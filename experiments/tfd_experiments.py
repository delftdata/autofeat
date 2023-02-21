import time
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from algorithms import BaseAlgorithm, TRAINING_FUNCTIONS, CART
from augmentation.ranking import Ranking, Rank
from config import JOIN_RESULT_FOLDER, PLOTS_FOLDER
from data_preparation.dataset_base import Dataset
from data_preparation.utils import get_join_path, prepare_data_for_ml
from experiments.base_experiment import BaseExperiment
from experiments.result_object import Result
from experiments.utils import hp_tune_join_all, map_features_scores
from helpers.util_functions import objects_to_dict


class TFDExperiment(BaseExperiment):
    def __init__(self, data: Dataset, learning_curve_depth_values):
        super().__init__(data, approach=Result.TFD, learning_curve_depth_values=learning_curve_depth_values)
        self.ranked_paths = None
        self.sensitivity_results: List[Result] = []
        self.learning_curve_train_tfd = []
        self.learning_curve_test_tfd = []
        self.learning_curve_train_tfd_path = []
        self.learning_curve_test_tfd_path = []
        self.cutoff_th_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.redundancy_th_values = [5, 7, 9, 10, 15, 20, 25, 30]

    def compute_results(self):
        print(f'======== TFD Pipeline ========')

        start = time.time()
        tfd_ranking = Ranking(self.dataset).start_ranking()
        end = time.time()
        join_time = end - start
        self.ranked_paths = tfd_ranking.ranked_paths

        top_1 = tfd_ranking.ranked_paths[0]
        joined_df, _ = self.__process_joined_data(top_1)
        X, y = prepare_data_for_ml(joined_df, self.dataset.target_column)
        self.__train_approach(X, y, Result.TFD_PATH, top_1.path, join_time)
        self.__learning_curve_results_tfd_path(X, y)

        print(f"Processing case 2: Remove all, but the ranked feature")

        aux_df = self.__process_joined_data_tfd(joined_df, top_1)
        X, y = prepare_data_for_ml(aux_df, self.dataset.target_column)
        self.__train_approach(X, y, Result.TFD, top_1.path, join_time)
        self.__learning_curve_results_tfd(X, y)

        print(f"======== Finished TFD Pipeline ========")

    def threshold_sensitivity_results(self, algorithm: BaseAlgorithm):
        for redundancy_th in self.redundancy_th_values:
            print(f"\n\tREDUNDANCY THRESHOLD: {redundancy_th}")
            for cutoff_th in self.cutoff_th_values:
                print(f"\n\tCUTOFF THRESHOLD: {cutoff_th}")
                ranking = Ranking(self.dataset.base_table_id, redundancy_th, cutoff_th)
                ranking.start_ranking()

                for i, ranked_path in enumerate(ranking.ranked_paths[0:3]):
                    # TFD - All in path
                    print(
                        f"Path: {i} - score: {ranked_path.score}\n\t{ranked_path.path}\n\t{ranked_path.features}"
                    )
                    print(f"Processing case 1: Keep the entire path")
                    joined_df, join_path = self.__process_joined_data(ranked_path)
                    X, y = prepare_data_for_ml(joined_df, self.dataset.target_column)
                    acc, params, feature_imp, _, _ = algorithm.train(X, y)

                    result = Result(
                        approach=Result.TFD_PATH,
                        data_path=join_path,
                        data_label=self.dataset.base_table_label,
                        algorithm=algorithm.LABEL,
                        depth=params["max_depth"],
                        rank=i,
                        accuracy=acc,
                        cutoff_threshold=cutoff_th,
                        redundancy_threshold=redundancy_th,
                        feature_importance=map_features_scores(feature_imp, X),
                    )
                    self.sensitivity_results.append(result)

                    print(f"Processing case 2: Remove all, but the ranked feature")
                    aux_df = self.__process_joined_data_tfd(joined_df, ranked_path)
                    X, y = prepare_data_for_ml(aux_df, self.dataset.target_column)
                    acc, params, feature_imp, _, _ = algorithm.train(X, y)

                    result = Result(
                        approach=Result.TFD,
                        data_path=join_path,
                        data_label=self.dataset.base_table_label,
                        algorithm=algorithm.LABEL,
                        depth=params["max_depth"],
                        rank=i,
                        accuracy=acc,
                        cutoff_threshold=cutoff_th,
                        redundancy_threshold=redundancy_th,
                        feature_importance=map_features_scores(feature_imp, X),
                    )
                    self.sensitivity_results.append(result)

    def plot_sensitivity_result(self, results: pd.DataFrame = None):
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 6))

        if results is None:
            results = pd.DataFrame(objects_to_dict(self.sensitivity_results))

        for i, cutoff_th in enumerate(self.cutoff_th_values):
            df = results[results["cutoff_th"] == cutoff_th]

            colors = ["red", "black", "green"]
            for j, rank in enumerate(df["rank"].unique()):
                df_ranks = df[df["rank"] == rank]

                marker = ["*", "H"]
                for k, approach in enumerate(df["approach"].unique()):
                    values = df_ranks[df_ranks["approach"] == approach]
                    if i == len(self.cutoff_th_values) - 1:
                        axs[i].plot(
                            self.redundancy_th_values,
                            values["accuracy"],
                            marker=marker[k],
                            color=colors[j],
                            label=f"{approach}-{rank}",
                        )
                    else:
                        axs[i].plot(
                            self.redundancy_th_values,
                            values["accuracy"],
                            marker=marker[k],
                            color=colors[j],
                        )
            axs[i].set_title(f"Cut-off Threshold = {cutoff_th}")

        fig.legend()
        fig.show()
        fig.savefig(PLOTS_FOLDER / f'sensitivity-plots-{self.dataset.base_table_label}.png', dpi=300,
                    bbox_inches="tight")

    def __process_joined_data_tfd(self, joined_df: pd.DataFrame, ranked_path: Rank):
        aux_df = joined_df.copy(deep=True)
        aux_features = list(joined_df.columns)
        aux_features.remove(self.dataset.target_column)
        columns_to_drop = [
            c
            for c in aux_features
            if (c not in self.dataset.base_table_features) and (c not in ranked_path.features)
        ]
        aux_df.drop(columns=columns_to_drop, inplace=True)
        return aux_df

    def __process_joined_data(self, ranked_path: Rank):
        join_path = get_join_path(ranked_path.path)

        print(f"Processing case 1: Keep the entire path")
        joined_df = pd.read_csv(
            JOIN_RESULT_FOLDER / join_path,
            header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
        )
        return joined_df, join_path

    def __train_approach(self, X, y, approach, data_path, join_time):
        for algorithm in TRAINING_FUNCTIONS:
            print(f"==== Model Name: {algorithm.LABEL} ====")
            accuracy, max_depth, feature_importances, train_time, _ = hp_tune_join_all(
                X, y, algorithm().train, False
            )
            entry = Result(
                approach=approach,
                data_path=data_path,
                data_label=self.dataset.base_table_label,
                algorithm=algorithm.LABEL,
                depth=max_depth,
                accuracy=accuracy,
                feature_importance=feature_importances,
                train_time=train_time,
                join_time=join_time,
            )
            self.results.append(entry)

    def __learning_curve_results_tfd_path(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        print(f"All in path features: {X.columns}")

        for i in self.learning_curve_depth_values:
            decision_tree = tree.DecisionTreeClassifier(max_depth=i)
            decision_tree.fit(X_train, y_train)
            # evaluate on train
            train_predict = decision_tree.predict(X_train)
            train_acc = accuracy_score(y_train, train_predict)
            self.learning_curve_train_tfd_path.append(train_acc)
            # evaluate on test
            test_predict = decision_tree.predict(X_test)
            test_acc = accuracy_score(y_test, test_predict)
            self.learning_curve_test_tfd_path.append(test_acc)
            print(f"Depth {i}:\tTrain acc: {train_acc}\tTest acc: {test_acc}")

    def __learning_curve_results_tfd(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        print(f"Best ranked features: {X.columns}")

        for i in self.learning_curve_depth_values:
            decision_tree = tree.DecisionTreeClassifier(max_depth=i)
            decision_tree.fit(X_train, y_train)
            # evaluate on train
            train_predict = decision_tree.predict(X_train)
            train_acc = accuracy_score(y_train, train_predict)
            self.learning_curve_train_tfd.append(train_acc)
            # evaluate on test
            test_predict = decision_tree.predict(X_test)
            test_acc = accuracy_score(y_test, test_predict)
            self.learning_curve_test_tfd.append(test_acc)
            print(f"Depth {i}:\tTrain acc: {train_acc}\tTest acc: {test_acc}")

    def run_sensitivity_experiments(self):
        self.threshold_sensitivity_results(CART())
        self.plot_sensitivity_result()
