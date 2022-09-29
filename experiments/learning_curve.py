import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import arda.arda
from augmentation.ranking import Ranking
from data_preparation.dataset_base import Dataset
from data_preparation.utils import prepare_data_for_ml, get_join_path
from experiments.datasets import Datasets
from utils.file_naming_convention import JOIN_RESULT_FOLDER

folder_name = os.path.abspath(os.path.dirname(__file__))


class LearningCurves:
    def __init__(self, dataset: Dataset, ranked_paths=None):
        self.dataset = dataset
        self.dataset.set_features()
        self.all_in_path_train_scores = []
        self.all_in_path_test_scores = []
        self.best_ranked_train_scores = []
        self.best_ranked_test_scores = []
        self.arda_train = []
        self.arda_test = []
        self.depth_values = [i for i in range(1, 21)]
        self.ranked_paths = ranked_paths if ranked_paths is not None else self.set_ranked_paths()

    def set_ranked_paths(self):
        return Ranking(self.dataset).start_ranking().ranked_paths

    def all_in_path_curves(self):
        print(f'======== All in path Pipeline ========')
        top_1 = self.ranked_paths[0]
        join_path = get_join_path(top_1.path)

        joined_df = pd.read_csv(
            f"../{JOIN_RESULT_FOLDER}/{join_path}",
            header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
        )
        X, y = prepare_data_for_ml(joined_df, self.dataset.target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        print(f"All in path features: {X.columns}")

        for i in self.depth_values:
            decision_tree = tree.DecisionTreeClassifier(max_depth=i)
            decision_tree.fit(X_train, y_train)
            # evaluate on train
            train_predict = decision_tree.predict(X_train)
            train_acc = accuracy_score(y_train, train_predict)
            self.all_in_path_train_scores.append(train_acc)
            # evaluate on test
            test_predict = decision_tree.predict(X_test)
            test_acc = accuracy_score(y_test, test_predict)
            self.all_in_path_test_scores.append(test_acc)
            print(f"Depth {i}:\tTrain acc: {train_acc}\tTest acc: {test_acc}")

    def best_ranked_curves(self):
        top_1 = self.ranked_paths[0]
        join_path = get_join_path(top_1.path)

        joined_df = pd.read_csv(
            f"../{JOIN_RESULT_FOLDER}/{join_path}",
            header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
        )

        aux_features = list(joined_df.columns)
        aux_features.remove(self.dataset.target_column)
        columns_to_drop = [
            c for c in aux_features if (c not in self.dataset.base_table_features) and (c not in top_1.features)
        ]
        joined_df.drop(columns=columns_to_drop, inplace=True)

        X, y = prepare_data_for_ml(joined_df, self.dataset.target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        print(f"Best ranked features: {X.columns}")

        for i in self.depth_values:
            decision_tree = tree.DecisionTreeClassifier(max_depth=i)
            decision_tree.fit(X_train, y_train)
            # evaluate on train
            train_predict = decision_tree.predict(X_train)
            train_acc = accuracy_score(y_train, train_predict)
            self.best_ranked_train_scores.append(train_acc)
            # evaluate on test
            test_predict = decision_tree.predict(X_test)
            test_acc = accuracy_score(y_test, test_predict)
            self.best_ranked_test_scores.append(test_acc)
            print(f"Depth {i}:\tTrain acc: {train_acc}\tTest acc: {test_acc}")

    def arda_results(self):
        print(f'======== ARDA Pipeline ========')

        X, y, _, _ = arda.arda.select_arda_features(self.dataset.base_table_id, self.dataset.target_column,
                                                    self.dataset.base_table_features)

        print(f"ARDA features: {X.columns}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        for i in self.depth_values:
            decision_tree = tree.DecisionTreeClassifier(max_depth=i)
            decision_tree.fit(X_train, y_train)
            # evaluate on train
            train_predict = decision_tree.predict(X_train)
            train_acc = accuracy_score(y_train, train_predict)
            self.arda_train.append(train_acc)
            # evaluate on test
            test_predict = decision_tree.predict(X_test)
            test_acc = accuracy_score(y_test, test_predict)
            self.arda_test.append(test_acc)
            print(f"Depth {i}:\tTrain acc: {train_acc}\tTest acc: {test_acc}")

    def plot_curves(self):
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12, 6))

        axs[0].set_title("Best Ranked Features")
        axs[0].plot(self.depth_values, self.best_ranked_train_scores, '-o')
        axs[0].plot(self.depth_values, self.best_ranked_test_scores, '-o')

        axs[1].set_title("All in path Features")
        axs[1].plot(self.depth_values, self.all_in_path_train_scores, '-o')
        axs[1].plot(self.depth_values, self.all_in_path_test_scores, '-o')

        axs[2].set_title("Arda")
        axs[2].plot(self.depth_values, self.arda_train, '-o', label='Train')
        axs[2].plot(self.depth_values, self.arda_test, '-o', label='Test')

        fig.legend()
        fig.show()
        fig.savefig(f'../plots/learning-curves-{self.dataset.base_table_label}.pdf', dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    titanic_lc = LearningCurves(Datasets.steel_plate_fault)
    titanic_lc.all_in_path_curves()
    titanic_lc.best_ranked_curves()
    titanic_lc.arda_results()
    titanic_lc.plot_curves()
