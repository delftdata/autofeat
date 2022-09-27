import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

from arda.arda import wrapper_algo
from augmentation.ranking import start_ranking
from data_preparation.join_data import join_all
from experiments.all_experiments import _get_join_path
from experiments.config import Datasets
from utils.file_naming_convention import MAPPING_FOLDER, ENUMERATED_PATHS, JOIN_RESULT_FOLDER
from utils.util_functions import prepare_data_for_ml

folder_name = os.path.abspath(os.path.dirname(__file__))


def get_paths():
    with open(f"{os.path.join(folder_name, '../', MAPPING_FOLDER)}/{ENUMERATED_PATHS}", 'r') as fp:
        all_paths = json.load(fp)
    return all_paths


class LearningCurves:
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.all_in_path_train_scores = []
        self.all_in_path_test_scores = []
        self.best_ranked_train_scores = []
        self.best_ranked_test_scores = []
        self.arda_train = []
        self.arda_test = []
        self.depth_values = [i for i in range(1, 21)]
        self.ranking = self._get_ranks()

    def _get_ranks(self):
        all_paths = get_paths()
        target_column = self.dataset_config["label_column"]
        ranking = start_ranking(self.dataset_config['id'], target_column, all_paths)
        sorted_ranks = dict(sorted(ranking.items(), key=lambda item: item[1][0]))
        return sorted_ranks

    def all_in_path_curves(self):
        print(f'======== All in path Pipeline ========')
        target_column = self.dataset_config["label_column"]
        top_1 = list(self.ranking.keys())[0]
        join_path = _get_join_path(top_1)

        joined_df = pd.read_csv(
            f"../{JOIN_RESULT_FOLDER}/{join_path}",
            header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
        )
        X, y = prepare_data_for_ml(joined_df, target_column)
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
        target_column = self.dataset_config["label_column"]

        base_table_features = pd.read_csv(
            self.dataset_config['id'],
            engine="python", encoding="utf8", quotechar='"', escapechar='\\', nrows=1).drop(
            columns=[target_column]).columns
        top_1 = list(self.ranking.keys())[0]
        score, features = self.ranking[top_1]
        join_path = _get_join_path(top_1)

        joined_df = pd.read_csv(
            f"../{JOIN_RESULT_FOLDER}/{join_path}",
            header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
        )

        aux_features = list(joined_df.columns)
        aux_features.remove(target_column)
        columns_to_drop = [
            c for c in aux_features if (c not in base_table_features) and (c not in features)
        ]
        joined_df.drop(columns=columns_to_drop, inplace=True)

        X, y = prepare_data_for_ml(joined_df, target_column)
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
        target_column = self.dataset_config['label_column']

        base_table_features = pd.read_csv(
            self.dataset_config['id'],
            engine="python", encoding="utf8", quotechar='"', escapechar='\\', nrows=1).drop(
            columns=[target_column]).columns

        dataset_df = join_all(self.dataset_config['id'])
        X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=target_column)
        print(X.shape)
        if X.shape[0] > 10000:
            _, X, _, y = train_test_split(X, y, test_size=10000, shuffle=True, stratify=y)
        print(X.shape)

        T = np.arange(0.0, 1.0, 0.1)
        indices = wrapper_algo(X, y, T)
        if len(indices) == 0:
            return

        fs_X = X.iloc[:, indices].columns
        columns_to_drop = [
            c for c in list(X.columns) if (c not in base_table_features) and (c not in fs_X)
        ]
        X.drop(columns=columns_to_drop, inplace=True)

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
        fig.savefig(f'../plots/learning-curves-{self.dataset_config["base_table_label"]}.pdf', dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    titanic_lc = LearningCurves(Datasets.steel_data)
    titanic_lc.all_in_path_curves()
    titanic_lc.best_ranked_curves()
    titanic_lc.arda_results()
    titanic_lc.plot_curves()

