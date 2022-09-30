from typing import List

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from arda.arda import select_arda_features
from data_preparation.dataset_base import Dataset
from experiments.result_object import Result
from experiments.utils import TRAINING_FUNCTIONS
from experiments.utils import hp_tune_join_all


class ArdaExperiment:
    def __init__(self, data: Dataset, learning_curve_depth_values):
        self.dataset = data
        self.approach = Result.ARDA
        self.results: List[Result] = []
        self.learning_curve_train = []
        self.learning_curve_test = []
        self.depth_values = learning_curve_depth_values
        self.selected_features = None

    def accuracy_results(self):
        print(f'======== ARDA Pipeline ========')

        X, y, join_time, fs_time, selected_features = select_arda_features(self.dataset.base_table_id,
                                                                           self.dataset.target_column,
                                                                           self.dataset.base_table_features)
        self.selected_features = selected_features
        for model_name, training_fun in TRAINING_FUNCTIONS.items():
            print(f"==== Model Name: {model_name} ====")
            entry = Result(self.approach, self.dataset.base_table_id, self.dataset.base_table_label, model_name)
            accuracy, max_depth, feature_importances, train_time, _ = hp_tune_join_all(X, y, training_fun, False)
            entry.set_depth(max_depth).set_accuracy(accuracy).set_feature_importance(
                feature_importances).set_train_time(train_time).set_feature_selection_time(fs_time).set_join_time(
                join_time)
            self.results.append(entry)

        print(f"======== Finished ARDA Pipeline ========")

    def learning_curve_results(self):
        print(f'======== ARDA Pipeline ========')

        X, y, _, _, _ = select_arda_features(self.dataset.base_table_id, self.dataset.target_column,
                                             self.dataset.base_table_features)

        print(f"ARDA features: {X.columns}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        for i in self.depth_values:
            decision_tree = tree.DecisionTreeClassifier(max_depth=i)
            decision_tree.fit(X_train, y_train)
            # evaluate on train
            train_predict = decision_tree.predict(X_train)
            train_acc = accuracy_score(y_train, train_predict)
            self.learning_curve_train.append(train_acc)
            # evaluate on test
            test_predict = decision_tree.predict(X_test)
            test_acc = accuracy_score(y_test, test_predict)
            self.learning_curve_test.append(test_acc)
            print(f"Depth {i}:\tTrain acc: {train_acc}\tTest acc: {test_acc}")
