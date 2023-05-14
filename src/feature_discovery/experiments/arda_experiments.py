from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from feature_discovery.algorithms import TRAINING_FUNCTIONS
from feature_discovery.arda.arda import select_arda_features_budget_join
from feature_discovery.data_preparation.dataset_base import Dataset
from feature_discovery.experiments.base_experiment import BaseExperiment
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils import hp_tune_join_all


class ArdaExperiment(BaseExperiment):
    def __init__(self, data: Dataset, learning_curve_depth_values, sample_size=1000):
        super().__init__(data, approach=Result.ARDA, learning_curve_depth_values=learning_curve_depth_values)
        self.learning_curve_train = []
        self.learning_curve_test = []
        self.selected_features = None
        self.sample_size = sample_size

    def compute_results(self):
        print(f'======== ARDA Pipeline ========')

        X, y, selected_features, join_time, fs_time = select_arda_features_budget_join(self.dataset.base_table_id,
                                                                                       self.dataset.target_column,
                                                                                       self.dataset.base_table_features,
                                                                                       self.sample_size)
        self.selected_features = selected_features
        for algorithm in TRAINING_FUNCTIONS:
            print(f"==== Model Name: {algorithm.LABEL} ====")
            accuracy, max_depth, feature_importances, train_time, _ = hp_tune_join_all(X, y, algorithm().train, False)
            entry = Result(
                approach=self.approach,
                data_path=self.dataset.base_table_id,
                data_label=self.dataset.base_table_label,
                algorithm=algorithm.LABEL,
                depth=max_depth,
                accuracy=accuracy,
                feature_importance=feature_importances,
                train_time=train_time,
                feature_selection_time=fs_time,
                join_time=join_time
            )
            self.results.append(entry)

        self.__learning_curve_results(X, y)

        print(f"======== Finished ARDA Pipeline ========")

    def __learning_curve_results(self, X, y):
        print(f'======== ARDA Learning Curve Results ========')
        print(f"Features: {X.columns}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        for i in self.learning_curve_depth_values:
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
