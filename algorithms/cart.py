import time
from subprocess import check_call

import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

from algorithms.base_algorithm import BaseAlgorithm
from algorithms.helpers import feature_selection
from config import PLOTS_FOLDER


class CART(BaseAlgorithm):
    LABEL = "CART"

    def __init__(self, num_cv: int = 10):
        super().__init__()
        self.num_cv: int = num_cv
        self.max_depth = None

    def train(self, X, y, do_sfs: bool = False):
        sfs_time = None

        if do_sfs:
            decision_tree = tree.DecisionTreeClassifier()
            X, sfs_time = feature_selection(X, y, decision_tree)

        print("Split data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        print(f"X train {X_train.shape}, X_test {X_test.shape}")
        print(f"Y uniqueness: {len(y_train.dropna()) / len(y_train)}")

        print("\tFinding best tree params")

        # parameters = {"criterion": ["entropy", "gini"], "max_depth": range(1, X_train.shape[1] + 1)}
        parameters = {"criterion": ["entropy", "gini"]}

        print(parameters)

        decision_tree = tree.DecisionTreeClassifier()
        grids = GridSearchCV(decision_tree, parameters, n_jobs=1, scoring="accuracy", cv=self.num_cv)
        grids.fit(X_train, y_train)
        params = grids.best_params_
        # self.max_depth = grids.best_params_["max_depth"]

        # TODO Store all the accuracies and the trees and see how the trees look
        print(f"Hyper-params: {params} for best score: {grids.best_score_}")

        print(f"\t Training ... ")

        self.algorithm = grids.best_estimator_
        start = time.time()
        cv_output = cross_validate(
            estimator=self.algorithm,
            X=X,
            y=y,
            scoring="accuracy",
            return_estimator=True,
            cv=self.num_cv,
            verbose=10,
            n_jobs=1,
        )
        end = time.time()
        train_time = end - start
        feature_importances = [estimator.feature_importances_ for estimator in cv_output["estimator"]]

        acc_decision_tree = np.mean(cv_output["test_score"])
        feature_importance = np.around(np.median(feature_importances, axis=0), 3)

        print(f"\t\tAccuracy CART: {acc_decision_tree}")

        return acc_decision_tree, params, feature_importance, train_time, sfs_time

    def print_tree(self, X_train, y_train):
        with open(PLOTS_FOLDER / tree.dot, "w") as f:
            tree.export_graphviz(
                self.algorithm,
                out_file=f,
                max_depth=self.max_depth,
                impurity=True,
                feature_names=list(X_train),
                class_names=list(map(lambda x: str(x), y_train.unique())),
                rounded=True,
                filled=True,
            )

        # Convert .dot to .png to allow display in web notebook
        check_call(["dot", "-Tpng", PLOTS_FOLDER / "tree.dot", "-o", PLOTS_FOLDER / "CART-tree.png"])

