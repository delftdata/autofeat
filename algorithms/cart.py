import time
from subprocess import check_call

import numpy as np
import pandas as pd
from sklearn import tree, base
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_validate, HalvingGridSearchCV

from algorithms.base_algorithm import BaseAlgorithm
from algorithms.helpers import feature_selection
from config import PLOTS_FOLDER


class CART(BaseAlgorithm):
    LABEL = "CART"

    def __init__(self, num_cv: int = 10):
        super().__init__()
        self.num_cv: int = num_cv
        self.max_depth = None

    def train(self, train_data: pd.DataFrame, target_data: pd.Series, do_sfs: bool = False, regression: bool = False):
        print("Starting CART training ... ")

        # TODO: Fix feature selection
        sfs_time = None
        if do_sfs:
            decision_tree = tree.DecisionTreeClassifier()
            train_data, sfs_time = feature_selection(train_data, target_data, decision_tree)

        print("\t 1. Spliting data ... ")
        X_train, X_test, y_train, y_test = train_test_split(train_data, target_data, test_size=0.2, random_state=10)
        print(f"\t\tX train {X_train.shape}, X_test {X_test.shape}")

        print("\t 2. Hyper-parameter tuning ...")
        if regression:
            params = self._hyper_para_tuning_regression(X_train, y_train)
            scoring = "neg_root_mean_squared_error"
        else:
            params = self._hyper_para_tuning_classification(X_train, y_train)
            scoring = "accuracy"

        print(f"\t 3. Training ... ")
        cv_output = cross_validate(
            estimator=self.algorithm,
            X=X_train,
            y=y_train,
            scoring=scoring,
            return_estimator=True,
            cv=self.num_cv,
            verbose=10,
            n_jobs=1,
        )

        feature_importances = [estimator.feature_importances_ for estimator in cv_output["estimator"]]
        feature_importance = np.around(np.median(feature_importances, axis=0), 3)
        acc_decision_tree = np.mean(cv_output["test_score"])

        return acc_decision_tree, params, feature_importance, sfs_time

    def _hyper_para_tuning_classification(self, train_data, train_labels):
        parameters = {"criterion": ["entropy", "gini"],
                      "max_depth": range(1, int(train_data.shape[1] / 2 + 1))
                      }
        # parameters = {"criterion": ["entropy", "gini"]}

        decision_tree = tree.DecisionTreeClassifier()
        grids = HalvingGridSearchCV(decision_tree, parameters, n_jobs=1, scoring="accuracy")
        grids.fit(train_data, train_labels)
        params = grids.best_params_
        self.algorithm = grids.best_estimator_
        self.max_depth = grids.best_params_["max_depth"]
        print(f"\t\tHyper-params: {params} for best score: {grids.best_score_}")

        return params

    def _hyper_para_tuning_regression(self, train_data, train_labels):
        parameters = {"criterion": ["squared_error", "friedman_mse"],
                      "max_depth": range(1, int(train_data.shape[1] / 2 + 2))
                      }
        # parameters = {"criterion": ["entropy", "gini"]}

        decision_tree = tree.DecisionTreeRegressor()
        grids = HalvingGridSearchCV(decision_tree, parameters, n_jobs=1, scoring="neg_root_mean_squared_error")
        grids.fit(train_data, train_labels)
        params = grids.best_params_
        self.algorithm = grids.best_estimator_
        self.max_depth = grids.best_params_["max_depth"]
        print(f"\t\tHyper-params: {params} for best score: {grids.best_score_}")

        return params


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

