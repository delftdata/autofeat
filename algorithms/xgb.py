import time

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from xgboost import XGBClassifier

from algorithms.helpers import feature_selection


class XGB:
    LABEL = "XGBoost"

    def __init__(self, num_cv: int = 10):
        self.num_cv: int = num_cv

    def train(self, X, y, do_sfs: bool = False):
        sfs_time = None

        if do_sfs:
            decision_tree = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                # Reduce execution time, otherwise it explodes
                max_depth=5,
                n_jobs=1,
            )
            X, sfs_time = feature_selection(X, y, decision_tree)

        print("Split data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        print(f"X train {X_train.shape}, X_test {X_test.shape}")
        print(f"Y uniqueness: {len(y_train.dropna()) / len(y_train)}")

        parameters = {"max_depth": range(1, X_train.shape[1] + 1)}
        print(parameters)
        decision_tree = XGBClassifier(
            objective="binary:logistic", eval_metric="auc", use_label_encoder=False, n_jobs=1
        )
        grids = GridSearchCV(decision_tree, parameters, n_jobs=1, scoring="accuracy", cv=10)
        grids.fit(X_train, y_train)
        params = grids.best_params_
        print(f"Hyper-params: {params} for best score: {grids.best_score_}")

        decision_tree = grids.best_estimator_

        start = time.time()
        cv_output = cross_validate(
            estimator=decision_tree,
            X=X,
            y=y,
            scoring="accuracy",
            return_estimator=True,
            cv=self.num_cv,
            verbose=10,
        )
        end = time.time()
        feature_importances = [estimator.feature_importances_ for estimator in cv_output["estimator"]]
        acc_decision_tree = np.mean(cv_output["test_score"])
        feature_importance = np.around(np.median(feature_importances, axis=0), 3)
        train_time = end - start

        print(f"\t\tAccuracy XGBoost: {acc_decision_tree}")

        return acc_decision_tree, params, feature_importance, train_time, sfs_time
