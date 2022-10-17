import time

import numpy as np
from sklearn.model_selection import cross_validate

from algorithms import BaseAlgorithm
from algorithms.helpers import feature_selection
from algorithms.id3_alg import GadId3Classifier


class ID3(BaseAlgorithm):
    LABEL = "ID3"

    def __init__(self, num_cv: int = 10):
        super().__init__()
        self.num_cv: int = num_cv
        self.max_depth = None

    def train(self, X, y, do_sfs: bool = False):
        sfs_time = None

        # Not supported yet. TODO: Adapt ID3
        if do_sfs:
            decision_tree = GadId3Classifier()
            X, sfs_time = feature_selection(X, y, decision_tree)

        self.algorithm = GadId3Classifier()
        start = time.time()
        cv_output = cross_validate(
            estimator=self.algorithm,
            X=X,
            y=y,
            scoring="accuracy",
            return_estimator=True,
            verbose=10,
            cv=self.num_cv,
            n_jobs=1,
        )
        end = time.time()
        train_time = end - start
        max_depths = [estimator.depth() for estimator in cv_output["estimator"]]
        params = {"max_depth": np.median(max_depths)}
        self.max_depth = params["max_depth"]

        acc_decision_tree = np.mean(cv_output["test_score"])

        print(f"\t\tAccuracy ID3: {acc_decision_tree}")
        print(max_depths)

        # Empty list to be consistent with other models
        return acc_decision_tree, params, [], train_time, sfs_time
