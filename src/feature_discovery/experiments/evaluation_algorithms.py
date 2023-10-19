import logging
import time

import numpy as np
import pandas as pd

from feature_discovery.config import AUTO_GLUON_FOLDER
from feature_discovery.experiments.dataset_object import REGRESSION
from feature_discovery.experiments.result_object import Result

from sklearn.model_selection import cross_validate


def run_auto_gluon(dataframe: pd.DataFrame, target_column: str, problem_type: str, algorithms_to_run: dict):
    from sklearn.model_selection import train_test_split
    from autogluon.tabular import TabularPredictor

    start = time.time()

    logging.debug(f"Train algorithms: {list(algorithms_to_run.keys())} with AutoGluon ...")

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=[target_column]),
        dataframe[[target_column]],
        test_size=0.2,
        random_state=10,
    )
    join_path_features = list(X_train.columns)
    X_train[target_column] = y_train
    X_test[target_column] = y_test

    predictor = TabularPredictor(label=target_column,
                                 problem_type=problem_type,
                                 verbosity=0,
                                 path=AUTO_GLUON_FOLDER / "models").fit(train_data=X_train,
                                                                        hyperparameters=algorithms_to_run)
    score_type = 'accuracy'
    if problem_type == REGRESSION:
        score_type = 'root_mean_squared_error'

    results = []

    model_names = predictor.get_model_names()
    for model in model_names[:-1]:
        result = predictor.evaluate(data=X_test, model=model)
        accuracy = abs(result[score_type])
        ft_imp = predictor.feature_importance(
            data=X_test, model=model, feature_stage="original"
        )
        entry = Result(
            algorithm=model,
            accuracy=accuracy,
            feature_importance=dict(zip(list(ft_imp.index), ft_imp["importance"])),
            join_path_features=join_path_features,
        )

        results.append(entry)

    end = time.time()

    return end-start, results


def run_svm(dataframe: pd.DataFrame, target_column: str):
    from sklearn.svm import LinearSVC

    start = time.time()

    X = dataframe.drop(columns=[target_column])
    y = dataframe[[target_column]].values.ravel()

    scores = cross_validate(LinearSVC(dual=False), X, y, cv=10, scoring="accuracy", return_estimator=True)
    estimator = scores["estimator"][np.argmax(scores['test_score'])]

    entry = Result(
        algorithm="SVM",
        accuracy=scores['test_score'].mean(),
        feature_importance=dict(zip(list(X.columns), estimator.coef_[0])),
        join_path_features=list(X.columns),
    )

    end = time.time()

    return end - start, entry


def run_naive_bayes(dataframe: pd.DataFrame, target_column: str):
    from sklearn.naive_bayes import GaussianNB

    start = time.time()

    X = dataframe.drop(columns=[target_column])
    y = dataframe[[target_column]].values.ravel()
    scores = cross_validate(GaussianNB(), X, y, cv=10, scoring="accuracy")

    entry = Result(
        algorithm="NaiveBayes",
        accuracy=scores['test_score'].mean(),
        join_path_features=list(X.columns),
    )

    end = time.time()

    return end - start, entry
