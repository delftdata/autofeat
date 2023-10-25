import logging
import time

import numpy as np
import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from feature_discovery.config import AUTO_GLUON_FOLDER
from feature_discovery.experiments.dataset_object import REGRESSION
from feature_discovery.experiments.result_object import Result

hyper_parameters = {"RF": {}, "GBM": {}, "XGB": {}, "XT": {}, 'KNN': {},
                    'LR': [{'penalty': 'L1'}, {'penalty': 'L2'}]
                    }


def run_auto_gluon(dataframe: pd.DataFrame, target_column: str, problem_type: str, algorithms_to_run: dict):
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

    return end - start, results


def run_svm(dataframe: pd.DataFrame, target_column: str, backward_sel=False, forward_sel=False):
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import RFECV, SequentialFeatureSelector

    start = time.time()

    X = dataframe.drop(columns=[target_column])
    y = dataframe[[target_column]].values.ravel()

    if backward_sel:
        selector = RFECV(LinearSVC(dual=False), cv=10)
        new_X = selector.fit_transform(X, y)
        X = pd.DataFrame(new_X, columns=selector.get_feature_names_out())

    if forward_sel:
        selector = SequentialFeatureSelector(LinearSVC(dual=False))
        new_X = selector.fit_transform(X, y)
        X = pd.DataFrame(new_X, columns=selector.get_feature_names_out())

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


def run_logistic_regression(dataframe: pd.DataFrame, target_column: str):
    from sklearn.linear_model import LogisticRegressionCV

    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=[target_column]),
        dataframe[[target_column]],
        test_size=0.2,
        random_state=10,
    )
    clf = LogisticRegressionCV(cv=10, random_state=0).fit(X_train, y_train.values.ravel())
    entry = Result(
        algorithm="LogisticRegression",
        accuracy=clf.score(X_test, y_test),
        join_path_features=list(X_train.columns),
        feature_importance=dict(zip(list(X_train.columns), clf.coef_[0])),
    )

    end = time.time()
    return end - start, entry


def evaluate_all_algorithms(dataframe: pd.DataFrame, target_column: str, problem_tye: str = None):
    df = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False, enable_text_ngram_features=False
    ).fit_transform(X=dataframe)

    # Run LightGBM, XGBoost, Random Forest, Extreme Randomised Trees
    logging.debug(f"Training AutoGluon ... ")
    runtime, results = run_auto_gluon(
        dataframe=df,
        target_column=target_column,
        algorithms_to_run=hyper_parameters,
        problem_type=problem_tye,
    )

    for res in results:
        res.train_time = runtime

    # # sklearn: Run SVM
    # logging.debug(f"Training SVM ... ")
    # runtime_svm, result_svm = run_svm(dataframe=df,
    #                                   target_column=target_column)
    # result_svm.train_time = runtime_svm
    # results.append(result_svm)
    #
    # # sklearn: Run Naive Bayes
    # logging.debug(f"Training Naive Bayes ... ")
    # runtime_nb, result_nb = run_naive_bayes(dataframe=df,
    #                                         target_column=target_column)
    # result_nb.train_time = runtime_nb
    # results.append(result_nb)
    #
    # # sklearn: Run Logistic Regression
    # logging.debug(f"Training Logistic Regression ... ")
    # runtime_lr, result_lr = run_logistic_regression(dataframe=df,
    #                                                 target_column=target_column)
    # result_lr.train_time = runtime_lr
    # results.append(result_lr)

    return results
