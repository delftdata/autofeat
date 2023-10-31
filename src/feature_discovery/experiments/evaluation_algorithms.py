import logging
import time

import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split

from feature_discovery.config import AUTO_GLUON_FOLDER
from feature_discovery.experiments.dataset_object import REGRESSION
from feature_discovery.experiments.result_object import Result

hyper_parameters = {
    "RF": {},
    "GBM": {},
    "XT": {},
    "XGB": {},
    'KNN': {},
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


def evaluate_all_algorithms(dataframe: pd.DataFrame, target_column: str, problem_tye: str = None):
    df = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False, enable_text_ngram_features=False
    ).fit_transform(X=dataframe)

    logging.debug(f"Training AutoGluon ... ")
    runtime, results = run_auto_gluon(
        dataframe=df,
        target_column=target_column,
        algorithms_to_run=hyper_parameters,
        problem_type=problem_tye,
    )

    for res in results:
        res.train_time = runtime
        res.total_time += res.train_time

    return results, df
