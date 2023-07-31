import logging

import pandas as pd

from feature_discovery.config import AUTO_GLUON_FOLDER
from feature_discovery.experiments.dataset_object import REGRESSION
from feature_discovery.experiments.result_object import Result


def run_auto_gluon(approach: str, dataframe: pd.DataFrame, target_column: str, data_label: str, join_name: str,
                   problem_type: str, algorithms_to_run: dict, value_ratio: float = None):
    from sklearn.model_selection import train_test_split
    from autogluon.tabular import TabularPredictor

    logging.debug(f"Train algorithms: {list(algorithms_to_run.keys())} with AutoGluon ...")

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=[target_column]),
        dataframe[[target_column]],
        test_size=0.2,
        random_state=10,
    )
    train = X_train.copy()
    train[target_column] = y_train

    test = X_test.copy()
    test[target_column] = y_test

    predictor = TabularPredictor(label=target_column,
                                 problem_type=problem_type,
                                 verbosity=0,
                                 path=AUTO_GLUON_FOLDER / "models").fit(train_data=train,
                                                                        hyperparameters=algorithms_to_run)
    score_type = 'accuracy'
    if problem_type == REGRESSION:
        score_type = 'root_mean_squared_error'

    highest_acc = 0
    best_model = None
    results = []

    model_names = predictor.get_model_names()
    for model in model_names[:-1]:
        result = predictor.evaluate(data=test, model=model)
        accuracy = abs(result[score_type])
        ft_imp = predictor.feature_importance(
            data=test, model=model, feature_stage="original"
        )
        entry = Result(
            algorithm=model,
            accuracy=accuracy,
            feature_importance=dict(zip(list(ft_imp.index), ft_imp["importance"])),
            approach=approach,
            data_label=data_label,
            data_path=join_name,
            join_path_features=list(X_train.columns),
        )
        if value_ratio:
            entry.cutoff_threshold = value_ratio

        if accuracy > highest_acc:
            highest_acc = accuracy
            best_model = entry

        results.append(entry)

    return best_model, results
