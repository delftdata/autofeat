import logging
import math
import time

import numpy as np
import pandas as pd

from feature_discovery.autofeat_pipeline.feature_selection import spearman_correlation
from feature_discovery.baselines.arda import select_arda_features_budget_join
from feature_discovery.baselines.join_all import JoinAll
from feature_discovery.experiments.dataset_object import Dataset, REGRESSION
from feature_discovery.experiments.evaluation_algorithms import evaluate_all_algorithms
from feature_discovery.experiments.init_datasets import init_datasets
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets


def join_all_bfs(dataset: Dataset, algorithm: str):
    all_results = []
    joinall = JoinAll(
        base_table_id=str(dataset.base_table_id),
        target_column=dataset.target_column,
    )
    dataframe = joinall.join_all_bfs(queue={str(dataset.base_table_id)})
    dataframe.drop(columns=joinall.join_keys[joinall.partial_join_name], inplace=True)
    print(dataframe.shape)

    # Evaluate Join-All with all features
    results, df = evaluate_all_algorithms(dataframe=dataframe,
                                          target_column=dataset.target_column,
                                          problem_type=dataset.dataset_type,
                                          algorithm=algorithm)
    for res in results:
        res.approach = Result.JOIN_ALL_BFS
        res.data_path = joinall.partial_join_name
        res.data_label = dataset.base_table_label
    all_results.extend(results)

    # Join-All with filter feature selection
    start = time.time()
    X = df.drop(columns=[dataset.target_column])
    y = df[dataset.target_column]
    sorted_features_scores = sorted(list(zip(list(X.columns), abs(spearman_correlation(np.array(X), np.array(y))))),
                                    key=lambda s: s[1], reverse=True)[:math.floor(len(X.columns) / 2)]
    spearman_features = list(map(lambda x: x[0], sorted_features_scores))
    selected_features = spearman_features.copy()
    selected_features.append(dataset.target_column)
    end = time.time()

    results, _ = evaluate_all_algorithms(dataframe=dataframe[selected_features],
                                         target_column=dataset.target_column,
                                         problem_type=dataset.dataset_type,
                                         algorithm=algorithm)
    for res in results:
        res.approach = Result.JOIN_ALL_BFS_F
        res.data_path = joinall.partial_join_name
        res.data_label = dataset.base_table_label
        res.join_path_features = spearman_features
        res.feature_selection_time = end - start
        res.total_time += res.feature_selection_time
    all_results.extend(results)

    return all_results


def non_augmented(dataframe: pd.DataFrame, dataset: Dataset, algorithm: str):
    results, _ = evaluate_all_algorithms(dataframe=dataframe,
                                         target_column=dataset.target_column,
                                         problem_type=dataset.dataset_type,
                                         algorithm=algorithm)
    for res in results:
        res.approach = Result.BASE
        res.data_path = dataset.base_table_label
        res.data_label = dataset.base_table_label

    return results


def arda(dataset: Dataset, algorithm: str, sample_size: int):
    logging.debug(f"ARDA result on table {dataset.base_table_id}")

    start = time.time()

    (
        dataframe,
        base_table_features,
        selected_features,
        join_name,
    ) = select_arda_features_budget_join(
        base_node_id=str(dataset.base_table_id),
        target_column=dataset.target_column,
        sample_size=sample_size,
        regression=(dataset.dataset_type == REGRESSION),
    )
    end = time.time()

    logging.debug(f"X shape: {dataframe.shape}\nSelected features:\n\t{selected_features}")
    features = selected_features.copy()
    features.append(dataset.target_column)
    features.extend(base_table_features)

    logging.debug(f"Running on ARDA Feature Selection result with AutoGluon")
    results, _ = evaluate_all_algorithms(dataframe=dataframe[features],
                                         target_column=dataset.target_column,
                                         algorithm=algorithm,
                                         problem_type=dataset.dataset_type)
    for result in results:
        result.approach = Result.ARDA
        result.data_label = dataset.base_table_label
        result.data_path = join_name
        result.feature_selection_time = end - start
        result.total_time += result.feature_selection_time

    return results


if __name__ == "__main__":
    init_datasets()
    dataset = filter_datasets(["credit"])[0]
    # join_all_bfs(dataset)
    # join_all_dfs(dataset)
