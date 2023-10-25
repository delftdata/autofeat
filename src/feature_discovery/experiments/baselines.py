import logging
import time
from pathlib import Path

import pandas as pd

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.baselines.join_all import JoinAll
from feature_discovery.config import RESULTS_FOLDER, DATA_FOLDER
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.experiments.evaluate_join_paths import evaluate_paths
from feature_discovery.experiments.evaluation_algorithms import run_auto_gluon, run_svm, evaluate_all_algorithms
from feature_discovery.experiments.init_datasets import init_datasets
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets


def join_all_bfs(dataset: Dataset):
    joinall = JoinAll(
        base_table_id=str(dataset.base_table_id),
        target_column=dataset.target_column,
    )
    dataframe = joinall.join_all_bfs(queue={str(dataset.base_table_id)})
    dataframe.drop(columns=joinall.join_keys[joinall.partial_join_name], inplace=True)

    results = []

    # Join All simple scenario
    runtime_svm, res = run_svm(dataframe=dataframe,
                               target_column=dataset.target_column)

    res.approach = Result.JOIN_ALL_BFS
    res.data_label = dataset.base_table_label
    res.data_path = joinall.partial_join_name
    res.train_time = runtime_svm
    results.append(res)

    # Join All with Forward Selection scenario
    runtime_svm, result = run_svm(dataframe=dataframe,
                                  target_column=dataset.target_column,
                                  backward_sel=True)
    result.train_time = runtime_svm
    result.data_path = joinall.partial_join_name
    result.approach = Result.JOIN_ALL_BFS_BWD
    result.data_label = dataset.base_table_label
    results.append(result)

    # Join All with Forward Selection scenario
    runtime_svm, result = run_svm(dataframe=dataframe,
                                  target_column=dataset.target_column,
                                  forward_sel=True)
    result.train_time = runtime_svm
    result.data_path = joinall.partial_join_name
    result.approach = Result.JOIN_ALL_BFS_FWD
    result.data_label = dataset.base_table_label
    results.append(result)

    # Save intermediate results
    pd.DataFrame(results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_join_all_BFS.csv", index=False)
    return results


def join_all_dfs(dataset: Dataset):
    joinall = JoinAll(
        base_table_id=str(dataset.base_table_id),
        target_column=dataset.target_column,
    )
    joinall.join_all_dfs()

    dataframe = pd.read_csv(
        Path(joinall.temp_dir.name) / joinall.join_name_mapping[joinall.partial_join_name],
        header=0,
        engine="python",
        encoding="utf8",
        quotechar='"',
        escapechar='\\',
    )
    dataframe.drop(columns=joinall.join_keys[joinall.partial_join_name], inplace=True)

    results = []

    # Join All simple scenario
    runtime_svm, res = run_svm(dataframe=dataframe,
                               target_column=dataset.target_column)

    res.approach = Result.JOIN_ALL_DFS
    res.data_label = dataset.base_table_label
    res.data_path = joinall.partial_join_name
    res.train_time = runtime_svm
    results.append(res)

    # Join All with Forward Selection scenario
    runtime_svm, result = run_svm(dataframe=dataframe,
                                  target_column=dataset.target_column,
                                  backward_sel=True)
    result.train_time = runtime_svm
    result.data_path = joinall.partial_join_name
    result.approach = Result.JOIN_ALL_DFS_BWD
    result.data_label = dataset.base_table_label
    results.append(result)

    # Join All with Forward Selection scenario
    runtime_svm, result = run_svm(dataframe=dataframe,
                                  target_column=dataset.target_column,
                                  forward_sel=True)
    result.train_time = runtime_svm
    result.data_path = joinall.partial_join_name
    result.approach = Result.JOIN_ALL_DFS_FWD
    result.data_label = dataset.base_table_label
    results.append(result)

    # Save intermediate results
    pd.DataFrame(results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_join_all_DFS.csv", index=False)
    return results


def non_augmented(dataframe: pd.DataFrame, dataset: Dataset):
    results = evaluate_all_algorithms(dataframe=dataframe,
                                      target_column=dataset.target_column,
                                      problem_tye=dataset.dataset_type)
    for res in results:
        res.approach = Result.BASE
        res.data_path = dataset.base_table_label
        res.data_label = dataset.base_table_label

    return results


if __name__ == "__main__":
    init_datasets()
    dataset = filter_datasets(["credit"])[0]
    # join_all_bfs(dataset)
    # join_all_dfs(dataset)

