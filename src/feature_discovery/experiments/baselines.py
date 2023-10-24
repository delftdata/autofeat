import logging
import time
from pathlib import Path

import pandas as pd

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.baselines.join_all import JoinAll
from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.experiments.evaluate_join_paths import evaluate_paths
from feature_discovery.experiments.evaluation_algorithms import run_auto_gluon, run_svm
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


def autofeat_pearson_mrmr(dataset: Dataset, value_ratio: float, top_k: int):
    logging.debug(f"Running on TFD (Transitive Feature Discovery) result with AutoGluon")

    start = time.time()
    bfs_traversal = AutoFeat(
        base_table_id=str(dataset.base_table_id),
        base_table_label=dataset.base_table_label,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        top_k=top_k,
        task=dataset.dataset_type,
        pearson=True
    )
    bfs_traversal.streaming_feature_selection(queue={str(dataset.base_table_id)})
    end = time.time()

    logging.debug("FINISHED TFD - Pearson + MRMR")

    all_results, top_k_paths = evaluate_paths(bfs_result=bfs_traversal,
                                              top_k=top_k,
                                              feat_sel_time=end - start,
                                              problem_type=dataset.dataset_type,
                                              approach=Result.TFD_Pearson)
    logging.debug(top_k_paths)

    logging.debug("Save results ... ")
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_tfd_pearson_mrmr.csv", index=False)

    return all_results


def autofeat_pearson_jmi(dataset: Dataset, value_ratio: float, top_k: int):
    logging.debug(f"Running on TFD (Transitive Feature Discovery) result with AutoGluon")

    start = time.time()
    bfs_traversal = AutoFeat(
        base_table_id=str(dataset.base_table_id),
        base_table_label=dataset.base_table_label,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        top_k=top_k,
        task=dataset.dataset_type,
        pearson=True,
        jmi=True
    )
    bfs_traversal.streaming_feature_selection(queue={str(dataset.base_table_id)})
    end = time.time()

    logging.debug("FINISHED TFD - Pearson + JMI")

    all_results, top_k_paths = evaluate_paths(bfs_result=bfs_traversal,
                                              top_k=top_k,
                                              feat_sel_time=end - start,
                                              problem_type=dataset.dataset_type,
                                              approach=Result.TFD_Pearson_JMI)
    logging.debug(top_k_paths)

    logging.debug("Save results ... ")
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_tfd_pearson_jmi.csv", index=False)

    return all_results


def autofeat_spearman_jmi(dataset: Dataset, value_ratio: float, top_k: int):
    logging.debug(f"Running on TFD (Transitive Feature Discovery) result with AutoGluon")

    start = time.time()
    bfs_traversal = AutoFeat(
        base_table_id=str(dataset.base_table_id),
        base_table_label=dataset.base_table_label,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        top_k=top_k,
        task=dataset.dataset_type,
        jmi=True
    )
    bfs_traversal.streaming_feature_selection(queue={str(dataset.base_table_id)})
    end = time.time()

    logging.debug("FINISHED TFD - Spearman + MRMR")

    all_results, top_k_paths = evaluate_paths(bfs_result=bfs_traversal,
                                              top_k=top_k,
                                              feat_sel_time=end - start,
                                              problem_type=dataset.dataset_type,
                                              approach=Result.TFD_JMI)
    logging.debug(top_k_paths)

    logging.debug("Save results ... ")
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_tfd_spearman_jmi.csv", index=False)

    return all_results


if __name__ == "__main__":
    init_datasets()
    dataset = filter_datasets(["credit"])[0]
    # join_all_bfs(dataset)
    # join_all_dfs(dataset)
    autofeat_pearson_jmi(dataset, value_ratio=0.65, top_k=15)

