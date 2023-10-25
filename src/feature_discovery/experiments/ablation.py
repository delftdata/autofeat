import logging
import time

import pandas as pd

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.experiments.evaluate_join_paths import evaluate_paths
from feature_discovery.experiments.init_datasets import init_datasets
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets


def autofeat_spearman(dataset: Dataset, value_ratio: float, top_k: int):
    logging.debug(f"Running on TFD (Transitive Feature Discovery) result with AutoGluon")

    start = time.time()
    bfs_traversal = AutoFeat(
        base_table_id=str(dataset.base_table_id),
        base_table_label=dataset.base_table_label,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        top_k=top_k,
        task=dataset.dataset_type,
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
    autofeat_pearson_jmi(dataset, value_ratio=0.65, top_k=15)
