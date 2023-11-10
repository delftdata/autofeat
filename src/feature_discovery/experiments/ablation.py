import logging
import time
from typing import Tuple, List

import pandas as pd

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.experiments.evaluate_join_paths import evaluate_paths
from feature_discovery.experiments.init_datasets import init_datasets
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets


def autofeat(
    dataset: Dataset,
    value_ratio: float,
    top_k: int,
    algorithm: str,
    approach: str = Result.TFD,
    pearson: bool = False,
    jmi: bool = False,
    no_relevance: bool = False,
    no_redundancy: bool = False,
    save_joins_to_disk: bool = True,
    use_polars: bool = True,
) -> Tuple[List[Result], List[Tuple]]:
    logging.debug(f"Running on TFD (Transitive Feature Discovery) result with AutoGluon")

    start = time.time()
    bfs_traversal = AutoFeat(
        base_table_id=str(dataset.base_table_id),
        base_table_label=dataset.base_table_label,
        save_joins_to_disk=save_joins_to_disk,
        use_polars=use_polars,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        top_k=top_k,
        task=dataset.dataset_type,
        pearson=pearson,
        jmi=jmi,
        no_redundancy=no_redundancy,
        no_relevance=no_relevance,
    )
    bfs_traversal.streaming_feature_selection(queue={str(dataset.base_table_id)})
    end = time.time()

    logging.debug(f"FINISHED {approach}")

    all_results, top_k_paths = evaluate_paths(
        bfs_result=bfs_traversal, problem_type=dataset.dataset_type, algorithm=algorithm
    )
    for result in all_results:
        result.approach = approach
        result.feature_selection_time = end - start
        result.total_time += result.feature_selection_time
        result.top_k = top_k
        result.data_label = dataset.base_table_label
        result.cutoff_threshold = value_ratio

    logging.debug("Save results ... ")
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_{approach}.csv", index=False)

    return all_results, top_k_paths


if __name__ == "__main__":
    init_datasets()
    dataset = filter_datasets(["credit"])[0]
    autofeat(dataset, value_ratio=0.65, top_k=15)
