from datetime import time
from typing import List, Optional

import numpy as np
import pandas as pd
import time

from feature_discovery.augmentation.bfs_pipeline import BfsAugmentation
from feature_discovery.augmentation.trial_error import run_auto_gluon, train_test_cart
from feature_discovery.config import DATA_FOLDER, RESULTS_FOLDER
from feature_discovery.data_preparation.dataset_base import Dataset
from feature_discovery.experiments.ablation_experiments import (
    ablation_study_enumerate_paths,
    ablation_study_enumerate_and_join,
    ablation_study_prune_paths,
    ablation_study_feature_selection,
    ablation_study_prune_join_key_level,
)
from feature_discovery.experiments.result_object import Result
from feature_discovery.tfd_datasets import CLASSIFICATION_DATASETS

hyper_parameters = {"RF": {}, "GBM": {}, "XGB": {}, "XT": {}}


def filter_datasets(dataset_labels: Optional[List[str]] = None):
    # `is None` is missing on purpose, because typer cannot return None default values for lists, only []
    if not dataset_labels:
        datasets = CLASSIFICATION_DATASETS
    else:
        datasets = [dataset for dataset in CLASSIFICATION_DATASETS if dataset.base_table_label in dataset_labels]

    return datasets


def get_base_results(dataset: Dataset, autogluon: bool = True):
    print(f"Base result on table {dataset.base_table_id}")
    dataframe = pd.read_csv(
        DATA_FOLDER / dataset.base_table_id,
        header=0,
        engine="python",
        encoding="utf8",
        quotechar='"',
        escapechar="\\",
    )

    if autogluon:
        best_result, all_results = run_auto_gluon(
            approach=Result.BASE,
            dataframe=dataframe,
            target_column=dataset.target_column,
            data_label=dataset.base_table_label,
            join_name=dataset.base_table_label,
            algorithms_to_run=hyper_parameters,
        )
        return all_results
    else:
        entry = train_test_cart(
            train_data=dataframe,
            target_column=dataset.target_column,
            regression=dataset.dataset_type,
        )
        entry.approach = Result.BASE
        entry.data_label = dataset.base_table_label
        entry.data_path = dataset.base_table_label

        return [entry]


def get_arda_results(dataset: Dataset, sample_size: int = 3000, autogluon: bool = True) -> List:
    from feature_discovery.arda.arda import select_arda_features_budget_join

    print(f"ARDA result on table {dataset.base_table_id}")

    start = time.time()
    (
        dataframe,
        dataframe_label,
        selected_features,
        join_name,
    ) = select_arda_features_budget_join(
        base_node_id=str(dataset.base_table_id),
        target_column=dataset.target_column,
        sample_size=sample_size,
        regression=dataset.dataset_type,
    )
    end = time.time()
    print(f"X shape: {dataframe.shape}\nSelected features:\n\t{selected_features}")

    if len(selected_features) == 0:
        from feature_discovery.algorithms import CART

        entry = Result(
            algorithm=CART.LABEL,
            accuracy=0,
            feature_importance={},
            feature_selection_time=end - start,
            approach=Result.ARDA,
            data_label=dataset.base_table_label,
            data_path=join_name,
            join_path_features=selected_features,
        )
        entry.total_time += entry.feature_selection_time
        return [entry]

    features = selected_features.copy()
    features.append(dataset.target_column)

    if autogluon:
        best_result, all_results = run_auto_gluon(
            approach=Result.ARDA,
            dataframe=dataframe[features],
            target_column=dataset.target_column,
            data_label=dataset.base_table_label,
            join_name=join_name,
            algorithms_to_run=hyper_parameters,
        )
        for result in all_results:
            result.feature_selection_time = end - start
            result.total_time += result.feature_selection_time
        return all_results
    else:
        entry = train_test_cart(
            train_data=dataframe[features],
            target_column=dataset.target_column,
            regression=dataset.dataset_type,
        )
        entry.feature_selection_time = end - start
        entry.total_time += entry.feature_selection_time
        entry.approach = Result.ARDA
        entry.data_label = dataset.base_table_label
        entry.data_path = join_name
        entry.join_path_features = selected_features

        print(entry)

        return [entry]


def get_tfd_results(dataset: Dataset, value_ratio: float = 0.55, auto_gluon: bool = True) -> List:
    print(f"BFS result with table {dataset.base_table_id}")

    start = time.time()
    bfs_traversal = BfsAugmentation(
        base_table_label=dataset.base_table_label,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        auto_gluon=auto_gluon,
    )
    bfs_traversal.bfs_traverse_join_pipeline(queue={str(dataset.base_table_id)})
    end = time.time()

    print("FINISHED BFS")

    # Aggregate results
    results = []
    for join_name in bfs_traversal.join_name_mapping.keys():
        aux = list(filter(lambda x: x.data_path == join_name, bfs_traversal.all_results))

        for item in aux:
            item.feature_selection_time = end - start
            item.total_time += item.feature_selection_time
            results.append(item)

    # Save results
    pd.DataFrame(results).to_csv(
        RESULTS_FOLDER / f"results_{dataset.base_table_label}_bfs_{value_ratio}_autogluon_all_mixed.csv",
        index=False,
    )
    pd.DataFrame.from_dict(bfs_traversal.join_name_mapping, orient="index", columns=["join_name"]).to_csv(
        RESULTS_FOLDER / f"join_mapping_{dataset.base_table_label}_bfs_{value_ratio}_autogluon_all_mixed.csv"
    )

    return results


def get_classification_results(
    value_ratio: float,
    dataset_labels: Optional[List[str]] = None,
    results_file: str = "all_results_autogluon.csv",
):
    all_results = []
    datasets = filter_datasets(dataset_labels)

    for dataset in datasets:
        result_bfs = get_tfd_results(dataset, value_ratio=value_ratio)
        all_results.extend(result_bfs)
        result_base = get_base_results(dataset)
        all_results.extend(result_base)
        result_arda = get_arda_results(dataset)
        all_results.extend(result_arda)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


def get_results_ablation_classification(value_ratio: float):
    all_results = []

    result = ablation_study_enumerate_paths(CLASSIFICATION_DATASETS, value_ratio=value_ratio)
    all_results.append(result)

    result = ablation_study_enumerate_and_join(CLASSIFICATION_DATASETS, value_ratio=value_ratio)
    all_results.append(result)

    result = ablation_study_prune_paths(CLASSIFICATION_DATASETS, value_ratio=value_ratio)
    all_results.append(result)

    result = ablation_study_feature_selection(CLASSIFICATION_DATASETS, value_ratio=value_ratio)
    all_results.append(result)

    result = ablation_study_prune_join_key_level(CLASSIFICATION_DATASETS, value_ratio=value_ratio)
    all_results.append(result)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"ablation_study_{value_ratio}_autogluon.csv", index=False)


def get_results_tune_value_ratio_classification():
    all_results = []
    value_ratio_threshold = np.arange(0.15, 1.05, 0.05)
    for threshold in value_ratio_threshold:
        for dataset in CLASSIFICATION_DATASETS:
            result_bfs = get_tfd_results(dataset, value_ratio=threshold)
            all_results.extend(result_bfs)
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"all_results_value_ratio_tuning.csv", index=False)
