from typing import List

from feature_discovery.augmentation.bfs_pipeline import BfsAugmentation
from feature_discovery.data_preparation.dataset_base import Dataset


def ablation_study_enumerate_paths(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.enumerate_all_paths(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time

    return results


def ablation_study_prune_paths(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate_prune"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.prune_paths(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time

    return results


def ablation_study_enumerate_and_join(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate_join"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.enumerate_and_join(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time
    return results


def ablation_study_feature_selection(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate_join_prune_fs"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.apply_feature_selection(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time
    return results


def ablation_study_prune_join_key_level(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate_join_prune_fs_rank_jk"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.prune_join_key_level(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time
    return results
