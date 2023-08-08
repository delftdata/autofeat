from typing import Optional, List

from feature_discovery.experiments.dataset_object import CLASSIFICATION, REGRESSION, Dataset
from feature_discovery.experiments.init_datasets import CLASSIFICATION_DATASETS, REGRESSION_DATASETS, ALL_DATASETS


def filter_datasets(dataset_labels: Optional[List[str]] = None, problem_type: Optional[str] = None) -> List[Dataset]:
    # `is None` is missing on purpose, because typer cannot return None default values for lists, only []
    if problem_type == CLASSIFICATION:
        return CLASSIFICATION_DATASETS if not dataset_labels else [dataset for dataset in CLASSIFICATION_DATASETS if
                                                                   dataset.base_table_label in dataset_labels]
    if problem_type == REGRESSION:
        return REGRESSION_DATASETS if not dataset_labels else [dataset for dataset in REGRESSION_DATASETS if
                                                               dataset.base_table_label in dataset_labels]
    if not problem_type and dataset_labels:
        return [dataset for dataset in ALL_DATASETS if dataset.base_table_label in dataset_labels]

    return ALL_DATASETS
