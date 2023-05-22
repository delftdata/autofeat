import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import tqdm

from feature_discovery.augmentation.bfs_pipeline import BfsAugmentation
from feature_discovery.augmentation.trial_error import run_auto_gluon, train_test_cart
from feature_discovery.config import DATA_FOLDER, RESULTS_FOLDER, JOIN_RESULT_FOLDER
from feature_discovery.data_preparation.dataset_base import Dataset
from feature_discovery.data_preparation.utils import get_path_length
from feature_discovery.experiments.ablation_experiments import (
    ablation_study_enumerate_paths,
    ablation_study_enumerate_and_join,
    ablation_study_prune_paths,
    ablation_study_feature_selection,
    ablation_study_prune_join_key_level,
)
from feature_discovery.experiments.result_object import Result
from feature_discovery.graph_processing.neo4j_transactions import export_dataset_connections, export_all_connections
from feature_discovery.tfd_datasets.init_datasets import CLASSIFICATION_DATASETS, init_datasets

logging.getLogger().setLevel(logging.WARNING)

hyper_parameters = {"RF": {}, "GBM": {}, "XGB": {}, "XT": {}}
# hyper_parameters = {"GBM": {}}

init_datasets()


def filter_datasets(dataset_labels: Optional[List[str]] = None) -> List[Dataset]:
    # `is None` is missing on purpose, because typer cannot return None default values for lists, only []
    if not dataset_labels:
        datasets = CLASSIFICATION_DATASETS
    else:
        datasets = [dataset for dataset in CLASSIFICATION_DATASETS if dataset.base_table_label in dataset_labels]

    return datasets


def get_base_results(dataset: Dataset, autogluon: bool = True):
    logging.debug(f"Base result on table {dataset.base_table_id}")

    dataframe = pd.read_csv(
        DATA_FOLDER / dataset.base_table_id,
        header=0,
        engine="python",
        encoding="utf8",
        quotechar='"',
        escapechar="\\",
    )

    if not autogluon:
        logging.debug(f"Base result on table {dataset.base_table_id}")

        entry = train_test_cart(
            train_data=dataframe,
            target_column=dataset.target_column,
            regression=dataset.dataset_type,
        )
        entry.approach = Result.BASE
        entry.data_label = dataset.base_table_label
        entry.data_path = dataset.base_table_label

        return [entry]

    features = list(dataframe.columns)
    if "Key_0_0" in features:
        features.remove("Key_0_0")

    _, results = run_auto_gluon(
        approach=Result.BASE,
        dataframe=dataframe[features],
        target_column=dataset.target_column,
        data_label=dataset.base_table_label,
        join_name=dataset.base_table_label,
        algorithms_to_run=hyper_parameters,
    )

    # Save intermediate results
    pd.DataFrame(results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_base.csv", index=False)

    return results


def get_arda_results(dataset: Dataset, sample_size: int = 3000, autogluon: bool = True) -> List:
    from feature_discovery.arda.arda import select_arda_features_budget_join

    logging.debug(f"ARDA result on table {dataset.base_table_id}")

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
    logging.debug(f"X shape: {dataframe.shape}\nSelected features:\n\t{selected_features}")

    if len(selected_features) == 0:
        logging.debug("No selected features ... ")
        entry = Result(
            algorithm="",
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

    if not autogluon:
        logging.debug("Running CART on ARDA feature selection ... ")
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

        return [entry]

    logging.debug(f"Running on ARDA Feature Selection result with AutoGluon")
    _, results = run_auto_gluon(
        approach=Result.ARDA,
        dataframe=dataframe[features],
        target_column=dataset.target_column,
        data_label=dataset.base_table_label,
        join_name=join_name,
        algorithms_to_run=hyper_parameters,
    )
    for result in results:
        result.feature_selection_time = end - start
        result.total_time += result.feature_selection_time

    pd.DataFrame(results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_arda.csv", index=False)

    return results


def evaluate_paths(bfs_result: BfsAugmentation, top_k: int, feat_sel_time: float):
    logging.debug(f"Evaluate top-{top_k} paths ... ")
    sorted_paths = sorted(bfs_result.ranking.items(), key=lambda r: (r[1], -get_path_length(r[0])), reverse=True)
    top_k_paths = sorted_paths if len(sorted_paths) < top_k else sorted_paths[:top_k]

    all_results = []
    for path in tqdm.tqdm(top_k_paths):
        join_name, _ = path
        if join_name == bfs_result.base_node_label:
            continue

        dataframe = pd.read_csv(JOIN_RESULT_FOLDER / bfs_result.join_name_mapping[join_name], header=0,
                                engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        features = bfs_result.partial_join_selected_features[join_name]
        features.append(bfs_result.target_column)
        logging.debug(f"Feature before join_key removal:\n{features}")
        features = list(set(features) - set(bfs_result.join_keys[join_name]))
        logging.debug(f"Feature after join_key removal:\n{features}")

        best_model, results = run_auto_gluon(approach=Result.TFD,
                                             dataframe=dataframe[features],
                                             target_column=bfs_result.target_column,
                                             data_label=bfs_result.base_table_label,
                                             join_name=join_name,
                                             algorithms_to_run=hyper_parameters,
                                             value_ratio=bfs_result.value_ratio)
        for result in results:
            result.feature_selection_time = feat_sel_time
            result.total_time += feat_sel_time

        all_results.extend(results)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{bfs_result.base_table_label}_tfd.csv", index=False)
    return all_results, top_k_paths


def get_tfd_results(dataset: Dataset, top_k: int = 10, value_ratio: float = 0.55, auto_gluon: bool = True) -> List:
    logging.debug(f"Running on TFD (Transitive Feature Discovery) result with AutoGluon")

    start = time.time()
    bfs_traversal = BfsAugmentation(
        base_table_label=dataset.base_table_label,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        auto_gluon=auto_gluon,
    )
    bfs_traversal.bfs_traverse_join_pipeline(queue={str(dataset.base_table_id)})
    end = time.time()

    logging.debug("FINISHED BFS")

    all_results, top_k_paths = evaluate_paths(bfs_result=bfs_traversal, top_k=top_k, feat_sel_time=end-start)
    logging.debug(top_k_paths)

    logging.debug("Save results ... ")
    pd.DataFrame(all_results).to_csv(
        RESULTS_FOLDER / f"results_{dataset.base_table_label}_bfs_{value_ratio}_autogluon_all_mixed.csv",
        index=False,
    )
    pd.DataFrame(top_k_paths, columns=['path', 'score']).to_csv(
        f"paths_tfd_{dataset.base_table_label}_{value_ratio}.csv", index=False)

    return all_results


def get_classification_results(
        value_ratio: float,
        dataset_labels: Optional[List[str]] = None,
        results_file: str = "all_results_autogluon.csv",
):
    all_results = []
    datasets = filter_datasets(dataset_labels)

    for dataset in tqdm.tqdm(datasets):
        result_bfs = get_tfd_results(dataset, value_ratio=value_ratio)
        all_results.extend(result_bfs)
        result_base = get_base_results(dataset)
        all_results.extend(result_base)
        result_arda = get_arda_results(dataset)
        all_results.extend(result_arda)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


def get_results_ablation_classification(value_ratio: float, dataset_labels: List[Dataset], results_filename: str,
                                        ml_model: dict):
    all_results = []

    for params in tqdm.tqdm(ml_model.items()):
        logging.debug(f"Running ablation study with algorithm {params[0]} with AutoGluon")
        hyper_param = dict([params])

        result = ablation_study_enumerate_paths(dataset_labels, value_ratio=value_ratio, ml_model=hyper_param)
        all_results.append(result)

        result = ablation_study_enumerate_and_join(dataset_labels, value_ratio=value_ratio, ml_model=hyper_param)
        all_results.append(result)

        result = ablation_study_prune_paths(dataset_labels, value_ratio=value_ratio, ml_model=hyper_param)
        all_results.append(result)

        result = ablation_study_feature_selection(dataset_labels, value_ratio=value_ratio, ml_model=hyper_param)
        all_results.append(result)

        result = ablation_study_prune_join_key_level(dataset_labels, value_ratio=value_ratio, ml_model=hyper_param)
        all_results.append(result)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_filename, index=False)


def get_results_tune_value_ratio_classification(datasets: List[Dataset], results_filename: str):
    all_results = []
    value_ratio_threshold = np.arange(0.15, 1.05, 0.05)
    for threshold in value_ratio_threshold:
        for dataset in datasets:
            result_bfs = get_tfd_results(dataset, value_ratio=threshold)
            all_results.extend(result_bfs)
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_filename, index=False)


def export_neo4j_connections(dataset_label: str = None):
    if dataset_label:
        result = export_dataset_connections(dataset_label)
    else:
        result = export_all_connections()

    pd.DataFrame(result).to_csv("all_connections.csv", index=False)


def transform_arff_to_csv(dataset_label: str, dataset_name: str):
    from scipy.io import arff
    data = arff.loadarff(DATA_FOLDER / dataset_label / dataset_name)
    dataframe = pd.DataFrame(data[0])
    catCols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    dataframe[catCols] = dataframe[catCols].apply(lambda x: x.str.decode('utf8'))
    dataframe.to_csv(DATA_FOLDER / dataset_label / f"{dataset_label}_original.csv", index=False)


if __name__ == "__main__":
    # transform_arff_to_csv("bioresponse", "bioresponse_dataset.arff")
    dataset = filter_datasets(["covertype"])[0]
    # get_tfd_results(dataset, value_ratio=0.65)
    # get_arda_results(dataset)
    get_base_results(dataset)
