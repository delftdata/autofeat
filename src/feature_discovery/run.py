import json
import logging
import time
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from feature_discovery.augmentation.autofeat import AutoFeat
from feature_discovery.augmentation.bfs_pipeline import BfsAugmentation
from feature_discovery.augmentation.trial_error import run_auto_gluon, train_test_cart
from feature_discovery.config import DATA_FOLDER, RESULTS_FOLDER, JOIN_RESULT_FOLDER
from feature_discovery.data_preparation.dataset_base import Dataset, REGRESSION, CLASSIFICATION
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
from feature_discovery.helpers.util_functions import get_df_with_prefix
from feature_discovery.tfd_datasets.init_datasets import CLASSIFICATION_DATASETS, init_datasets, ALL_DATASETS, \
    REGRESSION_DATASETS
from feature_discovery.visualisations.plot_data import parse_data, plot_accuracy, plot_time

logging.getLogger().setLevel(logging.WARNING)

hyper_parameters = {"RF": {}, "GBM": {}, "XGB": {}, "XT": {}}
# hyper_parameters = {"GBM": {}}

init_datasets()


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
        problem_type=dataset.dataset_type,
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
    start_ag = time.time()
    _, results = run_auto_gluon(
        approach=Result.ARDA,
        dataframe=dataframe[features],
        target_column=dataset.target_column,
        data_label=dataset.base_table_label,
        join_name=join_name,
        algorithms_to_run=hyper_parameters,
        problem_type=dataset.dataset_type
    )
    end_ag = time.time()
    for result in results:
        result.feature_selection_time = end - start
        result.train_time = end_ag - start_ag
        result.total_time += result.feature_selection_time
        result.total_time += result.train_time

    pd.DataFrame(results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_arda.csv", index=False)

    return results


def evaluate_paths(bfs_result: AutoFeat, top_k: int, feat_sel_time: float, problem_type: str, top_k_paths: int=15):
    logging.debug(f"Evaluate top-{top_k_paths} paths ... ")
    sorted_paths = sorted(bfs_result.ranking.items(), key=lambda r: (r[1], -get_path_length(r[0])), reverse=True)
    top_k_path_list = sorted_paths if len(sorted_paths) < top_k_paths else sorted_paths[:top_k_paths]
    base_features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]

    all_results = []
    for path in tqdm.tqdm(top_k_path_list):
        print(path)
        dataframe = None
        join_name, rank = path
        if join_name == bfs_result.base_table_id:
            continue

        features = bfs_result.partial_join_selected_features[join_name]
        features.append(bfs_result.target_column)
        features.extend(base_features)
        logging.debug(f"Feature before join_key removal:\n{features}")
        # features = list((set(features) - set(bfs_result.join_keys[join_name])).intersection(set(dataframe.columns)))
        # logging.debug(f"Feature after join_key removal:\n{features}")

        features_tables = [f"{feat.split('.csv')[0]}.csv" for feat in features]
        features_tables.sort()
        features_tables = set(features_tables)

        path_tables = {}
        for p in join_name.split("--"):
            aux = p.split("-")
            if len(aux) == 4:
                path_tables[aux[3]] = (aux[0], aux[1], aux[2], aux[3])

        path_list = []
        for table in features_tables:
            if table in path_list:
                continue
            path_aux = create_join_tree(table, path_tables)
            if not (type(path_aux) is list) and (path_aux not in path_tables.keys()):
                continue
            path_list.append(path_aux)

        dataframe = join_from_path(path_list, bfs_result.target_column, bfs_result.base_table_id)
        print(dataframe.shape)
        features = list(set(features).intersection(set(dataframe.columns)))

        if len(features) < 2:
            features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]
            features.append(bfs_result.target_column)

        start = time.time()
        best_model, results = run_auto_gluon(approach=Result.TFD,
                                             dataframe=dataframe[features],
                                             target_column=bfs_result.target_column,
                                             data_label=bfs_result.base_table_label,
                                             join_name=join_name,
                                             algorithms_to_run=hyper_parameters,
                                             value_ratio=bfs_result.value_ratio,
                                             problem_type=problem_type)
        end = time.time()
        for result in results:
            result.feature_selection_time = feat_sel_time
            result.train_time = end - start
            result.total_time += feat_sel_time
            result.total_time += end - start
            result.rank = rank
            result.top_k = top_k
            result.data_path = path_list

        all_results.extend(results)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{bfs_result.base_table_label}_tfd.csv", index=False)
    return all_results, top_k_path_list


def join_from_path(path, target, base_node):
    join_path = ''
    step = 3
    joined_df = None
    for p in path:
        if ".".join(p) in join_path:
            continue
        for i, el in enumerate(p[::step]):
            if (i * step) + 1 == len(p):
                continue

            aux = ".".join([p[i], p[i + 1], p[i + 2], p[(i + 1) * step]])
            if aux in join_path:
                continue

            if p[i * step] not in join_path:
                if p[i * step] == base_node:
                    left_table, _ = get_df_with_prefix(p[i * step], target)
                else:
                    left_table, _ = get_df_with_prefix(p[i * step])
            else:
                left_table = joined_df

            right_table, _ = get_df_with_prefix(p[(i + 1) * step])
            to_column = f"{p[(i+1) * step]}.{p[i * step + 2]}"
            right_table = right_table.groupby(to_column).sample(n=1, random_state=42)

            joined_df = pd.merge(
                left_table,
                right_table,
                how="left",
                left_on=f"{p[i * step]}.{p[i * step + 1]}",
                right_on=f"{p[(i+1) * step]}.{p[i * step + 2]}",
            )

            join_path += aux

    return joined_df


def create_join_tree(table, path_tables):
    # print(f"\t{table}")
    if table in path_tables.keys():
        value = path_tables[table]
        result = create_join_tree(value[0], path_tables)
        if type(result) is not list:
            result = [result]

        result.append(value[1])
        result.append(value[2])
        result.append(value[3])
        return result

    return table


def get_tfd_results(dataset: Dataset, top_k: int = 10, value_ratio: float = 0.55, auto_gluon: bool = True,
                    join_all: bool = True) -> List:
    logging.debug(f"Running on TFD (Transitive Feature Discovery) result with AutoGluon")

    start = time.time()
    bfs_traversal = None
    if join_all:
        # bfs_traversal.join_all_recursively(queue={str(dataset.base_table_id)})
        # bfs_traversal.streaming_feature_selection(queue={str(dataset.base_table_id)})
        bfs_traversal = AutoFeat(
            base_table_id=str(dataset.base_table_id),
            base_table_label=dataset.base_table_label,
            target_column=dataset.target_column,
            value_ratio=value_ratio,
            top_k=top_k,
            task=dataset.dataset_type
        )
        bfs_traversal.streaming_feature_selection(queue={str(dataset.base_table_id)})
    else:
        bfs_traversal = BfsAugmentation(
            base_table_label=dataset.base_table_label,
            target_column=dataset.target_column,
            value_ratio=value_ratio,
            auto_gluon=auto_gluon,
        )
        bfs_traversal.bfs_traverse_join_pipeline(queue={str(dataset.base_table_id)})
    end = time.time()

    logging.debug("FINISHED BFS")

    all_results, top_k_paths = evaluate_paths(bfs_result=bfs_traversal,
                                              top_k=top_k,
                                              feat_sel_time=end - start,
                                              problem_type=dataset.dataset_type)
    logging.debug(top_k_paths)

    logging.debug("Save results ... ")
    # pd.DataFrame(all_results).to_csv(
    #     RESULTS_FOLDER / f"results_{dataset.base_table_label}_bfs_{value_ratio}_autogluon_all_mixed.csv",
    #     index=False,
    # )
    pd.DataFrame(top_k_paths, columns=['path', 'score']).to_csv(
        f"paths_tfd_{dataset.base_table_label}_{value_ratio}.csv", index=False)

    return all_results


def get_all_results(
        value_ratio: float,
        problem_type: Optional[str],
        dataset_labels: Optional[List[str]] = None,
        results_file: str = "all_results_autogluon.csv",
):
    all_results = []
    datasets = filter_datasets(dataset_labels, problem_type)

    for dataset in tqdm.tqdm(datasets):
        result_bfs = get_tfd_results(dataset, value_ratio=value_ratio, join_all=True)
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
    value_ratio_threshold = np.arange(1, 1.05, 0.05)
    for threshold in value_ratio_threshold:
        print(f"==== value_ratio = {threshold} ==== ")
        for dataset in datasets:
            print(f"\tDataset = {dataset.base_table_label} ==== ")
            result_bfs = get_tfd_results(dataset, value_ratio=threshold, top_k=15)
            all_results.extend(result_bfs)
        pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"value_ratio_{threshold}_{results_filename}", index=False)
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_filename, index=False)


def get_results_tune_k(datasets: List[Dataset], results_filename: str):
    all_results = []
    top_k = np.arange(1, 21, 1)
    for k in top_k:
        print(f"==== k = {k} ==== ")
        for dataset in datasets:
            print(f"\tDataset = {dataset.base_table_label} ==== ")
            result_bfs = get_tfd_results(dataset, value_ratio=0.65, top_k=k)
            all_results.extend(result_bfs)
        # pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"k_{k}_{results_filename}", index=False)
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_filename, index=False)


def export_neo4j_connections(dataset_label: str = None):
    if dataset_label:
        result = export_dataset_connections(dataset_label)
    else:
        result = export_all_connections()

    pd.DataFrame(result).to_csv(RESULTS_FOLDER / "all_connections-basicdd.csv", index=False)


def transform_arff_to_csv(dataset_label: str, dataset_name: str):
    from scipy.io import arff
    data = arff.loadarff(DATA_FOLDER / dataset_label / dataset_name)
    dataframe = pd.DataFrame(data[0])
    catCols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    dataframe[catCols] = dataframe[catCols].apply(lambda x: x.str.decode('utf8'))
    dataframe.to_csv(DATA_FOLDER / dataset_label / f"{dataset_label}_original.csv", index=False)


def plot(results_file_name: str):
    dataframe = pd.read_csv(
        RESULTS_FOLDER / results_file_name, header=0, sep=";", engine="python", encoding="utf8", quotechar='"',
        escapechar="\\")
    dataframe = parse_data(dataframe)
    plot_accuracy(dataframe)
    plot_time(dataframe)


if __name__ == "__main__":
    # transform_arff_to_csv("superconduct", "superconduct_dataset.arff")
    dataset = filter_datasets(["covertype"])[0]
    get_tfd_results(dataset, value_ratio=0.65, join_all=True, top_k=15)
    # get_arda_results(dataset)
    # get_base_results(dataset)
    # export_neo4j_connections()
