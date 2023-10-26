import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import tqdm

from feature_discovery.config import DATA_FOLDER, RESULTS_FOLDER, ROOT_FOLDER
from feature_discovery.experiments.ablation import autofeat
from feature_discovery.experiments.baselines import non_augmented, join_all_bfs, join_all_dfs, arda
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.experiments.init_datasets import init_datasets
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets
from feature_discovery.graph_processing.neo4j_transactions import export_dataset_connections, export_all_connections

logging.getLogger().setLevel(logging.WARNING)

hyper_parameters = {"RF": {}, "GBM": {}, "XGB": {}, "XT": {}}

init_datasets()


def get_base_results(dataset: Dataset):
    logging.debug(f"Base result on table {dataset.base_table_id}")

    dataframe = pd.read_csv(
        DATA_FOLDER / dataset.base_table_id,
        header=0,
        engine="python",
        encoding="utf8",
        quotechar='"',
        escapechar="\\",
    )

    results = non_augmented(dataframe=dataframe, dataset=dataset)

    # Save intermediate results
    pd.DataFrame(results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_base.csv", index=False)

    return results


def get_join_all_results(dataset: Dataset):
    results_bfs = join_all_bfs(dataset)
    results_dfs = join_all_dfs(dataset)
    results_bfs.extend(results_dfs)

    # Save intermediate results
    pd.DataFrame(results_bfs).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_join_all.csv", index=False)
    return results_bfs


def get_arda_results(dataset: Dataset, sample_size: int = 3000) -> List:
    results = arda(dataset, sample_size)

    pd.DataFrame(results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_arda.csv", index=False)

    return results


def get_tfd_results(dataset: Dataset, top_k: int = 15, value_ratio: float = 0.65) -> List:
    spearman_mrmr_results, top_k_paths = autofeat(dataset=dataset, top_k=top_k, value_ratio=value_ratio)

    logging.debug("Save results ... ")
    pd.DataFrame(top_k_paths, columns=['path', 'score']).to_csv(
        f"paths_tfd_{dataset.base_table_label}_{value_ratio}.csv", index=False)
    pd.DataFrame(spearman_mrmr_results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_tfd.csv", index=False)

    return spearman_mrmr_results


def get_autofeat_ablation(dataset: Dataset, top_k: int = 15, value_ratio: float = 0.65):
    all_results = []
    pearson_mrmr_results, _ = autofeat(dataset=dataset, top_k=top_k, value_ratio=value_ratio,
                                       approach=Result.TFD_Pearson, pearson=True)
    all_results.extend(pearson_mrmr_results)
    pearson_jmi_results, _ = autofeat(dataset=dataset, top_k=top_k, value_ratio=value_ratio,
                                      approach=Result.TFD_Pearson_JMI, pearson=True, jmi=True)
    all_results.extend(pearson_jmi_results)
    spearman_jmi_results, _ = autofeat(dataset=dataset, top_k=top_k, value_ratio=value_ratio,
                                       approach=Result.TFD_JMI, jmi=True)
    all_results.extend(spearman_jmi_results)

    logging.debug("Save results ... ")
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_tfd_ablation.csv", index=False)
    return all_results


def get_autofeat_disable_components(dataset: Dataset, top_k: int = 15, value_ratio: float = 0.65):
    all_results = []
    non_redundant_features, _ = autofeat(dataset=dataset, top_k=top_k, value_ratio=value_ratio,
                                         approach=Result.TFD_RED, no_relevance=True)
    all_results.extend(non_redundant_features)
    relevant_features, _ = autofeat(dataset=dataset, top_k=top_k, value_ratio=value_ratio,
                                    approach=Result.TFD_REL, no_redundancy=True)
    all_results.extend(relevant_features)

    logging.debug("Save results ... ")
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_tfd_disable_components.csv",
                                     index=False)
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
        result_bfs = get_tfd_results(dataset, value_ratio=value_ratio)
        all_results.extend(result_bfs)
        result_base = get_base_results(dataset)
        all_results.extend(result_base)
        result_arda = get_arda_results(dataset)
        all_results.extend(result_arda)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


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
        pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"k_{k}_{results_filename}", index=False)
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_filename, index=False)


def export_neo4j_connections(dataset_label: str = None):
    if dataset_label:
        result = export_dataset_connections(dataset_label)
    else:
        result = export_all_connections()

    pd.DataFrame(result).to_csv(RESULTS_FOLDER / "all_connections-basicdd.csv", index=False)


def transform_arff_to_csv(save_path: str, dataset_path: str):
    from scipy.io import arff
    data = arff.loadarff(ROOT_FOLDER / dataset_path)
    dataframe = pd.DataFrame(data[0])
    catCols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    dataframe[catCols] = dataframe[catCols].apply(lambda x: x.str.decode('utf8'))
    dataframe.to_csv(ROOT_FOLDER / save_path, index=False)


if __name__ == "__main__":
    # transform_arff_to_csv("original_data/original/miniboone_dataset.csv",
    #                       "original_data/originals/miniboone_dataset.arff")
    dataset = filter_datasets(["credit"])[0]
    # get_tfd_results(dataset, value_ratio=0.65, top_k=15)
    get_autofeat_ablation(dataset)
    # get_arda_results(dataset)
    # get_base_results(dataset)
    # export_neo4j_connections()
