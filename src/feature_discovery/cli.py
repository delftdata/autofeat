from typing import Optional, List

import numpy as np

np.random.seed(42)

import pandas as pd
import tqdm
import typer
from typing_extensions import Annotated

from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.dataset_relation_graph.dataset_discovery import profile_valentine_dataset, profile_valentine_all
from feature_discovery.dataset_relation_graph.ingest_data import ingest_nodes, ingest_data_with_pk_fk
from feature_discovery.experiments.init_datasets import ALL_DATASETS
from feature_discovery.experiments.utils_dataset import filter_datasets
from feature_discovery.run import (
    get_arda_results,
    get_base_results,
    get_tfd_results,
    get_all_results,
    get_results_tune_value_ratio_classification,
    get_results_tune_k,
    get_join_all_results,
    get_autofeat_ablation,
)

app = typer.Typer()


@app.command()
def run_arda(
    dataset_labels: Annotated[
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
    ] = None,
    algorithm: Annotated[str, typer.Option(help="ML algorithm for evaluation: [RF, GBM, XT, XGB, KNN, LR1]")] = None,
    results_file: Annotated[str, typer.Option(help="CSV file where the results will be written")] = "results_arda.csv",
):
    """Runs the ARDA experiments."""
    all_results = []
    datasets = filter_datasets(dataset_labels)
    for dataset in tqdm.tqdm(datasets):
        all_results.extend(get_arda_results(dataset, algorithm))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_base(
    dataset_labels: Annotated[
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
    ] = None,
    algorithm: Annotated[str, typer.Option(help="ML algorithm for evaluation: [RF, GBM, XT, XGB, KNN, LR1]")] = None,
    results_file: Annotated[str, typer.Option(help="CSV file where the results will be written")] = "results_base.csv",
):
    """Runs the base experiments."""
    all_results = []
    datasets = filter_datasets(dataset_labels)
    for dataset in tqdm.tqdm(datasets):
        all_results.extend(get_base_results(dataset, algorithm))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_join_all(
    dataset_labels: Annotated[
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
    ] = None,
    algorithm: Annotated[str, typer.Option(help="ML algorithm for evaluation: [RF, GBM, XT, XGB, KNN, LR1]")] = None,
    results_file: Annotated[
        str, typer.Option(help="CSV file where the results will be written")
    ] = "results_join_all.csv",
):
    """Runs 4 experiments: join all tables in BFS and DFS order, then same approaches with filter
    feature selection: spearman, and with wrapper feature selection: forward selection"""

    all_results = []
    datasets = filter_datasets(dataset_labels)
    for dataset in tqdm.tqdm(datasets):
        all_results.extend(get_join_all_results(dataset, algorithm))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_ablation(
    top_k: Annotated[int, typer.Option(help="Number of features and paths")] = 15,
    dataset_labels: Annotated[
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
    ] = None,
    algorithm: Annotated[str, typer.Option(help="ML algorithm for evaluation: [RF, GBM, XT, XGB, KNN, LR1]")] = None,
    results_file: Annotated[
        str, typer.Option(help="CSV file where the results will be written")
    ] = "results_tfd_ablation.csv",
    value_ratio: Annotated[float, typer.Option(help="Value ratio to be used in the TFD experiments")] = 0.65,
):
    """Runs the 5 types of TFD experiments: Spearman + JMI, Pearson + MRMR, Pearson + JMI,
    (no relevance) MRMR, (no redundancy) Spearman
    """
    all_results = []
    datasets = filter_datasets(dataset_labels)
    for dataset in tqdm.tqdm(datasets):
        all_results.extend(get_autofeat_ablation(dataset, algorithm, top_k, value_ratio))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_autofeat(
    top_k: Annotated[int, typer.Option(help="Number of features and paths")] = 15,
    dataset_labels: Annotated[
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
    ] = None,
    algorithm: Annotated[str, typer.Option(help="ML algorithm for evaluation: [RF, GBM, XT, XGB, KNN, LR]")] = None,
    results_file: Annotated[
        str, typer.Option(help="CSV file where the results will be written")
    ] = "results_autofeat.csv",
    value_ratio: Annotated[float, typer.Option(help="Value ratio to be used in the TFD experiments")] = 0.65,
):
    """Runs the AutoFeat experiments."""
    all_results = []
    datasets = filter_datasets(dataset_labels)

    for dataset in tqdm.tqdm(datasets):
        all_results.extend(get_tfd_results(dataset, algorithm, top_k, value_ratio))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_all(
    dataset_labels: Annotated[
        Optional[List[str]],
        typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels"),
    ] = None,
    algorithm: Annotated[str, typer.Option(help="ML algorithm for evaluation: [RF, GBM, XT, XGB, KNN, LR]")] = None,
    results_file: Annotated[
        str, typer.Option(help="CSV file where the results will be written")
    ] = "all_results_autogluon.csv",
    join_all: Annotated[
        bool, typer.Option(help="Run the Join-All experiments")
    ] = True,
):
    """Runs all experiments (ARDA + Base + TFD + Join-All)."""
    get_all_results(dataset_labels, algorithm, results_file, join_all)


@app.command()
def run_tune_value_ratio(
    dataset_labels: Annotated[
        Optional[List[str]],
        typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels"),
    ] = None,
    results_file: Annotated[
        str, typer.Option(help="CSV file where the results will be written")
    ] = "all_results_value_ratio_tuning.csv",
):
    """Run experiment on the sensitivity of value_ratio hyper-parameter"""
    datasets = filter_datasets(dataset_labels)
    get_results_tune_value_ratio_classification(datasets, results_file)


@app.command()
def run_tune_top_k(
    dataset_labels: Annotated[
        Optional[List[str]],
        typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels"),
    ] = None,
    results_file: Annotated[
        str, typer.Option(help="CSV file where the results will be written")
    ] = "all_results_top_k_features_tuning.csv",
):
    """Run experiment on the sensitivity of value_ratio hyper-parameter"""
    datasets = filter_datasets(dataset_labels)
    get_results_tune_k(datasets, results_file)


@app.command()
def ingest_kfk_data(
    dataset_label: Annotated[
        Optional[str],
        typer.Option(help="The label of the dataset to ingest"),
    ] = None,
    discover_connections_dataset: Annotated[
        bool, typer.Option(help="Run dataset discovery to find more connections within the dataset")
    ] = False,
):
    """Ingest the new dataset into neo4j database"""
    if not dataset_label:
        datasets = ALL_DATASETS
    else:
        datasets = filter_datasets([dataset_label])

    if len(datasets) == 0:
        raise typer.BadParameter(
            "Incorrect dataset label. The label should have the same value with <base_table_label>."
        )

    for dataset in tqdm.tqdm(datasets):
        ingest_data_with_pk_fk(dataset=dataset, profile_valentine=discover_connections_dataset)


@app.command()
def ingest_data(
    data_discovery_threshold: Annotated[
        float,
        typer.Option(
            help="Run dataset discovery to find more connections within the entire data lake with given"
            " accuracy rate threshold"
        ),
    ] = None,
    discover_connections_data_lake: Annotated[
        bool, typer.Option(help="Run dataset discovery to find more connections within the entire data lake")
    ] = False,
):
    """
    Ingest all dataset from specified "data" folder.
    """
    ingest_nodes()

    if data_discovery_threshold and discover_connections_data_lake:
        profile_valentine_all(valentine_threshold=data_discovery_threshold)
        return

    if data_discovery_threshold and not discover_connections_data_lake:
        for dataset in ALL_DATASETS:
            profile_valentine_dataset(dataset.base_table_label, valentine_threshold=data_discovery_threshold)


if __name__ == "__main__":
    app()
