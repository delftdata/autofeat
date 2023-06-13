from typing import Optional, List

import pandas as pd
import typer
import tqdm
from typing_extensions import Annotated

from feature_discovery.augmentation.data_preparation_pipeline import ingest_data_with_pk_fk
from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.data_preparation.ingest_data import ingest_nodes, profile_valentine_all
from feature_discovery.run import (
    filter_datasets,
    get_arda_results,
    get_base_results,
    get_tfd_results,
    get_all_results,
    get_results_ablation_classification,
    get_results_tune_value_ratio_classification,
    plot,
)
from feature_discovery.tfd_datasets.init_datasets import ALL_DATASETS

app = typer.Typer()


@app.command()
def run_arda(
    dataset_labels: Annotated[
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
    ] = None,
    problem_type: Annotated[
        str, typer.Option(help="Type of prediction problem: binary, regression, None (automatically detect)")
    ] = None,
    results_file: Annotated[str, typer.Option(help="CSV file where the results will be written")] = "results_arda.csv",
):
    """Runs the ARDA experiments."""
    all_results = []
    datasets = filter_datasets(dataset_labels, problem_type)
    for dataset in tqdm.tqdm(datasets):
        all_results.extend(get_arda_results(dataset))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_base(
    dataset_labels: Annotated[
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
    ] = None,
    problem_type: Annotated[
        str, typer.Option(help="Type of prediction problem: binary, regression, None (automatically detect)")
    ] = None,
    results_file: Annotated[str, typer.Option(help="CSV file where the results will be written")] = "results_base.csv",
):
    """Runs the base experiments."""
    all_results = []
    datasets = filter_datasets(dataset_labels, problem_type)
    for dataset in tqdm.tqdm(datasets):
        all_results.extend(get_base_results(dataset))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_tfd(
    top_k: Annotated[int, typer.Option(help="Number of results (paths)")] = 10,
    dataset_labels: Annotated[
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
    ] = None,
    problem_type: Annotated[
        str, typer.Option(help="Type of prediction problem: binary, regression, None (automatically detect)")
    ] = None,
    results_file: Annotated[str, typer.Option(help="CSV file where the results will be written")] = "results_tfd.csv",
    value_ratio: Annotated[float, typer.Option(help="Value ratio to be used in the TFD experiments")] = 0.65,
    join_all: Annotated[bool, typer.Option(help="Whether to use Join-All-Recursively strategy")] = True,
):
    """Runs the TFD experiments."""
    all_results = []
    datasets = filter_datasets(dataset_labels, problem_type)
    for dataset in tqdm.tqdm(datasets):
        all_results.extend(get_tfd_results(dataset, top_k, value_ratio, join_all=join_all))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_all(
    dataset_labels: Annotated[
        Optional[List[str]],
        typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels"),
    ] = None,
    problem_type: Annotated[
        str, typer.Option(help="Type of prediction problem: binary, regression, None (automatically detect)")
    ] = None,
    results_file: Annotated[
        str, typer.Option(help="CSV file where the results will be written")
    ] = "all_results_autogluon.csv",
    value_ratio: Annotated[float, typer.Option(help="Value ratio to be used in the TFD experiments")] = 0.65,
):
    """Runs all experiments (ARDA + base + TFD)."""
    get_all_results(value_ratio, problem_type, dataset_labels, results_file)


@app.command()
def run_ablation(
    value_ratio: Annotated[float, typer.Option(help="Value ratio to be used in the TFD experiments")] = 0.65,
    dataset_labels: Annotated[
        Optional[List[str]],
        typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels"),
    ] = None,
    results_file: Annotated[
        str, typer.Option(help="CSV file where the results will be written")
    ] = "ablation_study_autogluon.csv",
    ml_model: Annotated[str, typer.Option(help="Model name from AutoGluon ML Hyper-parameters")] = 'GBM',
):
    """Run all the ablation study experiments"""
    datasets = filter_datasets(dataset_labels)
    get_results_ablation_classification(value_ratio, datasets, results_file, {ml_model: {}})


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
def ingest_data(
    dataset_label: Annotated[
        Optional[str],
        typer.Option(help="The label of the dataset to ingest"),
    ] = None,
    discover_connections_dataset: Annotated[
        bool, typer.Option(help="Run dataset discovery to find more connections within the dataset")
    ] = False,
    discover_connections_data_lake: Annotated[
        bool, typer.Option(help="Run dataset discovery to find more connections within the entire data lake")
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
        ingest_data_with_pk_fk(
            dataset=dataset, profile_valentine=discover_connections_dataset, mix_datasets=discover_connections_data_lake
        )


@app.command()
def ingest_all_data(
    data_discovery_threshold: Annotated[
        float,
        typer.Option(
            help="Run dataset discovery to find more connections within the entire data lake with given"
            " accuracy rate threshold"
        ),
    ] = None,
):
    """
    Ingest all dataset from "data" folder.
    """
    ingest_nodes()

    if data_discovery_threshold:
        profile_valentine_all(valentine_threshold=data_discovery_threshold)


@app.command()
def plot_data(
    results_filename: Annotated[
        str, typer.Option(help="The name of the file with results which is in result folder")
    ] = "all_results_autogluon.csv",
):
    """
    Plots the accuracy and time given the results.
    """
    plot(results_filename)


if __name__ == "__main__":
    app()
