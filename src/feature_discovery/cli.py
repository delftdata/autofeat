from typing import Optional, List

import pandas as pd
import typer
from typing_extensions import Annotated

from feature_discovery.augmentation.data_preparation_pipeline import ingest_data_with_pk_fk
from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.run import (
    filter_datasets,
    get_arda_results,
    get_base_results,
    get_tfd_results,
    get_classification_results, get_results_ablation_classification, get_results_tune_value_ratio_classification,
)

app = typer.Typer()


@app.command()
def run_arda(
        dataset_labels: Annotated[
            Optional[List[str]], typer.Option(
                help="Whether to run only on a list of datasets. Filters by dataset labels")
        ] = None,
        results_file: Annotated[
            str, typer.Option(help="CSV file where the results will be written")
        ] = "results_arda.csv",
):
    """Runs the ARDA experiments."""
    all_results = []
    datasets = filter_datasets(dataset_labels)
    for dataset in datasets:
        all_results.extend(get_arda_results(dataset))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_base(
        dataset_labels: Annotated[
            Optional[List[str]], typer.Option(
                help="Whether to run only on a list of datasets. Filters by dataset labels")
        ] = None,
        results_file: Annotated[
            str, typer.Option(help="CSV file where the results will be written")
        ] = "results_base.csv",
):
    """Runs the base experiments."""
    all_results = []
    datasets = filter_datasets(dataset_labels)
    for dataset in datasets:
        all_results.extend(get_base_results(dataset))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_tfd(
        dataset_labels: Annotated[
            Optional[List[str]], typer.Option(
                help="Whether to run only on a list of datasets. Filters by dataset labels")
        ] = None,
        results_file: Annotated[
            str, typer.Option(help="CSV file where the results will be written")] = "results_tfd.csv",
        value_ratio: Annotated[float, typer.Option(help="Value ratio to be used in the TFD experiments")] = 0.55,
        auto_gluon: Annotated[bool, typer.Option(help="Whether to use AutoGluon")] = True,
):
    """Runs the TFD experiments."""
    all_results = []
    datasets = filter_datasets(dataset_labels)
    for dataset in datasets:
        all_results.extend(get_tfd_results(dataset, value_ratio, auto_gluon))

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / results_file, index=False)


@app.command()
def run_all(
        dataset_labels: Annotated[
            Optional[List[str]],
            typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels"),
        ] = None,
        results_file: Annotated[
            str, typer.Option(help="CSV file where the results will be written")
        ] = "all_results_autogluon.csv",
        value_ratio: Annotated[float, typer.Option(help="Value ratio to be used in the TFD experiments")] = 0.55,
):
    """Runs all experiments (ARDA + base + TFD)."""
    get_classification_results(value_ratio, dataset_labels, results_file)


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
):
    """ Run all the ablation study experiments """
    datasets = filter_datasets(dataset_labels)
    get_results_ablation_classification(value_ratio, datasets, results_file)


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
    """ Run experiment on the sensitivity of value_ratio hyper-parameter"""
    datasets = filter_datasets(dataset_labels)
    get_results_tune_value_ratio_classification(datasets, results_file)


@app.command()
def ingest_data(
        dataset_label: Annotated[
            Optional[str],
            typer.Option(help="The label of the dataset to ingest"),
        ],
        discover_connections_dataset: Annotated[
            bool, typer.Option(help="Run dataset discovery to find more connections within the dataset")] = True,
        discover_connections_data_lake: Annotated[
            bool, typer.Option(help="Run dataset discovery to find more connections within the entire data lake")
        ] = False,
):
    """ Ingest the new dataset into neo4j database """
    datasets = filter_datasets([dataset_label])
    if len(datasets) == 0:
        raise typer.BadParameter(
            "Incorrect dataset label. The label should have the same value with <base_table_label>.")

    ingest_data_with_pk_fk(dataset=datasets[0],
                           profile_valentine=discover_connections_dataset,
                           mix_datasets=discover_connections_data_lake)


if __name__ == "__main__":
    app()
