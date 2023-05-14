from typing import Optional, List
from typing_extensions import Annotated

import pandas as pd
import typer

from feature_discovery.run import (
    filter_datasets,
    get_arda_results,
    get_base_results,
    get_tfd_results,
    get_classification_results,
)
from feature_discovery.config import RESULTS_FOLDER

app = typer.Typer()


@app.command()
def run_arda(
    dataset_labels: Annotated[
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
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
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
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
        Optional[List[str]], typer.Option(help="Whether to run only on a list of datasets. Filters by dataset labels")
    ] = None,
    results_file: Annotated[str, typer.Option(help="CSV file where the results will be written")] = "results_tfd.csv",
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


if __name__ == "__main__":
    app()
