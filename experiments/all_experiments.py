from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import PLOTS_FOLDER, MAPPING_FOLDER, ACCURACY_RESULTS_ALL_PNG
from data_preparation.dataset_base import Dataset
from experiments.arda_experiments import ArdaExperiment
from experiments.base_experiment import BaseExperiment
from experiments.datasets import Datasets
from experiments.result_object import Result
from experiments.tfd_experiments import TFDExperiment
from helpers.util_functions import objects_to_dict

approach_plot_order = {
    Result.TFD: 1,
    Result.TFD_PATH: 2,
    Result.ARDA: 3,
    Result.JOIN_ALL: 4,
    Result.JOIN_ALL_FS: 5,
    Result.BASE: 6,
}


class AllExperiments:
    def __init__(self):
        self.arda_experiments: Dict[Dataset, ArdaExperiment] = {}
        self.tfd_experiments: Dict[Dataset, TFDExperiment] = {}
        self.learning_curves_depth_values = [i for i in range(1, 21)]
        self.datasets: List[Dataset] = []
        self.dataset_results: Dict[Dataset, List] = {}

    def experiments_per_dataset(self, dataset: Dataset):
        self.datasets.append(dataset)
        if dataset not in self.dataset_results:
            self.dataset_results[dataset] = []

        queue = set()
        queue.update(BaseExperiment.__subclasses__())

        while len(queue) > 0:
            experiment_class = queue.pop()
            experiment = experiment_class(dataset, learning_curve_depth_values=self.learning_curves_depth_values)
            experiment.compute_results()
            # Save ARDA and TFD results for the learning curve plot
            if experiment.approach == Result.ARDA:
                self.arda_experiments[dataset] = experiment
            elif experiment.approach == Result.TFD:
                self.tfd_experiments[dataset] = experiment
            self.dataset_results[dataset].extend(objects_to_dict(experiment.results))
            if len(experiment_class.__subclasses__()) > 0:
                queue.update(experiment_class.__subclasses__())

    def plot_results(self, results_df: pd.DataFrame = None):
        columns = len(self.datasets)
        fig, axs = plt.subplots(nrows=1, ncols=columns, figsize=(12, 6))

        for i, dataset in enumerate(self.datasets):
            if results_df is None and dataset in self.dataset_results:
                results_df = pd.DataFrame(self.dataset_results[dataset])
            results_df.sort_values(by=['approach', 'algorithm'], key=lambda x: x.map(approach_plot_order).fillna(x),
                                   inplace=True)
            results_df.to_csv(MAPPING_FOLDER / f"acc-results-{dataset.base_table_label}.csv", index=False)

            if len(self.datasets) == 1:
                sns.barplot(data=results_df, x="algorithm", y="accuracy", hue="approach", ax=axs)
                axs.set_title(f"{dataset.base_table_label.title()}")
                axs.set_ylabel("Accuracy")
            else:
                sns.barplot(data=results_df, x="algorithm", y="accuracy", hue="approach", ax=axs[i])
                axs[i].set_title(f"{dataset.base_table_label.title()}")
                axs[i].set_ylabel("Accuracy")

        if len(self.datasets) == 1:
            handles, labels = axs.get_legend_handles_labels()
            axs.legend(handles, labels, bbox_to_anchor=(0, -0.25), loc=2, ncol=2, fontsize="xx-small")

            fig = axs.get_figure()
        else:
            handles, labels = axs[0].get_legend_handles_labels()
            axs[0].legend(handles, labels, bbox_to_anchor=(0, -0.25), loc=2, ncol=2, fontsize="xx-small")

            fig = axs[0].get_figure()

        fig.show()
        fig.savefig(ACCURACY_RESULTS_ALL_PNG, dpi=300, bbox_inches="tight")

    def plot_learning_curves(self):
        for dataset in self.datasets:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

            axs[0].set_title("Best Ranked Features")
            axs[0].plot(
                self.learning_curves_depth_values,
                self.tfd_experiments[dataset].learning_curve_train_tfd,
                "-o")
            axs[0].plot(
                self.learning_curves_depth_values,
                self.tfd_experiments[dataset].learning_curve_test_tfd,
                "-o")

            axs[1].set_title("All in path Features")
            axs[1].plot(
                self.learning_curves_depth_values,
                self.tfd_experiments[dataset].learning_curve_train_tfd_path,
                '-o')
            axs[1].plot(
                self.learning_curves_depth_values,
                self.tfd_experiments[dataset].learning_curve_test_tfd_path,
                '-o')

            axs[2].set_title("Arda")
            axs[2].plot(
                self.learning_curves_depth_values,
                self.arda_experiments[dataset].learning_curve_train,
                '-o',
                label='Train')
            axs[2].plot(
                self.learning_curves_depth_values,
                self.arda_experiments[dataset].learning_curve_test,
                '-o',
                label='Test')

            fig.legend()
            fig.show()
            fig.savefig(PLOTS_FOLDER / f'learning-curves-{dataset.base_table_label}.png', dpi=300, bbox_inches="tight")

    def plot_sensitivity_results(self, dataset):
        tfd = TFDExperiment(dataset, self.learning_curves_depth_values)
        tfd.run_sensitivity_experiments()


def plot_results_per_dataset(dataset: Dataset):
    result_df = pd.read_csv(MAPPING_FOLDER / f"acc-results-{dataset.base_table_label}.csv")
    exp = AllExperiments()
    exp.datasets.append(dataset)
    exp.plot_results(result_df)


if __name__ == "__main__":
    data = Datasets.school
    # plot_results_per_dataset(data)
    experiments = AllExperiments()
    experiments.experiments_per_dataset(data)
    experiments.plot_results()
    experiments.plot_learning_curves()
