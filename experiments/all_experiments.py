from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import PLOTS_FOLDER, MAPPING_FOLDER, ACCURACY_RESULTS_ALL_PNG
from data_preparation.dataset_base import Dataset
from experiments.arda_experiments import ArdaExperiment
from experiments.base_table_experiments import BaseTableExperiment
from experiments.datasets import Datasets
from experiments.join_all_experiments import JoinAllExperiment
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
        self.base_table_experiments: Dict[Dataset, BaseTableExperiment] = {}
        self.join_all_experiments: Dict[Dataset, JoinAllExperiment] = {}
        self.join_all_fs_experiments: Dict[Dataset, JoinAllExperiment] = {}
        self.tfd_experiments: Dict[Dataset, TFDExperiment] = {}
        self.learning_curves_depth_values = [i for i in range(1, 21)]
        self.datasets: List[Dataset] = []

    def experiments_per_dataset(self, dataset: Dataset):
        # TODO: Standardize experiments, create config containing list of experiments to run
        # Instead of instantiating every experiment separately, loop through the config and
        # collect the results
        self.datasets.append(dataset)

        base_table = BaseTableExperiment(dataset)
        base_table.compute_accuracy_results()
        self.base_table_experiments[dataset] = base_table

        join_all = JoinAllExperiment(dataset)
        join_all.compute_accuracy_results()
        self.join_all_experiments[dataset] = join_all

        join_all_fs = JoinAllExperiment(dataset, True)
        join_all_fs.compute_accuracy_results()
        self.join_all_fs_experiments[dataset] = join_all_fs

        arda = ArdaExperiment(dataset, self.learning_curves_depth_values)
        arda.compute_accuracy_results()
        arda.learning_curve_results()
        self.arda_experiments[dataset] = arda

        tfd = TFDExperiment(dataset, self.learning_curves_depth_values)
        tfd.compute_results()
        self.tfd_experiments[dataset] = tfd

    def __get_results(self):
        dataset_results = {}

        for dataset in self.datasets:
            dataset_results[dataset] = []

        for dataset in self.datasets:
            dataset_results[dataset].extend(objects_to_dict(self.base_table_experiments[dataset].results))
            dataset_results[dataset].extend(objects_to_dict(self.join_all_experiments[dataset].results))
            dataset_results[dataset].extend(objects_to_dict(self.join_all_fs_experiments[dataset].results))
            dataset_results[dataset].extend(objects_to_dict(self.arda_experiments[dataset].results))
            dataset_results[dataset].extend(objects_to_dict(self.tfd_experiments[dataset].results))

        return dataset_results

    def plot_results(self, results_df: pd.DataFrame = None):
        dataset_results = {}
        if results_df is None:
            dataset_results = self.__get_results()
            print(dataset_results)

        columns = len(self.datasets)
        fig, axs = plt.subplots(nrows=1, ncols=columns, figsize=(12, 6))

        for i, dataset in enumerate(self.datasets):
            if results_df is None and dataset in dataset_results:
                results_df = pd.DataFrame(dataset_results[dataset])
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
                "-o",
            )
            axs[0].plot(
                self.learning_curves_depth_values,
                self.tfd_experiments[dataset].learning_curve_test_tfd,
                "-o",
            )

            axs[1].set_title("All in path Features")
            axs[1].plot(self.learning_curves_depth_values, self.tfd_experiments[dataset].learning_curve_train_tfd_path,
                        '-o')
            axs[1].plot(self.learning_curves_depth_values, self.tfd_experiments[dataset].learning_curve_test_tfd_path,
                        '-o')

            axs[2].set_title("Arda")
            axs[2].plot(self.learning_curves_depth_values, self.arda_experiments[dataset].learning_curve_train, '-o',
                        label='Train')
            axs[2].plot(self.learning_curves_depth_values, self.arda_experiments[dataset].learning_curve_test, '-o',
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
    data = Datasets.titanic
    # plot_results_per_dataset(data)
    experiments = AllExperiments()
    experiments.experiments_per_dataset(data)
    experiments.plot_results()
    experiments.plot_learning_curves()
