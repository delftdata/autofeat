from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_preparation.dataset_base import Dataset
from experiments.arda_experiments import ArdaExperiment
from experiments.base_table_experiments import BaseTableExperiment
from experiments.datasets import Datasets
from experiments.join_all_experiments import JoinAllExperiment
from experiments.tfd_experiments import TFDExperiment
from utils_module.util_functions import objects_to_dict


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

        self.datasets.append(dataset)

        base_table = BaseTableExperiment(dataset)
        base_table.accuracy_results()
        self.base_table_experiments[dataset] = base_table

        join_all = JoinAllExperiment(dataset)
        join_all.accuracy_results()
        self.join_all_experiments[dataset] = join_all

        join_all_fs = JoinAllExperiment(dataset, True)
        join_all_fs.accuracy_results()
        self.join_all_fs_experiments[dataset] = join_all_fs

        arda = ArdaExperiment(dataset, self.learning_curves_depth_values)
        arda.accuracy_results()
        arda.learning_curve_results()
        self.arda_experiments[dataset] = arda

        tfd = TFDExperiment(dataset, self.learning_curves_depth_values)
        tfd.get_results()
        self.tfd_experiments[dataset] = tfd

    def __get_results(self):
        dataset_results = {}

        for dataset in self.datasets:
            dataset_results[dataset] = []

        for dataset in self.datasets:
            dataset_results[dataset].extend(
                objects_to_dict(self.base_table_experiments[dataset].results)
            )
            dataset_results[dataset].extend(
                objects_to_dict(self.join_all_experiments[dataset].results)
            )
            dataset_results[dataset].extend(
                objects_to_dict(self.join_all_fs_experiments[dataset].results)
            )
            dataset_results[dataset].extend(objects_to_dict(self.arda_experiments[dataset].results))
            dataset_results[dataset].extend(objects_to_dict(self.tfd_experiments[dataset].results))

        return dataset_results

    def plot_results(self):
        dataset_results = self.__get_results()
        print(dataset_results)
        columns = len(self.datasets)
        fig, axs = plt.subplots(nrows=1, ncols=columns, figsize=(12, 6))

        for i, dataset in enumerate(self.datasets):
            results_df = pd.DataFrame(dataset_results[dataset])
            results_df.to_csv(f"acc-results-{dataset.base_table_label}.csv", index=False)

            if len(self.datasets) == 1:
                sns.barplot(data=results_df, x="algorithm", y="accuracy", hue="approach", ax=axs)
                axs.set_title(f"{dataset.base_table_label.title()}")
                axs.set_ylabel("Accuracy")
            else:
                sns.barplot(data=results_df, x="algorithm", y="accuracy", hue="approach", ax=axs[i])
                axs[i].set_title(f"{dataset.base_table_label.title()}")
                axs[i].set_ylabel("Accuracy")

        if len(self.datasets) == 1:
            h, l = axs.get_legend_handles_labels()
            axs.legend(h, l, bbox_to_anchor=(0, -0.25), loc=2, ncol=2, fontsize="xx-small")

            fig = axs.get_figure()
        else:
            h, l = axs[0].get_legend_handles_labels()
            axs[0].legend(h, l, bbox_to_anchor=(0, -0.25), loc=2, ncol=2, fontsize="xx-small")

            fig = axs[0].get_figure()

        fig.show()
        fig.savefig(f"../plots/accuracy-results-all.png", dpi=300, bbox_inches="tight")

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
            axs[1].plot(
                self.learning_curves_depth_values,
                self.tfd_experiments[dataset].learning_curve_train_tfd_path,
                "-o",
            )
            axs[1].plot(
                self.learning_curves_depth_values,
                self.tfd_experiments[dataset].learning_curve_test_tfd_path,
                "-o",
            )

            axs[2].set_title("Arda")
            axs[2].plot(
                self.learning_curves_depth_values,
                self.arda_experiments[dataset].learning_curve_train,
                "-o",
                label="Train",
            )
            axs[2].plot(
                self.learning_curves_depth_values,
                self.arda_experiments[dataset].learning_curve_test,
                "-o",
                label="Test",
            )

            fig.legend()
            fig.show()
            fig.savefig(
                f"../plots/learning-curves-{dataset.base_table_label}.png",
                dpi=300,
                bbox_inches="tight",
            )

    def plot_sensitivity_results(self, dataset):
        tfd = TFDExperiment(dataset, self.learning_curves_depth_values)
        tfd.run_sensitivity_experiments()


if __name__ == "__main__":
    data = Datasets.titanic
    experiments = AllExperiments()
    experiments.experiments_per_dataset(data)
    experiments.plot_results()
    experiments.plot_learning_curves()
