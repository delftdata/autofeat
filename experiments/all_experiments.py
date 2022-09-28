import math
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from augmentation.ranking import Ranking
from data_preparation.utils import get_join_path, prepare_data_for_ml
from experiments.accuracy_experiments import AccuracyExperiments, _map_features_scores
from experiments.datasets import Datasets
from experiments.experiment_object import Experiment
from experiments.result_object import Result
from utils.file_naming_convention import JOIN_RESULT_FOLDER
from utils.util_functions import objects_to_dict


class AllExperiments:
    def __init__(self):
        self.results: List[Experiment] = []
        self.arda_features = None
        self.ranked_paths = None
        self.learning_curve_results = None
        self.cutoff_th_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.redundancy_th_values = [5, 7, 9, 10, 15, 20, 25, 30]

    def variate_thresholds(self, dataset):
        model = AccuracyExperiments.CART

        for redundancy_th in self.redundancy_th_values:
            print(f"\n\tREDUNDANCY THRESHOLD: {redundancy_th}")
            for cutoff_th in self.cutoff_th_values:
                print(f"\n\tCUTOFF THRESHOLD: {cutoff_th}")
                ranking = Ranking(dataset, redundancy_th, cutoff_th)
                ranking.start_ranking()

                for i, ranked_path in enumerate(ranking.ranked_paths[0:3]):
                    # TFD - All in path
                    print(f"Path: {i} - score: {ranked_path.score}\n\t{ranked_path.path}\n\t{ranked_path.features}")
                    print(f"Processing case 1: Keep the entire path")
                    join_path = get_join_path(ranked_path.path)

                    if ranked_path.score == math.inf:
                        continue

                    joined_df = pd.read_csv(
                        f"../{JOIN_RESULT_FOLDER}/{join_path}",
                        header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
                    )

                    X, y = prepare_data_for_ml(joined_df, dataset.target_column)
                    acc, params, feature_imp, _, _ = AccuracyExperiments.TRAINING_FUNCTIONS[model](X, y)

                    experiment = Experiment(join_path, dataset.base_table_label, model, acc, Result.TFD_PATH)
                    experiment.set_depth(params['max_depth']).set_rank(i).set_cutoff_th(cutoff_th).set_redundancy_th(
                        redundancy_th).set_features(_map_features_scores(feature_imp, X))
                    self.results.append(experiment)

                    print(f"Processing case 2: Remove all, but the ranked feature")
                    aux_df = joined_df.copy(deep=True)
                    aux_features = list(joined_df.columns)
                    aux_features.remove(dataset.target_column)
                    columns_to_drop = [
                        c for c in aux_features if
                        (c not in dataset.base_table_features) and (c not in ranked_path.features)
                    ]
                    aux_df.drop(columns=columns_to_drop, inplace=True)

                    X, y = prepare_data_for_ml(aux_df, dataset.target_column)
                    acc, params, feature_imp, _, _ = AccuracyExperiments.TRAINING_FUNCTIONS[model](X, y)

                    experiment = Experiment(join_path, dataset.base_table_label, model, acc, Result.TFD)
                    experiment.set_depth(params['max_depth']).set_rank(i).set_cutoff_th(cutoff_th).set_redundancy_th(
                        redundancy_th).set_features(_map_features_scores(feature_imp, X))
                    self.results.append(experiment)

    def plot_redundancy_sensitivity(self, dataset_label, results: pd.DataFrame = None):
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 6))

        if results is None:
            results = pd.DataFrame(objects_to_dict(self.results))

        for i, cutoff_th in enumerate(self.cutoff_th_values):
            df = results[results['cutoff_th'] == cutoff_th]

            colors = ['red', 'black', 'green']
            for j, rank in enumerate(df['rank'].unique()):
                df_ranks = df[df['rank'] == rank]

                marker = ['*', 'H'
                               '']
                for k, approach in enumerate(df['approach'].unique()):
                    values = df_ranks[df_ranks['approach'] == approach]
                    if i == len(self.cutoff_th_values)-1:
                        axs[i].plot(self.redundancy_th_values, values['accuracy'], marker=marker[k], color=colors[j],
                                    label=f"{approach}-{rank}")
                    else:
                        axs[i].plot(self.redundancy_th_values, values['accuracy'], marker=marker[k], color=colors[j])
            axs[i].set_title(f"Cut-off Threshold = {cutoff_th}")

        fig.legend()
        fig.show()
        fig.savefig(f'../plots/sensitivity-plots-{dataset_label}2.png', dpi=300, bbox_inches="tight")


def get_sensitivity_results(dataset):
    all_exp = AllExperiments()
    all_exp.variate_thresholds(dataset)
    result = objects_to_dict(all_exp.results)
    print(result)
    pd.DataFrame(result).to_csv(f"threshold-sensitivity-{dataset.base_table_label}2.csv", index=False)


def plot_sensitivity_results(dataset):
    results = pd.read_csv(f"threshold-sensitivity-{dataset.base_table_label}2.csv")
    all_exp = AllExperiments()
    all_exp.plot_redundancy_sensitivity(dataset.base_table_label, results)


if __name__ == "__main__":
    data = Datasets.steel_plate_fault
    get_sensitivity_results(data)
    plot_sensitivity_results(data)
