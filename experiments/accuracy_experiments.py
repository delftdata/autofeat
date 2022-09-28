import math
import os
import time
from pathlib import Path
from typing import Callable, List

import pandas as pd

import arda.arda
from augmentation.rank_object import Rank
from augmentation.ranking import Ranking
from augmentation.train_algorithms import train_CART, train_ID3, train_XGBoost
from data_preparation.join_data import join_all
from data_preparation.utils import prepare_data_for_ml, get_join_path
from experiments.datasets import Datasets
from experiments.result_object import Result
from utils.file_naming_convention import CONNECTIONS, JOIN_RESULT_FOLDER
from utils.util_functions import objects_to_dict

folder_name = os.path.abspath(os.path.dirname(__file__))


class AccuracyExperiments:
    CART = "CART"
    ID3 = "ID3"
    XGBOOST = "XGBoost"

    TRAINING_FUNCTIONS = {CART: train_CART, ID3: train_ID3, XGBOOST: train_XGBoost}

    def __init__(self, data, ranked_paths: List[Rank] = None):
        self.dataset = data
        self.dataset.set_base_table_df()
        self.results = []
        self.ranked_paths = ranked_paths

    def tfd_results(self):
        print(f'======== TFD Pipeline ========')

        start = time.time()
        tfd_ranking = Ranking(self.dataset).start_ranking()
        end = time.time()
        join_time = end - start
        self.ranked_paths = tfd_ranking.ranked_paths

        top_1 = tfd_ranking.ranked_paths[0]
        join_path = get_join_path(top_1.path)

        print(f"Processing case 1: Keep the entire path")
        joined_df = pd.read_csv(
            f"../{JOIN_RESULT_FOLDER}/{join_path}",
            header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
        )
        X, y = prepare_data_for_ml(joined_df, self.dataset.target_column)
        self.train_approach(X, y, Result.TFD_PATH, top_1.path, join_time)

        print(f"Processing case 2: Remove all, but the ranked feature")
        aux_df = joined_df.copy(deep=True)
        aux_features = list(joined_df.columns)
        aux_features.remove(self.dataset.target_column)
        columns_to_drop = [
            c for c in aux_features if (c not in self.dataset.base_table_features) and (c not in top_1.features)
        ]
        aux_df.drop(columns=columns_to_drop, inplace=True)

        X, y = prepare_data_for_ml(aux_df, self.dataset.target_column)
        self.train_approach(X, y, Result.TFD, top_1.path, join_time)

        print(f"======== Finished TFD Pipeline ========")

    def arda_results(self):
        print(f'======== ARDA Pipeline ========')

        X, y, join_time, fs_time = arda.arda.select_features(self.dataset.base_table_id, self.dataset.target_column,
                                                             self.dataset.base_table_features)
        self.train_approach(X, y, Result.ARDA, self.dataset.base_table_id, join_time, fs_time)

        print(f"======== Finished ARDA Pipeline ========")

    def train_approach(self, X, y, approach, data_path, join_time=None, fs_time=None):
        for model_name, training_fun in self.TRAINING_FUNCTIONS.items():
            print(f"==== Model Name: {model_name} ====")
            entry = Result(approach, data_path, self.dataset.base_table_label, model_name,
                           join_time)
            accuracy, max_depth, feature_importances, train_time, ffs_time = _hp_tune_join_all(X, y, training_fun,
                                                                                               False)
            entry.set_depth(max_depth).set_accuracy(accuracy).set_feature_importance(
                feature_importances).set_train_time(train_time)
            if fs_time:
                entry.set_feature_selection_time(fs_time)
            self.results.append(entry)

    def non_aug_results(self):
        print(f'======== NON-AUG Pipeline ========')

        if self.dataset.base_table_df is None:
            self.dataset.set_base_table_df()
        X, y = prepare_data_for_ml(dataframe=self.dataset.base_table_df, target_column=self.dataset.target_column)
        self.train_approach(X, y, Result.BASE, self.dataset.base_table_id)

        print(f"======== Finished NON-AUG Pipeline ========")

    def verify_ranking_func(self):
        # 0. Get the baseline params
        print(f"Processing case 0: Baseline")

        if self.dataset.base_table_df is None:
            self.dataset.set_base_table_df()
        X_b, y = prepare_data_for_ml(self.dataset.base_table_df, self.dataset.target_column)
        acc_b, params_b, feature_imp_b, _, _ = train_CART(X_b, y)
        entry = Result(Result.BASE, self.dataset.base_table_id, self.dataset.base_table_label, self.CART)
        entry.set_accuracy(acc_b).set_depth(params_b["max_depth"]).set_feature_importance(
            _map_features_scores(feature_imp_b, X_b))
        self.results.append(entry)

        if self.ranked_paths is None:
            tfd_ranking = Ranking(self.dataset)
            tfd_ranking.start_ranking()
            self.ranked_paths = tfd_ranking.ranked_paths

        for ranked_path in self.ranked_paths:
            join_path = get_join_path(ranked_path.path)

            if ranked_path.score == math.inf:
                continue

            joined_df = pd.read_csv(
                f"../{JOIN_RESULT_FOLDER}/{join_path}",
                header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
            )
            # Two type of experiments
            # 1. Keep the entire path
            print(f"Processing case 1: Keep the entire path")
            X, y = prepare_data_for_ml(joined_df, self.dataset.target_column)
            acc, params, feature_imp, _, _ = train_CART(X, y)

            entry = Result(Result.TFD_PATH, ranked_path.path, self.dataset.base_table_label, self.CART)
            entry.set_accuracy(acc).set_depth(params["max_depth"]).set_feature_importance(
                _map_features_scores(feature_imp, X))
            self.results.append(entry)

            # 2. Remove all, but the ranked feature
            print(f"Processing case 2: Remove all, but the ranked feature")
            aux_df = joined_df.copy(deep=True)
            aux_features = list(joined_df.columns)
            aux_features.remove(self.dataset.target_column)
            columns_to_drop = [
                c for c in aux_features if
                (c not in self.dataset.base_table_features) and (c not in ranked_path.features)
            ]
            aux_df.drop(columns=columns_to_drop, inplace=True)

            X, y = prepare_data_for_ml(aux_df, self.dataset.target_column)
            acc, params, feature_imp, _, _ = train_CART(X, y)

            entry = Result(Result.TFD, ranked_path.path, self.dataset.base_table_label, self.CART)
            entry.set_accuracy(acc).set_depth(params["max_depth"]).set_feature_importance(
                _map_features_scores(feature_imp, X))
            self.results.append(entry)
        return self

    def join_all_results(self, do_feature_selection=False):
        print(f'======== JOIN-ALL Dataset Pipeline - feature selection - {do_feature_selection} ========')
        start = time.time()
        dataset_df = join_all(self.dataset.base_table_id)
        end = time.time()
        join_time = end - start

        X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=self.dataset.target_column)

        for model_name, training_fun in self.TRAINING_FUNCTIONS.items():
            if do_feature_selection:
                # ID3 not supported for feature selection
                if model_name == self.ID3:
                    continue
            print(f"==== Model Name: {model_name} ====")
            if do_feature_selection:
                approach = Result.JOIN_ALL_FS
            else:
                approach = Result.JOIN_ALL
            entry = Result(approach, self.dataset.base_table_id, self.dataset.base_table_label, model_name, join_time)
            accuracy, max_depth, feature_importances, train_time, sfs_time = _hp_tune_join_all(
                X, y, training_fun, do_feature_selection
            )
            entry.set_depth(max_depth).set_accuracy(accuracy).set_feature_importance(
                feature_importances).set_train_time(train_time)
            if sfs_time:
                entry.set_feature_selection_time(sfs_time)
            self.results.append(entry)

        print(f"======== Finished dataset ========")

    def run_all_experiments(self):
        self.tfd_results()
        self.arda_results()
        self.non_aug_results()
        self.join_all_results()
        self.join_all_results(True)
        return self


def _suffix_column(column_name: str, table_name: str) -> str:
    return f"{column_name}_{table_name}"


def _join_all(data_path: str) -> pd.DataFrame:
    file_paths = list(Path().glob(f"../{data_path}/**/*.csv"))
    connections_df = None
    datasets = {}
    joined_df = pd.DataFrame()

    for file_path in file_paths:
        if CONNECTIONS == file_path.name:
            connections_df = pd.read_csv(file_path)
            break

    if connections_df is None:
        raise ValueError(f"{CONNECTIONS} not in folder")

    for f in file_paths:
        if f.name != CONNECTIONS:
            dataset = pd.read_csv(f)
            dataset_columns = dataset.columns
            # Add suffix to each column to uniquely identify them
            dataset = dataset.rename(
                columns={
                    column: _suffix_column(column_name=column, table_name=f.name)
                    for column in dataset_columns
                }
            )
            datasets[f.name] = dataset

    columns_to_drop = []
    first_col = None

    for _, row in connections_df.iterrows():
        # Reconstruct suffix that was built previously
        left_col = _suffix_column(
            column_name=str(row["left_col"]), table_name=str(row["left_table"])
        )
        right_col = _suffix_column(
            column_name=str(row["right_col"]), table_name=str(row["right_table"])
        )

        # First iteration: no existing df
        if joined_df.empty:
            joined_df = pd.merge(
                datasets[row["left_table"]],
                datasets[row["right_table"]],
                how="left",
                left_on=left_col,
                right_on=right_col,
                suffixes=("", ""),
            )
            first_col = left_col
            continue

        joined_df = pd.merge(
            joined_df,
            datasets[row["right_table"]],
            how="left",
            left_on=left_col,
            right_on=right_col,
            suffixes=("", ""),
        )
        # Drop foreign keys (redundant)
        columns_to_drop.append(right_col)

        # Drop primary keys (redundat).
        # With the exception of the PK of the primary table
        if left_col != first_col:
            columns_to_drop.append(left_col)

    joined_df.drop(columns=columns_to_drop, inplace=True)

    return joined_df


def _map_features_scores(feature_importances, X):
    final_feature_importances = (
        dict(zip(X.columns, feature_importances)) if len(feature_importances) > 0 else {}
    )
    final_feature_importances = {
        feature: importance
        for feature, importance in final_feature_importances.items()
        if importance > 0
    }
    return final_feature_importances


def _hp_tune_join_all(X, y, training_fun: Callable, do_sfs: bool):
    accuracy, params, feature_importances, train_time, sfs_time = training_fun(X, y, do_sfs)

    final_feature_importances = _map_features_scores(feature_importances, X)

    return accuracy, params["max_depth"], final_feature_importances, train_time, sfs_time


def run_all_experiments():
    all_results = []
    all_ranks = []

    for data in Datasets.ALL:
        experiment = AccuracyExperiments(data)
        experiment.run_all_experiments()
        all_results.extend(objects_to_dict(experiment.results))
        all_ranks.extend(objects_to_dict(experiment.ranked_paths))

    return all_results, all_ranks


if __name__ == "__main__":
    # dataset_configs = Datasets.titanic_data

    # There are 7 datasets
    # pool_size = min(multiprocessing.cpu_count(), 6)

    # with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    #     results = p.map(run_all_experiments, dataset_configs)

    # results, ranks = run_all_experiments()
    dataset = Datasets.football
    experiments = AccuracyExperiments(dataset).run_all_experiments()
    results = objects_to_dict(experiments.results)
    ranks = objects_to_dict(experiments.ranked_paths)
    # flattened_results = [entry for sub_list in results for entry in sub_list]
    label = dataset.base_table_label
    # label = "all"
    pd.DataFrame(ranks).to_csv(f"ranks_{label}.csv", index=False)
    pd.DataFrame(results).to_csv(f"accuracy_results_{label}.csv", index=False)
