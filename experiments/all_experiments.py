import json
import math
import multiprocessing
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from arda.arda import wrapper_algo
from augmentation.ranking import ranking_multigraph
from augmentation.train_algorithms import train_CART, train_ID3, train_XGBoost
from experiments.config import Datasets
from utils.file_naming_convention import CONNECTIONS, MAPPING, ENUMERATED_PATHS, JOIN_RESULT_FOLDER
from utils.util_functions import prepare_data_for_ml

folder_name = os.path.abspath(os.path.dirname(__file__))


def tfd_results(dataset_config):
    print(f'======== Dataset: {dataset_config["path"]} ========')

    with open(f"{os.path.join(folder_name, '../', dataset_config['mappings_folder_name'])}/{MAPPING}", 'r') as fp:
        mapping = json.load(fp)

    with open(f"{os.path.join(folder_name, '../', dataset_config['mappings_folder_name'])}/{ENUMERATED_PATHS}",
              'r') as fp:
        all_paths = json.load(fp)

    target_column = dataset_config["label_column"]

    start = time.time()
    ranking = ranking_multigraph(all_paths, mapping, f"{dataset_config['path']}/{dataset_config['base_table_name']}",
                                 dataset_config["label_column"],
                                 dataset_config['join_result_folder_path'])
    sorted_ranking = dict(sorted(ranking.items(), key=lambda item: item[1][2]))
    end = time.time()
    join_time = end - start

    base_table_features = pd.read_csv(f"../{dataset_config['path']}/{dataset_config['base_table_name']}",
                                      engine="python",
                                      encoding="utf8", quotechar='"', escapechar='\\', nrows=1).drop(
        columns=[target_column]).columns
    top_1 = list(sorted_ranking.keys())[0]
    join_path, features, score = sorted_ranking[top_1]

    ### CASE 1: KEEP THE ENTIRE PATH
    print(f"Processing case 1: Keep the entire path")
    joined_df = pd.read_csv(
        join_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
    )
    X, y = prepare_data_for_ml(joined_df, target_column)
    training_funs = {"CART": train_CART, "ID3": train_ID3, "XGBoost": train_XGBoost}
    results = []
    for model_name, training_fun in training_funs.items():
        print(f"==== Model Name: {model_name} ====")
        accuracy, max_depth, feature_importances, train_time = _hp_tune_join_all(X, y, training_fun)
        entry = {
            "approach": "all-in-path",
            "dataset": top_1,
            "algorithm": model_name,
            "depth": max_depth,
            "accuracy": accuracy,
            "join_time": join_time,
            "train_time": train_time,
            "total_time": train_time + join_time,
            "feature_importances": feature_importances,
        }
        results.append(entry)

    print(f"Processing case 2: Remove all, but the ranked feature")
    aux_df = joined_df.copy(deep=True)
    aux_features = list(joined_df.columns)
    aux_features.remove(target_column)
    columns_to_drop = [
        c for c in aux_features if (c not in base_table_features) and (c not in features)
    ]
    aux_df.drop(columns=columns_to_drop, inplace=True)

    X, y = prepare_data_for_ml(aux_df, target_column)
    for model_name, training_fun in training_funs.items():
        print(f"==== Model Name: {model_name} ====")
        accuracy, max_depth, feature_importances, train_time = _hp_tune_join_all(X, y, training_fun)
        entry = {
            "approach": "best-ranked",
            "dataset": top_1,
            "algorithm": model_name,
            "depth": max_depth,
            "accuracy": accuracy,
            "join_time": join_time,
            "train_time": train_time,
            "total_time": train_time + join_time,
            "feature_importances": feature_importances,
        }
        results.append(entry)

    print(f"======== Finished dataset: {dataset_config['path']} ========")

    return results


def arda_results(dataset_config):
    print(f'======== Dataset: {dataset_config["path"]} ========')
    start = time.time()
    dataset_df = _join_all(dataset_config["path"])
    end = time.time()
    join_time = end - start

    label_column = _suffix_column(dataset_config["label_column"], dataset_config["base_table_name"])
    X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=label_column)
    print(X.shape)
    if X.shape[0] > 10000:
        _, X, _, y = train_test_split(X, y, test_size=10000, shuffle=True, stratify=y)
    print(X.shape)

    start = time.time()
    T = np.arange(0.0, 1.0, 0.1)
    indices = wrapper_algo(X, y, T)
    fs_X = X.iloc[:, indices]
    end = time.time()
    fs_time = end - start

    training_funs = {"CART": train_CART, "ID3": train_ID3, "XGBoost": train_XGBoost}
    results = []
    for model_name, training_fun in training_funs.items():
        print(f"==== Model Name: {model_name} ====")
        accuracy, max_depth, feature_importances, train_time, _ = _hp_tune_join_all(
            fs_X, y, training_fun, False
        )
        entry = {
            "approach": "arda",
            "dataset": dataset_config["path"],
            "algorithm": model_name,
            "depth": max_depth,
            "accuracy": accuracy,
            "join_time": join_time,
            "train_time": train_time,
            "total_time": join_time + train_time + fs_time,
            "feature_importances": feature_importances,
            "fs_time": fs_time,
        }
        results.append(entry)

    print(f"======== Finished dataset: {dataset_config['path']} ========")

    return results


def non_aug_results(dataset_config):
    print(f'======== Dataset: {dataset_config["path"]} ========')
    dataset_df = pd.read_csv(
        f"../{dataset_config['path']}/{dataset_config['base_table_name']}",
        engine="python",
        encoding="utf8",
        quotechar='"',
        escapechar="\\",
    )

    X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=dataset_config["label_column"])

    training_funs = {"CART": train_CART, "ID3": train_ID3, "XGBoost": train_XGBoost}
    results = []
    for model_name, training_fun in training_funs.items():
        print(f"==== Model Name: {model_name} ====")
        accuracy, max_depth, feature_importances, train_time, _ = _hp_tune_join_all(
            X, y, training_fun, False
        )
        entry = {
            "approach": "non-aug",
            "dataset": dataset_config["path"],
            "algorithm": model_name,
            "depth": max_depth,
            "accuracy": accuracy,
            "join_time": None,
            "train_time": train_time,
            "total_time": train_time,
            "feature_importances": feature_importances,
        }
        results.append(entry)

    print(f"======== Finished dataset: {dataset_config['path']} ========")

    return results


def _get_join_path(path: str):
    # Path looks like table_source/table_name/key--table_source...
    # Join path looks like table_source--table_name--table_source...

    path_tokens = path.split("--")
    join_path_tokens = ["--".join(token.split('/')[:-1]) for token in path_tokens]
    return "--".join(join_path_tokens)


def verify_ranking_func(ranking: dict, base_table_name: str, target_column: str):
    data = {}
    # 0. Get the baseline params
    print(f"Processing case 0: Baseline")
    base_table_df = pd.read_csv(
        # os.path.join(folder_name, "../", base_table_name),
        base_table_name,
        header=0,
        engine="python",
        encoding="utf8",
        quotechar='"',
        escapechar="\\",
    )
    X, y = prepare_data_for_ml(base_table_df, target_column)
    acc_b, params_b, _, _, _ = train_CART(X, y)
    base_table_features = list(base_table_df.drop(columns=[target_column]).columns)

    for path in ranking.keys():
        # join_path, features, score = ranking[path]
        score, features = ranking[path]
        join_path = _get_join_path(path)

        if score == math.inf:
            continue

        result = {"base-table": (acc_b, params_b["max_depth"])}
        joined_df = pd.read_csv(
            # join_path,
            f"../{JOIN_RESULT_FOLDER}/{join_path}",
            header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
        )
        # Two type of experiments
        # 1. Keep the entire path
        print(f"Processing case 1: Keep the entire path")
        X, y = prepare_data_for_ml(joined_df, target_column)
        acc, params, _, _, _ = train_CART(X, y)
        result["keep-all"] = (acc, params["max_depth"])
        print(X.columns)
        print(result)

        # 2. Remove all, but the ranked feature
        print(f"Processing case 2: Remove all, but the ranked feature")
        aux_df = joined_df.copy(deep=True)
        aux_features = list(joined_df.columns)
        aux_features.remove(target_column)
        columns_to_drop = [
            c for c in aux_features if (c not in base_table_features) and (c not in features)
        ]
        aux_df.drop(columns=columns_to_drop, inplace=True)

        X, y = prepare_data_for_ml(aux_df, target_column)
        acc, params, _, _, _ = train_CART(X, y)
        result["keep-all-but-ranked"] = (acc, params["max_depth"])
        print(X.columns)
        print(result)
        data[path] = {"features": features}
        data[path].update(result)

    return data


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


def _hp_tune_join_all(X, y, training_fun: Callable, do_sfs: bool):
    accuracy, params, feature_importances, train_time, sfs_time = training_fun(X, y, do_sfs)

    final_feature_importances = (
        dict(zip(X.columns, feature_importances)) if len(feature_importances) > 0 else {}
    )
    final_feature_importances = {
        feature: importance
        for feature, importance in final_feature_importances.items()
        if importance > 0
    }

    return accuracy, params["max_depth"], final_feature_importances, train_time, sfs_time


def join_all_results(dataset_config, do_feature_selection=False):
    print(f'======== Dataset: {dataset_config["path"]} ========')
    start = time.time()
    dataset_df = _join_all(dataset_config["path"])
    end = time.time()
    join_time = end - start
    # do_sfs = True

    label_column = _suffix_column(dataset_config["label_column"], dataset_config["base_table_name"])
    X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=label_column)

    if do_feature_selection:
        # ID3 not supported for SFS
        training_funs = {"CART": train_CART, "XGBoost": train_XGBoost}
    else:
        training_funs = {"CART": train_CART, "ID3": train_ID3, "XGBoost": train_XGBoost}
    # training_funs = {"XGBoost": train_XGBoost}
    results = []
    for model_name, training_fun in training_funs.items():
        print(f"==== Model Name: {model_name} ====")
        accuracy, max_depth, feature_importances, train_time, sfs_time = _hp_tune_join_all(
            X, y, training_fun, do_feature_selection
        )
        total_time = join_time + train_time
        approach = "join_all"
        if sfs_time:
            total_time += sfs_time
            approach += "_ffs"

        entry = {
            "approach": approach,
            "dataset": dataset_config["path"],
            "algorithm": model_name,
            "depth": max_depth,
            "accuracy": accuracy,
            "join_time": join_time,
            "train_time": train_time,
            "total_time": total_time,
            "feature_importances": feature_importances,
            "fs_time": sfs_time,
        }
        results.append(entry)

    print(f"======== Finished dataset: {dataset_config['path']} ========")

    return results


def run_all_experiments(dataset_config):
    all_results = []

    non_aug = non_aug_results(dataset_config)
    all_results += non_aug
    best_ranked = tfd_results(dataset_config)
    all_results += best_ranked
    arda = arda_results(dataset_config)
    all_results += arda
    join_all = join_all_results(dataset_config)
    all_results += join_all
    join_all_fs = join_all_results(dataset_config, True)
    all_results += join_all_fs

    return all_results


if __name__ == "__main__":
    dataset_configs = [
        getattr(Datasets, entry) for entry in dir(Datasets) if not entry.startswith("__")
    ]

    # There are 7 datasets
    pool_size = min(multiprocessing.cpu_count(), 6)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = p.map(run_all_experiments, dataset_configs)

    flattened_results = [entry for sub_list in results for entry in sub_list]
    results_df = pd.DataFrame(flattened_results)
    results_df.to_csv("ranking_func_results.csv", index=False)
