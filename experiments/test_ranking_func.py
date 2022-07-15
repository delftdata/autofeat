import math
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from augmentation.train_algorithms import train_CART, train_ID3, train_XGBoost
from utils.file_naming_convention import CONNECTIONS
from utils.util_functions import prepare_data_for_ml
from experiments.config import Datasets

folder_name = os.path.abspath(os.path.dirname(__file__))


def verify_ranking_func(ranking: dict, base_table_name: str, target_column: str):
    data = {}
    # 0. Get the baseline params
    print(f"Processing case 0: Baseline")
    base_table_df = pd.read_csv(
        os.path.join(folder_name, "../", base_table_name),
        header=0,
        engine="python",
        encoding="utf8",
        quotechar='"',
        escapechar="\\",
    )
    X, y = prepare_data_for_ml(base_table_df, target_column)
    acc_b, params_b, _ = train_CART(X, y)
    base_table_features = list(base_table_df.drop(columns=[target_column]).columns)

    for path in ranking.keys():
        join_path, features, score = ranking[path]

        if score == math.inf:
            continue

        result = {"base-table": (acc_b, params_b["max_depth"])}
        joined_df = pd.read_csv(
            join_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\"
        )
        # Three type of experiments
        # 1. Keep the entire path
        print(f"Processing case 1: Keep the entire path")
        X, y = prepare_data_for_ml(joined_df, target_column)
        acc, params, _ = train_CART(X, y)
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
        acc, params, _ = train_CART(X, y)
        result["keep-all-but-ranked"] = (acc, params["max_depth"])
        print(X.columns)
        print(result)
        data[path] = {"features": features}
        data[path].update(result)

    return data


def suffix_column(column_name: str, table_name: str) -> str:
    return f"{column_name}_{table_name}"


def join_all(data_path: str) -> pd.DataFrame:
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
                    column: suffix_column(column_name=column, table_name=f.name)
                    for column in dataset_columns
                }
            )
            datasets[f.name] = dataset

    columns_to_drop = []
    first_col = None

    for _, row in connections_df.iterrows():
        # Reconstruct suffix that was built previously
        left_col = suffix_column(
            column_name=str(row["left_col"]), table_name=str(row["left_table"])
        )
        right_col = suffix_column(
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


def hp_tune_join_all(dataset_config: Dict[str, str], dataset_df: pd.DataFrame):
    label_column = suffix_column(dataset_config["label_column"], dataset_config["base_table_name"])
    X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=label_column)
    # _, _, feature_importances = train_CART(X, y)
    # train_ID3(X, y)
    train_XGBoost(X, y)
    # print(dict(zip(X.columns, feature_importances)))


if __name__ == "__main__":
    dataset_config = Datasets.football_data
    dataset_df = join_all(dataset_config["path"])
    hp_tune_join_all(dataset_config=dataset_config, dataset_df=dataset_df)
