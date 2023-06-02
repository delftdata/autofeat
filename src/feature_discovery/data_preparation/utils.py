import json
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from feature_discovery.config import (
    MAPPING_FOLDER,
    ENUMERATED_PATHS,
    JOIN_RESULT_FOLDER,
)

folder_name = os.path.abspath(os.path.dirname(__file__))


def get_paths():
    with open(MAPPING_FOLDER / ENUMERATED_PATHS, "r") as fp:
        all_paths = json.load(fp)
    return all_paths


def prepare_data_for_ml(dataframe: pd.DataFrame, target_column: str) -> tuple:
    df = dataframe.fillna(0)
    df = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)
    X = df.drop(columns=[target_column])

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    normalized_X = pd.DataFrame(scaled_X, columns=X.columns)
    # print(normalized_X)

    y = df[target_column].astype(float)

    # le = LabelEncoder()
    # y = pd.Series(le.fit_transform(y), name=target_column)

    return normalized_X, y


def get_join_path(path: str) -> str:
    # Path looks like table_source/table_name/key--table_source...
    # Join path looks like table_source--table_name--table_source...

    path_tokens = path.split("--")
    join_path_tokens = ["--".join(token.split("/")[:-1]) for token in path_tokens]
    return "--".join(join_path_tokens)


def get_path_length(path: str) -> int:
    # Path looks like table_source/table_name/key--table_source...
    path_tokens = path.split("--")
    # Length = 1 means that we have 2 tables
    return len(path_tokens) - 1


def compute_join_name(join_key_property: tuple, partial_join_name: str) -> str:
    """
    Compute the name of the partial join, given the properties of the new join and the previous join name.

    :param join_key_property: (neo4j relation property, outbound label, inbound label)
    :param partial_join_name: Name of the partial join.
    :return: The name of the next partial join
    """
    join_prop, _, to_table = join_key_property
    joined_path = f"{partial_join_name}--{join_prop['to_column']}-{to_table}"
    return joined_path


def join_and_save(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_column_name: str,
    right_column_name: str,
    join_path: str,
) -> pd.DataFrame or None:
    """
    Join two dataframes and save the result on disk.

    :param left_df: Left side of the join
    :param right_df: Right side of the join
    :param left_column_name: The left join column
    :param right_column_name: The right join column
    :param join_path: The path to save the join result.
    :return: The join result.
    """
    if left_df[left_column_name].dtype != right_df[right_column_name].dtype:
        return None

    partial_join = pd.merge(
        left_df,
        right_df,
        how="left",
        left_on=left_column_name,
        right_on=right_column_name,
    )
    # Save join result
    partial_join.to_csv(join_path, index=False)
    return partial_join
