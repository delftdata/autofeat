from pathlib import Path

import pandas as pd

from feature_discovery.graph_processing.neo4j_transactions import (
    get_pk_fk_nodes,
)
from feature_discovery.helpers.dict_utils import transform_node_to_dict


def join_directly_connected(base_table_id: str):
    nodes = get_pk_fk_nodes(base_table_id)
    partial_join = None
    for pk, fk in nodes:
        pk_node = transform_node_to_dict(pk)
        fk_node = transform_node_to_dict(fk)

        left_table = pd.read_csv(pk_node["source_path"])
        right_table = pd.read_csv(fk_node["source_path"])
        if partial_join is not None:
            left_table = partial_join

        partial_join = pd.merge(
            left_table,
            right_table,
            how="left",
            left_on=pk_node["name"],
            right_on=fk_node["name"],
            suffixes=("", "_b"),
        )
        columns_to_drop = [c for c in list(partial_join.columns) if c.endswith("_b")]
        partial_join.drop(columns=columns_to_drop, inplace=True)

    return partial_join


def join_and_save(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_column_name: str,
        right_column_name: str,
        join_path: Path,
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
