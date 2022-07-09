import json
import os

import pandas as pd

from data_ingestion import ingest_data
from utils.file_naming_convention import ENUMERATED_PATHS, JOINED_PATHS
from utils.neo4j_utils import drop_graph, init_graph, enumerate_all_paths, get_relation_properties

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
folder_name = os.path.abspath(os.path.dirname(__file__))


def data_preparation(base_table_name: str, label_column: str, path_to_data: str, mappings_folder_name: str,
                     join_result_folder_path: str):
    mapping = data_ingestion(path_to_data, mappings_folder_name)
    all_paths = path_enumeration(mappings_folder_name)

    all_joined_paths = []
    path = join_tables_recursive(all_paths, mapping, base_table_name, label_column, "", all_joined_paths,
                                 join_result_folder_path)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{JOINED_PATHS}", 'w') as fp:
        json.dump(all_joined_paths, fp)


def data_ingestion(path_to_data: str, mappings_folder_name: str, profile_valentine=False) -> dict:
    mapping = ingest_data.ingest_fabricated_data(path_to_data, mappings_folder_name)
    ingest_data.ingest_connections(path_to_data, mapping)

    if profile_valentine:
        ingest_data.profile_valentine_all(path_to_data, mapping)
    return mapping


def path_enumeration(mappings_folder_name: str, graph_name="graph") -> dict:
    try:
        drop_graph(graph_name)
    except Exception as err:
        print(err)

    init_graph(graph_name)
    result = enumerate_all_paths(graph_name)  # list of lists [from, to, distance]

    # transform the list of lists into a dictionary (from: [...to])
    all_paths = {}
    for pair in result:
        key, value, _ = pair
        all_paths.setdefault(key, []).append(value)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{ENUMERATED_PATHS}", 'w') as fp:
        json.dump(all_paths, fp)
    return all_paths


def join_tables_recursive(all_paths: dict, mapping, current_table, target_column, path, allp, join_result_path,
                          joined_mapping=None):
    if not joined_mapping:
        joined_mapping = {}

    # Join and save the join result
    if not path == "":
        # get the name of the table from the path we created
        left_table = path.split("--")[-1]
        # get the location of the table we want to join with
        partial_join = mapping[left_table]

        # If we already joined the tables on the path, we retrieve the join result
        if path in joined_mapping:
            partial_join = joined_mapping[path]

        # Add the current table to the path
        path = f"{path}--{current_table}"
        joined_path, _, _ = join_and_save(partial_join, mapping[left_table],
                                                                    mapping[current_table], join_result_path, path)
        joined_mapping[path] = joined_path
    else:
        # Just started traversing, the path is the current table
        path = current_table

    print(path)
    allp.append(path)

    # Depth First Search recursively
    for table in all_paths[current_table]:
        # Break the cycles in the data, only visit new nodes
        if table not in path:
            join_path = join_tables_recursive(all_paths, mapping, table, target_column, path, allp, join_result_path,
                                              joined_mapping)
            # print(f"{join_path}")
    return path


def join_and_save(partial_join_path, left_table_path, right_table_path, join_result_path, join_result_name):
    # Getting the join keys
    from_col, to_col = get_relation_properties(left_table_path, right_table_path)
    # Read left side table
    left_table_df = pd.read_csv(partial_join_path, header=0, engine="python", encoding="utf8", quotechar='"',
                                escapechar='\\')
    if from_col not in left_table_df.columns:
        print(f"ERROR! Key {from_col} not in table {left_table_df}")
        return None

    right_table_df = pd.read_csv(right_table_path, header=0, engine="python", encoding="utf8", quotechar='"',
                                 escapechar='\\')
    if to_col not in right_table_df.columns:
        print(f"ERROR! Key {to_col} not in table {right_table_df}")
        return None

    print(f"\tJoining {partial_join_path} with {right_table_path}\n\tOn keys: {from_col} - {to_col}")
    joined_df = pd.merge(left_table_df, right_table_df, how="left", left_on=from_col, right_on=to_col,
                         suffixes=("_b", ""))

    # If both tables have the same column, drop one of them
    duplicate_col = [col for col in joined_df.columns if col.endswith('_b')]
    # Drop the FK key from the left table
    # duplicate_col.append(from_col)
    joined_df.drop(columns=duplicate_col, inplace=True)
    # Save join result
    joined_path = f"{os.path.join(folder_name, '../', join_result_path)}/{join_result_name}"
    joined_df.to_csv(joined_path, index=False)

    return joined_path, joined_df, left_table_df
