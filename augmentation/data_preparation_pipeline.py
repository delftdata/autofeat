import json
import os

import pandas as pd

from data_preparation.ingest_data import ingest_fabricated_data, ingest_connections, profile_valentine_all
from data_preparation.join_data import join_tables_recursive
from utils.file_naming_convention import ENUMERATED_PATHS, JOINED_PATHS
from utils.neo4j_utils import drop_graph, init_graph, enumerate_all_paths, get_relation_properties

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
folder_name = os.path.abspath(os.path.dirname(__file__))


def data_preparation(base_table_name: str, label_column: str, path_to_data: str, mappings_folder_name: str,
                     join_result_folder_path: str):
    mapping = _data_ingestion(path_to_data, mappings_folder_name)
    all_paths = _path_enumeration(mappings_folder_name)

    all_joined_paths = []
    path = join_tables_recursive(all_paths, mapping, base_table_name, label_column, "", all_joined_paths,
                                 join_result_folder_path)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{JOINED_PATHS}", 'w') as fp:
        json.dump(all_joined_paths, fp)


def _data_ingestion(path_to_data: str, mappings_folder_name: str, profile_valentine=False) -> dict:
    mapping = ingest_fabricated_data(path_to_data, mappings_folder_name)
    ingest_connections(path_to_data, mapping)

    if profile_valentine:
        profile_valentine_all(path_to_data)
    return mapping


def _path_enumeration(mappings_folder_name: str, graph_name="graph") -> dict:
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





