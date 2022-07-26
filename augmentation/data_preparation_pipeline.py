import json
import os

from data_preparation.ingest_data import ingest_connections, profile_valentine_all
from utils.file_naming_convention import ENUMERATED_PATHS, MAPPING
from utils.neo4j_utils import drop_graph, init_graph, enumerate_all_paths

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
folder_name = os.path.abspath(os.path.dirname(__file__))


def data_preparation(path_to_data: str, mappings_folder_name: str):
    mapping = _data_ingestion(path_to_data)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{MAPPING}", 'w') as fp:
        json.dump(mapping, fp)

    all_paths = _path_enumeration()

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{ENUMERATED_PATHS}", 'r') as fp:
        all_paths = json.load(fp)


def _data_ingestion(path_to_data: str, profile_valentine=False) -> dict:
    # Deprecated
    # mapping = ingest_fabricated_data(path_to_data, mappings_folder_name)
    mapping = ingest_connections(path_to_data)

    if profile_valentine:
        mappings_valentine = profile_valentine_all(path_to_data)
        mapping.update(mappings_valentine)

    return mapping


def _path_enumeration(graph_name="graph") -> dict:
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

    return all_paths
