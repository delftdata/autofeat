import json

from config import ENUMERATED_PATHS, MAPPING, MAPPING_FOLDER
from data_preparation.ingest_data import ingest_connections, profile_valentine_all, ingest_fabricated_data
from helpers.neo4j_utils import drop_graph, init_graph, enumerate_all_paths


def data_preparation():
    mapping = _data_ingestion(True)

    with open(MAPPING_FOLDER / MAPPING, 'w') as fp:
        json.dump(mapping, fp)

    all_paths = _path_enumeration()

    with open(MAPPING_FOLDER / ENUMERATED_PATHS, 'w') as fp:
        json.dump(all_paths, fp)


def _data_ingestion(profile_valentine=False) -> dict:
    mapping = ingest_fabricated_data()
    ingest_connections()

    if profile_valentine:
        profile_valentine_all()

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
