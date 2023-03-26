import json

from config import ENUMERATED_PATHS, MAPPING, MAPPING_FOLDER, JSON
from data_preparation.dataset_base import Dataset
from data_preparation.ingest_data import profile_valentine_all, ingest_fabricated_data, ingest_connections, \
    ingest_tables, ingest_unprocessed_data
from graph_processing.neo4j_transactions import drop_graph, init_graph, enumerate_all_paths, find_graph
from tfd_datasets.classification import credit, steel, cylinder


def data_preparation(ingest_data: bool = True, profile_valentine: bool = False):
    if ingest_data or profile_valentine:
        mapping = _data_ingestion(ingest_data=ingest_data, profile_valentine=profile_valentine)

        with open(MAPPING_FOLDER / MAPPING, 'w') as fp:
            json.dump(mapping, fp)

    all_paths = _path_enumeration()

    with open(MAPPING_FOLDER / ENUMERATED_PATHS, 'w') as fp:
        json.dump(all_paths, fp)


def _data_ingestion(ingest_data: bool = True, profile_valentine: bool = False) -> dict:
    mapping = {}
    if ingest_data:
        mapping = ingest_fabricated_data()
        ingest_connections()

    if profile_valentine:
        profile_valentine_all()

    return mapping


def _path_enumeration(graph_name="graph") -> dict:
    graphs = find_graph(graph_name)
    if len(graphs) > 0:
        drop_graph(graph_name)

    print("Initiate graph ... ")
    init_graph(graph_name)
    print("Traverse graph and enumerate all paths ... ")
    result = enumerate_all_paths(graph_name)  # list of lists [from, to, distance]

    # transform the list of lists into a dictionary (from: [...to])
    print("Save paths ... ")
    all_paths = {}
    for pair in result:
        key, value, _ = pair
        all_paths.setdefault(key, []).append(value)

    return all_paths


def data_preparation_tables(ingest=True, enumerate_paths=True):
    if ingest:
        mapping = ingest_tables()

        with open(MAPPING_FOLDER / MAPPING, 'w') as fp:
            json.dump(mapping, fp)

    if enumerate_paths:
        all_paths = _path_enumeration()

        with open(MAPPING_FOLDER / ENUMERATED_PATHS, 'w') as fp:
            json.dump(all_paths, fp)


def ingest_data_with_connections(dataset: Dataset, profile_valentine=False):
    mapping = ingest_unprocessed_data(dataset.base_table_label)
    with open(MAPPING_FOLDER / f"{MAPPING + dataset.base_table_label + JSON}", 'w') as fp:
        json.dump(mapping, fp)

    if profile_valentine:
        profile_valentine_all(dataset.base_table_label)


if __name__ == "__main__":
    # data_preparation(ingest_data=False, profile_valentine=False)
    # data_preparation_tables(ingest=True, enumerate_paths=False)
    ingest_data_with_connections(dataset=cylinder, profile_valentine=True)
