import json

from data_preparation.join_data import enumerate_all
from experiments.config import Datasets
from utils.file_naming_convention import MAPPING_FOLDER, ENUMERATED_PATHS, ALL_PATHS


def enum_paths(dataset_config, all_paths):
    path = []
    enump = []
    visited = []
    enumerate_all(dataset_config['id'], all_paths, path, enump, visited)
    enumerated_paths = [p for p in enump if len(p) > 1]

    with open(f"../{MAPPING_FOLDER}/{ALL_PATHS}_{dataset_config['base_table_label']}.json", 'w') as fp:
        json.dump({'paths': enumerated_paths}, fp)

    return enumerated_paths


def join_and_train(paths):
    return


if __name__ == '__main__':
    with open(f"../{MAPPING_FOLDER}/{ENUMERATED_PATHS}", 'r') as fp:
        all_paths = json.load(fp)

    dataset_configs = [
        getattr(Datasets, entry) for entry in dir(Datasets) if not entry.startswith("__")
    ]

    for dataset in dataset_configs:
        enumerated_paths = enum_paths(dataset, all_paths)
        join_and_train(enumerated_paths)
