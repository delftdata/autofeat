import json
import os

import pandas as pd

from augmentation.data_preparation_pipeline import data_preparation, _data_ingestion, _path_enumeration
from augmentation.ranking import ranking_func, ranking_multigraph
from data_preparation.ingest_data import profile_valentine_all, ingest_connections
from utils.file_naming_convention import MAPPING, ENUMERATED_PATHS, RANKING_FUNCTION, RANKING_VERIFY

from experiments.config import Datasets
from experiments.test_ranking_func import verify_ranking_func

folder_name = os.path.abspath(os.path.dirname(__file__))


def pipeline(data: dict, prepare_data=False, test_ranking=False):
    join_result_folder_path = data['join_result_folder_path']
    label_column = data['label_column']
    base_table_name = data['base_table_name']
    path = data['path']
    mappings_folder_name = data['mappings_folder_name']

    if prepare_data:
        data_preparation(path, mappings_folder_name)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{MAPPING}", 'r') as fp:
        mapping = json.load(fp)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{ENUMERATED_PATHS}", 'r') as fp:
        all_paths = json.load(fp)

    allp = []
    jm = {}

    ranking_func(all_paths, mapping, base_table_name, label_column, "", allp, join_result_folder_path, jm, None)
    sorted_ranking = dict(sorted(jm.items(), key=lambda item: item[1][2]))
    print(sorted_ranking)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{RANKING_FUNCTION}", 'w') as fp:
        json.dump(sorted_ranking, fp)

    if test_ranking:
        data = verify_ranking_func(sorted_ranking, mapping, join_result_folder_path, base_table_name, label_column)
        pd.DataFrame.from_dict(data).transpose().reset_index().to_csv(
            f"{folder_name}/../{mappings_folder_name}/{RANKING_VERIFY}", index=False)


def pipeline_multigraph(data: dict, prepare_data=False, test_ranking=False):
    join_result_folder_path = data['join_result_folder_path']
    target_column = data['label_column']
    base_table_name = data['base_table_name']
    path = data['path']
    mappings_folder_name = data['mappings_folder_name']

    if prepare_data:
        mapping = ingest_connections(path)
        mapping2 = profile_valentine_all(path)
        mapping.update(mapping2)

        with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{MAPPING}", 'w') as fp:
            json.dump(mapping, fp)

        all_paths = _path_enumeration(mappings_folder_name)

        with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{ENUMERATED_PATHS}", 'w') as fp:
            json.dump(all_paths, fp)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{MAPPING}", 'r') as fp:
        mapping = json.load(fp)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{ENUMERATED_PATHS}", 'r') as fp:
        all_paths = json.load(fp)

    ranking = ranking_multigraph(all_paths, mapping, f"{path}/{base_table_name}", target_column, join_result_folder_path)
    sorted_ranking = dict(sorted(ranking.items(), key=lambda item: item[1][2]))
    print(ranking)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{RANKING_FUNCTION}", 'w') as fp:
        json.dump(sorted_ranking, fp)

    if test_ranking:
        data = verify_ranking_func(sorted_ranking, f"{path}/{base_table_name}", target_column)
        print(data)
        pd.DataFrame.from_dict(data).transpose().reset_index().to_csv(
            f"{folder_name}/../{mappings_folder_name}/{RANKING_VERIFY}", index=False)


def data_pipeline():

    prepare_data = False
    test_ranking = False
    # pipeline(football_data, prepare_data, test_ranking)
    pipeline_multigraph(Datasets.football_data, prepare_data, test_ranking)


if __name__ == '__main__':
    data_pipeline()
