import json
import os

import pandas as pd

from augmentation.data_preparation_pipeline import data_preparation
from augmentation.ranking import ranking_func
from experiments.test_ranking_func import verify_ranking_func
from utils.file_naming_convention import MAPPING, ENUMERATED_PATHS, RANKING_FUNCTION, RANKING_VERIFY

folder_name = os.path.abspath(os.path.dirname(__file__))


def pipeline(data: dict, prepare_data=False, test_ranking=False):
    join_result_folder_path = data['join_result_folder_path']
    label_column = data['label_column']
    base_table_name = data['base_table_name']
    path = data['path']
    mappings_folder_name = data['mappings_folder_name']

    if prepare_data:
        data_preparation(base_table_name, label_column, path, mappings_folder_name, join_result_folder_path)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{MAPPING}", 'r') as fp:
        mapping = json.load(fp)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{ENUMERATED_PATHS}", 'r') as fp:
        all_paths = json.load(fp)

    allp = []
    ranking = {}
    jm = {}

    ranking_func(all_paths, mapping, base_table_name, label_column, "", allp, join_result_folder_path, ranking, jm)
    sorted_ranking = dict(sorted(ranking.items(), key=lambda item: item[1][0]))
    print(sorted_ranking)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{RANKING_FUNCTION}", 'w') as fp:
        json.dump(sorted_ranking, fp)

    if test_ranking:
        data = verify_ranking_func(sorted_ranking, mapping, join_result_folder_path, base_table_name, label_column)
        pd.DataFrame.from_dict(data).transpose().reset_index().to_csv(
            f"{folder_name}/../{mappings_folder_name}/{RANKING_VERIFY}", index=False)


def data_pipeline():
    titanic_data = {
        'join_result_folder_path': 'joined-df/titanic',
        'label_column': "Survived",
        'base_table_name': "titanic.csv",
        'path': "other-data/decision-trees-split/titanic",
        'mappings_folder_name': "mappings/titanic"
    }

    steel_data = {
        'join_result_folder_path': 'joined-df/steel-plate-fault',
        'label_column': "Class",
        'base_table_name': "steel_plate_fault.csv",
        'path': "other-data/decision-trees-split/steel-plate-fault",
        'mappings_folder_name': "mappings/steel-plate-fault"
    }

    football_data = {
        'join_result_folder_path': 'joined-df/football',
        'label_column': "win",
        'base_table_name': "football.csv",
        'path': "other-data/decision-trees-split/football",
        'mappings_folder_name': "mappings/football"
    }

    kidney_data = {
        'join_result_folder_path': 'joined-df/kidney-disease',
        'label_column': "classification",
        'base_table_name': "kidney_disease.csv",
        'path': "other-data/decision-trees-split/kidney-disease",
        'mappings_folder_name': "mappings/kidney-disease"
    }

    prepare_data = False
    test_ranking = True
    pipeline(football_data, prepare_data, test_ranking)


if __name__ == '__main__':
    data_pipeline()
