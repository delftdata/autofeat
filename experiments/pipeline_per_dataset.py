import json
import os

import pandas as pd

from augmentation.data_preparation_pipeline import data_preparation
from augmentation.ranking import ranking_func, Ranking
from data_preparation.dataset_base import Dataset
from experiments.accuracy_experiments import Experiments
from experiments.datasets import Datasets
from utils.file_naming_convention import MAPPING, ENUMERATED_PATHS, RANKING_FUNCTION, RANKING_VERIFY, MAPPING_FOLDER
from utils.util_functions import objects_to_dict

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

    # if test_ranking:
    #     data = verify_ranking_func(sorted_ranking, mapping, join_result_folder_path, base_table_name, label_column)
    #     pd.DataFrame.from_dict(data).transpose().reset_index().to_csv(
    #         f"{folder_name}/../{mappings_folder_name}/{RANKING_VERIFY}", index=False)


def pipeline_multigraph(dataset: Dataset, test_ranking=False):
    ranking = Ranking(dataset)
    ranking.start_ranking()
    print(objects_to_dict(ranking.ranked_paths))

    if test_ranking:
        experiments = Experiments(dataset, ranking.ranked_paths).verify_ranking_func()
        data = objects_to_dict(experiments.results)
        print(data)
        # pd.DataFrame.from_dict(data).transpose().reset_index().to_csv(f"../{MAPPING_FOLDER}/{RANKING_VERIFY}", index=False)
        pd.DataFrame(data).to_csv(f"../{MAPPING_FOLDER}/{RANKING_VERIFY}", index=False)


def data_pipeline(prepare_data=False):
    if prepare_data:
        data_preparation()

    test_ranking = False
    pipeline_multigraph(Datasets.steel_plate_fault, test_ranking)


if __name__ == '__main__':
    data_pipeline()
