import json
import os

import pandas as pd

from augmentation import pipeline
from augmentation.data_preparation_pipeline import path_enumeration, join_tables_recursive, data_preparation
from augmentation.pipeline import ranking_join_no_pruning
from augmentation.weight_training_pipeline import create_features_dataframe, create_ground_truth, \
    train_logistic_regression
from data_ingestion import ingest_data

folder_name = os.path.abspath(os.path.dirname(__file__))
join_result_path = 'joined-df/wstitanic'
join_path = f"{folder_name}/{join_result_path}"
label_column = "Survived"
# base_table = "table_0_0.csv"
base_table = "titanic.csv"
# path = "other-data/auto-fabricated/titanic/random_overlap"
path = "other-data/decision-trees-split/titanic"
base_table_path = f"{os.path.join(folder_name, path, base_table)}"
mappings_path = "mappings/wstitanic"


def main():
    # mapping = ingest_data.ingest_fabricated_data(path)
    # ingest_data.ingest_connections(path, mapping)
    #
    # all_paths = pipeline.path_enumeration()
    #
    # allp = []
    # path = join_tables_recursive(all_paths, mapping, base_table, "", allp)
    # with open(f"{os.path.join(folder_name, 'mappings')}/joined-paths.json", 'w') as fp:
    #     json.dump(allp, fp)
    #
    # ranks = pipeline.train_and_rank(join_path, label_column)

    result = pipeline.train_baseline(base_table_path, label_column)
    print(result)


def test_ingest_data():
    mapping = ingest_data.ingest_fabricated_data(path, mappings_path)
    ingest_data.ingest_connections(path, mapping)


def test_profile_valentine():
    with open(f"{os.path.join(folder_name, mappings_path)}/mapping.json", 'r') as fp:
        mapping = json.load(fp)
    ingest_data.profile_valentine_all(path, mapping)


def test_path_enumeration():
    all_paths = path_enumeration(mappings_path)


def test_join_tables_recursively():
    allp = []
    joined_mapping = {}
    with open(f"{os.path.join(folder_name, mappings_path)}/mapping.json", 'r') as fp:
        mapping = json.load(fp)

    with open(f"{os.path.join(folder_name, mappings_path)}/enumerated-paths.json", 'r') as fp:
        all_paths = json.load(fp)
    path = join_tables_recursive(all_paths, mapping, base_table, label_column, "", allp, join_result_path,
                                 joined_mapping)

    with open(f"{os.path.join(folder_name, mappings_path)}/joined-paths.json", 'w') as fp:
        json.dump(allp, fp)


def test_train_baseline():
    result = pipeline.train_baseline(base_table_path, label_column)
    print(result)


def test_train_and_rank():
    ranks = pipeline.train_and_rank(join_path, label_column)


def test_prepare_data():
    create_features_dataframe(label_column, base_table_path, join_path, mappings_path)


def test_ground_truth():
    create_ground_truth(label_column, base_table_path, join_path, mappings_path)


def test_data_preparation_pipeline():
    join_result_path = 'joined-df/wstitanic'
    join_path = f"{folder_name}/{join_result_path}"
    label_column = "Survived"
    base_table = "titanic.csv"
    path = "other-data/decision-trees-split/titanic"
    base_table_path = f"{os.path.join(folder_name, path, base_table)}"
    mappings_path = "mappings/wstitanic"
    data_preparation(base_table, label_column, path, mappings_path, join_result_path)


def evaluate_ranking_join_no_pruning():
    with open(f"{os.path.join(folder_name, mappings_path)}/mapping.json", 'r') as fp:
        mapping = json.load(fp)

    with open(f"{os.path.join(folder_name, mappings_path)}/enumerated-paths.json", 'r') as fp:
        all_paths = json.load(fp)

    allp = []
    ranking = {}
    jm = {}
    ranking_join_no_pruning(all_paths, mapping, base_table, label_column, "", allp, join_result_path, ranking, jm)
    sorted_ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True))
    with open(f"{os.path.join(folder_name, mappings_path)}/ranking.json", 'w') as fp:
        json.dump(sorted_ranking, fp)
    print(ranking)


if __name__ == '__main__':
    # main()
    # test_ingest_data()
    # test_profile_valentine()
    # test_path_enumeration()
    # test_join_tables_recursively()
    # test_train_and_rank()
    # test_train_baseline()
    # test_prepare_data()
    # test_ground_truth()
    # train_logistic_regression(mappings_path)
    evaluate_ranking_join_no_pruning()
    # test_data_preparation_pipeline()
