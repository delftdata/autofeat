import json
import os

from augmentation import pipeline
from augmentation.pipeline import join_tables_recursive
from augmentation.weight_training import create_features_dataframe, create_ground_truth, train_logistic_regression
from data_ingestion import ingest_data

folder_name = os.path.abspath(os.path.dirname(__file__))
join_path = f"{folder_name}/joined-df/dt"
label_column = "Survived"
base_table = "table_0_0.csv"
path = "other-data/auto-fabricated/titanic/random_overlap"
base_table_path = f"{os.path.join(folder_name, path, base_table)}"


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
    mapping = ingest_data.ingest_fabricated_data(path)
    ingest_data.ingest_connections(path, mapping)


def test_profile_valentine():
    with open(f"{os.path.join(folder_name, 'mappings')}/mapping.json", 'r') as fp:
        mapping = json.load(fp)
    ingest_data.profile_valentine_all(path, mapping)


def test_path_enumeration():
    all_paths = pipeline.path_enumeration()


def test_join_tables_recursively():
    allp = []
    joined_mapping = {}
    with open(f"{os.path.join(folder_name, 'mappings')}/mapping.json", 'r') as fp:
        mapping = json.load(fp)

    with open(f"{os.path.join(folder_name, 'mappings')}/enumerated-paths.json", 'r') as fp:
        all_paths = json.load(fp)
    path = join_tables_recursive(all_paths, mapping, base_table, "", allp, joined_mapping)

    with open(f"{os.path.join(folder_name, 'mappings')}/joined-paths.json", 'w') as fp:
        json.dump(allp, fp)


def test_train_baseline():
    result = pipeline.train_baseline(base_table_path, label_column)
    print(result)


def test_train_and_rank():
    ranks = pipeline.train_and_rank(join_path, label_column)


def test_prepare_data():
    create_features_dataframe(base_table_path, join_path, label_column)


def test_ground_truth():
    create_ground_truth(join_path, label_column, base_table_path)


if __name__ == '__main__':
    # main()
    # test_ingest_data()
    # test_profile_valentine()
    # test_path_enumeration()
    # test_join_tables_recursively()
    # test_train_and_rank()
    # test_train_baseline()
    # test_prepare_data()
    test_ground_truth()
    # train_logistic_regression()
