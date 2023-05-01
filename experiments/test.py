import json
import time
from typing import List

import pandas as pd

from augmentation.bfs_pipeline import BFS_Augmentation
from augmentation.trial_error import dfs_traverse_join_pipeline, train_test_cart
from config import RESULTS_FOLDER, JOIN_RESULT_FOLDER
from data_preparation.dataset_base import Dataset
from experiments.result_object import Result
from graph_processing.traverse_graph import dfs_traversal
from tfd_datasets import CLASSIFICATION_DATASETS, REGRESSION_DATASETS, CLASSIFICATION_DATASETS_NEW, school_small


def test_base_accuracy(dataset: Dataset):
    print(f"Base result on table {dataset.base_table_id}")
    dataframe = pd.read_csv(dataset.base_table_id, header=0, engine="python", encoding="utf8", quotechar='"',
                            escapechar='\\')
    entry = train_test_cart(dataframe=dataframe,
                            target_column=dataset.target_column,
                            regression=dataset.dataset_type)
    entry.approach = Result.BASE
    entry.data_label = dataset.base_table_label
    entry.data_path = dataset.base_table_label

    return [entry]


def test_arda(dataset: Dataset, sample_size: int = 1000) -> List:
    from arda.arda import select_arda_features_budget_join

    print(f"ARDA result on table {dataset.base_table_id}")

    start = time.time()
    dataframe, dataframe_label, selected_features, join_name = select_arda_features_budget_join(
        base_node_id=str(dataset.base_table_id),
        target_column=dataset.target_column,
        base_table_features=dataset.base_table_features,
        sample_size=sample_size,
        regression=dataset.dataset_type)
    end = time.time()
    print(f"X shape: {dataframe.shape}\nSelected features:\n\t{selected_features}")

    features = [f"{dataframe_label}.{feat}" for feat in dataset.base_table_features]
    features.extend(selected_features)
    features.append(dataset.target_column)

    entry = train_test_cart(dataframe=dataframe[features],
                            target_column=dataset.target_column,
                            regression=dataset.dataset_type)
    entry.feature_selection_time = end - start
    entry.approach = Result.ARDA
    entry.data_label = dataset.base_table_label
    entry.data_path = join_name

    return [entry]


def test_dfs_pipeline(dataset: Dataset, value_ratio: float = 0.55, gini: bool = False) -> List:
    print(f"DFS result with table {dataset.base_table_id}")

    start = time.time()
    join_path_tree = {}
    dfs_traversal(base_node_id=str(dataset.base_table_id), discovered=[], join_tree=join_path_tree)

    join_name_mapping = {}
    paths_score = {}
    train_results = []
    all_paths = dfs_traverse_join_pipeline(base_node_id=str(dataset.base_table_id), target_column=dataset.target_column,
                                           base_table_label=dataset.base_table_label, join_tree=join_path_tree,
                                           join_name_mapping=join_name_mapping, value_ratio=value_ratio,
                                           paths_score=paths_score, gini=gini)
    end = time.time()
    print(f"FINISHED DFS")

    # Train, test each path
    print(f"TRAIN WITHOUT feature selection")
    for join_name in join_name_mapping.keys():
        joined_df = pd.read_csv(JOIN_RESULT_FOLDER / dataset.base_table_label / join_name_mapping[join_name], header=0,
                                engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        result = train_test_cart(dataframe=joined_df,
                                 target_column=dataset.target_column,
                                 regression=dataset.dataset_type)
        result.feature_selection_time = end - start
        result.approach = "TFD_DFS"
        result.data_label = dataset.base_table_label
        result.data_path = join_name
        train_results.append(result)

    # Save results
    pd.Series(list(all_paths), name="filename").to_csv(
        RESULTS_FOLDER / f"paths_{dataset.base_table_label}_dfs_{value_ratio}.csv", index=False)
    pd.DataFrame(train_results).to_csv(
        RESULTS_FOLDER / f"results_{dataset.base_table_label}_dfs_{value_ratio}.csv", index=False)
    pd.DataFrame.from_dict(join_name_mapping, orient='index', columns=["join_name"]).to_csv(
        RESULTS_FOLDER / f'join_mapping_{dataset.base_table_label}_dfs_{value_ratio}.csv')
    with open(RESULTS_FOLDER / f"all_paths_{dataset.base_table_label}_dfs_{value_ratio}.json", "w") as f:
        json.dump(paths_score, f)
    with open(f'join_tree_{dataset.base_table_label}_dfs.json', 'w') as f:
        json.dump(join_path_tree, f)

    return train_results


def test_bfs_pipeline(dataset: Dataset, value_ratio: float = 0.55, gini: bool = False) -> List:
    print(f"BFS result with table {dataset.base_table_id}")

    start = time.time()
    bfs_traversal = BFS_Augmentation(base_table_label=dataset.base_table_label,
                                     target_column=dataset.target_column,
                                     value_ratio=value_ratio)
    bfs_traversal.bfs_traverse_join_pipeline(queue={str(dataset.base_table_id)})
    end = time.time()

    print("FINISHED BFS")

    print(bfs_traversal.join_name_mapping)

    results = []
    # # Train, test each path
    # print(f"TRAIN WITHOUT feature selection")
    # for join_name in join_name_mapping.keys():
    #     joined_df = pd.read_csv(JOIN_RESULT_FOLDER / dataset.base_table_label / join_name_mapping[join_name], header=0,
    #                             engine="python",
    #                             encoding="utf8", quotechar='"', escapechar='\\')
    #     result = train_test_cart(dataframe=joined_df,
    #                              target_column=dataset.target_column,
    #                              regression=dataset.dataset_type)
    #     result.feature_selection_time = end - start
    #     result.approach = "TFD_BFS"
    #     result.data_label = dataset.base_table_label
    #     result.data_path = join_name
    #     results.append(result)
    #
    # # Save results
    # pd.DataFrame(results).to_csv(
    #     RESULTS_FOLDER / f"results_{dataset.base_table_label}_bfs_{value_ratio}.csv", index=False)
    # pd.DataFrame.from_dict(join_name_mapping, orient='index', columns=["join_name"]).to_csv(
    #     RESULTS_FOLDER / f'join_mapping_{dataset.base_table_label}_bfs_{value_ratio}.csv')
    # with open(RESULTS_FOLDER / f"all_paths_{dataset.base_table_label}_bfs_{value_ratio}.json", "w") as f:
    #     json.dump(all_paths, f)

    return results


def aggregate_results():
    all_results = []

    for dataset in CLASSIFICATION_DATASETS_NEW:
    # for dataset in REGRESSION_DATASETS:
        result_base = test_base_accuracy(dataset)
        all_results.extend(result_base)
        # result_arda = test_arda(dataset, sample_size=3000)
        # all_results.extend(result_arda)
        result_bfs = test_bfs_pipeline(dataset, value_ratio=0.5)
        all_results.extend(result_bfs)
        result_dfs = test_dfs_pipeline(dataset, value_ratio=0.5)
        all_results.extend(result_dfs)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"all_results_cls_new.csv", index=False)


test_bfs_pipeline(school_small, value_ratio=0.45)
# test_dfs_pipeline()
# test_base_accuracy(REGRESSION_DATASETS[0])
# test_arda()

# aggregate_results()
