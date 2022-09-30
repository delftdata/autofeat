import json
import math
import os

import pandas as pd

from augmentation.data_preparation_pipeline import data_preparation
from augmentation.ranking import ranking_func, Ranking
from augmentation.train_algorithms import train_CART
from data_preparation.dataset_base import Dataset
from data_preparation.utils import prepare_data_for_ml, get_join_path
from experiments.datasets import Datasets
from experiments.result_object import Result
from experiments.utils import CART, map_features_scores
from utils_module.file_naming_convention import JOIN_RESULT_FOLDER
from utils_module.file_naming_convention import (
    MAPPING,
    ENUMERATED_PATHS,
    RANKING_FUNCTION,
    RANKING_VERIFY,
    MAPPING_FOLDER,
)
from utils_module.util_functions import objects_to_dict

folder_name = os.path.abspath(os.path.dirname(__file__))


def pipeline(data: dict, prepare_data=False, test_ranking=False):
    join_result_folder_path = data["join_result_folder_path"]
    label_column = data["label_column"]
    base_table_name = data["base_table_name"]
    path = data["path"]
    mappings_folder_name = data["mappings_folder_name"]

    if prepare_data:
        data_preparation(path, mappings_folder_name)

    with open(f"{os.path.join(folder_name, '../', mappings_folder_name)}/{MAPPING}", "r") as fp:
        mapping = json.load(fp)

    with open(
        f"{os.path.join(folder_name, '../', mappings_folder_name)}/{ENUMERATED_PATHS}", "r"
    ) as fp:
        all_paths = json.load(fp)

    allp = []
    jm = {}

    ranking_func(
        all_paths,
        mapping,
        base_table_name,
        label_column,
        "",
        allp,
        join_result_folder_path,
        jm,
        None,
    )
    sorted_ranking = dict(sorted(jm.items(), key=lambda item: item[1][2]))
    print(sorted_ranking)

    with open(
        f"{os.path.join(folder_name, '../', mappings_folder_name)}/{RANKING_FUNCTION}", "w"
    ) as fp:
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
        results = verify_ranking_func(dataset, ranking.ranked_paths)
        data = objects_to_dict(results)
        print(data)
        pd.DataFrame(data).to_csv(f"../{MAPPING_FOLDER}/{RANKING_VERIFY}", index=False)


def verify_ranking_func(dataset, ranked_paths=None):
    results = []
    # 0. Get the baseline params
    print(f"Processing case 0: Baseline")

    if dataset.base_table_df is None:
        dataset.set_base_table_df()

    X_b, y = prepare_data_for_ml(dataset.base_table_df, dataset.target_column)
    acc_b, params_b, feature_imp_b, _, _ = train_CART(X_b, y)
    entry = Result(
        approach=Result.BASE,
        data_path=dataset.base_table_id,
        data_label=dataset.base_table_label,
        algorithm=CART,
        depth=params_b["max_depth"],
        accuracy=acc_b,
        feature_importance=map_features_scores(feature_imp_b, X_b),
    )
    results.append(entry)

    if ranked_paths is None:
        tfd_ranking = Ranking(dataset)
        tfd_ranking.start_ranking()
        ranked_paths = tfd_ranking.ranked_paths

    for ranked_path in ranked_paths:
        join_path = get_join_path(ranked_path.path)

        if ranked_path.score == math.inf:
            continue

        joined_df = pd.read_csv(
            f"../{JOIN_RESULT_FOLDER}/{join_path}",
            header=0,
            engine="python",
            encoding="utf8",
            quotechar='"',
            escapechar="\\",
        )
        # Two type of experiments
        # 1. Keep the entire path
        print(f"Processing case 1: Keep the entire path")
        X, y = prepare_data_for_ml(joined_df, dataset.target_column)
        acc, params, feature_imp, _, _ = train_CART(X, y)

        entry = Result(
            approach=Result.TFD_PATH,
            data_path=ranked_path.path,
            data_label=dataset.base_table_label,
            algorithm=CART,
            depth=params["max_depth"],
            accuracy=acc,
            feature_importance=map_features_scores(feature_imp, X),
        )
        results.append(entry)

        # 2. Remove all, but the ranked feature
        print(f"Processing case 2: Remove all, but the ranked feature")
        aux_df = joined_df.copy(deep=True)
        aux_features = list(joined_df.columns)
        aux_features.remove(dataset.target_column)
        columns_to_drop = [
            c
            for c in aux_features
            if (c not in dataset.base_table_features) and (c not in ranked_path.features)
        ]
        aux_df.drop(columns=columns_to_drop, inplace=True)

        X, y = prepare_data_for_ml(aux_df, dataset.target_column)
        acc, params, feature_imp, _, _ = train_CART(X, y)

        entry = Result(
            approach=Result.TFD,
            data_path=ranked_path.path,
            data_label=dataset.base_table_label,
            algorithm=CART,
            depth=params["max_depth"],
            accuracy=acc,
            feature_importance=map_features_scores(feature_imp, X),
        )
        results.append(entry)
    return results


def data_pipeline(prepare_data=False):
    if prepare_data:
        data_preparation()

    test_ranking = False
    pipeline_multigraph(Datasets.steel_plate_fault, test_ranking)


if __name__ == "__main__":
    data_pipeline()
