import os

import pandas as pd

from augmentation.pipeline import prepare_data_for_ml
from augmentation.train_algorithms import train_CART
from utils.util_functions import get_top_k_from_dict

folder_name = os.path.abspath(os.path.dirname(__file__))


def verify_join_no_pruning(ranking: dict, mapping: dict, joined_data_path: str, base_table_name: str,
                           target_column: str):
    data = {}
    top_k_ranking = get_top_k_from_dict(ranking, 5)
    # 0. Get the baseline params
    print(f"Processing case 0: Baseline")
    base_table_df = pd.read_csv(os.path.join(folder_name, '../', mapping[base_table_name]), header=0, engine="python",
                                encoding="utf8", quotechar='"', escapechar='\\')
    X, y = prepare_data_for_ml(base_table_df, target_column)
    acc_b, params_b, _ = train_CART(X, y)
    base_table_features = list(base_table_df.drop(columns=[target_column]).columns)
    for path in top_k_ranking.keys():
        result = {'base-table': (acc_b, params_b['max_depth'])}
        feature_name, joined_path, tables = _parse_ranking_path(path)
        joined_df = pd.read_csv(f"{folder_name}/../{joined_data_path}/{joined_path}", header=0,
                                engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        # Three type of experiments
        # 1. Keep the entire path
        print(f"Processing case 1: Keep the entire path")
        X, y = prepare_data_for_ml(joined_df, target_column)
        acc, params, _ = train_CART(X, y)
        result['keep-all'] = (acc, params['max_depth'])
        print(X.columns)
        print(result)

        # 2. Remove all, but the last table (which contains the ranked feature)
        print(f"Processing case 2: Remove all, but the last table")
        aux_df = joined_df.copy(deep=True)
        for i in range(1, len(tables)-1):
            aux_features = list(pd.read_csv(os.path.join(folder_name, '../', mapping[tables[i]]), header=0,
                                            engine="python", encoding="utf8", quotechar='"',
                                            escapechar='\\', nrows=1).columns)
            columns_to_drop = set(aux_df.columns).intersection(set(aux_features))
            aux_df.drop(columns=columns_to_drop, inplace=True)
        X, y = prepare_data_for_ml(aux_df, target_column)
        acc, params, _ = train_CART(X, y)
        result['keep-all-but-last'] = (acc, params['max_depth'])
        print(X.columns)
        print(result)

        # 3. Remove all, but the ranked feature
        print(f"Processing case 3: Remove all, but the ranked feature")
        aux_features = list(pd.read_csv(os.path.join(folder_name, '../', mapping[tables[-1]]), header=0,
                                        engine="python", encoding="utf8", quotechar='"',
                                        escapechar='\\', nrows=1).columns)
        aux_features.remove(feature_name)
        columns_to_drop = [c for c in list(aux_df.columns) if (c not in base_table_features) and (c in aux_features)]
        # columns_to_drop = set(aux_df.columns).intersection(set(aux_features))
        aux_df.drop(columns=columns_to_drop, inplace=True)
        X, y = prepare_data_for_ml(aux_df, target_column)
        acc, params, _ = train_CART(X, y)
        result['keep-all-but-ranked'] = (acc, params['max_depth'])
        print(X.columns)
        print(result)
        data[path] = result
    return data


def _parse_ranking_path(path: str):
    tokens = path.split('/')
    joined_path = tokens[0]
    feature_name = tokens[1]
    tables = joined_path.split('--')

    return feature_name, joined_path, tables


