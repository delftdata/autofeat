import math
import os

import pandas as pd

from augmentation.train_algorithms import train_CART
from utils.util_functions import prepare_data_for_ml

folder_name = os.path.abspath(os.path.dirname(__file__))


def verify_ranking_func(ranking: dict, mapping: dict, joined_data_path: str, base_table_name: str,
                        target_column: str):
    data = {}
    # 0. Get the baseline params
    print(f"Processing case 0: Baseline")
    base_table_df = pd.read_csv(os.path.join(folder_name, '../', mapping[base_table_name]), header=0, engine="python",
                                encoding="utf8", quotechar='"', escapechar='\\')
    X, y = prepare_data_for_ml(base_table_df, target_column)
    acc_b, params_b, _ = train_CART(X, y)
    base_table_features = list(base_table_df.drop(columns=[target_column]).columns)

    for path in ranking.keys():
        _, features, score = ranking[path]

        if score == math.inf:
            continue

        result = {'base-table': (acc_b, params_b['max_depth'])}
        joined_df = pd.read_csv(f"{folder_name}/../{joined_data_path}/{path}", header=0,
                                engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        # Three type of experiments
        # 1. Keep the entire path
        print(f"Processing case 1: Keep the entire path")
        X, y = prepare_data_for_ml(joined_df, target_column)
        acc, params, _ = train_CART(X, y)
        result['keep-all'] = (acc, params['max_depth'])
        print(X.columns)
        print(result)

        # 2. Remove all, but the ranked feature
        print(f"Processing case 2: Remove all, but the ranked feature")
        aux_df = joined_df.copy(deep=True)
        aux_features = list(joined_df.columns)
        aux_features.remove(target_column)
        columns_to_drop = [c for c in aux_features if (c not in base_table_features) and (c not in features)]
        aux_df.drop(columns=columns_to_drop, inplace=True)

        X, y = prepare_data_for_ml(aux_df, target_column)
        acc, params, _ = train_CART(X, y)
        result['keep-all-but-ranked'] = (acc, params['max_depth'])
        print(X.columns)
        print(result)
        data[path] = {'features': features}
        data[path].update(result)

    return data
