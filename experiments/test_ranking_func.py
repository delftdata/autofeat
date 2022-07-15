import glob
import math
import os
from pathlib import Path

import pandas as pd

from augmentation.train_algorithms import train_CART
from utils.file_naming_convention import CONNECTIONS
from utils.util_functions import prepare_data_for_ml

folder_name = os.path.abspath(os.path.dirname(__file__))


def verify_ranking_func(ranking: dict, base_table_name: str, target_column: str):
    data = {}
    # 0. Get the baseline params
    print(f"Processing case 0: Baseline")
    base_table_df = pd.read_csv(os.path.join(folder_name, '../', base_table_name), header=0, engine="python",
                                encoding="utf8", quotechar='"', escapechar='\\')
    X, y = prepare_data_for_ml(base_table_df, target_column)
    acc_b, params_b, _ = train_CART(X, y)
    base_table_features = list(base_table_df.drop(columns=[target_column]).columns)

    for path in ranking.keys():
        join_path, features, score = ranking[path]

        if score == math.inf:
            continue

        result = {'base-table': (acc_b, params_b['max_depth'])}
        joined_df = pd.read_csv(join_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
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


def join_all(data_path):
    file_paths = list(Path().glob(f"../{data_path}/**/*.csv"))
    connections_df = None
    mapping = {}
    datasets = {}
    joined_df = None
    left_joined = []
    right_joined = []


    for file_path in file_paths:
        if CONNECTIONS == file_path.name:
            connections_df = pd.read_csv(file_path)
            break

    if connections_df is None:
        raise ValueError(f"{CONNECTIONS} not in folder")

    for f in file_paths:
        if f.name != CONNECTIONS:
            datasets[f.name] = pd.read_csv(f)

    for index, row in connections_df.iterrows():
        temp_joined_df = pd.merge(
            datasets[row['left_table']],
            datasets[row['right_table']],
            how='left',
            left_on=row['left_col'],
            right_on=row['right_col'],
            suffixes=(f'_{row["left_table"]}', f'_{row["right_table"]}')
        )

        if joined_df is None:
            joined_df = temp_joined_df
            left_joined.append(row['left_table'])
            right_joined.append(row['right_table'])
            continue

        if row['left_table'] in left_joined:
            joined_df = pd.merge(
                joined_df,
                datasets[row['right_table']],
                how='left',
                left_on=f'{row["left_col"]}_{row["left_table"]}',
                right_on=row['right_col'],
                suffixes=('', f'_{row["right_table"]}')
            )
            right_joined.append(row['right_table'])
            continue

        if row['left_table'] in right_joined:
            joined_df = pd.merge(
                joined_df,
                datasets[row['right_table']],
                how='left',
                left_on=f'{row["left_col"]}_{row["left_table"]}',
                right_on=row["right_col"],
                suffixes=('', f'_{row["right_table"]}')
            )
            left_joined.append(row["right_table"])

    print(joined_df)



football_data = {
    'join_result_folder_path': 'joined-df/football',
    'label_column': "win",
    'base_table_name': "football.csv",
    'path': "other-data/decision-trees-split/football",
    'mappings_folder_name': "mappings/football"
}

if __name__ == '__main__':
    join_all(football_data['path'])


