import logging
import os
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
import pandas as pd

from utils.neo4j_utils import get_nodes_with_pk_from_table
from augmentation.train_algorithms import train_CART, train_ID3, train_XGBoost
from FeatSel import FeatSel
from augmentation.algorithms import CART, ID3, XGB

logging.basicConfig(level=logging.INFO)
folder_name = os.path.abspath(os.path.dirname(__file__))


def select_join_paths(dataset_path: str, label: str) -> dict:
    next_nodes = []
    visited = []
    selected_paths = {}

    next_nodes.append(dataset_path)

    while len(next_nodes) > 0:
        path = next_nodes.pop()

        join_path = get_nodes_with_pk_from_table(path)
        reduced_join_path = {}

        for source, target in join_path:
            if source not in reduced_join_path:
                reduced_join_path[source] = []
            reduced_join_path[source].append(target)

        print(reduced_join_path)

        base_keys = reduced_join_path.keys()
        for base_source in base_keys:
            if base_source in selected_paths:
                filepath = os.path.join(folder_name, f"../joined-df/{selected_paths[base_source][0]}")
            else:
                filepath = os.path.join(folder_name, f"../{base_source}")

            base_df = pd.read_csv(filepath, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
            base_df_columns = list(base_df.columns)
            base_df_columns.remove(label)
            visited.append(base_source)

            from_source = '%s' % base_source
            for target in reduced_join_path[base_source]:

                if base_source in selected_paths:
                    base_source = selected_paths[base_source][0]

                from_id = target[0]
                to_source = target[1]
                to_id = target[2]

                if to_source in visited:
                    continue

                next_nodes.append(to_source)

                neighbour_filepath = os.path.join(folder_name, f"../{to_source}")
                neighbour_df = pd.read_csv(neighbour_filepath, header=0, engine="python", encoding="utf8",
                                           quotechar='"',
                                           escapechar='\\')

                logging.info(
                    f'\rJoin dataset {base_source} and {to_source} on left column: {from_id} and right column: {to_id}')
                joined_df = pd.merge(base_df, neighbour_df, how="left", left_on=from_id, right_on=to_id,
                                     suffixes=("_b", ""))

                duplicate_col = [col for col in joined_df.columns if col.endswith('_b')]
                joined_df.drop(columns=duplicate_col, inplace=True)
                print(f'\t\t{joined_df.columns}')

                joined_filename = f"{base_source.split('/')[-1].split('.')[0]}-{to_source.split('/')[-1]}"
                joined_path = f"../joined-df/{joined_filename}"
                joined_df.to_csv(joined_path, index=False)
                selected_paths[to_source] = (joined_filename, [from_id, to_id], from_source, base_df_columns)

    print(selected_paths)
    return selected_paths


def process_join_paths(join_paths: dict):
    processed = {}

    for to_path, elements in join_paths.items():
        joined_df, ids, from_path, base_columns = elements
        from_id = ids[0]
        to_id = ids[1]
        paths = [from_path, to_path]
        ids = [from_id, to_id]

        if from_path in join_paths:
            paths, ids = update_path(join_paths, from_path, paths, ids)

        processed[joined_df] = [paths, ids, base_columns]

    print(processed)
    return processed


def update_path(join_paths, from_path, paths, ids):
    if from_path in join_paths:
        _, id_s, path, _ = join_paths[from_path]
        paths.insert(0, path)
        ids.insert(0, id_s[0])
        ids.insert(1, id_s[1])
        return update_path(join_paths, path, paths, ids)
    else:
        return paths, ids


def rank_join_paths(join_paths: dict, label: str, selection_method):
    ranked = {}

    for joined_df, elements in join_paths.items():
        ids = elements[1]
        base_columns = elements[2]

        filepath = os.path.join(folder_name, f"../joined-df/{joined_df}")
        base_df = pd.read_csv(filepath, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')

        columns_to_drop = list(set(base_df.columns) & set(ids))
        columns_to_drop = columns_to_drop + base_columns
        df = base_df.drop(columns=columns_to_drop)
        df = df.apply(lambda x: pd.factorize(x)[0])

        X = np.array(df.drop(columns=[label]))
        y = np.array(df[label])

        scores = FeatSel().feature_selection(selection_method, X, y)
        max_score = max(scores)

        ranked[joined_df] = max_score

    ranked = dict(sorted(ranked.items(), key=lambda item: item[1], reverse=True))
    print(ranked)
    return ranked


def get_top_k_from_dict(join_paths: dict, k: int):
    return {key: join_paths[key] for i, key in enumerate(join_paths) if i < k}


def train_augmented(ranked: dict, paths: dict, label: str, algorithm: str, runtime, feat_sel) -> list:
    tables_acc = []

    for filename, score in ranked.items():
        elements = paths[filename]
        sources = elements[0]
        ids = elements[1]

        start_train = timer()
        base_filepath = os.path.join(folder_name, f"../joined-df/{filename}")
        aug_df = pd.read_csv(base_filepath, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')

        columns_to_drop = list(set(aug_df.columns) & set(ids))
        dataset = aug_df.drop(columns=columns_to_drop)

        print("\tEncoding data")
        df = dataset.apply(lambda x: pd.factorize(x)[0])

        print("\tSplit X, y")
        y = df[label]
        X = df.copy().drop([label], axis=1)

        print(f'Shape: \t {X.shape}')
        print(X.columns)

        params = None
        if algorithm == CART:
            accuracy, params = train_CART(X, y)
        elif algorithm == ID3:
            accuracy = train_ID3(X, y)
        elif algorithm == XGB:
            accuracy, params = train_XGBoost(X, y)
        else:
            ValueError('The algorithm does not exist')
            return tables_acc

        end_train = timer()
        logging.info(f'Acc aug dataset: {accuracy}')
        res = {
            'dataset': filename,
            'accuracy': accuracy,
            'algorithm': algorithm,
            'score': score,
            'feat_sel': feat_sel,
            'runtime': timedelta(seconds=(runtime + (end_train - start_train))),
            'path': sources,
            'path_ids': ids,
        }

        if params:
            res.update(params)

        tables_acc.append(res)

    return tables_acc

