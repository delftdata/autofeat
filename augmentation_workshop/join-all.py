import logging
import os
from datetime import timedelta
from timeit import default_timer as timer

import pandas as pd

from utils.neo4j_utils import get_nodes_with_pk_from_table
from augmentation.train_algorithms import train_CART, train_ID3, train_XGBoost

datasets = {
    "other-data/decision-trees-split/football/football.csv": ["win", "id"],
    "other-data/decision-trees-split/kidney-disease/kidney_disease.csv": ["classification", "id"],
    "other-data/decision-trees-split/steel-plate-fault/steel_plate_fault.csv": ["Class", "index"],
    "other-data/decision-trees-split/titanic/titanic.csv": ["Survived", "PassengerId"]
}

folder_name = os.path.abspath(os.path.dirname(__file__))
algorithms = ['CART', 'ID3', 'XGBoost']


def join_all(dataset_path: str) -> ():
    next_nodes = []
    visited = []
    ids = []
    id_map = {}

    filename = dataset_path.split('/')[-1]
    join_all_filename = f"join-all-{filename}"
    joined_path = f"../joined-df/{join_all_filename}"
    joined_df = None

    next_nodes.append(dataset_path)

    while len(next_nodes) > 0:
        path = next_nodes.pop()

        join_path = get_nodes_with_pk_from_table(path)
        from_source = join_path[0][0]
        visited.append(from_source)

        reduced_join_path = []
        for source, target in join_path:
            reduced_join_path.append(target)

        print(reduced_join_path)

        for target in reduced_join_path:
            base_source = '%s' % join_all_filename
            filepath = os.path.join(folder_name, f"../joined-df/{base_source}")

            if joined_df is None:
                base_source = '%s' % from_source
                filepath = os.path.join(folder_name, f"../{base_source}")

            base_df = pd.read_csv(filepath, header=0, engine="python", encoding="utf8", quotechar='"',
                                  escapechar='\\')

            from_id = target[0]
            to_source = target[1]
            to_id = target[2]

            if to_source in visited:
                continue

            if from_source in id_map:
                from_ids = list(filter(lambda x: x[0] == from_id, id_map[from_source]))
                if len(from_ids) > 0:
                    from_id = from_ids[0][1]

                print(from_id)
            else:
                id_map[from_source] = []

            if to_source in id_map:
                to_ids = list(filter(lambda x: x[0] == to_id, id_map[to_source]))
                if len(to_ids) > 0:
                    to_id = to_ids[0][1]
                print(to_id)
            else:
                id_map[to_source] = []

            next_nodes.append(to_source)

            neighbour_filepath = os.path.join(folder_name, f"../{to_source}")
            neighbour_df = pd.read_csv(neighbour_filepath, header=0, engine="python", encoding="utf8",
                                       quotechar='"',
                                       escapechar='\\')

            logging.info(
                f'\rJoin dataset {base_source} and {to_source} on left column: {from_id} and right column: {to_id}')
            joined_df = pd.merge(base_df, neighbour_df, how="left", left_on=from_id, right_on=to_id,
                                 suffixes=("", f"/{to_source}"))

            for col in joined_df.columns:
                if f"{to_source}" in col:
                    id_map[to_source].append((col.split('/')[0], col))

            joined_df.to_csv(joined_path, index=False)
            ids.append(from_id)
            ids.append(to_id)

    return ids, joined_path


def curate_df(ids, joined_path):
    joined_df = pd.read_csv(joined_path, header=0, engine="python", encoding="utf8",
                            quotechar='"',
                            escapechar='\\')

    drop_columns = list(set(joined_df.columns) & set(ids))
    for col in ids:
        drop_columns = drop_columns + list(filter(lambda x: col in x, joined_df.columns))

    joined_df.drop(columns=drop_columns, inplace=True)
    joined_df.to_csv(joined_path, index=False)

    return joined_df


def join_all_baseline(data):
    results = []

    for path, features in data.items():
        label = features[0]
        ids = features[1]

        start_join = timer()
        ids, path = join_all(path)
        dataset = curate_df(ids, path)

        print("\tEncoding data")
        df = dataset.apply(lambda x: pd.factorize(x)[0])

        X = df.drop(columns=[label])
        print(X.columns)
        y = df[label]
        end_join = timer()

        start_train = timer()
        accuracy, params = train_CART(X, y)
        end_train = timer()
        res = {
            'dataset': path,
            'accuracy': accuracy,
            'algorithm': 'CART',
            'runtime': timedelta(seconds=((end_join - start_join) + (end_train - start_train)))
        }
        res.update(params)
        results.append(res)

        start_train = timer()
        accuracy = train_ID3(X, y)
        end_train = timer()
        res = {
            'dataset': path,
            'accuracy': accuracy,
            'algorithm': 'ID3',
            'runtime': timedelta(seconds=((end_join - start_join) + (end_train - start_train)))
        }
        results.append(res)

        start_train = timer()
        accuracy, params = train_XGBoost(X, y)
        end_train = timer()
        res = {
            'dataset': path,
            'accuracy': accuracy,
            'algorithm': 'XGBoost',
            'runtime': timedelta(seconds=((end_join - start_join) + (end_train - start_train)))
        }
        res.update(params)
        results.append(res)

    print(results)

    df = pd.DataFrame(results)
    df.to_csv('../results/join-all-baseline.csv', index=False)


join_all_baseline(datasets)
