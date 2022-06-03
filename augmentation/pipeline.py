import json
import os
from os import listdir
from os.path import isfile, join

import pandas as pd

from augmentation.train_algorithms import train_CART
from data_ingestion import ingest_data
from utils.neo4j_utils import init_graph, enumerate_all_paths, drop_graph, get_relation_properties

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
folder_name = os.path.abspath(os.path.dirname(__file__))
join_path = f"{sys_path}joined-df/dt"


def data_ingestion(path: str):
    mapping = ingest_data.ingest_fabricated_data(path)
    ingest_data.ingest_connections(path, mapping)
    ingest_data.profile_valentine_all(path, mapping)
    return mapping


def path_enumeration(graph_name="graph") -> dict:
    try:
        drop_graph(graph_name)
    except Exception as err:
        print(err)

    init_graph(graph_name)
    result = enumerate_all_paths(graph_name)  # list of lists [from, to, distance]

    # transform the list of lists into a dictionary (from: [...to])
    all_paths = {}
    for pair in result:
        key, value, _ = pair
        all_paths.setdefault(key, []).append(value)

    with open(f"{os.path.join(folder_name, '../mappings')}/enumerated-paths.json", 'w') as fp:
        json.dump(all_paths, fp)
    return all_paths


def join_tables(all_paths: dict, mapping: dict, base_table):
    visited = set()
    joined_mapping = {}
    tables = []
    queue = set([base_table])
    join_paths = []
    joined_path_dir = f"{sys_path}joined-df/titanic/"

    previous_path = None
    while len(queue) > 0:
        base_table = queue.pop()
        print(f"Next node from queue: {base_table}")
        # visited.add(base_table)
        tables.append(base_table)

        if not previous_path:
            next_path = base_table
        else:
            next_path = f"{previous_path.split('.')[0]}-{base_table}"

        ids = []

        if base_table in joined_mapping:
            base_table_path = joined_mapping[base_table][0]
            ids = joined_mapping[base_table][1]
        else:
            base_table_path = mapping[base_table]
        base_table_df = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"',
                                    escapechar='\\')

        for table in all_paths[base_table]:
            if table not in tables:
                join_result_name = f"{next_path.split('.')[0]}-{table}"
                from_col, to_col = get_relation_properties(mapping[base_table], mapping[table])
                joined_path = save_join_result(base_table_df, from_col, to_col, mapping[table], join_result_name)
                if joined_path is None:
                    continue
                joined_mapping[join_result_name] = [joined_path, [from_col, to_col] + ids, base_table_df.columns]
                # visited.add(table)
                queue.add(table)

        previous_path = next_path

    print(joined_mapping)
    return joined_mapping


def join_tables_recursive(all_paths: dict, mapping, base_table, path, allp, joined_mapping=None):
    if not joined_mapping:
        joined_mapping = {}

    if not path == "":
        left_table = path.split("--")[-1]
        from_col, to_col = get_relation_properties(mapping[left_table], mapping[base_table])

        if path in joined_mapping:
            partial_join = joined_mapping[path]
        else:
            partial_join = mapping[left_table]

        left_table_df = pd.read_csv(partial_join, header=0, engine="python", encoding="utf8", quotechar='"',
                                    escapechar='\\')
        path = f"{path}--{base_table}"
        joined_path = save_join_result(left_table_df, from_col, to_col, mapping[base_table], path)
        joined_mapping[path] = joined_path
    else:
        path = base_table

    print(path)
    allp.append(path)

    for table in all_paths[base_table]:
        if table not in path:
            join_path = join_tables_recursive(all_paths, mapping, table, path, allp, joined_mapping)
            # print(f"{join_path}")
    return path


def save_join_result(base_table_df, from_col, to_col, connected_table_path, join_result_name):
    print(f"Started join result function\n\tJoining on: {connected_table_path}\n\tWith keys: {from_col} - {to_col}")
    if from_col not in base_table_df.columns:
        print(f"ERROR! Key {from_col} not in table")
        return None

    conn_table_df = pd.read_csv(connected_table_path, header=0, engine="python", encoding="utf8", quotechar='"',
                                escapechar='\\')

    if to_col not in conn_table_df.columns:
        print(f"ERROR! Key {to_col} not in table {conn_table_df}")
        return None

    joined_df = pd.merge(base_table_df, conn_table_df, how="left", left_on=from_col, right_on=to_col, suffixes=("", "_b"))
    duplicate_col = [col for col in joined_df.columns if col.endswith('_b')]
    duplicate_col.append(from_col)
    joined_df.drop(columns=duplicate_col, inplace=True)
    # joined_filename = f"{base_table_path.split('/')[-1].split('.')[0]}-{connected_table_path.split('/')[-1]}"
    joined_path = f"{join_path}/{join_result_name}"
    joined_df.to_csv(joined_path, index=False)

    return joined_path


def train_and_rank(join_path, label_column):
    rank = {}
    # Read data
    for f in listdir(join_path):
        if isfile(join(join_path, f)):
            table = pd.read_csv(join(join_path, f), header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
            df = table.apply(lambda x: pd.factorize(x)[0])
            X = df.drop(columns=[label_column])
            y = df[label_column]

            acc_decision_tree, params = train_CART(X, y)
            rank[f] = (acc_decision_tree, params)

    rank = dict(sorted(rank.items(), key=lambda item: item[1][0], reverse=True))
    with open(f"{os.path.join(folder_name, '../mappings')}/ranks.json", 'w') as fp:
        json.dump(rank, fp)
    return rank


def train_baseline(base_table_path, label_column):
    table = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
    df = table.apply(lambda x: pd.factorize(x)[0])
    X = df.drop(columns=[label_column])
    y = df[label_column]

    return train_CART(X, y)