import json
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from augmentation.train_algorithms import train_CART, train_CART_and_print
from data_ingestion import ingest_data
from utils.neo4j_utils import init_graph, enumerate_all_paths, drop_graph, get_relation_properties
from feature_selection.feature_selection_algorithms import FSAlgorithms

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


def join_tables_recursive(all_paths: dict, mapping, current_table, target_column, path, allp, joined_mapping=None):
    if not joined_mapping:
        joined_mapping = {}

    # Join and save the join result
    if not path == "":
        # get the name of the table from the path we created
        left_table = path.split("--")[-1]
        # get the location of the table we want to join with
        partial_join = mapping[left_table]

        # If we already joined the tables on the path, we retrieve the join result
        if path in joined_mapping:
            partial_join = joined_mapping[path]

        # Add the current table to the path
        path = f"{path}--{current_table}"
        joined_path, joined_df, left_table_features = join_and_save(partial_join, mapping[left_table], mapping[current_table], path)
        joined_mapping[path] = joined_path
    else:
        # Just started traversing, the path is the current table
        path = current_table

    print(path)
    allp.append(path)

    # Depth First Search recursively
    for table in all_paths[current_table]:
        # Break the cycles in the data, only visit new nodes
        if table not in path:
            join_path = join_tables_recursive(all_paths, mapping, table, target_column, path, allp, joined_mapping)
            # print(f"{join_path}")
    return path


def apply_feat_sel(joined_dataframe, base_table_features, target_column):
    # Remove the features from the base table
    joined_table_no_base = joined_dataframe.drop(
        columns=set(joined_dataframe.columns).intersection(set(base_table_features)))
    # Transform data (factorize)
    df = joined_table_no_base.apply(lambda x: pd.factorize(x)[0])

    X = np.array(df.drop(columns=[target_column]))
    y = np.array(df[target_column])

    fs = FSAlgorithms()
    # create dataset to feed to the classifier
    result = {'columns': [], FSAlgorithms.T_SCORE: [], FSAlgorithms.CHI_SQ: [], FSAlgorithms.FISHER: [], FSAlgorithms.SU: [], FSAlgorithms.MIFS: [], FSAlgorithms.CIFE: []}
    result['columns'] = list(df.drop(columns=[target_column]).columns)

    # For each feature selection algorithm, get the score
    for alg in fs.ALGORITHMS:
        scores = fs.feature_selection(alg, X, y)
        result[alg] = list(scores)

    dataframe = pd.DataFrame.from_dict(result)
    return dataframe


def classify_and_rank(join_path_name, features_dataframe, classifier, ranking):
    features_dataframe['score'] = classifier.predict(features_dataframe.drop(columns=['columns']))
    n_rows = len(features_dataframe[features_dataframe['score'] > 0.5])
    max_score = features_dataframe['score'].max()
    ranking[join_path_name] = (max_score, n_rows)


def join_and_save(partial_join_path, left_table_path, right_table_path, join_result_name):
    # Getting the join keys
    from_col, to_col = get_relation_properties(left_table_path, right_table_path)
    # Read left side table
    left_table_df = pd.read_csv(partial_join_path, header=0, engine="python", encoding="utf8", quotechar='"',
                                escapechar='\\')
    if from_col not in left_table_df.columns:
        print(f"ERROR! Key {from_col} not in table")
        return None

    right_table_df = pd.read_csv(right_table_path, header=0, engine="python", encoding="utf8", quotechar='"',
                                escapechar='\\')
    if to_col not in right_table_df.columns:
        print(f"ERROR! Key {to_col} not in table {right_table_df}")
        return None

    print(f"\tJoining {partial_join_path} with {right_table_path}\n\tOn keys: {from_col} - {to_col}")
    joined_df = pd.merge(left_table_df, right_table_df, how="left", left_on=from_col, right_on=to_col, suffixes=("", "_b"))

    # If both tables have the same column, drop one of them
    duplicate_col = [col for col in joined_df.columns if col.endswith('_b')]
    # Drop the FK key from the left table
    duplicate_col.append(from_col)
    joined_df.drop(columns=duplicate_col, inplace=True)
    # Save join result
    joined_path = f"{join_path}/{join_result_name}"
    joined_df.to_csv(joined_path, index=False)

    return joined_path, joined_df, list(left_table_df.columns)


def train_and_rank(join_path, label_column):
    rank = {}
    # Read data
    for f in listdir(join_path):
        if isfile(join(join_path, f)):
            table = pd.read_csv(join(join_path, f), header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
            df = table.apply(lambda x: pd.factorize(x)[0])
            X = df.drop(columns=[label_column])
            y = df[label_column]

            # acc_decision_tree, params = train_CART(X, y)
            acc_decision_tree, params, feat_score = train_CART_and_print(X, y, f, f"{join_path}/trees")
            rank[f] = (acc_decision_tree, params, list(X.columns), list(feat_score))

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