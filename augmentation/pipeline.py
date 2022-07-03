import json
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from augmentation.train_algorithms import train_CART, train_CART_and_print
from utils.file_naming_convention import ENUMERATED_PATHS
from feature_selection.feature_selection_algorithms import FSAlgorithms

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
folder_name = os.path.abspath(os.path.dirname(__file__))


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