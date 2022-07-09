import json
import os
from os import listdir
from os.path import isfile, join

import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

from augmentation.train_algorithms import train_CART, train_CART_and_print
from data_preparation.join_data import join_and_save
from feature_selection.util_functions import apply_feat_sel
from utils.util_functions import prepare_data_for_ml

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
folder_name = os.path.abspath(os.path.dirname(__file__))


def ranking_join_no_pruning(all_paths: dict, mapping, current_table, target_column, path, allp, join_result_folder_path,
                            ranking, joined_mapping: dict):

    # Join and save the join result
    if not path == "":
        # get the name of the table from the path we created
        left_table = path.split("--")[-1]
        # get the location of the table we want to join with
        partial_join_path = mapping[left_table]

        # If we already joined the tables on the path, we retrieve the join result
        if path in joined_mapping:
            partial_join_path = joined_mapping[path]

        # Add the current table to the path
        path = f"{path}--{current_table}"

        # Recursion logic
        # 1. Join existing left table with the current table visiting
        joined_path, joined_df, left_table_df = join_and_save(partial_join_path, mapping[left_table],
                                                              mapping[current_table], join_result_folder_path, path)
        # 2. Apply filter-based feature selection and normalise data
        new_features_ranks = apply_feat_sel(joined_df, left_table_df, target_column, path)
        # 3. Use the scores from the feature selection to predict the rank and add it to the ranking set
        result = classify_and_rank(new_features_ranks)
        ranking.update(result)
        # 4. Save the join for future reference/usage
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
            join_path = ranking_join_no_pruning(all_paths, mapping, table, target_column, path, allp, join_result_folder_path,
                                                ranking, joined_mapping)
    return path


def classify_and_rank(features_dataframe: pd.DataFrame) -> dict:
    classifier = load(os.path.join(folder_name, '../mappings/regressor.joblib'))

    X = features_dataframe.drop(columns=['columns'])
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    normalized_X = pd.DataFrame(scaled_X, columns=X.columns)

    features_dataframe['score'] = classifier.predict(normalized_X)
    # n_rows = len(features_dataframe[features_dataframe['score'] > 0.5])
    # max_score = features_dataframe['score'].max()
    return dict(zip(features_dataframe['columns'], features_dataframe['score']))


def train_and_rank(join_path, label_column):
    rank = {}
    # Read data
    for f in listdir(join_path):
        if isfile(join(join_path, f)):
            table = pd.read_csv(join(join_path, f), header=0, engine="python", encoding="utf8", quotechar='"',
                                escapechar='\\')
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


def train_baseline(base_table_path, target_column):
    table = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
    X, y = prepare_data_for_ml(table, target_column)

    return train_CART(X, y)
