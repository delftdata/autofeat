import json
import math
import os
import statistics

import pandas as pd

from data_preparation.join_data import prune_or_join
from feature_selection.util_functions import compute_correlation, compute_relevance_redundancy
from utils.file_naming_convention import MAPPING_FOLDER, MAPPING
from utils.util_functions import normalize_dict_values, get_elements_higher_than_value

folder_name = os.path.abspath(os.path.dirname(__file__))


def ranking_multigraph(all_paths, mapping, base_table_root_path, target_column, join_result_folder_path):
    base_table_features = list(
        pd.read_csv(f"{folder_name}/../{base_table_root_path}", header=0, engine="python", encoding="utf8",
                    quotechar='"',
                    escapechar='\\', nrows=1).drop(columns=[target_column]).columns)
    base_table_name = '/'.join(base_table_root_path.split('/')[-2:])
    joined_mapping = {}
    for feature in base_table_features:
        feature_name = f"{base_table_name}/{feature}"
        if feature_name in all_paths:
            ranking_func(all_paths, mapping, feature_name, target_column, "", [], join_result_folder_path,
                         joined_mapping, base_table_features)
    return joined_mapping


def ranking_func(all_paths: dict, mapping, current_table, target_column, path, visited, join_result_folder_path,
                 joined_mapping: dict, base_table_features):
    # Join and save the join result
    print(f"New iteration: {current_table}\n\t{visited}")
    table_name = '/'.join(current_table.split('/')[0:-1])
    if not path == "":
        # get the name of the table from the path we created
        left_table = path.split("--")[-1]
        # get the location of the table we want to join with
        partial_join_path = mapping[left_table]
        selected_features = []

        # If we already joined the tables on the path, we retrieve the join result
        if path in joined_mapping:
            partial_join_path, selected_features, _ = joined_mapping[path]

        # Add the current table to the path
        path = f"{path}--{current_table}"

        # Recursion logic
        # 1. Join existing left table with the current table visiting
        result = prune_or_join(partial_join_path, left_table, current_table, mapping, join_result_folder_path)
        if result is None:
            return path

        visited.append(table_name)
        joined_path, joined_df, left_table_df = result

        # Get the features from base table excluding target
        left_table_features = list(left_table_df.drop(columns=[target_column]).columns)
        join_table_features = list(joined_df.drop(columns=[target_column]).columns)
        features_right = [feat for feat in join_table_features if feat not in left_table_features]

        # 2. Select dependent features
        dependent_features_scores = compute_correlation(left_table_features, joined_df, target_column)

        # 3. Compute the scores of the new features
        # Case 1: Foreign features vs selected features
        foreign_features_scores_1 = compute_relevance_redundancy(selected_features + base_table_features,
                                                                 features_right,
                                                                 joined_df, target_column)

        # 4. Select only the top uncorrelated features which are in the dependent features list
        score, features = _compute_ranking_score(dependent_features_scores, foreign_features_scores_1)
        print(f"Path score: {score} and selected features\n\t{features}")
        # TODO: Future work
        # # Case 2: Foreign features vs base table features only
        # foreign_features_scores_2 = select_uncorrelated_with_selected(base_table_features, features_right,
        #                                                               joined_df, target_column)
        #
        # # 4. Select only the top uncorrelated features which are in the dependent features list
        # score_2, features_2 = _compute_ranking_score(dependent_features_scores, foreign_features_scores_2)

        # 5. Save the join for future reference/usage
        # features, score = (features_1, score_1) if score_1 < score_2 else (features_2, score_2)
        joined_mapping[path] = (joined_path, selected_features + features, score)
    else:
        # Just started traversing, the path is the current table
        path = current_table
        visited.append(table_name)
        # base_table_features = list(
        #     pd.read_csv(f"{folder_name}/../{mapping[current_table]}", header=0, engine="python", encoding="utf8",
        #                 quotechar='"',
        #                 escapechar='\\', nrows=1).drop(columns=[target_column]).columns)

    print(path)

    # Depth First Search recursively
    for table in all_paths[current_table]:
        # Break the cycles in the data, only visit new nodes
        table_name = '/'.join(table.split('/')[0:-1])
        if table not in path and table_name not in visited:
            join_path = ranking_func(all_paths, mapping, table, target_column, path, visited, join_result_folder_path,
                                     joined_mapping, base_table_features)
            print(f"JOIN PATH IN FOR: {join_path}")
    return path


def _compute_ranking_score(dependent_features_scores: dict, foreign_features_scores: dict, threshold=0.2) -> tuple:
    # Create dataframe and normalise the data
    normalised_dfs = normalize_dict_values(dependent_features_scores)
    normalised_ffs = normalize_dict_values(foreign_features_scores)

    # Compute a score for each feature based on the dependent score and the foreign feature score
    feat_scores = {feat: normalised_dfs[feat] + normalised_ffs[feat] for feat in foreign_features_scores.keys() if
                   feat in dependent_features_scores}

    normalised_fs = normalize_dict_values(feat_scores)
    # Sort the features ascending based on the score
    normalised_fs = dict(sorted(normalised_fs.items(), key=lambda item: item[1], reverse=True))
    print(f"Normalised score on features:\n\t{normalised_fs}")
    selected_features = get_elements_higher_than_value(normalised_fs, threshold)
    ns = normalize_dict_values(selected_features)
    # Assign a score to each join path for ranking
    score = statistics.mean(ns.values()) if ns else -math.inf

    return score, list(selected_features.keys())
