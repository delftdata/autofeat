import math
import statistics

from augmentation.pipeline import apply_feat_sel, classify_and_rank
from data_preparation.join_data import join_and_save
from feature_selection.util_functions import select_dependent_right_features, select_uncorrelated_with_selected
from utils.util_functions import get_elements_less_than_value


def ranking_func(all_paths: dict, mapping, current_table, target_column, path, allp, join_result_folder_path,
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
        # Get the features from base table excluding target
        base_table_features = list(left_table_df.drop(columns=[target_column]).columns)

        # 2. Select dependent features
        dependent_features_scores = select_dependent_right_features(base_table_features, joined_df, target_column)

        # 3. Compute the scores of the new features
        foreign_features_scores = select_uncorrelated_with_selected(base_table_features, joined_df, target_column)

        # 4. Select only the top uncorrelated features which are in the dependent features list
        score, selected_features = _compute_ranking_score(dependent_features_scores, foreign_features_scores)

        # 5. Save the features of this path
        ranking[path] = (score, selected_features)

        # 6. Save the join for future reference/usage
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
            join_path = ranking_func(all_paths, mapping, table, target_column, path, allp, join_result_folder_path,
                                                ranking, joined_mapping)
    return path


def _compute_ranking_score(dependent_features_scores: dict, foreign_features_scores: dict, threshold=0.2) -> tuple:
    feat_scores = {}
    # Compute a score for each feature based on the dependent score and the foreign feature score
    for feat in foreign_features_scores.keys():
        if feat in dependent_features_scores:
            feat_scores[feat] = foreign_features_scores[feat] * dependent_features_scores[feat]

    # Sort the features ascending based on the score
    feat_scores = dict(sorted(feat_scores.items(), key=lambda item: item[1]))
    # TODO: Improve the selection
    selected_features = get_elements_less_than_value(feat_scores, threshold)

    # Assign a score to each join path for ranking
    score = statistics.mean(selected_features.values()) if selected_features else math.inf

    return score, list(selected_features.keys())
