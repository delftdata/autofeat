from typing import Dict, List

import pandas as pd
import tqdm

from algorithms import CART
from data_preparation.utils import compute_partial_join_filename, join_and_save, prepare_data_for_ml
from experiments.result_object import Result
from feature_selection.util_functions import compute_correlation
from graph_processing.neo4j_transactions import get_relation_properties_node_name
from helpers.util_functions import get_df_with_prefix, get_elements_higher_than_value


def traverse_join_pipeline(base_node_id: str, target_column: str, join_tree: Dict, train_results: List,
                           partial_join_name=None, partial_join=None):
    """
    Recursive function - the pipeline to traverse the graph give a base node_id, join with the new nodes during traversal,
    apply feature selection algorithm and check the algorithm effectiveness by training CART decision tree model.
    :param base_node_id: Starting point of the traversal.
    :param target_column: Target column containing the class labels for training.
    :param join_tree: The result of the DFS traversal.
    :param train_results: List used to store the results of training CART.
    :param partial_join_name: The name of the partial join use to compute the name for the next iterations.
    :param partial_join: The partial join used for the next iterations.
    :return: None
    """
    print(f"New iteration with {base_node_id}")

    # Get current dataframe
    if partial_join_name is None or partial_join is None:
        left_df, _ = get_df_with_prefix(base_node_id, target_column)
    else:
        left_df = partial_join

    # Trace the recursion
    if len(join_tree[base_node_id].keys()) == 0:
        print(f"End node: {base_node_id}")

    # Traverse
    for node in tqdm.tqdm(join_tree[base_node_id].keys()):
        print(f"\n\tJoining with {node}")
        join_keys = get_relation_properties_node_name(from_id=base_node_id, to_id=node)

        right_df, right_label = get_df_with_prefix(node)
        print(f"\tRight table shape: {right_df.shape}")

        for prop in tqdm.tqdm(join_keys):
            join_prop, from_table, to_table = prop
            if join_prop['from_label'] != from_table:
                continue
            print(f"\n\tJoin properties: {join_prop}")

            # Transform to 1:1 or M:1
            right_df = right_df.groupby(f"{right_label}.{join_prop['to_column']}").sample(n=1, random_state=42)

            # Join
            join_name = compute_partial_join_filename(prop=prop, partial_join_name=partial_join_name)
            print(f"\tJoin name: {join_name}")
            joined_df = join_and_save(left_df=left_df,
                                      right_df=right_df,
                                      left_column=f"{from_table}.{join_prop['from_column']}",
                                      right_column=f"{to_table}.{join_prop['to_column']}",
                                      join_name=join_name)

            # Select features
            left_table_features = [feat for feat in list(joined_df.columns) if
                                   feat not in list(right_df.columns) and (feat != target_column)]
            correlated_features = compute_correlation(left_table_features=left_table_features,
                                                      joined_df=joined_df,
                                                      target_column=target_column)
            # TODO: adjust value
            selected_features = get_elements_higher_than_value(dictionary=correlated_features, value=0.4)
            selected_features_df = None
            if len(selected_features) > 0:
                features_with_selected = left_table_features
                features_with_selected.extend(selected_features.keys())
                features_with_selected.append(target_column)
                selected_features_df = joined_df[features_with_selected]

            # Train, test - Without feature selection
            print(f"TRAIN WITHOUT feature selection")
            result = _train_test_cart(joined_df, target_column, join_name, Result.JOIN_ALL)
            train_results.append(result)

            # Train, test - With feature selection
            print(f"TRAIN WITH feature selection")
            if selected_features_df is None:
                print(f"\tNo selected features. Skipping ...")
            elif selected_features_df.shape == joined_df.shape:
                print(f"\tAll features were selected. Skipping ... ")
            else:
                result = _train_test_cart(selected_features_df, target_column, join_name, Result.TFD)
                train_results.append(result)

            # Continue traversal
            traverse_join_pipeline(node, target_column, join_tree[base_node_id], train_results, join_name, joined_df)


def _train_test_cart(dataframe: pd.DataFrame, target_column: str, join_name: str, approach: str) -> Result:
    """
    Train CART decision tree on the dataframe and save the result.
    :param dataframe: DataFrame for training
    :param target_column: Target/label column with the class labels
    :param join_name: The name of the join (for saving purposes)
    :param approach: The approach used to get the dataframe (string value under the Result class)
    :return: A Result object with the configuration and results of training
    """
    X, y = prepare_data_for_ml(dataframe, target_column)
    acc_decision_tree, params, feature_importance, train_time, _ = CART().train(X, y)
    features_scores = dict(zip(feature_importance, X.columns))
    print(f"\tAccuracy: {acc_decision_tree}\n\tFeature scores: \n{features_scores}\n\tTrain time: {train_time}")

    entry = Result(
        approach=approach,
        data_path=join_name,
        algorithm=CART.LABEL,
        depth=params['max_depth'],
        accuracy=acc_decision_tree,
        feature_importance=features_scores,
        train_time=train_time
    )
    return entry
