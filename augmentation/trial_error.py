from typing import Dict, List

import pandas as pd
import tqdm

from algorithms import CART
from config import JOIN_RESULT_FOLDER
from data_preparation.utils import compute_partial_join_filename, join_and_save, prepare_data_for_ml
from experiments.result_object import Result
from feature_selection.util_functions import compute_correlation
from graph_processing.neo4j_transactions import get_relation_properties_node_name
from helpers.util_functions import get_df_with_prefix, get_elements_higher_than_value


def bfs_traverse_join_pipeline(queue: set, target_column: str, join_tree: Dict, train_results: List,
                               join_name_mapping: dict, previous_queue=None):
    """
    Recursive function - the pipeline to: 1) traverse the graph given a base node_id, 2) join with the adjacent nodes,
    3) apply feature selection algorithms, and 4) check the algorithm effectiveness by training CART decision tree model.

    :param queue: Queue with one node, which is the starting point of the traversal.
    :param target_column: Target column containing the class labels for training.
    :param join_tree: The result of the BFS traversal.
    :param train_results: List used to store the results of training CART.
    :param join_name_mapping: Mapping with the name of the join and the corresponding name of the file containing the join result.
    :param previous_queue: Initially empty or None, the queue is used to store the partial join names between the iterations.
    :return: None
    """

    if len(queue) == 0:
        return

    current_node_id = queue.pop()
    print(f"New iteration with {current_node_id}")

    adjacent_nodes = join_tree[current_node_id]
    if len(adjacent_nodes) == 0:
        return

    left_df, node_label = get_df_with_prefix(current_node_id, target_column)

    if previous_queue is None:
        previous_queue = {node_label}

    for node in adjacent_nodes:
        print(f"Adjacent node: {node}")
        join_keys = get_relation_properties_node_name(from_id=current_node_id, to_id=node)

        right_df, right_label = get_df_with_prefix(node)
        print(f"\tRight table shape: {right_df.shape}")

        current_queue = set()
        while len(previous_queue) > 0:
            partial_join_name = previous_queue.pop()
            print(f"\tPartial join name: {partial_join_name}")

            if partial_join_name == node_label:
                partial_join = left_df
            else:
                partial_join = pd.read_csv(JOIN_RESULT_FOLDER / join_name_mapping[partial_join_name], header=0,
                                           engine="python", encoding="utf8", quotechar='"', escapechar='\\')

            for prop in join_keys:
                join_prop, from_table, to_table = prop
                if join_prop['from_label'] != from_table:
                    continue
                print(f"\t\tJoin properties: {join_prop}")

                # Transform to 1:1 or M:1
                right_df = right_df.groupby(f"{right_label}.{join_prop['to_column']}").sample(n=1, random_state=42)

                # Compute the name of the join
                join_name = compute_partial_join_filename(prop=prop, partial_join_name=partial_join_name)

                # File naming convention as the filename can be gigantic
                join_filename = f"join{len(join_name_mapping) + 1}.csv"
                join_name_mapping[join_name] = join_filename
                print(f"\tJoin name: {join_name}")

                # Join
                joined_df = join_and_save(left_df=partial_join,
                                          right_df=right_df,
                                          left_column=f"{from_table}.{join_prop['from_column']}",
                                          right_column=f"{to_table}.{join_prop['to_column']}",
                                          join_name=join_filename)

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

                current_queue.add(join_name)

        previous_queue = current_queue
        queue.add(node)

    bfs_traverse_join_pipeline(queue, target_column, join_tree[current_node_id], train_results,
                               join_name_mapping, previous_queue)


def dfs_traverse_join_pipeline(base_node_id: str, target_column: str, join_tree: Dict, train_results: List,
                               join_name_mapping: dict, partial_join_name=None, partial_join=None):
    """
    Recursive function - the pipeline to traverse the graph give a base node_id, join with the new nodes during traversal,
    apply feature selection algorithm and check the algorithm effectiveness by training CART decision tree model.
    :param base_node_id: Starting point of the traversal.
    :param target_column: Target column containing the class labels for training.
    :param join_tree: The result of the DFS traversal.
    :param train_results: List used to store the results of training CART.
    :param join_name_mapping:
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
        return partial_join, partial_join_name

    # Traverse
    for node in tqdm.tqdm(join_tree[base_node_id].keys()):
        print(f"\n\t{base_node_id} Joining with {node}")
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

            # Compute the name of the join
            join_name = compute_partial_join_filename(prop=prop, partial_join_name=partial_join_name)

            # File naming convention as the filename can be gigantic
            join_filename = f"join{len(join_name_mapping) + 1}.csv"
            join_name_mapping[join_filename] = join_name
            print(f"\tJoin name: {join_name}")

            # Join
            joined_df = join_and_save(left_df=left_df,
                                      right_df=right_df,
                                      left_column=f"{from_table}.{join_prop['from_column']}",
                                      right_column=f"{to_table}.{join_prop['to_column']}",
                                      join_name=join_filename)

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
            joined_df, join_name = dfs_traverse_join_pipeline(node, target_column, join_tree[base_node_id],
                                                              train_results, join_name_mapping, join_name, joined_df)
            print(f"\tEnd join properties iteration for {node}")

        partial_join_name = join_name
        partial_join = joined_df
        print(f"End depth iteration for {node}")

    partial_join_name = join_name
    partial_join = joined_df

    return partial_join, partial_join_name


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
