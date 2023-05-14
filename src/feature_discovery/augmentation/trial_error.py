import time
from typing import Dict

import numpy as np
import pandas as pd
import tqdm

from feature_discovery.algorithms import CART
from feature_discovery.config import JOIN_RESULT_FOLDER
from feature_discovery.data_preparation.utils import (
    compute_join_name,
    join_and_save,
    prepare_data_for_ml,
)
from feature_discovery.experiments.result_object import Result
from feature_discovery.feature_selection.gini import gini_index, feature_ranking
from feature_discovery.feature_selection.util_functions import compute_correlation
from feature_discovery.graph_processing.neo4j_transactions import (
    get_relation_properties_node_name,
)
from feature_discovery.helpers.util_functions import (
    get_df_with_prefix,
    get_elements_higher_than_value,
)


def dfs_traverse_join_pipeline(
    base_node_id: str,
    target_column: str,
    base_table_label: str,
    join_tree: Dict,
    join_name_mapping: dict,
    value_ratio: float,
    paths_score: Dict,
    gini: bool,
    smallest_gini_score=None,
    previous_paths=None,
) -> set:
    """
    Recursive function - the pipeline to traverse the graph give a base node_id, join with the new nodes during traversal,
    apply feature selection algorithm and check the algorithm effectiveness by training CART decision tree model.

    :param base_node_id: Starting point of the traversal.
    :param target_column: Target column containing the class labels for training.
    :param join_tree: The result of the DFS traversal.
    :param train_results: List used to store the results of training CART.
    :param join_name_mapping:
    :param value_ratio: Pruning threshold. It represents the ration between the number of non-null values in a column and the total number of values.
    :param previous_paths: The join paths used in previous iteration, which are used to create a join tree of paths.
    :return: Set of paths
    """
    print(f"New iteration with {base_node_id}")

    # Trace the recursion
    if len(join_tree[base_node_id].keys()) == 0:
        print(f"End node: {base_node_id}")
        return previous_paths

    # Get current dataframe
    left_df = None
    if previous_paths is None:
        left_df, left_label = get_df_with_prefix(base_node_id, target_column)
        previous_paths = {left_label}
        if gini:
            X, y = prepare_data_for_ml(left_df, target_column)
            smallest_gini_score = min(gini_index(X.to_numpy(), y))

    all_paths = previous_paths.copy()

    # Traverse
    for node in tqdm.tqdm(join_tree[base_node_id].keys()):
        print(f"\n\t{base_node_id} Joining with {node}")
        join_keys = get_relation_properties_node_name(from_id=base_node_id, to_id=node)

        right_df, right_label = get_df_with_prefix(node)
        print(f"\tRight table shape: {right_df.shape}")

        next_paths = set()
        for prop in tqdm.tqdm(join_keys):
            join_prop, from_table, to_table = prop
            if join_prop["from_label"] != from_table:
                continue
            print(f"\n\t\tJoin properties: {join_prop}")

            # Transform to 1:1 or M:1 based on the join property
            right_df = right_df.groupby(
                f"{right_label}.{join_prop['to_column']}"
            ).sample(n=1, random_state=42)

            current_paths = all_paths.copy()
            while len(current_paths) > 0:
                current_join_path = current_paths.pop()
                print(f"\t\t\tCurrent join path: {current_join_path}")

                if left_df is None:
                    left_df = pd.read_csv(
                        JOIN_RESULT_FOLDER
                        / base_table_label
                        / join_name_mapping[current_join_path],
                        header=0,
                        engine="python",
                        encoding="utf8",
                        quotechar='"',
                        escapechar="\\",
                    )

                # Compute the name of the join
                join_name = compute_join_name(
                    join_key_property=prop, partial_join_name=current_join_path
                )
                print(f"\t\t\tJoin name: {join_name}")

                # File naming convention as the filename can be gigantic
                join_filename = (
                    f"join_DFS_{value_ratio}_{len(join_name_mapping) + 1}.csv"
                )

                # Join
                joined_df = join_and_save(
                    left_df,
                    right_df,
                    left_column_name=f"{from_table}.{join_prop['from_column']}",
                    right_column_name=f"{to_table}.{join_prop['to_column']}",
                    label=base_table_label,
                    join_name=join_filename,
                )

                if (
                    joined_df[f"{to_table}.{join_prop['to_column']}"].count()
                    / joined_df.shape[0]
                    < value_ratio
                ):
                    print("\t\tRight column value ration below 0.5.\nSKIPPED Join")
                    continue

                if gini:
                    print("\tGini index computation ... ")
                    X, y = prepare_data_for_ml(joined_df, target_column)
                    scores = gini_index(X.to_numpy(), y)
                    indices = feature_ranking(scores)

                    if not np.any(scores < smallest_gini_score):
                        print(
                            f"\t\tNo feature with gini index smallest than {smallest_gini_score}.\nSKIPPED Join"
                        )
                        continue

                    paths_score[join_name] = scores[indices[0]] + scores[indices[1]]

                next_paths.add(join_name)
                join_name_mapping[join_name] = join_filename

            print(f"\tEnd join properties iteration for {node}")

        # Continue traversal
        current_paths = dfs_traverse_join_pipeline(
            node,
            target_column,
            base_table_label,
            join_tree[base_node_id],
            join_name_mapping,
            value_ratio,
            paths_score,
            gini,
            smallest_gini_score,
            next_paths,
        )
        all_paths.update(current_paths)
        print(f"End depth iteration for {node}")

    return all_paths


def _select_features_train(joined_df, right_df, target_column, join_name):
    results = []
    # Select features
    left_table_features = [
        feat
        for feat in list(joined_df.columns)
        if feat not in list(right_df.columns) and (feat != target_column)
    ]
    correlated_features = compute_correlation(
        left_table_features=left_table_features,
        joined_df=joined_df,
        target_column=target_column,
    )
    # TODO: adjust value
    selected_features = get_elements_higher_than_value(
        dictionary=correlated_features, value=0.4
    )
    selected_features_df = None
    if len(selected_features) > 0:
        features_with_selected = left_table_features
        features_with_selected.extend(selected_features.keys())
        features_with_selected.append(target_column)
        selected_features_df = joined_df[features_with_selected]

    # Train, test - With feature selection
    print(f"TRAIN WITH feature selection")
    if selected_features_df is None:
        print(f"\tNo selected features. Skipping ...")
    elif selected_features_df.shape == joined_df.shape:
        print(f"\tAll features were selected. Skipping ... ")
    else:
        result = train_test_cart(selected_features_df, target_column)
        result.data_path = join_name
        result.approach = Result.TFD
        result.data_label = join_name
        results.append(result)

    return results


def train_test_cart(
    train_data: pd.DataFrame,
    target_column: str,
    prepare_data: bool = True,
    regression: bool = False,
) -> Result:
    """
    Train CART decision tree on the dataframe and save the result.

    :param train_data: DataFrame for training (X)
    :param target: Series representing the label/target data (y)
    :param target_column: Target/label column with the class labels
    :param prepare_data: False - if train_data and target have been processed, True - if they need to be processed.
    :param regression: Bool - if True: a regressor is employed, if False: a classifier is employed
    :return: A Result object with the configuration and results of training
    """

    start = time.time()

    if prepare_data:
        X, y = prepare_data_for_ml(train_data, target_column)
    else:
        X = train_data
        y = train_data[[target_column]]

    acc_decision_tree, _, feature_importance, _ = CART().train(
        train_data=X, target_data=y, regression=regression
    )
    features_scores = dict(zip(X.columns, feature_importance))
    end = time.time()
    train_time = end - start
    print(
        f"\tAccuracy/RMSE: {abs(acc_decision_tree)}\n\tFeature scores: \n{features_scores}\n\tTrain time: {train_time}"
    )

    entry = Result(
        algorithm=CART.LABEL,
        accuracy=abs(acc_decision_tree),
        feature_importance=features_scores,
        train_time=train_time,
    )
    return entry


def run_auto_gluon(
    approach: str,
    dataframe: pd.DataFrame,
    target_column: str,
    data_label: str,
    join_name: str,
    algorithms_to_run: dict,
    value_ratio: float = None,
):
    from sklearn.model_selection import train_test_split
    from autogluon.tabular import TabularPredictor

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=[target_column]),
        dataframe[[target_column]],
        test_size=0.2,
        random_state=10,
    )
    train = X_train.copy()
    train[target_column] = y_train

    test = X_test.copy()
    test[target_column] = y_test

    predictor = TabularPredictor(
        label=target_column, problem_type="binary", verbosity=2
    ).fit(train_data=train, hyperparameters=algorithms_to_run)
    highest_acc = 0
    best_model = None
    results = []
    training_results = predictor.info()
    for model in training_results["model_info"].keys():
        accuracy = training_results["model_info"][model]["val_score"]
        ft_imp = predictor.feature_importance(
            data=test, model=model, feature_stage="original"
        )
        entry = Result(
            algorithm=model,
            accuracy=accuracy,
            feature_importance=dict(zip(list(ft_imp.index), ft_imp["importance"])),
            train_time=training_results["model_info"][model]["fit_time"],
            approach=approach,
            data_label=data_label,
            data_path=join_name,
            join_path_features=list(X_train.columns),
        )
        if value_ratio:
            entry.cutoff_threshold = value_ratio

        if accuracy > highest_acc:
            highest_acc = accuracy
            best_model = entry

        results.append(entry)

    return best_model, results
