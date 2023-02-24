import math
import time
from typing import List

import deprecation as deprecation
import numpy as np
import pandas as pd
import tqdm as tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_preparation.join_data import join_directly_connected
from data_preparation.utils import prepare_data_for_ml
from graph_processing.neo4j_transactions import get_relation_properties_node_name, get_adjacent_nodes, get_node_by_id


def gen_features(A: pd.DataFrame, eta: float):
    """
    Algorithm 2 from "ARDA: Automatic Relational Data Augmentation for Machine Learning"
    :param A: The (normalized) data matrix
    :param eta: The amount of features to generate
    :return: A matrix of generated random features, where each column represents one feature
    """

    L = []
    d = A.shape[1]
    m = np.mean(A, axis=1)
    s = np.cov(A)
    print(f"\t\tARDA: Generate: {math.ceil(eta * d)} features")
    for i in tqdm.tqdm(range(math.ceil(eta * d))):
        L.append(np.random.multivariate_normal(m, s))
    result = np.array(L).T
    print(f"\t\tARDA: Generated {result.shape}")
    return result


def _bin_count_ranking(feature_importance_scores: np.ndarray, mask: np.ndarray, bin_size: int) -> List:
    """
    Count how often the "real" features appear in front of the generated features
    :param feature_importance_scores: The rankings as determined by the ranking algorithm
    :param mask: The bit mask indicating which columns were randomly generated (True) and which ones are real features (False)
    :param bin_size: Size of the bin array, corresponds to the amount of columns in the data matrix (the amount of "real" features)
    :return:
    """

    # Get sorting indices for the rankings, flip order since we have feature importance scores
    indices = feature_importance_scores.argsort()[::-1]
    # Sort the mask, so we know where the generated columns are located in terms of ranking
    sorted_mask = mask[indices[::]]
    bins = np.zeros(bin_size)

    # Iterate through this mask until we hit a generated feature
    # Add 1 for all the original features that were in front
    for i, val in zip(indices, sorted_mask):
        if val:
            break
        else:
            bins[i] += 1

    return bins


def select_features(A: pd.DataFrame, y: pd.Series, tau=0.1, eta=0.2, k=10, regressor=RandomForestClassifier) -> List:
    """
    Algorithm 1 from "ARDA: Automatic Relational Data Augmentation for Machine Learning"
    :param A: The (normalized) data matrix
    :param y: The label/target column
    :param tau: Threshold for the fraction of how many times a feature appeared in front of synthesized features in ranking
    :param eta: Fraction of random features to inject (fraction of amount of features in A)
    :param k: Number of times ranking and counting is performed
    :param regressor: The sklearn regressor to use
    :return: A set of indices selected by thresholding the normalized frequencies by 'tau'
    """
    d = A.shape[1]
    print("\tARDA: Generate features")
    X = np.concatenate((A, gen_features(A, eta)), axis=1)  # This gives us A' from the paper

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[d:] = True  # We mark the columns that were generated
    counts = np.zeros(d)

    # Repeat process 'k' times, as in the algorithm
    print("\tARDA: Decide feature importance")
    for i in range(k):
        reg = regressor()
        reg.fit(X, y)
        counts += _bin_count_ranking(reg.feature_importances_, mask, d)
    return np.arange(d)[counts / k > tau]


def wrapper_algo(A: pd.DataFrame, y: pd.Series, T: List[float], eta=0.2, k=10, estimator=RandomForestClassifier,
                 regressor=RandomForestClassifier) -> List:
    """
    Algorithm 3 from "ARDA: Automatic Relational Data Augmentation for Machine Learning"
    :param A: The (normalized) data matrix
    :param y: The label/target column
    :param T: A list with thresholds (see tau in algo 2) to use
    :param eta: Fraction of random features to inject
    :param k: The number of times ranking and counting is performed
    :param estimator: An sklearn estimator to use (e.g. a classifier)
    :param regressor: An sklearn  regressor to use for ranking in algo 2
    :return: An array of indices, corresponding to selected features from A
    """

    if A.shape[0] != y.shape[0]:
        raise ValueError("Criterion/feature 'y' should have the same amount of rows as 'A'")

    last_accuracy = 0
    last_indices = []

    for t in sorted(T):
        X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2)
        print("\nARDA: Select features")
        indices = select_features(X_train, y_train, tau=t, eta=eta, k=k, regressor=regressor)

        # If this happens, the thresholds might have been too strict
        if len(indices) == 0:
            return last_indices

        if len(X_train.iloc[:, indices]) == 0:
            return last_indices

        print("ARDA: Train and score")
        model = estimator()
        model.fit(X_train.iloc[:, indices], y_train)
        accuracy = model.score(X_test.iloc[:, indices], y_test)
        if accuracy < last_accuracy:
            break
        else:
            last_accuracy = accuracy
            last_indices = indices
    return last_indices


@deprecation.deprecated(details="Use select_arda_features_budget_join instead")
def select_arda_features(base_table_id, target_column, base_table_features):
    print("ARDA - Join directly connected tables ... ")
    start = time.time()
    dataset_df = join_directly_connected(base_table_id)
    end = time.time()
    join_time = end - start

    print("ARDA - Prepare data for ML ... ")
    X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=target_column)
    print(X.shape)
    if X.shape[0] > 10000:
        _, X, _, y = train_test_split(X, y, test_size=10000, shuffle=True, stratify=y)
    print(X.shape)

    print("ARDA Feature selection - Started ... ")
    start = time.time()
    T = np.arange(0.0, 1.0, 0.1)
    indices = wrapper_algo(X, y, T)
    fs_X = X.iloc[:, indices].columns
    end = time.time()
    fs_time = end - start
    print("ARDA Feature selection - Ended ... ")

    columns_to_drop = [
        c for c in list(X.columns) if (c not in base_table_features) and (c not in fs_X)
    ]
    X.drop(columns=columns_to_drop, inplace=True)

    return X, y, join_time, fs_time, fs_X


def select_arda_features_budget_join(base_node_id: str, target_column: str, base_table_features: List,
                                     sample_size: int):
    random_state = 42
    final_selected_features = []
    all_columns = []
    join_time = 0
    fs_time = 0

    # Read base table, uniform sample, set budget size
    left_table = pd.read_csv(base_node_id, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
    left_table = left_table.sample(sample_size, random_state=random_state)
    budget_size = left_table.shape[0]

    # Get node, prepend the node label to columns and base table features for easy identification
    base_node = get_node_by_id(base_node_id)
    left_table = left_table.set_index([target_column]).add_prefix(f"{base_node.get('label')}.").reset_index()
    base_table_features = [f"{base_node.get('label')}.{feat}" for feat in base_table_features]

    # Get directly connected nodes
    nodes = get_adjacent_nodes(base_node_id)
    while len(nodes) > 0:
        feature_count = 0

        # Join every table according to the budget
        start = time.time()
        while feature_count <= budget_size and len(nodes) > 0:
            node_id = nodes.pop()
            print(f"Node id: {node_id}\n\tRemaining: {len(nodes)}")

            # Get the keys between the base node and connected node
            join_keys = get_relation_properties_node_name(from_id=base_node_id, to_id=node_id)[0:2]
            # Keep the key which matches the outgoing node
            join_prop, from_table, to_table = [k for k in join_keys if k[0]['from_label'] == k[1]][0]
            print(f"Join properties: {join_prop}")

            # Read right table, aggregate on the join key (reduce to 1:1 or M:1 join) by random sampling
            right_table = pd.read_csv(node_id, header=0, engine="python", encoding="utf8", quotechar='"',
                                      escapechar='\\')
            right_table = right_table.groupby(join_prop['to_column']).sample(n=1, random_state=random_state)

            # Prepend node label to every column for easy identification
            right_node = get_node_by_id(node_id)
            right_table = right_table.add_prefix(f"{right_node.get('label')}.")

            # Join tables, drop the right key as we don't need it anymore
            left_table = pd.merge(left_table, right_table, how="left",
                                  left_on=f"{from_table}.{join_prop['from_column']}",
                                  right_on=f"{to_table}.{join_prop['to_column']}")
            left_table.drop(columns=[f"{to_table}.{join_prop['to_column']}"], inplace=True)
            # Update feature count (subtract 1 for the deleted right key)
            feature_count += right_table.shape[1] - 1
            print(f"Feature count: {feature_count}")

        join_time += time.time() - start
        # Compute the columns of the batch and create the batch dataset
        columns = [c for c in list(left_table.columns) if
                   (c not in base_table_features) and (c not in all_columns)]
        print(f"{len(columns)} columns to select")

        joined_tables_batch = left_table[columns]
        print(f"shape: {joined_tables_batch.shape}")

        # Save the computed columns
        all_columns.extend(columns)
        all_columns.remove(target_column)

        # Prepare data
        X, y = prepare_data_for_ml(dataframe=joined_tables_batch, target_column=target_column)

        # Run ARDA - RIFS (Random Injection Feature Selection) algorithm
        start = time.time()
        T = np.arange(0.0, 1.0, 0.1)
        indices = wrapper_algo(X, y, T)
        fs_X = X.iloc[:, indices].columns
        fs_time += time.time() - start
        print(f"Selected columns: {fs_X}")

        # Save the selected columns of the batch
        final_selected_features.extend(fs_X)

    return left_table, left_table[target_column], final_selected_features, join_time, fs_time
