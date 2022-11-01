import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# algo 2
# A                     the (normalized) data matrix
# n                     the amount of features to generate
#
# Return: A matrix of generated random features, where each column represents one feature
from data_preparation.join_data import join_directly_connected
from data_preparation.utils import prepare_data_for_ml
from helpers.neo4j_utils import get_pk_fk_nodes
from helpers.util_functions import transform_node_to_dict


def gen_features(A, eta):
    L = []
    d = A.shape[1]
    m = np.mean(A, axis=1)
    s = np.cov(A)
    for i in range(round(eta * d)):
        L.append(np.random.multivariate_normal(m, s))
    return np.array(L).T


# Counts how often features
# rankings             the rankings as determined by the ranking algorithm
# mask                 the bit mask indicating which columns were randomly generated (True) and which ones are real features (False)
# bin_size             size of the bin array, corresponds to the amount of columns in the data matrix (the amount of "real" features)
def _bin_count_ranking(importances, mask, bin_size):
    indices = importances.argsort()[
              ::-1
              ]  # Then we obtain sorting indices for the rankings, flip order since we have importances
    sorted_mask = mask[
        indices[::]
    ]  # These indices are then used to sort the mask, so we know where the generated columns are located in terms of ranking
    bins = np.zeros(bin_size)

    # Then we iterate through this mask until we hit a generated/random feature, adding 1 for all the original features that were in front
    for i, val in zip(indices, sorted_mask):
        if val:
            break
        else:
            bins[i] += 1

    return bins


# algo 1
# A                     the (normalized) data matrix
# y                     the feature to use as criterion/dependent variable for the regressors
# tau                   threshold for how many times a feature appeared in front of synthesized features in ranking
# eta                   fraction of random features to inject (fraction of amount of features in A)
# k                     number of times ranking and counting is performed
# regressor             the regressor to use
#
# Returns: An array of indices, corresponding to selected features from A
def select_features(A, y, tau=0.1, eta=0.5, k=20, regressor=RandomForestClassifier):
    d = A.shape[1]
    X = np.concatenate((A, gen_features(A, eta)), axis=1)  # This gives us A' from the paper

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[d:] = True  # We mark the columns that were generated
    counts = np.zeros(d)

    # Repeat process 'k' times, as in the algorithm
    for i in range(k):
        reg = regressor()
        reg.fit(X, y)
        counts += _bin_count_ranking(reg.feature_importances_, mask, d)
    # Return a set of indices selected by thresholding the normalized frequencies with 'tau'
    return np.arange(d)[counts / k > tau]


# algo 3
# A                     the (normalized) data matrix
# y                     the feature to use as criterion/dependent variable for the regressors
# T                     A list with tresholds (see tau in algo 2) to use
# eta                   fraction of random features to inject (fraction of amount of features in A)
# k                     number of times ranking and counting is performed
# estimator             An sklearn estimator to use (e.g. a Regressor)
# regressor             The regressor to use for ranking in algo 2
#
# Returns: An array of indices, corresponding to selected features from A
def wrapper_algo(
        A, y, T, eta=0.2, k=10, estimator=RandomForestClassifier, regressor=RandomForestClassifier
):
    if A.shape[0] != y.shape[0]:
        raise ValueError("Criterion/feature 'y' should have the same amount of rows as 'A'")

    last_accuracy = 0
    last_indices = []
    for t in sorted(T):
        X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2)
        indices = select_features(X_train, y_train, tau=t, eta=eta, k=k, regressor=regressor)

        # If this happens, the thresholds might have been too strict
        if len(indices) == 0:
            return last_indices

        if len(X_train.iloc[:, indices]) == 0:
            return last_indices

        model = estimator()
        model.fit(X_train.iloc[:, indices], y_train)
        accuracy = model.score(X_test.iloc[:, indices], y_test)
        if accuracy < last_accuracy:
            break
        else:
            last_accuracy = accuracy
            last_indices = indices

    return last_indices


def select_arda_features(base_table_id, target_column, base_table_features):
    start = time.time()
    dataset_df = join_directly_connected(base_table_id)
    end = time.time()
    join_time = end - start

    X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=target_column)
    print(X.shape)
    if X.shape[0] > 10000:
        _, X, _, y = train_test_split(X, y, test_size=10000, shuffle=True, stratify=y)
    print(X.shape)

    start = time.time()
    T = np.arange(0.0, 1.0, 0.1)
    indices = wrapper_algo(X, y, T)
    fs_X = X.iloc[:, indices].columns
    end = time.time()
    fs_time = end - start

    columns_to_drop = [
        c for c in list(X.columns) if (c not in base_table_features) and (c not in fs_X)
    ]
    X.drop(columns=columns_to_drop, inplace=True)

    return X, y, join_time, fs_time, fs_X


def select_arda_features_budget_join(base_table_id, target_column, base_table_features, budget_size, sample_size):
    random_state = 42
    final_selected_features = []

    nodes = get_pk_fk_nodes(base_table_id)
    for pk, fk in nodes:
        pk_node = transform_node_to_dict(pk)
        fk_node = transform_node_to_dict(fk)

        # Read tables
        left_table = pd.read_csv(pk_node['source_path'])
        right_table = pd.read_csv(fk_node['source_path'])

        # Sample rows for each
        left_table = left_table.sample(sample_size, random_state=random_state)
        right_table = right_table.sample(sample_size, random_state=random_state)

        # Join the tables based on join column
        joined_tables = pd.merge(left_table, right_table, how="left", left_on=pk_node['name'],
                                right_on=fk_node['name'], suffixes=("", "_b"))

        columns_to_drop = [c for c in list(joined_tables.columns) if c.endswith("_b")]
        joined_tables.drop(columns=columns_to_drop, inplace=True)

        # Check if joined table feature size is less than budget size
        if joined_tables.shape[1] <= budget_size:
            X, y = prepare_data_for_ml(dataframe=joined_tables, target_column=target_column)

            T = np.arange(0.0, 1.0, 0.1)
            indices = wrapper_algo(X, y, T)
            fs_X = X.iloc[:, indices].columns

            final_selected_features = fs_X
        else:
            columns = list(joined_tables.columns)
            columns.remove(target_column)

            n = budget_size

            # Split columns into batches of budget size n
            final = [columns[i * n:(i + 1) * n] for i in range((len(columns) + n - 1) // n)]

            # Perform calculations for every batch
            for columns in final:
                columns.append(target_column)

                joined_tables_batch = joined_tables[columns]

                # Prepare data to use
                X, y = prepare_data_for_ml(dataframe=joined_tables_batch, target_column=target_column)

                T = np.arange(0.0, 1.0, 0.1)
                indices = wrapper_algo(X, y, T)
                fs_X = X.iloc[:, indices].columns

                final_selected_features.append(fs_X)

    return list(set([item for sublist in final_selected_features for item in sublist]))
