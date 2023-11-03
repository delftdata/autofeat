import logging
import math
from typing import List

import numpy as np
import pandas as pd
import tqdm as tqdm
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from feature_discovery.autofeat_pipeline.join_path_utils import compute_join_name
from feature_discovery.config import DATA_FOLDER
from feature_discovery.graph_processing.neo4j_transactions import (
    get_relation_properties_node_name,
    get_adjacent_nodes,
    get_node_by_id,
)

logging.getLogger().setLevel(logging.WARNING)


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
    logging.debug(f"\t\tARDA: Generate: {math.ceil(eta * d)} features")
    for i in tqdm.tqdm(range(math.ceil(eta * d))):
        L.append(np.random.multivariate_normal(m, s))
    result = np.array(L).T
    logging.debug(f"\t\tARDA: Generated {result.shape}")
    return result


def _bin_count_ranking(
        feature_importance_scores: np.ndarray, mask: np.ndarray, bin_size: int
) -> List:
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


def select_features(
        normalised_matrix: pd.DataFrame,
        y: pd.Series,
        tau=0.1,
        eta=0.2,
        k=10,
        regression: bool = False,
) -> List:
    """
    Algorithm 1 from "ARDA: Automatic Relational Data Augmentation for Machine Learning"

    :param normalised_matrix: The (normalized) data matrix
    :param y: The label/target column
    :param tau: Threshold for the fraction of how many times a feature appeared in front of synthesized features in ranking
    :param eta: Fraction of random features to inject (fraction of amount of features in A)
    :param k: Number of times ranking and counting is performed
    :param regression: bool - if True: Random Forest Regressor is used, if False: Random Forest Classifier is used
    :return: A set of indices selected by thresholding the normalized frequencies by 'tau'
    """

    if regression:
        estimator = RandomForestRegressor()
    else:
        estimator = RandomForestClassifier()

    d = normalised_matrix.shape[1]
    logging.debug("\tARDA: Generate features")
    X = np.concatenate(
        (normalised_matrix, gen_features(normalised_matrix, eta)), axis=1
    )  # This gives us A' from the paper

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[d:] = True  # We mark the columns that were generated
    counts = np.zeros(d)

    # Repeat process 'k' times, as in the algorithm
    logging.debug("\tARDA: Decide feature importance")
    for i in range(k):
        estimator.fit(X, y)
        counts += _bin_count_ranking(estimator.feature_importances_, mask, d)
    return np.arange(d)[counts / k > tau]


def wrapper_algo(
        normalised_matrix: pd.DataFrame,
        y: pd.Series,
        T: List[float],
        eta=0.2,
        k=10,
        regression: bool = False,
) -> List:
    """
    Algorithm 3 from "ARDA: Automatic Relational Data Augmentation for Machine Learning"

    :param normalised_matrix: The (normalized) data matrix
    :param y: The label/target column
    :param T: A list with thresholds (see tau in algo 2) to use
    :param eta: Fraction of random features to inject
    :param k: The number of times ranking and counting is performed
    :param regression: bool - if True: Random Forest Regressor is used, if False: Random Forest Classifier is used
    :return: An array of indices, corresponding to selected features from A
    """

    if normalised_matrix.shape[0] != y.shape[0]:
        raise ValueError(
            "Criterion/feature 'y' should have the same amount of rows as 'A'"
        )

    if regression:
        estimator = RandomForestRegressor()
    else:
        estimator = RandomForestClassifier()

    last_accuracy = 0
    last_indices = []

    for t in sorted(T):
        X_train, X_test, y_train, y_test = train_test_split(
            normalised_matrix, y, test_size=0.2
        )
        logging.debug("\nARDA: Select features")
        indices = select_features(
            X_train, y_train, tau=t, eta=eta, k=k, regression=regression
        )

        # If this happens, the thresholds might have been too strict
        if len(indices) == 0:
            return last_indices

        if len(X_train.iloc[:, indices]) == 0:
            return last_indices

        logging.debug("ARDA: Train and score")
        estimator.fit(X_train.iloc[:, indices], y_train)
        accuracy = estimator.score(X_test.iloc[:, indices], y_test)
        if accuracy < last_accuracy:
            break
        else:
            last_accuracy = accuracy
            last_indices = indices
    return last_indices


def select_arda_features_budget_join(
        base_node_id: str, target_column: str, sample_size: int, regression: bool = False
):
    random_state = 42
    final_selected_features = []
    all_columns = []

    # Read base table, uniform sample, set budget size
    left_table = pd.read_csv(
        str(DATA_FOLDER / base_node_id),
        header=0,
        engine="python",
        encoding="utf8",
        quotechar='"',
        escapechar="\\",
    )
    if sample_size and sample_size < left_table.shape[0]:
        left_table = left_table.sample(sample_size, random_state=random_state)
    budget_size = left_table.shape[0]

    # Get node, prepend the node label to columns and base table features for easy identification
    base_node = get_node_by_id(base_node_id)
    left_table = (
        left_table.set_index([target_column])
        .add_prefix(f"{base_node.get('id')}.")
        .reset_index()
    )
    base_table_columns = list(left_table.columns)
    base_table_columns.remove(target_column)

    join_name = base_node.get("id")

    join_keys = []
    # Get directly connected nodes
    nodes = get_adjacent_nodes(base_node_id)
    while len(nodes) > 0:
        feature_count = 0

        # Join every table according to the budget
        while feature_count <= budget_size and len(nodes) > 0:
            node_id = nodes.pop()
            logging.debug(f"Node id: {node_id}\n\tRemaining nodes: {len(nodes)}")

            # Get the keys between the base node and connected node
            join_key = get_relation_properties_node_name(
                from_id=base_node_id, to_id=node_id
            )[0]
            join_prop, from_table, to_table = join_key
            logging.debug(f"Join properties: {join_prop}")

            if join_prop["from_label"] == base_node.get("id"):
                if join_prop["from_column"] == target_column:
                    continue

            if join_prop["to_label"] == base_node.get("id"):
                if join_prop["to_column"] == target_column:
                    continue

            if join_prop["from_label"] == to_table:
                from_column = join_prop["to_column"]
                to_column = join_prop["from_column"]
            else:
                from_column = join_prop["from_column"]
                to_column = join_prop["to_column"]

            # Read right table, aggregate on the join key (reduce to 1:1 or M:1 join) by random sampling
            right_table = pd.read_csv(
                str(DATA_FOLDER / node_id),
                header=0,
                engine="python",
                encoding="utf8",
                quotechar='"',
                escapechar="\\",
            )
            right_table = right_table.groupby(to_column).sample(
                n=1, random_state=random_state
            )

            # Prepend node label to every column for easy identification
            right_node = get_node_by_id(node_id)
            right_table = right_table.add_prefix(f"{right_node.get('id')}.")

            # Join tables, drop the right key as we don't need it anymore
            if (
                    left_table[f"{from_table}.{from_column}"].dtype
                    != right_table[f"{to_table}.{to_column}"].dtype
            ):
                continue

            left_on = f"{from_table}.{from_column}"
            right_on = f"{to_table}.{to_column}"
            left_table = pd.merge(
                left_table,
                right_table,
                how="left",
                left_on=left_on,
                right_on=right_on,
            )
            join_keys.append(left_on)
            join_keys.append(right_on)

            # Compute the join name
            join_name = compute_join_name(
                join_key_property=join_key, partial_join_name=join_name
            )
            logging.debug(f"\t\t\tJoin name: {join_name}")

            # Update feature count (subtract 1 for the deleted right key)
            feature_count += right_table.shape[1] - 1
            logging.debug(f"Feature count: {feature_count}")

        # Compute the columns of the batch and create the batch dataset
        columns = set(left_table.columns) - set(all_columns) - set(base_table_columns)
        columns = list(set(columns) - set(join_keys))

        logging.debug(f"{len(columns)} columns to select")

        # If the algorithm failed
        if len(columns) == 0:
            logging.debug("No selected column")
            continue

        # If the algorithm doesn't find any new feature
        if len(columns) == 1 and target_column in columns:
            logging.debug("No selected column")
            continue

        # If the algorithm only selects one feature
        if len(columns) == 2 and target_column in columns:
            columns.remove(target_column)
            final_selected_features.extend(columns)
            logging.debug(f"Selected columns: {columns}")
            continue

        joined_tables_batch = left_table[columns]
        logging.debug(f"shape: {joined_tables_batch.shape}")

        # Save the computed columns
        all_columns.extend(columns)
        all_columns.remove(target_column)

        # Prepare data
        X = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False
        ).fit_transform(X=joined_tables_batch)

        # Imputation of nan - with mean if not categorical, else with most common value
        # df = df.apply(
        #     lambda x: x.fillna(x.mean()) if x.name not in df.select_dtypes(include='category').columns else x.fillna(
        #         x.value_counts().index[0]))

        y = joined_tables_batch[target_column]
        X = X.drop(columns=[target_column])

        # Run ARDA - RIFS (Random Injection Feature Selection) algorithm
        T = np.arange(0.0, 1.0, 0.1)
        indices = wrapper_algo(X, y, T, regression=regression)
        fs_X = X.iloc[:, indices].columns
        logging.debug(f"Selected columns: {fs_X}")

        # Save the selected columns of the batch
        final_selected_features.extend(fs_X)

    return left_table, base_table_columns, final_selected_features, join_name
