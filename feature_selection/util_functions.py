import numpy as np
import pandas as pd
from ITMO_FS.utils.information_theory import matrix_mutual_information

from data_preparation.utils import prepare_data_for_ml
from experiments.understand_cife import measure_conditional_dependency, measure_relevance
from feature_selection.feature_selection_algorithms import FSAlgorithms
from utils.util_functions import get_elements_higher_than_value


def apply_feat_sel(joined_df, base_table_df, target_column, path):
    # Get the features from base table
    base_table_features = base_table_df.drop(columns=[target_column]).columns

    # Remove the features from the base table
    joined_table_no_base = joined_df.drop(
        columns=set(joined_df.columns).intersection(set(base_table_features)))
    # Transform data (factorize)
    # df = joined_table_no_base.apply(lambda x: pd.factorize(x)[0])
    #
    # X = np.array(df.drop(columns=[target_column]))
    # y = np.array(df[target_column])

    fs = FSAlgorithms()
    # create dataset to feed to the classifier
    result = {'columns': [], FSAlgorithms.T_SCORE: [], FSAlgorithms.CHI_SQ: [], FSAlgorithms.FISHER: [],
              FSAlgorithms.SU: [], FSAlgorithms.MIFS: [], FSAlgorithms.CIFE: []}

    # For each feature selection algorithm, get the score
    X, y = prepare_data_for_ml(joined_table_no_base, target_column)
    df_features = list(map(lambda x: f"{path}/{x}", X.columns))
    result['columns'] = df_features
    for alg in fs.ALGORITHMS:
        result[alg] = fs.feature_selection(alg, X, y)

    X, y = prepare_data_for_ml(joined_df, target_column)
    X_features = dict(zip(list(range(0, len(X.columns))), X.columns))
    left_table_features = dict(filter(lambda x: x[1] in base_table_features, X_features.items()))
    right_table_features = dict(
        filter(lambda x: x[1] in joined_table_no_base.columns and x[1] not in base_table_features,
               X_features.items()))
    for alg in fs.ALGORITHM_FOREIGN_TABLE:
        print(f"Processing: {alg}")
        result[alg] = fs.feature_selection_foreign_table(alg, list(left_table_features.keys()),
                                                         list(right_table_features.keys()), X, y)

    dataframe = pd.DataFrame.from_dict(result)
    return dataframe


def select_dependent_features(X, y) -> dict:
    fs = FSAlgorithms()
    # Use Symmetrical uncertainty to compute dependency
    scores = fs.feature_selection(fs.SU, X, y)
    # Map the scores to the columns
    result = dict(zip(X.columns, scores))
    # Sort the scores in descending order
    sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    # print(f"Sorted result:\n\t{sorted_result}")
    # Select the columns with SU > 0.4 - empirically showed that features with SU > 0.4 are selected by decision trees
    # selected_columns = dict(filter(lambda x: x[1] > 0.4, sorted_result.items()))
    return sorted_result


def compute_correlation(left_table_features: list, joined_df: pd.DataFrame, target_column: str) -> dict:
    print(f"Selecting dependent features on column: {target_column}...")
    # Remove the features from the base table
    joined_table_no_base = joined_df.drop(columns=left_table_features)

    # Fill NaN with 0, normalise data and split into X, y
    X, y = prepare_data_for_ml(joined_table_no_base, target_column)

    # Get the score of dependency: foreign features on target column
    dependent_features = select_dependent_features(X, y)
    print(f"Dependent features:\n\t{dependent_features}")
    return dependent_features


def compute_relevance_redundancy(left_table_features, features_to_compare, joined_df, target_column,
                                 redundancy_threshold=10):
    print(f"Selecting un-correlated features...")
    all_columns = list(joined_df.columns)
    all_columns.remove(target_column)
    to_drop_features = [feat for feat in all_columns if
                        feat not in left_table_features and feat not in features_to_compare]
    df = joined_df.drop(columns=to_drop_features)

    X, y = prepare_data_for_ml(df, target_column)

    all_features = dict(zip(list(range(0, len(X.columns))), X.columns))
    left_features = dict(filter(lambda x: x[1] in left_table_features, all_features.items()))
    right_features = dict(filter(lambda x: x[1] in features_to_compare, all_features.items()))

    fs = FSAlgorithms()
    scores = fs.feature_selection_foreign_table(fs.CIFE, list(left_features.keys()), list(right_features.keys()), X, y)
    # Map the scores to the columns
    result = dict(zip(list(right_features.values()), scores))

    # Broke down the metrics from CIFE to understand how the measure works
    # metrics = _understand_metrics(left_features, right_features, X, y)

    redundancy_scores = measure_redundancy(list(left_features.keys()), list(right_features.keys()), X)
    redundant_features = dict(zip(list(right_features.values()), redundancy_scores))
    selected_redundant = get_elements_higher_than_value(redundant_features, redundancy_threshold).keys()

    # Sort the scores in ascending order
    selected_non_redundant = {k: v for k, v in result.items() if k not in selected_redundant}
    # sorted_result = dict(sorted(selected_non_redundant.items(), key=lambda item: abs(item[1])))
    print(f"CIFE score result:\n\t{selected_non_redundant}")

    return selected_non_redundant


def _understand_metrics(left_features, right_features, X, y):
    ###
    redn = measure_redundancy(list(left_features.keys()), list(right_features.keys()), X)
    feat_red = dict(zip(list(right_features.values()), redn))
    cnd_dep = measure_conditional_dependency(list(left_features.keys()), list(right_features.keys()), X, y)
    feat_cnd_dep = dict(zip(list(right_features.values()), cnd_dep))
    relv = measure_relevance(list(right_features.keys()), X, y)
    feat_relv = dict(zip(list(right_features.values()), relv))
    metrics = {
        'redundancy': feat_red,
        'cond-dependency': feat_cnd_dep,
        'relevance': feat_relv
    }
    ###
    return metrics


def measure_redundancy(selected_features, free_features, X):
    """

    :param selected_features: list of indices of the selected features
    :param free_features: list of indices of the features to compare
    :param X: dataframe - training data
    :return: list of indices and scores
    """

    selected = np.array(selected_features)
    free = np.array(free_features)
    arr_X = np.array(X)
    redundancy = np.vectorize(
        lambda free_feature: np.sum(matrix_mutual_information(arr_X[:, selected], arr_X[:, free_feature])))(
        free)
    print(f"Redundancy: {redundancy}")
    return redundancy
