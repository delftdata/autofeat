import pandas as pd

from feature_selection.feature_selection_algorithms import FSAlgorithms
from utils.util_functions import prepare_data_for_ml


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
