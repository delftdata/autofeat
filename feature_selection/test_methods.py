import os

import pandas as pd

from augmentation.train_algorithms import train_CART_and_print
from feature_selection.feature_selection_algorithms import FSAlgorithms
from sklearn.preprocessing import MinMaxScaler

folder_name = os.path.abspath(os.path.dirname(__file__))
path = "../other-data/original/"
base_table = "titanic-og.csv"
# base_table = "FutbolMatches.csv"
# label_column = "win"
label_column = "Survived"
base_table_path = f"{os.path.join(folder_name, path, base_table)}"
join_path = f"{folder_name}/../joined-df/test-fs"
plot_path = os.path.join(folder_name, "../joined-df/test-fs/trees")
result_name = f"all-fs-{base_table}"
# result_name = f"baseline-{base_table}"
result_path = os.path.join(folder_name, "../mappings", result_name)


def train_baseline(base_table_name, label_column, base_table_path, result_path, plot_path):
    table = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
    df = table.apply(lambda x: pd.factorize(x)[0])
    X = df.drop(columns=[label_column])
    y = df[label_column]
    acc, params, feat_imp = train_CART_and_print(X, y, base_table_name, plot_path)
    result = {'feature': X.columns, 'score': feat_imp, 'accuracy': acc, 'depth': params['max_depth']}
    pd.DataFrame(result).to_csv(result_path, index=False)


def feature_selection(base_table_name, label_column, base_table_path, result_path, plot_path):
    table = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
    table.fillna(0, inplace=True)
    df = table.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)
    # print(df)

    X = df.drop(columns=[label_column])

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    normalized_X = pd.DataFrame(scaled_X, columns=X.columns)
    print(normalized_X)

    y = df[label_column].astype(int)
    # print(y)

    acc, params, feat_imp = train_CART_and_print(X, y, base_table_name, plot_path)
    result = {'feature': X.columns, 'score': feat_imp, 'accuracy': acc, 'depth': params['max_depth']}
    pd.DataFrame(result).to_csv(result_path, index=False)

    fs = FSAlgorithms()

    for alg in fs.ALGORITHMS:
        print(f"Processing: {alg}")
        result[alg] = fs.feature_selection(alg, normalized_X, y)

    pd.DataFrame(result).to_csv(result_path, index=False)


def feature_selection_foreign_table(base_table_path, foreign_table_path, from_col, to_col, join_result_name, plot_path, result_path):
    left_table_df = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
    right_table_df = pd.read_csv(foreign_table_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')

    joined_df = pd.merge(left_table_df, right_table_df, how="left", left_on=from_col, right_on=to_col, suffixes=("_b", ""))
    duplicate_col = [col for col in joined_df.columns if col.endswith('_b')]
    joined_df.drop(columns=duplicate_col, inplace=True)

    joined_df.fillna(0, inplace=True)
    df = joined_df.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)
    # print(df)
    X = df.drop(columns=[label_column])

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    normalized_X = pd.DataFrame(scaled_X, columns=X.columns)
    # print(normalized_X)

    y = df[label_column].astype(int)
    # print(y)

    X_features = dict(zip(list(range(0, len(X.columns))), X.columns))
    print(X_features)
    left_table_features = dict(filter(lambda x: x[1] in left_table_df.columns, X_features.items()))
    print(left_table_features)
    right_table_features = dict(filter(lambda x: x[1] in right_table_df.columns and x[1] not in left_table_df.columns,
                                       X_features.items()))
    print(right_table_features)

    acc, params, feat_imp = train_CART_and_print(normalized_X, y, join_result_name, plot_path)
    print(f"Feature importance for {normalized_X.columns}\n{feat_imp}")
    result = {'feature': right_table_features.values(), 'accuracy': acc, 'depth': params['max_depth']}
    pd.DataFrame(result).to_csv(result_path, index=False)

    fs = FSAlgorithms()

    for alg in fs.ALGORITHM_FOREIGN_TABLE:
        print(f"Processing: {alg}")
        result[alg] = fs.feature_selection_foreign_table(alg, list(left_table_features.keys()), list(right_table_features.keys()), normalized_X, y)

    pd.DataFrame(result).to_csv(result_path, index=False)



if __name__ == '__main__':
    # train_baseline(base_table, label_column, base_table_path, result_path, plot_path)
    # feature_selection(base_table, label_column, base_table_path, result_path, plot_path)

    base_table = "titanic.csv"
    base_table_path = os.path.join(folder_name, "../other-data/decision-trees-split/titanic/", base_table)
    foreign_table = "passenger_info.csv"
    foreign_table_path = os.path.join(folder_name, "../other-data/decision-trees-split/titanic/", foreign_table)
    result_path = os.path.join(folder_name, "../mappings", "fs-titanic-cabin.csv")
    from_col = "PassengerId"
    to_col = "PassengerId"
    join_result_name = "titanic-cabin.csv"
    feature_selection_foreign_table(base_table_path, foreign_table_path, from_col, to_col, join_result_name, plot_path,
                                    result_path)