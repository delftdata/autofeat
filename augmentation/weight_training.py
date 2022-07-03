import os
from operator import itemgetter
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegressionCV, LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from joblib import dump, load

from augmentation.train_algorithms import train_CART_and_print, train_CART
from feature_selection.feature_selection_algorithms import FSAlgorithms

folder_name = os.path.abspath(os.path.dirname(__file__))
# mapping_path = f"{folder_name}/../mappings"
# joined_path = f"{folder_name}/../joined-df/football"
results_filename = "results.csv"


def create_features_dataframe(base_table_path, joined_data_path, target_column, mappings_path):
    base_table = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"',
                             escapechar='\\', nrows=1)
    base_table_features = list(base_table.drop(columns=[target_column]).columns)

    result = {'columns': [], FSAlgorithms.T_SCORE: [], FSAlgorithms.CHI_SQ: [], FSAlgorithms.FISHER: [], FSAlgorithms.SU: [], FSAlgorithms.MIFS: [], FSAlgorithms.CIFE: []}
    for f in listdir(joined_data_path):
        filename = join(joined_data_path, f)
        if isfile(filename):
            print(f"DATASET: {f}")
            joined_table = pd.read_csv(filename, header=0, engine="python", encoding="utf8", quotechar='"',
                                       escapechar='\\')

            df = joined_table.apply(lambda x: pd.factorize(x)[0])
            joined_table_no_base = df.drop(
                columns=set(df.columns).intersection(set(base_table_features)))
            df_features = list(map(lambda x: f"{filename}/{x}", joined_table_no_base.drop(columns=[target_column]).columns))

            result['columns'] = result['columns'] + df_features
            X = joined_table_no_base.drop(columns=[target_column])
            y = joined_table_no_base[target_column].astype(int)

            scaler = MinMaxScaler()
            scaled_X = scaler.fit_transform(X)
            normalized_X = pd.DataFrame(scaled_X, columns=X.columns)

            all_features = dict(zip(list(range(0, len(df.drop(columns=[target_column]).columns))), df.drop(columns=[target_column]).columns))
            left_features = dict(filter(lambda x: x[1] in base_table_features, all_features.items()))
            right_features = dict(filter(lambda x: x[1] in joined_table_no_base.columns and x[1] not in base_table_features,
                                       all_features.items()))

            fs = FSAlgorithms()
            for alg in fs.ALGORITHMS:
                # print(f"\tAlgorithm: {alg}")
                # print(f"\tColumns: {df_features}")
                scores = fs.feature_selection(alg, normalized_X, y)
                result[alg] = result[alg] + list(scores)
                # print(f"\t\tScores: {scores}")

            scaler = MinMaxScaler()
            scaled_X = scaler.fit_transform(df.drop(columns=[target_column]))
            normalized_X = pd.DataFrame(scaled_X, columns=df.drop(columns=[target_column]).columns)
            for alg in fs.ALGORITHM_FOREIGN_TABLE:
                scores = fs.feature_selection_foreign_table(alg, list(left_features.keys()), list(right_features.keys()), normalized_X, joined_table[target_column])
                result[alg] = result[alg] + list(scores)

    result_dataframe = pd.DataFrame.from_dict(result)
    filepath = f"{os.path.join(folder_name, '../', mappings_path)}/{results_filename}"
    previous_result = pd.read_csv(filepath)
    pd.concat([previous_result, result_dataframe]).to_csv(filepath, index=False)


def create_ground_truth(join_path, label_column, base_table_path, mappings_path):
    result_path = f"{os.path.join(folder_name, '../', mappings_path)}/{results_filename}"
    results_dataframe = pd.read_csv(result_path)
    base_table = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"',
                             escapechar='\\', nrows=1)
    base_table_features = list(base_table.drop(columns=[label_column]).columns)
    for f in tqdm(listdir(join_path)):
        filename = join(join_path, f)
        if isfile(filename):
            print(f"DATASET: {f}")
            table = pd.read_csv(filename, header=0, engine="python", encoding="utf8", quotechar='"',
                                escapechar='\\')
            df = table.apply(lambda x: pd.factorize(x)[0])
            X = df.drop(columns=[label_column])
            y = df[label_column]

            scaler = MinMaxScaler()
            scaled_X = scaler.fit_transform(X)
            normalized_X = pd.DataFrame(scaled_X, columns=X.columns)

            # acc_decision_tree, params = train_CART(X, y)
            # acc_decision_tree, params, feat_score = train_CART_and_print(X, y, f, f"{join_path}/trees")
            acc_decision_tree, params, feat_score = train_CART(normalized_X, y)
            columns_index = [i for i, col in enumerate(X.columns) if col not in base_table_features]
            column_names = itemgetter(*columns_index)(list(X.columns))
            column_mapping = list(map(lambda x: f"{filename}/{x}", column_names))
            results_dataframe.loc[results_dataframe['columns'].isin(column_mapping), 'score'] = itemgetter(
                *columns_index)(feat_score)
            results_dataframe.loc[results_dataframe['columns'].isin(column_mapping), 'accuracy'] = acc_decision_tree
            results_dataframe.loc[results_dataframe['columns'].isin(column_mapping), 'depth'] = params['max_depth']

    results_dataframe.to_csv(result_path, index=False)


def train_logistic_regression(mappings_path):
    result_path = f"{os.path.join(folder_name, '../', mappings_path)}/{results_filename}"
    dataframe = pd.read_csv(result_path, header=0, engine="python", encoding="utf8",
                            quotechar='"',
                            escapechar='\\')
    X = dataframe.drop(columns=['score', 'depth', 'columns', 'accuracy'])
    y = dataframe['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # clf = LogisticRegressionCV(cv=5, random_state=24, class_weight='balanced').fit(X, y)
    clf = LassoLarsCV(cv=10, normalize=False).fit(X, y)

    clf_path = f"{os.path.join(folder_name, '../', mappings_path)}/regressor.joblib"

    dump(clf, clf_path)
    y_pred = clf.predict(X_test)
    acc = mean_squared_error(y_test, y_pred, squared=False)
    print(acc)
    print(clf.coef_)
