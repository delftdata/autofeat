import os
from operator import itemgetter
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegressionCV, LassoLarsCV
from sklearn.model_selection import train_test_split


from augmentation.feat_sel import FeatSel
from augmentation.train_algorithms import train_CART_and_print

folder_name = os.path.abspath(os.path.dirname(__file__))
mapping_path = f"{folder_name}/../mappings"
joined_path = f"{folder_name}/../joined-df/dt"
results_filename = "results.csv"


def create_features_dataframe(base_table_path, joined_data_path, target_column):
    base_table = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"',
                             escapechar='\\', nrows=1)
    base_table_features = list(base_table.drop(columns=[target_column]).columns)
    result = {'columns': [], FeatSel.GINI: [], FeatSel.RELIEF: [], FeatSel.CORR: [], FeatSel.SU: [], FeatSel.GAIN: []}
    for f in listdir(joined_data_path):
        filename = join(joined_data_path, f)
        if isfile(filename):
            # print(f"DATASET: {f}")
            joined_table = pd.read_csv(filename, header=0, engine="python", encoding="utf8", quotechar='"',
                                       escapechar='\\')
            joined_table_no_base = joined_table.drop(
                columns=set(joined_table.columns).intersection(set(base_table_features)))
            df = joined_table_no_base.apply(lambda x: pd.factorize(x)[0])
            df_features = list(map(lambda x: f"{f}/{x}", df.drop(columns=[target_column]).columns))
            result['columns'] = result['columns'] + df_features
            X = np.array(df.drop(columns=[target_column]))
            y = np.array(df[target_column])

            for alg in FeatSel.ALGORITHMS:
                # print(f"\tAlgorithm: {alg}")
                # print(f"\tColumns: {df_features}")
                scores = FeatSel().feature_selection(alg, X, y)
                result[alg] = result[alg] + list(scores)
                # print(f"\t\tScores: {scores}")

    result_dataframe = pd.DataFrame.from_dict(result)
    previous_result = pd.read_csv(f"{mapping_path}/{results_filename}")
    pd.concat([previous_result, result_dataframe]).to_csv(f"{mapping_path}/{results_filename}", index=False)


def create_ground_truth(join_path, label_column, base_table_path):
    results_dataframe = pd.read_csv(f"{mapping_path}/{results_filename}")
    base_table = pd.read_csv(base_table_path, header=0, engine="python", encoding="utf8", quotechar='"',
                             escapechar='\\', nrows=1)
    base_table_features = list(base_table.drop(columns=[label_column]).columns)
    for f in listdir(join_path):
        if isfile(join(join_path, f)):
            table = pd.read_csv(join(join_path, f), header=0, engine="python", encoding="utf8", quotechar='"',
                                escapechar='\\')
            df = table.apply(lambda x: pd.factorize(x)[0])
            X = df.drop(columns=[label_column])
            y = df[label_column]

            # acc_decision_tree, params = train_CART(X, y)
            acc_decision_tree, params, feat_score = train_CART_and_print(X, y, f, f"{join_path}/trees")
            columns_index = [i for i, col in enumerate(X.columns) if col not in base_table_features]
            column_names = itemgetter(*columns_index)(list(X.columns))
            column_mapping = list(map(lambda x: f"{f}/{x}", column_names))
            results_dataframe.loc[results_dataframe['columns'].isin(column_mapping), 'score'] = itemgetter(
                *columns_index)(feat_score)
            results_dataframe.loc[results_dataframe['columns'].isin(column_mapping), 'accuracy'] = acc_decision_tree
            results_dataframe.loc[results_dataframe['columns'].isin(column_mapping), 'depth'] = params['max_depth']

    results_dataframe.to_csv(f"{mapping_path}/{results_filename}", index=False)


def train_logistic_regression():
    dataframe = pd.read_csv(f"{mapping_path}/{results_filename}", header=0, engine="python", encoding="utf8",
                            quotechar='"',
                            escapechar='\\')
    X = dataframe.drop(columns=['score', 'depth', 'columns', 'accuracy'])
    y = dataframe['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # clf = LogisticRegressionCV(cv=5, random_state=24, class_weight='balanced').fit(X, y)
    clf = LassoLarsCV(cv=10, normalize=False).fit(X, y)
    y_pred = clf.predict(X_test)
    acc = mean_squared_error(y_test, y_pred, squared=False)
    print(acc)
    print(clf.coef_)
