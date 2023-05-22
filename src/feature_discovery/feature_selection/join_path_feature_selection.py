from typing import List, Tuple

import numpy as np
import pandas as pd
from ITMO_FS.filters.multivariate import CMIM, JMI
from ITMO_FS.filters.univariate import information_gain
from ITMO_FS.utils.information_theory import entropy, conditional_mutual_information

from feature_discovery.augmentation.trial_error import train_test_cart
from feature_discovery.config import ROOT_FOLDER
from feature_discovery.data_preparation.utils import prepare_data_for_ml


def measure_relevance(dataframe: pd.DataFrame, feature_names: List[str], target_column: pd.Series):
    print("Measure relevance ... ")
    common_features = list(set(dataframe.columns).intersection(set(feature_names)))
    features = dataframe[common_features]
    scores = information_gain(np.array(features), np.array(target_column)) / max(entropy(features),
                                                                                 entropy(target_column))
    final_feature_scores = []
    final_features = []
    for value, name in list(zip(scores, common_features)):
        # value 0 means that the features are completely redundant
        # value 1 means that the features are perfectly correlated, which is still undesirable
        if 0 < value < 1:
            final_feature_scores.append((name, value))
            final_features.append(name)

    return final_feature_scores, final_features


def measure_conditional_redundancy(dataframe: pd.DataFrame, selected_features: List[str], new_features: List[str],
                                   target_column: pd.Series, conditional_redundancy_threshold: float = 0.5):
    selected_features_int = [i for i, value in enumerate(dataframe.columns) if value in selected_features]
    new_features_int = [i for i, value in enumerate(dataframe.columns) if value in new_features]

    scores = CMIM(np.array(selected_features_int), np.array(new_features_int), np.array(dataframe),
                  np.array(target_column))

    if np.all(np.array(scores) == 0):
        return None, []
    # normalise
    normalised_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    feature_scores = list(zip(np.array(dataframe.columns)[np.array(new_features_int)], normalised_scores))
    final_feature_scores = [(name, value) for name, value in feature_scores if value > 0]
    final_feature_names = [feat for feat, _ in final_feature_scores]

    return final_feature_scores, final_feature_names


def measure_redundancy(dataframe, feature_group: List[str], target_column) -> Tuple[List[float] or None, List[str]]:
    if len(feature_group) == 1 or len(feature_group) == 0:
        return None, feature_group

    scores = np.vectorize(lambda feature:
                          np.mean(np.apply_along_axis(lambda x, y, z: conditional_mutual_information(y, z, x), 0,
                                                     np.array(dataframe[
                                                                  np.setdiff1d(feature_group,
                                                                               feature)]),
                                                     np.array(dataframe[feature]),
                                                     np.array(target_column)
                                                     )))(feature_group)
    feature_scores = list(zip(np.array(feature_group), scores))
    final_feature_scores = [(name, value) for name, value in feature_scores if value > 0]
    final_feature_names = [feat for feat, _ in final_feature_scores]

    return final_feature_scores, final_feature_names


def measure_joint_mutual_information(dataframe: pd.DataFrame, selected_features: List[str], new_features: List[str],
                                     target_column: pd.Series):
    selected_features_int = [i for i, value in enumerate(dataframe.columns) if value in selected_features]
    new_features_int = [i for i, value in enumerate(dataframe.columns) if value in new_features]

    scores = JMI(np.array(selected_features_int), np.array(new_features_int), np.array(dataframe),
                 np.array(target_column))

    if np.all(np.array(scores) == 0):
        return None, []

    if np.max(scores) == np.min(scores):
        normalised_scores = (scores - np.min(scores)) / np.max(scores)
    else:
        normalised_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    feature_scores = list(zip(np.array(dataframe.columns)[np.array(new_features_int)], normalised_scores))
    final_feature_scores = [(name, value) for name, value in feature_scores if value > 0]
    final_feature_names = [feat for feat, _ in final_feature_scores]

    return final_feature_scores, final_feature_names


def test_relevance():
    target_column = "Survived"
    # target_column = "classification"
    # titanic_df = pd.read_csv(ROOT_FOLDER / "other-data/original/titanic-og.csv")
    # titanic_df = pd.read_csv(ROOT_FOLDER / "other-data/original/kidney_disease.csv")
    table1 = pd.read_csv(ROOT_FOLDER / "other-data/synthetic/titanic/titanic.csv")
    X, y = prepare_data_for_ml(table1, target_column)
    feature_score, selected_features = measure_relevance(table1, X.columns, y)
    print(feature_score)


    table2 = pd.read_csv(ROOT_FOLDER / "joined-df/titanic/titanic.csv--passenger_info.csv")
    X2, y2 = prepare_data_for_ml(table2, target_column)
    table2_columns = [col for col in X2.columns if col not in X.columns]
    feature_score2 = measure_relevance(table2, table2_columns, y2)
    print(feature_score2)

    feature_score_cr = measure_conditional_redundancy(dataframe=table2,
                                                      selected_features=selected_features,
                                                      new_features=table2_columns,
                                                      target_column=y2)
    print(feature_score_cr)

    new_selected_features = [feat for feat, val in feature_score_cr]
    # selected_features.extend(new_selected_features)
    #
    feature_score_jmi = measure_joint_mutual_information(dataframe=table2,
                                                         selected_features=selected_features,
                                                         new_features=table2_columns,
                                                         target_column=y2)
    print(feature_score_jmi)
    features_jmi = [feat for feat, value in feature_score_jmi]
    #
    # # feature_group_1 = ['PassengerId', 'Name', 'Ticket', 'Fare']
    # feature_group_2 = ['Ticket', 'Fare', 'Age', 'Sex', 'SibSp', 'Cabin']
    # # measure_redundancy(X, feature_group_1, y)
    feature_score_redundancy = measure_redundancy(table2, features_jmi, y2)
    print(feature_score_redundancy)


def test_features():
    target_column = "Survived"
    # target_column = "classification"
    # titanic_df = pd.read_csv(ROOT_FOLDER / "other-data/original/titanic-og.csv")
    # titanic_df = pd.read_csv(ROOT_FOLDER / "other-data/original/kidney_disease.csv")
    titanic_df = pd.read_csv(ROOT_FOLDER / "other-data/synthetic/titanic/titanic.csv")
    titanic_passenger = pd.read_csv(ROOT_FOLDER / "joined-df/titanic/titanic.csv--passenger_info.csv")

    entry2 = train_test_cart(dataframe=titanic_df,
                             target_column=target_column)

    entry1 = train_test_cart(dataframe=titanic_passenger[['TicketId', 'Age', 'Sex', 'SibSp', 'Survived']],
                             target_column=target_column)
    print(entry1)
    print(entry2)


if __name__ == "__main__":
    # test_relevance()
    test_features()
