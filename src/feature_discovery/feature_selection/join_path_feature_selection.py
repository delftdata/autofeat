from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from ITMO_FS.filters.multivariate import CMIM, JMI, MRMR
from ITMO_FS.filters.univariate import spearman_corr
from ITMO_FS.utils.information_theory import entropy, conditional_mutual_information, conditional_entropy

from feature_discovery.augmentation.trial_error import train_test_cart
from feature_discovery.config import ROOT_FOLDER
from feature_discovery.data_preparation.utils import prepare_data_for_ml


class RelevanceRedundancy:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.target_entropy = None
        self.dataframe_entropy = {}
        self.dataframe_conditional_entropy = {}

    def measure_relevance_and_redundancy(self,
                                         dataframe: pd.DataFrame,
                                         selected_features: List[str],
                                         new_features: List[str],
                                         target_column: pd.Series) -> Tuple[
        Optional[List[float]], List[str], Optional[List[float]], List[str]]:

        # Relevance
        if self.target_entropy is None:
            self.target_entropy = entropy(target_column)

        new_common_features = list(set(dataframe.columns).intersection(set(new_features)))
        if len(new_common_features) == 0:
            return None, [], None, []

        new_dataframe = dataframe[new_common_features]

        # h = hash(str(new_common_features))
        # if h in self.dataframe_entropy:
        #     entropy_x = self.dataframe_entropy[h]
        # else:
        #     entropy_x = entropy(new_dataframe)
        #     self.dataframe_entropy[h] = entropy_x
        #
        # hash_cond_entropy = hash(str((np.array(new_dataframe), np.array(target_column))))
        # if hash_cond_entropy in self.dataframe_conditional_entropy:
        #     cond_entropy = self.dataframe_conditional_entropy[hash_cond_entropy]
        # else:
        #     cond_entropy = np.apply_along_axis(conditional_entropy, 0, np.array(new_dataframe), np.array(target_column))
        #     self.dataframe_conditional_entropy[hash_cond_entropy] = cond_entropy

        # Normalised information gain
        # information_gain_score = (self.target_entropy - cond_entropy) / max(entropy_x, self.target_entropy)
        information_gain_score = abs(spearman_corr(np.array(new_dataframe), np.array(target_column)))

        final_feature_scores_rel = []
        final_features_rel = []
        for value, name in list(zip(information_gain_score, new_common_features)):
            # value 0 means that the features are completely redundant
            # value 1 means that the features are perfectly correlated, which is still undesirable
            if 0 < value < 1:
                final_feature_scores_rel.append((name, value))
                final_features_rel.append(name)

        if len(final_features_rel) == 0:
            return None, [], None, []

        # # Conditional redundancy
        selected_features_int = [i for i, value in enumerate(dataframe.columns) if value in selected_features]
        new_features_int = [i for i, value in enumerate(dataframe.columns) if value in final_features_rel]
        # vectorized_function = lambda free_feature: \
        #     min(np.vectorize(
        #         lambda selected_feature:
        #         self.cached_conditional_mutual_information(np.array(dataframe)[:, free_feature], target_column,
        #                                                    np.array(dataframe)[:, selected_feature]), otypes=[float])(
        #         selected_features_int))
        # cmim_scores = np.vectorize(vectorized_function, otypes=[float])(np.array(new_features_int))
        # if np.all(np.array(cmim_scores) == 0):
        #     return final_feature_scores_rel, final_features_rel, None, []
        #
        # # normalise
        # normalised_scores = (cmim_scores - np.min(cmim_scores)) / (np.max(cmim_scores) - np.min(cmim_scores))
        # feature_scores = list(zip(np.array(dataframe.columns)[np.array(new_features_int)], normalised_scores))
        # final_feature_scores_cmim = [(name, value) for name, value in feature_scores if value > 0]
        # final_features_cmim = [feat for feat, _ in final_feature_scores_cmim]

        # Joint mutual information

        relevance = np.apply_along_axis(self.cached_mutual_information, 0,
                                        np.array(dataframe)[:, np.array(new_features_int)], target_column)
        redundancy = np.vectorize(
            lambda free_feature: np.sum(np.apply_along_axis(self.cached_mutual_information, 0,
                                                            np.array(dataframe)[:, np.array(selected_features_int)],
                                                            np.array(dataframe)[:, free_feature])))(new_features_int)
        # cond_dependency = np.vectorize(
        #     lambda free_feature: np.sum(np.apply_along_axis(self.cached_conditional_mutual_information, 0,
        #                                                     np.array(dataframe)[:, np.array(selected_features_int)],
        #                                                     np.array(dataframe)[:, free_feature],
        #                                                     target_column)))(np.array(new_features_int))
        mrmr_scores = relevance - (1 / np.array(selected_features_int).size) * redundancy
        # + (1 / np.array(selected_features_int).size) * cond_dependency
        if np.all(np.array(mrmr_scores) == 0):
            return final_feature_scores_rel, final_features_rel, None, []

        if np.max(mrmr_scores) == np.min(mrmr_scores):
            normalised_scores = (mrmr_scores - np.min(mrmr_scores)) / np.max(mrmr_scores)
        else:
            normalised_scores = (mrmr_scores - np.min(mrmr_scores)) / (np.max(mrmr_scores) - np.min(mrmr_scores))

        feature_scores = list(zip(np.array(dataframe.columns)[np.array(new_features_int)], normalised_scores))
        final_feature_scores_mrmr = [(name, value) for name, value in feature_scores if value > 0]
        final_feature_names_mrmr = [feat for feat, _ in final_feature_scores_mrmr]

        return final_feature_scores_rel, final_features_rel, final_feature_scores_mrmr, final_feature_names_mrmr

    def cached_mutual_information(self, x, y):
        h = hash(str((x, y)))
        if h in self.dataframe_conditional_entropy:
            cond_entropy = self.dataframe_conditional_entropy[h]
        else:
            cond_entropy = conditional_entropy(x, y)
            self.dataframe_conditional_entropy[h] = cond_entropy

        h_entropy = hash(str(y))
        if h_entropy in self.dataframe_entropy:
            entr = self.dataframe_entropy[h_entropy]
        else:
            entr = entropy(y)
            self.dataframe_entropy[h_entropy] = entr
        return entr - cond_entropy

    def cached_conditional_mutual_information(self, x, y, z):
        h1 = hash(str(list(zip(x, z))))
        if h1 in self.dataframe_entropy:
            entropy_xz = self.dataframe_entropy[h1]
        else:
            entropy_xz = entropy(list(zip(x, z)))
            self.dataframe_entropy[h1] = entropy_xz

        h2 = hash(str(list(zip(y, z))))
        if h2 in self.dataframe_entropy:
            entropy_yz = self.dataframe_entropy[h2]
        else:
            entropy_yz = entropy(list(zip(y, z)))
            self.dataframe_entropy[h2] = entropy_yz

        h3 = hash(str(list(zip(x, y, z))))
        if h3 in self.dataframe_entropy:
            entropy_xyz = self.dataframe_entropy[h3]
        else:
            entropy_xyz = entropy(list(zip(x, y, z)))
            self.dataframe_entropy[h3] = entropy_xyz

        h4 = hash(str(z))
        if h4 in self.dataframe_entropy:
            entropy_z = self.dataframe_entropy[h4]
        else:
            entropy_z = entropy(z)
            self.dataframe_entropy[h4] = entropy_z

        return entropy_xz + entropy_yz - entropy_xyz - entropy_z


def measure_relevance(dataframe: pd.DataFrame,
                      feature_names: List[str],
                      target_column: pd.Series) -> Tuple[Optional[list], list]:
    common_features = list(set(dataframe.columns).intersection(set(feature_names)))

    if len(common_features) == 0:
        return None, []

    features = dataframe[common_features]
    # scores = information_gain(np.array(features), np.array(target_column)) / max(entropy(features),
    #                                                                              entropy(target_column))
    scores = abs(spearman_corr(np.array(features), np.array(target_column)))

    final_feature_scores = []
    final_features = []
    for value, name in list(zip(scores, common_features)):
        # value 0 means that the features are completely redundant
        # value 1 means that the features are perfectly correlated, which is still undesirable
        if 0 < value < 1:
            final_feature_scores.append((name, value))
            final_features.append(name)

    return final_feature_scores, final_features


def measure_conditional_redundancy(dataframe: pd.DataFrame,
                                   selected_features: List[str],
                                   new_features: List[str],
                                   target_column: pd.Series,
                                   conditional_redundancy_threshold: float = 0.5) -> Tuple[Optional[list], list]:
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


def measure_redundancy(dataframe, feature_group: List[str], target_column) -> Tuple[Optional[list], List[str]]:
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


def measure_joint_mutual_information(dataframe: pd.DataFrame,
                                     selected_features: List[str],
                                     new_features: List[str],
                                     target_column: pd.Series) -> Tuple[Optional[list], list]:
    selected_features_int = [i for i, value in enumerate(dataframe.columns) if value in selected_features]
    new_features_int = [i for i, value in enumerate(dataframe.columns) if value in new_features]

    scores = MRMR(np.array(selected_features_int), np.array(new_features_int), np.array(dataframe),
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
