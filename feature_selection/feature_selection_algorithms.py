import math

from ITMO_FS.filters.univariate import reliefF_measure, chi2_measure, f_ratio_measure, su_measure
from ITMO_FS.filters.multivariate import MIFS, CIFE, FCBFDiscreteFilter, TraceRatioFisher
import numpy as np
import pandas as pd




# Imported from https://github.com/ctlab/ITMO_FS as it is not available in the pip version of the package
from typing import List


def modified_t_score(x, y):
    """Calculate the Modified T-score for each feature. Bigger values mean
    more important features.
    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples. There can be only 2 classes.
    Returns
    -------
    array-like, shape (n_features,) : feature scores
    See Also
    --------
    For more details see paper <https://dergipark.org.tr/en/download/article-file/261247>.
    Examples
    --------
    """
    classes = np.unique(y)

    size_class0 = y[y == classes[0]].size
    size_class1 = y[y == classes[1]].size

    mean_class0 = np.mean(x[y == classes[0]], axis=0)
    mean_class0 = np.nan_to_num(mean_class0)
    mean_class1 = np.mean(x[y == classes[1]], axis=0)
    mean_class1 = np.nan_to_num(mean_class1)

    std_class0 = np.std(x[y == classes[0]], axis=0)
    std_class0 = np.nan_to_num(std_class0)
    std_class1 = np.std(x[y == classes[1]], axis=0)
    std_class1 = np.nan_to_num(std_class1)

    corr_with_y = np.apply_along_axis(
        lambda feature: abs(np.corrcoef(feature, y)[0][1]), 0, x)
    corr_with_y = np.nan_to_num(corr_with_y)

    corr_with_others = abs(np.corrcoef(x, rowvar=False))
    corr_with_others = np.nan_to_num(corr_with_others)

    mean_of_corr_with_others = (
        corr_with_others.sum(axis=1)
        - corr_with_others.diagonal()) / (len(corr_with_others) - 1)

    t_score_numerator = abs(mean_class0 - mean_class1)
    t_score_denominator = np.sqrt(
        (size_class0 * np.square(std_class0) + size_class1 * np.square(
            std_class1)) / (size_class0 + size_class1))
    modificator = corr_with_y / mean_of_corr_with_others

    t_score = t_score_numerator / t_score_denominator * modificator
    t_score = np.nan_to_num(t_score)

    return t_score


def _fcbf(X, y):
    fcbf = FCBFDiscreteFilter()
    return fcbf.fit_transform(X, y, feature_names=X.columns)
    # return fcbf.transform(X)


def _variance(dataframe: pd.DataFrame):
    return dataframe.var()


def _trace_ration_filter(X, y):
    tracer = TraceRatioFisher(math.floor(len(X.columns)/4))
    features = np.array(X)
    label = np.array(y)
    tracer.fit(X, y, feature_names=X.columns)
    selected = tracer.transform(X)
    return selected, selected.columns


def _relief_f(X, y, neighbors=1):
    neighbors = len(X.columns)
    features = np.array(X)
    label = np.array(y)
    return reliefF_measure(features, label, neighbors)


def _chi_squared_score(X, y):
    features = np.array(X)
    label = np.array(y)
    return chi2_measure(features, label)


def _fisher_score(X, y):
    features = np.array(X)
    label = np.array(y)
    return f_ratio_measure(X, y)


def _symmetrical_uncertainty(X, y):
    features = np.array(X)
    label = np.array(y)
    return su_measure(features, label)


def _mutual_info_feat_sel(selected_features, free_features, X, y):
    beta = 0.3
    return MIFS(np.array(selected_features), np.array(free_features), np.array(X), np.array(y), beta)


def _conditional_infonax_feat_extraction(selected_features, free_features, X, y):
    return CIFE(np.array(selected_features), np.array(free_features), np.array(X), np.array(y))


def _t_score(X, y):
    if len(X.columns) == 1:
        return np.array(0)
    else:
        return modified_t_score(X, y)


class FSAlgorithms:

    # Statistical based - Chi and T score have very high values for the features selected by the decision tree
    # VARIANCE = 'variance' # REMOVED - for normalised data, variance is useless
    CHI_SQ = 'chi-squared score' # values > 5
    T_SCORE = 't-score' # values > 0.5

    # Information theoretically based
    # for negative values: the highest, the better. If positive values, the smallest?
    MIFS = 'mututal information feature selection' # params: selected, free, X, y, coef for redundancy
    # for negative values: the highest, the better. If positive values, the smallest?
    CIFE = 'conditional infomax feature extraction' # params: selected, free, X, y
    # FCBF = 'fast-correlation based filter' # REMOVED - does not return scores -- Replaced with SU
    SU = 'symmetrical uncertainty' # features with scores > 0.4 are selected by the decision trees

    # Similarity based
    FISHER = 'fisher score' # values > 0.1 for the features selected by the decision tree
    # TR_CRITERION = 'trace ration criterion' # param: amount of feat to filter -- REMOVED - does not return scores
    # RELIEF = 'reliefF' # param: number of neighbors to consider -- REMOVED - un-informative

    ALGORITHMS = [CHI_SQ, T_SCORE, SU, FISHER]
    DEPENDENCY_ALG = [CHI_SQ, T_SCORE, SU]
    ALL = [CHI_SQ, T_SCORE, FISHER, MIFS, CIFE]
    ALGORITHM_FOREIGN_TABLE = [MIFS, CIFE]

    def feature_selection(self, selection_method, X, y):
        feat_sel = self._get_feat_sel(selection_method)
        return feat_sel(X, y)

    def _get_feat_sel(self, selection_method):
        if selection_method == self.CHI_SQ:
            return _chi_squared_score
        elif selection_method == self.T_SCORE:
            return _t_score
        elif selection_method == self.SU:
            return _symmetrical_uncertainty
        elif selection_method == self.FISHER:
            return _fisher_score
        else:
            raise ValueError(selection_method)

    def feature_selection_foreign_table(self, selection_method, selected_feat: List[int], free_features: List[int], X, y):
        feat_sel = self._get_feature_selection_foreign_table(selection_method)
        return feat_sel(selected_feat, free_features, X, y)

    def _get_feature_selection_foreign_table(self, selection_method):
        if selection_method == self.MIFS:
            return _mutual_info_feat_sel
        elif selection_method == self.CIFE:
            return _conditional_infonax_feat_extraction
        else:
            raise ValueError(selection_method)



