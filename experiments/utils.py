from typing import Callable

from augmentation.train_algorithms import train_CART, train_ID3, train_XGBoost


def hp_tune_join_all(X, y, training_fun: Callable, do_sfs: bool):
    accuracy, params, feature_importances, train_time, sfs_time = training_fun(X, y, do_sfs)

    final_feature_importances = map_features_scores(feature_importances, X)

    return accuracy, params["max_depth"], final_feature_importances, train_time, sfs_time


def map_features_scores(feature_importances, X):
    final_feature_importances = (
        dict(zip(X.columns, feature_importances)) if len(feature_importances) > 0 else {}
    )
    final_feature_importances = {
        feature: importance
        for feature, importance in final_feature_importances.items()
        if importance > 0
    }
    return final_feature_importances

