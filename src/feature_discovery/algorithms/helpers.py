import time
from sklearn.feature_selection import SequentialFeatureSelector


def feature_selection(X, y, training_func):
    start = time.time()
    # sfs = SFS(estimator=training_func, k_features="best", n_jobs=1, cv=5)
    sfs = SequentialFeatureSelector(estimator=training_func, n_features_to_select="auto", scoring="accuracy")
    sfs.fit(X, y)
    X = sfs.transform(X)
    end = time.time()
    sfs_time = end - start
    print("==== Finished SFS =====")
    return X, sfs_time