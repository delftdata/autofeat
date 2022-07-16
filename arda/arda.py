import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging
from sklearn.model_selection import train_test_split


# algo 2
# A                     the (normalized) data matrix
# n                     the amount of features to generate
#
# Return: A matrix of generated random features, where each column represents one feature
def gen_features(A, eta):
    L = []
    d = A.shape[1]
    m = np.mean(A, axis=1)
    s = np.cov(A)
    for i in range(round(eta * d)):
        L.append(np.random.multivariate_normal(m, s))
    return np.array(L).T


# Counts how often features
# rankings             the rankings as determined by the ranking algorithm
# mask                 the bit mask indicating which columns were randomly generated (True) and which ones are real features (False)
# bin_size             size of the bin array, corresponds to the amount of columns in the data matrix (the amount of "real" features) 
def _bin_count_ranking(importances, mask, bin_size):
    indices = importances.argsort()[::-1]  # Then we obtain sorting indices for the rankings, flip order since we have importances
    sorted_mask = mask[indices[::]]  # These indices are then used to sort the mask, so we know where the generated columns are located in terms of ranking
    bins = np.zeros(bin_size)

    # Then we iterate through this mask until we hit a generated/random feature, adding 1 for all the original features that were in front
    for i, val in zip(indices, sorted_mask):
        if val:
            break
        else:
            bins[i] += 1

    return bins


# algo 1
# A                     the (normalized) data matrix
# y                     the feature to use as criterion/dependent variable for the regressors
# tau                   threshold for how many times a feature appeared in front of synthesized features in ranking
# eta                   fraction of random features to inject (fraction of amount of features in A)
# k                     number of times ranking and counting is performed
# regressor             the regressor to use
#
# Returns: An array of indices, corresponding to selected features from A
def select_features(A, y, tau=0.1, eta=0.5, k=20, regressor=RandomForestClassifier):
    d = A.shape[1]
    X = np.concatenate((A, gen_features(A, eta)), axis=1)  # This gives us A' from the paper

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[d:] = True  # We mark the columns that were generated
    counts = np.zeros(d)

    # Repeat process 'k' times, as in the algorithm
    for i in range(k):
        reg = regressor()
        reg.fit(X, y)
        counts += _bin_count_ranking(reg.feature_importances_, mask, d)
    # Return a set of indices selected by thresholding the normalized frequencies with 'tau'
    return np.arange(d)[counts/k > tau]


# algo 3
# A                     the (normalized) data matrix
# y                     the feature to use as criterion/dependent variable for the regressors
# T                     A list with tresholds (see tau in algo 2) to use
# eta                   fraction of random features to inject (fraction of amount of features in A)
# k                     number of times ranking and counting is performed
# estimator             An sklearn estimator to use (e.g. a Regressor)
# regressor             The regressor to use for ranking in algo 2
#
# Returns: An array of indices, corresponding to selected features from A
def wrapper_algo(A, y, T, eta=0.2, k=10, estimator=RandomForestClassifier, regressor=RandomForestClassifier):
    if (A.shape[0] != y.shape[0]):
        raise ValueError("Criterion/feature 'y' should have the same amount of rows as 'A'")
    
    last_accuracy = 0
    last_indices = [] 
    for t in sorted(T):
        X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2)
        indices = select_features(X_train, y_train, tau=t, eta=eta, k=k, regressor=regressor)

        # If this happens, the thresholds might have been too strict
        if (len(X_train.iloc[:, indices])==0):
            return last_indices

        model = estimator()
        model.fit(X_train.iloc[:, indices], y_train)
        accuracy = model.score(X_test.iloc[:, indices], y_test)
        if accuracy < last_accuracy:
            break
        else:
            last_accuracy = accuracy
            last_indices = indices

    return last_indices