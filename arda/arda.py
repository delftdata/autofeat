import numpy as np
from sklearn.ensemble import RandomForestRegressor


# algo 2
# A                     the (normalized) data matrix
# n                     the amount of features to generate
#
# Return: A matrix of generated random features, where each column represents one feature
def gen_features(A, n):
    L = []
    d = A.shape[1]
    m = np.mean(A, axis=1)
    s = np.cov(A)
    for i in range(n * d):
        L.append(np.random.multivariate_normal(m, s))
    return np.array(L).T


# algo 1
# A                     the (normalized) data matrix
# y                     the feature to use as criterion/dependent variable for the regressors
# tau                   threshold for how many times a feature appeared in front of synthesized features in ranking
# eta                   fraction of random features to inject (fraction of amount of features in A)
# k                     number of times ranking and counting is performed
# regressor             the regressor to use
#
# Returns: An array of indices, corresponding to selected features from A
def select_features(A, y, tau=0.1, eta=0.5, k=20, regressor=RandomForestRegressor):
    d = A.shape[1]
    augment_count = round(eta * d)
    X = np.concatenate((A, gen_features(A, augment_count)), axis=1)  # This gives us A' from the paper
    agg = np.zeros(d)

    # Repeat process 'k' times, as in the algorithm
    for i in range(k):
        reg = regressor()
        mask = np.zeros(d + augment_count, dtype=bool)
        mask[d:] = True  # We mark the columns that were generated
        reg.fit(X, y)
        indices = reg.feature_importances_.argsort()[::-1]  # Then we obtain sorting indices for the rankings
        sorted_mask = mask[indices[::]]  # These indices are then used to sort the mask, so we know where the generated columns are located in terms of ranking
        
        # Then we iterate through this mask until we hit a generated feature, adding 1 for all the original features that were in front
        for i, val in zip(indices, sorted_mask):
            if val:
                break
            else:
                agg[i] += 1

    # Return a set of indices selected by thresholding the normalized frequencies with 'tau'
    return np.arange(d)[agg/k > tau]


# algo 3
# A                     the (normalized) data matrix
# y                     the feature to use as criterion/dependent variable for the regressors
# train_method(X, y)    the method that performs training/testing/etc... We expect the first return var to be the accuracy
# T                     A list with tresholds (see tau in algo 2) to use
# regressor             The regressor to use for ranking in algo 2
#
# Returns: An array of indices, corresponding to selected features from A
def wrapper_algo(A, y, train_method, T, regressor=RandomForestRegressor):
    if (A.shape[0] != y.shape[0]):
        raise ValueError("Criterion/feature 'y' should have the same amount of rows as 'A'")
    
    last_accuracy = 0
    last_indices = [] 
    for t in sorted(T):
        indices = select_features(A, y, tau=t, regressor=RandomForestRegressor)
        result = train_method(A[:, indices], y)

        # Assume accuracy is the first return value
        accuracy = result[0]
        if accuracy < last_accuracy:
            break
        else:
            last_accuracy = accuracy
            last_indices = indices
    return last_indices