import numpy as np
from ITMO_FS.utils.information_theory import conditional_mutual_information, mutual_information


def measure_conditional_dependency(selected_features, free_features, X, y):
    selected = np.array(selected_features)
    free = np.array(free_features)
    arr_X = np.array(X)
    arr_Y = np.array(y)

    cond_dependency = np.vectorize(
        lambda free_feature: np.sum(np.apply_along_axis(conditional_mutual_information, 0,
                                                        arr_X[:, selected], arr_X[:, free_feature], arr_Y)))(
        free)
    print(f"Conditional dependency: {cond_dependency}")
    return cond_dependency


def measure_relevance(free_features, X, y):
    free = np.array(free_features)
    arr_X = np.array(X)
    arr_Y = np.array(y)
    relevance = np.apply_along_axis(mutual_information, 0, arr_X[:, free], arr_Y)
    print(f"Relevace: {relevance}")
    return relevance
