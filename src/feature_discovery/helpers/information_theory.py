import numpy as np

from numba import jit


def elog(value: np.ndarray) -> np.ndarray:
    log_values = np.zeros_like(value)
    mask = value > 0
    log_values[mask] = np.log(value[mask])
    return value * log_values


def conditional_entropy(x_j: np.ndarray, y: np.ndarray) -> np.float64:
    # Find unique values and their corresponding indices
    unique_x, inverse_x = np.unique(x_j, return_inverse=True)
    unique_y, inverse_y = np.unique(y, return_inverse=True)

    # Create a contingency table
    contingency_table = np.zeros((len(unique_x), len(unique_y)), dtype=int)
    np.add.at(contingency_table, (inverse_x, inverse_y), 1)

    # Compute counts for each unique value in x_j
    counts_x = np.sum(contingency_table, axis=1)

    # Calculate the probabilities P(Y|X)
    prob_y_given_x = contingency_table / counts_x[:, None]

    # Compute entropy for each unique value in x_j
    part_entropies = -np.sum(elog(prob_y_given_x), axis=1)

    # Compute the weighted average of the part_entropies
    entropy = np.sum(counts_x / len(x_j) * part_entropies)

    return entropy
