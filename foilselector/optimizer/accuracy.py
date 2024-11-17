import numpy as np

def get_accuracy(response_matrix: np.ndarray) -> int:
    # do something with this
    nonsingular_and_singular_values = np.linalg.svdvals(foil_response_matrix)
    return np.count_nonzero(nonsingular_and_singular_values>...)
    # OR, use some sort of scaled tanh function??