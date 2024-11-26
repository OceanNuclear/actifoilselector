import numpy as np


def get_precision_weight_vector(
    gs_array: np.ndarray, *, log_flux: bool = False, const_lethargy: bool = False
):
    """Get the weight vector w for calculating precision."""
    diff = np.diff(gs_array, axis=1).flatten()
    log_diff = np.diff(np.log(gs_array), axis=1).flatten()
    if const_lethargy:
        if log_flux:
            return log_diff**3
        return log_diff * log_diff * diff
    elif log_flux:
        return log_diff * diff * diff
    return diff**3


def get_precision(
    foil_response_matrix: np.ndarray,
    measured_variance: np.ndarray[float],
    weight_vector: np.ndarray,
) -> float:
    """
    Parameters
    ----------
    foil_response_matrix:
        foil response matrix, where row = count in gamma-bin measured, column = flux of incoming radiation.
    measured_variance:
        expected variance of the response_vector. Can be naively obtained by folding the
        response matrix with the a priori fluence.
    weight_vector:
        Obtained by get_precision_weight_vector
    """
    if len(measured_variance) == 0:
        return 0.0
    return weight_vector @ np.diag(
        foil_response_matrix.T @ np.diag(1 / measured_variance) @ foil_response_matrix
    )
