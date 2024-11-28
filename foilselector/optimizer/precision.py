import numpy as np


def get_precision_weight_vector(
    gs_array: np.ndarray, *, log_flux: bool = False, const_lethargy: bool = False
) -> np.ndarray:
    """
    Get the weight vector w for calculating precision.

    Parameters
    ----------
    gs_array:
        group structure showing the lower and upper bound of each bin, with shape==(n, 2)
    log_flux:
        whether we consider each unit log(flux/reference flux) [dimensionless] to have
        the same importance (True), or each unit flux [cm^-2 eV^-1] to have the same
        importance (False).
    const_lethargy:
        whether we consider each unit energy [eV] of the neutron spectrum to have the
        same importance (False), or each unit lethargy=log(energy/reference energy)
        [dimensionless] of the neutron flux to have the same importance (True).

    Returns
    -------
    :
        a weight vector (1d array) that is used by get_precision.
    """
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
        foil response matrix, where row = count in gamma-bin measured, column = per unit
        flux of incoming radiation.
    measured_variance:
        expected variance of the response_vector. Can be naively obtained by folding the
        response matrix with the a priori fluence.
        TODO: replace with full covariance matrix later.
    weight_vector:
        Obtained by get_precision_weight_vector

    Returns
    -------
    :
        precision/sensitivity value measured in the unit:
        [cm^2 eV^3] (log_flux==False, const_lethargy==False)
        [cm^2 eV^2] (log_flux==False, const_lethargy==True )
        [cm^2 eV^1] (log_flux==True , const_lethargy==False)
        [cm^2 eV^3] (log_flux==True , const_lethargy==True )
        where log_flux, const_lethargy are parameters used in get_precision_weight_vector.
    """
    if len(measured_variance) == 0:
        return 0.0
    return weight_vector @ np.diag(
        foil_response_matrix.T @ np.diag(1 / measured_variance) @ foil_response_matrix
    )
