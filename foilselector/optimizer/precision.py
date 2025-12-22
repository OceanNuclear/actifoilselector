"""All functions required to calculate the precision metric for a response matrix."""

import numpy as np
import uncertainties

__all__ = [
    "get_precision_contributions",
    "get_precision_unit",
    "get_precision_weight_vector",
]


def get_precision_weight_vector(
    gs_array: np.ndarray,
    *,
    log_flux: bool = False,
    const_lethargy: bool = False,
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
    if log_flux:
        return log_diff * diff * diff
    return diff**3


def get_precision_unit(*, log_flux: bool = False, const_lethargy: bool = False):
    """Get the unit applicable to the precision metric chosen by the user.

    For a given setting of log_flux and const_lethargy, the precision parameter will have
    a corresponding unit:
    [cm^2 eV^3] (log_flux==False, const_lethargy==False)
    [cm^2 eV^2] (log_flux==True , const_lethargy==False)
    [cm^2 eV^1] (log_flux==False, const_lethargy==True )
    [cm^2]      (log_flux==True , const_lethargy==True )

    Parameters
    ----------
    log_flux:
        see :func: `get_precision_weight_vector`
    const_lethargy:
        see :func: `get_precision_weight_vector`

    Returns
    -------
    :
        The string stating the unit.
    """
    if (not log_flux) and (not const_lethargy):
        return "cm^2 eV^3"
    if (log_flux) and (not const_lethargy):
        return "cm^2 eV^2"
    if (not log_flux) and (const_lethargy):
        return "cm^2 eV^1"
    # log_flux and const_lethargy
    return "cm^2"


def get_precision_contributions(
    foil_response_matrix: np.ndarray,
    foil_response_vector: np.ndarray[uncertainties.core.AffineScalarFunc],
    weight_vector: np.ndarray[float],
) -> np.ndarray[float]:
    """
    Calculate the precision metric as defined in the thesis. Don't perform the summation
    yet.

    Parameters
    ----------
    foil_response_matrix:
        foil response matrix, where row = count in gamma-bin measured, column = per unit
        flux of incoming radiation.
    foil_response_vector:
        The foil response vector (i.e. the number of counts under every detectable peak)
        as obtained after being irradiated by the a priori fluence.
    weight_vector:
        Obtained by get_precision_weight_vector

    Returns
    -------
    :
        precision/sensitivity value measured in the unit specified by get_precision_unit,
        where log_flux, const_lethargy are parameters used in get_precision_weight_vector

    Note
    ----
        This function gets the 'precision' scalar quantity. This differs from the same
        'precision' quantity mentioned in the thesis by a simple multiplicative factor of
        'precision in program' = 'precision (in thesis)' * (irradiation duration)^2,
        because here the a priori fluence is used instead of the a priori flux.
        This change is minor and does not cause any meaningful difference to the result.
    """
    if len(foil_response_matrix) == 0:
        return np.zeros_like(weight_vector)
    covariance_matrix = uncertainties.covariance_matrix(foil_response_vector)
    return weight_vector * np.diag(
        foil_response_matrix.T
        @ np.linalg.pinv(covariance_matrix)
        @ foil_response_matrix,
    )
