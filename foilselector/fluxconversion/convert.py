"""
Function to convert between different representations of
1. flux
2. gs
"""  # noqa: D400

import numpy as np
import pandas as pd
from numpy import log as ln

from foilselector.constants import MeV, keV

__all__ = ["convert_arbitrary_gs_from_means", "flux_conversion"]

_ASSERT_STR = "format 'i' must be one of the following 4:"
_ACCEPTED_FMTS = "'integrated'|'PUL'(per unit lethargy)|'per ({})eV'"


def flux_conversion(flux_in, gs_in_eV, in_fmt: str, out_fmt: str):
    """
    Convert flux from a one representation into another.

    Parameters
    ----------
    flux_in:
        flux to be converted into out_fmt. A pd.DataSeries/DataFrame, a numpy array or a
        list of (n) flux values. The meaning of each of the n flux values is should be
        specified by the parameter in_fmt (see in_fmt below).

    gs_in_eV:
        group-structure with shape = (n, 2), which denotes the upper and lower energy
        boundaries of the bin

    in_fmt, out_fmt : string describing the format of the flux when inputted/outputted
                      Accepted argument for fmt's:
                        "per MeV",
                        "per eV",
                        "per keV",
                        "integrated",
                        "PUL"

    Returns
    -------
    flux_out:
        The flux converted into the output format.

    Raises
    ------
    ValueError
        Unaccepted input/output format.
    """
    flux = (
        flux_in.to_numpy().T
        if isinstance(flux_in, pd.DataFrame | pd.Series)
        else flux_in
    )
    # convert all of them to per eV
    if in_fmt == "per MeV":
        flux_per_eV = flux / MeV
    elif in_fmt == "integrated":
        flux_per_eV = flux / np.diff(gs_in_eV, axis=1).flatten()
    elif in_fmt == "PUL":
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux * leth_space
        flux_per_eV = flux_conversion(
            flux_integrated,
            gs_in_eV,
            "integrated",
            "per eV",
        )  # reuse the same function, but via a different path.
    elif in_fmt == "per keV":
        flux_per_eV = flux / keV
    elif in_fmt == "per eV":
        flux_per_eV = flux
    else:
        raise ValueError("the input " + _ASSERT_STR + _ACCEPTED_FMTS.format("k/M"))

    # convert from per eV back into output format
    if out_fmt == "per MeV":
        flux_out = flux_per_eV * MeV
    elif out_fmt == "integrated":
        flux_out = flux_per_eV * np.diff(gs_in_eV, axis=1).flatten()
    elif out_fmt == "PUL":
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux_conversion(
            flux_per_eV,
            gs_in_eV,
            "per eV",
            "integrated",
        )  # reuse the same function, but via a different path.
        flux_out = flux_integrated / leth_space
    elif out_fmt == "per eV":
        flux_out = flux_per_eV
    else:
        raise ValueError("the input " + _ASSERT_STR + _ACCEPTED_FMTS.format("M"))
        # does not allow per keV output, because that's not a standard/commonly used
        # energy unit.

    # give it back as the original type
    if isinstance(flux_in, pd.DataFrame | pd.Series):  # check type
        flux_out = type(flux_in)(flux_out)
        name_or_col = "column" if isinstance(flux_in, pd.DataFrame) else "name"
        setattr(flux_out, name_or_col, getattr(flux_in, name_or_col))
    return flux_out


def convert_arbitrary_gs_from_means(gs_means: np.ndarray):
    """
    Create a group structure (n bins, with upper and lower bounds each) from a list of
    n numbers. This is done by taking the first (n-1) numbers as the upper bounds of the
    last (n-1) bins, and the last (n-1) numbers as the lower bounds of the first (n-1)
    bins. The first bin's lower bound and the last bin's upper bound is obtained by
    extrapolating the bin width of the second bin and the penultimate bin respectively.

    Parameters
    ----------
    gs_means : a list of n numbers, describing the class-mark of each bin.

    Returns
    -------
    gs_array:
        The group structure inferred.
    """
    first_bin_size, last_bin_size = np.diff(gs_means)[[0, -1]]
    mid_points = gs_means[:-1] + np.diff(gs_means) / 2
    gs_min = np.hstack([gs_means[0] - first_bin_size / 2, mid_points])
    gs_max = np.hstack([mid_points, gs_means[-1] + last_bin_size / 2])
    return np.array([gs_min, gs_max]).T
