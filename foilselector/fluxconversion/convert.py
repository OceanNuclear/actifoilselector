"""
function to convert between differetn representations of 
1. flux
2. gs
"""

import pandas as pd
import numpy as np
from numpy import log as ln
from foilselector.constants import MeV, keV


def flux_conversion(flux_in, gs_in_eV, in_fmt, out_fmt):
    """
    Convert flux from a one representation into another.

    Parameters
    ----------
    flux_in : flux to be converted into out_fmt. A pd.DataSeries/DataFrame, a numpy array or a list of (n) flux values.
              The meaning of each of the n flux values is should be specified by the parameter in_fmt (see in_fmt below).

    gs_in_eV : group-structure with shape = (n, 2), which denotes the upper and lower energy boundaries of the bin

    in_fmt, out_fmt : string describing the format of the flux when inputted/outputted
                      Accepted argument for fmt's:
                        "per MeV",
                        "per eV",
                        "per keV",
                        "integrated",
                        "PUL"
    """
    if isinstance(flux_in, (pd.DataFrame, pd.Series)): #check type
        flux = flux_in.values.T
    else:
        flux = flux_in
    # convert all of them to per eV
    if in_fmt == "per MeV":
        flux_per_eV = flux / MeV
    elif in_fmt == "integrated":
        flux_per_eV = flux / np.diff(gs_in_eV, axis=1).flatten()
    elif in_fmt == "PUL":
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux * leth_space
        flux_per_eV = flux_conversion(
            flux_integrated, gs_in_eV, "integrated", "per eV"
        )  # reuse the same function, but via a different path.
    elif in_fmt == "per keV":
        flux_per_eV = flux / keV
    else:
        assert (
            in_fmt == "per eV"
        ), "the input format 'i' must be one of the following 4='integrated'|'PUL'(per unit lethargy)|'per (k/M)eV'"
        flux_per_eV = flux

    # convert from per eV back into output format
    if out_fmt == "per MeV":
        flux_out = flux_per_eV * MeV
    elif out_fmt == "integrated":
        flux_out = flux_per_eV * np.diff(gs_in_eV, axis=1).flatten()
    elif out_fmt == "PUL":
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux_conversion(
            flux_per_eV, gs_in_eV, "per eV", "integrated"
        )  # reuse the same function, but via a different path.
        flux_out = flux_integrated / leth_space
    else:
        assert (
            out_fmt == "per eV"
        ), "the input format 'i' must be one of the following 4='integrated'|'PUL'(per unit lethargy)|'per (M)eV'"
        # does not allow per keV output, because that's not a standard/common method to use.
        flux_out = flux_per_eV

    # give it back as the original type
    if isinstance(flux_in, (pd.DataFrame, pd.Series)): #check type
        flux_out = type(flux_in)(flux_out)
        name_or_col = "column" if isinstance(flux_in, pd.DataFrame) else "name"
        setattr(flux_out, name_or_col, getattr(flux_in, name_or_col))
    return flux_out


def convert_arbitrary_gs_from_means(gs_means):
    """
    Create a group structure (n bins, with upper and lower bounds each) from a list of n numbers.
    This is done by taking the first (n-1) numbers as the upper bounds of the last (n-1) bins,
    and the last (n-1) numbers as the lower bounds of the first (n-1) bins.
    The first bin's lower bound and the last bin's upper bound is obtained by
    extrapolating the bin width of the second bin and the penultimate bin respectively.

    parameters
    ----------
    gs_means : a list of n numbers, describing the class-mark of each bin.
    """
    first_bin_size, last_bin_size = np.diff(gs_means)[[0, -1]]
    mid_points = gs_means[:-1] + np.diff(gs_means) / 2
    gs_min = np.hstack([gs_means[0] - first_bin_size / 2, mid_points])
    gs_max = np.hstack([mid_points, gs_means[-1] + last_bin_size / 2])
    return np.array([gs_min, gs_max]).T
