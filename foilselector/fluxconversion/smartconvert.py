"""
More advanced functions to convert flux from one representation into anthoer.
This module is named smartconvert because it interactively converts to other scales,
thus relies on the interactions and convert module in the same sub-package.
"""
# numpy functions
import numpy as np
from numpy import array as ary

# custom functions
from foilselector.fluxconversion.interactions import ask_question as _ask_question, get_column_interactive as _get_column_interactive
from foilselector.fluxconversion.convert import convert_arbitrary_gs_from_means as _convert_arbitrary_gs_from_means
from foilselector.constants import MeV, keV



def scale_to_eV_interactive(gs_ary):
    """
    Scales the group structure values so that it describes the group structure in the correct unit (eV).

    returns
    -------
    gs_ary : group structure array, of the same shape as the input gs_ary, but now each value describes the energy bin bounds in eV.
    """
    gs_ary_fmt_unit_question = (
        f"Were the group structure values \n{gs_ary}\n given in 'eV', 'keV', or 'MeV'?"
    )
    # scale the group structure up
    gs_ary_fmt_unit = _ask_question(gs_ary_fmt_unit_question, ["eV", "keV", "MeV"])
    if gs_ary_fmt_unit == "MeV":
        gs_ary *= MeV
    elif gs_ary_fmt_unit == "keV":
        gs_ary *= keV
    return gs_ary


def ask_for_gs(directory):
    """
    Check directory for a file containing a group structure; and interact with the program user to obtain said group structure.
    """
    gs_fmt_question = "Were the flux values provided along with the mean energy of the bin ('class mark'), or the upper and lower ('class boundaries')?"
    gs_fmt = _ask_question(gs_fmt_question, ["class mark", "class boundaries"])

    if gs_fmt == "class mark":
        gs_mean = _get_column_interactive(
            directory, "a priori spectrum's mean energy of each bin"
        )
        bin_sizes = np.diff(gs_mean)
        assert all(
            bin_sizes > 0
        ), "The a priori spectrum must be given in ascending order of energy."
        # deal with two special cases: lin space and log space
        if all(np.isclose(np.diff(bin_sizes), 0, atol=1e-3)):  # second derivative = 0
            print("equal spacing in energy space detected")
            average_step = np.mean(bin_sizes)
            start_point, end_point = gs_mean[[0, -1]]
            gs_min = np.linspace(
                start_point - average_step / 2,
                end_point - average_step / 2,
                num=len(gs_mean),
                endpoint=True,
            )
            gs_max = np.linspace(
                start_point + average_step / 2,
                end_point + average_step / 2,
                num=len(gs_mean),
                endpoint=True,
            )
        elif all(
            np.isclose(np.diff(np.diff(gs_mean)), 0, atol=1e-3)
        ):  # second derivative of log(E) = 0
            average_step = np.mean(np.diff(np.log10(gs_mean)))
            start_point, end_point = np.log10(gs_mean[[0, -1]])
            gs_min = np.logspace(
                start_point - average_step / 2,
                end_point - average_step / 2,
                num=len(gs_mean),
                endpoint=True,
            )
            gs_max = np.logspace(
                start_point + average_step / 2,
                end_point + average_step / 2,
                num=len(gs_mean),
                endpoint=True,
            )
        else:  # neither log-spaced nor lin-spaced
            gs_min, gs_max = _convert_arbitrary_gs_from_means(gs_mean).T

    elif gs_fmt == "class boundaries":
        while True:
            gs_min = _get_column_interactive(
                directory, "lower bounds of the energy groups"
            )  # , first_time_use=True)
            gs_max = _get_column_interactive(
                directory, "upper bounds of the energy groups"
            )
            if (gs_min < gs_max).all():
                break
            print(
                "The left side of the bins must have strictly lower energy than the right side of the bins. Please try again:"
            )
    gs_ary = scale_to_eV_interactive(ary([gs_min, gs_max]).T)
    return gs_ary
