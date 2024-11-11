"""
The mass of the foil needs to be chosen before the optimization can begin.
This mass rescales the response matrix appropriately for the response matrix precision
and accuracy calculation.
"""

from numpy import typing as npt

from openmc.data import atomic_mass


def calculate_max_num_reactants(
    response_matrix: npt.NDArray,
    apriori_fluence: npt.NDArray[float],
    max_gamma_count_rate: float,
    acquisition_duration: float,
):
    """
    Returns the maximum number of reactants in the reactant foil without breaking the
    max-gamma-count-rate threshold during the acquisition period.

    Parameters
    ----------
    response_matrix:
        2D np.array that represents the number of counts that would be collected into
        each bin of the gamma-ray spectrum during the acquisition period per reactant
        nucleus present initially present in the foil.
    apriori_fluence:
        np.array that represents the a priori spectrum (vector) * irradiation duration
        (scalar).
    max_gamma_count_rate:
        gamma-count rate capability of the gamma-ray spectrum acquisition set-up.

    acquisition_duration:
        length of the acquisition duration.

    Returns
    -------
    scale_factor
        How many reactants can the foil have.
    """
    total_counts_per_reactant = (response_matrix @ apriori_fluence).sum()
    num_reactants_in_foils = (
        max_num_counts(max_gamma_count_rate, acquisition_duration)
        / total_counts_per_reactant
    )
    return num_reactants_in_foils


def mass_of_one_reactant_atom(composition_dict):
    """Calculate the weighted average of the reactant nuclides' masses."""
    return sum(
        convert_to_mass(isotope) * fraction
        for isotope, fraction in composition_dict.items()
    )


def convert_to_mass(isotope_name):
    """Returns the atomic mass of an isotope in grams"""
    return atomic_mass(isotope_name) * 1.660538921e-24


def max_num_counts(max_gamma_count_rate, acquisition_duration):
    """
    The response matrix outputs the total number of gamma counts during the acquisition
    duration.
    Assuming the gamma-ray count rate is constant over the acquisition duration, in order
    to not let the gamma-ray spectrum acquisition set-up reach the saturation threshold
    (above which the energy resolution degrades), the max number of gamma counts is
    max_gamma_count_rate * acquisition_duration

    The user is expected to have inputted sensible durations such that the count rate
    difference between the start of gamma-ray spectrum acquisition and the end of
    acquisition would not be too dramatic, and therefore this approximation would not be
    too severely violated.
    (If the decay curve of the selected foil did end up to have a steep decay curve, then
    the experimentalist can simply program the gamma-ray spectrum to stop and re-start
    its acquisition at the right time to avoid the noise generated from one peak to not
    overpower the rest of the peaks.)
    """
    return max_gamma_count_rate * acquisition_duration
