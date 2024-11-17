"""
The mass of the foil needs to be chosen before the optimization can begin.
This mass rescales the response matrix appropriately for the response matrix precision
and accuracy calculation.
"""

from __future__ import annotations

from openmc.data import atomic_mass
from collections.abc import Callable
from typing import TYPE_CHECKING
from foilselector.constants import amu
from foilselector.openmcextension.table import Tab1DExtended
from uncertainties import nominal_value as nom
if TYPE_CHECKING:
    from foilselector.simulation.efficiency import EfficiencyCurve
    from foilselector.openmcextension.library_reader import DiscreteRadiation, ContinuousRadiationDistribution

__all__ = ["choose_num_reactant_in_foil", "mass_from_num_atoms", "max_num_counts"]

def choose_num_reactant_in_foil(
    foil_response_matrix: dict[DiscreteRadiation, np.ndarray[float]],
    foil_background: dict[ContinuousRadiationDistribution, np.ndarray[float]],
    apriori_fluence: np.ndarray[float],
    max_counts_per_foil: float,
    compton_peak_ratio: Callable[[float | npt.NDArray], float | npt.NDArray] | None = None,
):
    """
    Returns the maximum number of reactants in the reactant foil without breaking the
    max-gamma-count-rate threshold during the acquisition period.

    Parameters
    ----------
    foil_response_matrix, foil_background:
        response matrix that gives the number of counts that would be collected into
        the gamma-ray spectrum during the acquisition period per reactant nucleus present
        initially present in the foil. Each column = sensitivity to each neutron bin
        energy. Each row is a different gamma-ray energy line outputted.
        foil_response_matrix gives discrete gamma-ray lines, while
        foil_background gives distribution of gamma-rays continuous over a range of
        gamma-ray energies.
    apriori_fluence:
        np.array that represents the a priori spectrum (vector) * irradiation duration
        (scalar).
    max_gamma_count_rate:
        gamma-count rate capability of the gamma-ray spectrum acquisition set-up.
    compton_peak_ratio:
        optional to include, accounts for the extra counts that 
    
    Returns
    -------
    num_reactants_in_foil:
        How many reactants can the foil have.
    largest_foil_response:
        similar to input foil_response_matrix, but each row's response is scaled up by
        num_reactants_in_foil to get the foil's total response if it were at max. mass.
    largest_foil_response:
        similar to input foil_background, but each row's response is scaled up by
        num_reactants_in_foil to get the foil's total response if it were at max. mass.
    """

    total_counts_per_reactant = 0.0  # we don't need fsum because it's purely increasing.
    for line, resp in foil_response_matrix.items():
        num_released_per_reactant = (resp @ apriori_fluence) * nom(line.intensity)
        total_counts_per_reactant += num_released_per_reactant
        if compton_peak_ratio:
            total_counts_per_reactant += compton_peak_ratio(nom(line.energy)) * num_released_per_reactant
    for dist, resp in foil_background.items():
        # spectrum per decay of end-of-chain isotope
        num_detected_per_decay = Integrate(dist).definite_integral(*minmax(dist))
        total_counts_per_reactant += num_detected_per_decay * (resp @ apriori_fluence)
        if compton_peak_ratio:
            compton_per_decay = dist.apply_scaling(compton_peak_ratio)
            total_counts_per_reactant += Integrate(compton_per_decay).definite_integral(*minmax(compton_per_decay)) * (resp @ apriori_fluence)

    num_reactants_in_foil = max_counts_per_foil/total_counts_per_reactant
    return (
        num_reactants_in_foil, 
        {line: resp*num_reactants_in_foil for line, resp in foil_response_matrix.items()}
        {dist: resp*num_reactants_in_foil for dist, resp in foil_background.items()}
    )


def convert_to_mass(isotope_name):
    """Returns the atomic mass of an isotope in grams"""
    return atomic_mass(isotope_name) * amu


def mass_of_one_reactant_atom(composition_dict):
    """Calculate the weighted average of the reactant nuclides' masses, in grams."""
    return sum(
        convert_to_mass(isotope) * fraction
        for isotope, fraction in composition_dict.items()
    )

def mass_from_num_atoms(num_atoms:float, isotope_composition: dict[str, float]) -> float:
    """
    Parameters
    ----------
    num_atoms:
        Total number of atoms in that foil.
    isotope_composition:
        isotope composition of foil, in the format {reactant_isotope: fraction}, such
        that sum(fraction)==1.

    Returns
    -------
    mass:
        unit [g]
    """
    return num_atoms * mass_of_one_reactant_atom(isotope_composition)

def max_num_counts(max_gamma_count_rate, acquisition_duration):
    """
    The response matrix outputs the total number of gamma counts during the acquisition
    duration.
    Assuming the gamma-ray count rate remains constant over the acquisition duration, in
    order to not let the gamma-ray spectrum acquisition set-up reach the saturation
    threshold (above which the energy resolution degrades), the max number of gamma
    counts is
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
