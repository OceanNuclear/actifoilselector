from typing import TYPE_CHECKING

from collections import defaultdict
from collections.abc import Callable
import scipy
import numpy as np
from uncertainties import nominal_value as nom
from foilselector.constants import me_eV
from foilselector.openmcextension.library_reader import (
    DiscreteRadiation,
    ContinuousRadiationDistribution,
)

if TYPE_CHECKING:
    from unceratinties.core import Variable, AffineScalarFunc


def get_response_matrix_and_peaks_without_uncertainty(
    discrete_response_matrix: dict[DiscreteRadiation, np.ndarray],
) -> tuple[np.ndarray, dict[float | Variable, list[str]]]:
    """Disregarding all uncertainties."""
    matrix, radiations = [], defaultdict(list)
    for line, resp in discrete_response_matrix.items():
        matrix.append(np.array(nom(line.intensity) * resp))
        radiations[nom(line.energy)].append(line.source)
    return np.array(matrix, dtype=float), radiations


def simulate_peaks_with_uncertainties(
    discrete_response_matrix: dict[DiscreteRadiation, np.ndarray],
    apriori_fluence: np.ndarray[float],
) -> list[tuple[float | Variable, float | Variable, str]]:
    """
    Simulate the irradiation by the fluence, and include the uncertainties in both the energy and number of counts in each peak.
    """
    matrix = np.array(list(discrete_response_matrix.values()), dtype=float)
    peaks_multiplier = matrix @ apriori_fluence

    radiations = []
    for line, multiplier in zip(list(discrete_response_matrix.keys()), peaks_multiplier):
        radiations.append(
            DiscreteRadiation(line.energy, multiplier * line.intensity, line.source)
        )
    return radiations


def merge_delete_peaks(
    peak_list: list[DiscreteRadiation],
    resolution_curve: Callable[[float | np.ndarray], float | np.ndarray],
) -> list[DiscreteRadiation]:
    """Merge nearby peaks"""
    peak_buffer = []
    for peak in peak_list:
        peak_buffer.append(peak.copy())
    return


def fold_background(
    background: dict[ContinuousRadiationDistribution, np.ndarray[float]],
    apriori_fluence: np.ndarray,
) -> list[ContinuousRadiationDistribution]:
    """Similar to simulate_peaks_with_uncertainties, but with background instead."""
    matrix = np.array(list(background.values()), dtype=float)
    bg_multipler = matrix @ apriori_fluence

    bgs = []
    for dist, multiplier in zip(list(background.keys()), bg_multipler):
        bgs.append(
            ContinuousRadiationDistribution(dist.distribution * multiplier, dist.source)
        )
    return bgs


def corresponding_background_level(
    peak_list: list[DiscreteRadiation],
    folded_background: list[ContinuousRadiationDistribution],
    compton_from_peak_curve: Callable[
        [float | np.ndarray | AffineScalarFunc], float | np.ndarray | AffineScalarFunc
    ],
    test_locations: list[float] | None = None,
) -> list[AffineScalarFunc]:
    """
    Get the background heights at the test_locations.

    Parameters
    ----------
    peak_list:
        list of peaks, including their energy (eV) and intensity (counts).
    background:
        list of continuous peaks, including their energy and intensity.
    compton_from_peak_curve:
        Curve showing the Compton-to-peak ratio. See
        foilselector.simulation.detector.Compton_to_peak_curve_factory.
    test_locations
        Places on the gamma-ray energy axis where we want to calculate the background
        heights at. (unit: eV) If not provided, this is copied from the energies of
        peak_list.

    Returns
    -------
    bg_heights:
        list of background levels at the test_locations, given in unit [counts/eV].
    """
    bg_heights = []
    if test_locations is None:
        test_locations = [nom(peak.energy) for peak in peak_list]
    else:
        test_locations = [nom(energy_l) for energy_l in peak_list]

    for energy_l in test_locations:
        this_height = 0.0
        # background due to Compton-scattering.
        for peak_zeta in peak_list:
            zeta_edge_E = compton_edge(nom(peak_zeta.energy))
            if zeta_edge_E >= energy_l:
                compton_total = (
                    compton_from_peak_curve(peak_zeta.energy) * peak_zeta.intensity
                )
                this_height += compton_total / zeta_edge_E
        # background due to continuous peaks
        for dist, _source in folded_background:
            this_height += dist(energy_l)

        bg_heights.append(this_height)
    return bg_heights


def compton_edge(peak_energy: float) -> float:
    """
    Calculate the Compton edge energy corresponding to a photopeak, where a the photon recoils 180° and loses as much
    energy to the electron as possible.

    Parameters
    ----------
    peak_energy:
        energy of the photopeak, in eV.

    Returns
    -------
    Compton_edge_energy:
        energy of the compton edge corresponding to the thing.
    """
    factor = 1.0 + 2 * peak_energy / me_eV
    return peak_energy * (1 - 1 / factor)


def integrate_bg_area(
    background_level: float | AffineScalarFunc,
    width: float,
) -> AffineScalarFunc:
    return add_Poisson_error(background_level * width)


def contained_by_interval(standard_score: float) -> float:
    """
    The 68-95-99.7 Rule

    Returns
    -------
    area enclosed by a unit-Gaussian function (i.e. a normal distribution whose integral
    under the curve = 1) from -standard_score*sigma to +standard_score*sigma.
    """
    erf_z = scipy.special.erf(standard_score / np.sqrt(2))
    1 / 2 * (1 + erf_z)
    return


def integrate_peak_area(
    peak_list: list[DiscreteRadiation],
    resolution_curve: Callable[[float | np.ndarray], float | np.ndarray],
    background_levels: list[float | AffineScalarFunc],
    *,
    how_many_fwhm: float = 1.0,
) -> list[float | AffineScalarFunc]:
    """
    Get the number of background counts under a peak.

    Parameters
    ----------
    peak_list:
        Location (energy, eV) and total number of counts expected of the peaks.
    background_levels:
        A list of scalar that represents the background levels, which are assumed to be
        locally constant.
        unit: [counts/eV]
    resolution_curve:
        resolution curve that gives the FWHM at energy E (eV).
    how_many_fwhm:
        how many times of the FWHM to integrate over.

    Returns
    -------
    truncated_net_peak_areas:
        Net area of the peak from E=(energy-FWHM/2*how_many_fwhm)
        to E=(energy+FWHM/2*how_many_fwhm), with the appropriate uncertainties expected
        of it.
    """
    truncated_net_peak_areas = []
    for peak, bg_lvl in zip(peak_list, background_levels):
        bg_area = integrate_bg_area(
            bg_lvl, resolution_curve(peak.energy) * how_many_fwhm
        )
        truncated_peak_area = peak.intensity * contained_by_interval(how_many_fwhm)
        full_trunc_peak_area = add_Poisson_error(truncated_peak_area + nom(bg_area))
        truncated_net_peak_areas.append(full_trunc_peak_area - bg_area)
    return truncated_net_peak_areas


def add_Poisson_error(count_rate: float | AffineScalarFunc) -> AffineScalarFunc:
    """
    Assume Poisson distribution, count_rate is the mean number of events counted,
    therefore a factor of np.sqrt(count_rate) will be added onto the error of the output.
    """
    return count_rate + Variable(0.0, np.sqrt(nom(count_rate)))


def discoverable(count: AffineScalarFunc, *, num_sigmas=3) -> bool:
    """
    Parameters
    ----------
    num_sigmas:
    In this case,
    3-sigmas = 99.7% chance of rejecting an undetectible peak, i.e.
    (true negatives)/(true negatives + false positives) = 99.7%.
    """
    return num_sigmas * count.s < count.n
