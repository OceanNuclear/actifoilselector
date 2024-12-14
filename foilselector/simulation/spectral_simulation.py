"""Simulates the gamma-ray spectrum, including how the peak broadens."""

from collections import defaultdict
from collections.abc import Callable
import warnings

import scipy
import numpy as np
from uncertainties import nominal_value as nom
from foilselector.constants import keV, FWHM_SIGMA
from foilselector.openmcextension.library_reader import (
    DiscreteRadiation,
    ContinuousRadiationDistribution,
)
from uncertainties.core import Variable, AffineScalarFunc
import matplotlib.pyplot as plt
from foilselector.simulation.compton import compton_edge, make_sharp_compton_distribution


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
    Simulate the irradiation by the fluence, and include the uncertainties in both the
    energy and number of counts in each peak.

    Returns
    -------
    radiations:
        A list of peaks (DiscreteRadiation) (which may be degenerate, i.e. multiple peaks
        can have the same energy at this stage) expected from a priori irradiation by the
        specified duration, where the energy and counts of each peak is specified only.
    """
    matrix = np.array(list(discrete_response_matrix.values()), dtype=float)
    if len(matrix) == 0:
        return []
    peaks_multiplier = matrix @ apriori_fluence

    radiations = []
    for line, multiplier in zip(list(discrete_response_matrix.keys()), peaks_multiplier):
        radiations.append(
            DiscreteRadiation(line.energy, multiplier * line.intensity, line.source)
        )
    return radiations


def merge_peaks(
    peak_list: list[DiscreteRadiation],
    resolution_curve: Callable[[float | np.ndarray], float | np.ndarray],
    full_response_matrix: dict[DiscreteRadiation, np.ndarray],
) -> tuple[list[DiscreteRadiation], dict[DiscreteRadiation, np.ndarray]]:
    """
    Merge peaks that are close together.
    Among each group of peak that gets merged into a single resultant peak, any one pair
    of adjacent peaks must satisfy the criteria laid out in peaks_are_distinct; but the
    peaks_are_distinct(peak_group[0], peak_group[-1]) may evaluate to False, i.e. the
    first peak and final peak may not actually be close enough to each other if they're so far away.
    We will write down a list of such offenders below.

    Parameters
    ----------
    peak_list:
        The list of DiscreteRadiation (gamma-ray energy and gamma-ray energy) that
        should've been created (each via a unique decay pathway) when the foil is
        irradiated by the a priori neutron spectrum, according to the nuclear data
        libraries.
    """
    if len(peak_list) <= 1:
        return [peak.copy() for peak in peak_list], {
            rad.copy(): response.copy() for rad, response in full_response_matrix.items()
        }

    response_matrix_list = list(full_response_matrix.items())
    # containers
    merged_peaks, merged_response_matrix = [], {}
    buffer, index_buffer = [peak_list[0].copy()], [0]
    # standard lengths
    len_response = len(response_matrix_list[0][1])

    def clear_buffers():
        """
        Private function to wrap up the content of the buffer and add them to the
        merged_peaks, merged_response_matrix queues.
        """
        nonlocal buffer, index_buffer
        nonlocal merged_peaks, merged_response_matrix
        if len(buffer) == 1:
            merged_peaks.append(buffer.pop())
            i = index_buffer.pop()
            radiation, response = response_matrix_list[i]
            merged_response_matrix[radiation.copy()] = response.copy()
        else:
            peak_list, response_dict = merge_peak_group(
                buffer,
                [response_matrix_list[i][0] for i in index_buffer],
                [response_matrix_list[i][1] for i in index_buffer],
                len_response,
                resolution_curve,
            )
            merged_peaks.extend(peak_list)
            merged_response_matrix.update(response_dict)

        buffer, index_buffer = [], []
        return

    # iterate through the whole list
    for j, peak in enumerate(peak_list[1:]):
        if peaks_are_distinct(
            peak, buffer[-1], fwhm_to_sigma(resolution_curve(nom(peak.energy)))
        ):
            clear_buffers()
        buffer.append(peak.copy())
        index_buffer.append(j + 1)
    clear_buffers()  # final buffer clearing
    return merged_peaks, merged_response_matrix


def merge_peak_group(
    peak_buffer: np.ndarray[DiscreteRadiation],
    radiations: np.ndarray[DiscreteRadiation],
    responses: np.ndarray[np.ndarray[float]],
    len_response: int,
    resolution_curve: Callable[[float | np.ndarray], float | np.ndarray],
) -> tuple[list[DiscreteRadiation], dict[DiscreteRadiation, np.ndarray[float]]]:
    """
    Merge a list of (possibly interfering) peaks into a list of mutually separaable
    peaks, which is shorter in length by at least one.

    Parameters
    ----------
    peak_buffer:
        list of peaks sorted by energies (ascending).
    radiations:
        list of peaks sorted by energies (ascending); same as peak_buffer, but
        may differ in intensity.
    responses:
        response matrix, each row matching `radiations`.

    Returns
    -------
    merged_peaks:
        A list of radiations, shorter than peak_buffer, with minimum length = 1.
    merged_response_matrix:
        A dictionary (each key = peak that the response row is suppose to represeent).
        Matching the length of merged_peaks.
    """
    merged_peaks, merged_response_matrix = [], {}

    # check meticulously for any roots in the gradient plot.
    gradient = gradient_factory(peak_buffer, resolution_curve)
    checked_energies = np.linspace(
        nom(peak_buffer[0].energy), nom(peak_buffer[-1].energy)
    )
    gradient_samples = gradient(checked_energies)
    grad_signs = np.sign(gradient_samples)
    # The two conditions required to trigger a "new peak"
    derivative_crosses_zero = np.diff(grad_signs) > 0
    previously_negative = grad_signs[:-1] == -1

    # separate into 3 list of lists, each nested list allows peak_list --merge--> peak.
    cuts = np.where(np.logical_and(derivative_crosses_zero, previously_negative))[0]
    lower_bound = np.hstack([nom(peak_buffer[0].energy) - 1, checked_energies[cuts]])
    upper_bound = np.hstack([checked_energies[cuts], nom(peak_buffer[-1].energy) + 1])
    mean_E_array = np.array([nom(peak.energy) for peak in peak_buffer])

    for low, upp in zip(lower_bound, upper_bound):
        chosen_peaks = np.logical_and(low <= mean_E_array, mean_E_array < upp)
        if any(chosen_peaks):
            chosen_slice = mask_to_slice(chosen_peaks)
            this_merged_peak, this_merged_row = merge_peaks_and_responses(
                peak_buffer[chosen_slice],
                radiations[chosen_slice],
                responses[chosen_slice],
                len_response,
            )
            merged_peaks.append(this_merged_peak)
            merged_response_matrix[this_merged_peak] = this_merged_row
    return merged_peaks, merged_response_matrix


def mask_to_slice(mask: np.ndarray[bool]) -> slice:
    """
    Turn a mask consisting of a SINGLE, contiguous block of Trues surrounded by Falses,
    into into a slice object.
    """
    boolean_as_int = np.diff(np.array(np.hstack([False, mask, False]), dtype=int))
    start_index = np.where(boolean_as_int == +1)[0][0]
    end_index = np.where(boolean_as_int == -1)[0][0]
    return slice(start_index, end_index)


def merge_peaks_and_responses(
    peak_list: list[DiscreteRadiation],
    radiations: list[DiscreteRadiation],
    responses: list[np.ndarray[float]],
    len_response: int,
) -> tuple[DiscreteRadiation, np.ndarray[float]]:
    """
    Merge a big group of peaks as a single peak.
    This differ from combine_as_single_peak by also merging the response.

    Parameters
    ----------
    peak_list:
        list
    """
    this_peak = combine_as_single_peak(peak_list)
    this_row_response = np.zeros(len_response, dtype=float)
    normalization_factor = nom(this_peak.intensity)
    if normalization_factor:
        for rad, resp in zip(radiations, responses):
            this_row_response += nom(rad.intensity) / normalization_factor * resp
    # else: this_row_response = np.zeros(len_response)
    return this_peak, this_row_response


def gradient_factory(
    peak_list: list[DiscreteRadiation],
    resolution_curve: Callable[[float | np.ndarray], float | np.ndarray],
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """
    A function factory that returns a function that calculates the gradient.
    """
    curve_container = []
    for peak in peak_list:
        curve_container.append(
            gradient_contribution(
                nom(peak.energy),
                fwhm_to_sigma(resolution_curve(nom(peak.energy))),
                nom(peak.intensity),
            )
        )

    def total_gradient_calculator(x: float | np.ndarray) -> float | np.ndarray:
        """Function that takes in energy and output the gradient at that energy due to
        contributions from every single peak listed."""
        return np.sum([curve(x) for curve in curve_container], axis=0)

    return total_gradient_calculator


def gradient_contribution(mu: float, sigma: float, amplitude: float):
    """
    Function factory that calculates the contribution to gradient of a single peak.
    """
    gaussian = normal_dist_factory(mu, sigma)

    def gradient_calculator(x):
        """Functoin that outputs the gradient contribution from a single peak."""
        return amplitude * (mu - x) / sigma * gaussian(x)

    return gradient_calculator


def fwhm_to_sigma(fwhm: float) -> float:
    """
    Given FWHM of a peak, get the sigma (standard deviation) of the same peak, assuming
    it's a gaussian peak.
    """
    return fwhm / FWHM_SIGMA


def combine_as_single_peak(peak_group: list[DiscreteRadiation]) -> DiscreteRadiation:
    """
    Calculate the (weighted) average energy, and the total number of counts, of the new
    DiscreteRadiation line created by merging this group of peaks into a single line.

    Parameters
    ----------
    peak_group:
        list of peaks to be merged as a single peak.

    Returns
    -------
    :
        merged single peak.
    """
    # new interval must span the entire group's old interval.
    total_counts = sum(nom(p.intensity) for p in peak_group)
    if total_counts:  # not equal to zero
        peak_energy = sum(p.energy * nom(p.intensity) for p in peak_group) / total_counts
    else:
        # do an unweighted average
        total_peak_energy = sum(peak.energy for peak in peak_group)
        total_E = nom(total_peak_energy)
        peak_energy = total_peak_energy - total_E + total_E / len(peak_group)
    return DiscreteRadiation(
        peak_energy,
        sum(p.intensity for p in peak_group),
        "; ".join([f"{p.source} at {nom(p.energy)}" for p in peak_group]),
    )


def peaks_are_distinct(
    peak_1: DiscreteRadiation, peak_2: DiscreteRadiation, sigma: float
) -> bool:
    """
    Check if two peaks are distinct from each other (True), or shall be merged as one
    (False).

    Notes
    -----
    I have decided that any two peaks shall be considered distinct if their centroids
    are 2σ away from each other.
    Proof is in https://www.desmos.com/calculator/rxzhj0a0w0
    where, peaks are shown to be
    {clearly separatable when separation>2σ,
        clearly mergable when separation<2σ}
        when sum(peak2)/sum(peak1)=1.0;
    {clearly separatable when separation>2.2993σ
        clearly mergable when separation<1.987σ}
        when sum(peak2)/sum(peak1)=0.9;
    {clearly separatable when separation>2.1825σ
        clearly mergable when separation<1.997σ}
        when sum(peak2)/sum(peak1)=0.8;
    {clearly separatable when separation>2.40707σ
        clearly mergable when separation<1.967σ}
        when sum(peak2)/sum(peak1)=0.7;
    {clearly separatable when separation>2.51463σ
        clearly mergable when separation<1.9283σ}
        when sum(peak2)/sum(peak1)=0.6;
    {clearly separatable when separation>2.62751σ
        clearly mergable when separation<1.8498σ}
        when sum(peak2)/sum(peak1)=0.5.
    The point is, if we assert there is a magical separation value for ALL values of
    sum(peak2)/sum(peak1) ratios, below which peaks are 100% mergable and above which
    peaks are 100% separable, then we have ot choose 2.00 σ as that magical separation
    value.
    """
    if abs(nom(peak_2.energy) - nom(peak_1.energy)) < (2 * sigma):
        return False
    if nom(peak_1.energy) < nom(peak_2.energy):
        smaller_E, larger_E = peak_1.energy, peak_2.energy
    else:
        smaller_E, larger_E = peak_2.energy, peak_1.energy
    if (smaller_E.n + smaller_E.s) > (
        larger_E.n - larger_E.s
    ):  # if unc interval overlaps
        return False
    return True


def delete_peaks(
    peak_list: list[DiscreteRadiation],
    response_matrix: dict[DiscreteRadiation, np.ndarray],
    *,
    gamma_range: list[float] = [20.0, 2800.0],
    exclusion_zone_511: float = 0.5,
) -> tuple[list[DiscreteRadiation], dict[DiscreteRadiation, np.ndarray]]:
    """
    Delete peaks that are not useful in the analysis, even if they're detectible in theory.

    Parameters
    ----------
    peak_list, response_matrix:
        The list of detectible radiation and the associated response matrix that
        generates it.
    gamma_range:
        The range of detectible gamma-ray energies [keV].

    exclusion_zone_511:
        The energy range around 511 keV, any peaks that falls within
        510.998950 keV +/- exclusion_zone_511 keV would be deleted
    """
    new_peak_list, new_response_matrix = [], {}
    for peak, (radiation, reaction) in zip(peak_list, response_matrix.items()):
        if nom(peak.energy) < (gamma_range[0] * keV) or nom(peak.energy) > (
            gamma_range[1] * keV
        ):
            continue
        if exclusion_zone_511 and np.isclose(
            nom(peak.energy), 510.998950 * keV, atol=exclusion_zone_511 * keV, rtol=0.0
        ):
            continue
        new_peak_list.append(peak.copy())
        new_response_matrix[radiation.copy()] = reaction.copy()
    return new_peak_list, new_response_matrix


def fold_background(
    background: dict[ContinuousRadiationDistribution, np.ndarray[float]],
    apriori_fluence: np.ndarray,
) -> list[ContinuousRadiationDistribution]:
    """Similar to simulate_peaks_with_uncertainties, but with background instead."""
    matrix = np.array(list(background.values()), dtype=float)
    if len(matrix) == 0:
        return []
    bg_multipler = matrix @ apriori_fluence

    bgs = []
    for dist, multiplier in zip(list(background.keys()), bg_multipler):
        bgs.append(
            ContinuousRadiationDistribution(dist.distribution * multiplier, dist.source)
        )
    return bgs


def make_compton_distribution(
    photopeak_energy: AffineScalarFunc,
    test_energies: np.ndarray[float],
    resolution_curve: Callable[[float | np.ndarray], float | np.ndarray],
) -> np.ndarray[float]:
    """
    Create a normalized distribution that would properly simulate the Compton continuum,
    as it includes the blurring of the Compton edge.

    Parameters
    ----------
    photopeak_energy:
        the energy of the gamma ray that's causing this Compton continuum.
    test_energies:
        The array of energies for which we have to compute the Compton continuum for.
    resolution_curve:
        Function that takes resolution curve as input and outputs -> FWHM at that energy.
    """
    compton_dist = make_sharp_compton_distribution(photopeak_energy, test_energies)
    Eg = nom(photopeak_energy)
    Ecomp = compton_edge(Eg)
    sigma = fwhm_to_sigma(resolution_curve(Ecomp))
    sigma_at_lower_bound = fwhm_to_sigma(resolution_curve(Ecomp - 6 * sigma))

    if not np.isclose(sigma_at_lower_bound, sigma, rtol=0.4, atol=0):
        warnings.warn(
            "The kernel width changes too much over the smearing area!"
            "Simulated gamma-spectrum may yield an inaccurate Compton continuum."
        )
    smearing_range = np.logical_and(
        test_energies >= (Ecomp - 6 * sigma), test_energies <= (Ecomp + 10 * sigma)
    )
    smearing_scaler = get_smeared_multiplier(
        (test_energies[smearing_range] - Ecomp) / sigma
    )
    rhs_of_smearing_range = np.logical_and(test_energies > Ecomp, smearing_range)
    compton_dist[rhs_of_smearing_range] = make_sharp_compton_distribution(
        photopeak_energy, np.array([Ecomp])
    )[0]
    compton_dist[smearing_range] = smearing_scaler * compton_dist[smearing_range]
    return compton_dist


def get_smeared_multiplier(displacement_in_terms_of_sigma):
    """
    Convolving a step function that is +1 at x<0, 0 at x>0, with a normal distribution
    (with unit area) of standard deviation = sigma.
    """
    return scipy.special.erf(-displacement_in_terms_of_sigma / np.sqrt(2)) / 2 + 0.5


def get_broadening_matrix(
    resolution_curve: Callable[[float | np.ndarray], float | np.ndarray],
    test_energies: np.ndarray[float],
) -> np.ndarray[float]:
    """
    A quick way to simulate Gaussian broadening due to the resolution limit of the
    detector without using integration, by simply broadening from the curent sampled
    points on to their neighbouring points.

    Parameters
    ----------
    test_energies:
        energies where we want to evaluate the Gaussian broadening matrix's points over,
        unit: [eV].
    """
    n = len(test_energies)
    weights = np.zeros([n, n])
    # Normal distributions lying in the COLUMN direction. Every new column = a new normal distribution.
    for i in range(n):
        mu, sigma = test_energies[i], fwhm_to_sigma(resolution_curve(test_energies[i]))
        normal = normal_dist_factory(mu, sigma)(test_energies)
        # Normalize each column
        weights[:, i] += normal / normal.sum()
    return weights


def corresponding_background_level(
    peak_list: list[DiscreteRadiation],
    folded_background: list[ContinuousRadiationDistribution],
    compton_peak_ratio_curve: Callable[
        [float | np.ndarray | AffineScalarFunc], float | np.ndarray | AffineScalarFunc
    ],
    resolution_curve: Callable[[float | np.ndarray], float | np.ndarray],
    test_energies: np.ndarray[float],
    *,
    # broadening_matrix: np.ndarray | None = None,
    include_uncertainties: bool = False,
) -> list[AffineScalarFunc]:
    """
    TODO: speed up
    Get the background heights at the test_energies.

    Parameters
    ----------
    peak_list:
        list of peaks, including their energy [eV] and intensity (counts).
    background:
        list of continuous peaks, including their energy and intensity.
    compton_peak_ratio_curve:
        Curve showing the Compton-to-peak ratio. See
        :class:`foilselector.simulation.compton.ComptonToPeakRatioCurve`.
    test_energies
        Places on the gamma-ray energy axis where we want to calculate the background
        heights at. (unit: eV)
    include_uncertainties:
        Whether the outputted bg_heights should include uncertainties or not.

    Returns
    -------
    bg_heights:
        list of background levels at the test_energies, given in unit [counts/eV].
    """
    bg_heights = np.zeros(
        len(test_energies), dtype=object if include_uncertainties else float
    )

    for peak_zeta in peak_list:
        c_counts = compton_peak_ratio_curve(peak_zeta.energy) * peak_zeta.intensity
        comp_dist = make_compton_distribution(
            peak_zeta.energy, test_energies, resolution_curve
        )

        bg_heights += comp_dist * (c_counts if include_uncertainties else nom(c_counts))
    for dist, _source in folded_background:
        bg_heights += dist(test_energies)

    return bg_heights


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
    return scipy.special.erf(standard_score / np.sqrt(2))


def simulate_full_spectrum(
    peak_list: list[DiscreteRadiation],
    folded_background: list[ContinuousRadiationDistribution],
    compton_peak_ratio_curve: Callable[
        [float | np.ndarray | AffineScalarFunc], float | np.ndarray | AffineScalarFunc
    ],
    resolution_curve: Callable[[float | np.ndarray], float | np.ndarray],
    sampling_points: np.ndarray[float],
    # broadening_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    resolution_curve:
        The curve that takes in the centroid energy [eV] of the peak and outputs the
        FWHM (width) of the peak [eV].
    sampling_points:
        gamma-ray energies [eV] where we want to know the count density [1/eV].
    # broadening_matrix:
    #     See get_broadening_matrix. Shape must match sampling_points^2.
    all other parameters:
        See corresponding_background_level

    Returns
    -------
    spectrum:
        count density [1/eV] at the specified sampling_points
    """
    spectrum = np.zeros_like(sampling_points)
    spectrum += corresponding_background_level(
        peak_list,
        folded_background,
        compton_peak_ratio_curve,
        resolution_curve,
        sampling_points,
        # broadening_matrix=broadening_matrix,
    )
    for peak in peak_list:
        spectrum += normal_dist_factory(
            nom(peak.energy),
            fwhm_to_sigma(resolution_curve(nom(peak.energy))),
            nom(peak.intensity),
        )(sampling_points)
    return spectrum


def plot_spectrum(
    sampling_points_keV: np.ndarray[float],
    spectrum: np.ndarray[float],
    peak_labels: list[DiscreteRadiation],
    *,
    plot_min: float = 0.05,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot the entire gamma-ray spectrum, in unit [1/keV], with a log y-axis.

    Parameters
    ----------
    sampling_points_keV:
        where the gamma-count per-keV is actually sampled. [keV]
    spectrum:
        gamma-count per-eV.
    peak_labels:
        list of DiscreteRadiation documenting the location and size of each peak,
        including their uncertainties and origin.
    ax:
        The axis on which the gamma-ray spectrum is getting plotted.
    plot_min:
        The lower bound of the y-axis if plotting in log-scale.

    Returns
    -------
    ax:
        plt.Axes object on which the spectrum is plotted.
    """
    ax = ax or plt.subplot()
    spectrum_new_scale = spectrum * keV
    ax.semilogy(sampling_points_keV, spectrum_new_scale)
    # plotting parameters
    y_max = max(spectrum_new_scale)
    log_data_height = np.log(y_max / plot_min)
    plot_max = y_max * np.exp(log_data_height * 0.1)  # 10% taller than max in log scale
    ax.set_ylim(*sorted([plot_min, plot_max]))
    for peak in peak_labels:
        E = nom(peak.energy)
        i = np.argmin(abs(sampling_points_keV * keV - E))
        x = sampling_points_keV[i]
        ytip = spectrum_new_scale[i] * np.exp(log_data_height * 0.1)
        yend = spectrum_new_scale[i] * np.exp(log_data_height * 0.2)
        ax.annotate(
            peak.plot_label_format(),
            xy=(x, ytip),
            xytext=(x, yend),
            arrowprops=dict(arrowstyle="->"),
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("counts /keV")
    ax.set_xlabel(r"$E_\gamma$ (keV)")
    return ax


def plot_in_sqrt_scale(
    sampling_points_keV: np.ndarray,
    spectrum: np.ndarray,
    peak_labels: list[DiscreteRadiation],
    *,
    ax: plt.Axes | None = None,
):
    """
    Plot the entire gamma-ray spectrum, in unit [1/keV], with a sqrt(counts) y-axis.

    Parameters
    ----------
    sampling_points_keV:
        where the gamma-count per-keV is actually sampled. [keV]
    spectrum:
        gamma-count per-eV.
    peak_labels:
        list of DiscreteRadiation documenting the location and size of each peak,
        including their uncertainties and origin.
    ax:
        The axis on which the gamma-ray spectrum is getting plotted.

    Returns
    -------
    ax:
        plt.Axes object on which the spectrum is plotted.
    """
    ax = ax or plt.subplot()
    ax.plot(sampling_points_keV, spectrum)
    ax.set_yscale(
        "function",
        functions=[
            lambda x: np.sign(x) * np.sqrt(np.abs(x)),
            lambda x: np.sign(x) * np.square(np.abs(x)),
        ],
    )
    ax.set_ylabel("counts /keV")
    ax.set_xlabel("gamma energy (keV)")
    y_max_sqrt = np.sqrt(spectrum.max())
    for peak in peak_labels:
        E = nom(peak.energy)
        i = np.argmin(abs(sampling_points_keV * keV - E))
        x = sampling_points_keV[i]
        ytip = (np.sqrt(spectrum[i]) + y_max_sqrt * 0.00) ** 2
        yend = (np.sqrt(spectrum[i]) + y_max_sqrt * 0.10) ** 2
        ax.annotate(
            peak.plot_label_format(),
            xy=(x, ytip),
            xytext=(x, yend),
            arrowprops=dict(arrowstyle="->"),
            ha="center",
            va="bottom",
        )
    return ax


def normal_dist_factory(
    mu0: float, sigma: float, area: float = 1.0
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """
    Function factory that makes normal distributions of the specified mean location,
    width, and height.
    """
    normal_dist_generated = scipy.stats.norm(loc=mu0, scale=sigma)

    def normal_dist_with_specified_params(x: float | np.ndarray) -> float | np.ndarray:
        """Returns that normal distribution evaluated at the required x."""
        return normal_dist_generated.pdf(x) * area

    return normal_dist_with_specified_params


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
        resolution curve that gives the FWHM [eV] at energy E [eV].
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
    counts = nom(count_rate)
    return count_rate + Variable(0.0, np.sqrt(np.clip(counts, 0, np.inf)))


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
