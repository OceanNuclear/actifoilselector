"""
The goal of this script is to read from a nuclear data library,
    extract all of the relevant information (decay data and neutron spectrum)
    and save them in a useful format for later processing.
It 'primes' the working directory with data, hence its name.

Use the following script if you
1. You have a reactor/ beamline with a constant power level (i.e. stable neutron spectrum)
2. You have foils made of more than 1 material that you can put into the neutron field to be activated.
3. You have at least one gamma detector to measure its radioactive decay products when it leaves.
____
A nuance that I have to clear up: if a(n advaned) user _knows_ that there's a specific
    natural-composition isotope (e.g. 'Cd0') entry in one of their --library folders,
    and would like to use that instead of adding up the constituent isotopes' microsopic cross-sections
        in the openmc-specified isotopic abundance ratios,
    then they must specify that isotope rather than the
        (e.g. 'Cadmium': {'Cd0':1.0} rather than 'Cadmium':{'Cd':1.0})

"""

import json
from pathlib import Path
from typing import TYPE_CHECKING
from collections import defaultdict
from tqdm import tqdm
from numpy import array as ary
from uncertainties import nominal_value as nom

from foilselector.foldermanagement import *
from foilselector.reactionnaming import (
    specify_isotopic_composition,
    commonname_to_atnum_massnum,
)
from foilselector.openmcextension import *  # collapse_single_xs
from foilselector.simulation.spectral_simulation import *
from foilselector.optimizer.choose_mass import *
from foilselector.optimizer.precision import get_precision_weight_vector, get_precision
from foilselector.optimizer.accuracy import get_accuracy
from foilselector.openmcextension.extended_io import serialize_radiation_dict
from foilselector.constants import BARN
from foilselector.generic import sorted_dict
from foilselector.simulation.decay.bateman import mat_exp_num_decays
from foilselector.simulation.decay import linearize_decay_chain, build_decay_chain_tree
from foilselector.openmcextension.library_reader import (
    DiscreteRadiation,
    ContinuousRadiationDistribution,
)

if TYPE_CHECKING:
    pass

default_gamma_energy_limits_keV = [20, 4600]


def load_relevant_xs_and_decay_info(
    composition_dict: dict, libraries: list[Path]
) -> tuple[dict, dict]:
    """
    Open the libraries, load in only the relevant cross-sections and decay data.

    Returns
    -------
    xs_dict: dict[str, openmc.data.Tabulated1D]
        See sparsely_load_xs_and_decay_dict
    decay_dict: dict[str, dict]
        See sparsely_load_xs_and_decay_dict
    """
    # stage 3.1: find what isotopes need to be extracted.
    isotope_of_interest = set()
    for isotopes in composition_dict.values():
        for iso in isotopes.keys():
            atnum_and_massnum = commonname_to_atnum_massnum(iso)
            isotope_of_interest.add(atnum_and_massnum)
    # stage 3.2: extract them
    return sparsely_load_xs_and_decay_dict(isotope_of_interest, libraries)


def calculate_response_matrix(
    foil_composition: dict,
    xs_dict: dict,
    decay_dict: dict,
    gs_array: np.ndarray,
    a,
    b,
    c,
) -> tuple[dict, dict]:
    """
    For each gamma-ray energy or gamma-ray continuum, sum up the cross-sections of
    all of the pathways that can create it. Store these gamma:cross-sections in two
    dictionaries, one for the discrete lines, another for the continua.

    Parameters
    ----------
    foil_composition: dict[str, float]
        fraction (float) of each isotope (name of isotope stored as a str) in the foil.
    xs_dict, decay_dict:
        see `sparsely_load_xs_and_decay_dict`
    gs_array:
        float array of shape (len(gs_array), 2) storing the lower and upper bounds of
        each neutron group's energy.
    a, b, c:
        Times when irradiation stops, acquisition starts, and acquisition stops.
        See `mat_exp_num_decays`.
    efficiency_curve:

    Returns
    -------
    this_foil: dict[DiscreteRadiation, np.ndarray[float]]
        cross-section inducing each gamma-line, indexed by the gamma-line (energy and
        intensity) and its source, stored as a DiscreteRadiation.

    this_background: dict[ContinuousRadiationDistribution, np.ndarray[float]]
        cross-section inducing each gamma-continuum, indexed by the gamma-continuum and
        its source, stored as a ContinuousRadiationDistribution.
    """

    # stage 4
    def xs_template_generator() -> np.ndarray[float]:
        """
        For creating an empty array representing the cross-section for generating a
        single count in whichever gamma-line of interest.
        """
        return np.zeros(len(gs_array), dtype=float)

    this_foil = defaultdict(xs_template_generator)
    this_background = defaultdict(xs_template_generator)
    for isotope, atomic_fraction in foil_composition.items():
        for rx_name, rx_xs in tqdm(
            reactions_matching(xs_dict, isotope).items(),
            desc=f"Calculating all possible reactions for reactant={isotope} and all decay pathways for each reaction.",
            leave=False,
        ):
            decay_pathways = linearize_decay_chain(
                build_decay_chain_tree(decay_dict, rx_name.split("-")[1])
            )
            for pathway in decay_pathways:
                # calculate how many decays of PRODUCT are measured per REACTANT ATOM
                # initially present in the foil when irradiated by 1 cm^-2 s^-1 flux
                # in each bin from time t=0-a seconds, and then measured from time
                # t= b-c seconds.
                decay_correction_factor = mat_exp_num_decays(
                    pathway.branching_ratios, pathway.decay_constants, a, b, c
                )
                if nom(decay_correction_factor):
                    scaled_collapsed_xs = (
                        collapse_single_xs(rx_xs, gs_array) * BARN * atomic_fraction
                    )
                    path_string = isotope + "+n->" + "->".join(pathway.names)
                    for line in pathway.discrete_photon_spectrum:
                        scaled_line = DiscreteRadiation(
                            line.energy,
                            line.intensity * decay_correction_factor,
                            path_string + " " + line.source,
                        )
                        this_foil[scaled_line] += scaled_collapsed_xs * efficiency_curve(
                            line.energy
                        )

                    for continuum in pathway.background_photon_spectrum:
                        # the intensity correction value of the background is already built into background_dist
                        this_background[
                            ContinuousRadiationDistribution(
                                continuum.distribution.apply_scaling(efficiency_curve),
                                # path_string+" "+
                                continuum.source,
                            )
                        ] += scaled_collapsed_xs * nom(decay_correction_factor)
    return this_foil, this_background


def main(
    composition: dict,
    libraries: list[Path],
    irradiation_duration: float,
    transit_duration: float,
    measurement_duration: float,
):
    cwd = Path.cwd()
    # stage 1: read outputs of step1.
    gs_array = read_gs(".gs.csv")
    w_vector = get_precision_weight_vector(gs_array)
    # stage 2: break down the foil composition into its consituent isotopes.
    with open(composition) as j:
        _composition_used_here = json.load(j)
    processed_composition = {
        foil_name: specify_isotopic_composition(foil_comp)
        for foil_name, foil_comp in _composition_used_here.items()
    }
    # save a version of the processed_composition dictionary

    save_atomic_composition_json(processed_composition)  # for future reference

    xs_dict, decay_dict = load_relevant_xs_and_decay_info(
        processed_composition, libraries
    )

    apriori_flux, apriori_fluence = get_apriori(cwd, irradiation_duration)
    resolution_coefficients, max_count_rate = ResolutionMaxCountRate.load()
    resolution_curve = resolution_curve_factory(resolution_curve_factory)
    max_counts_per_foil = max_num_counts(max_count_rate, measurement_duration)
    eff_curve = EfficiencyCurve.from_file(find_efficiency_file())
    compton_from_peak = Compton_to_peak_curve_factory(PeakToComptonCoefficients.load())

    # stage 4: get respones matrices.
    every_foil_response_matrix, every_foil_background, mass_record = {}, {}, {}
    effective_foil_matrices = {}
    foil_precision, foil_accuracy = [], []
    for foil_name, foil_comp in tqdm(
        processed_composition.items(), desc="Processing each foil individually"
    ):
        this_foil, this_background = calculate_response_matrix(
            foil_comp,
            xs_dict,
            decay_dict,
            gs_array,
            irradiation_duration,
            irradiation_duration + transit_duration,
            irradiation_duration + transit_duration + measurement_duration,
        )

        # can't sort gamma-continua against each other, so won't be sorting on background
        foil_num_atoms, final_response_matrix, final_background = (
            choose_num_reactant_in_foil(
                sorted_dict(this_foil),
                this_background,
                apriori_fluence,
                max_counts_per_foil,
                compton_from_peak,
            )
        )
        every_foil_response_matrix[foil_name] = final_response_matrix

        every_foil_background[foil_name] = final_background
        mass_record[foil_name] = {
            "number of atoms": foil_num_atoms,
            "mass (g)": mass_from_num_atoms(foil_num_atoms, foil_comp),
        }
        # stage 4.2: calculate only effective counts.
        full_peak_list = simulate_peaks_with_uncertainties(
            final_response_matrix, apriori_fluence
        )
        detectible_peaks, detectible_response_matrix = merge_delete_peaks(
            full_peak_list, resolution_curve, final_response_matrix
        )
        background_levels = corresponding_background_level(
            full_peak_list,
            fold_background(final_background, apriori_fluence),
            compton_from_peak,
            test_locations=detectible_peaks,
        )
        net_peak_areas = integrate_peak_area(
            detectible_peaks, resolution, background_levels
        )

        effective_matrix, reaction_info = [], []
        for net_area, (peak, xs) in zip(
            net_peak_areas, detectible_response_matrix.items()
        ):
            if discoverable(net_area):
                effective_matrix.append(nom(peak.intensity) * xs)
                reaction_info.append(
                    DiscreteRadiation(peak.energy, net_area, peak.source)
                )

        effective_matrix = np.array(effective_matrix, dtype=float)
        effective_foil_matrices[foil_name] = {
            "matrix": effective_matrix,
            "photons": reaction_info,
        }
        foil_precision[foil_name] = get_precision(
            effective_matrix,
            ary([1 / (peak.intensity.s) ** 2 for peak in reaction_info]),
            weight_vector,
        )
        foil_accuracy[foil_name] = get_accuracy(effective_matrix, reaction_info)

    # stage 4.3: Store response matrices and background spectra response matrices
    print("Writing to response matrix...", end="\r")
    with open(".response_matrices.json", "w") as j:
        json.dump(serialize_radiation_dict(every_foil_response_matrix), j)
    print("Written to response matrix, writing to background radiation...", end="\r")
    with open(".background_response_matrices.json", "w") as j:
        json.dump(serialize_radiation_dict(every_foil_background), j)
    print("Written response and background radiation, Done!              ")
    with open(".mass_records.json", "w") as j:
        json.dump(mass_record, j)
    with open(".effective_response_matrix.json", "w") as j:
        json.dump(effective_foil_matrices, j)

    return every_foil_response_matrix, every_foil_background, mass_record
