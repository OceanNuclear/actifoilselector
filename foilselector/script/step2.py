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
from collections import defaultdict
from tqdm import tqdm
from numpy import array as ary
from uncertainties import nominal_value as nom

from foilselector.foldermanagement import *
from foilselector.reactionnaming import (
    specify_isotopic_composition,
    commonname_to_atnum_massnum,
)
from foilselector.openmcextension import * # collapse_single_xs
from foilselector.openmcextension.table import Tab1DExtended
from foilselector.openmcextension.extended_io import serialize_radiation_dict, deserialize_radiation_dict
from foilselector.constants import BARN
from foilselector.generic import sorted_dict
from foilselector.simulation.decay.bateman import mat_exp_num_decays
from foilselector.simulation.decay import linearize_decay_chain, build_decay_chain_tree

default_gamma_energy_limits_keV = [20, 4600]


def load_relevant_xs_and_decay_info(composition_dict: dict, libraries: list[Path]) -> tuple[dict, dict]:
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
    foil_composition:dict,
    xs_dict: dict,
    decay_dict: dict,
    gs_array:np.ndarray,
    a, b, c
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

    Returns
    -------
    this_foil: dict[DiscreteRadiation, np.ndarray[float]]
        cross-section inducing each gamma-line, indexed by the gamma-line (energy and
        intensity) and its source, stored as a DiscreteRadiation.

    this_background: dict[ContinuousRadiationDistribution, np.ndarray[float]]
        cross-section inducing each gamma-continuum, indexed by the gamma-continuum and
        its source, stored as a ContinuousRadiationDistribution.
    """

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
                leave=False
            ):
            decay_pathways = linearize_decay_chain(build_decay_chain_tree(decay_dict, rx_name.split("-")[1]))
            for pathway in decay_pathways:
                # calculate how many decays of PRODUCT are measured per REACTANT ATOM
                # initially present in the foil when irradiated by 1 cm^-2 s^-1 flux
                # in each bin from time t=0-a seconds, and then measured from time
                # t= b-c seconds.
                decay_correction_factor = mat_exp_num_decays(
                    pathway.branching_ratios,
                    pathway.decay_constants,
                    a, b, c
                )
                scaled_collapsed_xs = (
                    collapse_single_xs(rx_xs, gs_array)
                    * BARN
                    * atomic_fraction
                    * decay_correction_factor
                )
                for peak_energy, peak_intensity, source in pathway.discrete_photon_spectrum:
                    this_foil[(peak_energy, source)] += scaled_collapsed_xs * peak_intensity
                for background_dist, source in pathway.background_photon_spectrum:
                    # the intensity correction value of the background is already built into background_dist
                    this_background[(background_dist, source)] += scaled_collapsed_xs
    return this_foil, this_background

def main(
    composition: dict,
    libraries: list[Path],
    irradiation_duration: float,
    transit_duration: float,
    measurement_duration: float
):

    # stage 1: read outputs of step1.
    gs_array = read_gs(".gs.csv")
    # stage 2: break down the foil composition into its consituent isotopes.
    with open(composition) as j:
        _composition_used_here = json.load(j)
    processed_composition = {
        foil_name: specify_isotopic_composition(foil_comp)
        for foil_name, foil_comp in _composition_used_here.items()
    }
    # save a version of the processed_composition dictionary

    save_atomic_composition_json(processed_composition)  # for future reference

    xs_dict, decay_dict = load_relevant_xs_and_decay_info(processed_composition, libraries)
    # TODO: rewrite this stage without openmc, and possibly implement an alternative using FISPACT-II.


    every_foil_response_matrix = {}
    for foil_name, foil_comp in tqdm(processed_composition.items(), desc="Processing each foil individually"):
        this_foil, this_background = calculate_response_matrix(
            foil_comp,
            xs_dict,
            decay_dict,
            gs_array,
            irradiation_duration,
            irradiation_duration+transit_duration,
            irradiation_duration+transit_duration+measurement_duration,
        )

        every_foil_response_matrix[foil_name] = sorted_dict(this_foil)
        # can't sort gamma-continua against each other, so no sorting on this_background.
        every_foil_background[foil_name] = this_background

    # Store response matrices and background spectra response matrices
    print("Writing to response matrix...", end="\r")
    with open(".response_matrices.json", "w") as j:
        json.dump(serialize_radiation_dict(every_foil_response_matrix), j)
    print("Written to response matrix, writing to background radiation...", end="\r")
    with open(".response_matrices.json", "w") as j:
        json.dump(serialize_radiation_dict(every_foil_response_matrix), j)
    print("Written response and background radiation, Done!")

