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
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy import array as ary
from tqdm import tqdm
from uncertainties import nominal_value as nom

from foilselector.constants import BARN, keV
from foilselector.foldermanagement import (
    BG_RESPONSE_MATRICES,
    EFFECTIVE_RESPONSE_MATRICES,
    RAW_RESPONSE_MATRICES,
    append_to_csv,
    append_to_json,
    get_gamma_spec_directory,
    read_apriori,
    read_gs,
    save_atomic_composition_json,
)
from foilselector.generic import sorted_dict
from foilselector.openmcextension.extended_io import (
    reactions_matching,
    serialize_radiation_dict,
    serialize_radiation_list,
    sparsely_load_xs_and_decay_dict,
)
from foilselector.openmcextension.library_reader import (
    ContinuousRadiationDistribution,
    DiscreteRadiation,
    collapse_single_xs,
)
from foilselector.optimizer.accuracy import get_accuracy_upper_bound
from foilselector.optimizer.choose_mass import (
    choose_num_reactant_in_foil,
    mass_from_num_atoms,
    max_num_counts,
)
from foilselector.optimizer.precision import (
    get_precision_contributions,
    get_precision_unit,
    get_precision_weight_vector,
)
from foilselector.reactionnaming import (
    commonname_to_atnum_massnum,
    specify_isotopic_composition,
)
from foilselector.simulation.compton import (
    ComptonToPeakRatioCurve,
    PeakToComptonCoefficients,
)
from foilselector.simulation.decay import build_decay_chain_tree, linearize_decay_chain
from foilselector.simulation.decay.bateman import mat_exp_num_decays
from foilselector.simulation.efficiency import EfficiencyCurve, find_efficiency_file
from foilselector.simulation.resolution import (
    ResolutionMaxCountRate,
    resolution_curve_factory,
)
from foilselector.simulation.spectral_simulation import (
    corresponding_background_level,
    delete_peaks,
    discoverable,
    fold_background,
    integrate_peak_area,
    merge_peaks,
    plot_spectrum,
    simulate_full_spectrum,
    simulate_peaks_with_uncertainties,
)


def load_relevant_xs_and_decay_info(
    composition_dict: dict,
    libraries: list[Path],
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
        for iso in isotopes:
            atnum_and_massnum = commonname_to_atnum_massnum(iso)
            isotope_of_interest.add(atnum_and_massnum)
    # stage 3.2: extract them
    return sparsely_load_xs_and_decay_dict(isotope_of_interest, libraries)


def calculate_response_matrix(
    foil_composition: dict,
    xs_dict: dict,
    decay_dict: dict,
    gs_array: np.ndarray,
    a: float,
    b: float,
    c: float,
    efficiency_curve: EfficiencyCurve,
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
        Efficiency v.s. gamma-ray energy curve.

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

        Returns
        -------
        :
            A blank row (placeholder in the response matrix).
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
                build_decay_chain_tree(decay_dict, rx_name.split("-")[1]),
            )
            for pathway in decay_pathways:
                # calculate how many decays of PRODUCT are measured per REACTANT ATOM
                # initially present in the foil when irradiated by 1 cm^-2 s^-1 flux
                # in each bin from time t=0-a seconds, and then measured from time
                # t= b-c seconds.
                decay_correction_factor = mat_exp_num_decays(
                    pathway.branching_ratios,
                    pathway.decay_constants,
                    a,
                    b,
                    c,
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
                            line.energy,
                        )

                    for continuum in pathway.background_photon_spectrum:
                        # the intensity correction value of the background is already built into background_dist
                        this_background[
                            ContinuousRadiationDistribution(
                                continuum.distribution.apply_scaling(efficiency_curve),
                                path_string + " " + continuum.source,
                            )
                        ] += scaled_collapsed_xs * nom(decay_correction_factor)
    return this_foil, this_background


class ScriptExecutedOutOfOrderError(RuntimeError):
    pass


def read_inputs(
    directory: Path,
    irradiation_duration: float,
    measurement_duration: float,
) -> tuple[
    np.ndarray,
    np.ndarray[float],
    Callable[[float | np.ndarray], float | np.ndarray],
    float,
    EfficiencyCurve,
    ComptonToPeakRatioCurve,
]:
    """
    Parameters
    ----------
    directory:
        The working directory

    Raises
    ------
    ScriptExecutedOutOfOrderError
        script not executed int he order of: 1. input 2. simulate 3. examine.
    """
    try:
        gs_array = read_gs()
        _apriori_flux, apriori_fluence = read_apriori(directory, irradiation_duration)
    except FileNotFoundError as e:
        raise ScriptExecutedOutOfOrderError(
            "Missing group structure and flux. Please complete the input step first!",
        ) from e

    try:
        resolution_coefficients, max_count_rate = ResolutionMaxCountRate.load(directory)
    except FileNotFoundError as e:
        raise ScriptExecutedOutOfOrderError(
            "Missing resolution and maximum count rate information,"
            " please complete the input step properly!",
        ) from e
    resolution_curve = resolution_curve_factory(resolution_coefficients)
    max_counts_per_foil = max_num_counts(max_count_rate, measurement_duration)

    try:
        eff_curve = EfficiencyCurve.from_file(find_efficiency_file(directory))
    except StopIteration as e:
        raise ScriptExecutedOutOfOrderError(
            "Missing efficiency file, please complete the input step properly!",
        ) from e

    try:
        compton_from_peak = ComptonToPeakRatioCurve(
            PeakToComptonCoefficients.load(directory),
        )
    except FileNotFoundError as e:
        raise ScriptExecutedOutOfOrderError(
            "Missing peak-to-Compton ratio curve,"
            " please complete the input step properly!",
        ) from e

    return (
        gs_array,
        apriori_fluence,
        resolution_curve,
        max_counts_per_foil,
        eff_curve,
        compton_from_peak,
    )


def main(  # TODO @OceanNuclear: PLR0914, PLR0915; need refactor.
    composition: Path,
    libraries: list[Path],
    irradiation_duration: float,
    transit_duration: float,
    measurement_duration: float,
    gamma_spectrum_parameters: list[float],
) -> tuple[
    dict[str, float],
    dict[str, np.ndarray],
    dict[str, list[DiscreteRadiation]],
    dict[str, float],
    dict[str, float],
]:
    """Main script of step2:simulate.

    Parameters
    ----------
    composition:
        Path to the composition JSON file, used by specify_isotopic_composition.
    libraries:
        Path to the nuclear data libraries.
    irradiation_duration:
        Length of time (in seconds) when the samples is expected to be irradiated
        continuously.
    transit_duration:
        Length of time (in seconds) expected to span between the end of the irradiation
        and the start of the gamma-ray spectrum acquisition.
    measurement_duration:
        Length of time (in seconds) of the gamma-ray spectrum acquisition.
    gamma_spectrum_parameters:
        Either a 2-tuple or 3-tuple.
        The first two parameters are the lowest and highest energies of gamma-rays
        (in keV) that the gamma-ray detector can record.

    Returns
    -------
    mass_record:
        Dictionary of recommended masses used by each foil.

    effective_foil_matrices:
        Dictionary of response matrix for generating each visible peak on the gamma-ray
        spectrum for each foil.

    effective_foil_peaks:
        Dictionary of list of visible peaks on the gamma-ray spectrum for each foil.

    foil_precision:
        Dictionary of each foil's contribution to the precision score.

    foil_accuracy:
        Dictionary of each foil's contribution to the accurcy score.
    """  # noqa: D401
    # Stage 1.0: load data from last step.
    cwd = Path.cwd()
    gspec_directory = get_gamma_spec_directory(cwd)

    (
        gs_array,
        apriori_fluence,
        resolution_curve,
        max_counts_per_foil,
        eff_curve,
        compton_from_peak,
    ) = read_inputs(cwd, irradiation_duration, measurement_duration)
    w_vector = get_precision_weight_vector(gs_array)
    precision_unit = get_precision_unit()

    # Stage 1.1: preparation of the gamma-ray spectrum simulation energies, and
    # the broadening matrix.
    gamma_simulation_energies_keV = np.arange(*gamma_spectrum_parameters)
    gamma_simulation_energies = gamma_simulation_energies_keV * keV

    # stage 2: break down the foil composition into its consituent isotopes.
    with Path(composition).open() as j:
        _composition_used_here = json.load(j)
    processed_composition = {
        foil_name: specify_isotopic_composition(foil_comp)
        for foil_name, foil_comp in _composition_used_here.items()
    }
    # stage 2.2: save a version of the processed_composition dictionary
    save_atomic_composition_json(
        processed_composition,
        directory=cwd,
    )  # for future reference

    # stage 3: get nuclear data
    xs_dict, decay_dict = load_relevant_xs_and_decay_info(
        processed_composition,
        libraries,
    )

    # stage 4.1: for each foil, optimize mass, and then whittle down radiation list.
    mass_record = {}
    effective_foil_matrices, effective_foil_peaks = {}, {}
    foil_precision_cont, foil_accuracy = {}, {}
    Path(cwd, RAW_RESPONSE_MATRICES).unlink(missing_ok=True)
    Path(cwd, BG_RESPONSE_MATRICES).unlink(missing_ok=True)
    Path(cwd, EFFECTIVE_RESPONSE_MATRICES).unlink(missing_ok=True)
    Path(cwd, "each_foil.csv").unlink(missing_ok=True)
    for foil_name, foil_comp in (
        pbar := tqdm(
            processed_composition.items(),
            desc="Processing each foil individually",
        )
    ):
        pbar.set_postfix_str(foil_name)
        this_foil, this_background = calculate_response_matrix(
            foil_comp,
            xs_dict,
            decay_dict,
            gs_array,
            irradiation_duration,
            irradiation_duration + transit_duration,
            irradiation_duration + transit_duration + measurement_duration,
            eff_curve,
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
        append_to_json(
            serialize_radiation_dict({foil_name: final_response_matrix}),
            json_path=Path(cwd, RAW_RESPONSE_MATRICES),
        )

        append_to_json(
            serialize_radiation_dict({foil_name: final_background}),
            json_path=Path(cwd, BG_RESPONSE_MATRICES),
        )
        mass_record[foil_name] = {
            "number of atoms": foil_num_atoms,
            "mass (g)": mass_from_num_atoms(foil_num_atoms, foil_comp),
        }
        # stage 4.2: calculate all gamma-peaks (and the response matrix for that).
        full_peak_list = simulate_peaks_with_uncertainties(
            final_response_matrix,
            apriori_fluence,
        )
        # stage 4.3: (final_response_matrix, full_peak_list) -merge-delete-> (detectible_peaks, detectible_response_matrix)
        detectible_peaks, detectible_response_matrix = merge_peaks(
            full_peak_list,
            resolution_curve,
            final_response_matrix,
        )
        detectible_peaks, detectible_response_matrix = delete_peaks(
            detectible_peaks,
            detectible_response_matrix,
            gamma_range=gamma_spectrum_parameters[0:2],
            exclusion_zone_511=0.0,
        )
        folded_bg = fold_background(final_background, apriori_fluence)
        background_levels = corresponding_background_level(
            full_peak_list,
            folded_bg,
            compton_from_peak,
            resolution_curve,
            test_energies=np.array([nom(peak.energy) for peak in detectible_peaks]),
            include_uncertainties=True,
        )
        net_peak_areas = integrate_peak_area(
            detectible_peaks,
            resolution_curve,
            background_levels,
        )

        # stage 4.4: (detectible_peaks, detectible_response_matrix) -merge-delete-> (effective_matrix, reaction_info)
        effective_matrix, reaction_info = [], []
        for net_area, (peak, xs) in zip(
            net_peak_areas,
            detectible_response_matrix.items(),
            strict=False,
        ):
            if discoverable(net_area):
                effective_matrix.append(nom(peak.intensity) * xs)
                reaction_info.append(
                    DiscreteRadiation(peak.energy, net_area, peak.source),
                )

        effective_matrix = np.array(effective_matrix, dtype=float)
        # During merging, a mixing ratio is decided upon.
        # This mixing ratio is only correct if neutron spectrum == a priori.
        # Hence, if we want to reload the effective response matrix during the
        # experiment, the best course of action is to reconstruct from
        # RAW_RESPONSE_MATRICES + BG_RESPONSE_MATRICES
        effective_foil_matrices[foil_name] = effective_matrix
        effective_foil_peaks[foil_name] = reaction_info

        append_to_json(
            serialize_radiation_dict({foil_name: effective_matrix}),
            json_path=Path(cwd, EFFECTIVE_RESPONSE_MATRICES),
        )
        foil_precision_cont[foil_name] = get_precision_contributions(
            effective_matrix,
            ary([peak.intensity for peak in reaction_info]),
            w_vector,
        )
        foil_accuracy[foil_name] = get_accuracy_upper_bound(
            effective_matrix,
            reaction_info,
        )
        append_to_csv(
            foil_name,
            {
                "recommended number of atoms": mass_record[foil_name]["number of atoms"],
                "recommended mass (mg)": mass_record[foil_name]["mass (g)"] * 1000,
                f"sensitivity ({precision_unit})": foil_precision_cont[foil_name].sum(),
                f"specificity upper limit (max value = {len(gs_array)})": foil_accuracy[
                    foil_name
                ],
                "number of detectable peaks": len(reaction_info),
            },
            csv_path=Path(cwd, "each_foil.csv"),
        )

        if not Path(gspec_directory, foil_name + ".pdf").exists():
            gspec_directory.mkdir(exist_ok=True)
            # simulating it at such high resolution will blow up the RAM,
            # so we'll have to dump the data as we create them, keeping memory usage low.
            spectrum = simulate_full_spectrum(
                full_peak_list,
                folded_bg,
                compton_from_peak,
                resolution_curve,
                gamma_simulation_energies,
            )
            with Path(gspec_directory, foil_name + ".json").open("w") as j:
                json.dump(
                    {
                        "energy (keV)": gamma_simulation_energies_keV.tolist(),
                        "spectrum": (spectrum * keV).tolist(),
                        "radiation": serialize_radiation_list(reaction_info),
                    },
                    j,
                )
            if spectrum.sum() > 0:  # only bother to create the plot if counts=non-zero.
                ax = plot_spectrum(
                    gamma_simulation_energies_keV,
                    spectrum,
                    peak_labels=reaction_info,
                )
                ax.set_title(
                    foil_name + f"\n(foil mass= {mass_record[foil_name]['mass (g)']} g)",
                )
                ax.get_figure().set_size_inches(20, 12)
                plt.savefig(Path(gspec_directory, foil_name + ".pdf"))
                plt.close()

    # stage 4.3: Store response matrices and background spectra response matrices

    return (
        mass_record,
        effective_foil_matrices,
        effective_foil_peaks,
        foil_precision_cont,
        foil_accuracy,
    )
