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
from tqdm import tqdm
from numpy import array as ary
from uncertainties import nominal_value as nom
from uncertainties.core import AffineScalarFunc, Variable

from foilselector.foldermanagement import *
from foilselector.reactionnaming import (
    specify_isotopic_composition,
    commonname_to_atnum_massnum,
)
from foilselector.openmcextension import * # collapse_single_xs
from foilselector.openmcextension.table import Tab1DExtended
from foilselector.simulation import EfficiencyCurve
from foilselector.constants import BARN
from foilselector.simulation.decay.bateman import mat_exp_num_decays
from foilselector.simulation.decay import linearize_decay_chain, build_decay_chain_tree

default_gamma_energy_limits_keV = [20, 4600]


class RadiationXSEncoder(json.JSONEncoder):
    """Encodes discrete radiation lines, radiation continua, and cross-sections."""
    def default(self, obj):
        if isinstance(obj, AffineScalarFunc):
            return str(obj)
        elif isinstance(obj, Tab1DExtended):
            return dict(x=obj.x, y=obj.y, inteprolation=obj.inteprolation)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (DiscreteRadiation, ContinuousRadiationDistribution)):
            return {k: self.default(v) for k, v in obj._asdict()}
        elif isinstance(obj, dict):
            # a dict of reactions and their xs: turn into list instead
            if len(obj)==0:
                return {}
            k0 = list(obj.keys())[0]
            if isinstance(k0, (DiscreteRadiation, ContinuousRadiationDistribution)):
                return [{type(k).__name__:self.default(k), "xs":self.default(v)} for k, v in obj.items()]
        return super().default(obj)

def serialize_radiation_dict(obj):
    """Turn radiation dict into something that can be saved as a JSON file."""
    if isinstance(obj, AffineScalarFunc):
        # AffineScalarFunc -> dict{'n':float, 's':float}
        return {"n":obj.n, "s":obj.s}
    elif isinstance(obj, np.ndarray):
        # np.ndarray -> list[float] | list[int]
        return obj.tolist()
    elif isinstance(obj, (DiscreteRadiation, ContinuousRadiationDistribution, Tab1DExtended)):
        # namedtuple | Tab1DExtended -> dict
        return {k: serialize_radiation_dict(v) for k, v in obj._asdict().items()}
    elif isinstance(obj, dict):
        # a dict of reactions and their xs: turn into list instead
        if len(obj)==0:
            return {}
        k0 = list(obj.keys())[0]
        if isinstance(k0, (DiscreteRadiation, ContinuousRadiationDistribution)):
            # dict -> list[{             'DiscreteRadiation' : dict, 'xs':list[float]}, ...]
            # dict -> list[{'ContinuousRadiationDistribution': dict, 'xs':list[float]}, ...]
            return [{type(k).__name__:serialize_radiation_dict(k), "xs":serialize_radiation_dict(v)} for k, v in obj.items()]
        # dict[str:'foil_name', dict] -> dict[str: 'foil_name', list]
        return {k:serialize_radiation_dict(v) for k,v in obj.items()}
    # str -> str
    return obj

def deserialize_radiation_dict(obj):
    """Turn JSON file back into radiation dict."""
    if isinstance(obj, dict):
        keys = obj.keys()
        if tuple(keys)==("n", "s"):
            return Variable(obj["n"], obj["s"])
        elif "DiscretRadiation" in keys:
            return {
                DiscreteRadiation(**obj["DiscreteRadiation"]):
                np.array(obj["xs"])
            }
        elif "ContinuousRadiationDistribution" in keys:
            return {
                ContinuousRadiationDistribution(**obj["ContinuousRadiationDistribution"]):
                np.array(obj["xs"])
            }
        elif sorted(keys)==sorted(Tab1DExtended._fields):
            return Tab1DExtended(x=obj["x"], y=obj["y"], interpolation=obj["interpolation"])
        return {k:deserialize(v) for k,v in obj.items()}

    if isinstance(obj, list):
        if isinstance(obj[0], (float, int)):
            return np.array(obj)
        # should be a list of len==2 dicts left at this stage.
        d = {}
        for item in obj:
            prev_len = len(d)
            d.update(deserialize_radiation_dict(item))
            if (prev_len+1)!=len(d):
                raise ValueError("Programmer error! List that was supposed to represent a dict contains repeated items.")
        return d
        return {deserialize_radiation_dict(k): deserialize_radiation_dict(v) for k, v in obj.items()}
        return {k: deserialize_radiation_dict(v) for k, v in obj.items()}
    return obj

class RadiationXSDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, object_hook=self.object_hook, **kwargs)
    def object_hook(self, obj):
        if isinstance(obj, dict):
            return

def serialize_item(item):
    if isinstance(item, str):
        return item
    elif isinstance(item, AffineScalarFunc):
        return str(item)
    elif isinstance(item, Tab1DExtended):
        return dict(x=item.x, y=item.y, inteprolation=item.inteprolation)
    else:
        raise TypeError(f"Undocumented item type {type(tiem)}.")

def deserialize_item(item):
    if isinstance(item, dict):
        if all(("x" in item.keys()), ("y" in item.keys()), ("interpolation" in item.keys())):
            return Tab1DExtended(**item)
        else:
            raise ValueError("Deserializing tuple containing nested dicts is not allowed!")
    elif isinstance(item, str):
        if "+/-" in item:
            return deserialize_variable(item)
        else:
            return item
    elif isinstance(item, list):
        return np.array(item)
    else:  # str
        raise ValueError(f"cannot deserialize item type {type(item)}!")

def deserialize_variable(variable) -> openmc.core.Variable:
    """
    Restore an openmc.core.Variable/openmc.core.AffineScalarFunc variable from str format
    (used to preserve it as the content of a json file) into openmc.core.Variable.
    """
    if ")" in variable:
        multiplier = float("1" + variable.split(")")[1])
        variable_stripped = variable.split(")")[0].strip("(")
    else:
        multiplier = 1.0
        variable_stripped = variable
    return Variable(*[float(i) * multiplier for i in variable_stripped.split("+/-")])        

def serialize_bg_radiation_dict(radiation_dict: dict):
    """
    Returns
    -------
    dictionary of tuples, where each dictionary is a list.
    """
    if len(radiation_dict)==0:
        return {}
    if isinstance(list(radiation_dict.keys())[0], str):
        # dict key=str, value=dict
        return {k: serialize_bg_radiation_dict(v) for k,v in radiation_dict.items()}
    elif isinstance(list(radiation_dict.keys())[0], tuple):
        # dict key=tuple, value=np.array
        return [(*[serialize_item(i) for i in k], v.tolist()) for k, v in radiation_dict.items()]

def deserialize_bg_radiation_dict(radiation_dict: dict):
    if len(radiation_dict)==0:
        return {}


def main(
    composition, library, irradiation_duration, transit_duration, measurement_duration
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

    # TODO: rewrite this stage without openmc, and possibly implement an alternative using FISPACT-II.
    # stage 3.1: find what isotopes need to be extracted.
    isotope_of_interest = set()
    for isotopes in processed_composition.values():
        for iso in isotopes.keys():
            atnum_and_massnum = commonname_to_atnum_massnum(iso)
            isotope_of_interest.add(atnum_and_massnum)
    # stage 3.2: extract them
    xs_dict, decay_dict = sparsely_load_xs_and_decay_dict(isotope_of_interest, library)

    def xs_template_generator():
        """
        For creating an empty array representing the cross-section for generating a
        single count in whichever gamma-line of interest.
        """
        return np.zeros(len(gs_array), dtype=float)

    every_foil_response_matrix = {}
    for foil_name, foil_comp in tqdm(processed_composition.items(), desc="Processing every foil individually..."):
        # TODO: double tqdm here.
        this_foil = defaultdict(xs_template_generator)
        this_background = defaultdict(xs_template_generator)
        for isotope, atomic_fraction in foil_comp.items():
            for rx_name, rx_xs in reactions_matching(xs_dict, isotope).items():
                decay_pathways = linearize_decay_chain(build_decay_chain_tree(decay_dict, rx_name.split("-")[1]))
                for pathway in decay_pathways:
                    # calculate how many decays of PRODUCT are measured per REACTANT ATOM
                    # initially present in the foil when irradiated by 1 cm^-2 s^-1 flux
                    # in each bin from time t=0-a seconds, and then measured from time
                    # t= b-c seconds.
                    decay_correction_factor = mat_exp_num_decays(
                        pathway.branching_ratios,
                        pathway.decay_constants,
                        irradiation_duration,
                        irradiation_duration+transit_duration,
                        irradiation_duration+transit_duration+measurement_duration,
                    )
                    scaled_collapsed_xs = (
                        collapse_single_xs(rx_xs, gs_array)
                        * BARN
                        * atomic_fraction
                        * nom(pathway.branching_fraction)
                        * decay_correction_factor
                    )
                    for peak_energy, peak_intensity, source in pathway.discrete_photon_spectrum:
                        this_foil[(peak_energy, source)] += scaled_collapsed_xs * peak_intensity
                    for background_dist, source in pathway.background_photon_spectrum:
                        # the intensity correction value of the background is already built into background_dist
                        this_background[(background_dist, source)] += scaled_collapsed_xs
        every_foil_response_matrix[foil_name] = sorted_dict(this_foil)
        every_foil_background[foil_name] = this_background

    # Store response matrices and background spectra response matrices
    with open(".response_matrices.json", "w") as j:
        json.write(json.dumps)
    gamma_energy_limits = np.array(default_gamma_energy_limits_keV) * keV
    if gamma_energy_limits_keV:
        gamma_energy_limits = np.array(sorted(gamma_energy_limits_keV)) * keV
    decay_info = {}
    for name, dec_file in tqdm(
        decay_dict.items(),
        desc="Summarizing the decay gamma spectra into a single scalar: countable number of pulses.",
    ):
        decay_info[name] = condense_spectrum_copy(
            dec_file, eff_curve, gamma_lims=gamma_energy_limits
        )
        background_continua.json

    sigma_df, selfshielding_dict = collapse_xs(xs_dict, gs_array)
    # ^ we must make sure to extract the max sigma from the the raw xs before collapsing it to the right group structure.
    # This is because the process of collapsing it to the appropriate group structure destroys that information.

    # save the rest of the useful informations into files. Needed in step 3+
    print(
        "Writing to decay_info.json, sigma_df.csv, and self-shielding.json...", end="\r"
    )
    save_decay_info(decay_info)
    save_microscopic_cross_section_csv(sigma_df)
    save_self_shielding(selfshielding_dict)
    print("Writing to decay_info.json, sigma_df.csv, and self-shielding.json... Done!")
