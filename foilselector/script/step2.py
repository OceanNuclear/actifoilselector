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
from tqdm import tqdm
from numpy import array as ary

from foilselector.foldermanagement import *
from foilselector.reactionnaming import (
    specify_isotopic_composition,
    commonname_to_atnum_massnum,
)
from foilselector.openmcextension import *
from foilselector.simulation import EfficiencyCurve


default_gamma_energy_limits_keV = [20, 4600]


def main(
    composition, library, photopeak_efficiency, gamma_energy_limits_keV, group_structure
):
    # sub-step 1: break down the foil composition into its consituent isotopes.
    with open(composition) as j:
        _composition_used_here = json.load(j)
    processed_composition = {
        foil_name: specify_isotopic_composition(foil_comp)
        for foil_name, foil_comp in _composition_used_here.items()
    }
    # save a version of the processed_composition dictionary

    save_atomic_composition_json(processed_composition)  # needed for step 3+

    # sub-step 2: find what isotopes need to be extracted.
    _isotope_concerned = set()
    for isotopes in processed_composition.values():
        for iso in isotopes.keys():
            atnum_and_massnum = commonname_to_atnum_massnum(iso)
            _isotope_concerned.add(atnum_and_massnum)
    xs_dict, decay_dict = sparsely_load_xs_and_decay_dict(_isotope_concerned, library)

    # save decay_radiation
    save_decay_radiation(decay_dict)  # needed for step 5: simulating gamma spec.
    # sub-step 3: apply efficiency curve to condense the decay-dict into a smaller dictionary
    try:
        eff_curve = EfficiencyCurve.from_file(str(photopeak_efficiency))
    except TypeError as e:
        print("Incorrect file path. Try giving a valid file to the -e argument?")
        raise e
        sys.exit()
    decay_info = {}
    for name, dec_file in tqdm(
        decay_dict.items(),
        desc="Summarizing the decay gamma spectra into a single scalar: countable number of pulses.",
    ):
        gamma_energy_limits = (
            sorted(gamma_energy_limits_keV)
            if gamma_energy_limits_keV
            else default_gamma_energy_limits_keV
        )
        decay_info[name] = condense_spectrum_copy(
            dec_file, eff_curve, gamma_lims=ary(gamma_energy_limits) * 1000
        )

    # sub-step 4: re-bin into the correct group structure
    gs_array = read_gs(group_structure)
    assert (gs_array[:, 0] < gs_array[:, 1]).all(), (
        "The -G, --group-structure file must be provided in ascending bin order! (And the flux file used in the next step must match it.)"
    )

    sigma_df, selfshielding_dict = collapse_xs(xs_dict, gs_array)
    # ^ we must make sure to extract the max sigma from the the raw xs before collapsing it to the right group structure.
    # This is because the process of collapsing it to the appropriate group structure destroys that information.

    # sub-step 5: Merge
    sigma_df = merge_identical_parent_products(sigma_df)

    # save the rest of the useful informations into files. Needed in step 3+
    print(
        "Writing to decay_info.json, sigma_df.csv, and self-shielding.json...", end="\r"
    )
    save_decay_info(decay_info)
    save_microscopic_cross_section_csv(sigma_df)
    save_self_shielding(selfshielding_dict)
    print("Writing to decay_info.json, sigma_df.csv, and self-shielding.json... Done!")
