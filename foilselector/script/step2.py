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
import json, os, inspect
from tqdm import tqdm
from numpy import array as ary

import foilselector
from foilselector.foldermanagement import *
from foilselector.reactionnaming import specify_isotopic_composition, commonname_to_atnum_massnum
from foilselector.openmcextension import *
from foilselector.simulation import EfficiencyCurve

import argparse
from pathlib import Path
parg = argparse.ArgumentParser(description="""Extract the relevant cross-sections and decay data from the nuclear data library.
Then save them as functionst ath will never be used again.
This is analogous to the 'collapse' and 'condense' step in FISPACT.""")
parg.add_argument('-c', '--composition', type=Path, required=True, help=".json file specifiying the composition.")
parg.add_argument('-L', '--library'    , type=Path, required=True, help="directory(ies) where the cross-sections are stored", action='extend', nargs='+')
parg.add_argument('-e', '--photopeak-efficiency', type=Path, help="""file with data of the absolute photopeak efficiency of the gamma detector used in its current configuration.
The accepted file types are:
.csv (energy in MeV in column 0, efficiency in column 1.)
.dat (same as .csv, but space delimited instead)
.o (mcnp output)
.ecc (GENIE/ISOCS output) """,
    default=os.path.join(os.path.dirname(inspect.getsourcefile(foilselector)), "physicalparameters","photopeak_efficiency","Absolute_photopeak_efficiencyMeV.csv"))
parg.add_argument('-g', '--gamma-energy-limits-keV', type=float, action='extend', nargs=2, help="The minimum and maximum gamma energies (keV) that the detector can detect.\nThe defaults are 20 keV - 4600 keV.")
default_gamma_energy_limits_keV = [20, 4600]
parg.add_argument('-G', '--group-structure', type=Path, help="newline-separated file listing the pairs of group boundaries (comma-separated) in ascending energies.")

if __name__=="__main__":
    cl_arg = parg.parse_args()
    # a quick check to make sure all arguments are valid.
    assert os.path.exists(cl_arg.composition),                  f"the -c, --composition argument ({cl_arg.composition}) must be a valid file path!"
    assert all(os.path.exists(lib) for lib in cl_arg.library),  f"the -L, --library argument ({cl_arg.library}) must all be valid file paths!"
    assert os.path.exists(cl_arg.photopeak_efficiency),         f"the -e, --photopeak-efficiency argument ({cl_arg.photopeak_efficiency}) must be a valid file path!"
    assert os.path.exists(cl_arg.group_structure),              f"the -G, --group-structure argument ({cl_arg.group_structure}) must be a valid file path!"
    # sub-step 1: break down the foil composition into its consituent isotopes.
    with open(cl_arg.composition) as j:
        _composition_used_here = json.load(j)
    processed_composition = {foil_name:specify_isotopic_composition(foil_comp) for foil_name, foil_comp in _composition_used_here.items()}
    # save a version of the processed_composition dictionary
    from os import path
    save_atomic_composition_json(processed_composition) # needed for step 3+

    # sub-step 2: find what isotopes need to be extracted.
    _isotope_concerned = set()
    for isotopes in processed_composition.values():
        for iso in isotopes.keys():
            atnum_and_massnum = commonname_to_atnum_massnum(iso)
            _isotope_concerned.add(atnum_and_massnum)
    xs_dict, decay_dict = sparsely_load_xs_and_decay_dict(_isotope_concerned, cl_arg.library)

    # save decay_radiation
    save_decay_radiation(decay_dict) # needed for step 5: simulating gamma spec.
    # sub-step 3: apply efficiency curve to condense the decay-dict into a smaller dictionary
    try:
        eff_curve = EfficiencyCurve.from_file(str(cl_arg.photopeak_efficiency))
    except TypeError as e:
        print("Incorrect file path. Try giving a valid file to the -e argument?")
        raise e
        sys.exit()
    decay_info = {}
    for name, dec_file in tqdm(decay_dict.items(), desc="Summarizing the decay gamma spectra into a single scalar: countable number of pulses."):
        gamma_energy_limits = sorted(cl_arg.gamma_energy_limits_keV) if cl_arg.gamma_energy_limits_keV else default_gamma_energy_limits_keV
        decay_info[name] = condense_spectrum_copy(dec_file, eff_curve, gamma_lims=ary(gamma_energy_limits)*1000)

    # sub-step 4: re-bin into the correct group structure
    gs_array = read_gs(cl_arg.group_structure)
    assert (gs_array[:,0]<gs_array[:,1]).all(), "The -G, --group-structure file must be provided in ascending bin order! (And the flux file used in the next step must match it.)"
    
    sigma_df, selfshielding_dict = collapse_xs(xs_dict, gs_array)
    # ^ we must make sure to extract the max sigma from the the raw xs before collapsing it to the right group structure.
    # This is because the process of collapsing it to the appropriate group structure destroys that information.

    # sub-step 5: Merge
    sigma_df = merge_identical_parent_products(sigma_df)

    # save the rest of the useful informations into files. Needed in step 3+
    print("Writing to decay_info.json, sigma_df.csv, and self-shielding.json...", end="\r")
    save_decay_info(decay_info)
    save_microscopic_cross_section_csv(sigma_df)
    save_self_shielding(selfshielding_dict)
    print("Writing to decay_info.json, sigma_df.csv, and self-shielding.json... Done!")
