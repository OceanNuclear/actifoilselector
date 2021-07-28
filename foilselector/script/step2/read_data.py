# typical system/python stuff
import os, sys, json, time
from tqdm import tqdm
import warnings, gc
from io import StringIO
from collections import OrderedDict, defaultdict
# typical python numerical stuff
from numpy import array as ary; import numpy as np
from numpy import log as ln
import pandas as pd
# openmc stuff
import openmc
from openmc.data import IncidentNeutron, Decay, Evaluation, ATOMIC_SYMBOL
from openmc.data.reaction import REACTION_NAME
# uncertainties
import uncertainties
from uncertainties.core import Variable

# local modules
from foilselector.openmcextension import Integrate
from foilselector.foldermanagement import save_parameters_as_json
from foilselector.openmcextension import tabulate, detabulate
from foilselector.generic import ordered_set
from foilselector.physicalparameters import HPGe_eff_file, HPGe_efficiency_curve_generator
from foilselector.selfshielding import MaxSigma
                         
def sort_and_trim_ordered_dict(ordered_dict, trim_length=3):
    """
    sort an ordered dict AND erase the first three characters (the atomic number) of each name
    """
    return OrderedDict([(key[trim_length:], val)for key,val in sorted(ordered_dict.items())])

def extract_decay(dec_file):
    """
    extract the useful information out of an openmc.data.Decay entry
    """
    half_life = Variable(np.clip(dec_file.half_life.n, 1E-23, None), dec_file.half_life.s) # increase the half-life to a non-zero value.
    decay_constant = ln(2)/(half_life)
    modes = {}
    for mode in dec_file.modes:
        modes[mode.daughter] = mode.branching_ratio # we don't care what mechanism is used to transmute it. We just care about the respective branching ratios.
    return dict(decay_constant=decay_constant, branching_ratio=modes, spectra=dec_file.spectra)

def rename_branching_ratio(decay_dict, isomeric_to_excited_state):
    """
    dec_file
    isomeric_to_excited_state_translator : a dictionary that translates from isomeric state to excited state.
    """
    for parent in tqdm(decay_dict.keys()):
        products = decay_dict[parent]["branching_ratio"]
        renamed = {}
        for prod, ratio in products.items():
            e_name = isomeric_to_excited_state.get(prod, prod.split("_")[0])
            if e_name in renamed:
                renamed[e_name] += ratio
            else:
                renamed[e_name] = ratio
        decay_dict[parent]["branching_ratio"] = renamed
    return decay_dict

def condense_spectrum(dec_file, photopeak_eff_curve):
    """
    We will explicitly ignore all continuous distributions because they do not show up as clean gamma lines.
    Note that the ENDF-B/decay/ directory stores a lot of spectra (even those with very clear lines) as continuous,
        so it may lead to the following method ignoring it.
        The workaround is just to use another library where the evaluators aren't so lazy to not use the continuous_flag=='both' :/
    """
    count = Variable(0.0,0.0)
    if ("gamma" in dec_file['spectra']) and ("discrete" in dec_file['spectra']['gamma']):
        norm_factor = dec_file['spectra']['gamma']['discrete_normalization']
        for gamma_line in dec_file['spectra']['gamma']['discrete']:
            # if the gamma_line is within the wanted energy range
            if np.clip(gamma_line['energy'].n, *gamma_E)==gamma_line['energy'].n:
                count += photopeak_eff_curve(gamma_line['energy']) * gamma_line['intensity']/100 * norm_factor
    if ('xray' in dec_file['spectra']) and ('discrete' in dec_file['spectra']['xray']):
        norm_factor = dec_file['spectra']['xray']['discrete_normalization']
        for xray_line in dec_file['spectra']['xray']['discrete']:
            # if the xray_line is within the wanted energy range
            if np.clip(xray_line['energy'].n, *gamma_E)==xray_line['energy'].n:
                additional_counts = photopeak_eff_curve(xray_line['energy']) * xray_line['intensity']/100 * norm_factor
                if not additional_counts.s<=additional_counts.n:
                    additional_counts = Variable(additional_counts.n, additional_counts.n) # clipping the uncertainty so that std never exceed the mean. This takes care of the nan's too.
                count += additional_counts
    del dec_file['spectra']
    dec_file['countable_photons'] = count #countable photons per decay of this isotope
    return

def deduce_daughter_from_mt(parent_atomic_number, parent_atomic_mass, mt):
    """
    Given the atomic number and mass number, get the daughter in the format of 'Ag109'.
    """
    if mt in MT_to_nuc_num.keys():
        element_symbol = openmc.data.ATOMIC_SYMBOL[parent_atomic_number + MT_to_nuc_num[mt][0]]
        product_mass = str(parent_atomic_mass + MT_to_nuc_num[mt][1])
        if len(MT_to_nuc_num[mt])>2 and MT_to_nuc_num[mt][2]>0: # if it indicates an excited state
            excited_state = '_e'+str(MT_to_nuc_num[mt][2])
            return element_symbol+product_mass+excited_state
        else:
            return element_symbol+product_mass
    else:
        return None

def extract_xs(parent_atomic_number, parent_atomic_mass, rx_file, tabulated=True):
    """
    For a given (mf, mt) file,
    Extract only the important bits of the informations:
    actaul cross-section, and the yield for each product.
    Outputs a list of these modified cross-sections (which are multiplied onto the thing if possible)
        along with all their names.
    The list can then be added into the final reaction dictionary one by one.
    """
    appending_name_list, xs_list = [], []
    xs = rx_file.xs['0K']
    if isinstance(xs, openmc.data.ResonancesWithBackground):
        xs = xs.background # When shrinking the group structure, this contains everything you need. The Resonance part of xs can be ignored (only matters for self-shielding.)
    if len(rx_file.products)==0: # if no products are already available, then we can only assume there is only one product.
        daughter_name = deduce_daughter_from_mt(parent_atomic_number, parent_atomic_mass, rx_file.mt)
        if daughter_name: # if the name is not None or False, i.e. a matching MT number is found.
            name = daughter_name+'-MT='+str(rx_file.mt) # deduce_daughter_from_mt will return the ground state value
            appending_name_list.append(name)
            xs_list.append(detabulate(xs) if (not tabulated) else xs)
    else:
        for prod in rx_file.products:
            appending_name_list.append(prod.particle+'-MT='+str(rx_file.mt))
            partial_xs = openmc.data.Tabulated1D(xs.x, prod.yield_(xs.x) * xs.y,
                                                breakpoints=xs.breakpoints, interpolation=xs.interpolation)
            xs_list.append(detabulate(partial_xs) if (not tabulated) else partial_xs)
    return appending_name_list, xs_list

def collapse_xs(xs_dict, gs_ary):
    """
    Calculates the group-wise cross-section in the given group-structure
    by averaging the cross-section within each bin.
    """
    collapsed_sigma, max_xs_dict = {}, MaxSigma()
    print("Collapsing the cross-sections to the desired group-structure:")
    for parent_product_mt, xs in tqdm(xs_dict.items()):
        # perform the integration
        I = Integrate(xs)
        sigma = I.definite_integral(*gs_ary.T)/np.diff(gs_ary, axis=1).flatten()

        collapsed_sigma[parent_product_mt] = sigma
        max_xs_dict[parent_product_mt] = max(xs.y)

    del xs_dict; gc.collect() # remove to save memory
    return pd.DataFrame(collapsed_sigma).T, max_xs_dict

def merge_identical_parent_products(object_with_key): # can I speed it up by reducing the number of (hidden) for-loops? (matching_reactions is a sort of for loop)
    # I worry that it can only be sped up using Fortran. Not python, IMO.
    # get the parent_product string and mt number string as two list, corresponding to each row in the sigma_df.
    parent_product_list, mt_list = [], []
    for parent_product_mt in object_with_key.index:
        parent_product_list.append("-".join(parent_product_mt.split("-")[:2]))
        mt_list.append(parent_product_mt.split("=")[1])
    parent_product_list, mt_list = ary(parent_product_list), ary(mt_list) # make them into array to make them indexible.

    partial_reaction_array = object_with_key.values
    parent_product_all = ordered_set(parent_product_list)

    sigma_unique = {}
    tprint("Condensing the sigma_xs dataframe to merge together reactions with identical (parent, product) pairs:")
    for parent_product in tqdm(parent_product_all):
        matching_reactions = parent_product_list==parent_product
        mt_name = "-MT=({})".format(",".join(mt_list[matching_reactions]))
        sigma_unique[parent_product+mt_name] = partial_reaction_array[matching_reactions].sum(axis=0)
    del object_with_key; del partial_reaction_array; gc.collect()
    return pd.DataFrame(sigma_unique).T

class MF10(object):
    def __getitem__(self, key):
        return self.reactions.__getitem__(key)

    def __len__(self):
        return self.reactions.__len__()

    def __iter__(self):
        return self.reactions.__iter__()

    def __reversed__(self):
        return self.reactions.__reversed__()

    def __contains__(self, key):
        return self.reactions.__contains__(key)

    __slots__ = ["number_of_reactions", "za", "target_mass", "target_isomeric_state", "reaction_mass_difference", "reaction_q_value", "reactions"]
    # __slots__ created for memory management purpose in case there are many suriving instances of MF10 all present at once.
    def __init__(self, mf10_mt5_section):
        if mf10_mt5_section is not None:
            file_stream = StringIO(mf10_mt5_section)
            za, target_mass, target_iso, _, ns, _ = openmc.data.get_head_record( file_stream ) # read the first line, i.e. the head record for MF=10, MT=5.
            self.number_of_reactions = ns
            self.za = za
            self.target_mass = target_mass
            self.target_isomeric_state = target_iso
            self.reaction_mass_difference, self.reaction_q_value = {}, {}
            self.reactions = {}
            for reaction_number in range(ns):
                (mass_diff, q_value, izap, isomeric_state), tab = openmc.data.get_tab1_record(file_stream)
                self.reaction_mass_difference[(izap, isomeric_state)] = mass_diff
                self.reaction_q_value[(izap, isomeric_state)] = q_value
                self.reactions[(izap, isomeric_state)] = tab
        else:
            self.reactions = {}

    def keys(self):
        return self.reactions.keys()

    def items(self):
        return self.reactions.items()

    def values(self):
        return self.reactions.values()

def merge_xs(low_E_xs, high_E_xs, debug_info=""):
    """
    Specifically created to append E>=30MeV xs onto the normally obtained data.
    Adds together two different cross-section profiles.
    The high_E_xs is expected to start at a minimum energy E_min,
    while low_E_xs is expected to have its xs zeroed at higher than xs.

    Otherwise... well fuck.
    I'm going to have to add them together and figure out some sort of interpoation scheme.
    """
    E_min = high_E_xs.x.min()
    low_E_range = low_E_xs.x<E_min
    strictly_high_E_range = low_E_xs.x>E_min # ignore the point E = E_min
    # assert all(low_E_xs.y[low_E_range]==0.0), "The high_E_xs record must start at E_min, and the low_E_xs record is expected to have zeroed all y values above at x>=E_min."
    if all(low_E_xs.y[strictly_high_E_range]==0.0):
        low_x, low_y, low_interp = low_E_xs.x[low_E_range], low_E_xs.y[low_E_range], ary(detabulate(low_E_xs)["interpolation"], dtype=int)[low_E_range[:-1]]
        high_x, high_y, high_interp = high_E_xs.x, high_E_xs.y, ary(detabulate(high_E_xs)["interpolation"], dtype=int)
        return tabulate({
            "x":np.hstack([low_x, high_x]),
            "y":np.hstack([low_y, high_y]),
            "interpolation":np.hstack([low_interp, high_interp]),
            })
    else:
        from matplotlib import pyplot as plt
        from misc_library import plot_tab
        plot_tab(low_E_xs)
        plot_tab(high_E_xs)
        plt.title(debug_info)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()
        return low_E_xs

# must load decay informatino and neutron incidence information in a single python script,
# because the former is needed to compile the dictionary of isomeric_to_excited_state, which the latter use when saving the reaction rates products..
