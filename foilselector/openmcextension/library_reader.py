# typical system/python stuff
import os, sys, json, time, warnings
from tqdm import tqdm
import gc
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
from foilselector.generic import ordered_set
from foilselector.selfshielding import MaxSigma
from foilselector.simulation import LineTuple

__all__ = ["condense_spectrum_copy", "collapse_xs", "merge_identical_parent_products", "simplify_spectrum_copy"]

def condense_spectrum_copy(dec_file, photopeak_eff_curve, gamma_lims=[20*1E3, 4.6*1E6]):
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
            if np.clip(gamma_line['energy'].n, *gamma_lims)==gamma_line['energy'].n:
                count += photopeak_eff_curve(gamma_line['energy']) * gamma_line['intensity'] * norm_factor
    if ('xray' in dec_file['spectra']) and ('discrete' in dec_file['spectra']['xray']):
        norm_factor = dec_file['spectra']['xray']['discrete_normalization']
        for xray_line in dec_file['spectra']['xray']['discrete']:
            # if the xray_line is within the wanted energy range
            if np.clip(xray_line['energy'].n, *gamma_lims)==xray_line['energy'].n:
                additional_counts = photopeak_eff_curve(xray_line['energy']) * xray_line['intensity'] * norm_factor
                if not additional_counts.s<=additional_counts.n:
                    additional_counts = Variable(additional_counts.n, additional_counts.n) # clipping the uncertainty so that std never exceed the mean. This takes care of the nan's too.
                count += additional_counts
    dec_file_copy = dec_file.copy()
    # replace the spectra attribute with countable_photon attribute.
    del dec_file_copy['spectra']
    dec_file_copy['countable_photons'] = count # countable photons per decay of this isotope
    return dec_file_copy

def simplify_spectrum_copy(dec_file, isotope_name, photopeak_eff_curve, gamma_lims=[20*1E3, 4.6*1E6]):
    """
    We will explicitly ignore all continuous distributions because they do not show up as clean gamma lines.
    Note that the ENDF-B/decay/ directory stores a lot of spectra (even those with very clear lines) as continuous,
        so it may lead to the following method ignoring it.
        The workaround is just to use another library where the evaluators aren't so lazy to not use the continuous_flag=='both' :/
    """
    spectra = dec_file['spectra']
    lines = [] # container of lines extracted

    if ('xray' in spectra) and ('discrete' in spectra['xray']): # find dec_file['spectrum']['xray']['discrete']
        norm_factor = spectra['xray']['discrete_normalization']
        for xray_line in spectra['xray']['discrete']:
            # if the xray_line is within the wanted energy range
            if np.clip(xray_line['energy'].n, *gamma_lims)==xray_line['energy'].n:
                counts_per_decay = photopeak_eff_curve(xray_line['energy']) * xray_line['intensity'] * norm_factor
                if not counts_per_decay.s<=counts_per_decay.n: # clipping the uncertainty so that std never exceed the mean. This takes care of the nan's too.
                    # it's only a problem for x rays due to the low energies.
                    counts_per_decay = Variable(counts_per_decay.n, counts_per_decay.n)
                lines.append(LineTuple(xray_line['energy'].n,
                                        counts_per_decay,
                                        isotope_name,
                                        'xray'))
    if ("gamma" in spectra) and ("discrete" in spectra['gamma']): # find dec_file['spectrum']['gamma']['discrete']
        norm_factor = spectra['gamma']['discrete_normalization']
        for gamma_line in spectra['gamma']['discrete']:
            # if the gamma_line is within the wanted energy range
            if np.clip(gamma_line['energy'].n, *gamma_lims)==gamma_line['energy'].n:
                lines.append(LineTuple(gamma_line['energy'].n,
                                        gamma_line['intensity'] * norm_factor,
                                        isotope_name,
                                        'gamma'))
    return lines

def collapse_xs(xs_dict, gs_ary):
    """
    Calculates the group-wise cross-section (production rate of product per unit flux) in the given group-structure
    by averaging the cross-section within each bin.
    """
    collapsed_sigma, max_xs_dict = {}, MaxSigma()
    with warnings.catch_warnings(record=True) as w_list:
        for parent_product_mt, xs in tqdm(xs_dict.items(), desc="Collapsing the cross-sections to the desired group-structure:"):
            # perform the integration
            I = Integrate(xs)
            sigma = I.definite_integral(*gs_ary.T)/np.diff(gs_ary, axis=1).flatten()

            collapsed_sigma[parent_product_mt] = sigma
            max_xs_dict[parent_product_mt] = max(xs.y)
    # ignore the w_list of caught warnings
    return pd.DataFrame(collapsed_sigma).T, max_xs_dict

def merge_identical_parent_products(loose_collection_of_rx):
    """
    Parameters
    ----------
    loose_collection_of_rx: a dataframe of reaction cross-sections. 
    The same parent-product pair can occur multiple times, e.g.
    'Mn55-Cr51-MT=154'
    'Mn55-Cr51-MT=5'
    These are two ways of getting the same isotopes
    Returns
    -------
    a pandas dataframe that only have one unique pair of parent-product line
    e.g. 'Mn55-Cr51-MT=(154,5)' only appears once

    This is an idempotent operation.
    """
    # can I speed it up by reducing the number of (hidden) for-loops? (matching_reactions is a sort of for loop)
    # I worry that it can only be sped up using Fortran. Not python, IMO.
    # get the parent_product string and mt number string as two list, corresponding to each row in the sigma_df.
    parent_product_list, mt_list = [], []
    for parent_product_mt in loose_collection_of_rx.index:
        parent_product_list.append("-".join(parent_product_mt.split("-")[:2]))
        mt_list.append(parent_product_mt.split("=")[1])
    parent_product_list, mt_list = ary(parent_product_list), ary(mt_list) # make them into array to make them indexible.

    partial_reaction_array = loose_collection_of_rx.values
    parent_product_all = ordered_set(parent_product_list)

    sigma_unique = {}
    print("Condensing the sigma_xs dataframe to merge together reactions with identical (parent, product) pairs:")
    for parent_product in tqdm(parent_product_all):
        matching_reactions = parent_product_list==parent_product
        mt_name = "-MT=({})".format(",".join(mt_list[matching_reactions]))
        sigma_unique[parent_product+mt_name] = partial_reaction_array[matching_reactions].sum(axis=0)
    del loose_collection_of_rx; del partial_reaction_array; gc.collect()
    return pd.DataFrame(sigma_unique).T
