import numpy as np
from numpy import array as ary
import pandas as pd
from collections import defaultdict
from functools import partial
from foilselector.constants import BARN
from foilselector.simulation.decay import build_decay_chain_tree, linearize_decay_chain, mat_exp_num_decays

__all__ = ["default_dict_zero_array", "get_macroscopic_xs_all_rx", "calculate_num_decays_measured_per_product_present", "merge_same_isotope_populations"]

def default_dict_zero_array(length):
    """Return a defaultdict that would be initialized with np.zeros(length) when a new key is used.
    length = length of the zero-filled array we require new key to have."""
    empty_xs_matching_gs = partial(np.zeros, length)
    return defaultdict(empty_xs_matching_gs)    

def get_macroscopic_xs_all_rx(isotope_number_densities, sigma_df):
    """
    Calculate the macroscopic cross-section of all reactions involved.
    We start from the microscopic cross-section value stored in sigma_df,
    and use the formula:
    Formula
    -------
    Sigma(E) = sigma(E) * N_d # unit: cm^-1
    
    Returns
    -------
    Sigma(E) of each reaction, as a dataframe.
    starts with parents_product_mts, which is MULTIPLE parents giving the same product through MULTIPLE mts,
    And ends up with a SINGLE line of macroscopic.

    DeprecationWarning: this method probably would only work for the current (2021-07-27 22:50:20) implementation
        as I don't plan to keep a sigma_df (pd.DataFrame object) in a microscopic_xs.csv forever, because this will take up space.
    """

    # create a cross-section accumulator that if there's no existing entry for the matching product,
    # an all-zero cross-section vector will be created before the '+=' operation.
    macroscopic_xs = default_dict_zero_array(len(sigma_df.columns))

    parent_list, product_list = ary([name.split("-")[0:2] for name in sigma_df.index]).T
    for isotope_name, density in isotope_number_densities.items():
        matching_parent = sigma_df[parent_list==isotope_name]

        for reaction_name, xs_profile in matching_parent.iterrows():
            product = reaction_name.split("-")[1]
            macroscopic_xs[product] += xs_profile.values * (BARN * density) # accumulate cross-section

    return macroscopic_xs

def calculate_num_decays_measured_per_product_present(root_product, schedule, decay_info, time_resolved=False):
    """
    Calculate the number of decays caused by a single root product atom, measured during
        the measurement periods specified in schedule.
    Parameters
    ----------
    root_product: a str name of the root product of interest created during the irradiation.
    schedule: a foilselector.simulation.schedule.Schedule object
                containing information about the irradiation, cooling, and measurement schedule.
    
    decay_info: dictionary of decay_info from which the number of counts per second can be extracted
    time_resolved: boolean to choose whether to show A->B as a separate pathway than A->C->B and A->C->D->B, etc.
                default: False, because we can't see where they're coming from.

    Output
    ------
    A dictionary showing the number of decays that each of the isotopes in the DAG decay graph experiences
    during the measurement time.

    Currently this function (calculate_num_decays_measured_per_product_present) relies on decay_info to be read multiple times,
    So I will have to revamp decay_info into a class of its own
        and attach this function as a method instead to make it cleaner.
        Honestly I don't think it's worth the effort.
    """
    effective_step_list = schedule.measurable_irradiation_effects()
    total_fluence = sum(step.fluence for step in effective_step_list)
    decay_tree = build_decay_chain_tree(root_product, decay_info)
    collected_num_decays = defaultdict(float)
    collected_num_photons= defaultdict(float)

    for subchain in linearize_decay_chain(decay_tree):
        # for the same subchain (i.e. target product element),
        num_decays_during_measurement = sum(
                    # sum the effect from all irradiation steps.
                    mat_exp_num_decays(
                    subchain.branching_ratios, subchain.decay_constants,
                    step.a, step.b, step.c
                    ) * step.fluence/total_fluence # normalized to 1 unit fluence
                    for step in effective_step_list
                )
        if time_resolved: 
            isotope_name = arrow.join(subchain.names) # give the full pathway where the isotope came from.
        else:
            isotope_name = subchain.names[-1]
        collected_num_decays[isotope_name] += num_decays_during_measurement
        collected_num_photons[isotope_name]+= num_decays_during_measurement * subchain.countable_photons
    num_decays_series = pd.Series(collected_num_decays, name=root_product)
    num_photons_series = pd.Series(collected_num_photons, name=root_product)

    return pd.DataFrame({"number of decays":num_decays_series,
                "number of photons measurable":num_photons_series})

def merge_same_isotope_populations(dict_of_df, sort_key="number of photons measurable"):
    """
    Given the dictionary of many different dataframes
        (each one representing the populations and counts of all isotopes created by the decay of one root-product),
    Accumulate all the end products to create one big dataframe containing all of the isotopes (once each).
    dict_of_df: a dictionary of pd.DataFrame objects
    sort_key: sort the final outputted dataframe of products by this key in descending order.
    """
    dict_items = list(dict_of_df.items())
    if len(dict_items)==0: return {}

    final_isotope_population_df = dict_items[0][1].copy()

    for root_isotope, df in dict_items[1:]:
        for final_isotope, populations in df.iterrows():
            if final_isotope not in final_isotope_population_df.index:
                final_isotope_population_df = final_isotope_population_df.append(populations)
            else:
                final_isotope_population_df.loc[final_isotope] += populations
    if sort_key:
        final_isotope_population_df.sort_values(sort_key, ascending=False, inplace=True)# sort descendingly
    return final_isotope_population_df
