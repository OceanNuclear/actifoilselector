"""
outputs the expected spectra (and effectiveness) of the finalized foil selection
"""
if __name__=="__main__":
    from finalize_selection import get_num_reactants_dict
else:
    from .__init__ import *
"""
Instruction:
This module is expected to be used after the following steps
- ```python -m foilselector.scripts.step4 my_directory/```
- removing the dictionaries stating information about unwanted foils from selected_foils.json
- Changing the thicknesses values listed in selected_foils.json into the actual thicknesses of the foils that will be purchased.
"""
import sys, os
import json
from collections import defaultdict
from tqdm import tqdm

from numpy import array as ary
import numpy as np
from scipy.special import expm1

import pandas as pd
from foilselector.foldermanagement import *
from foilselector.constants import BARN
from foilselector.openmcextension import unserialize_dict
from foilselector.physicalparameters import HPGe_efficiency_curve_generator
from foilselector.reactionnaming import unpack_reactions
from foilselector.simulation import GammaSpectrum, SingleDecayGammaSignature
from foilselector.simulation.gamma import arrow
from foilselector.simulation.schedule import *
from foilselector.simulation.decay import (
                        pprint_tree,
                        linearize_decay_chain,
                        build_decay_chain_tree,
                        mat_exp_num_decays,
                        )

NO_TIME_RESOLUION = True

def main(dirname):
    with open(os.path.join(dirname, "decay_info.json")) as j:
        decay_info = unserialize_dict(json.load(j))
    with open(os.path.join(dirname, "decay_radiation.json")) as j:
        decay_dict = unserialize_dict(json.load(j)) # the actual gamma spectrum
    with open(os.path.join(dirname, "selected_foils.json")) as j:
        selected_foils = json.load(j)
    with open(os.path.join(dirname, "irradiation_schedule.txt")) as f:
        schedule_of_materials, efficiency_curve_dict = read_fispact_irradiation_schedule(f.read())

    gs = get_gs(dirname)
    sigma_df = get_microscopic_cross_sections_df(dirname)
    apriori_flux = get_apriori(dirname, None)
    neutron_energy_pdf_apriori = apriori_flux/apriori_flux.sum()

    # the microscopic cross-sections in barns

    parent_list, product_list = ary([name.split("-")[0:2] for name in sigma_df.index]).T

    def get_macroscopic_xs_all_rx(isotope_number_densities):
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
        from functools import partial
        empty_xs_matching_gs = partial(np.zeros, len(sigma_df.columns))
        macroscopic_xs = defaultdict(empty_xs_matching_gs)

        for isotope_name, density in isotope_number_densities.items():
            matching_parent = sigma_df[parent_list==isotope_name]

            for reaction_name, xs_profile in matching_parent.iterrows():
                product = reaction_name.split("-")[1]
                macroscopic_xs[product] += xs_profile.values * (BARN * density) # accumulate cross-section

        return macroscopic_xs

    def calculate_num_decays_measured_per_product_present(root_product, schedule):
        """
        Calculate the number of decays measured during the measurement periods specified in schedule.
        Parameters
        ----------
        root_product: a str name of the root product of interest created during the irradiation.
        schedule: a foilselector.simulation.schedule.Schedule object
                    containing information about the irradiation, cooling, and measurement schedule.

        Output
        ------
        A dictionary showing the number of decays that each of the isotopes in the DAG decay graph experiences
        during the measurement time.

        Relies on decay_info to be read multiple times,
        So I will have to revamp decay_info into a class of its own before I can make this function cleaner.
        """
        effective_step_list = schedule.measurable_irradiation_effects()
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
                        ) * step.fluence
                        for step in effective_step_list
                    )
            if NO_TIME_RESOLUION: 
                isotope_name = subchain.names[-1]
            else:
                isotope_name = arrow.join(subchain.names) # give the full pathway where the isotope came from.
            collected_num_decays[isotope_name] += num_decays_during_measurement
            collected_num_photons[isotope_name]+= num_decays_during_measurement * subchain.countable_photons
        num_decays_series = pd.Series(collected_num_decays, name=root_product)
        num_photons_series = pd.Series(collected_num_photons, name=root_product)

        return pd.DataFrame({"number of decays":num_decays_series,
                    "number of photons measurable":num_photons_series})

    def merge_same_isotope_populations(dict_of_df, sort=True):
        """
        Given the dictionary of many different dataframes
            (each one representing the populations and counts of all isotopes created by the decay of one root-product),
        Accumulate all the end products to create one big dataframe containing all of the isotopes (once each).
        If 
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
        if sort:
            final_isotope_population_df.sort_values("number of photons measurable", ascending=False, inplace=True)# sort descendingly
        return final_isotope_population_df

    num_decays_sorted_by_root_product = {}

    for foil_name, foil in selected_foils.items():
        print("for foil =", foil_name, "a schedule is detected as the following:", schedule_of_materials[foil_name])

        thickness = foil["thickness (cm)"]
        area = foil["area (cm^2)"]
        num_density_dict = foil["number density (cm^-3)"]
        macroscopic_xs = get_macroscopic_xs_all_rx(num_density_dict) # is a dict

        macroscopic_xs_df = pd.DataFrame(macroscopic_xs).T
        # dataframe currently contains macroscopic cross-section values (Sigma(E)):
        # each row contains the macroscopic cross-section profile of reactions that generate 1 product,
        # while each column denotes the bin number.

        # now to calculate for the number of reactions that happened in a way that accounts for self-sheilding:
        # P(E, interaction) = 1 - exp(-Sigma(E) * t)
        # number of interactions = @ P(E, interactions)
        num_reaction_per_unit_fluence = pd.Series(
            (-expm1(-macroscopic_xs_df * thickness)).values @ neutron_energy_pdf_apriori,
            index=macroscopic_xs_df.index,
            name=foil_name
        )

        # number of decays of one type of root-product is given by
        # = (number of root-products created per unit fluence) * (number of decays per root-product
        # given the current irradiation-cooling-and-measurement schedule);
        # where the fluence value is already embedded in the schedule.
        num_decays_sorted_by_root_product[foil_name] = {root_product :
            num_reactions * calculate_num_decays_measured_per_product_present(
                root_product, schedule_of_materials[foil_name]
                )
            for root_product, num_reactions in num_reaction_per_unit_fluence.items()
        }

    summed_isotope_populations, num_decays_spec, num_detected_spec = {}, {}, {}
    for foil_name, reaction_products in num_decays_sorted_by_root_product.items():
        print(f"Summing together the population of isotopes obtained from different pathways in {foil_name}...")
        summed_isotope_populations[foil_name] = merge_same_isotope_populations(reaction_products)
        num_decays_spec[foil_name] = GammaSpectrum()

        for final_isotope, populations in summed_isotope_populations[foil_name].iterrows():
            spectrum = decay_dict[final_isotope.split(arrow)[-1]].get("spectra", {})
            num_decays_spec[foil_name] += SingleDecayGammaSignature(spectrum, final_isotope) * populations["number of decays"]
        num_detected_spec[foil_name] = num_decays_spec[foil_name] * efficiency_curve_dict[foil_name]

    return num_detected_spec, summed_isotope_populations

def main_old(dirname):
    # # get the user-defined foil selection, and their thicknesses
    # with open(os.path.join(dirname, "parameters_used.json")) as J:
    #     PARAMS_DICT = json.load(J)
    # reading data
    with open(os.path.join(dirname, "decay_radiation.json")) as j:
        decay_dict = unserialize_dict(json.load(j)) # the actual gamma spectrum
    with open(os.path.join(dirname, "decay_info.json")) as j:
        decay_info = unserialize_dict(json.load(j)) # the decay counts given the stated starting and ending measurement times.
    with open(os.path.join(dirname, "final_selection.json")) as j:
        selected_foils = json.load(j)
        
    # with open(os.path.join(dirname, "post-measurement_population.json")) as j:
    #     post_measurement_population = json.load(j)
    abs_efficieny_curve = HPGe_efficiency_curve_generator()

    # get irradiation schedules
    PARAMS_DICT = get_parameters_json(dirname)
    IRRADIATION_DURATION = PARAMS_DICT["IRRADIATION_DURATION"]
    a = IRRADIATION_DURATION
    b = a + PARAMS_DICT["TRANSIT_DURATION"]
    c = b + PARAMS_DICT["MEASUREMENT_DURATION"]

    # get the irradiating flux
    APRIORI_FLUX, APRIORI_FLUENCE = get_apriori(dirname, IRRADIATION_DURATION)
    sigma_df = get_microscopic_cross_sections_df(dirname)
    gs = get_gs(dirname)

    def get_macro_and_num_reactions(parents_product_mts, foil_num_density_dict):
        """
        Calculate the macroscopic cross-section and partial reaction rate contribution variables.
        """
        macroscopic_xs = np.zeros(len(gs))
        for parent_product_mt in unpack_reactions(parents_product_mts):
            microscopic_xs = sigma_df[sigma_df.index==parent_product_mt].values[0]*BARN
            macroscopic_xs += microscopic_xs * foil_num_density_dict[parent_product_mt.split("-")[0]]
        num_reactions = macroscopic_xs * APRIORI_FLUENCE
        return macroscopic_xs, num_reactions
        
    collected_spectra = {}
    for foil_name, foil in selected_foils.items():
        print("for foil =", foil_name)
        num_reactants = get_num_reactants_dict(foil)
        print("FISPACT input file should use the value of:")
        print(reactant_to_FUEL(num_reactants))

        total_counts_dict = foil[[k for k in foil.keys() if k.startswith("total counts at thickness =")][0]]

        # ax = plt.subplot()
        foil_spectrum_w_eff = GammaSpectrum()

        for parents_product_mts in total_counts_dict.keys():
            product = parents_product_mts.split("-")[1]

            macroscopic_xs, num_reactions = get_macro_and_num_reactions(parents_product_mts, foil["number density (cm^-3)"])
            if PLOT_MACRO:
                ax = plt.subplot()
                ax.semilogx(gs.flatten(), np.repeat(macroscopic_xs/(macroscopic_xs).sum(), 2))
                ax.set_ylabel("max-normalized macroscopic cross-section")
                ax.set_xlabel(r"$E_n$ (eV)")
                ax.set_title(parents_product_mts)
                plt.show()
            if PLOT_CONTR:
                ax = plt.subplot()
                ax.semilogx(gs.flatten(), np.repeat(num_reactions/(num_reactions).max(), 2))
                ax.set_ylabel("Fraction of contribution to the reaction rates")
                ax.set_xlabel(r"$E_n$ (eV)")
                ax.set_title(parents_product_mts)
                plt.show()

            decay_tree = build_decay_chain_tree(product, decay_info)
            pprint_tree(decay_tree)

            final_spec = GammaSpectrum()
            for subchain in linearize_decay_chain(decay_tree):
                num_decays_recorded_per_reaction = mat_exp_num_decays(subchain.branching_ratios, subchain.decay_constants, IRRADIATION_DURATION, b, c)
                num_decays_recorded = (num_reactions).sum() * num_decays_recorded_per_reaction

                spectrum_of = subchain.names[-1]
                spectrum = decay_dict[spectrum_of].get("spectra", {})
                # final_spec += (SingleDecayGammaSignature(spectrum, spectrum_of) * num_decays_recorded) # use the final isotope as name.
                final_spec += (SingleDecayGammaSignature(spectrum, subchain.names) * num_decays_recorded) # use the entire subchain as the isotope name

            if PLOT_INDIVIDUAL_SPEC:
                ax, lines = final_spec.plot()
                ax.set_title("Decay of {} in {}\n before multiplying by efficiency".format(product, foil_name))
                plt.show()

                ax, lines = (final_spec * abs_efficieny_curve).plot()
                ax.set_title("Decay of {} in {}\n after multiplying by efficiency".format(product, foil_name))
                plt.show()

            foil_spectrum_w_eff += (final_spec * abs_efficieny_curve).signatures

        collected_spectra[foil_name] = foil_spectrum_w_eff
        ax, lines = foil_spectrum_w_eff.plot(sqrt_scale=True)
        ax.set_title(foil_name+" spectrum after applying efficiency curve")
        ax.legend()
        ax.set_ylim(bottom=1E0)
        ax.set_xlim(left=0, right=2.8E3)
        plt.show()
        print("-"*40)
        if input("enter 'return' to stop at this point;")=="return":
            return collected_spectra
    return collected_spectra

if __name__=="__main__":
    final_spec, summed_isotope_populations = main(*sys.argv[1:])
