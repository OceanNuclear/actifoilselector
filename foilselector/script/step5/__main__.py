"""
outputs the expected spectra (and effectiveness) of the finalized foil selection
"""
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

import pandas as pd
from foilselector.foldermanagement import *
from foilselector.constants import BARN
from foilselector.openmcextension import unserialize_dict
from foilselector.physicalparameters import HPGe_efficiency_curve_generator
from foilselector.reactionnaming import unpack_reactions
from foilselector.simulation import GammaSpectrum, SingleDecayGammaSignature
from foilselector.simulation.schedule import *
from foilselector.simulation.decay import (
                        pprint_tree,
                        linearize_decay_chain,
                        build_decay_chain_tree,
                        mat_exp_num_decays,
                        )

def main(dirname):
    with open(os.path.join(dirname, "decay_info.json")) as j:
        decay_info = unserialize_dict(json.load(j))
    with open(os.path.join(dirname, "final_selection.json")) as j:
        selected_foils = json.load(j)
    with open(os.path.join(dirname, "irradiation_schedule.txt")) as f:
        schedule_of_materials = read_fispact_irradiation_schedule(f.read())
    sigma_df = get_microscopic_cross_sections_df(dirname)
    gs = get_gs(dirname)

    def get_macro(parents_product_mts, foil_num_density_dict):
        """
        Calculate the macroscopic cross-section and partial reaction rate contribution variables.
        starts with parents_product_mts, which is MULTIPLE parents giving the same product through MULTIPLE mts,
        And ends up with a SINGLE line of macroscopic.
        """
        macroscopic_xs = np.zeros(len(gs))
        for parent_product_mt in unpack_reactions(parents_product_mts):
            microscopic_xs = sigma_df[sigma_df.index==parent_product_mt].values[0]*BARN
            macroscopic_xs += microscopic_xs * foil_num_density_dict[parent_product_mt.split("-")[0]]
        return macroscopic_xs, num_reactions

    for foil_name, foil in selected_foils.items():
        print("for foil =", foil_name)
            num_reactants = get_num_reactants_dict(foil)
            macroscopic_xs = get_macro(parents_product_mts, foil["number density (cm^-3)"])

            num_decays_checksum = 0.0
            for step_contribution in schedule_of_materials[foil_name]
                neutrons_per_cm2 = step_contribution.flux * step_contribution.a
                num_decays_checksum += step_contribution.flux
                neutrons_per_cm2 * macroscopic_xs
    return

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
    main(*sys.argv[1:])