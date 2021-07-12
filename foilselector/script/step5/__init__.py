from .finalize_selection import *

def main(dirname):
    PLOT_CONTR = False
    PLOT_MACRO = False
    PLOT_INDIVIDUAL_SPEC = False

    """
    with open(os.path.join(dirname, "parameters_used.json")) as J:
        PARAMS_DICT = json.load(J)
    # get the user-defined foil selection, and their thicknesses
    """
    # reading data
    with open(os.path.join(dirname, "final_selection.json")) as j:
        selected_foils = json.load(j)
    with open(os.path.join(dirname, "decay_radiation.json")) as j:
        decay_dict = unserialize_dict(json.load(j))
    with open(os.path.join(dirname, "decay_info.json")) as j:
        decay_info = unserialize_dict(json.load(j)) # the decay counts given the stated starting and ending measurement times.
    with open(os.path.join(dirname, "post-measurement_population.json")) as j:
        post_measurement_population = json.load(j)
    abs_efficieny_curve = HPGe_efficiency_curve_generator()
    PARAMS_DICT = get_parameters_json(dirname)
    IRRADIATION_DURATION = PARAMS_DICT["IRRADIATION_DURATION"]
    b, c = IRRADIATION_DURATION+PARAMS_DICT["TRANSIT_DURATION"], IRRADIATION_DURATION+PARAMS_DICT["TRANSIT_DURATION"]+PARAMS_DICT["MEASUREMENT_DURATION"]
    APRIORI_FLUX, APRIORI_FLUENCE = get_apriori(dirname, IRRADIATION_DURATION)
    sigma_df = pd.read_csv(os.path.join(dirname, "microscopic_xs.csv"), index_col=[0])
    gs = pd.read_csv(os.path.join(dirname, 'gs.csv')).values

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
        volume_cm3 = foil["thickness (cm)"] * foil["area (cm^2)"]
        number_density = foil["number density (cm^-3)"]
        num_reactants = { species : num_density * volume_cm3 for species, num_density in number_density.items() }
        print("FISPACT input file should use the value of:")
        print(fispact_FUEL_dict_string(num_reactants))

        total_counts_dict = foil[[k for k in foil.keys() if k.startswith("total counts at thickness =")][0]]

        # ax = plt.subplot()
        foil_spectrum_w_eff = GammaSpectrum()

        for parents_product_mts in total_counts_dict.keys():
            product = parents_product_mts.split("-")[1]

            macroscopic_xs, num_reactions = get_macro_and_num_reactions(parents_product_mts, number_density)
            if PLOT_MACRO:
                ax = plt.subplot()
                ax.semilogx(gs.flatten(), np.repeat(macroscopic_xs/(macroscopic_xs).sum(), 2))
                ax.set_ylabel("Fraction of contribution to the reaction rates")
                ax.set_xlabel(r"$E_n$ (eV)")
                ax.set_title(parents_product_mts)
                plt.show()
            if PLOT_CONTR:
                ax = plt.subplot()
                ax.semilogx(gs.flatten(), np.repeat(num_reactions/(num_reactions).max(), 2))
                ax.set_ylabel("max-normalized macroscopic cross-section")
                ax.set_xlabel(r"$E_n$ (eV)")
                ax.set_title(parents_product_mts)
                plt.show()

            decay_tree = build_decay_chain_tree(product, decay_info)
            pprint.pprint(trim_tree(decay_tree), sort_dicts=False)

            final_spec = GammaSpectrum()
            for subchain in linearize_decay_chain(decay_tree):
                # decay_mat_exp_population_convolved(subchain.branching_ratios, subchain.decay_constants, IRRADIATION_DURATION, t)
                num_decays_recorded_per_reaction = decay_mat_exp_num_decays(subchain.branching_ratios, subchain.decay_constants, IRRADIATION_DURATION, b, c)
                num_decays_recorded = (num_reactions).sum() * num_decays_recorded_per_reaction

                spectrum_of = subchain.names[-1]
                spectrum = decay_dict[spectrum_of].get("spectra", {})
                # final_spec += (SingleDecayGammaSignature(spectrum, spectrum_of) * num_decays_recorded)
                final_spec += (SingleDecayGammaSignature(spectrum, subchain.names) * num_decays_recorded)

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
        print()
