from foilselector.foldermanagement import *
from .__init__ import *
# other parameters
photopeak_eff_curve = HPGe_efficiency_curve_generator() # read the default file for HPGe
gamma_E = [20*1E3, 4.6*1E6] # range of gamma energy considered as detectable.

# Timing
prog_start_time = time.time()

def tprint(*msg):
    print("\n[t=+{:2.2f}s]".format(time.time()-prog_start_time), *msg)
    
def main(output_directory, *read_from_endf_directories):

    ######################### stage 1: read data from the provided file paths ############
    stage_1_outputs_exist = all(os.path.exists(os.path.join(sys.argv[-1], FULL_DECAY_INFO_FILE)),
                            os.path.exists(os.path.join(sys.argv[-1], CONDENSED_DECAY_INFO_FILE)),
                            os.path.exists(os.path.join(sys.argv[-1], MICROSCOPIC_XS_CSV)),
                            os.path.exists(os.path.join(sys.argv[-1], MAX_XS_FILE))
                            )
    if not stage_1_outputs_exist:
        from foilselector.openmcextension import EncoderOpenMC, MT_to_nuc_num, load_endf_directories, FISSION_MTS, AMBIGUOUS_MT

        prog_start_time = time.time()
        assert os.path.exists(expected_gs_path), "Must first manually create gs.csv at the output directory!"
        gs = get_gs(sys.argv[-1])
        if SORT_BY_REACTION_RATE:
            _msg = f"Output directory must already have integrated_apriori.csv in order to sort the {MICROSCOPIC_XS_CSV} in descending order of expected-radionuclide-population later on."
            assert os.path.exists(os.path.join(sys.argv[-1], 'integrated_apriori.csv')), _msg
            apriori = pd.read_csv(os.path.join(sys.argv[-1], 'integrated_apriori.csv'))['value'].values
        endf_file_list = load_endf_directories(*read_from_endf_directories)
        print(f"Loaded {len(endf_file_list)} different material files,\n")

        # First compile the decay records
        tprint("Stage 1: Compiling the decay information as decay_dict, and recording the excited-state to isomeric-state information:")
        decay_dict = OrderedDict() # dictionary of decay data
        isomeric_to_excited_state = OrderedDict() # a dictionary that translates all 
        with warnings.catch_warnings(record=True) as w_list:
            for file in tqdm(endf_file_list):
                name = str(file.target["atomic_number"]).zfill(3) + ATOMIC_SYMBOL[file.target["atomic_number"]] + str(file.target["mass_number"])
                isomeric_name = name # make a copy
                if file.target["isomeric_state"]>0: # if it is not at the lowest isomeric state: add the _e behind it too.
                    isomeric_name += "_m"+str(file.target["isomeric_state"])
                    name += "_e"+str(file.target["state"])
                isomeric_to_excited_state[isomeric_name] = name[3:] # trim the excited state name

                if file.info['sublibrary']=="Radioactive decay data": # applicable to materials with (mf, mt) = (8, 457) file section
                    dec_f = Decay.from_endf(file)
                    decay_dict[name] = extract_decay(dec_f)
        if w_list:
            print(w_list[0].filename+", line {}, {}'s:".format(w_list[0].lineno, w_list[0].category.__name__))
            for w in w_list:
                print("    "+str(w.message))
        decay_dict = sort_and_trim_ordered_dict(decay_dict) # reorder it so that it conforms to the 
        isomeric_to_excited_state = sort_and_trim_ordered_dict(isomeric_to_excited_state)
        tprint("Renaming the decay products from isomeric state names to excited state names:")
        decay_dict = rename_branching_ratio(decay_dict, isomeric_to_excited_state)

        # Save said decay records
        with open(os.path.join(sys.argv[-1], FULL_DECAY_INFO_FILE), 'w') as j:
            tprint("Saving the decay spectra as {} ...".format(FULL_DECAY_INFO_FILE))


        # turn decay records into number of counts
        tprint("Condensing each decay spectrum...")
        for name, dec_file in tqdm(decay_dict.items()):
            condense_spectrum(dec_file, photopeak_eff_curve)

        with open(os.path.join(sys.argv[-1], CONDENSED_DECAY_INFO_FILE), 'w') as j:
            tprint("Saving the condensed decay information as {} ...".format(CONDENSED_DECAY_INFO_FILE))

            
        # Then compile the Incident-neutron records
        tprint("Compiling the raw cross-section dictionary.")
        xs_dict = OrderedDict()
        for file in tqdm(endf_file_list):
            # mf10 = {}
            if file.info['sublibrary']=="Incident-neutron data":
                inc_f = IncidentNeutron.from_endf(file)
                nuc_sort_name = str(inc_f.atomic_number).zfill(3)+inc_f.name

                # get the higher-energy range values of xs as well if available.
                mf10_mt5 = MF10(file.section.get((10, 5), None)) # default value = None if (10, 5 doesn't exist.)
                for (izap, isomeric_state), xs in mf10_mt5.items():
                    atomic_number, mass_number = divmod(izap, 1000)
                    if atomic_number>0 and mass_number>0: # ignore the weird products that means nothing meaningful
                        isomeric_name = ATOMIC_SYMBOL[atomic_number]+str(mass_number)
                        if isomeric_state>0: 
                            isomeric_name += "_m"+str(isomeric_state)
                        e_name = isomeric_to_excited_state.get(isomeric_name, isomeric_name.split("_")[0]) # return the ground state name if there is no corresponding excited state name for such isomer.
                        long_name = nuc_sort_name+"-"+e_name+"-MT=5"
                        xs_dict[long_name] = xs

                # get the normal reactions, found in mf=3
                for mt, rx in inc_f.reactions.items():
                    if any([(mt in AMBIGUOUS_MT), (mt in FISSION_MTS), (301<=mt<=459)]):
                        continue # skip the cases of AMBIGUOUS_MT, fission mt, and heating information. They don't give us useful information about radionuclides produced.

                    append_name_list, xs_list = extract_xs(inc_f.atomic_number, inc_f.mass_number, rx, tabulated=True)
                    # add each product into the dictionary one by one.
                    for name, xs in zip(append_name_list, xs_list):
                        xs_dict[nuc_sort_name + '-' + name] = xs

        # memory management
        print("Deleting endf_file_list since it will no longer be used in this script, in an attempt to reduce memory usage")
        del endf_file_list; gc.collect()
        # del decay_dict

        xs_dict = sort_and_trim_ordered_dict(xs_dict)

        tprint("Collapsing the cross-section to the group structure specified by 'gs.csv' and then saving it as '{}' ...".format(MICROSCOPIC_XS_CSV))
        sigma_df, max_xs = collapse_xs(xs_dict, gs)
        with open(os.path.join(sys.argv[-1], MAX_XS_FILE), "w") as j:
            json.dump(max_xs, j)
        del xs_dict; gc.collect()

        if not SHOW_SEPARATE_MT_REACTION_RATES:
            sigma_df = merge_identical_parent_products(sigma_df)
            # Need to make merge_identical_parent_products to work with max_xs as well.

        if SORT_BY_REACTION_RATE:
            sigma_df = sigma_df.loc[ary(sigma_df.index)[np.argsort(sigma_df.values@apriori)[::-1]]]
        print(f"Saving the cross-sections in the required group structure to file as '{MICROSCOPIC_XS_CSV}'...")
        sigma_df.to_csv(os.path.join(sys.argv[-1], MICROSCOPIC_XS_CSV))
        # saves the number of radionuclide produced per (neutron cm^-2) of fluence flash-irradiated in that given bin.

        # save parameters at the end.
        PARAMS_DICT = dict(HPGe_eff_file=HPGe_eff_file,
                gamma_E=gamma_E,
                FISSION_MTS=FISSION_MTS,
                AMBIGUOUS_MT=AMBIGUOUS_MT,
                SORT_BY_REACTION_RATE=SORT_BY_REACTION_RATE,
                SHOW_SEPARATE_MT_REACTION_RATES=SHOW_SEPARATE_MT_REACTION_RATES,
                CONDENSED_DECAY_INFO_FILE=CONDENSED_DECAY_INFO_FILE,
                FULL_DECAY_INFO_FILE=FULL_DECAY_INFO_FILE,
                MAX_XS_FILE=MAX_XS_FILE
                )
        PARAMS_DICT.update({sys.argv[0]+" argv": sys.argv[1:]})
        save_parameters_as_json(sys.argv[-1], PARAMS_DICT)
        tprint("Stage 1: Data reading from", *sys.argv[1:], "complete!")
    else:
        tprint(f"Assuming that Stage 1 is complete. Reading {MICROSCOPIC_XS_CSV} as sigma_df and {CONDENSED_DECAY_INFO_FILE} as decay_dict...")
        from foilselector.openmcextension import unserialize_dict
        sigma_df = get_microscopic_cross_sections_df(sys.argv[-1])
        with open(os.path.join(sys.argv[-1], CONDENSED_DECAY_INFO_FILE)) as j:
            decay_dict = json.load(j)
            decay_dict = unserialize_dict(decay_dict)
        with open(os.path.join(sys.argv[-1], MAX_XS_FILE)) as j:
            max_xs = MaxSigma(json.load(j))

    #########################################################################
    ################################ stage_2 ################################
    #########################################################################

    from foilselector.constants import BARN, MM_CM
    from foilselector.simulation.decay import decay_mat_exp_num_decays, decay_mat_exp_population_convolved, build_decay_chain_tree, linearize_decay_chain

    tprint("Stage 2: Calculating information about each reaction:")

    apriori_flux, apriori_fluence = get_apriori(sys.argv[-1], IRRADIATION_DURATION)

    # get the production rate for each reaction, and build that into a dataframe (which will be expanded upon)
    population = pd.DataFrame({'production of primary product per reactant atom':(sigma_df.values*BARN) @ apriori_fluence}, index=sigma_df.index)
    del sigma_df; gc.collect() # sigma_df is now used and will not be called again; remove it from memory to reduce memory usage/ stop stuffing up RAM.

    # create containers to contain the calculated activities
    # detected_counts_per_primary_product = {}
    population_pre, population_post = defaultdict(dict), defaultdict(dict) # key = the name of the subchain
    total_counts, total_counts_post_meas, activity_pre, activity_post, cnt_rate_pre, cnt_rate_post = {}, {}, {}, {}, {}, {}

    tprint("Calculating the expected number of photopeak counts for each type of product created:")
    product_set = set([parent_product_mt.split("-")[1] for parent_product_mt in population.index])
    PRE_MEASUREMENT  = IRRADIATION_DURATION + TRANSIT_DURATION
    POST_MEASUREMENT = IRRADIATION_DURATION + TRANSIT_DURATION + MEASUREMENT_DURATION

    for product in tqdm(product_set):
        detected_counts_per_primary_product = []
        for subchain in linearize_decay_chain(build_decay_chain_tree(product, decay_dict)):
            new_pathway = {"pathway" : "-".join(subchain.names),
                            "counts during measurement": decay_mat_exp_num_decays(
                                subchain.branching_ratios,
                                subchain.decay_constants,
                                IRRADIATION_DURATION,
                                PRE_MEASUREMENT,
                                POST_MEASUREMENT
                                )*subchain.countable_photons,
                            "counts in 1 hr immediately after measurement": decay_mat_exp_num_decays(
                                subchain.branching_ratios,
                                subchain.decay_constants,
                                IRRADIATION_DURATION,
                                POST_MEASUREMENT,
                                POST_MEASUREMENT+3600
                                )*subchain.countable_photons
                            }
            population_pre[product][new_pathway["pathway"]] = decay_mat_exp_population_convolved(
                                subchain.branching_ratios,
                                subchain.decay_constants,
                                IRRADIATION_DURATION,
                                PRE_MEASUREMENT)
            population_post[product][new_pathway["pathway"]] = decay_mat_exp_population_convolved(
                                subchain.branching_ratios,
                                subchain.decay_constants,
                                IRRADIATION_DURATION,
                                POST_MEASUREMENT)

            new_pathway["decay rate pre-measurement" ] = subchain.decay_constants[-1] * population_pre[product][new_pathway["pathway"]]
            new_pathway["decay rate post-measurement"] = subchain.decay_constants[-1] * population_post[product][new_pathway["pathway"]]
            new_pathway["count rate pre-measurement" ] = subchain.countable_photons * new_pathway["decay rate pre-measurement" ]
            new_pathway["count rate post-measurement"] = subchain.countable_photons * new_pathway["decay rate post-measurement"]

            detected_counts_per_primary_product.append(new_pathway)

        total_counts[product] = sum([path["counts during measurement"] for path in detected_counts_per_primary_product])
        total_counts_post_meas[product] = sum([path["counts in 1 hr immediately after measurement"] for path in detected_counts_per_primary_product])
        activity_pre[product]  = sum([path["decay rate pre-measurement" ] for path in detected_counts_per_primary_product])
        activity_post[product] = sum([path["decay rate post-measurement"] for path in detected_counts_per_primary_product])
        cnt_rate_pre[product]  = sum([path["count rate pre-measurement" ] for path in detected_counts_per_primary_product])
        cnt_rate_post[product] = sum([path["count rate post-measurement"] for path in detected_counts_per_primary_product])

    # clean out reactions that can't be detected.
    tprint("Removing all reactions whose products is the same as the product (i.e. elastic scattering reactions):")
    population = population[ary([parent_product_mt.split("-")[0] != parent_product_mt.split("-")[1] for parent_product_mt in population.index])]
    # add the total counts of gamma photons detectable per primary product column

    tprint("Matching the reactions to their decay product count...")
    # add the final counts measured by detector PPP column
    rearranged_total_cnts = ary([total_counts[parent_product_mt.split("-")[1]] for parent_product_mt in population.index])
    population['final counts measured by detector PPP'] = rearranged_total_cnts

    # sort by activity and remove all nans
    tprint("Re-ordering the dataframe by the final counts measured by detector PPP and removing the entries with zero counts...")
    population.sort_values('final counts measured by detector PPP', inplace=True, ascending=False)
    population = population[population["final counts measured by detector PPP"]>0.0] # keeping only those with positive counts.

    # save the population breakdown in total_counts
    tprint("Saving the population breakdowns as .json files...")
    all_significant_products = ordered_set([parent_product_mt.split("-")[1] for parent_product_mt in population.index])
    with open(os.path.join(sys.argv[-1], PRE_MEASUREMENT_POPULATION_FILE), "w") as j:
        json.dump({product : population_pre[product] for product in all_significant_products} , j, cls=EncoderOpenMC)
        # del population_pre; gc.collect()
    with open(os.path.join(sys.argv[-1], POST_MEASUREMENT_POPULATION_FILE), "w") as j:
        json.dump({product : population_post[product] for product in all_significant_products}, j, cls=EncoderOpenMC)
        # del population_post; gc.collect()

    # add the rest of the information
    rearranged_post_cnt_meas = ary([total_counts_post_meas[parent_product_mt.split("-")[1]] for parent_product_mt in population.index])
    rearranged_activity_pre  = ary([ activity_pre[parent_product_mt.split("-")[1]] for parent_product_mt in population.index])
    rearranged_activity_post = ary([activity_post[parent_product_mt.split("-")[1]] for parent_product_mt in population.index])
    rearranged_cnt_rate_pre  = ary([ cnt_rate_pre[parent_product_mt.split("-")[1]] for parent_product_mt in population.index])
    rearranged_cnt_rate_post = ary([cnt_rate_post[parent_product_mt.split("-")[1]] for parent_product_mt in population.index])
    rearranged_max_xs        = ary([max_xs[parent_product_mt] for parent_product_mt in population.index])
    population["counts accumulated in the 1 hour following the detection period PPP"] = rearranged_post_cnt_meas # column used to quickly extrapolate the post-measurement decay rate
    population["activity before measurement PPP"] = rearranged_activity_pre
    population["activity after measurement PPP" ] = rearranged_activity_post
    population["detector count rate before measurement PPP"] = rearranged_cnt_rate_pre
    population["detector count rate after measurement PPP" ] = rearranged_cnt_rate_post
    population["max microscopic cross-section"] = rearranged_max_xs

    tprint("Saving as 'counts.csv'...")
    try:
        population.to_csv(os.path.join(sys.argv[-1], 'counts.csv'), index_label='rname')
    except ZeroDivisionError:
        tprint("Minor issue when trying to print the values which are too small. Plese wait for a couple more minutes...")
        not_uncertain_columns = ["production of primary product per reactant atom",
                            "max microscopic cross-section",]
        for col in population.columns:
            if col not in not_uncertain_columns:
                # when trying to express uncertainties.core.Variable using the  __str__ method, it will try to factorize it.
                # But if the GCD between the norminal value and uncertainty is rounded down to 1E-323 or smaller, it will lead to ZeroDivisionError.
                floating_point_problem = population[col]<12.5E-324 # therefore we set all small values
                # this method is harsher than it needs to because it forces items
                # with nominal value < 12.5E-324 but error > 12.5E-324 to be 0 as well,
                # even though they are perfectly expressible as strings without errors.
                population[col][floating_point_problem] = 0
        population.to_csv(os.path.join(sys.argv[-1], 'counts.csv'), index_label='rname')
    save_parameters_as_json(sys.argv[-1], dict(
        IRRADIATION_DURATION=IRRADIATION_DURATION,
        TRANSIT_DURATION=TRANSIT_DURATION,
        MEASUREMENT_DURATION=MEASUREMENT_DURATION,
        # PRE_MEASUREMENT_POPULATION_FILE=PRE_MEASUREMENT_POPULATION_FILE,
        # POST_MEASUREMENT_POPULATION_FILE=POST_MEASUREMENT_POPULATION_FILE,
        )
    )
    tprint("Run complete. See results in 'counts.csv'.")

if __name__=="__main__":
    import sys
    main(*sys.argv[1:])
