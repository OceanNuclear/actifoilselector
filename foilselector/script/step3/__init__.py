__all__= ["get_reaction_rates"]
def run():
    def tprint(*msg):
        print("\n[t=+{:2.2f}s]".format(time.time()-prog_start_time), *msg)

    from .get_reaction_rates import *
    # typical system/python stuff
    import sys, os, time, json
    from tqdm import tqdm

    # numerical packages
    from numpy import array as ary; import numpy as np
    from numpy import log as ln; from numpy import sqrt
    import pandas as pd

    # openmc/specialist packages
    import openmc
    import uncertainties
    from uncertainties.core import Variable
    #collections
    from collections import namedtuple, OrderedDict
    # foilselector
    from misc_library import save_parameters_as_json, unserialize_dict
    from misc_library import BARN, MM_CM, get_apriori, decay_mat_exp_num_decays, Bateman_convolved_generator

    # main script    
    prog_start_time = time.time()
    apriori_flux, apriori_fluence = get_apriori(sys.argv[-1], IRRADIATION_DURATION)

    with open(os.path.join(sys.argv[-1], CONDENSED_DECAY_INFO_FILE), 'r') as f:
        decay_dict = json.load(f)
        decay_dict = unserialize_dict(decay_dict)
    sigma_df = pd.read_csv(os.path.join(sys.argv[-1], 'response.csv'), index_col=[0])

    count_contribution_per_primary_product, total_counts_per_primary_product = {}, {}
    tprint("Calculating the expected number of photopeak counts for each type of product created:")
    product_set = set([parent_product_mt.split("-")[1] for parent_product_mt in sigma_df.index])
    for product in tqdm(product_set):
        count_contribution_per_primary_product[product] = [{
                    'pathway': '-'.join(subchain.names),
                            # (# of photons detected per nuclide n decayed) = (# of photons detected per decay of nuclide n) * lambda_n * \int(population)dT
                    # 'counts':Bateman_num_decays_factorized(subchain.branching_ratios, subchain.decay_constants,
                    'counts':np.product(subchain.branching_ratios[1:])*
                            decay_mat_exp_num_decays(subchain.decay_constants, 
                            IRRADIATION_DURATION,
                            IRRADIATION_DURATION+TRANSIT_DURATION,
                            IRRADIATION_DURATION+TRANSIT_DURATION+MEASUREMENT_DURATION
                            )*subchain.countable_photons,
                    } for subchain in linearize_decay_chain(build_decay_chain(product, decay_dict))]
        total_counts_per_primary_product[product] = sum([path["counts"] for path in count_contribution_per_primary_product[product]])
    # get the production rate for each reaction
    population = pd.DataFrame({'production of primary product per reactant atom':(sigma_df.values*BARN) @ apriori_fluence}, index=sigma_df.index)
    # clean out reactions that can't be detected.
    tprint("Removing all reactions whose products is the same as the product (i.e. elastic scattering reactions):")
    population = population[ary([parent_product_mt.split("-")[0] != parent_product_mt.split("-")[1] for parent_product_mt in population.index])]
    # add the total counts of gamma photons detectable per primary product column
    tprint("Matching the reactions to their decay product count...")
    gamma_counts_at_measurement_per_reactant = ary([total_counts_per_primary_product[parent_product_mt.split("-")[1]] for parent_product_mt in sigma_df.index])
    # add the final counts accumulated per reactant atom column
    population['final counts accumulated per reactant atom'] = gamma_counts_at_measurement_per_reactant * population['production of primary product per reactant atom']
    # sort by activity and remove all nans
    tprint("Re-ordering the dataframe by the final counts accumulated per reactant atom and removing the entries with zero counts...")
    population.sort_values('final counts accumulated per reactant atom', inplace=True, ascending=False)
    population = population[gamma_counts_at_measurement_per_reactant>0.0] # keeping only those with positive counts.

    if GUESS_MATERIAL:
        from misc_library import extract_elem_from_string, pick_material, PHYSICAL_PROP_FILE, get_physical_property
        # read the physical property file to get the number densities.
        tprint(f"Reading ./{os.path.relpath(PHYSICAL_PROP_FILE)} to extract the physical parameters about various solids.")
        physical_prop = get_physical_property(PHYSICAL_PROP_FILE)

        # select the default materials and get its relevant parameters
        default_material, partial_number_density = [], []

        tprint("Selecting the default material to be used:")
        for parent_product_mt in tqdm(population.index):
            parent = parent_product_mt.split('-')[0]
            if parent[len(extract_elem_from_string(parent)):]=='0': # take care of the species which are a MIXED natural composition of materials, e.g. Gd0
                parent = parent[:-1]
            if ENRICH_TO_100_PERCENT: # allowing enrichment means 100% of that element being made of the specified isotope only
                parent = extract_elem_from_string(parent)
            if parent not in physical_prop.columns:
                # if there isn't a parent material
                default_material.append('Missing (N/A)')
                partial_number_density.append(0.0)
                continue
            material_info = pick_material(parent, physical_prop)
            default_material.append(material_info.name+" ("+material_info['Formula']+")")
            partial_number_density.append(material_info['Number density-cm-3'] * material_info[parent]) # material_info[parent] chooses the fraction of atoms which is made of .

        population["default material"] = default_material
        population["partial number density (cm^-3)"] = partial_number_density
        population["gamma counts per volume of foil (cm^-3)"] = population["final counts accumulated per reactant atom"] * population["partial number density (cm^-3)"]
        population["gamma counts per unit thickness of foil (mm^-1)"] = population["gamma counts per volume of foil (cm^-3)"] * MAX_AREA * MM_CM# assuming the area = Foil Area
        tprint("Re-ordering the dataframe according to the counts per volume...")
        population.sort_values("gamma counts per volume of foil (cm^-3)", inplace=True, ascending=False) # sort again, this time according to the required volume

    tprint("Saving as 'counts.csv'...")
    try:
        population.to_csv(os.path.join(sys.argv[-1], 'counts.csv'), index_label='rname')
    except ZeroDivisionError:
        tprint("Minor issue when trying to print the values which are too small. Plese wait for a couple more minutes...")
        uncertain_columns = ["final counts accumulated per reactant atom"]
        if GUESS_MATERIAL:
            uncertain_columns+= ["gamma counts per volume of foil (cm^-3)", "gamma counts per unit thickness of foil (mm^-1)"]
        for col in uncertain_columns:
            less_than_mask = population[col]<12.5E-324 # when trying to express uncertainties.core.Variable using the  __str__ method, it will try to factorize it.
            # But if the GCD (between the norminal value and its counterpart) is rounded down to 1E-323 or smaller, it will lead to ZeroDivisionError.
            population[col][less_than_mask] = uncertainties.core.Variable(0.0, 0.0)
        population.to_csv(os.path.join(sys.argv[-1], 'counts.csv'), index_label='rname')

    # save parameters at the end.
    save_parameters_as_json(sys.argv[-1], dict(
        IRRADIATION_DURATION=IRRADIATION_DURATION,
        TRANSIT_DURATION=TRANSIT_DURATION,
        MEASUREMENT_DURATION=MEASUREMENT_DURATION,
        MAX_AREA=MAX_AREA,
        ENRICH_TO_100_PERCENT=ENRICH_TO_100_PERCENT,
        )
    )
    tprint("Run complete. See results in 'counts.csv'.")