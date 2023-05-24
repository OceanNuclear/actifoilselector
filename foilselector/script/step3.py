"""
Assumption:
Each unactivated foil will be placed in the irradiation environment for a fixed amount of time (--irradiation-duration) that you have decided upon, 
and its transferral onto the gamma detector also takes a predetermined amount of time (--transit-duration).
The gamma acquisition also lasts a pre-determined amount of time (--measurement-duration),
    at a fixed distance (so that the efficiency curve can be approximated by --photopeak-efficiency).
______________________________________________________________________________________________________
In the following text, "detectible decay" = decay with sufficiently long half-lives (on the order of minutes to hours), and emits a non-negligible number of gamma photons per decay.

Here we will list a series of scenarios, and comments about the difficulty of differentiating them from one another.
    - Any gamma spectroscopy system worth its salt can tell the difference between these:
        Different root products with entirely different decay pathways:
            A -n-> C -dec-> E -> ... -> G
            B -n-> D -dec-> F -> ... -> H
        E.g.1: As long as C->G and D->H both contains "detectible decays", then A->C can be differentiated from B->D by ANY gamma spectroscopy system.
    - However, if the two decay pathways converge, then we have a more complex situation:
        Different rectants that yield different root products, but the decay chain of root products overlaps:
            A -n-> C -dec-> E -dec-> ... -dec-> G -> ...
            B -n-> D -dec-> F -dec-> ... -dec-> G -> ...
        E.g.2: if C->G and/or D->G contain(s) detectible decays, then A->C can be differentitated from B->D by ANY gamma spectroscopy system.
        E.g.3: else if neither C->G and D->G contains detectible decays, but they have sufficiently different effective half-lives, then a time-resolved gamma spectroscopy system is needed.
        E.g.4: else C->G and D->G contains no detectible decays and both have very short half-lives (less than 1 second), then they cannot be differentiated from each other by any gamma spectroscopy system.
    - Finally, in the following scenario, the two reaction is impossible to tell apart:
            A -n-> C -> ...
            B -n-> C -> ...
        E.g.5: Different reactants that yield the same root products
    Summary: To recap, these are the capabilities of a time-resolved
    ____________________________________________________
    |Scenario|non-time-resolved HPGe|time-resolved HPGe|
    |________|______________________|__________________|
    |E.g.1   |         Yes          |       Yes        |
    |________|______________________|__________________|          
    |E.g.2   |         Yes          |       Yes        |
    |________|______________________|__________________|
    |E.g.3   |          No          |       Yes        |
    |________|______________________|__________________|
    |E.g.4   |          No          |       No         |
    |________|______________________|__________________|
    |E.g.5   |          No          |       No         |
    |________|______________________|__________________|
    
In this foil selector script, for simplicity, we assume that when a time-resolved system is used, all reaction (that has passed the --min-counts-threshold in step4) with different root product will always be differentiable from each other.
i.e. We assume that the scenarios in e.g.4 does not occur frequently enough to affect the accuracy of the foil selection efficiency computation.
"""
import json, os
from tqdm import tqdm
import pandas as pd
from numpy import array as ary
from foilselector.foldermanagement import *
from foilselector.constants import BARN
from foilselector.simulation.decay import summed_cnt_and_rates
from foilselector.optimizer import max_num_atoms_from_count_rate_limit
from foilselector.selfshielding import sigma_to_thickness
from uncertainties import nominal_value as nom

import argparse
from pathlib import Path
parg = argparse.ArgumentParser(description="""Calculate the number of decays from each reaction.""")

parg.add_argument('-f', '--a-priori-flux'    , type=Path, required=True, help="""newline-separated file listing the total flux expected in each bin of ascending energy.
The number of bins (n) must match the number of bin boundaries (n+1) in the previous step.""")
parg.add_argument('-R', '--max-gamma-count-rate', type=float, default=10000, help="Maximum pulse rate that the gamma detector can handle without losing its resolution.")
parg.add_argument('-I', '--irradiation-duration', type=float, required=True, help="Number of seconds the foils spend getting activated in the neutron field (in the beamline/reactor).")
parg.add_argument('-T', '--transit-duration', type=float, required=True, help="Number of seconds the required to get the foil out of the neutron field onto the gamma detector.")
parg.add_argument('-D', '--measurement-duration', type=float, required=True, help="Number of seconds the detector spend acquiring a spectrum of the activated foil sample.")

if __name__=="__main__":
    cl_arg = parg.parse_args()
    
    expected_files = [".atomic_composition.json", ".decay_info.json", ".sigma_df.csv", ".self-shielding.json"] # ".decay_radiation.json" isn't needed
    assert all(os.path.exists(file) for file in expected_files), f"step1 must've been ran first at the current directory to generate the following list of files:\n{' '.join(expected_files)}"
    POST_IRRADIATION = cl_arg.irradiation_duration
    PRE_MEASUREMENT  = cl_arg.irradiation_duration + cl_arg.transit_duration
    POST_MEASUREMENT = cl_arg.irradiation_duration + cl_arg.transit_duration + cl_arg.measurement_duration
    assert os.path.exists(cl_arg.a_priori_flux), f"the -f, --a-priori-flux argument ({cl_arg.a_priori_flux}) must be a valid file path!"

    print(f"Loading {expected_files}...", end="\r")
    processed_composition = read_atomic_composition_json()
    decay_info = read_decay_info()
    ss_info = read_self_shielding()
    sigma_df = read_microscopic_cross_section_csv()

    print("Removing all reactions whose products are the same as the reactants (i.e. elastic scattering reactions):")
    _trivial_rx = ary(sigma_df.index)[[parent_product_mt.split("-")[0] == parent_product_mt.split("-")[1] for parent_product_mt in sigma_df.index]]
    # eslastic scattering reactions are 'trivial', i.e. imparts no observable change.
    sigma_df.drop(_trivial_rx, axis='index', inplace=True)

    flux = read_flux(cl_arg.a_priori_flux)
    assert len(flux)==sigma_df.shape[1], "The number of bins in the a priori neutron spectrum the must match the group structure (n+1 boundaries)."
    fluence = ary(flux) * cl_arg.irradiation_duration
    save_vector(fluence, ".fluence.txt") # needed in step 4: optimization
    print(f"Loading {expected_files}... Done!")

    ##################################################################
    # sub-step 1: multiply with flux to get reaction rates
    population = pd.DataFrame({'production of primary product per reactant atom':(sigma_df.values*BARN) @ fluence}, index=sigma_df.index)
    # sigma_df has already merged identical reactant-product entries, so each reaction channel is already unique.

    ##################################################################
    # sub-step 2: get the individual root-product's decay rates and photon-emission rates, as well as their counts.

    # create containers to contain the calculated activities
    total_counts = {}
    total_counts_post_meas = {}
    activity_pre = {}
    activity_post = {}
    cnt_rate_pre = {}
    cnt_rate_post = {}
    product_set = set([parent_product_mt.split("-")[1] for parent_product_mt in population.index])

    # And then write them into a dataframe (smaller than sigma_df).
    for product in tqdm(product_set, desc="Calculating the expected number of photopeak counts for each type of product created:"):
        count_calculation_results = summed_cnt_and_rates(product, decay_info,
                                                        a=POST_IRRADIATION,
                                                        b=PRE_MEASUREMENT,
                                                        c=POST_MEASUREMENT )
        total_counts[product] = count_calculation_results.total_counts
        total_counts_post_meas[product] = count_calculation_results.total_counts_in_the_1h_following_measurement
        activity_pre[product]  = count_calculation_results.activity_pre
        activity_post[product] = count_calculation_results.activity_post
        cnt_rate_pre[product]  = count_calculation_results.pre_measurement_count_rate
        cnt_rate_post[product] = count_calculation_results.post_measurement_count_rate
    # add the total counts of gamma photons detectable per primary product column

    print("Matching the reactions to their decay product count...")
    # add the first column of the dataframe
    population['final counts measured by detector PPP'] = ary([total_counts[parent_product_mt.split("-")[1]] for parent_product_mt in population.index])
    # PPP stands for 'Per Primary Product' (i.e. root-product).

    # sort an apply threshold on the first column of the dataframe,
    # so that nans are removed,
    # and reactions are sorted by the detectability.
    print("Re-ordering the dataframe by the final counts measured by detector PPP and removing the entries with zero counts...")
    population.sort_values('final counts measured by detector PPP', inplace=True, ascending=False)

    # shorthands.
    parent_product_mt_list = list(population.index)
    product_list = [parent_product_mt.split("-")[1] for parent_product_mt in parent_product_mt_list]
    reactant_list = [parent_product_mt.split("-")[0] for parent_product_mt in parent_product_mt_list]

    # add the rest of the information
    population["counts accumulated in the 1 hour following the detection period PPP"] = ary([total_counts_post_meas[prod] for prod in product_list])
    # column used to quickly extrapolate the post-measurement decay rate.
    # Used only for radiological protection purpose in legacy code.
    # no harm in adding in; but it will be a lot of effort to re-add if I've removed it.
    # So I've decided to keep it.
    population["activity before measurement PPP"]           = ary([ activity_pre[prod] for prod in product_list])
    population["activity after measurement PPP" ]           = ary([activity_post[prod] for prod in product_list])
    population["detector count rate before measurement PPP"]= ary([ cnt_rate_pre[prod] for prod in product_list])
    population["detector count rate after measurement PPP" ]= ary([cnt_rate_post[prod] for prod in product_list])
    population["max microscopic cross-section"]             = ary([ss_info[parent_product_mt] for parent_product_mt in parent_product_mt_list])

    print("Saving as '.counts.csv'...")
    save_counts_csv(population, comments=[f" irradiation_duration = {cl_arg.irradiation_duration}",
        f" transit_duration = {cl_arg.transit_duration}",
        f" measurement_duration = {cl_arg.measurement_duration}",
        ])
    # the .counts.csv file is kept only for diagnostic purposes for now, and can be deleted.
    # But in the future we can expand the program by basing more functionalities off it.

    ##################################################################
    #####################################################################
    # sub-step 3: determine max. foil dimensions to not cause self-shielding (done by limiting the thickness)
    # or exceed the detector's max useful count rate (controlled by total volume).
    foil_dimension_limits = {}
    for foil_name, foil_comp in processed_composition.items():
        max_sigma_so_far, init_count_rate_per_reactant = 0, 0
        for component, fraction in foil_comp.items():
            matching_list = ary([r==component for r in reactant_list])
            proportional_counts_xs = population[matching_list] * fraction
            _this_max_sigma = proportional_counts_xs["max microscopic cross-section"].max()
            max_sigma_so_far = max([max_sigma_so_far, _this_max_sigma]) # still working in BARN
            init_count_rate_per_reactant += (
                proportional_counts_xs['production of primary product per reactant atom'] * proportional_counts_xs['detector count rate before measurement PPP']
                ).sum()
        # # determine the max. number of atoms to have in the foil
        # max_thickness_in_atoms = sigma_to_thickness_in_num_atoms(max_sigma_so_far, foil_comp["number density (cm^-3)"])
        # foil_dimension_limits[foil_name] = {"max. thickness in number of atoms":max_thickness_in_atoms, "max. number of atoms": max_num_atoms}

        # Simply let the user determine the thickness themselves, but provide them with the maximum microscopic cross-section.
        max_num_atoms = max_num_atoms_from_count_rate_limit(nom(init_count_rate_per_reactant), cl_arg.max_gamma_count_rate)
        foil_dimension_limits[foil_name] = {"max. sigma(E) (energy-dependent cross-section) (barns)":max_sigma_so_far,
                                            "max. number of atoms": max_num_atoms}
    with open("dimension_upper_limits.json", "w") as j:
        # and write the resulting limits into a human-readable dictionary.
        json.dump(foil_dimension_limits, j, indent=1)

    # bearing in mind that, as long as the thickness of the foil is kept below the limit of self-shielding conern,
    # it does not affect the efficiency computation.
    # the user can then use their own number density to determine their desired foil sizes,