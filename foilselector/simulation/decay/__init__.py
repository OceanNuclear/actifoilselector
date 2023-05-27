from .bateman import *
from .chaining import *
from collections import namedtuple

# define a few functions that doesn't make sense to be in either, and also depends on the function in both.

def cnt_and_cnt_rates_for_each_pathway(root_product, decay_info_dict, a, b, c):
    """Create a list of dictionaries related to all decay pathways of one root-product,
    where each dictionary describes the decay rate and count rate of that pathway.
    root_product: the product that originates
    decay_info_dict: the decay info dictionary formed by applying
        foilselector.openmcextension.condense_spectrum_copy on each item of the decay_dict extracted by sparsely_load_xs_and_decay_dict"""
    rates_per_root_product = []
    for subchain in linearize_decay_chain(build_decay_chain_tree(root_product, decay_info_dict)):
        new_pathway = {"pathway" : "-".join(subchain.names),
                        "counts during measurement": mat_exp_num_decays(
                            subchain.branching_ratios,
                            subchain.decay_constants,
                            a,
                            b,
                            c
                            )*subchain.countable_photons,
                        "counts in 1 hr immediately after measurement": mat_exp_num_decays(
                            subchain.branching_ratios,
                            subchain.decay_constants,
                            a,
                            c,
                            c+3600
                            )*subchain.countable_photons
                        }
        pre_population_of_this_pathway = mat_exp_population_convolved(
                            subchain.branching_ratios,
                            subchain.decay_constants,
                            a,
                            b)
        post_population_of_this_pathway = mat_exp_population_convolved(
                            subchain.branching_ratios,
                            subchain.decay_constants,
                            a,
                            c)

        # population_pre[root_product][new_pathway["pathway"]] = pre_population_of_this_pathway
        # population_post[root_product][new_pathway["pathway"]] = post_population_of_this_pathway

        new_pathway["decay rate pre-measurement" ] = subchain.decay_constants[-1] * pre_population_of_this_pathway
        new_pathway["decay rate post-measurement"] = subchain.decay_constants[-1] * post_population_of_this_pathway
        new_pathway["count rate pre-measurement" ] = subchain.countable_photons * new_pathway["decay rate pre-measurement" ]
        new_pathway["count rate post-measurement"] = subchain.countable_photons * new_pathway["decay rate post-measurement"]

        rates_per_root_product.append(new_pathway)
    return rates_per_root_product

Summed_reaction_rates_results = namedtuple("Summed_reaction_rates_results",
                                            ["total_counts",
                                            "total_counts_in_the_1h_following_measurement",
                                            "activity_pre",
                                            "activity_post",
                                            "pre_measurement_count_rate",
                                            "post_measurement_count_rate"])

def summed_cnt_and_rates(root_product, decay_info_dict, a, b, c):
    """
    Calculates the outputted quantities of a SINGLE root-product atom.
    Parameters
    ----------
    root_product: the name of the product that is produced.
    decay_info_dict, a, b, c: see cnt_and_cnt_rates_for_each_pathway

    Returns
    -------
    See Summed_reaction_rates_results.
    """
    detected_counts_per_primary_product = cnt_and_cnt_rates_for_each_pathway(root_product, decay_info_dict,
                                                                            a=a,
                                                                            b=b,
                                                                            c=c
                                            )
    total_counts = sum([path["counts during measurement"] for path in detected_counts_per_primary_product])
    total_counts_in_the_1h_following_measurement = sum([path["counts in 1 hr immediately after measurement"] for path in detected_counts_per_primary_product])
    activity_pre  = sum([path["decay rate pre-measurement" ] for path in detected_counts_per_primary_product])
    activity_post = sum([path["decay rate post-measurement"] for path in detected_counts_per_primary_product])
    pre_measurement_count_rate  = sum([path["count rate pre-measurement" ] for path in detected_counts_per_primary_product])
    post_measurement_count_rate = sum([path["count rate post-measurement"] for path in detected_counts_per_primary_product])
    return Summed_reaction_rates_results(
            total_counts,
            total_counts_in_the_1h_following_measurement,
            activity_pre,
            activity_post,
            pre_measurement_count_rate,
            post_measurement_count_rate
            )