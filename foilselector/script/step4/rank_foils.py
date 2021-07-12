# system and display
import os, sys, itertools, functools, json, time
from tqdm import tqdm
from operator import add, itemgetter
from dataclasses import dataclass
# numpy
from numpy import array as ary; import numpy as np
from numpy import log as ln
from numpy import mean
import numpy.linalg as la
# dataframe
import pandas as pd
# plotting
from matplotlib import pyplot as plt
import seaborn as sns
from collections import OrderedDict
# number of permutation/combination functions, for calculating the run time when optimizing
from math import factorial as fac
from scipy.special import comb as n_choose_k
from scipy.special import perm as n_perm_k

# uncertainties is a module that openmc uses. Thus the values are 
from uncertainties.unumpy import nominal_values
# local libraries
from misc_library import (BARN,
                        MM_CM,
                        ordered_set,
                        get_apriori,
                        unserialize_dict,
                        PHYSICAL_PROP_FILE,
                        get_parameters_json,
                        get_physical_property,
                        save_parameters_as_json,
                        unserialize_pd_DataFrame)
from comb_sum_ranker import top_n_sums_of_dict

SATURATION_COUNT_RATE = 10000 # maximum number of gamma countable accurately per second
# MAX_THICKNESS = 0.1 # mm
# MAX_THICKNESS should be determined by the thickness at which no self-shielding occurs;
# which should in turn be determined by the max(microscopic cross-sections)
"""
if RANK_BY_DETERMINANT: calculate a non-singular matrix representing the curvature,
And then performance of each set of foil combination is then quantified by the determinant of its curvature matrix.
else: use various approach.
"""
COUNT_THRESHOLD = 300 # reactions with less than this many count will not be considered as valid contributors that increase the precision of the unfolding.
FOIL_AREA = 2.5*2.5 # cm^2
RANK_BY_DETERMINANT = False
LIST_PRICE = True # also list the price in the resulting dictionary.
MAX_INTERACTION_PROBABILITY = 0.1
IGNORE_SELF_SHIELDING_OF_OTHER_REACTIONS = True

if RANK_BY_DETERMINANT:
    def D_KL(test_spectrum, apriori_spectrum):
        """ Calculates the Kullback-Leibler divergence.
        """
        fDEF = apriori_spectrum/sum(apriori_spectrum)
        f = test_spectrum/sum(test_spectrum)
        from autograd import numpy as ag_np
        log_ratio = ag_np.nan_to_num(ag_np.log(fDEF/f))
        return ag_np.dot(fDEF, log_ratio)
else:
    def D_KL(test_spectrum, apriori_spectrum):
        fDEF = apriori_spectrum/sum(apriori_spectrum)
        f = test_spectrum/sum(test_spectrum)
        log_ratio = np.nan_to_num(ln(fDEF/f))
        return np.dot(fDEF, log_ratio)

def fractional_curvature_matrix(R, S_N_inv, apriori):
    """
    parameters
    ----------
    R : response matrix, with the appropriate thickness information alreay included.
    S_N_inv : inverse of the covariance matrix of the reaction rates vector N
    apriori : apriori_fluence of 
    """
    apriori_multiplier = np.diag(apriori)
    return apriori_multiplier @ R.T @ S_N_inv @ R @ apriori_multiplier

def orthogonality_matrix(R):
    """
    The matrix that measures how easily one (absolute) unit of flux change in one bin is going to be (mis-)interpreted
    as a one unit of flux change in another bin in the unfolded flux.
    
    Parameters
    ----------
    R: Response matrix (m*n), no need to normalize. The thickness does not matter.

    Returns
    -------
    orthogonality matrix, dimension = (n*n) where n = number of bins.
    """
    return la.pinv(R) @ R

def confusion_matrix_linear(R, apriori):
    """
    Defunct function to calculate
    the fractional change in the unfolded flux given a fractional change in the measured flux.
    d(unfolded flux fractional change)/d(measured flux fractional change) | at measured flux = a priori
    outputs a matrix (Jacobian).
    """
    return np.diag(1/apriori) @ orthogonality_matrix(R) @ np.diag(apriori)

def confusion_matrix_log(R, apriori):
    """
    a matrix stating how much fractional change does a bin in the unfolded spectrum experience when
        another bin leads a fractional change.
    Parameters and Returns: same as orthogonality_matrix
    """
    Rinv_R = la.pinv(R) @ R # would've been identity matrix if fully determined
    RR_ap_vec = Rinv_R @ apriori # the a priori vector after being transformed by the Rinv_R
    if (RR_ap_vec==0).any():
        return la.pinv(np.diag(RR_ap_vec)) @ Rinv_R @ np.diag( apriori )
    else:
        return np.diag(1/RR_ap_vec) @ Rinv_R @ np.diag( apriori )

# there should be one more confusion matrix definition, which is even more complicated

def confusion_matrix_correctly_identified_diagonal(R, apriori, matrix_type=confusion_matrix_log):
    return np.clip(np.diag(matrix_type(R, apriori)), 0, 1).sum()/R.shape[1]

def scalar_curvature(R, S_N_inv, apriori):
    """
    Assume no correlations between different reaction channels, which, of course is a big lie, but I'll think about the consequence of this lie later.
    This function takes the del^2 chi2.
    This is different from taking the determinant of the chi2 curvature matrix.
    Parameters
    ----------
    R : response matrix associated with the foil, that turns the incident neutron flux into measured gamma counts.
    S_N_inv : covariance matrix of the measured gamma counts, which is expected to be obtained by np.diag(1/(R @ apriori)).

    Explanation: Curvature DOES decrease wrt. number of counts. This is a feature of a Poisson system, not a bug.
    The more absolute number of count a channel has, the harder it is to differentiate one extra count from the rest.
    (law of deminishing return)
    """
    # S_N_inv = np.diag(1/counts) # = la.inv(np.diag(counts))
    return np.diag(fractional_curvature_matrix(R, S_N_inv, apriori)).sum()

class ABCFoil():
    def __add__(self, foil_like):
        assert isinstance(foil_like, (ABCFoil, FoilSet)), "Can only add Foil/FoilSet onto another Foil/FoilSet to create another FoilSet."
        return FoilSet(self, foil_like)

    def get_reaction_filter(self, threshold_count=COUNT_THRESHOLD):
        """
        Remove reactions whose total counts (at the given foil volume) doesn't reach the required counts.
        """
        detectable_reactions = self.counts >= threshold_count
        return detectable_reactions

@dataclass
class _Foil(ABCFoil):
    """
    Class that contains all the useful information about a foil.
    use CMS for all units (cm, cm^2, cm^-2 s^-1, etc.) for all numbers stated;
        except for microscopic_xs, which uses barns (1E-24 cm^2).
    This include the slice of the microscopic cross-section dataframe of reaction useful that foil,
        and physical properties relevant to that foil.
    count_rates_per_primary_product : photopeaks' counting rate at the beginning and end of the measurement.
        we're'assuming the foil's max count rate will be achieved at either the beginning or the end.
    counts_per_primary_product: total number of counts accumulated across the entire measurement duration
    max_micro_xs : maxium microscopic cross-section listed by the ENDF file before it was collapsed by ReadData.py
    """
    material_name : str
    number_densities : pd.DataFrame
    microscopic_xs : pd.DataFrame
    primary_product_per_reactant : pd.Series # already encodes fluence and irradiation duration information in it.
    counts_per_primary_product : pd.Series # encodes decay information
    count_rates_per_primary_product: pd.Series
    max_micro_xs : pd.Series
    area : float = FOIL_AREA
    melting_point: float = np.nan
    price : float = np.nan

    def __repr__(self):
        return "<{} foil at {}>".format(self.material_name, id(self))

class Foil(_Foil):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Merging can only be done after thickness is determined so that we freeze the macroscopic cross-section.

        Two methods:
        1. record all reactions, determine thickness, merge them, and then filter out those that doesn't give enough counts.
        2. Set a lenient thickness, discard all reactions that still gives not enough counts
            (thus their self-shielding effects won't be considered when re-considering the thickness),
            and then determine thickness -> merge.
        Using option 1 here...
        """
        if IGNORE_SELF_SHIELDING_OF_OTHER_REACTIONS:
            # method one: check the value.
            # set a very lenient thickness, limited only by the count-rate.
            self.thickness = self.determine_thickness(max_interaction_prob=None)
            # self.filter_reactions()# filter away at reactions that still has too few counts even if
            # we max out the thickness while ignoring self-shielding.
            # However this does have the danger of ignoring the effect of reactions with tall sharp peaks but very low total counts.
        # self.thickness = self.determine_thickness() # now we can take into account self-shielding.
        # self.merge_duplicates_and_filter()

    # thickness independent properties
    @property
    def macroscopic_xs(self):
        """
        macroscopic cross-section for that reaction
        Generated at getattr time
        """
        return BARN * self.microscopic_xs.mul(self.number_densities, axis=0) # expressed in cm^-1

    @property
    def counts_per_volume(self):
        """
        calculate the number of counts collected at the photopeak for the entire measurement duration
            per volume of foil (cm^-3).
        """
        # multiplying together two pd.Series with matching indices
        p_p_per_volume = self.number_densities.mul( self.primary_product_per_reactant, axis=0 ) # p_p = primary product
        counts_per_volume_series = p_p_per_volume.mul( self.counts_per_primary_product, axis=0 )
        counts_per_volume_series.name = "counts cm^-3"
        return counts_per_volume_series

    def limit_thickness_by_count_rates(self, saturation_count_rate=SATURATION_COUNT_RATE):
        p_p_per_volume = self.primary_product_per_reactant.mul(self.number_densities, axis=0) # p_p = primary product
        # multiplying together two pd.Series with matching indices: safest way to do so is by turning both into np.array
        count_rates_per_thickness = self.area * self.count_rates_per_primary_product.mul(p_p_per_volume, axis=0)
        # sum over the count rate, and choose the larger of the two (at the beginning or at the end)
        max_count_rate_per_thickness = count_rates_per_thickness.values.sum(axis=0).max()
        if max_count_rate_per_thickness>0: # non-zero value
            count_rate_limited_thickness = saturation_count_rate/max_count_rate_per_thickness
        else: # edge-case handling
            count_rate_limited_thickness = np.inf
        return count_rate_limited_thickness

    def limit_thickness_by_self_shielding(self, max_interaction_prob):
        max_macroscopic_xs = BARN * self.max_micro_xs.mul(self.number_densities, axis=0) # two pd.Series multiplied together.
        # 1 - e^(Sigma * t) = probability of absorption # where Sigma = macroscopic cross-section
        # inverting the equation yields the following
        self_shielding_limited_thickness = -ln(1-max_interaction_prob)/max_macroscopic_xs
        return self_shielding_limited_thickness

    # determining the thickness
    def determine_thickness(self, max_interaction_prob=MAX_INTERACTION_PROBABILITY):
        """
        Provide a maximum thickness that the foil can have,
        limited by the SATURATION_COUNT_RATE and the self_shielding_limited_thickness.
        assume the max count rate is achieved at either the very beginning or the end of the irradiation duration.

        use max_interaction_prob = None to not account for the self-shielding when deciding the thickness.
        """
        count_rate_limited_thickness = self.limit_thickness_by_count_rates()
        if max_interaction_prob is None:
            return count_rate_limited_thickness # ignore self-shielding..
        else:
            self_shielding_limited_thickness = self.limit_thickness_by_self_shielding(max_interaction_prob)

            # choose a thickness where self-shielding is not a problem (max_interaction_prob below requirement).
            if len(self_shielding_limited_thickness)>0:
                min_thickness = min(count_rate_limited_thickness, ary(self_shielding_limited_thickness).min())
            else:
                min_thickness = count_rate_limited_thickness
            if np.isinf(min_thickness):
                return 0
            else:
                return min_thickness

    # thickness dependent properties
    @property
    def response_per_unit_flux(self):
        """
        Get the response per unit flux for that given thickness and area of foil 
        """
        volume = self.area * self.thickness
        response = volume * self.macroscopic_xs.multiply(self.counts_per_primary_product, axis=0)
        return response
        
    @property
    def counts(self):
        counts_series = self.counts_per_volume * self.area * self.thickness
        counts_series.name = "counts"
        return counts_series

    def merge_duplicates_and_filter(self, threshold_count=COUNT_THRESHOLD):
        """
        Merge reaction with the same products, since the same radioisotope generated from two different reactions
        in the same foil cannot be differentiated from one another by the gamma detector.
        """
        if len(self.microscopic_xs.index)==0:
            # take care of the zero-reaction case.
            return FoilFixedDimensions(self.material_name, self.area, self.thickness,
                self.microscopic_xs, self.counts_per_primary_product, self.counts, 
                self.price, melting_point=self.melting_point, number_densities=self.number_densities.to_dict())
        selection_matrix, name_series = duplicates_matrix_and_new_names(self.microscopic_xs.index)

        merged_macroscopic_xs = pd.DataFrame(selection_matrix @ self.macroscopic_xs.values, index=name_series)
        cnt_per_p_p_array = self.counts_per_primary_product.values
        merged_counts_per_primary_product = pd.Series([cnt_per_p_p_array[row][0] for row in selection_matrix],
                    index=name_series, name=self.counts_per_primary_product.name)
        merged_counts = pd.Series(selection_matrix@self.counts.values, index=name_series, name=self.counts.name)
        detectable_reactions = merged_counts >= threshold_count

        new_foil = FoilFixedDimensions(
            # scalars
            self.material_name,
            self.area,
            self.thickness,
            # non-scalars
            merged_macroscopic_xs[detectable_reactions],
            merged_counts_per_primary_product[detectable_reactions],
            merged_counts[detectable_reactions],
            # other properties
            price = self.price,
            melting_point = self.melting_point,
            number_densities = {parent_product_mt.split("-")[0]:num_density for parent_product_mt, num_density in self.number_densities.items()}
            )
        return new_foil

    def filter_reactions(self, threshold_count=COUNT_THRESHOLD):
        detectable_reactions = self.get_reaction_filter(threshold_count)
        for pd_attr in ("number_densities",
                "microscopic_xs",
                "primary_product_per_reactant",
                "counts_per_primary_product",
                "count_rates_per_primary_product",
                "max_micro_xs"):
            setattr(self, pd_attr, getattr(self, pd_attr)[detectable_reactions])
        return

def duplicates_matrix_and_new_names(parent_product_mt_index):
    parent, reactant_names, mts_long = ary([parent_product_mt.split("-") for parent_product_mt in parent_product_mt_index]).T
    mts = ary([mt.split("=")[1] for mt in mts_long])

    equality_matrix = ary([reactant_names==reactant for reactant in reactant_names])
    condensed_matrix, name_series = [], []
    for index, row in enumerate(equality_matrix):
        if row[:index].sum()==0:
            condensed_matrix.append(row)

            new_name =  "({})".format( ",".join(parent[row]) )
            new_name += "-{}-MT=".format(reactant_names[index])
            new_name += "({})".format( ",".join(mts[row]) )
            name_series.append(new_name)

    return ary(condensed_matrix), name_series

@dataclass
class FoilFixedDimensions(ABCFoil):
    # scalars
    material_name : str
    area : float
    thickness : float
    # non-scalars
    macroscopic_xs : pd.DataFrame
    counts_per_primary_product : pd.Series
    counts : pd.Series
    # other properties
    price : float = np.nan
    melting_point : float = np.nan
    number_densities : dict = None

    @property
    def response_per_unit_flux(self):
        """
        Get the response per unit flux for that given thickness and area of foil 
        """
        volume = self.area * self.thickness
        response = volume * self.macroscopic_xs.multiply(self.counts_per_primary_product, axis=0)
        return response

    def __repr__(self):
        return "<{} foil with {}cm^2 x {}cm at {}>".format(self.material_name, self.area, self.thickness, id(self))

class FoilSet():
    def __init__(self, *foils):
        # quantities necessary for deciding its use
        self.response_per_unit_flux = []
        self.counts = []
        # other quantities related to the foil
        self.material_name = []
        self.price = []
        self.melting_point = []
        self.area = []
        self.thickness = []

        for foil_like in foils:
            self.response_per_unit_flux.append(foil_like.response_per_unit_flux)
            self.counts.extend( list(foil_like.counts.values) )
            for attr in "material_name", "thickness", "price", "area", "melting_point":
                curr_attr = getattr(self, attr)
                new_attr = getattr(foil_like, attr)
                curr_attr.extend(new_attr) if isinstance(new_attr, list) else curr_attr.append(new_attr)
                setattr(self, attr, curr_attr)
        self.response_per_unit_flux = np.concatenate(self.response_per_unit_flux)

    def __repr__(self):
        return "<foil set of {} foils at {}>".format(len(self.material_name), id(self))

    def __add__(self, foil_like):
        return FoilSet(self, foil_like)

    def get_price_expr(self):
        num_POA, total_price = 0, 0.0
        for price in self.price:
            if np.isnan(price):
                num_POA += 1
            else:
                total_price += price
        total_price = str(total_price)
        if num_POA: # if there are non-zero number of missing records
            total_price += " + {} POA's".format(num_POA)
        return total_price
        
    def get_min_melting_point(self):
        return min(self.melting_point)

def get_foilset_condensed_name(foil_list, space_then_symbol_in_bracket=True):
    if space_then_symbol_in_bracket:
        return "-".join(i.split()[1].strip("()") for i in foil_list)
    else:
        return "-".join(i[:2] for i in foil_list)

if RANK_BY_DETERMINANT:
    import autograd
    D_KL_hessian_getter = autograd.hessian(lambda x: D_KL(x, apriori_fluence.copy()))
    
def det_curvature(R, apriori, include_D_KL_contribution=RANK_BY_DETERMINANT, custom_hessian_contribution=None):
    hess_matrix = fractional_curvature_matrix(R, la.inv(np.diag(R @ apriori_fluence)))
    if include_D_KL_contribution:
        hess_matrix += D_KL_hessian_getter(apriori_fluence)
    if custom_hessian_contribution:
        hess_matrix += custom_hessian_contribution
    return la.det(hess_matrix)

chain_sum = lambda iterable: functools.reduce(add, iterable)

descending_odict = lambda d: OrderedDict(sorted(d.items(), key=itemgetter(1), reverse=True)) # key=itemgetter(1) picks out the dict value rather than the dict key to use as sorting key.

def choose_top_n_pretty(func, target_chosen_length, policy, choices, verbose=True):
    # number of possible reactions = 
    num_combinations = int(n_choose_k(len(choices), target_chosen_length))
    if type(policy)==int:
        print("Attempting to choose an optimal combination of foils by choosing the top {} foils at every move, in the solution space of {} possible combinations".format(policy, num_combinations))
        print("which should return a dictionary of length with an upper limit = {}".format(policy, np.clip(policy**target_chosen_length, num_combinations, None)))
        print("This can take up to {} evaluations ...".format( min(policy**target_chosen_length, int(n_perm_k(len(choices), target_chosen_length))) ))
    elif type(policy)==float:
        print("Attempting to choose an optimal combination of foils by choosing the top {} % of foils, in the solution space of {} possible combinations.".format(policy, num_combinations))
    return choose_top_n(func, target_chosen_length, policy, choices, verbose=verbose) # and then one can use descending_odict to find the most promising combination at next(iter( .items()))

def choose_top_n(func, target_chosen_length, policy, choices, chosen=set(), verbose=True):
    """
    Find a chosen combination of strings that give the highest output value when plugged into func,
    using the "greedy" step approach, taking the step that leads to the greatest increase in the output.
    ("Taking a step" here refers to updating the set of strings by adding one element from "choices" to "chosen".)

    Parameters
    ----------
    func: function that takes in a list of strings (all of which are present in the list variable choices) in any arbitrary order and output a scalar.
    target_chosen_length: number of elements to be placed.
    policy:
        if int: n = int(policy), n must be >=1. Will take a step forward in each of the top n most promising steps.
        if float: f = float(policy), f must be between 0 and 1. Will take a step forward on each of the top 100*f% most promising steps.

    returns
    """
    # calculate scalar outputs for all possible steps
    output_scalars = {name:func(name, *chosen) for name in choices}
    # and then choose the names to be used as the next steps.
    if type(policy)==int:
        output_scalars = descending_odict(output_scalars)
        next_steps = OrderedDict((name, out) for name, out in list(output_scalars.items())[:policy]) # names of the chosen elements
    elif type(policy)==float and (0<=policy<=1):
        out_max, out_min = max(output_scalars.values()), min(output_scalars.values())
        diff = out_max - out_min
        threshold = out_max - policy*diff
        next_steps = OrderedDict((name, out) for name, out in output_scalars.items() if out>=threshold)
    else:
        raise ValueError("policy must be either: 1. choose the top n most promising steps (int), or 2. choose the top (100*f)% most promising steps (0<=float<=1)")

    # termination condition
    if (len(chosen)+1)>=target_chosen_length:
        if verbose:
            for name, out in next_steps.items():
                print("Branch ends! Foils chosen in this branch = {} with value = {}".format([name, *chosen], out))
        return { tuple(sorted([name, *chosen])): out for name, out in next_steps.items() }

    # recursion
    else:
        combos_used_and_their_output = OrderedDict()
        for name in next_steps.keys():
            added_chosen, remaining_choices = chosen.copy(), choices.copy()
            # transfer the chosen name into the added set; and remove it from the remaining_choices set.
            added_chosen.add(name), remaining_choices.remove(name)
            if verbose:
                print("foils chosen in this branch so far = {} with value = {}".format(added_chosen, next_steps[name]))
            combos_used_and_their_output.update( choose_top_n(func, target_chosen_length, policy, remaining_choices, added_chosen, verbose) )
        return combos_used_and_their_output

def choose_top_1(func, target_chosen_length, choices, chosen=[], verbose=True):
    """
    Same as choose_top_n, but policy is fixed at int(1) and the chosen is now a list, allowing the order in which items were added to be preserved.
    """
    output_scalars = {name:func(name, *chosen) for name in choices}
    max_val = max(output_scalars.values())
    # reverse lookup dict to find what name gave the max output value.
    name = [name for name, out in output_scalars.items() if out==max_val][0]
    #shift the name from choices to chosen
    chosen, choices = chosen.copy(), choices.copy() # unlink from the list from the function call above.
    # If I don't unlink, it'll spazz out and reuse the old copy of chosen, choices.
    chosen.append(name)
    print(len(chosen))
    if verbose:
        print("foil chosen so far =", chosen)
    choices.remove(name)
    if len(chosen)>=target_chosen_length: # termination condition
        if len(chosen)>target_chosen_length:
            print("Hold up, how did you get here?")
        return (chosen, output_scalars[name])
    # recursion condition
    else:
        return choose_top_1(func, target_chosen_length, choices, chosen, verbose)

def choose_custom(func, target_chosen_length, custom_func, choices, chosen=set(), verbose=True):
    """
    same __doc__ as choose_top_1
    Parameters
    ----------
    custom_func: function that takes in a descendingly-sorted OrderedDict (sorted according to its values) and spits out a trimmed copy of this dict (preferably ordered dict).
        This OrderedDict stores loss values (values) for each set of arguments (keys) used,
        so the custom_func is expected to be choosing the most effective few items of the dict.
    """
    output_scalars = {name:func(name, *chosen) for name in choices}
    # and then choose the names to be used as the next steps.
    output_scalars = descending_odict(output_scalars)
    next_steps = custom_func(output_scalars)

    # termination condition
    if (len(chosen)+1)>=target_chosen_length:
        if verbose:
            for name, out in next_steps.items():
                print("Branch ends! Foils chosen in this branch = {} with value = {}".format([name, *chosen], out))
        return { tuple(sorted([name, *chosen])): out for name, out in next_steps.items() }

    # recursion
    else:
        combos_used_and_their_output = OrderedDict()
        for name in next_steps.keys():
            added_chosen, remaining_choices = chosen.copy(), choices.copy()
            # transfer the chosen name into the added set; and remove it from the remaining_choices set.
            added_chosen.add(name), remaining_choices.remove(name)
            if verbose:
                print("foils chosen in this branch so far = {} with value = {}".format(added_chosen, next_steps[name]))
            combos_used_and_their_output.update( choose_custom(func, target_chosen_length, custom_func, remaining_choices, added_chosen, verbose) )
        return combos_used_and_their_output

def perturb_top_1_order(func, target_chosen_length, choices, mix_length=[0, 1], verbose=True, skip_evaluation=False):
    """
    Parameters
    ----------
    func, target_chosen_length, choices, verbose: see choose_top_1.__doc__
    mix_length: for each int in mix_length, let's call that m.
        Assume that the target_chosen_length = r
        and the total number of choices = n.
        for each m,
        We will generate rCm different combinations mixed with (n-r)Cm to give rCm*(n-r)Cm total possible combinations,
            each of which preserves (r-m) of the original r chosen, removing m of them from this list, and then
            replacing them with m chosen from the negated list.
    """
    greedy_choices = choose_top_1(func, target_chosen_length, choices, verbose=verbose)[0]
    negated_choices = [c for c in choices if c not in greedy_choices]
    combos_used_and_their_output = {}
    num_combinations = [int(n_choose_k(target_chosen_length, m_len) * n_choose_k(len(choices)-target_chosen_length, m_len)) for m_len in mix_length]
    print("\nChoosing mixing lengths = {}, ".format(mix_length))
    print("\nNumber of evaluations required = {} = total of {} ".format(num_combinations, sum(num_combinations)) )
    for m_len, num_comb in zip(mix_length, num_combinations):
        # mix in m_len of the negated_choices, thus displacing m_len from the original greedy_choices
        with tqdm(total=num_comb) as progress_bar:
            for old_choices in itertools.combinations(greedy_choices, target_chosen_length - m_len):
                for new_choices in itertools.combinations(negated_choices, m_len):
                    names = tuple(sorted([*old_choices, *new_choices]))
                    if skip_evaluation:
                        combos_used_and_their_output[names] = None
                    else:
                        combos_used_and_their_output[names] = func(*names)
                    progress_bar.update(1)
    return combos_used_and_their_output

def false_apriori_generator(apriori, seed=None):
    """
    pick a wrong apriori so that the unfolding can proceed
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.poisson(apriori).astype(float)

def tprint(*msg):
    print("\n[t=+{:2.2f}s]".format(time.time()-prog_start_time), *msg)

"""
There are nCk solutions.

# Verify that "greedy" appraoch is a reasonable? done.
1. greedy appraoch -> greedy choice reordering
2. relaxation of choices: branch at each foil-num increment: 
    include n-1 more optimal choices
    -> generates at most n C k combos
3. relaxation of order: using the greedy choice ordering,
    monte carlo approach of mixing it up?
        (Because there are n! ways of mixing it up)
        genetic algorithm
""" 

PLOT_SAVE_FOLDER = "sensitivity_vs_specificity_plot/"