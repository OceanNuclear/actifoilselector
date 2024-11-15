# typical system/python stuff
import warnings
from tqdm import tqdm
from collections import namedtuple
import gc

# typical python numerical stuff
from numpy import array as ary
import numpy as np
import pandas as pd
from numpy import typing as npt

# openmc stuff
from openmc.data import Tabulated1D
# uncertainties
from uncertainties.core import Variable
from uncertainties import nominal_value as nom

# local modules
from foilselector.openmcextension.table import Integrate, Tab1DExtended
from foilselector.openmcextension.gamma import LineTuple
from foilselector.generic import ordered_set
from foilselector.selfshielding import MaxSigma

__all__ = [
    "condense_spectrum_copy",
    "collapse_xs",
    "collapse_single_xs",
    "merge_identical_parent_products",
    "simplify_spectrum_copy",
    "flatten_photon_spectrum",
]


def condense_spectrum_copy(
    dec_file, photopeak_eff_curve, gamma_lims=[20 * 1e3, 4.6 * 1e6]
):
    """
    We will explicitly ignore all continuous distributions because they do not show up as clean gamma lines.
    Note that the ENDF-B/decay/ directory stores a lot of spectra (even those with very clear lines) as continuous,
        so it may lead to the following method ignoring it.
        The workaround is just to use another library where the evaluators aren't so lazy to not use the continuous_flag=='both' :/
    """
    count = Variable(0.0, 0.0)
    if ("gamma" in dec_file["spectra"]) and ("discrete" in dec_file["spectra"]["gamma"]):
        norm_factor = dec_file["spectra"]["gamma"]["discrete_normalization"]
        for gamma_line in dec_file["spectra"]["gamma"]["discrete"]:
            # if the gamma_line is within the wanted energy range
            if np.clip(gamma_line["energy"].n, *gamma_lims) == gamma_line["energy"].n:
                count += (
                    photopeak_eff_curve(gamma_line["energy"])
                    * gamma_line["intensity"]
                    * norm_factor
                )
    if ("xray" in dec_file["spectra"]) and ("discrete" in dec_file["spectra"]["xray"]):
        norm_factor = dec_file["spectra"]["xray"]["discrete_normalization"]
        for xray_line in dec_file["spectra"]["xray"]["discrete"]:
            # if the xray_line is within the wanted energy range
            if np.clip(xray_line["energy"].n, *gamma_lims) == xray_line["energy"].n:
                additional_counts = (
                    photopeak_eff_curve(xray_line["energy"])
                    * xray_line["intensity"]
                    * norm_factor
                )
                if not additional_counts.s <= additional_counts.n:
                    additional_counts = Variable(
                        additional_counts.n, additional_counts.n
                    )  # clipping the uncertainty so that std never exceed the mean. This takes care of the nan's too.
                count += additional_counts
    dec_file_copy = dec_file.copy()
    # replace the spectra attribute with countable_photon attribute.
    del dec_file_copy["spectra"]
    dec_file_copy["countable_photons"] = (
        count  # countable photons per decay of this isotope
    )
    return dec_file_copy


def simplify_spectrum_copy(
    dec_file, isotope_name, photopeak_eff_curve, gamma_lims=[20 * 1e3, 4.6 * 1e6]
):
    """
    We will explicitly ignore all continuous distributions because they do not show up as clean gamma lines.
    Note that the ENDF-B/decay/ directory stores a lot of spectra (even those with very clear lines) as continuous,
        so it may lead to the following method ignoring it.
        The workaround is just to use another library where the evaluators aren't so lazy to not use the continuous_flag=='both' :/
    """
    spectra = dec_file["spectra"]
    lines = []  # container of lines extracted

    if ("xray" in spectra) and (
        "discrete" in spectra["xray"]
    ):  # find dec_file['spectrum']['xray']['discrete']
        norm_factor = spectra["xray"]["discrete_normalization"]
        for xray_line in spectra["xray"]["discrete"]:
            # if the xray_line is within the wanted energy range
            if np.clip(xray_line["energy"].n, *gamma_lims) == xray_line["energy"].n:
                counts_per_decay = (
                    photopeak_eff_curve(xray_line["energy"])
                    * xray_line["intensity"]
                    * norm_factor
                )
                if (
                    not counts_per_decay.s <= counts_per_decay.n
                ):  # clipping the uncertainty so that std never exceed the mean. This takes care of the nan's too.
                    # it's only a problem for x rays due to the low energies.
                    counts_per_decay = Variable(counts_per_decay.n, counts_per_decay.n)
                lines.append(
                    LineTuple(
                        xray_line["energy"].n, counts_per_decay, isotope_name, "xray"
                    )
                )
    if ("gamma" in spectra) and (
        "discrete" in spectra["gamma"]
    ):  # find dec_file['spectrum']['gamma']['discrete']
        norm_factor = spectra["gamma"]["discrete_normalization"]
        for gamma_line in spectra["gamma"]["discrete"]:
            # if the gamma_line is within the wanted energy range
            if np.clip(gamma_line["energy"].n, *gamma_lims) == gamma_line["energy"].n:
                lines.append(
                    LineTuple(
                        gamma_line["energy"].n,
                        gamma_line["intensity"] * norm_factor,
                        isotope_name,
                        "gamma",
                    )
                )
    return lines


def collapse_xs(xs_dict, gs_ary):
    """
    Calculates the group-wise cross-section (production rate of product per unit flux) in the given group-structure
    by averaging the cross-section within each bin.
    """
    collapsed_sigma, max_xs_dict = {}, MaxSigma()
    with warnings.catch_warnings(record=True) as w_list:
        for parent_product_mt, xs in tqdm(
            xs_dict.items(),
            desc="Collapsing the cross-sections to the desired group-structure:",
        ):
            # perform the integration
            I = Integrate(xs)
            sigma = I.definite_integral(*gs_ary.T) / np.diff(gs_ary, axis=1).flatten()

            collapsed_sigma[parent_product_mt] = sigma
            max_xs_dict[parent_product_mt] = max(xs.y)
    # ignore the w_list of caught warnings
    return pd.DataFrame(collapsed_sigma).T, max_xs_dict

def collapse_single_xs(xs_entry: Tabulated1D, gs_array: npt.NDArray):
    """
    Parameters
    ----------
    xs_entry:
        A single openmc.data.Tabulated1D instance found in
        `openmc.data.IncidentNeutron.from_endf(file).reactions[...].xs["0K"]`
    gs_array:
        Group structure showing the lower and upper bound of each bin,
        shape = len(gs) * 2.

    Returns
    -------
    cross-section:

    """
    return integrate(xs_entry).definite_integral(*gs_array.T) / np.diff(gs_array, axis=1).flatten()

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
    parent_product_list, mt_list = (
        ary(parent_product_list),
        ary(mt_list),
    )  # make them into array to make them indexible.

    partial_reaction_array = loose_collection_of_rx.values
    parent_product_all = ordered_set(parent_product_list)

    sigma_unique = {}
    print(
        "Condensing the sigma_xs dataframe to merge together reactions with identical (parent, product) pairs:"
    )
    for parent_product in tqdm(parent_product_all):
        matching_reactions = parent_product_list == parent_product
        mt_name = "-MT=({})".format(",".join(mt_list[matching_reactions]))
        sigma_unique[parent_product + mt_name] = partial_reaction_array[
            matching_reactions
        ].sum(axis=0)
    del loose_collection_of_rx
    del partial_reaction_array
    gc.collect()
    return pd.DataFrame(sigma_unique).T

class DiscreteRadiation(namedtuple("Radiation", ["energy", "intensity", "source"])):
    """
    Attributes
    ----------
    energy: openmc.core.Variable
        mean energy of this discrete radiation line.
    intensity: openmc.core.Variable
        number of this radiation line released per decay of the immediate parent.
    source: str
        description of where the radiation originated from.
        decay radiation type, immediate parent's name, and decay mode inducing the
        release of this radiation. e.g. "gamma from Y101 beta-"
    """
    pass

class ContinuousRadiationDistribution(namedtuple("RadiationDistribution", ["distribution", "source"])):
    """
    Attributes
    ----------
    distribution: `foilselector.openmcextension.table.Tab1DExtended`
        Distribution of gamma-lines(x=energy (eV), y=intensity) released per decay of
        the immediate parent.
        Area under the distribution integrates to the value given by
        `nom(openmc_decay_spectrum[radiation_type]["continuous_normalization"])`.
    source: str
        description of where the radiation originated from.
        decay radiation type, immediate parent's name, and decay mode inducing the
        release of this radiation. e.g. "gamma from Y101 beta-"
    """

def flatten_photon_spectrum(openmc_decay_spectrum: dict, isotope_name: str) -> tuple[list[DiscreteRadiation], list[ContinuousRadiationDistribution]]:
    """
    Turn a openmc.data.Decay.from_endf(...).spectra from a nested dictionary into
    a lists of photon (xray and gamma) lines.

    Parameters
    ----------
    openmc_decay_spectrum:
        dictionary obtained by openmc.data.Decay.from_endf(isotope).spectra.
    isotope_name:
        name of the isotope to which the spectrum belongs.
    
    Returns
    -------
    discrete_spec: list[3-tuple]
        DiscreteRadiation(energy (eV), intensity, source description)

    continuous_spec: list[2-tuple]
        ContinuousRadiationDistribution(photon energy distribution, source description)
    """
    discrete_spec, continuous_spec = [], []
    def extend_discrete_lines(discrete: list[dict], discrete_normalization: Variable, source: str):
        """Flatten a decay["spectrum"][radiation_type]["discrete"]"""
        if nom(discrete_normalization):
            for line in discrete:
                discrete_spec.append(
                    (line["energy"], line["intensity"] * discrete_normalization, f"{source} from {isotope_name} {','.join(line['from_mode'])}")
                )

    if "xray" in openmc_decay_spectrum:
        if "discrete" in openmc_decay_spectrum["xray"]:
            extend_discrete_lines(
                openmc_decay_spectrum["xray"]["discrete"],
                openmc_decay_spectrum["xray"]["discrete_normalization"],
                "xray"
            )
        if "continuous" in openmc_decay_spectrum["xray"]:
            prob_table = openmc_decay_spectrum["xray"]["continuous"]["probability"]
            xray_dist = Tab1DExtended(
                prob_table.x,
                prob_table.y * nom(openmc_decay_spectrum["xray"]["continuous_normalization"]),
                breakpoints=prob_table.breakpoints,
                interpolation=prob_table.interpolation,
            )

            continuous_spec.append(
                (xray_dist, f"xray from {isotope_name} {','.join(line['from_mode'])}",)
            )
    if "gamma" in openmc_decay_spectrum:
        if "discrete" in openmc_decay_spectrum["gamma"]:
            extend_discrete_lines(
                openmc_decay_spectrum["gamma"]["discrete"],
                openmc_decay_spectrum["gamma"]["discrete_normalization"],
                "gamma"
            )
        if "continuous" in openmc_decay_spectrum["gamma"]:
            prob_table = openmc_decay_spectrum["gamma"]["continuous"]["probability"]
            gamma_dist = Tab1DExtended(
                prob_table.x,
                prob_table.y * nom(openmc_decay_spectrum["gamma"]["continuous_normalization"]),
                breakpoints=prob_table.breakpoints,
                interpolation=prob_table.interpolation,
            )

            # num_counts = Integrate(gamma_dist).definite_integral(*minmax(gamma_dist))
            # warnings.warn(
            #     "Continuous gamma-ray distribution found in the decay gamma-ray spectrum"
            #     f"! This distribution sums up to {num_counts * nom()} gamma-rays "
            #     f"released per decay of {isotope_name}."
            # )
            continuous_spec.append(
                (gamma_dist, f"gamma from {isotope_name} {','.join(line['from_mode'])}",)
            )
            
    return discrete_spec, continuous_spec