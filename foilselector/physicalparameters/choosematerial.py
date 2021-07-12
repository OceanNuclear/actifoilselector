from foilselector.openmcextension.constants import NATURAL_ABUNDANCE, atomic_mass, ATOMIC_NUMBER
from .filepaths import PHYSICAL_PROP_FILE as default_physical_prop_flie
import numpy as np
import pandas as pd

"""Module used to convert materials parameters into nuclide informations and choose materials"""

"""Text processing functions solely for supporting the physical properties inquiring functions below"""
def split_at_cap(string):
    """Cut a string up into chunks;
    each chunk of the string contains a single element and its fraction,
    which is expected to start with a capital letter.
    """
    composition_list = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        string = string.replace(letter, ","+letter)
    string = string.lstrip(',')
    return string.split(',')

def get_fraction(composition):
    """Given a string like Al22.2, get the fraction i.e. 22.2
    """
    numerical_value = "".join([i for i in composition[1:] if (i.isnumeric() or i=='.')])
    return float(numerical_value) if numerical_value!="" else 1.0

def extract_elem_from_string(composition):
    """ Given a string like Al22.2, get the element symbol, i.e. Al
    """
    return "".join([i for i in composition[:2] if i.isalpha()])

def expand_bracket(solid):
    """
    expand the compositions of an sub-alloy inside the bracket
    """
    if len(left_bracket_split:=solid.split("("))>1:
        in_brackets, behind_brackets = left_bracket_split[1].split(")")
        # find the text inside the bracket and the NUMBER that immediately follow the bracket.
        fraction_of_bracketed = ""

        for char in behind_brackets: # we will be looping through a COPY of the behind_brackets, so we won't have to worry about mutating the string while looping.
            if char.isnumeric() or char==".":
                fraction_of_bracketed += char
            if char == "/":
                break # stop when it hits a '/' just in case there are more components behind it.
            behind_brackets = behind_brackets[1:] # that character is considered "Used", and is removed from the behind_brackets variable.
        fraction_of_bracketed = float(fraction_of_bracketed)

        modified_in_bracket = ""
        num_compositions = len(in_brackets.split("+"))
        for composition in in_brackets.split("+"):
            modified_in_bracket += composition
            modified_in_bracket += str(fraction_of_bracketed/num_compositions)
            modified_in_bracket += "/"
        modified_in_bracket = modified_in_bracket.rstrip('/') #remove the extranous "/"
        
        return "".join([left_bracket_split[0], modified_in_bracket, behind_brackets])
    else:
        return solid

"""Functions to find matching physical properties about the element"""
def get_natural_average_atomic_mass(elem, verbose=False, **kwargs):
    """ returns the average atomic mass of the element when it is in its naturally occurring ratio of isotopic abundances.
    **kwargs is plugged into the numpy.isclose function ot modify rtol and atol.
    """
    isotopic_abundance = {isotope: abundance for isotope, abundance in NATURAL_ABUNDANCE.items() if extract_elem_from_string(isotope)==elem}
    if sum(isotopic_abundance.values())!=1.0:
        if verbose:
            print("Expected the isotopic abundances of {} to sum to 1.0, got {} instead.".format(elem, sum(isotopic_abundance.values())))
        if not np.isclose(sum(isotopic_abundance.values()), 1.0, **kwargs):
            raise ValueError("The sum of abundances for element '{}' = {} is not close to unity.".format(elem, sum(isotopic_abundance.values())))
    return sum([atomic_mass(isotope)*abundance for isotope, abundance in isotopic_abundance.items()])

def get_elemental_fractions(formula):
    if ("/" not in formula) or ("(Atomic%)" in formula):
        # use atomic percentage
        compositions = split_at_cap(formula.replace("(Atomic%)", "")) # could've used removesuffix in python 3.9; but I've not installed that yet.
        elements = [extract_elem_from_string(comp) for comp in compositions]

        raw_mole_ratios = [get_fraction(comp) for comp in compositions] # interpret them as mole ratios

        elemental_fractions = {elem:mole_ratio/sum(raw_mole_ratios) for mole_ratio, elem in zip(raw_mole_ratios, elements)}
    else:
        # is an alloy, use wt. percentage
        compositions = expand_bracket(formula).split("/")
        elements = [extract_elem_from_string(comp) for comp in compositions]

        weight_fractions = [get_fraction(comp)/100.0 for comp in compositions] # interpret them as weight fraction
        average_mass = 1/sum([weight/get_natural_average_atomic_mass(elem) for weight, elem in zip(weight_fractions, elements)]) # obtain the average mass which is then used as a multiplier below:

        elemental_fractions = { elem: weight/get_natural_average_atomic_mass(elem)*average_mass for weight, elem in zip(weight_fractions, elements) }
    return elemental_fractions

def convert_elemental_to_isotopic_fractions(dict_of_elemental_fractions):
    isotopic_fractions = {}
    for elem, fraction in dict_of_elemental_fractions.items():
        matching_isotopes = [iso for iso in NATURAL_ABUNDANCE.keys() if extract_elem_from_string(iso)==elem]
        for isotope in matching_isotopes:
            isotopic_fractions[isotope] = NATURAL_ABUNDANCE[isotope]*fraction
    return isotopic_fractions

def get_average_atomic_mass_from_isotopic_fractions(dict_of_isotopic_fractions):
    return sum([atomic_mass(isotope)*fraction for isotope, fraction in dict_of_isotopic_fractions.items()])

def get_average_atomic_mass_from_elemental_fractions(dict_of_elemental_fractions):
    return sum([get_natural_average_atomic_mass(elem)*fraction for elem, fraction in dict_of_elemental_fractions.items()])

"""Functions to find out quantities about the relevant material"""
def pick_material(species, expanded_physical_prop_df):
    """Choose the material with the highest atomic fraction of this species
    """
    material = expanded_physical_prop_df[species].idxmax()
    if expanded_physical_prop_df[species][material]>0:
        return expanded_physical_prop_df.loc[material]
    else:
        dummy_data_series = pd.Series({col: float('nan') for col in expanded_physical_prop_df.columns})
        dummy_data_series['Formula'] = 'N/A'
        dummy_data_series.name = 'Missing'
        return dummy_data_series

def get_physical_property(phy_prop_file=default_physical_prop_flie):
    physical_prop = pd.read_csv(phy_prop_file, index_col=[0])
    return physical_prop

def pick_all_usable_materials(species, expanded_physical_prop_df):
    """Pick all materials that has even a trace of the species of interest.
    """
    material = expanded_physical_prop_df[species]>0
    return expanded_physical_prop_df.loc[material]