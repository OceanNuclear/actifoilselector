from collections import defaultdict
from foilselector.openmcextension.constants import isotopes as ISOTOPES, ATOMIC_NUMBER

def specify_isotopic_composition(user_composition_input):
    """Expand a user input (typed in a user-friendly syntax) dictionary of atomic composition
    into a purely "isotope:fraction" format.
    (user may input in either 'isotope:fraction' or 'chemical-element:fraction' format.)
    This requires breaking down elements into its natural isotopic abundances,
    and then normalizing the output composition so that it adds up to unit.

    Parameters
    ----------
    user_composition_input: dict
    """
    unnormed_composition = defaultdict(float)
    for element_or_isotope, atomic_fraction in user_composition_input.items():
        if any(c.isnumeric() for c in element_or_isotope):
            isotope = element_or_isotope # is an isotope
            unnormed_composition[isotope]+= atomic_fraction
        else:
            element = element_or_isotope # is an element
            for isotope, sub_fraction in ISOTOPES(element):
                unnormed_composition[isotope]+= sub_fraction*atomic_fraction
    norm_factor = sum(unnormed_composition.values())
    return {isotope: round(fraction/norm_factor, 7) for isotope, fraction in unnormed_composition.items()}

def commonname_to_atnum_massnum(common_name):
    """convert str('Ag109') to (int(47), int(109))"""
    elem, mass_num = [c for c in common_name if c.isalpha()], [d for d in common_name if d.isdecimal()]
    at_num = ATOMIC_NUMBER["".join(elem)]
    return int(at_num), int("".join(mass_num))

def unpack_reactions(parent_product_mt):
    """
    Turn a reaction from the format of
    '(Cd106, Cd104)-Nb84-MT=((5), (5))'
    to
    ('Cd106-Nb84-MT=(5)', )
    """
    if parent_product_mt.startswith("("):
        parent_list, product, mts = parent_product_mt.split("-")
        mt_list = mts[len("-MT=("):-1]
        broken_down_reactions = []
        for parent, mt_values in zip(parent_list.strip("()").split(","), mt_list.split("),(")):
            broken_down_reactions.append("{}-{}-MT=({})".format(parent, product, mt_values.strip("()")))
        return tuple(broken_down_reactions)
    else:
        return (parent_product_mt,)

def strip_mt_brackets(parent_product_mt):
    """
    turn a reaction from the format of
    'Cd106-Nb84-MT=(5)'
    to
    ('Cd106-Nb84-MT=5', )
    """
    parent, product, mts = parent_product_mt.split("-")
    broken_down_reactions = []
    mt_list = mts[len("-MT="):-1]
    for mt in mt_list.strip("()").split(","):
        broken_down_reactions.append("{}-{}-MT={}".format(parent, product, mt))
    return tuple(broken_down_reactions)