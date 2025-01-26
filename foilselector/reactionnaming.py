"""
Functions to help converting between different types of naming convention for
isotopes and reactions.
"""

from collections import defaultdict

from openmc.data import ATOMIC_NUMBER
from openmc.data import isotopes as ISOTOPES  # noqa: N812


def specify_isotopic_composition(
    user_composition_input: dict[str, float],
) -> dict[str, float]:
    """Expand a user input (typed in a user-friendly syntax) dictionary of atomic
    composition into a purely "isotope:fraction" format.
    (user may input in either 'isotope:fraction' or 'chemical-element:fraction' format.)
    This requires breaking down elements into its natural isotopic abundances,
    and then normalizing the output composition so that it adds up to unit.

    Parameters
    ----------
    user_composition_input:
        A dictionary of atomic compositions, where the keys are isotope names OR
        element names.

    Returns
    -------
    :
        A dictionary of atomic compositions, where the keys are isotope names OR
        element names.
    """
    unnormed_composition = defaultdict(float)
    for element_or_isotope, atomic_fraction in user_composition_input.items():
        if any(c.isnumeric() for c in element_or_isotope):
            isotope = element_or_isotope  # is an isotope
            unnormed_composition[isotope] += atomic_fraction
        else:
            element = element_or_isotope  # is an element
            for isotope, sub_fraction in ISOTOPES(element):
                unnormed_composition[isotope] += sub_fraction * atomic_fraction
    norm_factor = sum(unnormed_composition.values())
    return {
        isotope: round(fraction / norm_factor, 7)
        for isotope, fraction in unnormed_composition.items()
    }


def commonname_to_atnum_massnum(common_name: str) -> tuple[int, int]:
    """
    Convert isotope name into (atomic number, mass number) representation.

    Parameters
    ----------
    common_name:
        Name of the isotope, e.g. 'Ag109'

    Returns
    -------
    :
        tuple of atomic number and mass number, e.g. (int(47), int(109))
    """
    elem, mass_num = (
        [c for c in common_name if c.isalpha()],
        [d for d in common_name if d.isdecimal()],
    )
    at_num = ATOMIC_NUMBER["".join(elem)]
    return int(at_num), int("".join(mass_num))
