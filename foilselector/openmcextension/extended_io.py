"""
These includes functions to
1. Convert between uncertainties and python friendly objects (tuple containing the
    nominal_value and std_dev)
2. Load files as openmc class objects procedurally (sparsely_load_xs_and_decay_dict),
3. supported by functions that rename the reactions and isotopes to the appropriate
    formats.
"""

import warnings
from collections import OrderedDict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import numpy as np
import openmc
from openmc.data import ATOMIC_SYMBOL
from tqdm import tqdm
from uncertainties import nominal_value as nom
from uncertainties.core import AffineScalarFunc, Variable

from foilselector.openmcextension.constants import (
    AMBIGUOUS_MT,
    FISSION_MTS,
    MT_to_nuc_num,
)
from foilselector.openmcextension.library_reader import (
    ContinuousRadiationDistribution,
    DiscreteRadiation,
)
from foilselector.openmcextension.table import Tab1DExtended, detabulate

__all__ = [
    "MF10",
    "deduce_daughter_from_mt",
    "deserialize_radiation_dict",
    "deserialize_radiation_list",
    "endf_data_list_to_xs_dict",
    "reactions_matching",
    "serialize_radiation_dict",
    "serialize_radiation_list",
    "sparsely_load_xs_and_decay_dict",
]


def _sort_and_trim_ordered_dict(
    ordered_dict: dict,
    trim_length: int = 3,
) -> OrderedDict:
    """Sort an ordered dict AND erase the first three characters (the atomic number) of
    each name.

    Parameters
    ----------
    ordered_dict:
        The dictionary to be sorted.
    trim_length:
        how many characters to trim off of the beginning of each key.

    Returns
    -------
    :
        A shallow copy of the ordered_dict
    """
    return OrderedDict([
        (key[trim_length:], val) for key, val in sorted(ordered_dict.items())
    ])


def _extract_decay(dec_file: openmc.data.Decay) -> dict:
    """Extract the useful information out of an openmc.data.Decay entry.

    Parameters
    ----------
    dec_file:
        The openmc Decay data.

    Returns
    -------
    :
        The dictionary containing only the important information, i.e. decay constant (
        scalar, which is an AffineScalarFunc), branching ratios (dict[str, float]), and
        the spectra (dict).
    """
    decay_constant = Variable(
        np.nan_to_num(
            dec_file.decay_constant.n,
            nan=np.nan,
            posinf=1e23,
        ),  # remove infinities
        np.nan_to_num(
            float(dec_file.decay_constant.s),
        ),  # remove the nans which are often used as the uncertainties on the stable isotopes.
    )
    modes = {}
    for mode in dec_file.modes:
        modes[mode.daughter] = (
            mode.branching_ratio
        )  # we don't care what mechanism is used to transmute it. We just care about the respective branching ratios.
    return dict(
        decay_constant=decay_constant,
        branching_ratio=modes,
        spectra=dec_file.spectra,
    )


def _rename_branching_ratio(
    decay_dict: dict, isomeric_to_excited_state: dict[str, str]
) -> dict:
    """Modify the 'branching_ratio' entry of the decay_dict to use the correct names,
    showing the excited state rather than the metastable/isomeric state.

    Parameters
    ----------
    decay_dict:
        a dictionary of decay_dict
    isomeric_to_excited_state:
        a dictionary that translates from isomeric state to excited state.

    Returns
    -------
    decay_dict:
        The same decay dict, but with the 'branching_ratio' entry modified in place.
    """
    for parent in decay_dict:
        products = decay_dict[parent]["branching_ratio"]
        renamed = {}
        for prod, ratio in products.items():
            e_name = isomeric_to_excited_state.get(prod, prod.split("_")[0])
            if e_name in renamed:
                renamed[e_name] += ratio
            else:
                renamed[e_name] = ratio
        decay_dict[parent]["branching_ratio"] = renamed
    return decay_dict


def sparsely_load_xs_and_decay_dict(
    required_isotopes: Iterable[str], folder_list: Iterable[Path]
) -> tuple[dict[str, openmc.data.Tabulated1D], dict]:
    """
    Load in ONLY cross-sections of the required isotopes from a list of folders.
    This massively reduce the memory usage (therefore the prefix 'sparsely' in its name.)

    Parameters
    ----------
    required_isotopes:
        An iterable of isotopes, represented by a tuple of (atomic number, mass number)
    folder_list:
        An literable of directories where the endf files can be found.

    Returns
    -------
    xs_dict:
        {isotope_name-product_name-MT=?? : openmc.data.Tabulated1D(microscopic
            cross-section in barns)}
    decay_dict:
        dictionary of {isotope_name : openmc.data.Decay.from_endf(isotope)}
    """
    # first, get the list of ALL files that can be read.
    max_mass_number = (
        max(at_num_and_mass_num[1] for at_num_and_mass_num in required_isotopes) + 1
    )
    endf_file_list = []
    for folder in folder_list:
        endf_file_list.extend(list(Path(folder).iterdir()))

    micro_xs = []
    decay_dict = OrderedDict()
    isomeric_to_excited_state = OrderedDict()

    with warnings.catch_warnings(record=True) as w_list:
        # catching warnings occuring at the Decay.from_endf stage.
        for path in tqdm(
            endf_file_list,
            desc="Reading ENDF files from the provided directories",
        ):
            try:
                # this is likely a incident neutron file:
                this_endf_data = openmc.data.get_evaluations(path)
            except ValueError:
                # this is either an invalid (non-ENDF data) file, or
                # a decay data file that only be loaded as follows:
                this_endf_data = [openmc.data.Evaluation(path)]

            for isotope_data in this_endf_data:
                # find the mass and atomic number to see if they need to be included.
                atnum = isotope_data.target["atomic_number"]
                massnum = isotope_data.target["mass_number"]

                if isotope_data.info["sublibrary"] == "Incident-neutron data":
                    # only collect relevant cross-sections, nothing else.
                    if (atnum, massnum) in required_isotopes:
                        micro_xs.append(isotope_data)

                # indiscriminantly collect every single isotope below the max. mass number.
                elif isotope_data.info["sublibrary"] == "Radioactive decay data":
                    if mass_number <= max_mass_number:
                        # extract only if we can reach this mass number by decaying.
                        dec_f = openmc.data.Decay.from_endf(isotope_data)
                        # just for convenience of figuring out the isomeric names, which
                        # is very important for later use.
                        name = _name_from_at_mass(atnum, massnum)
                        isomeric_name = _add_isomeric_state(
                            name, isotope_data.target["isomeric_state"],
                        )
                        excited_name = _add_excited_state(
                            name, isotope_data.target["state"]
                        )
                        isomeric_to_excited_state[isomeric_name] = excited_name[3:]
                        decay_dict[excited_name] = _extract_decay(dec_f)

    # echo back the errors so that it doesn't fail silently.
    if w_list:
        print(
            w_list[0].filename
            + f", line {w_list[0].lineno}, {w_list[0].category.__name__}'s:",
        )
        for w in w_list:
            print("    " + str(w.message))

    # sort dicts to increase ease of use and debugging.
    decay_dict = _sort_and_trim_ordered_dict(decay_dict)
    isomeric_to_excited_state = _sort_and_trim_ordered_dict(isomeric_to_excited_state)
    decay_dict = _rename_branching_ratio(decay_dict, isomeric_to_excited_state)

    # sort to increase ease of finding things the user needs.
    xs_dict = endf_data_list_to_xs_dict(micro_xs, isomeric_to_excited_state)
    xs_dict = _sort_and_trim_ordered_dict(xs_dict)
    return xs_dict, decay_dict

def _name_from_at_mass(atomic_number: int, mass_number: int) -> tuple[str, str]:
    """Create name of the isotope from atomic number and mass number alone.
    Parameters
    ----------
    atomic_number:
        atomic number of the nucleus
    mass_number:
        mass number of the nucleus

    Returns
    -------
    :
        A string representation of the ground-state nuclide, e.g. "002He4", where
        002 = atomic number,
        He = atomic symbol,
        4 = mass number
    """
    return str(atomic_number).zfill(3) + ATOMIC_SYMBOL[atomic_number] + str(mass_number)

def _add_isomeric_state(name: str, isomeric_state: int) -> str:
    """Append the isomeric state onto the end of the name.

    Parameters
    ----------
    name:
        str that we want to attach into
    isomeric_state:
        THe nuclide is in the n-th isomeric (i.e. metastable) state, where if n=0, it is
        in the ground state.

    Returns
    -------
    :
        The original name appended with the isomeric state
    """
    if isomeric_state:
        return name + f"_m{isomeric_state}"
    return name

def _add_excited_state(name: str, excited_state: int) -> str:
    """Append the isomeric state onto the end of the name.

    Parameters
    ----------
    name:
        str that we want to attach into
    excited_state:
        THe nuclide is in the n-th isomeric (i.e. metastable) state, where if n=0, it is
        in the ground state.

    Returns
    -------
    :
        The original name appended with the excited state
    """
    if excited_state:
        return name + f"_m{excited_state}"
    return name

def endf_data_list_to_xs_dict(
    inc_nuc_list: Iterable[openmc.data.endf.Evaluation],
    isomeric_to_excited_state: dict[str, str]
) -> dict[str, openmc.data.Tabulated1D]:
    """
    Unpack openmc.data.IncidentNeutron objects into a dictionary of xs_dict.

    Parameters
    ----------

    Returns
    -------
    xs_dict:
        dictionary of primary-product production cross-sections, where
        key = "068Fe-"
        value= Tabulated1D (x=energy in eV, y=xs in barns)
    """
    xs_dict = OrderedDict()
    for file in tqdm(
        inc_nuc_list,
        desc="Compiling the cross-sections from each nuclear data files",
    ):
        inc_f = openmc.data.IncidentNeutron.from_endf(file)
        nuc_sort_name = str(inc_f.atomic_number).zfill(3) + inc_f.name

        # get the higher-energy range values of xs as well if available.
        mf10_mt5 = MF10(
            file.section.get((10, 5), None),
        )  # default value = None if (10, 5 doesn't exist.)
        for (izap, isomeric_state), xs in mf10_mt5.items():
            atomic_number, mass_number = divmod(izap, 1000)
            if (
                atomic_number > 0 and mass_number > 0
            ):  # ignore the weird products that means nothing meaningful
                gnd_name = ATOMIC_SYMBOL[atomic_number] + str(mass_number)
                isomeric_name = _add_isomeric_state(gnd_name, isomeric_state)
                e_name = isomeric_to_excited_state.get(isomeric_name,
                    isomeric_to_excited_state.get(gnd_name, gnd_name)
                )
                # default to using the ground state's name if N/A.
                long_name = nuc_sort_name + "-" + e_name + "-MT=5"
                xs_dict[long_name] = xs

        # get the normal reactions, found in mf=3
        for mt, rx in inc_f.reactions.items():
            if any([(mt in AMBIGUOUS_MT), (mt in FISSION_MTS), (301 <= mt <= 459)]):
                continue  # skip the cases of AMBIGUOUS_MT, fission mt, and heating information. They don't give us useful information about radionuclides produced.

            append_name_list, xs_list = _extract_xs(
                inc_f.atomic_number,
                inc_f.mass_number,
                rx,
                tabulated=True,
            )
            # add each product into the dictionary one by one.
            for name, xs in zip(append_name_list, xs_list, strict=False):
                xs_dict[nuc_sort_name + "-" + name] = xs
    return xs_dict


def reactions_matching(xs_dict: dict, isotope: str) -> dict:
    """
    Get the subset of the dictionary containing only reactions that uses the specified
    isotopes as the reactant.
    """
    return {k: v for k, v in xs_dict.items() if k.split("-")[0] == isotope}


def _extract_xs(
    parent_atomic_number, parent_atomic_mass, rx_file, tabulated=True
) -> tuple[list[str], list[openmc.data.Tabulated1D]]:
    """
    For a given (mf, mt) file,
    Extract only the important bits of the informations:
    actaul cross-section, and the yield for each product.
    Outputs a list of these modified cross-sections (which are multiplied onto the thing if possible)
        along with all their names.
    The list can then be added into the final reaction dictionary one by one.
    """
    appending_name_list, xs_list = [], []
    xs = rx_file.xs["0K"]
    if isinstance(xs, openmc.data.ResonancesWithBackground):
        xs = xs.background
        # When shrinking the group structure, xs.background contains everything you need.
        # The Resonance part of xs can be ignored (only matters for self-shielding.)
    daughter_name = deduce_daughter_from_mt(
        parent_atomic_number,
        parent_atomic_mass,
        rx_file.mt,
    )
    # if a suitable MT number is found, the daughter's ground state's name will be given.
    if daughter_name:
        name = daughter_name + "-MT=" + str(rx_file.mt)
        appending_name_list.append(name)
        xs_list.append(detabulate(xs) if (not tabulated) else xs)
    return appending_name_list, xs_list


def deduce_daughter_from_mt(parent_atomic_number, parent_atomic_mass, mt):
    """
    Given the atomic number and mass number, get the daughter in the format of 'Ag109'.
    """
    if mt in MT_to_nuc_num.keys():
        element_symbol = openmc.data.ATOMIC_SYMBOL[
            parent_atomic_number + MT_to_nuc_num[mt][0]
        ]
        product_mass = str(parent_atomic_mass + MT_to_nuc_num[mt][1])
        if (
            len(MT_to_nuc_num[mt]) > 2 and MT_to_nuc_num[mt][2] > 0
        ):  # if it indicates an excited state
            excited_state = "_e" + str(MT_to_nuc_num[mt][2])
            return element_symbol + product_mass + excited_state
        return element_symbol + product_mass
    return None


class MF10:
    def __getitem__(self, key):
        return self.reactions.__getitem__(key)

    def __len__(self):
        return self.reactions.__len__()

    def __iter__(self):
        return self.reactions.__iter__()

    def __reversed__(self):
        return self.reactions.__reversed__()

    def __contains__(self, key):
        return self.reactions.__contains__(key)

    __slots__ = [
        "number_of_reactions",
        "reaction_mass_difference",
        "reaction_q_value",
        "reactions",
        "target_isomeric_state",
        "target_mass",
        "za",
    ]

    # __slots__ created for memory management purpose in case there are many suriving instances of MF10 all present at once.
    def __init__(self, mf10_mt5_section):
        if mf10_mt5_section is not None:
            file_stream = StringIO(mf10_mt5_section)
            za, target_mass, target_iso, _, ns, _ = openmc.data.get_head_record(
                file_stream,
            )  # read the first line, i.e. the head record for MF=10, MT=5.
            self.number_of_reactions = ns
            self.za = za
            self.target_mass = target_mass
            self.target_isomeric_state = target_iso
            self.reaction_mass_difference, self.reaction_q_value = {}, {}
            self.reactions = {}
            for reaction_number in range(ns):
                (mass_diff, q_value, izap, isomeric_state), tab = (
                    openmc.data.get_tab1_record(file_stream)
                )
                self.reaction_mass_difference[izap, isomeric_state] = mass_diff
                self.reaction_q_value[izap, isomeric_state] = q_value
                self.reactions[izap, isomeric_state] = tab
        else:
            self.reactions = {}

    def keys(self):
        return self.reactions.keys()

    def items(self):
        return self.reactions.items()

    def values(self):
        return self.reactions.values()


def serialize_radiation_dict(obj):
    """Turn radiation dict into something that can be saved as a JSON file."""
    if isinstance(obj, AffineScalarFunc):
        # AffineScalarFunc -> dict{'n':float, 's':float}
        return {"n": obj.n, "s": obj.s}
    if isinstance(obj, np.ndarray):
        if obj.dtype.name == "object":
            return [nom(var) for var in obj]
        # np.ndarray -> list[float] | list[int]
        return obj.tolist()
    if isinstance(
        obj,
        DiscreteRadiation | ContinuousRadiationDistribution | Tab1DExtended,
    ):
        # namedtuple | Tab1DExtended -> dict
        return {k: serialize_radiation_dict(v) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        # a dict of reactions and their xs: turn into list instead
        if len(obj) == 0:
            return {}
        k0 = list(obj.keys())[0]
        if isinstance(k0, DiscreteRadiation | ContinuousRadiationDistribution):
            # dict -> list[{             'DiscreteRadiation' : dict, 'xs':list[float]}, ...]
            # dict -> list[{'ContinuousRadiationDistribution': dict, 'xs':list[float]}, ...]
            return [
                {
                    type(k).__name__: serialize_radiation_dict(k),
                    "xs": serialize_radiation_dict(v),
                }
                for k, v in obj.items()
            ]
        # dict[str:'foil_name', dict] -> dict[str: 'foil_name', list]
        return {k: serialize_radiation_dict(v) for k, v in obj.items()}
    # str -> str
    return obj


def deserialize_radiation_dict(obj):
    """Turn JSON file back into radiation dict."""
    if isinstance(obj, dict):
        keys = obj.keys()
        if tuple(keys) == ("n", "s"):
            return Variable(obj["n"], obj["s"])
        if "DiscreteRadiation" in keys:
            return {
                DiscreteRadiation(**{
                    k: deserialize_radiation_dict(v)
                    for k, v in obj["DiscreteRadiation"].items()
                }): np.array(obj["xs"]),
            }
        if "ContinuousRadiationDistribution" in keys:
            return {
                ContinuousRadiationDistribution(**{
                    k: deserialize_radiation_dict(v)
                    for k, v in obj["ContinuousRadiationDistribution"].items()
                }): np.array(obj["xs"]),
            }
        if sorted(keys) == sorted(Tab1DExtended._fields):
            return Tab1DExtended(
                x=obj["x"],
                y=obj["y"],
                interpolation=obj["interpolation"],
            )
        return {k: deserialize_radiation_dict(v) for k, v in obj.items()}

    if isinstance(obj, list):
        if isinstance(obj[0], float | int):
            return np.array(obj)
        # should be a list of len==2 dicts left at this stage.
        d = {}
        for item in obj:
            prev_len = len(d)
            d.update(deserialize_radiation_dict(item))
            if (prev_len + 1) != len(d):
                raise ValueError(
                    "Programmer error! This list is supposed to represent a dict; "
                    "but this list contains repeated items!",
                )
        return d
    return obj


def serialize_radiation_list(obj):
    if isinstance(obj, list):
        return [serialize_radiation_list(o) for o in obj]
    if isinstance(obj, DiscreteRadiation | ContinuousRadiationDistribution):
        return {k: serialize_radiation_dict(v) for k, v in obj._asdict().items()}


def deserialize_radiation_list(obj):
    if isinstance(obj, list):
        return [deserialize_radiation_list(o) for o in obj]
    if isinstance(obj, dict):
        if tuple(obj.keys()) == DiscreteRadiation._fields:
            return DiscreteRadiation(**{
                k: deserialize_radiation_dict(v) for k, v in obj.items()
            })
        if tuple(obj.keys()) == ContinuousRadiationDistribution._fields:
            return ContinuousRadiationDistribution(**{
                k: deserialize_radiation_dict(v) for k, v in obj.items()
            })
        return {k: deserialize_radiation_list(v) for k, v in obj.items()}
    return obj
