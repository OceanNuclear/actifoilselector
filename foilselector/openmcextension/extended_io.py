# default system packages
import os, sys, json, warnings
from io import StringIO
from collections import defaultdict, OrderedDict
# special numerical computing packages
from tqdm import tqdm
import numpy as np
from numpy import array as ary
import uncertainties
from uncertainties.core import Variable
import pandas as pd
import openmc
from .constants import AMBIGUOUS_MT, FISSION_MTS, ATOMIC_SYMBOL, MT_to_nuc_num

__all__ = ["sparsely_load_xs_and_decay_dict",
            "endf_data_list_to_xs_dict",
            "deduce_daughter_from_mt",
            "MF10",
            "detabulate",
            "tabulate",
            "EncoderOpenMC",
            "DecoderOpenMC",
            "serialize_dict",
            "unserialize_dict",
            "save_csv_with_uncertainty",
            "unserialize_pd_DataFrame",
            ]

"""Functions to 
1. convert between uncertainties and python friendly objects (str representations)
2. load files as openmc class objects procedurally (load_endf_directories) # (wait no this should be a script?)
"""
def load_endf_directories(*folder_list):
    """
    Typical usage of this function:
    # in script.py
    if __name__=="__main__":
        load_endf_directories(*sys.argv[1:])
    Such that when users use script.py, they can call
    python script.py some/ folders/ with/ endf/ files/ or/specific_endf

    Within each folder, it will read all of the files that doesn't end in .json or .csv,
        as these are the two types of data outputted by foilselector.
    """
    if len(folder_list)==0:
        print("usage:")
        print("'python "+sys.argv[0]+" output/ folders/ containing/ endf/ tapes/ in/ ascending/ order/ of/ priority/'")
        print("Thus 'folders/ containing/ endf/ tapes/ in/ ascending/ order/ of/ priority/' will be read.")
        # print("where the outputs-saving directory 'output/' is only requried when read_apriori_and_gs_df is used.")
        print("Use wildcard (*) to replace directory as .../*/ if necessary")
        print("The endf files in question can be downloaded from online sources.")
        # by running the make file.")
        # from https://www.oecd-nea.org/dbforms/data/eva/evatapes/eaf_2010/ and https://www-nds.iaea.org/IRDFF/
        # Currently does not support reading h5 files yet, because openmc does not support reading ace/converting endf into hdf5/reading endf using NJOY yet
        sys.exit()
    #Please update your python to 3.6 or later to use f-strings
    print(f"Reading from {len(folder_list)} folders,")
    endf_file_list = []
    for folder in folder_list:
        # reads all endf data or decay data files.
        endf_file_list += [os.path.join(folder, file) for file in os.listdir(folder) if not (file.endswith('.json') or file.endswith('.csv'))]

    print(f"Found {len(endf_file_list)} regular files (ignoring files ending in '.json' or '.csv'). Assuming these are all endf data/decay data, reading them ...")
    #read in each file:
    endf_data = []
    for path in tqdm(endf_file_list):
        try:
            endf_data += openmc.data.get_evaluations(path)#works with IRDFF/IRDFFII.endf and EAF/*.endf
        except ValueError: # get_evaluations works doesn't work on decay/decay/files.
            endf_data += [openmc.data.Evaluation(path),] # works with decay/decay_2012/*.endf
    return endf_data

def _sort_and_trim_ordered_dict(ordered_dict, trim_length=3):
    """
    sort an ordered dict AND erase the first three characters (the atomic number) of each name
    """
    return OrderedDict([(key[trim_length:], val) for key,val in sorted(ordered_dict.items())])

def _extract_decay(dec_file):
    """
    extract the useful information out of an openmc.data.Decay entry
    """
    decay_constant = Variable(
                        np.nan_to_num(dec_file.decay_constant.n, nan=np.nan, posinf=1E23), # remove infinities
                        np.nan_to_num(float(dec_file.decay_constant.s)), # remove the nans which are often used as the uncertainties on the stable isotopes.
                        )
    modes = {}
    for mode in dec_file.modes:
        modes[mode.daughter] = mode.branching_ratio # we don't care what mechanism is used to transmute it. We just care about the respective branching ratios.
    return dict(decay_constant=decay_constant, branching_ratio=modes, spectra=dec_file.spectra)

def _rename_branching_ratio(decay_dict, isomeric_to_excited_state):
    """
    decay_dict : a dictionary of decay_dict
    isomeric_to_excited_state : a dictionary that translates from isomeric state to excited state.
    """
    for parent in decay_dict.keys():
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

def sparsely_load_xs_and_decay_dict(required_isotopes, folder_list):
    """
    Load in ONLY cross-sections of the required isotopes from a list of folders.
    This massively reduce the memory usage (therefore the prefix 'sparsely' in its name.)
    Parameters
    ----------
    required_isotopes: an iterable of isotopes,
                represented by a tuple of (atomic number, mass number)
    folder_list: an literable of directories where endf files can be found.
    Returns
    -------
    xs_dict: {isotope_name-product_name-MT=?? : openmc.data.Tabulated1D(microscopic cross-section in barns)}
    decay_dict: dictionary of {isotope_name : openmc.data.Decay.from_endf(isotope)}
    """
    # first, get the list of ALL files that can be read.
    max_mass_number = max(at_num_and_mass_num[1] for at_num_and_mass_num in required_isotopes) + 1
    endf_file_list = []
    for folder in folder_list:
        endf_file_list += [os.path.join(folder, file) for file in os.listdir(folder) if not (file.endswith('.json') or file.endswith('.csv'))]

    micro_xs, decay_dict, isomeric_to_excited_state = [], OrderedDict(), OrderedDict() # containers

    with warnings.catch_warnings(record=True) as w_list:
        # catching warnings occuring at the Decay.from_endf stage.
        for path in tqdm(endf_file_list, desc="Reading ENDF files from the provided directories"):
            try:
                # this is likely a incident neutron file:
                this_endf_data = openmc.data.get_evaluations(path)
            except ValueError:
                # this is either an invalid (non-ENDF data) file, or
                # a decay data file that only be loaded as follows:
                this_endf_data = [openmc.data.Evaluation(path)]

            for isotope_data in this_endf_data:
                # find the mass and atomic number to see if they need to be included.
                atnum_massnum = isotope_data.target['atomic_number'], isotope_data.target['mass_number']

                if isotope_data.info['sublibrary']=="Incident-neutron data":
                    # only collect relevant cross-sections, nothing else.
                    if atnum_massnum in required_isotopes:
                        micro_xs.append(isotope_data)

                # indiscriminantly collect every single isotope below the max. mass number.
                elif isotope_data.info['sublibrary']=="Radioactive decay data":
                    if atnum_massnum[1]<=max_mass_number: #mass number is within the allowed range of mass numbers.
                        dec_f = openmc.data.Decay.from_endf(isotope_data)
                        name = (str(isotope_data.target["atomic_number"]).zfill(3)
                                + ATOMIC_SYMBOL[isotope_data.target["atomic_number"]]
                                + str(isotope_data.target["mass_number"]) )
                        # just for convenience of figuring out the isomeric names, which is very important for later use.
                        isomeric_name = name
                        if isotope_data.target["isomeric_state"]>0: # if it is not at the lowest isomeric state: add the _e behind it too.
                            isomeric_name += "_m"+str(isotope_data.target["isomeric_state"])
                            name += "_e"+str(isotope_data.target["state"])
                        isomeric_to_excited_state[isomeric_name] = name[3:] # trim the excited state name

                        isomeric_to_excited_state[isomeric_name] = name[3:]
                        decay_dict[name] = _extract_decay(dec_f)

    # echo back the errors so that it doesn't fail silently.
    if w_list:
        print(w_list[0].filename+", line {}, {}'s:".format(w_list[0].lineno, w_list[0].category.__name__))
        for w in w_list:
            print("    "+str(w.message))

    # sort dicts to increase ease of use and debugging.
    decay_dict = _sort_and_trim_ordered_dict(decay_dict)
    isomeric_to_excited_state = _sort_and_trim_ordered_dict(isomeric_to_excited_state)
    decay_dict = _rename_branching_ratio(decay_dict, isomeric_to_excited_state)

    xs_dict = endf_data_list_to_xs_dict(micro_xs, isomeric_to_excited_state)
    xs_dict = _sort_and_trim_ordered_dict(xs_dict) # sort to increase ease of finding things the user needs.
    return xs_dict, decay_dict

def endf_data_list_to_xs_dict(inc_nuc_list, isomeric_to_excited_state):
    """Unpack openmc.data.IncidentNeutron objects into a dictionary of xs_dict.
    Returns: primary-product production cross-sections."""
    xs_dict = OrderedDict()
    for file in tqdm(inc_nuc_list, desc="Compiling the raw cross-section dictionary"):
        inc_f = openmc.data.IncidentNeutron.from_endf(file)
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

            append_name_list, xs_list = _extract_xs(inc_f.atomic_number, inc_f.mass_number, rx, tabulated=True)
            # add each product into the dictionary one by one.
            for name, xs in zip(append_name_list, xs_list):
                xs_dict[nuc_sort_name + '-' + name] = xs
    return xs_dict

def _extract_xs(parent_atomic_number, parent_atomic_mass, rx_file, tabulated=True):
    """
    For a given (mf, mt) file,
    Extract only the important bits of the informations:
    actaul cross-section, and the yield for each product.
    Outputs a list of these modified cross-sections (which are multiplied onto the thing if possible)
        along with all their names.
    The list can then be added into the final reaction dictionary one by one.
    """
    appending_name_list, xs_list = [], []
    xs = rx_file.xs['0K']
    if isinstance(xs, openmc.data.ResonancesWithBackground):
        xs = xs.background # When shrinking the group structure, this contains everything you need. The Resonance part of xs can be ignored (only matters for self-shielding.)
    if len(rx_file.products)==0: # if no products are already available, then we can only assume there is only one product.
        daughter_name = deduce_daughter_from_mt(parent_atomic_number, parent_atomic_mass, rx_file.mt)
        if daughter_name: # if the name is not None or False, i.e. a matching MT number is found.
            name = daughter_name+'-MT='+str(rx_file.mt) # deduce_daughter_from_mt will return the ground state value
            appending_name_list.append(name)
            xs_list.append(detabulate(xs) if (not tabulated) else xs)
    else:
        for prod in rx_file.products:
            appending_name_list.append(prod.particle+'-MT='+str(rx_file.mt))
            partial_xs = openmc.data.Tabulated1D(xs.x, prod.yield_(xs.x) * xs.y,
                                                breakpoints=xs.breakpoints, interpolation=xs.interpolation)
            xs_list.append(detabulate(partial_xs) if (not tabulated) else partial_xs)
    return appending_name_list, xs_list

def deduce_daughter_from_mt(parent_atomic_number, parent_atomic_mass, mt):
    """
    Given the atomic number and mass number, get the daughter in the format of 'Ag109'.
    """
    if mt in MT_to_nuc_num.keys():
        element_symbol = openmc.data.ATOMIC_SYMBOL[parent_atomic_number + MT_to_nuc_num[mt][0]]
        product_mass = str(parent_atomic_mass + MT_to_nuc_num[mt][1])
        if len(MT_to_nuc_num[mt])>2 and MT_to_nuc_num[mt][2]>0: # if it indicates an excited state
            excited_state = '_e'+str(MT_to_nuc_num[mt][2])
            return element_symbol+product_mass+excited_state
        else:
            return element_symbol+product_mass
    else:
        return None

class MF10(object):
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

    __slots__ = ["number_of_reactions", "za", "target_mass", "target_isomeric_state", "reaction_mass_difference", "reaction_q_value", "reactions"]
    # __slots__ created for memory management purpose in case there are many suriving instances of MF10 all present at once.
    def __init__(self, mf10_mt5_section):
        if mf10_mt5_section is not None:
            file_stream = StringIO(mf10_mt5_section)
            za, target_mass, target_iso, _, ns, _ = openmc.data.get_head_record( file_stream ) # read the first line, i.e. the head record for MF=10, MT=5.
            self.number_of_reactions = ns
            self.za = za
            self.target_mass = target_mass
            self.target_isomeric_state = target_iso
            self.reaction_mass_difference, self.reaction_q_value = {}, {}
            self.reactions = {}
            for reaction_number in range(ns):
                (mass_diff, q_value, izap, isomeric_state), tab = openmc.data.get_tab1_record(file_stream)
                self.reaction_mass_difference[(izap, isomeric_state)] = mass_diff
                self.reaction_q_value[(izap, isomeric_state)] = q_value
                self.reactions[(izap, isomeric_state)] = tab
        else:
            self.reactions = {}

    def keys(self):
        return self.reactions.keys()

    def items(self):
        return self.reactions.items()

    def values(self):
        return self.reactions.values()

def detabulate(tabulated_object):
    """
    Convert a openmc.data.tabulated_object into something json serialize-able.
    """
    scheme = np.zeros(len(tabulated_object.x)-1, dtype=int)
    for point, scheme_number in list(zip(tabulated_object.breakpoints, tabulated_object.interpolation))[::-1]:
        scheme[:point-1] = scheme_number # note the -1: it describes the interp-scheme of all cells *before* it.

    return dict(x=tabulated_object.x.tolist(), y=tabulated_object.y.tolist(), interpolation=scheme.tolist())

def tabulate(detabulated_dict):
    """
    parameters
    ----------
    detabulated_dict : should have 3 columns: "x" (list of len=n+1), "y" (list of len=n+1), "interpolation" (list of len=n).
    """
    interpolation_long = list(detabulated_dict["interpolation"]) # make a copy
    # pad the interpolation_long to the lenght of n+1
    interpolation_long.append(0) # outside of the specified data range, the interpolation scheme = N/A; use zero as placeholder for N/A.
    breakpoints = sorted(np.argwhere(np.diff(interpolation_long)).flatten()+2)
    interpolation = ary(interpolation_long)[ary(breakpoints)-2].tolist()
    return openmc.data.Tabulated1D(detabulated_dict["x"], detabulated_dict["y"], breakpoints, interpolation)

class EncoderOpenMC(json.JSONEncoder):
    def default(self, o):
        """
        The object will be sent to the .default() method if it can't be handled
        by the native ,encode() method.
        The original JSONEncoder.default method only raises a TypeError;
        so in this class we'll make sure it handles these specific cases (numpy, openmc and uncertainties)
        before defaulting to the JSONEncoder.default's error raising.
        """
        # numpy types
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.float64):
            return float(o)
        # uncertainties types
        elif isinstance(o, uncertainties.core.AffineScalarFunc):
            try:
                return str(o)
            except ZeroDivisionError:
                return "0.0+/-0"
        # openmc
        elif isinstance(o, openmc.data.Tabulated1D):
            return detabulate(o)
        # return to default error
        else:
            return super().default(o)

class DecoderOpenMC(json.JSONDecoder):
    def decode(self, o):
        """
        Catch the uncertainties
        Edit: This class doesn't work because the cpython/decoder.py is not written in a open-for-expansion principle.
        I suspect this is because turning variables (which aren't strings) into strings is an
            imporper way to use jsons; but I don't have any other solutions for EncoderOpenMC.

        In any case, DecoderOpenMC will be replaced by unserialize_dict below.
        """
        if '+/-' in o:
            if ')' in o:
                multiplier = float('1'+o.split(')')[1])
                o = o.split(')')[0].strip('(')
            else:
                multiplier = 1.0
            return Variable(*[float(i)*multiplier for i in o.split('+/-')])
        else:
            return super().decode(o)

def serialize_dict(mixed_object):
    """
    Deprecated as its functionality is entirely covered by EncoderOpenMC.
    """
    if isinstance(mixed_object, dict):
        for key, val in mixed_object.items():
            mixed_object[key] = serialize_dict(val)
    elif isinstance(mixed_object, list):
        for ind, item in enumerate(mixed_object):
            mixed_object[ind] = serialize_dict(item)
    elif isinstance(mixed_object, uncertainties.core.AffineScalarFunc):
        mixed_object = str(mixed_object) # rewrite into str format
    elif isinstance(mixed_object, openmc.data.Tabulated1D):
        mixed_object = detabulate(mixed_object)
    else: # can't break it down, and probably is a scalar.
        pass
    return mixed_object

def unserialize_dict(mixed_object):
    """
    Turn the string representation of the uncertainties back into uncertainties.core.Variable 's.
    """
    if isinstance(mixed_object, dict):
        if tuple(mixed_object.keys())==("x", "y", "interpolation"): # if the mixed_object is a openmc.data.Tabulated1D object in disguise:
            mixed_object = tabulate(mixed_object)
        else:
            for key, val in mixed_object.items():
                mixed_object[key] = unserialize_dict(val) # recursively un-serialize
    elif isinstance(mixed_object, list):
        for ind, item in enumerate(mixed_object):
            mixed_object[ind] = unserialize_dict(item)
    elif isinstance(mixed_object, str):
        if '+/-' in mixed_object: # is an uncertainties.core.Variable object
            if ')' in mixed_object:
                multiplier = float('1'+mixed_object.split(')')[1])
                mixed_object_stripped = mixed_object.split(')')[0].strip('(')
            else:
                multiplier = 1.0
                mixed_object_stripped = mixed_object
            mixed_object = Variable(*[float(i)*multiplier for i in mixed_object_stripped.split('+/-')])
        else:
            pass # just a normal string
    else: # unknown type
        pass
    return mixed_object

def save_csv_with_uncertainty(df, filename, *args, **kwargs):
    """Handles saving dataframes with uncertain values as csv. This is the counterpart of unserialize_pd_DataFrame
    df: pandas.DataFrame object to be saved, possibly with uncertainties.core.AffineFunc values in one of the columns.
    """
    try:
        df.to_csv(filename, *args, **kwargs)
    except ZeroDivisionError:
        print("Minor issue when trying to write values which are too small. Plese allow several_extra_minutes...")
        cols = df.columns
        is_uncertain = ary([isinstance(df.iloc[0][col], uncertainties.core.AffineScalarFunc) for col in cols])
        uncertain_columns = cols[is_uncertain]
        for col in tqdm(df.columns, desc="Setting almost infinitesimally small values to zero"):
            if col in uncertain_columns:
                # when trying to express uncertainties.core.Variable using the  __str__ method, it will try to factorize it.
                # But if the GCD between the norminal value and uncertainty is rounded down to 1E-323 or smaller, it will lead to ZeroDivisionError.
                floating_point_problem = df[col]<12.5E-324 # therefore we set all small values
                # this method is harsher than it needs to because it forces items
                # with nominal value < 12.5E-324 but error > 12.5E-324 to be 0 as well,
                # even though they are perfectly expressible as strings without errors.
                df[col][floating_point_problem] = 0
        df.to_csv(filename, *args, **kwargs)

def unserialize_pd_DataFrame(df):
    """Convert the strings that represent uncertain values in a csv read in by pd.read_csv
    into unc.core.Variable(those strings)"""
    new_values = []
    for col in df.values.T:
        new_values.append(unserialize_dict(list(col)))
    return pd.DataFrame(ary(new_values).T, index=df.index, columns=df.columns)