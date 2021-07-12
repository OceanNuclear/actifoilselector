# default system packages
import os, sys, json
# special numerical computing packages
import numpy as np
from numpy import array as ary
import uncertainties
from uncertainties.core import Variable
import pandas as pd
import openmc

"""Functions to 
1. convert between uncertainties and python friendly objects (str representations)
2. load files as openmc class objects procedurally (load_endf_directories) # (wait no this should be a script?)
"""
from tqdm import tqdm
def load_endf_directories(folder_list):
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
        print("'python "+sys.argv[0]+" folders/ containing/ endf/ tapes/ in/ ascending/ order/ of/ priority/ output/'")
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
        elif isinstance(o, np.float):
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

def unserialize_pd_DataFrame(df):
    new_values = []
    for col in df.values.T:
        new_values.append(unserialize_dict(list(col)))
    return pd.DataFrame(ary(new_values).T, index=df.index, columns=df.columns)