"""functions used for reading and saving data.
All of the read_* and save_* functions all saves at the current directory by default,
    unless their save filepath is changed"""
import json as json
import re
import pandas as pd
import numpy as np
from os.path import exists, join
from foilselector.openmcextension.extended_io import *
from foilselector.selfshielding import MaxSigma

# sigma_df
def save_microscopic_cross_section_csv(sigma_df, csv_path=".sigma_df.csv"):
    return sigma_df.to_csv(csv_path)

def read_microscopic_cross_section_csv(csv_path=".sigma_df.csv"):
    return pd.read_csv(csv_path, index_col=[0])

# atomic_composition
def save_atomic_composition_json(processed_composition, json_path=".atomic_composition.json"):
    from os import path
    print(path.abspath(json_path))
    with open(json_path, "w") as j:
        return json.dump(processed_composition, j, indent=1)

def read_atomic_composition_json(json_path=".atomic_composition.json"):
    with open(json_path) as j:
        return json.load(j)

# masses of foils
def read_csv_with_header(csv_name):
    return pd.read_csv(csv_name, index_col=[0])

# decay_radiation (un-condensed version of decay_dict that include the gamma spectrum)
def save_decay_radiation(decay_radiation, json_path=".decay_radiation.json", verbose=True):
    """Save the dictionary of isotope:{decay constant, decay_spectra, etc} info into a .decay_radiation.json file.
    verbose: because this function takes a bit of time, verbose gives feedback to the user so they know what's going on."""
    with open(json_path, "w") as j:
        if verbose: print("Writing to decay_radiation.json... ", end="")
        exit_code = json.dump(decay_radiation, j, cls=EncoderOpenMC)
        if verbose: print("Done!")
        return exit_code

def read_decay_radiation(json_path=".decay_radiation.json"):
    """Read decay_radiation"""
    with open(json_path) as j:
        return unserialize_dict(json.load(j))

# decay_info (condensed version of decay_dict, only includes the number of countable photons as a scalar.)
def save_decay_info(decay_info, json_path=".decay_info.json"):
    with open(json_path, "w") as j:
        return json.dump(decay_info, j, cls=EncoderOpenMC) # we're dumping the condensed decay dict, i.e. called the decay_info, here.

def read_decay_info(json_path=".decay_info.json"):
    with open(json_path) as j:
        return unserialize_dict(json.load(j))

# selfshielding_dict
def save_self_shielding(selfshielding_dict, json_path=".self-shielding.json"):
    with open(json_path, "w") as j:
        return json.dump(selfshielding_dict, j)

def read_self_shielding(json_path=".self-shielding.json"):
    with open(json_path) as j:
        return MaxSigma(json.load(j))

def save_counts_csv(population_df, csv_path=".counts.csv", comments: str=None):
    if comments:
        if isinstance(comments, str):
            comments = comments.split("\n") # break strings at newlines.
        elif isinstance(comments, (list, tuple)):
            assert all([isinstance(c, str) for c in comments]), "Comment must a list or tuple of strings."
        else:
            raise TypeError("Comment provided must be a string, or list/tuple of strings.")

    exit_code = save_csv_with_uncertainty(population_df, csv_path, index_label="rname")

    if comments:
        with open(csv_path, "r") as f: # open file and read it.
            new_text = "#"+"\n#".join(comments)+"\n"+f.read()
        with open(csv_path, "w") as f:# overwrite existing csv file
            exit_code = f.write(new_text)

    return exit_code

def read_counts_csv(csv_path=".counts.csv", return_parameters_stored_in_comments=False):
    """
    return_parameters_stored_in_comments: return the comments to terminal if comments are available.
    """
    if return_parameters_stored_in_comments:
        comments = []
        with open(csv_path) as f:
            this_line = f.readline()
            while this_line.startswith("#"):
                # print(this_line.rstrip("\n"), end="\n") # only print one newline character
                comments.append(this_line.lstrip("# ").rstrip("\n")) # instead of printing it, we want to return it.
                this_line = f.readline()
            f.seek(0) # force of habit: I want to put the cursor back to the front.
            # close the file automatically when we exit the 'with' clause.
        parameters = {}
        for line in comments:
            parameters[line.split("=")[0].strip()] = line.split("=")[-1].strip()
        return unserialize_pd_DataFrame(pd.read_csv(csv_path, index_col=[0], comment="#")), parameters
    else:
        return unserialize_pd_DataFrame(pd.read_csv(csv_path, index_col=[0], comment="#"))

def read_vector(file_path):
    """Read a file that contains newline/space/comma delimited scalar values,
    and then return it as a list of numbers."""
    with open(file_path) as f:
        scalar_list = re.split("\n|,| ", f.read()) # with possible empty entries if there are blank lines.
    return [float(val) for val in scalar_list if len(val)>0]
    
def save_vector(vector, file_path, format_func=str):
    """Save a vector into a newline-delimited file."""
    with open(file_path, "w") as f:
        for val in vector:
            f.write(format_func(val)+"\n")

def read_gs(file_path):
    return pd.read_csv(file_path).values

def read_flux(file_path):
    return np.squeeze(pd.read_csv(file_path).values)

#### The rest of these functions below aren't going to be needed. ####
def get_apriori_from_folder(folder, irradiation_duration=None):
    """
    Given the file location and irradiation duration,
    return the apriori_flux and the apriori_fluence
    """
    assert exists(join(folder, 'integrated_apriori.csv')), "Output directory must already have integrated_apriori.csv for calculating the radionuclide populations."
    print("Reading integrated_apriori.csv as the fluence, i.e. total number of neutrons/cm^2/eV/s, averaged over the IRRADIATION_DURATION = {} s\n".format(irradiation_duration))
    apriori_flux = pd.read_csv(join(folder, 'integrated_apriori.csv'))['value'].values # integrated_apriori.csv is a csv with header = value and number of rows = len(gs); no index.
    if irradiation_duration is None:
        return apriori_flux
    else:
        apriori_fluence = apriori_flux * irradiation_duration
        return apriori_flux, apriori_fluence

# def get_gs_and_flux(file_path, directory="."):
#     return pd.read_csv(file_path, index_col=[0], comment="#")

def get_microscopic_cross_sections_df(directory="."):
    """
    Read the .csv of microscopic cross-sections from stated directory,
    And return it as a pandas dataframe.
    """
    expected_microscopic_xs_path = join(directory, "microscopic_xs.csv")
    assert exists(expected_microscopic_xs_path),"Output directory must already contain microscopic_xs.csv"
    microscopic_xs = pd.read_csv(expected_microscopic_xs_path, index_col=[0])
    return microscopic_xs

def get_parameters_json(directory):
    """
    Open the "parameters_used.json" file if it exists at the directory provided.
    Else return an empty file.
    """
    json_filename = join(directory, "parameters_used.json")
    # read the json file if it exist
    if exists(json_filename):
        with open(json_filename) as f:
            json_data = json.load(f)
    else:
        json_data = {}
    return json_data

def save_parameters_as_json(directory, parameter_dict):
    """
    Saves the parameter used in this run.
    search for parameters_used.json in the directory, open it and save the parameter_dict.
    """
    json_filename = join(directory,"parameters_used.json")
    json_data = get_parameters_json(directory)

    # update the content
    json_data.update(parameter_dict)

    json_filename = join(directory,"parameters_used.json")
    with open(json_filename, "w") as f:
        json.dump(json_data, f)
    return
