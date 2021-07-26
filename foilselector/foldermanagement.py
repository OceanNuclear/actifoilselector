"""functions used for reading and saving data"""
import pandas as pd
import os
import json

def get_apriori(file_location, irradiation_duration=None):
    """
    Given the file location and irradiation duration,
    return the apriori_flux and the apriori_fluence
    """
    assert os.path.exists(os.path.join(file_location, 'integrated_apriori.csv')), "Output directory must already have integrated_apriori.csv for calculating the radionuclide populations."
    print("Reading integrated_apriori.csv as the fluence, i.e. total number of neutrons/cm^2/eV/s, averaged over the IRRADIATION_DURATION = {} s\n".format(irradiation_duration))
    apriori_flux = pd.read_csv(os.path.join(file_location, 'integrated_apriori.csv'))['value'].values # integrated_apriori.csv is a csv with header = value and number of rows = len(gs); no index.
    if irradiation_duration is None:
        return apriori_flux
    else:
        apriori_fluence = apriori_flux * irradiation_duration
        return apriori_flux, apriori_fluence

def get_gs(directory):
    expected_gs_path = os.path.join(directory, "gs.csv")
    assert os.path.exists(expected_gs_path), "Output directory must already have gs.csv"
    gs = pd.read_csv(expected_gs_path).values
    return gs

def get_microscopic_cross_sections_df(directory):
    """
    Read the .csv of microscopic cross-sections from stated directory,
    And return it as a pandas dataframe.
    """
    expected_microscopic_xs_path = os.path.join(directory, "microscopic_xs.csv")
    assert os.path.exists(expected_microscopic_xs_path),"Output directory must already contain microscopic_xs.csv"
    microscopic_xs = pd.read_csv(expected_microscopic_xs_path, index_col=[0])
    return microscopic_xs

def get_parameters_json(directory):
    """
    Open the "parameters_used.json" file if it exists at the directory provided.
    Else return an empty file.
    """
    json_filename = os.path.join(directory,"parameters_used.json")
    # read the json file if it exist
    if os.path.exists(json_filename):
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
    json_filename = os.path.join(directory,"parameters_used.json")
    json_data = get_parameters_json(directory)

    # update the content
    json_data.update(parameter_dict)

    json_filename = os.path.join(directory,"parameters_used.json")
    with open(json_filename, "w") as f:
        json.dump(json_data, f)
    return
