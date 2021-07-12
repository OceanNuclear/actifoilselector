"""functions used for reading and saving data"""
import pandas as pd
import os
import json

def get_apriori(file_location, irradiation_duration):
    """
    Given the file location and irradiation duration,
    return the apriori_flux and the apriori_fluence
    """
    assert os.path.exists(os.path.join(file_location, 'integrated_apriori.csv')), "Output directory must already have integrated_apriori.csv for calculating the radionuclide populations."
    print("Reading integrated_apriori.csv as the fluence, i.e. total number of neutrons/cm^2/eV/s, averaged over the IRRADIATION_DURATION = {} s\n".format(irradiation_duration))
    apriori_flux = pd.read_csv(os.path.join(file_location, 'integrated_apriori.csv'))['value'].values # integrated_apriori.csv is a csv with header = value and number of rows = len(gs); no index.
    apriori_fluence = apriori_flux * irradiation_duration
    return apriori_flux, apriori_fluence

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