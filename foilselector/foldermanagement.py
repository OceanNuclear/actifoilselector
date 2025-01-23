"""functions used for reading and saving data.
All of the read_* and save_* functions all saves at the current directory by default,
    unless their save filepath is changed
"""

import json as json
import os
from glob import glob
from os.path import exists, join
from pathlib import Path

import numpy as np
import pandas as pd

from foilselector.selfshielding import MaxSigma


# atomic_composition
def save_atomic_composition_json(
    processed_composition,
    json_filename: str = ".atomic_composition.json",
    cwd: str | None = None,
):
    cwd = cwd or Path.cwd()
    json_fullpath = Path(cwd, json_filename)
    print("saving the processed composition file to", json_fullpath)
    with open(json_fullpath, "w") as j:
        return json.dump(processed_composition, j, indent=1)


def read_atomic_composition_json(json_filename=".atomic_composition.json"):
    with open(json_filename) as j:
        return json.load(j)


# selfshielding_dict
def save_self_shielding(selfshielding_dict, json_path=".self-shielding.json"):
    with open(json_path, "w") as j:
        return json.dump(selfshielding_dict, j)


def read_self_shielding(json_path=".self-shielding.json"):
    with open(json_path) as j:
        return MaxSigma(json.load(j))


def read_gs(file_path):
    return pd.read_csv(file_path).values


def read_flux(file_path):
    return np.squeeze(pd.read_csv(file_path).values)


def get_durations_from_csv(file: str):
    """Get the durations saved at the comments before the header of the csv file.

    Parameters
    ----------
    file:
        a plain text file (.csv file) that should start with at least 3 comments lines,
        each of them started with a '#' character.

    """
    with open(file) as csv:
        while True:
            comment_line = csv.readline()
            if not comment_line.startswith("#"):
                break
            if "duration" in comment_line:
                if "irradiation" in comment_line:
                    irradiation_duration = float(comment_line.split()[-1])
                elif "trans" in comment_line:
                    transit_duration = float(comment_line.split()[-1])
                elif ("measurement" in comment_line) or ("acquisition" in comment_line):
                    acquisition_duration = float(comment_line.split()[-1])
    return irradiation_duration, transit_duration, acquisition_duration


#### The rest of these functions below aren't going to be needed. ####
def get_apriori(directory: Path, irradiation_duration: float | None = None):
    """Given the file location and irradiation duration, return the apriori_flux and
    the apriori_fluence.
    """
    assert exists(
        join(directory, ".integrated_apriori.csv"),
    ), "Output directory must already have integrated_apriori.csv for calculating the radionuclide populations."
    print(
        f"Reading integrated_apriori.csv as the fluence, i.e. total number of neutrons/cm^2/eV/s, averaged over the IRRADIATION_DURATION = {irradiation_duration} s\n",
    )
    apriori_flux = pd.read_csv(
        join(directory, ".integrated_apriori.csv"),
    )[
        "value"
    ].values  # integrated_apriori.csv is a csv with header = value and number of rows = len(gs); no index.
    if irradiation_duration is None:
        return apriori_flux
    apriori_fluence = apriori_flux * irradiation_duration
    return apriori_flux, apriori_fluence


class ResolutionMaxCountRate:
    """Data on the resolution curve of the gamma-ray detector, and the maximum count rate
    at which this resolution can be achieved without degredation.
    """

    def __init__(self, resolution_coefficients: list[float], max_count_rate: float):
        self.resolution_coefficients = resolution_coefficients
        self.max_count_rate = max_count_rate

    def save(self, directory="."):
        """Store data as plain text file."""
        with open(join(directory, ".gamma-resolution-count-rate-coefs.txt"), "w") as f:
            for i, coef in enumerate(self.resolution_coefficients):
                f.write(f"x_{i}={coef}\n")
            f.write(f"max. count rate={self.max_count_rate}\n")

    @staticmethod
    def load(directory="."):
        """Load data back from the '.gamma-resolution-count-rate-coefs.txt' file"""
        with open(join(directory, ".gamma-resolution-count-rate-coefs.txt")) as f:
            text = f.readlines()
        resolution_coefficients = []
        while text:
            if text[0].startswith("x"):
                resolution_coefficients.append(float(text.pop(0).split("=")[1]))
            else:
                break
        assert text[0].startswith("max"), "Expected x_0=...,x_1=...,max. count rate=..."
        max_count_rate = float(text.pop(0).split("=")[1])
        return resolution_coefficients, max_count_rate


class PeakToComptonCoefficients:
    """Data on the resolution curve of the gamma-ray detector, and the maximum count rate
    at which this resolution can be achieved without degredation.
    """

    def __init__(self, peak_to_Compton_coefficients: list[float]):
        self.peak_to_Compton_coefficients = peak_to_Compton_coefficients

    def save(self, directory="."):
        """Store data as plain text file."""
        with open(join(directory, ".gamma-Compton-to-peak-coefs.txt"), "w") as f:
            for i, coef in enumerate(self.peak_to_Compton_coefficients):
                f.write(f"logx_{i}={coef}\n")

    @staticmethod
    def load(directory="."):
        """Load data back from the '.gamma-Compton-to-peak-coefs.txt' file"""
        with open(join(directory, ".gamma-Compton-to-peak-coefs.txt")) as f:
            text = f.readlines()
        peak_to_Compton_coefficients = []
        while text:
            if text[0].startswith("logx"):
                peak_to_Compton_coefficients.append(float(text.pop(0).split("=")[1]))
            else:
                break
        return peak_to_Compton_coefficients


def find_efficiency_file():
    return glob(".efficiency.*")[0]


# def get_gs_and_flux(file_path, directory="."):
#     return pd.read_csv(file_path, index_col=[0], comment="#")


def get_microscopic_cross_sections_df(directory="."):
    """Read the .csv of microscopic cross-sections from stated directory,
    And return it as a pandas dataframe.
    """
    expected_microscopic_xs_path = join(directory, "microscopic_xs.csv")
    assert exists(
        expected_microscopic_xs_path,
    ), "Output directory must already contain microscopic_xs.csv"
    microscopic_xs = pd.read_csv(expected_microscopic_xs_path, index_col=[0])
    return microscopic_xs


def get_parameters_json(directory):
    """Open the ".parameters_used.json" file if it exists at the directory provided.
    Else return an empty file.
    """
    json_filename = join(directory, ".parameters_used.json")
    # read the json file if it exist
    if exists(json_filename):
        with open(json_filename) as f:
            json_data = json.load(f)
    else:
        json_data = {}
    return json_data


def save_parameters_as_json(directory, parameter_dict):
    """Saves the parameter used in this run.
    search for .parameters_used.json in the directory, open it and save the parameter_dict.
    """
    json_filename = join(directory, ".parameters_used.json")
    json_data = get_parameters_json(directory)

    # update the content
    json_data.update(parameter_dict)

    json_filename = join(directory, ".parameters_used.json")
    with open(json_filename, "w") as f:
        json.dump(json_data, f)


def append_to_json(obj: dict, json_path: Path) -> None:
    if not Path(json_path).exists():
        create_template_json(json_path)
    with open(json_path, "r+b") as j:
        if j.seek(-2, os.SEEK_END) > 1:
            starting = ","
        else:
            starting = ""
        content = json.dumps(obj, indent=1)[1:-1]
        closing = "}"
        j.write((starting + content + closing).encode())


def append_to_csv(row_name: str, data: dict, csv_path: Path = Path("each_foil.csv")):
    """Append to a .csv, where the column names order are supposed to match the ordering
    of the keys of the data dictionary.
    Create the .csv file if it doesn't already exist.

    Parameters
    ----------
    row_name:
        the title of that row
    data:
        A dictionary (ordered by default since python 3.7), whose keys should match the
        column names of the csv file, in the correct order.

    """
    if not Path(csv_path).exists():
        with open(csv_path, "w") as csv:
            csv.write("foil_name," + ",".join([str(i) for i in data]) + "\n")
    with open(csv_path, "a") as csv:
        csv.write(row_name + "," + ",".join([str(i) for i in data.values()]) + "\n")


def create_template_json(json_path: Path) -> None:
    with open(json_path, "w") as j:
        j.write("{\n}")
