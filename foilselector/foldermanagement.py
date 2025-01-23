"""functions used for reading and saving data.
All of the read_* and save_* functions all saves at the current directory by default,
    unless their save filepath is changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from foilselector.selfshielding import MaxSigma

if TYPE_CHECKING:
    from collections.abc import Iterable

GAMMA_RES_AND_COUNT_RATE_FILENAME = ".gamma-resolution-count-rate-coefs.txt"
PEAK_TO_COMPTON_FILENAME = ".gamma-Compton-to-peak-coefs.txt"
APRIORI_FILENAME = ".integrated_apriori.csv"
SIGMA_CSV = "microscopic_xs.csv"
PARAM_JSON_FILE = ".parameters_used.json"


# atomic_composition
def save_atomic_composition_json(
    processed_composition,
    json_filename: str = ".atomic_composition.json",
    cwd: str | None = None,
):
    """Save the atomic composition as a json file."""
    cwd = cwd or Path.cwd()
    json_fullpath = Path(cwd, json_filename)
    print("saving the processed composition file to", json_fullpath)
    with json_fullpath.open("w") as j:
        json.dump(processed_composition, j, indent=1)


def read_atomic_composition_json(json_filename=".atomic_composition.json"):
    with Path(json_filename).open() as j:
        return json.load(j)


# selfshielding_dict
def save_self_shielding(selfshielding_dict, json_path=".self-shielding.json"):
    with Path(json_path).open("w") as j:
        return json.dump(selfshielding_dict, j)


def read_self_shielding(json_path=".self-shielding.json"):
    with Path(json_path).open() as j:
        return MaxSigma(json.load(j))


def read_gs(file_path: Path):
    return pd.read_csv(Path(file_path)).to_numpy()


def read_flux(file_path: Path):
    return np.squeeze(pd.read_csv(Path(file_path)).values)


def get_durations_from_csv(file: Path):
    """Get the durations saved at the comments before the header of the csv file.

    Parameters
    ----------
    file:
        a plain text file (.csv file) that should start with at least 3 comments lines,
        each of them started with a '#' character.

    Returns
    -------
    irradiation_duration:
        The length of the irradiation, in [seconds].
    transport_duration:
        The length of the transport period, in [seconds].
    measurement_duration:
        The length of the measurement/gamma-spectrum acquisition period, in [seconds].
    """
    with Path(file).open() as csv:
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


# The rest of these functions below aren't going to be needed. ####
def get_apriori(directory: Path, irradiation_duration: float | None = None):
    """Get the apriori neutron spectrum, as flux (and potentially fluence).

    Parameters
    ----------
    directory:
        The location where the apriorio
    irradiation_duration:
        The length of the irradiation duration. flux * irradiation_duration = fluence

    Returns
    -------
    apriori_flux:
        Neutron flux in each bin per second of irradiation. Unit: [neutrons s^-1 cm^-2]
    apriori_fluence:
        Neutron fluence in each bin over the course of the entire irradiation duration.
        Unit: [neutrons cm^-2]
        Only outputted if irradiation_duration is provided.

    Raises
    ------
    FileNotFoundError
        Raised if apriori file
    """
    apriori_path = Path(directory, APRIORI_FILENAME)
    if not apriori_path.exists():
        raise FileNotFoundError(
            f"Output directory must already have {APRIORI_FILENAME} for "
            "calculating the radionuclide populations.",
        )
    print(
        f"Reading {APRIORI_FILENAME} as the fluence, i.e. "
        "total number of neutrons/cm^2/eV/s,"
        f" averaged over the IRRADIATION_DURATION = {irradiation_duration} s\n",
    )
    # A csv file with header = "value" and number of rows = len(gs); no index.
    apriori_flux = pd.read_csv(apriori_path)["value"].to_numpy()
    if irradiation_duration:
        apriori_fluence = apriori_flux * irradiation_duration
        return apriori_flux, apriori_fluence
    return apriori_flux


class ResolutionMaxCountRate:
    """Data on the resolution curve of the gamma-ray detector, and the maximum count rate
    at which this resolution can be achieved without degredation.
    """

    def __init__(self, resolution_coefficients: Iterable[float], max_count_rate: float):
        """
        Initialize from an Iterable of resolution curve coefficients and the maximum
        count rate.

        Paraemters
        ----------
        resolution_coefficients:
            The polynomial coefficients that get the resolution (expressed as FWHM) as
            FWHM = sqrt(polynomial(energy)).
            See :func:`~resolution_curve_factory` for more details.
        """
        self.resolution_coefficients = resolution_coefficients
        self.max_count_rate = max_count_rate

    def save(self, directory="."):
        """Store data as plain text file."""
        with Path(directory, GAMMA_RES_AND_COUNT_RATE_FILENAME).open("w") as f:
            for i, coef in enumerate(self.resolution_coefficients):
                f.write(f"x_{i}={coef}\n")
            f.write(f"max. count rate={self.max_count_rate}\n")

    @staticmethod
    def load(directory=".") -> tuple[list[float], float]:
        """Load data back from the GAMMA_RES_AND_COUNT_RATE_FILENAME file.

        Returns
        -------
        Directly return the two objects:
            resolution_coefficients, max_count_rate [float].

        Raises
        ------
        ValueError
            Raised when the text inside the GAMMA_RES_AND_COUNT_RATE_FILENAME does not
            match the expected text.
        """
        with Path(directory, GAMMA_RES_AND_COUNT_RATE_FILENAME).open() as f:
            text = f.readlines()
        resolution_coefficients = []
        while text:
            if text[0].startswith("x"):
                resolution_coefficients.append(float(text.pop(0).split("=")[1]))
            else:
                break
        if not text[0].startswith("max"):
            raise ValueError("Expected x_0=...\nx_1=...\n...\nmax. count rate=...")
        max_count_rate = float(text.pop(0).split("=")[1])
        return resolution_coefficients, max_count_rate


class PeakToComptonCoefficients:
    """Data on the resolution curve of the gamma-ray detector, and the maximum count rate
    at which this resolution can be achieved without degredation.
    """

    def __init__(self, peak_to_Compton_coefficients: Iterable[float]):
        """
        Initialize object from an Iterable of coefficients describing the peak-to-Compton
        curve.

        Paraemters
        ----------
        peak_to_Compton_coefficients:
            The polynomial coefficients that get the efficiency as
            log(peak-to-Compton ratio) = polynomial(log(energy)).
            See :func:`~ComptonToPeakRatioCurve` for more details.
        """
        self.peak_to_Compton_coefficients = peak_to_Compton_coefficients

    def save(self, directory="."):
        """Store data as plain text file."""
        with Path(directory, PEAK_TO_COMPTON_FILENAME).open("w") as f:
            for i, coef in enumerate(self.peak_to_Compton_coefficients):
                f.write(f"logx_{i}={coef}\n")

    @staticmethod
    def load(directory=".") -> list[float]:
        """
        Load data back from the PEAK_TO_COMPTON_FILENAME file.

        Returns
        -------
        peak_to_Compton_coefficients:
            Directly return the coefficients describing the peak-to-Compton curve.
        """
        with Path(directory, PEAK_TO_COMPTON_FILENAME).open() as f:
            text = f.readlines()
        peak_to_Compton_coefficients = []
        while text:
            if text[0].startswith("logx"):
                peak_to_Compton_coefficients.append(float(text.pop(0).split("=")[1]))
            else:
                break
        return peak_to_Compton_coefficients


def find_efficiency_file():
    return Path(Path.cwd()).glob(".efficiency.*")[0]


# def get_gs_and_flux(file_path, directory="."):
#     return pd.read_csv(file_path, index_col=[0], comment="#")


def get_microscopic_cross_sections_df(directory="."):
    """Read the .csv of microscopic cross-sections from stated directory,
    And return it as a pandas dataframe.

    Parameters
    ----------
    directory:
        directory to load the SIGMA_CSV from.

    Returns
    -------
    microscopic_xs:
        The entire database of microscopic cross-sections used.

    Raises
    ------
    FileNotFoundError
        Raised if SIGMA_CSV does not already exist at the directory.
    """
    expected_microscopic_xs_path = Path(directory, SIGMA_CSV)
    if not expected_microscopic_xs_path.exists():
        raise FileNotFoundError(f"Output directory must already contain {SIGMA_CSV}")
    return pd.read_csv(expected_microscopic_xs_path, index_col=[0])


def get_parameters_json(directory) -> dict:
    """Open the PARAM_JSON_FILE file if it exists at the directory provided.
    Else return an empty dict.

    Returns
    -------
    json_data:
        The dictionary stored in the PARAM_JSON_FILE file.
    """
    json_filename = Path(directory, PARAM_JSON_FILE)
    # read the json file if it exist
    if json_filename.exists():
        with json_filename.open() as f:
            json_data = json.load(f)
    else:
        json_data = {}
    return json_data


def save_parameters_as_json(directory, parameter_dict):
    """
    Search for PARAM_JSON_FILE in the directory, open it and append the
    parameter_dict, and then save at the same location.

    Parameters
    ----------
    parameter_dict:
        The dictionary of data to be added to the existing parameter dict.
    """
    json_data = get_parameters_json(directory)

    # update the content
    json_data.update(parameter_dict)

    json_filename = Path(directory, PARAM_JSON_FILE)
    with json_filename.open("w") as f:
        json.dump(json_data, f)


def append_to_json(obj: dict, json_path: Path) -> None:
    """Open a json file, and append to its data dictionary, while adhering to the
    indent=1 json syntax, without deleting previous data.
    Create it if it does not already exist.

    Parameters
    ----------
    obj:
        data to be appended to the json file.
    json_path:
        path to the json file.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        _create_template_json(json_path)
    with json_path.open("r+b") as j:
        starting = "," if j.seek(-2, os.SEEK_END) > 1 else ""
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
    csv_path = Path(csv_path)
    if not csv_path.exists():
        with csv_path.open("w") as csv:
            csv.write("foil_name," + ",".join([str(i) for i in data]) + "\n")
    with csv_path.open("a") as csv:
        csv.write(row_name + "," + ",".join([str(i) for i in data.values()]) + "\n")


def _create_template_json(json_path: Path) -> None:
    """Support function used to create the json file if it does not exist in
    :func:`~append_to_json`.
    """
    with Path(json_path).open("w") as j:
        j.write("{\n}")
