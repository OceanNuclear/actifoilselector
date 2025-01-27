"""functions used for reading and saving data.
All of the read_* and save_* functions all saves at the current directory by default,
    unless their save filepath is changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_STRUCTURE_FILENAME = ".gs.csv"
APRIORI_FILENAME = ".integrated_apriori.csv"
CONT_APRIORI_FILENAME = ".continuous_apriori.csv"
ATOMIC_COMPOSITION_FILENAME = ".atomic_composition.json"

# TODO @OceanNuclear: convert these to Enums?
RAW_RESPONSE_MATRICES = ".response_matrices.json"
BG_RESPONSE_MATRICES = ".background_response_matrices.json"
EFFECTIVE_RESPONSE_MATRICES = ".effective_response_matrices.json"

RESULT_CSV = "each_foil.csv"


# atomic_composition
def save_atomic_composition_json(
    processed_composition: dict[str, float],
    json_filename: Path | str = ATOMIC_COMPOSITION_FILENAME,
    cwd: str | None = None,
) -> None:
    """Save the atomic composition as a json file."""
    cwd = cwd or Path.cwd()
    json_fullpath = Path(cwd, json_filename)
    print("saving the processed composition file to", json_fullpath)
    with json_fullpath.open("w") as j:
        json.dump(processed_composition, j, indent=1)


def read_atomic_composition_json(
    json_filename: Path | str = ATOMIC_COMPOSITION_FILENAME,
) -> dict[str, float]:
    with Path(json_filename).open() as j:
        return json.load(j)


def read_gs(file_path: Path | str = GROUP_STRUCTURE_FILENAME) -> pd.DataFrame:
    return pd.read_csv(Path(file_path)).to_numpy()


def save_gs(
    gs_df: pd.DataFrame,
    file_path: Path | str = GROUP_STRUCTURE_FILENAME,
) -> None:
    return gs_df.to_csv(Path(file_path), index=False)


def read_flux(file_path: Path) -> np.ndarray[float]:
    return np.squeeze(pd.read_csv(Path(file_path)).values)


def get_durations_from_csv(file: Path) -> float:
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


def save_apriori(
    apriori_vector_df: pd.DataFrame,
    file_path: Path = APRIORI_FILENAME,
) -> None:
    return apriori_vector_df.to_csv(Path(file_path), index=False)


def read_apriori(
    directory: Path,
    irradiation_duration: float | None = None,
) -> np.ndarray[float] | tuple[np.ndarray[float], np.ndarray[float]]:
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


def find_efficiency_file() -> Path:
    return next(Path(Path.cwd()).glob(".efficiency.*"))


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


def append_to_csv(
    row_name: str,
    data: dict,
    csv_path: Path | str = RESULT_CSV,
) -> None:
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
