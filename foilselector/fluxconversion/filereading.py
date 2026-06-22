"""Functions used when reading .csv files."""

import csv
from pathlib import Path

import pandas as pd

__all__ = ["list_dir_csv", "open_csv"]


def list_dir_csv(directory: Path):
    """Pretty print the list of all csv's in a specified directory.

    Parameters
    ----------
    directory:
        The location from which we want to get all csv files from.

    Returns
    -------
    fname:
        The list of csv files in the specified directory.
    """
    fnames = list(Path(directory).glob("*.csv"))
    print("########################")
    print("------------------------")
    for f in fnames:
        print(f.name)
    print("------------------------")
    print("########################")
    return fnames


def open_csv(fname: Path):
    """
    General function that opens any csv, where ",|±" are all intepreted as separators,
    and Header is optional.

    Returns
    -------
    df:
        the full dataframe
    df.columns:
        the column names of the dataframe.
    """
    sniffer = csv.Sniffer()
    with open(fname) as f:
        text = f.read()
    if sniffer.has_header(text):
        df = pd.read_csv(fname, sep=",|±", skipinitialspace=True, engine="python")
    else:
        df = pd.read_csv(
            fname,
            sep=",|±",
            header=None,
            skipinitialspace=True,
            engine="python",
        )
    return df, df.columns
