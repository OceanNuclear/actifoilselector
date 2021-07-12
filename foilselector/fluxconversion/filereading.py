from glob import glob
import csv
import os
import pandas as pd

join, base = os.path.join, os.path.basename


def list_dir_csv(directory):
    """
    Pretty print the list of all csv's in a specified directory.
    """
    fnames = glob(join(directory, "*.csv"))
    print("########################")
    print("------------------------")
    for f in fnames:
        print(os.path.basename(f))
    print("------------------------")
    print("########################")
    return fnames


def open_csv(fname):
    """
    General function that opens any csv, where ",|±" are all intepreted as separators, and Header is optional.
    """
    sniffer = csv.Sniffer()
    if sniffer.has_header(fname):
        df = pd.read_csv(fname, sep=",|±", skipinitialspace=True, engine="python")
    else:
        df = pd.read_csv(
            fname, sep=",|±", header=None, skipinitialspace=True, engine="python"
        )
    return df, df.columns


def get_integrated_apriori_value_only(directory):
    """
    Find integrated_apriori.csv in a specified directory and load in its values.
    Written and saved in this module so that it can be imported by other modules in the future.
    (I.e. written here for future compatibility)
    """
    full_file_path = join(directory, "integrated_apriori.csv")
    integrated_apriori = pd.read_csv(
        full_file_path, sep="±|,", engine="python", skipinitialspace=True
    )
    return integrated_apriori["value"].values


def get_gs_ary(directory):
    """
    Find the group structure of a specified directory and load in its values.
    Written and saved in this module so that it can be imported by other modules in the future.
    (I.e. written here for future compatibility).
    """
    full_file_path = join(directory, "gs.csv")
    gs = pd.read_csv(full_file_path, skipinitialspace=True)
    return gs.values


def get_continuous_flux(directory):
    """
    retrieve the continuous flux distribution (openmc.data.Tabulated1D object, stored in "continuous_apriori.csv") from a directory,
    and turn it back into an openmc.data.Tabulated1D object.
    """
    from foilselector.openmcextension import tabulate

    detabulated_apriori_df = pd.read_csv(join(directory, "continuous_apriori.csv"))
    detabulated_apriori = {
        "x": detabulated_apriori_df["x"].tolist(),
        "y": detabulated_apriori_df["y"].tolist(),
        "interpolation": detabulated_apriori_df["interpolation"].tolist(),
    }
    detabulated_apriori["interpolation"].pop()
    return tabulate(detabulated_apriori)
