"""
Getting the name various interpolation schemes /their indices in openmc and endf.
"""
from openmc.data import INTERPOLATION_SCHEME


def get_interpolation_scheme(scheme):
    """
    invert the dictionary from {1:"log-log", ..., 5:"histogramic"}
    to {"log-log":1, ..., "histogramic":5},
    and then get the scheme index corresponding to the variable 'scheme'.
    """
    interpolation_to_index = {v: k for k, v in INTERPOLATION_SCHEME.items()}
    return interpolation_to_index[scheme]


loglog = get_interpolation_scheme("log-log")
histogramic = get_interpolation_scheme("histogram")
