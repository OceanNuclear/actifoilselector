"""
Functions used to support graph plotting,
which is used to display the flux and its binning so to let the user decide
    if the graph makes sense or not.
"""
import numpy as np

def intuitive_logspace(start, stop, *args, **kwargs):
    """
    plug in the actual start and stop limit to the logspace function, without having to take log first.
    """
    logstart, logstop = np.log10([start, stop])
    return np.logspace(logstart, logstop, *args, **kwargs)

def percentile_of(minmax_of_range, percentile=50):
    """
    Find the value that's a certain percent above the minimum of that range.
    e.g. percentile_of([-1,1], 10) = -1 + (2)*0.1 = -0.8.

    parameters
    ----------
    minmax_of_range: list containing the minimum value and maximum value of the range.
    percentile : what percent above the minium is wanted.

    output
    ------
    value corresponding to that percentile value required.
    """
    perc = float(percentile/100)
    return min(minmax_of_range) + perc*abs(np.diff(minmax_of_range))

def minmax(array):
    """
    Quick function to return the minimum and maximum among all values in an array. 
    parameters
    ----------
    array : any shaped array

    returns
    -------
    tuple containing a scalar (min) and a scalar (max)
    """
    return min(array), max(array)