import numpy as np
import contextlib # to silence numpy error


def ordered_set(sequence):
    """
    Get the sorted set, sorted according to the order of element first appearing in the sequence.
    source:
    http://www.martinbroadhurst.com/removing-duplicates-from-a-list-while-preserving-order-in-python.html
    date accessed website: 2021-01-19 11:44:23
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
    # GENIUS!
    # (x in seen) -> stop evaluating;
    # (x not in seen) -> add x to seen -> bracket returns False
    #   -> negated by "not" in front of bracket -> adds element to list
    # This should be an O(n) operation.

def sorted_dict(dictionary: dict):
    """
    Python dictionaries are sorted by default now, so we don't need to import OrderedDict
    """
    sorted_keys = sorted(list(dictionary.keys()))
    new_dict = {}
    while sorted_keys:
        next_lowest = sorted_keys.pop(0)
        new_dict[next_lowest] = dictionary[next_lowest]
    return new_dict

def minmax(array):
    """
    Alias function to quickly return the minimum and maximum among all values in an array.
    parameters
    ----------
    array : any shaped array

    returns
    -------
    tuple containing a min (scalar) and a max (scalar)
    """

    return np.min(array), np.max(array)

class SilenceNumpyDivisionError(contextlib.ContextDecorator):
    """Context manager to suppress warnings and errors for dividing by zero."""
    def __enter__(self):
        self.prev_divide_error_state = np.geterr()["divide"] # record current state of error handling style
        np.seterr(divide="ignore") # force ignore all division errors
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        np.seterr(divide=self.prev_divide_error_state) # undo error silencing
        if exc_type is None:
            return True
        else: # any type of error
            return False
