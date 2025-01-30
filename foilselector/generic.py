"""Generic functions and context managers."""

from __future__ import annotations

import contextlib  # to silence numpy error
from typing import TYPE_CHECKING

import numpy as np
from uncertainties import nominal_value as nom

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "SilenceNumpyDivisionError",
    "SilenceNumpyInvalidError",
    "minmax",
    "ordered_set",
    "sorted_dict",
    "vectorized_nom",
]

vectorized_nom = np.vectorize(nom)


def kahan_sum(a: np.ndarray, axis: int = 0) -> float:
    """Carefully add together sum to avoid floating point precision problem.

    Retrieved (and then modified) from
    https://github.com/numpy/numpy/issues/8786

    Returns
    -------
    t:
        The sum
    """
    s = np.zeros(a.shape[:axis] + a.shape[axis + 1 :])
    c = np.zeros(s.shape)
    for i in range(a.shape[axis]):
        # http://stackoverflow.com/42817610/353337
        y = a[(slice(None),) * axis + (i,)] - c
        t = s + y
        c = (t - s) - y
    return t


def ordered_set(sequence: Iterable) -> list:
    """
    Get the sorted set, sorted according to the order of element first appearing in the
    sequence.

    Source:
    http://www.martinbroadhurst.com/removing-duplicates-from-a-list-while-preserving-order-in-python.html
    date accessed website: 2021-01-19 11:44:23

    Parameters
    ----------
    Sequence:
        An iterable that possibly involve repeated elements.

    Returns
    -------
    :
        A list with no repeated elements, sorted in the order of first appearance in
        `sequence`.
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
    # GENIUS!
    # (x in seen) -> stop evaluating;
    # (x not in seen) -> add x to seen -> bracket returns False
    #   -> negated by "not" in front of bracket -> adds element to list
    # This should be an O(n) operation.


def sorted_dict(dictionary: dict) -> dict:
    """
    Sort a dictionary by its keys.
    Python dictionaries are sorted by default, so the output does not need to be an
    OrderedDict, just a normal `dict` will do.

    Parameters
    ----------
    dictionary:
        The dictionary to be sorted.

    Returns
    -------
    new_dict:
        A rearranged dict of the input dictionary, where each item is copied.
    """
    sorted_keys = sorted(dictionary.keys())
    new_dict = {}
    while sorted_keys:
        next_lowest = sorted_keys.pop(0)
        new_dict[next_lowest] = dictionary[next_lowest]
    return new_dict


def minmax(array: Iterable[float]) -> tuple[float, float]:
    """
    Alias function to quickly return the min. and max. among all values in an array.

    Parameters
    ----------
    array : any shaped array

    Returns
    -------
    tuple containing a min (scalar) and a max (scalar)
    """
    return np.min(array), np.max(array)


class SilenceNumpyDivisionError(contextlib.ContextDecorator):
    """Context manager to suppress warnings and errors for dividing by zero."""

    def __enter__(self):
        """Set to ignore error."""
        self.prev_divide_error_state = np.geterr()["divide"]
        np.seterr(divide="ignore")  # force ignore all division errors
        return self  # noqa: DOC201

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        """Unset error status."""
        np.seterr(divide=self.prev_divide_error_state)
        return exc_type is None  # noqa: DOC201


class SilenceNumpyInvalidError(contextlib.ContextDecorator):
    """
    Context manager to suppress warning specifically about invalid values.
    Dangerous to use.
    """

    def __enter__(self):
        """Set to ignore error."""
        self.prev_invalid_error_state = np.geterr()["invalid"]
        np.seterr(invalid="ignore")
        return self  # noqa: DOC201

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        """Unset error status."""
        np.seterr(invalid=self.prev_invalid_error_state)
        return exc_type is None  # noqa: DOC201
