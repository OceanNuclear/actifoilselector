from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from foilselector.openmcextension.library_reader import DiscreteRadiation


def get_all_reactions(expected_peak_list: list[DiscreteRadiation]) -> set:
    """
    Compile together all distinct reactions that is used to generate the list of expected
    peaks.

    Parameters
    ----------
    expected_peak_list:
        List of expected peaks, where neighbouring peaks have already been merged
        together and undetectable peaks removed.

    Returns
    -------
    source_set:
        Set of all reaction pathways (each stored as a string)
    """
    # TODO: @OceanNuclear:
    # Using str to store and transfer reaction information is not the most secure, nor
    # has this syntax of "Z101 -> X101 -> Y101 gamma from Y101 beta-; ..." been
    # formalized/protected by checks at initialization. Therefore this function needs
    # to be protected by a test, and/or DiscreteRadiation.source needs to have a setter
    # that validates this syntax is used. Or better yet, make sure
    # DiscreteRadiation.source stores a specialized class rather than a str.
    source_set = set()
    for peak in expected_peak_list:
        for source in peak.source.split(";"):
            reaction_chain = source.split(" from")[0][:-5].strip().split("->")
            source_set.add("->".join(reaction_chain[:2]))
    return source_set


def get_accuracy_upper_bound(
    response_matrix: np.ndarray[float], expected_peak_list: list[DiscreteRadiation]
) -> int:
    """
    The rank of the response matrix is bounded above by the number of distinct reactions
    used to construct that matrix.
    Each reaction, assuming that it is linearly independent to the rest of the reactions,
    contribute another rank to the response matrix.

    Parameters
    ----------
    response_matrix:
        response matrix where each line is the generation of each expected peak.
    expected_peak_list:
        List of expected peaks, where neighbouring peaks have already been merged
        together and undetectable peaks removed.
    """
    non_zero_bins = response_matrix.sum(axis=0) > 0.0
    return min([non_zero_bins.sum(), len(get_all_reactions(expected_peak_list))])
