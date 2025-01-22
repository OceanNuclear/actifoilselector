from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from foilselector.generic import SilenceNumpyDivisionError, SilenceNumpyInvalidError
from uncertainties.core import AffineScalarFunc, Variable
from uncertainties import nominal_value as nom, std_dev

FLOAT_THRESHOLD = np.finfo(float).resolution * 10

if TYPE_CHECKING:
    from foilselector.openmcextension.library_reader import DiscreteRadiation


def response_matrix_rank_at_given_vector(
    response_matrix: Iterable[Iterable[float]],
    response_vector: Iterable[AffineScalarFunc],
    error_principle_ratio_threshold: float = 0.2,
    *,
    rank_counter: int = 0,
):
    """
    Find the number of useful directions of the response matrix near the a priori vector,
    i.e. the number of non-singular directions of the transformation represented by the
    response matrix in the phi-space near phi=a priori.

    Given a response matrix R, the response_vector = (R@phi) in real-life would be a
    measured quantity with some associated uncertainty. We then make an Augmented Matrix
    (response_matrix|response_vector), and perform a row reduction procedure.

    Because both the response_matrix and the response_vector are all-positive, the
    scaling factor used during the subtraction substep of each step in the reduction
    procedure will still be positive.

    Parameters
    ----------
    response_matrix:
        np.ndarray

    Example
    -------
    Below is a text example without including the uncertainties:

    a priori = [1,2,3,4,5].T
    response matrix = [
    [0,2,2,2,2],
    [5,2,1,1,1],
    [0,2,3,1,1],
    ]
    Augmented matrix = (
    0 2 2 2 2|28
    5 2 1 1 1|21
    0 2 3 1 1|22
    )
    => subtract scaled copies of the most effective row (
    0       2       2       2       2      |28        <- can ignore this row now
    5-0.5*0 2-0.5*2 1-0.5*2 1-0.5*2 1-0.5*2|21-0.5*28
    0-0.5*0 2-0.5*2 3-0.5*2 1-0.5*2 1-0.5*2|22-0.5*28
    )
    => remove most effective row (
    5 1 0 0 0|7
    0 1 2 0 0|8
    )
    = subtract scaled copies of the most effective row (
    5-0 1-0 0-0 0-0 0-0|7-0
    0   1   2   0   0  |8   <- can ignore this row now
    )
    => remove most effective row (
    5 1 0 0 0|7
    )
    => subtract scaled copies of the most effective row (
    5 1 0 0 0|7 <- can ignore this row now
    )
    => remove most effective row (
    )
    DONE
    """
    if len(response_matrix) == 0:
        return rank_counter

    error_principle_ratios = safely_get_error_principle_ratio(response_vector)
    # Keep only rows with a small enough fraction of uncertainty, i.e.
    # remove rows of large uncertainty fraction
    keep_row = abs(error_principle_ratios) < error_principle_ratio_threshold
    if sum(keep_row) == 0:
        # Recursion termination condition:
        # stop if no rows is left to be used for the row reduction
        return rank_counter
    most_effective_row_index = np.argmin(error_principle_ratios)
    most_effective_row = response_matrix[most_effective_row_index]
    most_effective_count = response_vector[most_effective_row_index]
    remaining_response_matrix, remaining_response_vector = [], []
    for i, (response_row, count) in enumerate(zip(response_matrix, response_vector)):
        if i == most_effective_row_index:  # remove most effective row.
            continue
        if keep_row[i]:
            num_basis_contained = get_max_num_basis(response_row, most_effective_row)
            row_after_reduction = response_row - num_basis_contained * most_effective_row
            count_after_reduction = count - num_basis_contained * most_effective_count
            if count_after_reduction<0:
                count_after_reduction += abs(nom(count_after_reduction)) # zero it.
            remaining_response_matrix.append(np.clip(row_after_reduction, 0.0, np.inf))
            remaining_response_vector.append(count_after_reduction)
    return response_matrix_rank_at_given_vector(
        np.array(remaining_response_matrix),
        np.array(remaining_response_vector),
        error_principle_ratio_threshold=error_principle_ratio_threshold,
        rank_counter=rank_counter + 1,
    )

def response_matrix_rank_new(
    response_matrix, response_vector, *, error_principle_ratio_threshold=0.2
):
    """
    Trying to put the response matrix in row echelon form.
    """
    rank = 0
    for _ in range(len(response_matrix)):
        error_principle_ratios = safely_get_error_principle_ratio(response_vector)
        rows_in_play = error_principle_ratios < error_principle_ratio_threshold
        most_effective_row_index = np.argmin(error_principle_ratios)
        most_effective_row = response_matrix[most_effective_row_index]
        most_effective_count = response_vector[most_effective_row_index]
        for i, this_row_still_in_play in enumerate(rows_in_play):
            if this_row_still_in_play and i!=most_effective_row_index:
                num_basis_contained = get_max_num_basis(response_matrix[i], most_effective_row)
                response_matrix[i] -= num_basis_contained * most_effective_row
                response_vector[i] -= num_basis_contained * most_effective_count
        response_matrix[most_effective_row_index] -= most_effective_row
        response_vector[most_effective_row_index] = Variable(0.0, 0.1)
        response_matrix = np.clip(response_matrix, 0, np.inf)
        rank += 1
    return rank

def get_accuracy(
    response_matrix: np.ndarray, response_vector: np.ndarray[AffineScalarFunc]
) -> int:
    """
    Finds the accuracy score of a given set of response-matrix and response-vector.
    """
    reduced_matrix, reduced_vector = reduce_matrix_vector(response_matrix, response_vector)
    accuracy_score = 0
    for count in reduced_vector:
        if nom(count)<FLOAT_THRESHOLD:
            accuracy_score += 1
    return accuracy_score

def safely_get_error_principle_ratio(vector: Iterable[AffineScalarFunc]) -> Iterable[AffineScalarFunc]:
    """
    Get the ratio of the standard deviation and value of the vector for each element in
    a vector, while silencing any division related error or warnings.

    Parameters
    ----------
    vector:
        A vector of AffineScalarFunc's.

    Returns
    -------
    :
        A vector of floats, each representing the ratio between the std-deviation and the
        nominal value of the random variable. Any division with 0 as the denominator
        (i.e. nominal value = 0 for the random variable) will output +ve infinity.
        Co-domain/ output range: (0.0, 1.0]
    """
    ratios = []
    for v in vector:
        if v.s:
            ratios.append(v.s/np.clip(v.n, v.s, np.inf))
        else:
            ratios.append(1.0)
    return np.array(ratios)

def get_max_num_basis(row_to_be_subtracted: np.ndarray[float], basis: np.ndarray[float]) -> float:
    """
    Count how many copies of 'basis' we can take out of row_to_be_subtracted before any
    element goes negative.

    Parameters
    ----------
    row_to_be_subtracted:
        A vector of nonnegative float values.
    num_basis_contained:
        A vector of nonnegative float values of the same length as row_to_be_subtracted.

    Returns
    -------
    num_bases_contained:
        A number such that min(row_to_be_subtracted - num_bases_contained * basis)== 0.0.
        row_to_be_subtracted - num_bases_contained * basis >= row_to_be_subtracted,
        element-wise.
    """
    with SilenceNumpyDivisionError():
        # sometimes 0/0 yields invalid rather than division error.
            with SilenceNumpyInvalidError():
                num_basis_contained = np.nanmin(row_to_be_subtracted / basis)
    return num_basis_contained

def num_bases_contained(matrix: np.ndarray, basis: np.ndarray[float]) -> np.ndarray[float]:
    """
    Number of copies of the chosen row ('basis') that can be subtracted from
    each row of the matrix.

    Parameters
    ----------
    matrix:
        2D nonnegative matrix, with shape (m,n)
    basis:
        1D nonnegative vector, with shape (n).

    Returns
    -------
    :
        Max number of basis contained by each row.
    """
    return np.array([get_max_num_basis(row, basis) for row in matrix])

def num_bases_row_contained(matrix: np.narray, index: int) -> np.ndarray[float]:
    """
    Number of basis row contained by every other row of the matrix.
    Since we know at matrix[index]==basis, this number should be 1.0, so we want to
    *ignore* this row, and return 0.0 instead of 1.0 for this position.

    Parameters
    ----------
    matrix:
        2D nonnegative matrix, with shape (m,n)
    index:
        index of the row in the 2D matrix that we're trying to use as the basis.

    Returns
    -------
    num_copies_subtractable:
        Number of copies that can be subtracted 
    """
    basis_row = matrix[index]
    num_copies_subtractable = num_bases_contained(matrix, basis_row)
    num_copies_subtractable[index] = 0.0
    return num_copies_subtractable

def subtract_row(matrix: np.ndarray, vector: np.ndarray[AffineScalarFunc], index: int, num_copies_subtractable: np.ndarray[float]) -> np.ndarray:
    """Subtract off a row.
    This function modifies the original matrix, so the usage is
    matrix = subtract_row(matrix, i, num_copies)

    Parameters
    ----------
    matrix:
        The 2D matrix that we intend to modify.
    
    index:
        The index of the row that we would like to subtract off of the whole matrix.
    num_copies_subtractable:
        A pre-calculated constant, such that 

    Returns
    -------
    matrix:
        Same matrix, but reduced
    """
    basis_row, basis_count = matrix[index], vector[index]
    matrix -= np.outer(num_copies_subtractable, basis_row)
    vector -= num_copies_subtractable * basis_count
    return np.clip(matrix, 0, np.inf), vector

def reduce_matrix_vector(matrix: np.ndarray, vector: np.ndarray[AffineScalarFunc]) -> np.ndarray:
    """
    Reduce a response matrix and its associated response vector to a form sufficient to
    be rearranged into row-echelon form.

    Parameters
    ----------
    response_matrix:
        the response matrix that we want to find out the rank for.
    vector:
        The response vector that is obtained by response matrix @ apriori fluence, but
        with associated errors as computed by other simulation steps.

    Returns
    -------
    matrix:
        The same response matrix, but *missing*.
    """
    matrix, vector = matrix.copy(), vector.copy() # don't overwrite the original.
    matrix = matrix.T[matrix.sum(axis=0)>0.0].T
    while True:
        row_indices_to_be_reduced = get_valid_rows_id(vector)
        for i in row_indices_to_be_reduced:
            num_copies = num_bases_row_contained(matrix, i)
            if num_copies.sum()>0.0:
                matrix, vector = subtract_row(matrix, vector, i, num_copies)
        else:
            break # outside while loop
    return matrix, vector

def get_valid_rows_id(remaining_response_vector: np.ndarray[float]):
    """
    Find the indices of rows that are still valid, in ascending order of error:principle
    ratio, A.K.A. descending order of effective accuracy contribution.

    Parameters
    ----------
    remaining_response_vector:
        The response vector after being modified by the reduce_matrix_vector function,
        where copies of the thing were subtracted
    error_principle_ratio_threshold:
        The threshold which, if the ratio is larger than, 

    Returns
    -------
    :
        Indices of rows which still has low enough error:principle ratio.
    """
    ratios = safely_get_error_principle_ratio(remaining_response_vector)
    return np.argsort(ratios)

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
            source_set.add(source.split(" from")[0][:-5].strip())
    return source_set

def get_accuracy_upper_bound(
    response_matrix: np.ndarray[float],
    expected_peak_list: list[DiscreteRadiation]
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
    non_zero_bins = response_matrix.sum(axis=0)>0.0
    return min([non_zero_bins.sum(), len(get_all_reactions(expected_peak_list))])