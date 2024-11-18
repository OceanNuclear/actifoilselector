from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from uncertainties.core import AffineScalarFunc

def response_matrix_rank_at_given_vector(
    response_matrix: np.ndarray,
    response_vector: np.ndarray[AffineScalarFunc],
    error_principle_ratio_threshold: float=1.0,
    *, 
    rank_counter: int=0,
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
    if len(response_matrix)==0:
        return rank_counter

    error_principle_ratios = np.array([count.s/count.n for count in response_vector])
    # remove rows that are too small to be effective.
    keep_row = error_principle_ratios<error_principle_ratio_threshold
    if sum(keep_row)==0: 
        # Recursion termination condition:
        # stop if no rows is left to be used for the row reduction
        return rank_counter
    most_effective_row_index = np.argmin(error_principle_ratios)
    most_effective_row = response_matrix[most_effective_row_index]
    most_effective_count = response_vector[most_effective_row_index]
    remaining_response_matrix, remaining_response_vector = [], []
    for i, (response_row, count) in enumerate(zip(response_matrix, response_vector)):
        if i==most_effective_row_index:  # remove most effective row.
            continue
        if keep_row[i]:
            with SilenceNumpyDivisionError():
                number_of_basis_contained = np.min(response_row/most_effective_row)
                remaining_response_matrix.append(
                    response_row - number_of_basis_contained * most_effective_row
                )
                remaining_response_vector.append(
                    count - number_of_basis_contained * most_effective_count
                )
    return response_matrix_rank_at_given_vector(
        np.array(remaining_response_matrix),
        np.array(remaining_response_vector),
        error_principle_ratio_threshold=error_principle_ratio_threshold,
        rank_counter=rank_counter+1
    )


def get_accuracy(response_matrix: np.ndarray, response_vector: np.ndarray[AffineScalarFunc]) -> int:
    return response_matrix_rank_at_given_vector(response_matrix, response_vector)
    # singular_values = np.linalg.svdvals(foil_response_matrix)
    # accuracy = np.sum(np.tanh(singular_values/singular_values[0]))