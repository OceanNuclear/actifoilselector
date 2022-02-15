import itertools
from tqdm import tqdm
import numpy as np
from numpy import array as ary
from matplotlib import pyplot as plt
from scipy.special import binom as binomial_coefficient
from projection import projection_matrix

def stars_and_bars(num_objects_remaining, num_bins_remaining, filled_bins=()):
    """
    Stars and Bars (https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)) problem can be thought of as
    1. Putting m-1 bars ("|") in amongst a row of n stars
    OR
    2. Putting n balls into m bins.

    The code below is partially modified from
    @MISC {4096469,
    TITLE = {Number of ways to write n as a sum of k nonnegative integers},
    AUTHOR = {Ben Paul Thurston (https://math.stackexchange.com/users/288232/ben-paul-thurston)},
    HOWPUBLISHED = {Mathematics Stack Exchange},
    NOTE = {URL:https://math.stackexchange.com/q/4096469 (version: 2021-04-11)},
    EPRINT = {https://math.stackexchange.com/q/4096469},
    URL = {https://math.stackexchange.com/q/4096469}
    }
    """
    if num_bins_remaining > 1:
        # case 1: more than one bins left

        for num_distributed_to_the_next_bin in range(0, num_objects_remaining + 1):
            # try putting in anything between 0 to num_objects_remaining of objects into the next bin.
            yield from stars_and_bars(
                    num_objects_remaining - num_distributed_to_the_next_bin,
                    num_bins_remaining - 1,
                    filled_bins=filled_bins + (num_distributed_to_the_next_bin,),
                )
    else:
        # case 2: reached last bin. Termintae recursion.
        # return a single tuple enclosed in a list.
        yield (
            filled_bins + (num_objects_remaining,)
        )

def analytical_number_of_ways(n, r):
    """
    Analytically known equation from the Stars and Bars problem.
    https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)
    https://math.stackexchange.com/questions/217597/number-of-ways-to-write-n-as-a-sum-of-k-nonnegative-integers
    """
    return binomial_coefficient(n + r - 1, r - 1)

def tri_upper_matrix(dimension):
    """
    Create a matrix that has an
    upper-half (including the diagonal)==True but
    lower-half (excluding diagonal)==False.
    """
    index_matrix = np.arange(dimension).repeat(dimension).reshape([dimension, dimension])
    # index_matrix is a matrix where each element = its row index.
    # index_matrix.T is a matrix where each element = its column index.
    return index_matrix<=index_matrix.T

def zero_allowed_partition(n, r):
    """
    All way of writing n as a sum of positive integers is called the partitions of n.
    https://math.stackexchange.com/questions/217597/number-of-ways-to-write-n-as-a-sum-of-k-nonnegative-integers
    https://en.wikipedia.org/wiki/Partition_%28number_theory%29
    However the typical definition of "Partition" doesn't allow zero to be used, which is not a positive integer.
    So our attempt to write n as a sum of r nonnegative integers is a slightly differently defined "partition",
    which I will call zero-allowed partition.
    """
    all_possible_combinations = ary(
        list(itertools.combinations_with_replacement(range(r + 1), n))
    )
    matching_coordinates = all_possible_combinations[
        all_possible_combinations.sum(axis=1) == r
    ]

    permuted_all_matches = set()
    for match in matching_coordinates:
        for perm in itertools.permutations(match):
            permuted_all_matches.add(perm)
    return ary(list(permuted_all_matches))


if __name__ == "__main__":
    import sys

    n, r = int(sys.argv[1]), int(
        sys.argv[2]
    )  # choose r foils from n possible materials.

    allowed_coordinates = ary(list(stars_and_bars(r, n)))
    print(f"Found {len(allowed_coordinates)} matches.")

    # for dim, coord in enumerate(allowed_coordinates.T):
    initial_points = np.tile(allowed_coordinates, len(allowed_coordinates)).reshape([len(allowed_coordinates), *allowed_coordinates.shape])
    # same point's coordinates is repeated along its respective row

    print("Calculating the number of duplicate lines...")
    displacements = (initial_points - initial_points.transpose([1,0,2]))
    L1_distances = abs(displacements).sum(axis=2)
    # calculate the boolean array denoting who are neighbours of each other.
    neighbours = L1_distances==2
    lines_drawn = np.logical_and(tri_upper_matrix(len(allowed_coordinates)), neighbours)

    start_points, end_points = initial_points[lines_drawn], initial_points.transpose([1,0,2])[lines_drawn]

    P = projection_matrix(n, method="asymmetric_linear")
    print("Projection matrix has shape=", P.shape)
    print("Data points has shape=", allowed_coordinates.shape)

    (start_x, start_y), (end_x, end_y) = P@start_points.T, P@end_points.T
    node_x, node_y = P @ allowed_coordinates.T

    ALLOW_PLOTLY = False
    USE_PLOTLY = True if len(neighbours)<500 else False # few enough number of connections
    print("Calling the plotting functions")
    if PLOTLY:
        from plotly import express as px, graph_objects as go
        scatter_fig = px.scatter(x=node_x, y=node_y, hover_name=[",".join(coord) for coord in allowed_coordinates.astype(str)])

        # import pandas as pd # unused because we're adding trace instead
        # pd.DataFrame(...)
        # line_fig = px.line(df, x=..., y=...)

        for x0, y0, x1, y1 in tqdm(zip(start_x, start_y, end_x, end_y), total=len(start_x)):
            scatter_fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines'))
            # a better method than a for-loop is if we can plot connectivity without using keys - Otherwise the load time would scale linearly with the number of connections.
            # Or we can plot all of the connectivity using A SINGLE line.
        scatter_fig.show()

    else:
        plt.scatter(node_x, node_y)
        plt.plot(ary([start_x, end_x]), ary([start_y, end_y]))
        plt.show()
        # Try using plotly instead to allow labelling?