import itertools
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

    P = projection_matrix(n, method="asymmetric_linear")
    print(P.shape)
    print(allowed_coordinates.shape)
    plt.plot(*(P @ allowed_coordinates.T))
    plt.show()
    # n choose r foils from a combination of
