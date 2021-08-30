from numpy import array as ary
import itertools
from projection import projection_matrix
from matplotlib import pyplot as plt

def zero_allowed_partition(n, r):
    """
    All way of writing n as a sum of positive integers is called the partitions of n.
    https://math.stackexchange.com/questions/217597/number-of-ways-to-write-n-as-a-sum-of-k-nonnegative-integers
    https://en.wikipedia.org/wiki/Partition_%28number_theory%29
    However the typical definition of "Partition" doesn't allow zero to be used, which is not a positive integer.
    So our attempt to write n as a sum of r nonnegative integers is a slightly differently defined "partition",
    which I will call zero-allowed partition.
    """
    all_possible_combinations = ary(list(itertools.combinations_with_replacement(range(r+1), n)))
    matching_coordinates = all_possible_combinations[all_possible_combinations.sum(axis=1)==r]

    permuted_all_matches = []
    for match in matching_coordinates:
        for perm in itertools.permutations(match):
            if perm not in permuted_all_matches:
                permuted_all_matches.append(perm)
    return ary(permuted_all_matches)

if __name__=="__main__":
    import sys
    n, r = int(sys.argv[1]), int(sys.argv[2]) # choose r foils from n possible materials.

    allowed_coordinates = zero_allowed_partition(n, r)
    print(f"Found {len(allowed_coordinates)} matches.")

    P = projection_matrix(n, "symmetric_linear")
    print(P.shape)
    print(allowed_coordinates.shape)
    plt.scatter(*(P @ allowed_coordinates.T))
    plt.show()
    # n choose r foils from a combination of 

if False:
    import matplotlib.animation as manimation
    writer = manimation.writers['ffmpeg'](fps=15, metadata={"title":"Video_name", "comment":"#", "artist":"Ocean Wong"})

    grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
    fig, (ax, cbar_ax) = plt.subplots(1,2, gridspec_kw=grid_kws)

    with writer.saving(fig, "Video_name.mp4", 300):
        for data in data_list:
            sns.heatmap(data, ax=ax, cbar_ax=cbar_ax)
            fig.suptitle("Title")
            ax.set_xlabel("")
            ax.set_ylabel("")

            writer.grab_frame()
            ax.clear()
            cbar_ax.clear()
    fig.clf()