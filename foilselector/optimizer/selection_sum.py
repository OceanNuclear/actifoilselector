import numpy as np
from numpy.linalg import norm
from numpy import sqrt, array as ary
from matplotlib import pyplot as plt

def find_new_vec(ref_vector, *perp_vectors):
    """
    Find a vector that is perpendicular to all of the perp_vectors,
    and is a projection of the ref_vector on the remaining hyperplane.
    """
    coef_builder = [ref_vector, *perp_vectors] # = basis to express our desired unit_vector in.
    first_n_minus_1_lines = [[basis_vec.dot(vec) for basis_vec in coef_builder] for vec in perp_vectors]
    # the last line cannot be expressed in linear simultaneous equation :(
    # unit_vector.dot(unit_vector) = 1, which leads to a lot of 2nd degree terms of coefficients
    # need sympy to solve :(
    return NotImplementedError("Unfortunately we need sympy to solve it, so this visualization will be a project that takes more time than I have.")
if __name__=="__main__":
    N = 5 # number_of_candidate_foils

    prev_coordinates = list(np.diag(np.ones(num_dim)))
    new_coordinates = [sqrt(1/num_dim)*np.ones(num_dim), ]
    while len(new_coordinates) < num_dim:
        reference_vec = prev_coordinates.pop() # must be the projection of this vector on the remaining hyperplane space.
        find_new_vec(reference_vec, *new_coordinates)
        new_coordinates.append()

if False:
    import matplotlib.animation as manimation
    import seaborn as sns
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