import numpy as np
from numpy import array as ary
from uncertainties.core import Variable

from foilselector.optimizer import accuracy, precision

def test_accuracy():
    np.random.seed(1)
    mock_reaction_xs = ary([[1, 1, 0.8, 0.7, 0, 0, 0, 0],
            [0, 1, 1, 0.9, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 2, 1, 1],
            [0, 0, 0, 0, 1, 2, 1, 1]]
    )

    # make a 10 row response matrix.
    mock_response_matrix = np.zeros([10, mock_reaction_xs.shape[1]])
    for i in range(10):
        if i<len(mock_reaction_xs):
            mock_response_matrix[i]+= mock_reaction_xs[i]
        for j in range(len(mock_reaction_xs)):
            if np.random.rand()<0.4:
                weight = np.random.randint(1, 20)
                mock_response_matrix[i] += weight * mock_reaction_xs[j]

    mock_apriori = np.random.rand(mock_reaction_xs.shape[1])
    mock_response = mock_response_matrix @ mock_apriori
    mock_response = ary([Variable(resp, 0.01) for resp in mock_response])
    return accuracy.reduce_matrix_vector(mock_response_matrix, mock_response)

def test_num_bases_contained():
    a = ary([1, 2, 3, 0], dtype=float)
    b = ary([1, 2, 3, 0], dtype=float)
    assert accuracy.get_max_num_basis(a, b)==1.0
    a = ary([1, 2, 5, 0], dtype=float)
    b = ary([1, 2, 3, 0], dtype=float)
    assert accuracy.get_max_num_basis(a, b)==1.0
    assert np.isclose(accuracy.get_max_num_basis(b, a), 0.6)
    a = ary([1, 2, 3, 0], dtype=float)
    b = ary([0, 1, 2, 5], dtype=float)
    assert accuracy.get_max_num_basis(a, b)==0.0
    a = ary([1, 2, 0, 1E-90])
    b = ary([1, 2, 5, 1E-81])
    assert accuracy.get_max_num_basis(a, b)==0.0
    a = ary([1, 2, 5, 1E-5])
    b = ary([1, 2, 5, 1.0])
    assert np.isclose(accuracy.get_max_num_basis(a, b), 1E-5, atol=0.0)
    a, b = np.random.rand(5), np.random.rand(5)
    assert accuracy.get_max_num_basis(a, b)>=0.0

# from foilselector.optimizer.accuracy import get_accuracy
# from foilselector.foldermanagement import get_apriori
# from pathlib import Path
# cwd = Path.cwd()
# apriori_flux, apriori_fluence = get_apriori(cwd, 28512000)
# def get_ratioed(name):
#     R = effective_foil_matrices[name]
#     P = effective_foil_peaks[name]
#     return [peak.intensity.n for peak in P] / (R@apriori_fluence)

# for foil_name in effective_foil_peaks.keys():
#     source_set, accuracy = get_theoretical_max_lines(foil_name)
#     print(foil_name, accuracy - len(source_set))
# PV, response_matrix = effective_foil_peaks["Ga"], effective_foil_matrices["Ga"]
# # mock_resp = response_matrix @ apriori_fluence
# import numpy as np
# from numpy import array as ary
# from uncertainties.core import Variable
# response_vector = [peak.intensity for peak in PV]
# R = response_matrix.T[response_matrix.sum(axis=0)>0].T
# # mock_response = [Variable(peak_counts, np.sqrt(peak_counts)) for peak_counts in mock_resp]
# # with np.printoptions(linewidth=120, precision=4):
# #     print(ary([response_vector, mock_response]).T)
# # error_principle_ratio_threshold = 0.2

def test_precision_weight_vector():
    gs_array = ary([[1,2,3],[2,3,4]]).T
    np.testing.assert_allclose(precision.get_precision_weight_vector(gs_array), [1,1,1])
    log_gs_array = np.log(gs_array)
    log_diff = np.diff(log_gs_array).flatten()
    np.testing.assert_allclose(
        precision.get_precision_weight_vector(
            gs_array, log_flux=True, const_lethargy=True
        ),
        log_diff**3,
    )
    gs_array = ary([[1, 10, 100], [10, 100, 1000]]).T

    log_diff = np.diff(np.log(gs_array)).flatten()
    diff = np.diff(gs_array).flatten()
    np.testing.assert_allclose(
        precision.get_precision_weight_vector(
            gs_array, log_flux=False, const_lethargy=True
        ),
        log_diff**2 * diff,
    )
    np.testing.assert_allclose(
        precision.get_precision_weight_vector(
            gs_array, log_flux=True, const_lethargy=False
        ),
        log_diff * diff**2,
    )

