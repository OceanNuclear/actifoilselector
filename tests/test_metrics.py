import numpy as np
from numpy import array as ary

from foilselector.optimizer import precision

# from numpy import array as ary
# from foilselector.foldermanagement import get_apriori
# from uncertainties.core import Variable
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
# mock_resp = response_matrix @ apriori_fluence
# import numpy as np
# response_vector = [peak.intensity for peak in PV]
# R = response_matrix.T[response_matrix.sum(axis=0)>0].T
# mock_response = [Variable(peak_counts, np.sqrt(peak_counts)) for peak_counts in mock_resp]
# with np.printoptions(linewidth=120, precision=4):
#     print(ary([response_vector, mock_response]).T)
# error_principle_ratio_threshold = 0.2


def test_precision_weight_vector():
    gs_array = ary([[1, 2, 3], [2, 3, 4]]).T
    np.testing.assert_allclose(
        precision.get_precision_weight_vector(gs_array), [1, 1, 1]
    )
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
