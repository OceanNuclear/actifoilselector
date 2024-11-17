from foilselector.optimizer import comb_sum
from foilselector.optimizer import selection_sum
from foilselector.optimizer import choose_mass
from numpy import log as ln

def get_precision_weight_vector(gs_array: np.ndarray, *, log_flux: bool=False, const_lethargy: bool=False):
    """Get the weight vector w for calculating precision. """
    diff = np.diff(gs_ary, axis=1).flatten()
    log_diff = np.diff(ln(gs_ary), axis=1).flatten()
    if const_lethargy:
        if log_flux:
            return log_diff**3
        return log_diff*log_diff*diff
    elif log_flux:
        return log_diff*diff*diff
    return diff**3