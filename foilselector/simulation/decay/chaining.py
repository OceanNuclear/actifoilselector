import numpy as np
from uncertainties.core import Variable
from collections import namedtuple

class IsotopeDecay(namedtuple('IsotopeDecay', ['names', 'branching_ratios', 'decay_constants', 'countable_photons'])):
    """
    Quickly bodged together class that contains 4 attributes:
        names as a list,
        branching_ratios as a list,
        decay_constants as a list,
        countable_photons of the last isotope as a uncertainties.core.AffineFunc object (or a scalar).
    Made so that the __add__ method would behave differently than a normal tuple.
    """
    def __add__(self, subordinate):
        return IsotopeDecay(self.names+subordinate.names, self.branching_ratios+subordinate.branching_ratios, self.decay_constants+subordinate.decay_constants, subordinate.countable_photons)

def build_decay_chain_tree(decay_parent, decay_dict, decay_constant_threshold=1E-23):
    """
    Build the entire decay chain for a given starting isotope.

    decay_parent : str
        names of the potentially unstable nuclide
    decay_dict : dictionary
        the entire decay_dict containing all of the that there is.
    """
    if not decay_parent in decay_dict:
        # decay_dict does not contain any decay data record about the specified isotope (decay_parent), meaning it is (possibly) stable.
        return_dict = {'name':decay_parent, 'decay_constant':Variable(0.0,0.0), 'countable_photons':Variable(0.0,0.0), 'modes':{}}
    else: # decay_dict contains the specified decay_parent
        parent = decay_dict[decay_parent]
        return_dict = {'name':decay_parent, 'decay_constant':parent['decay_constant'], 'countable_photons':parent['countable_photons']} # countable_photons per decay of this isotope
        if decay_dict[decay_parent]['decay_constant']<=decay_constant_threshold:
            # if this isotope is rather stable.
            return return_dict
        else:
            return_dict['modes'] = []
            for name, branching_ratio in decay_dict[decay_parent]['branching_ratio'].items():
                if name!=decay_parent:
                    return_dict['modes'].append( {'daughter': build_decay_chain_tree(name, decay_dict), 'branching_ratio': branching_ratio} )
    return return_dict

def linearize_decay_chain(decay_file):
    """
    Return a comprehensive list of path-to-nodes. Each path starts from the origin, and ends at the node of interest.
    Each node in the original graph must be must be recorded as the end node exactly once.
    The initial population is always assumed as 1.
    """
    self_decay = IsotopeDecay([decay_file['name']],
                            [Variable(1.0, 0.0)], # branching ratio dummy value
                            [decay_file['decay_constant']],
                            decay_file['countable_photons'])
    all_chains = [self_decay]
    if 'modes' in decay_file: # expand the decay modes if there are any.
        for mode in decay_file['modes']:
            this_branch = linearize_decay_chain(mode['daughter']) # note that this is a list, so we need to unpack it.
            for subbranch in this_branch:
                subbranch.branching_ratios[0] = mode['branching_ratio']
                all_chains.append(self_decay+subbranch)
    return all_chains # returns a list

def check_differences(decay_constants):
    difference_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)[0]
    return abs(difference_matrix)[np.triu_indices(len(decay_constants), 1)]