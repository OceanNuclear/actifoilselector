"""
Module to turn build the decay graph (a directed-acyclic-graph (DAG)),
turn it into a tree,
and then linearize the tree
    so that each way of reaching each node in the tree is accounted for using a subchain.
"""
import pprint
from collections import namedtuple

import numpy as np
from uncertainties.core import Variable

class IsotopeDecay(namedtuple('IsotopeDecayTuple', ['names',
                                                    'branching_ratios',
                                                    'decay_constants',
                                                    'countable_photons'])
                                ):
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

class DecayInfo(dict):
    def __init__(self):
        raise NotImplementedError("To be implemented at the revamp stage - when I invert the foilselector to do material selection as the first stage.")

def build_decay_chain_tree(decay_parent, decay_dict, decay_constant_threshold=1E-23):
    """
    Build the entire decay chain (recursively) *AS A TREE* for a given starting isotope to decay from.

    This turns the decay chain (a directed-acyclic graph) into a tree.
    In order to do so sometimes we may end up with the same isotope on several different branches,
    but that's a mathematically expected behaviour,
        because if you add up the branching ratio leading down to each of these isotopes you'll get back to unity.
    We must sum across all of these instances of the identical isotopes to accurately represent
        the total number of isotopes at any point in time/ total number of decay experienced across any two points in time.

    decay_parent : str
        names of the potentially unstable nuclide
    decay_dict : dictionary
        the entire decay_dict containing all of material that forms the radiodict that there is.
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
                subbranch.branching_ratios[0] = mode['branching_ratio'] # only a 'mode["branching_ratio"]' number of them goes into this subbranch.
                all_chains.append(self_decay+subbranch)
    return all_chains # returns a list

def check_differences(decay_constants):
    difference_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)[0]
    return abs(difference_matrix)[np.triu_indices(len(decay_constants), 1)]

def trim_tree(tree):
    """
    Given a tree built from build_decay_chain_tree,
    such that when pretty-printed by pprint,
    it is easy to read and retains only the key information for each daughter nuclide (name, branching ratio)
    """
    results = {"name" : tree["name"]}
    decay_const = tree["decay_constant"]
    if decay_const>1E-23:
        results.update({"t1/2" : ln2/decay_const})
    else:
        results.update({"t1/2" : np.inf})
    if "modes" in tree.keys():
        results.update({" L=>" : [(mode["branching_ratio"], trim_tree(mode["daughter"])) for mode in tree["modes"]]})
    return results

def pprint_tree(decay_tree):
    """
    Print the decay tree in a pretty/readable format
    """
    return pprint.pprint(trim_tree(decay_tree), sort_dicts=False)