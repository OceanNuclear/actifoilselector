import numpy as np
from numpy import array as ary
from collections import OrderedDict, namedtuple
from operator import itemgetter
from math import factorial as fac
from matplotlib import pyplot as plt

class CombinationNode(object):
    __slots__ = ["plot_coordinates", "_occupancy", "_numbered_occupancy", "children_instantiated"]
    def __init__(self, occupancy, plot_coordinates):
        """
        must initialize with a plot_coordinates (int, int)
        """
        self.occupancy = occupancy
        self.plot_coordinates = plot_coordinates
        self.children_instantiated = False

    def __repr__(self):
        return "<"+"".join("1" if tf else "0" for tf in self.occupancy)+">"

    @property
    def right_mobility_array(self):
        """ Return array stating which bits can move right"""
        return np.hstack([np.diff(self.occupancy.astype(int))==-1, False])

    @property
    def left_mobility_array(self):
        """Return array stating which bits can move left"""
        return np.hstack([False, np.diff(self.occupancy.astype(int))==1])

    @property
    def right_movable_numbers(self):
        return self.numbered_occupancy[self.right_mobility_array]

    @property
    def left_movable_numbers(self):
        return self.numbered_occupancy[self.left_mobility_array]

    @property
    def occupancy(self):
        return self._occupancy

    @occupancy.setter
    def occupancy(self, _occupancy):
        self._occupancy = np.array(_occupancy, dtype=bool)
        numbering = np.cumsum(self._occupancy)
        numbering[~self._occupancy] = int(0)
        self._numbered_occupancy = numbering

    @property
    def numbered_occupancy(self):
        return self._numbered_occupancy

    @property
    def decimal_occupancy(self):
        return int("".join("1" if tf else "0" for tf in self.occupancy), base=2)

    def get_children_coordinates(self, right=True):
        if right:
            return [(self.plot_coordinates[0]+(num - (self.occupancy.sum()+1)/2), self.plot_coordinates[1]-1) for num in self.right_movable_numbers]
        else:
            return [(self.plot_coordinates[0]-(num - (self.occupancy.sum()+1)/2), self.plot_coordinates[1]+1) for num in self.left_movable_numbers]

    def get_children(self, right=True):
        children = []
        movable_numbers = self.right_movable_numbers if right else self.left_movable_numbers
        for num_moved, child_coord in zip(movable_numbers, self.get_children_coordinates(right)):
            new_occupancy = self.occupancy.copy()
            remover = self.numbered_occupancy == num_moved
            new_occupancy[remover] = False
            adder = np.roll(remover, (-1)**(1-right)) # roll to the right if right==True, to the left if right==False.
            assert (not adder[-1+right]) and adder.sum()==1, "Expected to move only one occupancy bit, withOUT wrapping around."
            new_occupancy[adder] = True
            # this part would've been made more expandable (i.e. don't need to specify future arguments such as sorted_list) if we use the inspect module.
            # however, I suspect list(inspect.signature(self.__init__).parameters)[1:] would massively slow things down.
            # So I choose to manually specify rather than use automatic introspection, in a similar vein to how when using __slots__ I gave up expandability for performance.
            init_list = [new_occupancy, child_coord]
            if hasattr(self, "sorted_list"):
                init_list.append(self.sorted_list)
            children.append( self.__class__(*init_list) )
        self.children_instantiated = True
        return children

class CombinationSumNode(CombinationNode):
    __slots__ = ["plot_coordinates", "_occupancy", "_numbered_occupancy", "children_instantiated", "sorted_list"]
    def __init__(self, occupancy, plot_coordinates, sorted_list):
        super().__init__(occupancy, plot_coordinates)
        self.sorted_list = np.array(sorted_list)

    def sum(self):
        return self.sorted_list[self.occupancy].sum()

CombinationGraphState = namedtuple("CombinationGraphState", ["all_nodes", "sorted_sum_result", "expectant_parents"], )
def top_n_sums_generator(sorted_list, target_chosen_length : int, skip_n : int, largest=True):
    """
    Get the top n largest (or smallest) sums by choosing target_chosen_length out of the sorted_list of values.
    Parameters
    ----------
    sorted_list : a list of real numbers, sorted descendingly. The goal of this function is to return the top n combinations that will give the largest/smallest sums by picking from this list.
    target_chosen_length : number of elements to pick out of this sorted_list.
    skip_n : number of iterations to skip, to save time/reduce memory usage
    largest: boolean indicating whether we start by selecting the leftmost (biggest) target_chosen_length elements of the sorted list and then move each bit to the right;
             or the rightmost (smallest) target_chosen_length element and move each bit to the left.

    Returns:
    -------
    acts as a generator, yielding the following 3 dictionaries at each stage.
    at the n-th iteration, it returns:
    all_nodes: dictionary containing all instances of CombinationSumNode created
    sorted_sum_result: an OrderedDict with the top n nodes' values
    expectant_parents: a dict with values of all of the nodes that has been evaluated but not yet included into the sorted dict becuase their values aren't large/small enough.
    """
    all_nodes = dict() # dictionary storing every node, for the sake of plotting sake.
    sorted_sum_result = OrderedDict() # value of nodes that are already sorted
    expectant_parents = dict() # value of nodes that are instantiated but not selected into sorted_sum_result yet because their values are still too low.

    # create first node. In i-th iterations where i>1 , this would be replaced by the get_children() method.
    first_occupancy = np.zeros(len(sorted_list), dtype=bool)
    if largest:
        first_occupancy[:target_chosen_length] = True
        extrumum_func = np.argmax
    else:
        extrumum_func = np.argmin
        first_occupancy[-target_chosen_length:] = True
    children = [ CombinationSumNode(first_occupancy, (0, 0), sorted_list), ] # NULL births only 1 child: the origin (the first node).

    n_max = fac(len(sorted_list))/(fac(len(sorted_list)-target_chosen_length) * fac(target_chosen_length))
    skip_n = np.clip(skip_n, 0, n_max) # do not generate the first skip_n cases.
    while len(sorted_sum_result)<n_max:
        # evaluate the children
        for node in children:
            if tuple(node.occupancy) not in all_nodes:
                node_name = tuple(node.occupancy)
                all_nodes[node_name] = node
                expectant_parents[node_name] = node.sum()

        # sort the un-inserted nodes, and insert the best one into sorted_sum_result:
        unsorted_name, unsorted_sums = list(zip(*expectant_parents.items()))
        opt_ind = extrumum_func(unsorted_sums)
        node_name, node_value = unsorted_name[opt_ind], unsorted_sums[opt_ind]
        # remove from expectant_parents and place in sorted_sum_result
        expectant_parents.pop(node_name)
        sorted_sum_result[node_name] = node_value
        # get its children
        children = all_nodes[node_name].get_children(right=largest)

        if len(sorted_sum_result)>=skip_n:
            yield CombinationGraphState(all_nodes.copy(), sorted_sum_result.copy(), expectant_parents.copy())

def top_n_sums(sorted_list, target_chosen_length, n, largest=True):
    return next(top_n_sums_generator(sorted_list, target_chosen_length, n, largest)).sorted_sum_result

def top_n_sums_of_dict(un_sorted_dict, target_chosen_length, n, largest=True):
    descending_sorted_dict = OrderedDict(sorted(un_sorted_dict.items(), key=itemgetter(1), reverse=True))
    top_n_combos = top_n_sums(list(descending_sorted_dict.values()), target_chosen_length, n, largest)
    selectable_keys = ary(list(descending_sorted_dict.keys()))

    sort_result_as_return_dict = OrderedDict()
    for boolean_mask, sum_value in top_n_combos.items():
        sort_result_as_return_dict[tuple(selectable_keys[ary(boolean_mask)])] = sum_value

    return sort_result_as_return_dict
