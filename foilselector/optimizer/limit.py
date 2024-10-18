"""Generates the limit of how many atoms can be present in the foil; and hence the volume etc."""
from math import inf
def max_num_atoms_from_count_rate_limit(count_rate_per_reactant, count_rate_limit):
    """Calculates the maximum number of atoms that can be present in the foil such that the count-rate limit is not exceeded."""
    if count_rate_per_reactant>0:
        return count_rate_limit/count_rate_per_reactant
    else:
        return inf