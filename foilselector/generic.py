def ordered_set(sequence):
    """
    Get the sorted set, sorted according to the order of element first appearing in the sequence.
    source:
    http://www.martinbroadhurst.com/removing-duplicates-from-a-list-while-preserving-order-in-python.html
    date accessed website: 2021-01-19 11:44:23
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
    # GENIUS!
    # (x in seen) -> stop evaluating;
    # (x not in seen) -> add x to seen -> bracket returns False
    #   -> negated by "not" in front of bracket -> adds element to list
    # This should be an O(n) operation.

def minmax(array):
    """
    Alias function to quickly return the minimum and maximum among all values in an array. 
    parameters
    ----------
    array : any shaped array

    returns
    -------
    tuple containing a min (scalar) and a max (scalar)
    """
    return min(array), max(array)