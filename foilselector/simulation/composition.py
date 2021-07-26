"""
(A very short) module to read fispact in/outputs about compositions of the foils
"""
def reactant_to_FUEL(reactant_dict):
    """
    Turns a reactant_dict consisting of {isotope:number} key:value pairs
    into a FUEL block in fispact input files.
    """
    reactant_descriptions = "\n".join("{isotope} {number:.9e}".format(isotope=key, number=val)
                                            for key, val in reactant_dict.items())
    reactant_descriptions =  f"FUEL {len(reactant_dict)}\n" + reactant_descriptions
    return reactant_descriptions