def unpack_reactions(parent_product_mt):
    """
    Turn a reaction from the format of
    '(Cd106, Cd104)-Nb84-MT=((5), (5))'
    to
    ('Cd106-Nb84-MT=(5)', )
    """
    if parent_product_mt.startswith("("):
        parent_list, product, mts = parent_product_mt.split("-")
        mt_list = mts[len("-MT=("):-1]
        broken_down_reactions = []
        for parent, mt_values in zip(parent_list.strip("()").split(","), mt_list.split("),(")):
            broken_down_reactions.append("{}-{}-MT=({})".format(parent, product, mt_values.strip("()")))
        return tuple(broken_down_reactions)
    else:
        return (parent_product_mt,)

def strip_mt_brackets(parent_product_mt):
    """
    turn a reaction from the format of
    'Cd106-Nb84-MT=(5)'
    to
    ('Cd106-Nb84-MT=5', )
    """
    parent, product, mts = parent_product_mt.split("-")
    broken_down_reactions = []
    mt_list = mts[len("-MT="):-1]
    for mt in mt_list.strip("()").split(","):
        broken_down_reactions.append("{}-{}-MT={}".format(parent, product, mt))
    return tuple(broken_down_reactions)