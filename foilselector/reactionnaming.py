def unpack_reactions(parent_product_mt):
    if parent_product_mt.startswith("("):
        parent_list, product, mts = parent_product_mt.split("-")
        mt_list = mts[len("-MT=("):-1]
        broken_down_reactions = []
        for parent, mt_values in zip(parent_list.strip("()").split(","), mt_list.split("),(")):
            broken_down_reactions.append("{}-{}-MT=({})".format(parent, product, mt_values.strip("()")))
        return tuple(broken_down_reactions)
    else:
        return (parent_product_mt,)