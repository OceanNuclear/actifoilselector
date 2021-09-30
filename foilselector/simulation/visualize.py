"""
Visualize the irradiation phase and decay phase using networkx.
"""
import networkx as nx
import matplotlib.pyplot as plt
from foilselector.simulation.decay import build_decay_chain_tree

# the three types of identities that each isotopes can be, for visualization purposes:
reactant, direct_product, indirect_product = "reactant", "direct product", "indirect product"
stable_product = "stable product"

# newly added functionality for plotting the decay graphs rather than pprinting.

def get_all_products(foil_dict):
    """
    TODO: only a temporary function, shouldn't permenantly rely on this.
    """
    return [i.split('-')[1] for i in foil_dict.keys()]

def leaf_collector(tree):
    """
    Collects the names of all in a decay tree
    """
    fruit_basket = set() # the container for picking things from the tree is called a fruit basket, in a tongue-in-cheek manner
    fruit_basket.add(tree['name'])
    if 'modes' in tree:
        for mode in tree['modes']: # tree['modes'] is a list
            fruit_basket = fruit_basket.union( leaf_collector(mode['daughter']) )
    return fruit_basket

def harvest_multiple_trees(primary_products, decay_dict):
    """
    primary_products: a list of isotope names, each of which is a primary product produced directly from neutron's reaction
         can be obtained by
        `primary_products = get_all_products(foil.keys())`
    decay_dict : dictionary
        Each entry in decay_dict has the name of the decaying isotope as the key,
        and a dictionary of its decay-related parameters (decay_constant, branching_ratio, countable_photons) as the values.
            branching_ratio is a dictionary with the name of the daughter as each key and the value of branching ratio as each value.
    """
    leaf_sets = [leaf_collector( build_decay_chain_tree(iso, decay_dict)) for iso in primary_products]
    isotopes_involved = set.union(*leaf_sets)
    return {iso:decay_dict[iso] for iso in isotopes_involved}

def as_graph(num_reactions, selected_decay_dict):
    """
    num_reactions : a dictionary with parent_product_mts as the keys and the number of such reaction happening as the value.
    selected_decay_dict : a dicationary of SELECTED decay info files,
        same as the selected_decay_dict argument in build_decay_chain_tree,
        except the irrelevant isotopes are filtered out (otherwise the graph plotted is going to be very big)
    """
    from uncertainties import nominal_value
    dg = nx.DiGraph()
    reactant_set, direct_product_set, indirect_product_set = set(), set(), set()
    half_life = {}
    # add the parent->direct product (neutron induced reaction) edges
    for parent_product_mts, num_reactions in num_reactions.items():
        foil_reactant, primary_product, mts = parent_product_mts.split("-")
        reactant_set.add(foil_reactant)
        direct_product_set.add(primary_product)
        dg.add_edge(foil_reactant,
                    primary_product,
                    weight=nominal_value(num_reactions),
                    str_weight=str(num_reactions),
                    pathway=mts,
                    )

    # add the direct product->indirect product (decay) edges
    for parent, decay_info in selected_decay_dict.items():
        try:
            half_life[parent] = 0.6931471805599453/decay_info['decay_constant'] # log 2 divided by that
        except ZeroDivisionError:
            import math
            half_life[parent] = math.inf
        for product, branching_ratio in decay_info['branching_ratio'].items():
            indirect_product_set.add(product)
            dg.add_edge(parent, product,
                weight=nominal_value(branching_ratio),
                str_weight=str(branching_ratio),
                )

    # Label the node by it's identity(ies).
    identity = {}
    for node in dg.nodes:
        identity[node] = []
        for iden, iden_set in zip((reactant, direct_product, indirect_product),
                    (reactant_set, direct_product_set, indirect_product_set)):
            if node in iden_set:
                identity[node].append(iden)
                # this identity attribute will be used to decide its color later on.
    nx.set_node_attributes(dg, identity, 'identity')
    nx.set_node_attributes(dg, half_life, 'half_life')

    """
    Objective:
        1. Keep the reactants central,
        2. The primary products should be arranged in a ring around the centre.
        3. Minimize cross-over of edges

    Try the methods listed first. Worry about seeding later.

    OR simplest method: 
        0. Use planar algorithm on products only?
        1. spring algorithm to form product ring naturally.
        2. Simply force the reactants into the centre, without further spring algorithm iteration.
            BAD!
        3. Calculate to minimize overlap?
            HARD!
        Do some sort of check to minimize the possibility of overlap.
    """
    # (Fruchterman-Reingold force-directed algorithm)
    # exclude_reactant = not_reactants(dg)
    # # hole_in_centre_layout = nx.drawing.layout.shell_layout(exclude_reactant)
    # hole_in_centre_layout = nx.drawing.layout.spring_layout(exclude_reactant)
    # pos = nx.drawing.layout.spring_layout(dg, pos=hole_in_centre_layout, iterations=50)

    # Other alternatives planned for testing are:
    pos = nx.drawing.layout.spring_layout(dg)
    # pos = nx.drawing.layout.spectral_layout(dg)
    # pos = nx.drawing.layout.kamada_kawai_layout(dg)

    nx.set_node_attributes(dg, pos, name="pos")
    return dg

def default_cmap(attr_dict):
    """
    Given, an attribute dictionary of the node,
    determine the colour to be used.
    color can be 
    """
    identities = attr_dict['identity']
    if reactant in identities:
        return "C0"
    elif direct_product in identities:
        return "C1"
    elif indirect_product in identities:
        return "C2"

def draw_networkx_weighted_directed_graph(graph, ax, minarrow=0.1, maxarrow=1, cmap=default_cmap, edge_labels=False):
    """
    Draws a networkx weighted directed graph.
    graph: the graph to draw
    ax: a matplotlib.pyplot.Axes instance to draw on.
    minarrow: minimum arrow size, to be parsed by as the mutation_scale argument.
    cmap : a color mapping function,
        whose input is the node's attribute dictionary
        and the returned value is a str, or hex values. 
    """
    pos = nx.get_node_attributes(graph, 'pos')
    assert len(pos)==len(graph.nodes), f"All nodes must have a 'pos' attribute as given by one of the nx.drawing.layout functions!"
    weights = nx.get_edge_attributes(graph, 'weight')
    assert len(weights)==len(graph.edges), "All edeges must have a 'weight' attribute, to represent either the branching ratio (decay) or the number of reactions (neutron-induced reaction)."

    # draw the nodes, and label them.
    nx.draw_networkx_nodes(graph, pos, node_color=[cmap(attr) for n, attr in graph.nodes(data=True)], ax=ax)
    half_life = nx.get_node_attributes(graph, 'half_life')
    half_life_str = { iso:"\n"+str(hl) for iso, hl in half_life.items()}
    label_text = { iso:iso+half_life_str.get(iso, "") for iso in graph.nodes }
    nx.draw_networkx_labels(graph, pos, labels=label_text, ax=ax)

    # draw edges, width determined by normalized weight
    minw, maxw = min(weights.values()) , max(weights.values())
    scale_factor = (maxarrow - minarrow)/(maxw - minw)
    normalize_weights_into_widths = lambda w: minarrow + (w-minw)*scale_factor
    unique_weights = set(data['weight'] for n1, n2, data in graph.edges(data=True))
    for weight in unique_weights:
        #4 d. Form a filtered list with just the weight you want to draw
        # (shamelessly stolen from https://qxf2.com/blog/drawing-weighted-graphs-with-networkx/)
        weighted_edges = [ (n1, n2) for (n1, n2, edge_attr) in graph.edges(data=True) if edge_attr['weight']==weight ]
        width = normalize_weights_into_widths(weight)
        nx.draw_networkx_edges(graph, pos, edgelist=weighted_edges, width=width)

    # draw edge labels
    if edge_labels:
        if isinstance(edge_labels, bool):
            edge_labels = nx.get_edge_attributes(graph, 'str_weight')
        elif isinstance(edge_labels, dict):
            pass
        else:
            raise ValueError("Unrecognized edge_labels! Can only use True (in which case node.str_weight will be used), False/None, or a dictionary to represent what labels to use.")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)
    return ax

def irradiation_subgraph(graph):
    """ Return the part of the graph that represent neutron-induced reactions."""
    subgraph_list = [node for node, dat in graph.nodes(data=True) if (reactant in dat['identity']) or (direct_product in dat['identity']) ]
    return graph.subgraph(subgraph_list)

def decay_subgraph(graph):
    """ Return the part of the graph that represent natural decay of isotopes after their production."""
    subgraph_list = [node for node, dat in graph.nodes(data=True) if (direct_product in dat['identity']) or (indirect_product in dat['identity']) ]
    return graph.subgraph(subgraph_list)

def not_reactants(graph):
    subgraph_list = [node for node, dat in graph.nodes(data=True) if reactant not in dat['identity']]
    return graph.subgraph(subgraph_list)