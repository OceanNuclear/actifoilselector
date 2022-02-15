"""
Visualize the irradiation phase and decay phase using networkx.
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from foilselector.simulation.decay import build_decay_chain_tree
from uncertainties import nominal_value
from .positioning import vogels_model, Disk
from pprint import pprint 

# the three types of identities that each isotopes can be, for visualization purposes:
reactant, direct_product, indirect_product = 'reactant', 'direct product', 'indirect product'
stable_product = 'stable product'
identity = 'identity' # name of the attribute which contains the identity of the node.
EXCLUSION_MARGIN = 0.1 # How much bigger the exclusion zone should be than the disk encircling the central nodes (reactants)
# newly added functionality for plotting the decay graphs rather than pprinting.

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
    identities = {}
    for node in dg.nodes:
        identities[node] = []
        for iden, iden_set in zip((reactant, direct_product, indirect_product),
                    (reactant_set, direct_product_set, indirect_product_set)):
            if node in iden_set:
                identities[node].append(iden)
                # this identity attribute will be used to decide its color later on.
    nx.set_node_attributes(dg, identities, identity)
    nx.set_node_attributes(dg, half_life, 'half_life')

    other_isotopes_set = indirect_product_set.union(direct_product_set)-reactant_set
    reactant_pos, other_pos = position_initializer(reactant_set, other_isotopes_set)
    pos = nx.spring_layout(dg, pos={**reactant_pos, **other_pos}, fixed=list(reactant_pos.keys()), weight=None, iterations=200) # (Fruchterman-Reingold force-directed algorithm)
    pos = shell_layout(dg, pos)
    nx.set_node_attributes(dg, pos, name="pos")
    return dg

def shell_layout(graph, init_pos, finer_division=20):
    """
    graph: the graph to create a layout for
    init_pos : a dictionary containing the {names of node: 2D positions of nodes}.
    finer_division : How many segments to cut every shell into.
        The number of nodes in 1 segment = 1
        e.g. if the 3rd ring is found to have at most 8 nodes in one segment,
    Planning thought:
    There are five types of nodes that we want to lay out nicely:
    # if they are reactants, leave at the center where they already are. (no scaling should be needed, in theory)
    # if they are Judas nodes, they will stay in the closest ring.
    # if they are direct products only, they will be in the 3rd ring.
    # if they are indirect AND direct products, they will be on the second-furthest ring.
    # if they are stable and are only indirect products, they will be on the furthest ring.
    Judas nodes refers to nodes that stay too damn close to the reactants,
        like how Judas is painted to be uncomfortably close to Jesus in all of the picture of his betrayal.
    """
    from numpy import array as ary; tau = 2*np.pi
    # isotope_type = {}
    isotope_type = {iso_type:[] for iso_type in [reactant, "Judas", direct_product, direct_product+indirect_product, indirect_product]}
    # the dictionary is by default ordered. No need to use odict.

    # iterate through once to filter out and find reactant nodes' distances.
    reactant_pos = {node:init_pos[node] for node, data in graph.nodes(data=True) if reactant in data[identity]}
    max_reactant_distance = max(abs(complex(*pos)) for pos in reactant_pos.values())

    # sort them into different shells
    for node, data in graph.nodes(data=True):
        if reactant in data[identity]:
            isotope_type[reactant].append(node)
        else:
            # if abs(complex(*init_pos[node])) <= (1+EXCLUSION_MARGIN)*max_reactant_distance:
            if abs(complex(*init_pos[node])) <= 1:
                # if it lies within the exclusion zone, then it's considered a "Judas" node.
                isotope_type["Judas"].append(node)
            elif (direct_product in data[identity]) and (indirect_product not in data[identity]):
                isotope_type[direct_product].append(node)
            elif (direct_product in data[identity]) and (indirect_product in data[identity]):
                isotope_type[direct_product+indirect_product].append(node)
            elif (direct_product not in data[identity]) and (indirect_product in data[identity]):
                isotope_type[indirect_product].append(node)
            else:
                raise ValueError(f"Missed this case which has node.{identity}=\n{data[identity]}\n(Poor programming on my part!)")

    segmented_shells = [[{} for j in range(finer_division)] for shell_key in isotope_type.keys()]
    # ignore the reactant shell, which we've already decided the coordinates for.
    for i, (shell_key, isotopes) in enumerate(isotope_type.items()):
        # if shell_key==reactant:
        #     continue
        #     # gain a tiny little computational speed-up by skipping the part that we know we don't need: reactants which forms the 0th shell.
        init_angles = {}
        for iso in isotopes:
            phase = np.angle(complex(*init_pos[iso]))
            phase = phase+(tau) if phase<0 else phase # turn the negative angles into positive.
            init_angles[iso] = phase
        init_angles = dict(sorted(init_angles.items(), key=lambda x:x[1])) # sort it so it is ordered in anti-clockwise.

        nodes, angle_values = ary(list(init_angles.items()), dtype='O').reshape([-1 ,2]).T
        # This step relies strongly on the fact that the dictionary `init_angles` remains sorted even when called as .items().
        for j in range(finer_division):
            in_this_segment = np.floor_divide(angle_values, tau/finer_division)==j
            segmented_shells[i][j] = dict(zip(nodes[in_this_segment], angle_values[in_this_segment])) 

    max_num_node_per_any_segment = max(max(len(j) for j  in i) for i in segmented_shells)
    num_bins_per_ring = max_num_node_per_any_segment * finer_division
    bin_width = tau / num_bins_per_ring
    bin_comb = np.arange(max_num_node_per_any_segment)*bin_width

    def sort_to_nearestbin(dict_of_positions, bin_centroids):
        """
        Motivation/ description
        -----------------------
        Given a list of m objects and n bins,
            we want to put each object into a nearest bin
            without overfilling any bin with more than 1 object.

        Parameters
        ----------
        dict_of_positions : dictionary {node name: node position (scalar)}
        bin_centroids : a list of scalars, each one denoting the centroid of each bin.

        Returns
        -------
        The occupancy vector (vector of booleans) for for the bins.
        """
        scalars, bins = ary(list(dict_of_positions.values())), ary(bin_centroids)
        unassigned, empty = np.ones(scalars.shape, dtype=bool), np.ones(bins.shape, dtype=bool)
        def reconstruct_index(index):
            reind0 = np.arange(len(scalars), dtype=int)[unassigned][index[0]]
            reind1 = np.arange(len(bins), dtype=int)[empty][index[1]]
            return reind0, reind1
        while any(unassigned):
            distance_to_bins = abs(np.subtract.outer(scalars[unassigned], bin_centroids[empty]))
            nearest_match_ind = np.unravel_index(distance_to_bins.argmin(), distance_to_bins.shape)
            reconstructed_indices = reconstruct_index(nearest_match_ind)
            unassigned[reconstructed_indices[0]] = False
            empty[reconstructed_indices[1]] = False
        return ~empty

    final_shells_order = []
    filler_counter = 0
    for this_ring in segmented_shells:
        shells = []
        for j in range(finer_division):
            angle_offset = j*bin_width*max_num_node_per_any_segment 
            bin_centroids = bin_comb + angle_offset + bin_width/2
            occupancy_vector = sort_to_nearestbin(this_ring[j], bin_centroids)
            # and then fill in the shell
            node_names_in_segment = list(this_ring[j].keys())
            for this_angle, occupied in zip(bin_comb + angle_offset, occupancy_vector):
                if occupied:
                    shells.append(node_names_in_segment.pop())
                else:
                    shells.append("filler"+str(filler_counter))
                    filler_counter += 1 # prevent name collision and merging of filler points.
        final_shells_order.append(shells)
    filled_shell_pos = nx.shell_layout(graph, nlist=final_shells_order[1:], rotate=0)
    shell_pos = {k: v for k, v in filled_shell_pos.items() if not k.startswith("filler")}
    return {**reactant_pos, **shell_pos}

def position_initializer(reactants, other_isotopes, seed=None):
    """
    reactants: iterable of the reactant's names
    other_isotopes: iterable of the reactant's names
    Generates the initial positions of all isotopes.
    reactants are treated differently than other isotopes:
        reactants' positions: initilized as a fermat spiral using vogel's model.
            This circle will be scaled down so that it matches the point-density everywhere else.
        other isotopes positions: initilized as a uniform distribution of points,
            placed around [-1, 1], [-1, 1].
    """
    fermat_spiral = vogels_model(len(reactants))
    fdisk = Disk(fermat_spiral[-3:]) # define the boudning circle, using only the last 3 points of the fermat spiral.
    fermat_spiral_area = np.pi*(fdisk.radius**2)
    if fermat_spiral_area==0:
        fermat_spiral_area = np.pi # if there is only 1 point, then we force the area to be non-zero.

    # shift and shrink down the spiral so that it matches the point density.
    node_density = (len(reactants)+len(other_isotopes))/4 # the [-1, 1], [-1, 1] square has area =4.
    scale_factor = (len(reactants)/node_density/fermat_spiral_area)**0.5 # need to scale down by this much.
    centered_spiral = fermat_spiral - fdisk.center
    transformed_spiral = centered_spiral * scale_factor

    reactant_pos = dict(zip(reactants, transformed_spiral))

    # cover [-1, 1], [-1, 1] with a random distribution of points, but with an exclusion zone.
    np.random.seed(seed)
    # exclusion zone is a circle slightly bigger than the space that the spiral occupies
    exclusion_disk = Disk.from_parameters([0.0, 0.0], fdisk.radius*scale_factor*1+EXCLUSION_MARGIN) # hard-coded value of 15% more. Can change later.
    other_isotope_list = list(other_isotopes)
    other_pos = {}
    while other_isotope_list:
        new_pos = np.random.uniform(-1, 1, size=2)
        if new_pos not in exclusion_disk:
            other_pos[other_isotope_list.pop()] = new_pos
    return reactant_pos, other_pos

def default_cmap(attr_dict):
    """
    Given, an attribute dictionary of the node,
    determine the colour to be used.
    color can be 
    """
    identities = attr_dict[identity]
    if reactant in identities:
        return "C0"
    elif direct_product in identities:
        return "C1"
    elif indirect_product in identities:
        return "C2"

def draw_networkx_weighted_directed_graph(graph, ax, minarrow=0.1, maxarrow=1, cmap=default_cmap, write_half_life=False, edge_labels=False):
    """
    Draws a networkx weighted directed graph.
        The positions are ALREADY fixed when the nodes are generated in as_graph and saved as the .pos attribute.
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
    if write_half_life:
        half_life = nx.get_node_attributes(graph, 'half_life')
        half_life_str = { iso:"\n"+str(hl) for iso, hl in half_life.items()}
        label_text = { iso:iso+half_life_str.get(iso, "") for iso in graph.nodes }
        nx.draw_networkx_labels(graph, pos, labels=label_text, ax=ax)
    else:
        nx.draw_networkx_labels(graph, pos, ax=ax)

    # draw edges, width determined by normalized weight
    if len(graph.edges)>0:
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

def irradiationView(graph):
    """Return a view of all isotopes involved in the irradiation phase of the experiment."""
    subgraph_list = [node for node, dat in graph.nodes(data=True) if (reactant in dat[identity]) or (direct_product in dat[identity]) ]
    subgraph = graph.subgraph(subgraph_list)
    ignore_edges = []
    for innode, outnode in subgraph.edges:
        # include only edges that originates from the reactants
        if reactant not in subgraph.nodes[innode][identity]:
            ignore_edges.append((innode, outnode)) # exclude all other edges them.
    filtered_graph = nx.restricted_view(subgraph, [], ignore_edges)
    return filtered_graph

def decayView(graph, exclude_further_decay=False):
    """
    Return a view of all isotopes involved in the decay phase of the experiment.
    exclude_further_decay : ignore indirect decays.
    """
    subgraph_list = [node for node, dat in graph.nodes(data=True) if (direct_product in dat[identity]) or (indirect_product in dat[identity]) ]
    subgraph = graph.subgraph(subgraph_list)
    ignore_edges = []
    for innode, outnode in subgraph.edges:
        if exclude_further_decay:
            # include only edges that originates from direct decay products
            if direct_product not in subgraph.nodes[innode][identity]:
                ignore_edges.append((innode, outnode)) # exclude all other edges them.
        else:
            if reactant in subgraph.nodes[innode][identity]:
                ignore_edges.append((innode, outnode))
    filtered_graph = nx.restricted_view(subgraph, [], ignore_edges)
    return filtered_graph

def edgelessView(graph):
    """Return a view of the graph that has all the edges removed."""
    restricted_graph = nx.restricted_view(graph, [], graph.edges)
    return restricted_graph

def reactantlessView(graph):
    """Return a view of the graph that excludes the reactants."""
    reactant_list = []
    for node, data in graph.nodes(data=True):
        if reactant in data[identity]:
            reactant_list.append(node)
    restricted_graph = nx.restricted_view(graph, reactant_list, [])
    return restricted_graph

def isolateRemovedView(graph):
    isolates = nx.isolates(graph)
    return nx.restricted_view(graph, isolates, [])