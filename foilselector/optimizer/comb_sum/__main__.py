def plot_nodes(all_nodes, sorted_sum_result, expectant_parents, largest=True, ax=None, plot_dangling_links=True, offset=(0, 0), transform_matrix=np.identity(2), text=True):
    """
    Parameters
    ----------
    all_nodes: dictionary of all of the instances of CombinationSumNode created for a given sorted list.
    all_nodes = 1
    """
    sample_node = list(all_nodes.values())[0]
    sorted_list, target_chosen_length = sample_node.sorted_list, sample_node.occupancy.sum()
    biggest_sum, smallest_sum = sorted_list[:target_chosen_length].sum(), sorted_list[-target_chosen_length:].sum()
    def normalize_size(x, output_min=5, output_max=50):
        m = (output_max - output_min)/(biggest_sum - smallest_sum)
        return m * (x - smallest_sum) + output_min

    birthed, unbirthed = [], []
    birthed_sum, unbirthed_sum = [], []
    birthed_children_links, unbirthed_children_links = [], []
    birthed_names, unbirthed_names = [], []

    for node_name, node in all_nodes.items():
        node_coordinates = node.plot_coordinates
        # if node_name in sorted_sum_result:
        if node.children_instantiated:
            birthed.append(node_coordinates)
            birthed_sum.append(sorted_sum_result[node_name])
            birthed_names.append(repr(node))
            for child_coord in node.get_children_coordinates(right=largest):
                birthed_children_links.append(node_coordinates)
                birthed_children_links.append(child_coord)
        # elif node_name in expectant_parents:
        else:
            unbirthed.append(node_coordinates)
            unbirthed_sum.append(expectant_parents[node_name])
            unbirthed_names.append(repr(node))
            for child_coord in node.get_children_coordinates(right=largest):
                unbirthed_children_links.append(node_coordinates)
                unbirthed_children_links.append(child_coord)
        # else:
        #     print("{} is instantiated, but doesn't have its evaluated sum in either dicts!".format(node_name) )

    handles, labels = [], []

    if ax is None:
        ax = plt.subplot()
    b_node = transform_matrix @ (ary(birthed) + offset).T
    ax.scatter(*b_node, s=normalize_size(ary(birthed_sum)), marker="o", facecolors='none', edgecolors=primary_color) # nodes instantiated but not sorted.

    # draw the nodes
    b_edgex, b_edgey = ( transform_matrix @ (ary(birthed_children_links) + offset).T ).reshape([2, -1, 2])
    b_lines = ax.plot(b_edgex.T, b_edgey.T, color=primary_color)
    handles.append(b_lines[0])
    labels.append("nodes already inserted into sorted_sum_result")
    if unbirthed:
        u_node = transform_matrix @ (ary(unbirthed) + offset).T
        ax.scatter(*u_node, s=normalize_size(ary(unbirthed_sum)), marker="o", facecolors='none', edgecolors="C0") # nodes on the leading edge, instantiated but not sorted, whose children has not yet been instantiated
        if plot_dangling_links:
            u_edgex, u_edgey = ( transform_matrix @ (ary(unbirthed_children_links) + offset).T ).reshape([2, -1, 2])
            u_lines = ax.plot(u_edgex.T, u_edgey.T, linestyle="--", color="C0") # link to children which aren't instantiated yet
            handles.append(u_lines[0])
            labels.append("link to yet un-instantiated children")
    if text:
        # write the names directly below them
        for location, name in zip(b_node.T, birthed_names):
            ax.annotate(name, location, ha="center", va="top")
        if unbirthed:
            for location, name in zip(u_node.T, unbirthed_names):
                ax.annotate(name, location, ha="center", va="top")
        ax.legend(handles, labels, loc=6)
    return ax

def get_full_graph_size(sorted_list_length, target_chosen_length):
    graph_depth = target_chosen_length * (sorted_list_length - target_chosen_length)
    cur_step_size = (target_chosen_length-1)/2
    graph_width = 0
    while cur_step_size>0:
        graph_width += cur_step_size * (sorted_list_length - target_chosen_length)
        cur_step_size -= 1
    return graph_width, graph_depth

def main(list_length, n_chosen, artisitc_plot=False, _MAKE_ANIMATION=False):
    # (8, 6): nice spiral with only one touching point
    # (8, 5): donut (2 touching points)
    # (6, 4): a very simple single touch point
    # (7, 4): a more pixellated donut, but good for demo
    # (10, 7): the prettiest donut, but good for demo

    if _ARTISTIC_PLOT:
        primary_color = "white"
        plt.style.use("dark_background")
        # secondary_color = "C0"
    else:
        primary_color = "black"
        plt.style.use("default")
        # secondary_color = "C0"
    np.random.seed(1)

    sorted_list = sorted(np.random.rand(list_length), reverse=True)
    if _MAKE_ANIMATION:
        vid_title = "{}_choose_{}".format(len(sorted_list), n_chosen)
        vid_save_name = "combination_graph/"+vid_title+".mp4"
        vid_comment = "#"

    graph_width, graph_depth = get_full_graph_size(len(sorted_list), n_chosen)

    if _MAKE_ANIMATION:
        import matplotlib.animation as manimation
        writer = manimation.writers['ffmpeg'](fps=15, metadata={"title":vid_title, "comment":vid_comment, "artist":"Ocean Wong"})
        # grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
        fig, ax = plt.subplots()#gridspec_kw=grid_kws)
        fig.suptitle("{} choose {}".format(len(sorted_list), n_chosen))


        with writer.saving(fig, vid_save_name, 300):
            for n, (down_dict, up_dict) in enumerate(zip(top_n_sums_generator(sorted_list, n_chosen, 0, True), top_n_sums_generator(sorted_list, n_chosen, 0, False))):
                if _ARTISTIC_PLOT:
                    ax.set_xticks([]), ax.set_yticks([])
                ax.set_xlim( (ary([-(graph_width+0), graph_width])*1.15) )
                ax.set_ylim( (ary([0, graph_depth])*1.05) - graph_depth*0.05/2 - graph_depth)
                ax = plot_nodes(*down_dict, largest=True, ax=ax, text=not _ARTISTIC_PLOT)
                if _ARTISTIC_PLOT:
                    ax = plot_nodes( *up_dict, largest=False, ax=ax, offset=(+0, -graph_depth), transform_matrix=ary([[-1, 0],[0, 1]]), text=not _ARTISTIC_PLOT)
                for duplicate in range(duplicate_sampling_scheduler(n)):
                    writer.grab_frame()
                ax.clear()
        fig.clf()
    else:
        print(sorted_list, n_chosen)

        for (down_dict, up_dict) in list(zip(top_n_sums_generator(sorted_list, n_chosen, 0, True), top_n_sums_generator(sorted_list, n_chosen, 0, False)))[-1:]:
            dpi = 100
            fig, ax = plt.subplots(figsize=(3200/dpi, 2800/dpi))
            ax = plot_nodes(*down_dict, largest=True, ax=ax, text=not _ARTISTIC_PLOT)
            ax = plot_nodes( *up_dict, largest=False, ax=ax, offset=(+0, -graph_depth), transform_matrix=ary([[-1, 0],[0, 1]]), text=not _ARTISTIC_PLOT)
            if _ARTISTIC_PLOT:
                ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlim( (ary([-(graph_width+0), graph_width])*1.15) )
            ax.set_ylim( (ary([0, graph_depth])*1.05) - graph_depth*0.05/2 - graph_depth)
            # plt.savefig("combination_graph/{}_choose_{}.svg".format(list_length, n_chosen), format="svg")
            plt.savefig("combination_graph/{}_choose_{}.png".format(list_length, n_chosen))
            plt.show()
            ax.cla()

from .__init__ import *

main(*sys.argv[1:])