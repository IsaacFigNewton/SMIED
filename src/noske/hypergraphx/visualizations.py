from typing import Dict, List, Optional, Union
import random
import networkx as nx
import matplotlib.pyplot as plt
from hypergraphx.linalg import *
from hypergraphx.representations.projections import clique_projection
from hypergraphx.generation.random import *

from noske.hypergraphx.Object import Object
from noske.SemanticHypergraph import SemanticHypergraph

def draw(
    # main parameters
    hypergraph: SemanticHypergraph,
    figsize: tuple = (12, 7),
    ax: Optional[plt.Axes] = None,

    # node position parameters
    pos: Optional[dict] = None,
    iterations: int = 100,
    seed: int = 10,
    scale: int = 1,
    k: float = 0.5,

    # node styling
    with_node_labels: bool = False,
    node_size: Union[int, np.array] = 150,
    node_color: Union[str, np.array] = "#E2E0DD",
    node_facecolor: Union[str, np.array] = "black",
    node_shape: str = "o",

    # edge styling
    with_pairwise_edge_labels: bool = False,
    pairwise_edge_color: str = "lightgrey",
    pairwise_edge_width: float = 1.2,
    
    # hyperedge styling
    # Set color hyperedges of size > 2 (order > 1).
    with_hyperedge_labels: bool = False,
    hyperedge_color_by_order: dict = {2: "#FFBC79", 3: "#79BCFF", 4: "#4C9F4C"},
    hyperedge_facecolor_by_order: dict = {2: "#FFBC79", 3: "#79BCFF", 4: "#4C9F4C"},
    hyperedge_alpha: Union[float, np.array] = 0.8,

    # other styling parameters
    label_size: float = 10,
    label_col: str = "black",
):
    """Visualize a hypergraph."""
    # Initialize figure.
    if ax is None:
        plt.figure(figsize=figsize)
        plt.subplot(1, 1, 1)
        ax = plt.gca()

    # Extract node positions based on the hypergraph clique projection.
    if pos is None:
        pos = nx.spring_layout(
            clique_projection(hypergraph.G, keep_isolated=True),
            iterations=iterations,
            seed=seed,
            scale=scale,
            k=k,
        )


    # Initialize a networkx graph with the nodes and only the pairwise interactions of the hypergraph.
    pairwise_G = hypergraph.to_nx()

    # Plot the pairwise graph.
    if type(node_shape) == str:
        node_shape = {n: node_shape for n in pairwise_G.nodes()}
    for nid, n in enumerate(list(pairwise_G.nodes())):
        nx.draw_networkx_nodes(
            pairwise_G,
            pos,
            [n],
            with_labels=with_node_labels,
            labels=hypergraph.get_node_labels() if with_node_labels else None,
            node_size=node_size,
            node_shape=node_shape[n],
            node_color=node_color,
            alpha=0.8,
            linewidths=2,
            edgecolors=node_facecolor,
            ax=ax,
        )

    # Plot the edges of the pairwise graph.
    if with_pairwise_edge_labels:
        nx.draw_networkx_edge_labels(
            pairwise_G,
            pos,
            edge_labels=hypergraph.get_pairwise_edge_labels(),
            font_size=label_size,
            font_color=label_col,
            ax=ax,
        )
    nx.draw_networkx_edges(
        pairwise_G,
        pos,
        width=pairwise_edge_width,
        edge_color=pairwise_edge_color,
        alpha=0.8,
        ax=ax,
    )

    # Plot the hyperedges (size>2/order>1).
    for hye in list(hypergraph.G.get_edges()):
        if len(hye) > 2:
            x1, y1, color, facecolor = hypergraph.get_hyperedge_styling_data(
                hye,
                pos,
                hyperedge_color_by_order,
                hyperedge_facecolor_by_order
            )
            ax.fill(
                x1,
                y1,
                alpha=hyperedge_alpha,
                c=color,
                edgecolor=facecolor
            )
            if with_hyperedge_labels:
                ax.annotate(
                    hypergraph.get_hyperedge_labels(),
                    (pos[n][0] - 0.1, pos[n][1] - 0.06),
                    fontsize=label_size,
                    color=label_col,
                )
    
    # Set the aspect ratio of the plot to be equal.
    ax.axis("equal")
    plt.axis("equal")
    plt.title("Semantic Knowledge Graph")
    plt.show()


def get_hyperedge_styling_data(
        hye,
        pos: Dict[int, tuple],
        hyperedge_color_by_order: Dict[int, str],
        hyperedge_facecolor_by_order: Dict[int, str]) -> List[tuple]:
    """
    Get the fill data for a hyperedge.
    """
    points = []
    for node in hye:
        points.append((pos[node][0], pos[node][1]))
        # Center of mass of points.
        x_c = np.mean([x for x, y in points])
        y_c = np.mean([y for x, y in points])
    # Order points in a clockwise fashion.
    points = sorted(points, key=lambda x: np.arctan2(x[1] - y_c, x[0] - x_c))

    if len(points) == 3:
        points = [
            (x_c + 2.5 * (x - x_c), y_c + 2.5 * (y - y_c)) for x, y in points
        ]
    else:
        points = [
            (x_c + 1.8 * (x - x_c), y_c + 1.8 * (y - y_c)) for x, y in points
        ]
    Cartesian_coords_list = points + [points[0]]

    obj = Object(Cartesian_coords_list)
    Smoothed_obj = obj.Smooth_by_Chaikin(number_of_refinements=12)
    
    # Extract x and y coordinates from the smoothed object.
    x1 = [i for i, j in Smoothed_obj]
    y1 = [j for i, j in Smoothed_obj]

    order = len(hye) - 1

    if order not in hyperedge_color_by_order.keys():
        std_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        hyperedge_color_by_order[order] = std_color

    if order not in hyperedge_facecolor_by_order.keys():
        std_face_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        hyperedge_facecolor_by_order[order] = std_face_color

    color = hyperedge_color_by_order[order]
    facecolor = hyperedge_facecolor_by_order[order]

    return x1, y1, color, facecolor