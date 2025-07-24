import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from hypergraphx import Hypergraph

def to_nx(g:Hypergraph) -> nx.Graph|nx.DiGraph:
    """
    Convert a directed Hypergraph to a NetworkX Graph.
    """
    G = nx.DiGraph()
    for node in g.nodes():
        G.add_node(node, **g.get_node_metadata(node))
    
    for edge in g.get_edges(order=1, metadata=True):
        G.add_edge(edge, **g.get_edge_metadata(edge))
    
    return G