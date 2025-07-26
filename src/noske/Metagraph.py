import ast
from typing import Dict, List, Any, Optional
import json

from hypergraphx.linalg import *
from hypergraphx.generation.random import *

from noske.MetagraphUtils import (
    wrap,
    _is_edge,
    _flatten_edge,
    _get_required_node_fields,
    _get_required_edge_fields,
)

class Metagraph:
    """
    Base class for metagraph structures using HypergraphX.
    
    A metagraph is a hypergraph that can contain metavertices (nodes representing hyperedges)
    and supports hierarchical structures through nested edges.
    """
    
    def __init__(self, json_data: Dict[str, Any] | None = None):
        """
        Initialize a new Metagraph.
        
        Args:
            json_data: Optional JSON data to load the metagraph from
        """
        # HypergraphX Hypergraph instance
        self.G = Hypergraph()
        
        if json_data is not None:
            self.from_json(json_data)
    
    def to_json(self) -> str:
        """Convert the hypergraph to JSON format"""
        node_data = {
            k: v
            for k, v in self.get_nodes().items()
        }
        edge_data = {
            str(k): v
            for k, v in self.get_edges().items()
        }
        combined_dict = {
            "nodes": node_data,
            "edges": edge_data
        }
        return json.dumps(
            combined_dict,
            indent=4,
            default=str
        )
    
    @staticmethod
    def from_json(json_data: str) -> 'Metagraph':
        """
        Load a Metagraph from JSON format.
        
        Args:
            json_data: JSON dict containing nodes and edges data
            
        Returns:
            New Metagraph instance
        """
        json_data = json.loads(json_data)

        # load the nodes' json data as a dict
        node_data = json_data["nodes"]
        nodes = list()
        nodes_metadata = list()
        for k, v in node_data.items():
            nodes.append(k)
            nodes_metadata.append(v)

        # load the edges' json data as a dict
        edge_data = json_data["edges"]
        edges = list()
        edges_metadata = list()
        for k, v in edge_data.items():
            # load the edge keys as tuples instead of strings
            edges.append(ast.literal_eval(k))
            edges_metadata.append(v)
        
        # build the metagraph from the node and edge data
        new_graph = Metagraph()
        new_graph.add_nodes(nodes, metadata=nodes_metadata)
        new_graph.add_edges(edges, metadata=edges_metadata)
        return new_graph
    

    # Node management methods
    def add_node(self,
                 node: int | None = None,
                 metadata: Dict[str, Any] = None,
                 node_type: str = "regular"):
        """
        Add a single node to the metagraph.
        
        Args:
            node: The node identifier
            metadata: Optional metadata dictionary for the node
            node_type: Type of node ("regular" or "meta")
        """
        if not node and not metadata:
            raise ValueError("Must provide node id or metadata in order to create a node.")
        if node is None:
            node = len(self.get_nodes())
        if metadata is None:
            metadata = {}
        
        metadata.update(_get_required_node_fields(
            id=node,
            node_type=node_type
        ))
        self.G.add_node(node, metadata=metadata)

    def add_nodes(self,
                  nodes: List[int] | None= None,
                  metadata: List[Dict[str, Any]] = None,
                  node_type: str = "regular"):
        """
        Add multiple nodes to the metagraph.
        
        Args:
            nodes: List of node identifiers
            metadata: Optional list of metadata dictionaries for each node
            node_type: Type of nodes ("regular" or "meta")
        """
        if not nodes and not metadata:
            raise ValueError("Must provide node id or metadata in order to create a node.")
        if nodes is None:
            nodes = [i + len(self.get_nodes()) for i in range(len(metadata))]
        if metadata is None:
            metadata = [{}] * len(nodes)
        
        # Update metadata with required fields
        for i in range(len(nodes)):
            metadata[i].update(_get_required_node_fields(
                id=nodes[i],
                node_type=node_type
            ))
        
        # Convert list-based metadata to dict-based metadata for HypergraphX
        metadata_dict = {nodes[i]: metadata[i] for i in range(len(nodes))}
        self.G.add_nodes(nodes, metadata=metadata_dict)

    def _add_metavert(self,
                      edge: tuple,
                      metadata: Dict[str, Any]):
        """
        Add a metavertex (node representing a hyperedge) to the graph.
        
        Args:
            edge: The hyperedge to create a metavertex for
            metadata: Metadata for the metavertex
        """
        # add required fields to the metavertex's metadata
        metavert_metadata = metadata.copy()
        metavert_metadata.update(_get_required_node_fields(
            id=edge,
            node_type="meta"
        ))
        
        # add the metavert to the graph
        metavert_idx = len(self.G.get_nodes())
        self.G.add_node(metavert_idx, metadata=metavert_metadata)
        
        # Convert all elements to strings to avoid sorting issues
        flattened_edge = tuple(e for e in _flatten_edge(edge))
        metavert_tuple = (metavert_idx, )
        
        # link the metavertex to the hyperedge
        parent_child_e = (metavert_tuple, flattened_edge)
        self.add_edge(
            edge=parent_child_e,
            metadata=_get_required_edge_fields(
                id=parent_child_e,
                edge_type="metavert_to_hye"
            )
        )
        
        # link the hyperedge to the metavertex
        child_parent_e = (flattened_edge, metavert_tuple)
        self.add_edge(
            edge=child_parent_e,
            metadata=_get_required_edge_fields(
                id=child_parent_e,
                edge_type="hye_to_metavert"
            )
        )
    

    # Edge management methods
    def add_edge(self,
                 edge: tuple,
                 metadata: Dict[str, Any] | None = None):
        """
        Add a single edge to the metagraph.
        
        Args:
            edge: Tuple representing the edge
            metadata: Optional metadata dictionary for the edge
        """
        if metadata is None:
            metadata = {}
        
        # create a list indicating whether each element of the edge is an edge itself
        subedge_mask = [_is_edge(e) for e in edge]
        # wrap naked node indices (integers) in tuples to avoid comparison issues
        edge = tuple(wrap(e) for e in edge)

        # if it's a pairwise edge between nodes
        if len(edge) == 2 and not any(subedge_mask):
            metadata.update(_get_required_edge_fields(edge, "regular"))
            self.G.add_edge(edge, metadata=metadata)
        # if it's an undirected hyperedge
        elif not any(subedge_mask):
            metadata.update(_get_required_edge_fields(edge, "hyper"))
            try:
                self.G.add_edge(edge, metadata=metadata)
                self._add_metavert(edge, metadata=metadata)
            except TypeError:
                raise TypeError(f"TypeError, likely due to invalid comparison while sorting edge: {edge}")
        # if it's a directed hyperedge
        elif len(edge) == 2:
            metadata.update(_get_required_edge_fields(edge, "hyper"))
            self.G.add_edge(edge, metadata=metadata)
        # if len(edge) != 2 and the edge contains a nested edge
        #   ie if it's a metaedge/metavertex
        else:
            raise ValueError("Can't add a metavertex/metaedge (hyperedge with nested children) to edge list.")
    
    def add_edges(self,
                  edges: List[tuple],
                  metadata: List[Dict[str, Any]] | None= None):
        """
        Add multiple edges to the metagraph.
        
        Args:
            edges: List of edge tuples
            metadata: Optional list of metadata dictionaries for each edge
        """
        if metadata is None:
            metadata = [{}] * len(edges)
        
        # ensure the metadata provided is of the same length as the edgelist
        if len(edges) != len(metadata):
            raise Exception("len(edges) != len(metadata)")
        
        for i in range(len(edges)):
            self.add_edge(edges[i], metadata[i])
    


    # Query methods
    # Node query methods
    def get_nodes(self) -> dict[Any, dict[Any, Any]]:
        """Get all nodes with their metadata"""
        return self.G.get_nodes(metadata=True)  # type: ignore
    
    def get_node_with_id(self, id: Any) -> tuple[int, dict] | None:
        """Find the node with the given id"""
        for key, data in self.get_nodes().items():
            if data["id"] == id:
                return (key, data)
        return None
    

    # Edge query methods
    def get_edges(self) -> dict[Any, dict[Any, Any]]:
        """Get all edges with their metadata"""
        return self.G.get_edges(metadata=True)  # type: ignore
    
    def get_edge_with_id(self, id: Any) -> tuple[int, dict] | None:
        """Find the edge with the given id"""
        for key, data in self.get_edges().items():
            if data["id"] == id:
                return (key, data)
        return None
    

    # Metavertex query methods
    def get_all_metaverts(self) -> dict[int, dict]:
        """Get all metavertices (nodes of type 'meta')"""
        return {
            key: data
            for key, data in self.get_nodes().items()
            if data.get("type") == "meta"
        }