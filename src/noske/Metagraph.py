import ast
from typing import Dict, List, Any, Optional
import json

from hypergraphx.core.directed_hypergraph import DirectedHypergraph
from hypergraphx.linalg import *
from hypergraphx.generation.random import *

from noske.MetagraphUtils import (
    wrap,
    _is_edge,
    _flatten_edge,
    _get_required_node_fields,
    _get_required_edge_fields,
)

class Metagraph(DirectedHypergraph):
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
        metavert_idx = len(self.get_nodes())
        self.add_node(metavert_idx, metadata=metavert_metadata)
        
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

        return metavert_idx
    

    def _add_metaedge(self,
            edge: tuple,
            metadata=dict(),
        ):
        # create a list indicating whether each element of the edge is an edge itself
        subedge_mask = [_is_edge(e) for e in edge]

        new_edge = list()
        for nested_edge_idx, is_e in enumerate(subedge_mask):
            if not is_e:
                # if it's not an edge, add it to the list of members of the new hyperedge
                new_edge.append(edge[nested_edge_idx])
            else:
                # if it is an edge, recurse on the nested edge
                self.add_edge(edge[nested_edge_idx])
                # get the most recently created edge (should be the outer-most shell of the metaedge onion)
                most_recent_edge_idx = self._next_edge_id - 1
                most_recent_edge = self.get_edge_with_edge_list_idx(most_recent_edge_idx)
                # add a metavertex for it
                metavert_idx = self._add_metavert(most_recent_edge, metadata=metadata)
                # add the metavert's idx to the new edge
                new_edge.append(metavert_idx)

        # add the new hyperedge containing only regular vertices to the metagraph
        metadata.update(_get_required_edge_fields(edge, "hyper"))
        super().add_edge(tuple(new_edge), metadata=metadata)


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

        # if it's a directed edge
        if len(edge) == 2:

            # if it's a pairwise edge between nodes (ex: (1, 2))
            if not any(subedge_mask):
                metadata.update(_get_required_edge_fields(edge, "regular"))
                super().add_edge(edge, metadata=metadata)
            
            # if it's a hyperedge or metaedge
            else:
                
                # if it's a hyperedge (ex: ((1, 2, 3), (4, 5, 6)))
                if not any([_is_edge(e) for e in edge[0]])\
                    and not any([_is_edge(e) for e in edge[1]]):
                    metadata.update(_get_required_edge_fields(edge, "hyper"))
                    super().add_edge(edge, metadata=metadata)
                
                # if it's a directed metaedge (ex: ((1, 2, (3, 4)), (5, 6, 7)))
                else:
                    self._add_metaedge(edge, metadata)
        
        # if it's an undirected hyperedge or metaedge
        else:

            # if it's an undirected hyperedge (ex: (1, 2, 3))
            if not any(subedge_mask):
                metadata.update(_get_required_edge_fields(edge, "hyper"))
                # add a directed edge from the contents of the edge to itself
                super().add_edge((edge, edge), metadata=metadata)
            
            # if it's an undirected metaedge (ex: ((1, 2, (3, 4)), (5, 6, 7), 8, 9))
            else:
                self._add_metaedge(edge, metadata)


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
        return super().get_nodes(metadata=True)  # type: ignore
    
    def get_node_with_id(self, id: Any) -> tuple[int, dict] | None:
        """Find the node with the given id"""
        for key, data in self.get_nodes().items():
            if data["id"] == id:
                return (key, data)
        return None
    

    # Edge query methods
    def get_edges(self) -> dict[Any, dict[Any, Any]]:
        """Get all edges with their metadata"""
        return super().get_edges(metadata=True)  # type: ignore
    
    def get_edge_with_id(self, id: Any) -> tuple[int, dict] | None:
        """Find the edge with the given id"""
        for key, data in self.get_edges().items():
            if data["id"] == id:
                return (key, data)
        return None
    
    def get_edge_with_edge_list_idx(self, idx: int) -> tuple[int, dict] | None:
        """Find the edge with the given index"""
        for key, data in self._edge_list.items():
            if data == idx:
                return key
        return None
    

    # Metavertex query methods
    def get_all_metaverts(self) -> dict[int, dict]:
        """Get all metavertices (nodes of type 'meta')"""
        return {
            key: data
            for key, data in self.get_nodes().items()
            if data.get("type") == "meta"
        }