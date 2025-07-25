from typing import Dict, List, Any, Optional
import random
import json
import networkx as nx
from spacy.tokens import Doc, Token
import matplotlib.pyplot as plt

import hypergraphx as hgx
from hypergraphx.generation.scale_free import scale_free_hypergraph
from hypergraphx.linalg import *
from hypergraphx.representations.projections import bipartite_projection, clique_projection
from hypergraphx.generation.random import *
from hypergraphx.readwrite.save import save_hypergraph
from hypergraphx.readwrite.load import load_hypergraph
from hypergraphx.viz.draw_hypergraph import draw_hypergraph

from noske.hypergraphx import Object

def _is_edge(edge: Any) -> bool:
    return isinstance(edge, tuple) and len(edge) > 1

def _flatten_edge(edge:Any) -> List[Any]:
    flattened_edge = list()
    for e in edge:
        if not _is_edge(e):
            flattened_edge.append(e)
        else:
            flattened_edge = flattened_edge + _flatten_edge(e)
    return flattened_edge

def _get_required_node_fields(id: int | str | tuple,
                                node_type: str) -> Dict[Any, Any]:
    # regular vertex ids:   v1
    # metavertex ids:      (v1, v2)
    #                       (v1, v2, v3, ...)
    #                       ((v11, ...), (v21, ...))
    #                       (((v11, ...), (v21, ...)), v31)
    #                       (((v11, ...), (v21, ...)), (v31, ...))
    #                       (((v11, ...), (v21, ...)), (v31, ...), (v41, ...)))
    #                       ...
    valid_types = {"regular", "meta"}
    if node_type not in valid_types:
        raise ValueError(f"Invalid node type: {node_type}. Valid types: {valid_types}")
    return {
        "id": id,
        "type": node_type
    }

def _get_required_edge_fields(id: tuple,
                                edge_type: str) -> Dict[Any, Any]:
    # regular edge ids:     (v1, v2)
    # hyperedge ids:        (v1, v2, v3, ...)
    #                       ((v11, ...), (v21, ...))
    # metaedge ids:         (((v11, ...), (v21, ...)), v31)
    #                       (((v11, ...), (v21, ...)), (v31, ...))
    #                       (((v11, ...), (v21, ...)), (v31, ...), (v41, ...)))
    #                       ...
    valid_types = {"regular", "hyper", "meta", "metavert_to_hye", "hye_to_metavert"}
    if edge_type not in valid_types:
        raise ValueError(f"Invalid edge type: {edge_type}. Valid types: {valid_types}")
    return {
        "id": id,
        "type": edge_type
    }

class SemanticMetagraph:
    """
    Semantic Metagraph representation of a knowledge graph
    """
        
    def __init__(self,
                 doc:Doc|None=None,
                 json_data:Dict[str, Any]|None=None):
        # HypergraphX Hypergraph instance
        self.G = Hypergraph()
        if doc is not None:
            # load the parsed SpaCy document into the hypergraph
            self.add_doc(doc)
        elif json_data is not None:
            # load the hypergraph from JSON
            self.from_json(json_data)
    
    def add_doc(self, doc:Doc):
        # add tokens and their relations
        for t in doc:
            # define and add the token's node to G
            node_dict = {
                "text": t.text,
                "pos": t.pos_,
                "head": t.head.i,
                "lemma": t.lemma_
            }
            node_dict.update(self.get_token_tags(t))
            self.add_node(t.i, metadata=node_dict)

            # add token NER relations
            if t.ent_type_:
                # add the entity type as a node if it does not exist
                self.add_node(
                    t.ent_type_,
                    metadata={"text": t.ent_type_}
                )
                # will aggregate the outgoing edges of the entity type into a single hyperedge
                self.add_edges(
                    [
                        ((t.i), (t.ent_type_)),
                        ((t.ent_type_), (t.i)),
                    ],
                    metadata=[
                        {"type": "hasEntityType"},
                        {"type": "contains"}
                    ]
                )
            # add dependency relations
            dep_edges, dep_edge_metadata = self.get_dep_edges(t)
            self.add_edges(
                dep_edges,
                metadata=dep_edge_metadata
            )
        
        # convert the outgoing edges of the t.ent_type_ nodes into hyperedges
        for ent in doc.ents:
            # get all nodes that are connected to the entity type
            connected_nodes = list(self.G.get_neighbors(ent.label_))
            if len(connected_nodes) > 1:
                # create a hyperedge for the entity type
                self.add_edge(
                    (node for node in connected_nodes),
                    metadata={
                        "class": "hasEntityType",
                        "subclass": ent.label_,
                    }
                )
                # remove the individual entity type node and its edges
                self.G.remove_node(ent.label_)


    def to_json(self) -> Dict[str, Any]:
        """Convert the hypergraph to JSON format"""
        nodes = json.dumps(list(self.get_nodes()), indent=4)
        edges = json.dumps(list(self.get_edges()), indent=4)
        return {"nodes": nodes, "edges": edges}
    

    @staticmethod
    def from_json(json_data: Dict[str, Any]):
        """Load the hypergraph from JSON format"""
        node_data = json.loads(json_data["nodes"])
        nodes = list(node_data.keys())
        nodes_metadata = [node_data[node] for node in nodes]
        edge_data = json.loads(json_data["edges"])
        edges = list(edge_data.keys())
        edges_metadata = [edge_data[edge] for edge in edges]
        
        new_graph = SemanticMetagraph()
        new_graph.add_nodes(nodes, metadata=nodes_metadata)
        new_graph.add_edges(edges, metadata=edges_metadata)
        return new_graph
    

    @staticmethod
    def get_token_tags(t: Token) -> Dict[str, Any]:
        tags = dict()

        # get token case
        if t.is_lower:
            tags["case"] = "lower"
        elif t.is_upper:
            tags["case"] = "upper"
        elif t.is_title:
            tags["case"] = "title"

        # get token class
        if t.is_currency:
            tags["class"] = "currency"
        elif t.like_url:
            tags["class"] = "url"
        elif t.like_email:
            tags["class"] = "email"
        elif t.is_alpha:
            tags["class"] = "word"
        elif t.like_num:
            tags["class"] = "num"
        elif t.is_space:
            tags["class"] = "whitespace"
        elif t.is_punct:
            tags["class"] = "punct"
            if t.is_left_punct:
                tags["subclass_features"] = ["left"]
            elif t.is_right_punct:
                tags["subclass_features"] = ["right"]
            if t.is_bracket:
                tags["subclass_features"].append("bracket")
            elif t.is_quote:
                tags["subclass_features"].append("quote")
        
        # get morphologic analysis as lists of tags
        morph_dict = t.morph.to_dict()
        morph_dict = {k: v.split(",") for k, v in morph_dict.items()}
        tags.update(morph_dict)
        return tags


    @staticmethod
    def get_dep_edges(t: Token) -> tuple[List[tuple], List[Dict[str, str]]]:
        edge_list = list()
        metadata_list = list()
        for child in t.lefts:
            e = (t.i, child.i)
            edge_list.append(e)
            metadata_list.append({
                "relation": child.dep_,
                "rel_pos": "after"
            })
            metadata_list[-1].update(_get_required_edge_fields(
                id=e,
                edge_type="regular"
            ))
        for child in t.rights:
            e = (t.i, child.i)
            edge_list.append(e)
            metadata_list.append({
                "relation": child.dep_,
                "rel_pos": "before"
            })
            metadata_list[-1].update(_get_required_edge_fields(
                id=e,
                edge_type="regular"
            ))
        return edge_list, metadata_list


    # Add nodes and edges to the hypergraph
    # Add nodes to graph
    def add_node(self,
                  node: Any,
                  metadata: Dict[str, Any] = dict(),
                  node_type:str="regular"):
        metadata.update(_get_required_node_fields(
            id=node,
            node_type=node_type
        ))
        self.G.add_node(node, metadata=metadata)
    def add_nodes(self,
                  nodes: List[Any],
                  metadata: List[Dict[str, Any]] = list(),
                  node_type:str="regular"):
        for i in range(len(metadata)):
            metadata[i].update(_get_required_node_fields(
                id=nodes[i],
                node_type=node_type
            ))
        self.G.add_nodes(nodes, metadata=metadata)
    
    # Add a hypernode to the the graph
    def _add_metavert(self,
                     edge: tuple,
                     metadata: Dict[str, Any]):
        # add required fields to the metavertex's metadata
        metadata.update(_get_required_node_fields(
            id=edge,
            node_type="meta"
        ))
        # add the metavert to the graph
        self.G.add_node(str(edge), metadata=metadata)
        # link the metavertex to the hyperedge
        parent_child_e = (str(edge), _flatten_edge(edge))
        self.add_edge(
            edge=parent_child_e,
            metadata=_get_required_edge_fields(
                id=parent_child_e,
                edge_type="metavert_to_hye"
            )
        )
        # link the hyperedge to the metavertex
        child_parent_e = (_flatten_edge(edge), str(edge))
        self.add_edge(
            edge=child_parent_e,
            metadata=_get_required_edge_fields(
                id=child_parent_e,
                edge_type="hye_to_metavert"
            )
        )
        

    # Add edges to graph
    def add_edge(self,
                  edge: tuple,
                  metadata: Dict[str, Any] = dict()):
        # create a list indicating whether each element of the edge is an edge itself
        subedge_mask = [_is_edge(e) for e in edge]

        # if it's a pairwise edge between nodes
        if len(edge) == 2 and not any(subedge_mask):
            metadata.update(_get_required_edge_fields(edge, "regular"))
            self.G.add_edge(edge, metadata=metadata)
        # if it's an undirected hyperedge
        elif not any(subedge_mask):
            metadata.update(_get_required_edge_fields(edge, "hyperedge"))
            self.G.add_edge(edge, metadata=metadata)
            self._add_metavert(edge, metadata=metadata)
        # if it's a directed hyperedge
        elif len(edge) == 2:
            metadata.update(_get_required_edge_fields(edge, "hyperedge"))
            self.G.add_edge(edge, metadata=metadata)
            self._add_metavert(edge, metadata=metadata)
        # if len(edge) != 2 and the edge contains a nested edge
        #   ie if it's a metaedge/metavertex
        else:
            raise ValueError("Can't add a metavertex/metaedge (hyperedge with nested children) to edge list.")
    def add_edges(self,
                  edges: List[tuple],
                  metadata: List[Dict[str, Any]] = list()):
        # if edge metadata provided
        if metadata:
            # ensure the metadata provided is of the same length as the edgelist
            if len(edges) != len(metadata):
                raise Exception("len(edges) != len(metadata)")
            for i in range(len(edges)):
                self.add_edge(edges[i], metadata[i])
        # if no edge metadata provided
        for i in range(len(edges)):
            self.add_edge(edges[i], metadata[i])


    # Get nodes and edges from the hypergraph
    # Get nodes
    def get_nodes(self) -> dict[Any, dict[Any, Any]]:
        return self.G.get_nodes(metadata=True)  # type: ignore
    def get_all_metaverts(self) -> dict[int, dict]:
        return {
            key: data
            for key, data in self.get_nodes().items()
            if data["type"] == "meta"
        }
    def get_node_with_id(self, id:Any) -> tuple[int, dict] | None:
        # Find the node with the given id
        for key, data in self.get_nodes().items():
            if data["id"] == id:
                return (key, data)
        return None
    
    # Get edges
    def get_edges(self) -> dict[Any, dict[Any, Any]]:
        return self.G.get_edges(metadata=True)  # type: ignore
    def get_edge_with_id(self, id:Any) -> tuple[int, dict] | None:
        # Find the edge with the given id
        for key, data in self.get_edges().items():
            if data["id"] == id:
                return (key, data)
        return None
    
    
    # Get metagraph information
    def get_metavert_metadata(self, hyperedge:tuple) -> dict[Any, Any] | None:
        return self.get_nodes().get(str(hyperedge))
