from typing import Dict, List, Any, Optional, Union
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

class SemanticHypergraph:
    """
    Semantic Hypergraph representation of a knowledge graph
    TODO: Integrate HypergraphX
    """
        
    def __init__(self,
                 doc:Doc|None=None,
                 json_data:Dict[str, Any]|None=None):
        # HypergraphX Hypergraph instance
        #    directed by default
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
                        "type": "hasEntityType",
                        "subtype": ent.label_,
                    }
                )
                # remove the individual entity type node and its edges
                self.G.remove_node(ent.label_)


    def to_json(self) -> Dict[str, Any]:
        """Convert the hypergraph to JSON format"""
        nodes = json.dumps(list(self.get_nodes(metadata=True)), indent=4)
        edges = json.dumps(list(self.get_edges(metadata=True)), indent=4)
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
        
        new_graph = SemanticHypergraph()
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

        # get token type
        if t.is_currency:
            tags["type"] = "currency"
        elif t.like_url:
            tags["type"] = "url"
        elif t.like_email:
            tags["type"] = "email"
        elif t.is_alpha:
            tags["type"] = "word"
        elif t.like_num:
            tags["type"] = "num"
        elif t.is_space:
            tags["type"] = "whitespace"
        elif t.is_punct:
            tags["type"] = "punct"
            if t.is_left_punct:
                tags["subtype_features"] = ["left"]
            elif t.is_right_punct:
                tags["subtype_features"] = ["right"]
            if t.is_bracket:
                tags["subtype_features"].append("bracket")
            elif t.is_quote:
                tags["subtype_features"].append("quote")
        
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
            edge_list.append((
                t.i,
                child.i
            ))
            metadata_list.append({
                "type": child.dep_,
                "rel_pos": "after"
            })
        for child in t.rights:
            edge_list.append((
                t.i,
                child.i
            ))
            metadata_list.append({
                "type": child.dep_,
                "rel_pos": "before"
            })
        return edge_list, metadata_list


    # Add nodes and edges to the hypergraph 
    def add_node(self,
                  node,
                  metadata: Optional[Dict[str, Any]] = None):
        if metadata:
            metadata["node_id"] = str(node)
        self.G.add_node(node, metadata=metadata)
    def add_nodes(self,
                  nodes: Union[List[Any], List[int]],
                  metadata: Optional[List[Dict[str, Any]]] = None):
        if metadata:
            for i in range(len(metadata)):
                metadata[i]["node_id"] = str(nodes[i])
        self.G.add_nodes(nodes, metadata=metadata)
    def add_edge(self,
                  edge,
                  metadata: Optional[Dict[str, Any]] = None):
        self.G.add_node(edge, metadata=metadata)
    def add_edges(self,
                  edges,
                  metadata: Optional[List[Dict[str, Any]]] = None):
        self.G.add_edges(edges, metadata=metadata)
    

    # Get nodes and edges from the hypergraph
    def get_nodes(self,
                  metadata:bool = True) -> list[Any]\
                                            | dict[Any, dict[Any, Any]]:
        return self.G.get_nodes(metadata=metadata)
    def get_edges(self,
                  metadata:bool = True) -> Hypergraph\
                                            | list[Any]\
                                            | dict[Any, dict[Any, Any]]:
        return self.G.get_edges(metadata=metadata)

    def get_node_with_idx(self, node_idx) -> dict:
        return self.get_nodes()[node_idx]