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
from hypergraphx.viz import Object

class SemanticHypergraph:
    """
    Semantic Hypergraph representation of a knowledge graph
    TODO: Integrate HypergraphX
    """
        
    def __init__(self, doc: Doc):
        """
        Initialize from a spaCy Doc object
        """
        self.G = Hypergraph()
        self.directed = True  # Hypergraph is directed by default

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
            self.G.add_node(t.i, metadata=node_dict)

            # add token NER relations
            if t.ent_type_:
                # add the entity type as a node if it does not exist
                self.G.add_node(
                    t.ent_type_,
                    metadata={"text": t.ent_type_}
                )
                # will aggregate the outgoing edges of the entity type into a single hyperedge
                self.G.add_edges(
                    [
                        (t.i, t.ent_type_),
                        (t.ent_type_, t.i),
                    ],
                    metadata=[
                        {"type": "hasEntityType"},
                        {"type": "contains"}
                    ]
                )
            # add dependency relations
            dep_edges, dep_edge_metadata = self.get_dep_edges(t)
            self.G.add_edges(
                dep_edges,
                metadata=dep_edge_metadata
            )
        
        # convert the outgoing edges of the t.ent_type_ nodes into hyperedges
        for ent_type in doc.ents:
            if ent_type.ent_type_:
                # get all nodes that are connected to the entity type
                connected_nodes = list(self.G.get_neighbors(ent_type.ent_type_))
                if len(connected_nodes) > 1:
                    # create a hyperedge for the entity type
                    self.G.add_edge(
                        (node for node in connected_nodes),
                        metadata={
                            "type": "hasEntityType",
                            "subtype": ent_type.ent_type_,
                        }
                    )
                    # remove the individual entity type node and its edges
                    self.G.remove_node(ent_type.ent_type_)
    
    def to_nx(self) -> nx.Graph|nx.DiGraph:
        """
        Convert a Hypergraph to a NetworkX Graph.
        """
        G = nx.DiGraph() if self.directed else nx.Graph()
        for node in self.G.nodes():
            G.add_node(node, **self.G.get_node_metadata(node))
        
        for edge in self.G.get_edges(order=1, metadata=True):
            G.add_edge(edge, **self.G.get_edge_metadata(edge))
        
        return G

    def to_json(self) -> Dict[str, Any]:
        """Convert the hypergraph to JSON format"""
        nodes = json.dumps(list(self.G.nodes(metadata=True)), indent=4)
        edges = json.dumps(list(self.G.edges(metadata=True)), indent=4)
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
        
        new_graph = SemanticHypergraph(Hypergraph())
        new_graph.G.add_nodes(nodes, metadata=nodes_metadata)
        new_graph.G.add_edges(edges, metadata=edges_metadata)
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
    

    def get_node_labels(self, key:str="text") -> Dict[int, str]:
        """
        Get node labels for visualization.
        """
        return {
            node: metadata.get(key, '')
            for node, metadata in self.G.nodes(metadata=True).items()
            if key in metadata.keys()
        }
    

    def get_pairwise_edge_labels(self, key:str="type") -> Dict[tuple, str]:
        """
        Get edge labels for edges of order 1 (standard edge pairs) to use for visualization.
        """
        return {
            edge: metadata.get(key, '')
            for edge, metadata in self.G.get_edges(order=1, metadata=True).items()
            if key in metadata.keys()
        }


    def get_hyperedge_labels(self, key:str="type") -> Dict[tuple, str]:
        """
        Get hyperedge labels for visualization.
        """
        return {
            edge: metadata.get(key, '')
            for edge, metadata in self.G.edges(metadata=True).items()
            if key in metadata.keys() and len(edge) > 2
        }