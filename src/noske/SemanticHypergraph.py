from typing import Dict, List, Any
import json
import networkx as nx
from spacy.tokens import Doc, Token
import matplotlib.pyplot as plt

class SemanticHypergraph:
    """
    Semantic Hypergraph representation of a knowledge graph
    TODO: Integrate HypergraphX
    """
        
    def __init__(self, doc: Doc):
        """
        Initialize from a spaCy Doc object
        """
        self.G = nx.DiGraph()

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
            self.G.add_nodes_from([(t.i, node_dict)])

            # add token NER relations
            if t.ent_type_:
                self.G.add_nodes_from([
                    (t.ent_type_, {"text": t.ent_type_})
                ])
                self.G.add_edges_from([
                    (t.i, t.ent_type_, {"type":"type"}),
                ])
            # add dependency relations
            self.G.add_edges_from(self.get_dep_edges(t))
    

    def to_json(self) -> Dict[str, Any]:
        """Convert the hypergraph to JSON format"""
        nodes = json.dumps(list(self.G.nodes(data=True)), indent=4)
        edges = json.dumps(list(self.G.edges(data=True)), indent=4)
        return {"nodes": nodes, "edges": edges}
    

    @staticmethod
    def from_json(json_data: Dict[str, Any]):
        """Load the hypergraph from JSON format"""
        nodes = json.loads(json_data["nodes"])
        edges = json.loads(json_data["edges"])
        
        new_graph = SemanticHypergraph(nx.DiGraph())
        new_graph.G.add_nodes_from(nodes)
        new_graph.G.add_edges_from(edges)
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
    def get_dep_edges(t: Token) -> List[tuple]:
        edge_list = list()
        for child in t.lefts:
            edge_list.append((
                t.i,
                child.i,
                {"type": child.dep_, "rel_pos": "after"}
            ))
        for child in t.rights:
            edge_list.append((
                t.i,
                child.i,
                {"type": child.dep_, "rel_pos": "before"}
            ))
        return edge_list


    def plot(self):
        # Extract node labels and edge labels
        node_labels = {
            node: self.G.nodes[node]['text']
            for node in self.G.nodes()
            if 'text' in self.G.nodes[node]
        }
        edge_labels = {
            (u, v): d['type']
            for u, v, d in self.G.edges(data=True)
        }

        # Position nodes using spring layout
        pos = nx.spring_layout(self.G, k=50)

        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            labels=node_labels,
            node_size=1500,
            node_color="skyblue",
            alpha=0.8,
            linewidths=2,
            edge_color="gray"
        )
        nx.draw_networkx_edge_labels(
            self.G,
            pos,
            edge_labels=edge_labels,
            font_color='red'
        )
        plt.title("Semantic Knowledge Graph")
        plt.axis("off")
        plt.show()