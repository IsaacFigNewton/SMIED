from typing import Dict, List, Any, Tuple, Optional
import json
from spacy.tokens import Doc, Token
import matplotlib.pyplot as plt
import networkx as nx

from .DirectedMetagraph import DirectedMetagraph

class SemanticMetagraph(DirectedMetagraph):
    """
    Semantic Metagraph representation of a knowledge graph
    """
        
    def __init__(self, doc: Optional[Doc] = None, vert_list: Optional[List[Tuple]] = None):
        """
        Initialize from a spaCy Doc object or a list of vertices
        """
        if doc is not None:
            # Build metaverts from spaCy Doc
            vert_list = self._build_metaverts_from_doc(doc)
        
        # Initialize parent class with the vertex list
        super().__init__(vert_list)
        
        # Store doc for reference if provided
        self.doc = doc
    
    def _build_metaverts_from_doc(self, doc: Doc) -> List[Tuple]:
        """
        Build metavert list from a spaCy Doc object
        """
        metaverts = []
        token_idx_map = {}  # Map token indices to metavert indices
        
        # First pass: add all tokens as atomic metaverts
        mv_idx = 0
        for t in doc:
            # Create metadata for the token
            metadata = {
                "text": t.text,
                "pos": t.pos_,
                "head": t.head.i,
                "lemma": t.lemma_,
                "idx": t.i  # Original token index
            }
            metadata.update(self.get_token_tags(t))
            
            # Add as atomic metavert (string with metadata)
            metaverts.append((t.text, metadata))
            token_idx_map[t.i] = mv_idx
            mv_idx += 1
        
        # Second pass: add NER and dependency relations as complex metaverts
        for t in doc:
            curr_idx = token_idx_map[t.i]
            
            # Add NER type relations if present
            if t.ent_type_:
                # Check if we already have this entity type
                ent_type_mv = None
                for i, mv in enumerate(metaverts):
                    if isinstance(mv[0], str) and len(mv) > 1 and mv[1].get("is_entity_type") and mv[0] == t.ent_type_:
                        ent_type_mv = i
                        break
                
                # If not, add it
                if ent_type_mv is None:
                    metaverts.append((t.ent_type_, {"is_entity_type": True}))
                    ent_type_mv = mv_idx
                    mv_idx += 1
                
                # Add relation between token and entity type
                metaverts.append((
                    (curr_idx, ent_type_mv),
                    {"relation": "has_entity_type"}
                ))
                mv_idx += 1
            
            # Add dependency relations
            for child in t.children:
                child_idx = token_idx_map[child.i]
                rel_pos = "before" if child.i < t.i else "after"
                metaverts.append((
                    (curr_idx, child_idx),
                    {"relation": child.dep_, "rel_pos": rel_pos}
                ))
                mv_idx += 1
        
        return metaverts
    

    def to_json(self) -> Dict[str, Any]:
        """Convert the metagraph to JSON format"""
        # Convert metaverts to JSON-serializable format
        json_metaverts = []
        for mv_id, mv in self.metaverts.items():
            mv_data = {"id": mv_id}
            if len(mv) == 1:
                mv_data.update({"type": "atomic", "value": mv[0]})
            elif len(mv) == 2:
                if isinstance(mv[0], str):
                    mv_data.update({"type": "atomic", "value": mv[0], "metadata": mv[1]})
                elif isinstance(mv[0], tuple):
                    mv_data.update({"type": "directed", "source": mv[0][0], "target": mv[0][1], "metadata": mv[1]})
                elif isinstance(mv[0], list):
                    mv_data.update({"type": "undirected", "nodes": mv[0], "metadata": mv[1]})
            json_metaverts.append(mv_data)
        
        return {"metaverts": json.dumps(json_metaverts, indent=4)}
    

    @staticmethod
    def from_json(json_data: Dict[str, Any]):
        """Load the metagraph from JSON format"""
        json_metaverts = json.loads(json_data["metaverts"])
        
        # Convert JSON back to metavert format
        metaverts = []
        for jmv in json_metaverts:
            if jmv["type"] == "atomic":
                if "metadata" in jmv:
                    metaverts.append((jmv["value"], jmv["metadata"]))
                else:
                    metaverts.append((jmv["value"],))
            elif jmv["type"] == "directed":
                metaverts.append(((jmv["source"], jmv["target"]), jmv["metadata"]))
            elif jmv["type"] == "undirected":
                metaverts.append((jmv["nodes"], jmv["metadata"]))
        
        return SemanticMetagraph(vert_list=metaverts)
    

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
        """Plot the semantic metagraph using networkx"""
        # Convert to networkx graph
        G = self.to_nx()
        
        # Extract node labels
        node_labels = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            if 'label' in node_data:
                node_labels[node] = node_data['label']
            elif 'text' in node_data:
                node_labels[node] = node_data['text']
            else:
                node_labels[node] = str(node)
        
        # Extract edge labels
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if 'label' in d:
                edge_labels[(u, v)] = d['label']
            elif 'type' in d:
                edge_labels[(u, v)] = d['type']
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw the graph
        plt.figure(figsize=(14, 10))
        
        # Draw nodes with different colors based on type
        node_colors = []
        for node in G.nodes():
            node_data = G.nodes[node]
            if 'is_entity_type' in node_data:
                node_colors.append('lightgreen')
            elif 'relation' in node_data:
                node_colors.append('lightcoral')
            else:
                node_colors.append('skyblue')
        
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=node_labels,
            node_size=1500,
            node_color=node_colors,
            alpha=0.8,
            linewidths=2,
            edge_color="gray",
            font_size=8,
            font_weight='bold'
        )
        
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_color='red',
                font_size=7
            )
        
        plt.title("Semantic Metagraph", fontsize=16, fontweight='bold')
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    def get_tokens(self) -> List[Dict[str, Any]]:
        """Get all token metaverts with their metadata"""
        tokens = []
        for i, mv in self.metaverts.items():
            if isinstance(mv[0], str) and len(mv) > 1 and 'idx' in mv[1]:
                tokens.append({"metavert_idx": i, "token_idx": mv[1]['idx'], "text": mv[0], "metadata": mv[1]})
        return tokens
    
    def get_relations(self) -> List[Dict[str, Any]]:
        """Get all relation metaverts"""
        relations = []
        for i, mv in self.metaverts.items():
            if len(mv) > 1 and 'relation' in mv[1]:
                if isinstance(mv[0], tuple):
                    relations.append({
                        "metavert_idx": i,
                        "type": "directed",
                        "source": mv[0][0],
                        "target": mv[0][1],
                        "relation": mv[1]['relation'],
                        "metadata": mv[1]
                    })
                elif isinstance(mv[0], list):
                    relations.append({
                        "metavert_idx": i,
                        "type": "undirected",
                        "nodes": mv[0],
                        "relation": mv[1]['relation'],
                        "metadata": mv[1]
                    })
        return relations