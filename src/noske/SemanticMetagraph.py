from typing import Dict, List, Any
from spacy.tokens import Doc, Token

from hypergraphx.linalg import *
from hypergraphx.generation.random import *

from noske.Metagraph import Metagraph
from noske.MetagraphUtils import _get_required_edge_fields

class SemanticMetagraph(Metagraph):
    """
    Specialized metagraph for semantic and linguistic structures.
    
    Extends the base Metagraph class with functionality for processing
    SpaCy documents and creating semantic relationships.
    """
    
    def __init__(self,
                 doc: Doc | None = None,
                 json_data: Dict[str, Any] | None = None):
        """
        Initialize a new SemanticMetagraph.
        Args:
            doc: Optional SpaCy document to load into the metagraph
            json_data: Optional JSON data to load the metagraph from
        """
        super().__init__(json_data)
        if doc is not None:
            self.add_doc(doc)
    

    def add_doc(self, doc: Doc):
        """
        Load a parsed SpaCy document into the hypergraph.
        Args:
            doc: SpaCy Doc object containing parsed text
        """
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
                    tuple(connected_nodes),
                    metadata={
                        "class": "hasEntityType",
                        "subclass": ent.label_,
                    }
                )
                # remove the individual entity type node and its edges
                self.G.remove_node(ent.label_)
    

    @staticmethod
    def get_token_tags(t: Token) -> Dict[str, Any]:
        """
        Extract linguistic tags and features from a SpaCy token.
        Args:
            t: SpaCy Token object
        Returns:
            Dictionary of token features and tags
        """
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
            tags["subclass_features"] = []
            if t.is_left_punct:
                tags["subclass_features"].append("left")
            elif t.is_right_punct:
                tags["subclass_features"].append("right")
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
        """
        Extract dependency edges from a SpaCy token.
        Args:
            t: SpaCy Token object
        Returns:
            Tuple of (edge_list, metadata_list) for dependency relations
        """
        edge_list = list()
        metadata_list = list()
        
        # add all dependencies left of the current token to the Metagraph
        for child in t.lefts:
            e = (t.i, child.i)
            edge_list.append(e)
            metadata_dict = {
                "relation": child.dep_,
                "rel_pos": "after"
            }
            metadata_dict.update(_get_required_edge_fields(
                id=e,
                edge_type="regular"
            ))
            metadata_list.append(metadata_dict)
        
        # add all dependencies right of the current token to the Metagraph
        for child in t.rights:
            e = (t.i, child.i)
            edge_list.append(e)
            metadata_dict = {
                "relation": child.dep_,
                "rel_pos": "before"
            }
            metadata_dict.update(_get_required_edge_fields(
                id=e,
                edge_type="regular"
            ))
            metadata_list.append(metadata_dict)
        
        return edge_list, metadata_list