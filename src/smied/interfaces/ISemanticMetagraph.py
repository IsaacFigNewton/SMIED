from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from spacy.tokens import Doc, Token
from .IDirectedMetagraph import IDirectedMetagraph


class ISemanticMetagraph(IDirectedMetagraph):
    """
    Interface for Semantic Metagraph representation of a knowledge graph.
    
    Defines the contract for building semantic metagraphs from spaCy documents,
    serializing to/from JSON, extracting token and relation information,
    and visualizing the metagraph structure.
    """
    
    @abstractmethod
    def __init__(self, doc: Optional[Doc] = None, vert_list: Optional[List[Tuple]] = None):
        """
        Initialize from a spaCy Doc object or a list of vertices.
        
        Args:
            doc: Optional spaCy Doc object to build metagraph from
            vert_list: Optional list of vertices in metagraph format
        """
        pass
    
    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the metagraph to JSON format.
        
        Returns:
            Dictionary containing JSON-serializable metagraph data
        """
        pass
    
    @staticmethod
    @abstractmethod
    def from_json(json_data: Dict[str, Any]) -> 'ISemanticMetagraph':
        """
        Load the metagraph from JSON format.
        
        Args:
            json_data: Dictionary containing metagraph JSON data
            
        Returns:
            New SemanticMetagraph instance loaded from JSON
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_token_tags(t: Token) -> Dict[str, Any]:
        """
        Extract linguistic tags and features from a spaCy Token.
        
        Args:
            t: spaCy Token object
            
        Returns:
            Dictionary containing token case, type, and morphological features
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_dep_edges(t: Token) -> List[tuple]:
        """
        Extract dependency edges for a token.
        
        Args:
            t: spaCy Token object
            
        Returns:
            List of tuples representing dependency edges with metadata
        """
        pass
    
    @abstractmethod
    def plot(self):
        """
        Plot the semantic metagraph using networkx and matplotlib.
        
        Creates a visual representation of the metagraph with different colors
        for different node types (tokens, entity types, relations).
        """
        pass
    
    @abstractmethod
    def get_tokens(self) -> List[Dict[str, Any]]:
        """
        Get all token metaverts with their metadata.
        
        Returns:
            List of dictionaries containing token information including metavert index,
            token index, text, and metadata
        """
        pass
    
    @abstractmethod
    def get_relations(self) -> List[Dict[str, Any]]:
        """
        Get all relation metaverts.
        
        Returns:
            List of dictionaries containing relation information including type,
            source/target or nodes, relation name, and metadata
        """
        pass
    
