from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Tuple, Union, Optional
from smied.PatternLoader import PatternLoader
from smied.SemanticMetagraph import SemanticMetagraph


class IPatternMatcher(ABC):
    """
    Interface for pattern matching with JSON-based configuration.
    
    Defines the contract for pattern matching operations enhanced to work with 
    metavertex structure, including pattern registration, node/edge matching,
    metavertex chain matching, and semantic analysis.
    """
    
    @abstractmethod
    def add_pattern(self,
                    name: str,
                    pattern: List[Dict[str, Any]], 
                    description: str = "",
                    category: str = "custom") -> None:
        """
        Add a new pattern to the loader.
        
        Args:
            name: Pattern name identifier
            pattern: List of pattern dictionaries defining the matching criteria
            description: Optional pattern description
            category: Pattern category for organization
        """
        pass

    @abstractmethod
    def metavertex_matches(self, mv_idx: int, pattern_attrs: dict) -> bool:
        """
        Check if a metavertex matches the given pattern attributes.
        
        Args:
            mv_idx: Metavertex index to check
            pattern_attrs: Dictionary of pattern attributes to match against
            
        Returns:
            True if metavertex matches the pattern, False otherwise
        """
        pass
    
    @abstractmethod
    def node_matches(self, node_attrs: dict, pattern_attrs: dict) -> bool:
        """
        Node matching with additional pattern types.
        
        Args:
            node_attrs: Dictionary of node attributes
            pattern_attrs: Dictionary of pattern attributes to match against
            
        Returns:
            True if node matches the pattern, False otherwise
        """
        pass

    @abstractmethod
    def edge_matches(self, edge_attrs: dict, pattern_attrs: dict) -> bool:
        """
        Edge matching against pattern attributes.
        
        Args:
            edge_attrs: Dictionary of edge attributes
            pattern_attrs: Dictionary of pattern attributes to match against
            
        Returns:
            True if edge matches the pattern, False otherwise
        """
        pass

    @abstractmethod
    def match_metavertex_chain(self, query: List[dict]) -> List[List[int]]:
        """
        Find sequences of metavertices matching the given pattern.
        Works directly with metavertex indices and relationships.
        
        Args:
            query: List of pattern dictionaries defining the chain to match
            
        Returns:
            List of metavertex index sequences that match the pattern
        """
        pass

    @abstractmethod
    def is_metavertex_related(self, mv_idx1: int, mv_idx2: int, pattern: dict) -> bool:
        """
        Check if two metavertices are related according to the pattern.
        
        Args:
            mv_idx1: First metavertex index
            mv_idx2: Second metavertex index
            pattern: Pattern dictionary defining relationship criteria
            
        Returns:
            True if metavertices are related per pattern, False otherwise
        """
        pass
    
    @abstractmethod
    def match_chain(self, query: List[dict]) -> List[List]:
        """
        Find all paths in graph matching the alternating node/edge attribute requirements.
        Enhanced to work with JSON-loaded patterns.
        
        Args:
            query: List of alternating node/edge pattern dictionaries
            
        Returns:
            List of paths matching the chain pattern
        """
        pass

    @abstractmethod
    def get_pattern_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get a summary of match counts for all patterns.
        
        Returns:
            Nested dictionary with category -> pattern_name -> match_count structure
        """
        pass

    @abstractmethod
    def match_metavertex_pattern(self, pattern_dict: dict) -> List[List[int]]:
        """
        Match a pattern dictionary against metavertices.
        
        Args:
            pattern_dict: Dictionary containing pattern definition
            
        Returns:
            List of metavertex index sequences matching the pattern
        """
        pass

    @abstractmethod
    def get_metavertex_context(self, mv_indices: List[int]) -> Dict[str, Any]:
        """
        Get context information for a sequence of metavertex indices.
        
        Args:
            mv_indices: List of metavertex indices
            
        Returns:
            Dictionary containing context information including indices, metaverts, and summary
        """
        pass

    @abstractmethod
    def __call__(self,
                 category: str = None,
                 pattern_name: str = None) -> Union[Dict[str, Dict[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
        """
        Enhanced pattern matching that returns metavertex information.
        
        Args:
            category: Optional category to match patterns from
            pattern_name: Optional specific pattern name to match
            
        Returns:
            Pattern matching results - structure depends on parameters provided
        """
        pass

    @abstractmethod
    def find_atomic_metavertices(self, **filters) -> List[int]:
        """
        Find atomic metavertices (string content) matching filters.
        
        Args:
            **filters: Keyword arguments for filtering metavertices
            
        Returns:
            List of metavertex indices for atomic metavertices matching filters
        """
        pass
    
    @abstractmethod
    def find_relation_metavertices(self, relation_type: str = None, **filters) -> List[int]:
        """
        Find relation metavertices matching criteria.
        
        Args:
            relation_type: Optional specific relation type to match
            **filters: Additional keyword arguments for filtering
            
        Returns:
            List of metavertex indices for relation metavertices matching criteria
        """
        pass
    
    @abstractmethod
    def get_metavertex_chain(self, start_idx: int, max_depth: int = 3) -> List[List[int]]:
        """
        Get chains of metavertices starting from a given index.
        
        Args:
            start_idx: Starting metavertex index
            max_depth: Maximum chain depth to explore
            
        Returns:
            List of metavertex chains starting from the given index
        """
        pass
    
    @abstractmethod
    def analyze_metavertex_patterns(self) -> Dict[str, Any]:
        """
        Analyze the metavertex structure and provide insights.
        
        Returns:
            Dictionary containing analysis results including counts, distributions, and metrics
        """
        pass