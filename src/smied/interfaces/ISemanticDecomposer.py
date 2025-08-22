from abc import ABC, abstractmethod
from typing import Optional, Tuple
import networkx as nx


class ISemanticDecomposer(ABC):
    """
    Interface for semantic decomposition using WordNet paths.
    
    Defines the contract for the main orchestrator that integrates pathfinding, 
    beam building, embedding analysis, and gloss parsing to find semantic paths 
    between subject-predicate-object triples.
    """
    
    @abstractmethod
    def find_connected_shortest_paths(
        self,
        subject_word: str,
        predicate_word: str,
        object_word: str,
        model=None,
        g: nx.DiGraph = None,
        max_depth: int = 10,
        max_self_intersection: int = 5,
        beam_width: int = 3,
        max_results_per_pair: int = 3,
        len_tolerance: int = 1,
        relax_beam: bool = False
    ) -> Tuple[Optional[list], Optional[list], Optional[object]]:
        """
        Main entry point for finding semantic paths between subject-predicate-object triples.
        
        Args:
            subject_word: Subject word string
            predicate_word: Predicate word string  
            object_word: Object word string
            model: Optional embedding model
            g: Optional pre-built synset graph
            max_depth: Maximum search depth
            max_self_intersection: Maximum allowed path intersections
            beam_width: Beam width for search
            max_results_per_pair: Maximum results per synset pair
            len_tolerance: Length tolerance for path selection
            relax_beam: Whether to relax beam constraints
            
        Returns:
            Tuple of (best_subject_path, best_object_path, best_predicate) or (None, None, None)
        """
        pass
    
    @abstractmethod
    def build_synset_graph(self) -> nx.DiGraph:
        """
        Build a directed graph of synsets with their lexical relations.
        
        Returns:
            NetworkX DiGraph of synsets with relation edges
        """
        pass