from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple
import networkx as nx

# Type aliases
SynsetName = str
Path = List[SynsetName]
BeamElement = Tuple[Tuple[SynsetName, str], Tuple[SynsetName, str], float]
GetNewBeamsFn = callable


class IPairwiseBidirectionalAStar(ABC):
    """
    Interface for bidirectional A* pathfinding between synset pairs.
    
    Defines the contract for beam-constrained, gloss-seeded bidirectional A* search
    for finding paths between synset pairs in WordNet graphs.
    """
    
    @abstractmethod
    def find_paths(self, max_results: int = 3, len_tolerance: int = 0) -> List[Tuple[Path, float]]:
        """
        Run bidirectional beam+depth constrained search and return up to max_results unique paths.
        
        Args:
            max_results: Maximum number of paths to return
            len_tolerance: Integer extra hops allowed beyond the best (shortest) path length
            
        Returns:
            List of (path, cost) tuples where path is a list of synset names
        """
        pass