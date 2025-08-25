from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional, Tuple
import networkx as nx

# Type aliases
SynsetName = str
Path = List[SynsetName]
BeamElement = Tuple[Tuple[SynsetName, str], Tuple[SynsetName, str], float]
GetNewBeamsFn = Callable[[nx.DiGraph, SynsetName, SynsetName], List[BeamElement]]


class IPairwiseBidirectionalAStar(ABC):
    """
    Interface for bidirectional A* pathfinding between synset pairs.
    
    Defines the contract for beam-constrained, gloss-seeded bidirectional A* search
    for finding paths between synset pairs in WordNet graphs.
    """
    
    @abstractmethod
    def __init__(
        self,
        g: nx.DiGraph,
        src: SynsetName,
        tgt: SynsetName,
        get_new_beams_fn: Optional[GetNewBeamsFn] = None,
        gloss_seed_nodes: Optional[Iterable[SynsetName]] = None,
        beam_width: int = 10,  # Optimized: increased from 3 to 10
        max_depth: int = 10,   # Optimized: increased from 6 to 10 
        relax_beam: bool = True,  # Optimized: changed from False to True
        heuristic_type: str = "hybrid",  # New: hybrid, embedding, wordnet, uniform
        embedding_helper: Optional[object] = None,  # For hybrid heuristic
    ):
        """
        Initialize the bidirectional A* pathfinder.
        
        Args:
            g: nx.DiGraph — graph of synsets (nodes as synset names)
            src, tgt: synset node ids (strings)
            get_new_beams_fn: function to produce embedding-based beam pairs (optional)
            gloss_seed_nodes: explicit list of synset names seeded from glosses (optional)
            beam_width: beam width for initial seeding (optimized default: 10)
            max_depth: maximum hops allowed per side (optimized default: 10)
            relax_beam: if True, allow exploring nodes outside the allowed beams (optimized default: True)
            heuristic_type: type of heuristic function ("hybrid", "embedding", "wordnet", "uniform")
            embedding_helper: embedding helper instance for hybrid heuristic calculations
        """
        pass
    
    @abstractmethod
    def find_paths(self, max_results: int = 3, len_tolerance: int = 3) -> List[Tuple[Path, float]]:
        """
        Run bidirectional beam+depth constrained search and return up to max_results unique paths.
        
        Args:
            max_results: Maximum number of paths to return
            len_tolerance: Integer extra hops allowed beyond the best (shortest) path length
            
        Returns:
            List of (path, cost) tuples where path is a list of synset names
        """
        pass