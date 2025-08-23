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
        beam_width: int = 3,
        max_depth: int = 6,
        relax_beam: bool = False,
    ):
        """
        Initialize the bidirectional A* pathfinder.
        
        Args:
            g: nx.DiGraph — graph of synsets (nodes as synset names)
            src, tgt: synset node ids (strings)
            get_new_beams_fn: function to produce embedding-based beam pairs (optional)
            gloss_seed_nodes: explicit list of synset names seeded from glosses (optional)
            beam_width: beam width for initial seeding (passed to get_new_beams if used)
            max_depth: maximum hops allowed (total across both sides; enforced per side)
            relax_beam: if True, allow exploring nodes outside the allowed beams
        """
        pass
    
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