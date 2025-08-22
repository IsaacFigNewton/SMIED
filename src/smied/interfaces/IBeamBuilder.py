from abc import ABC, abstractmethod
from typing import List, Tuple
import networkx as nx


class IBeamBuilder(ABC):
    """
    Interface for constructing embedding-based beams for synset pair searches.
    
    Defines the contract for building beams by analyzing lexical relations 
    between synsets using embedding similarities.
    """
    
    @abstractmethod
    def get_new_beams(
        self,
        g: nx.DiGraph,
        src: str,
        tgt: str,
        model,
        beam_width: int = 3
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str], float]]:
        """
        Get the k closest pairs of lexical relations between 2 synsets.

        Args:
            g: NetworkX graph of synsets
            src: Source synset name (e.g., 'dog.n.01')
            tgt: Target synset name (e.g., 'cat.n.01')
            model: Embedding model
            beam_width: max number of pairs to return

        Returns:
            List of tuples of the form:
              (
                (synset1, lexical_rel),
                (synset2, lexical_rel),
                relatedness
              )
        """
        pass