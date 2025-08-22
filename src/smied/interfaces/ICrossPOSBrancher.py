from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class ICrossPOSBrancher(ABC):
    """
    Interface for cross-part-of-speech path finding and legacy/experimental functionality.
    
    Defines the contract for handling cross-POS search functionality and legacy methods
    that may be useful for specialized use cases or as fallback methods.
    
    This interface delegates gloss parsing functionality to a GlossParser instance.
    """
    
    @abstractmethod
    def path_syn_to_syn(self, start_synset, end_synset, max_depth: int = 6) -> Optional[List]:
        """
        Find shortest path between synsets of the same POS using bidirectional BFS.
        
        Args:
            start_synset: Starting synset
            end_synset: Target synset
            max_depth: Maximum search depth
            
        Returns:
            List of synsets forming the path, or None if no path found
        """
        pass
    
    @abstractmethod
    def get_all_neighbors(self, synset) -> List:
        """
        Get all lexically related neighbors of a synset.
        Delegates to GlossParser instance.
        
        Args:
            synset: WordNet synset object
            
        Returns:
            List of neighbor synsets
        """
        pass
    
    @abstractmethod
    def extract_subjects_from_gloss(self, gloss_doc) -> Tuple[List, List]:
        """
        Extract subject tokens from a parsed gloss.
        Delegates to GlossParser instance.
        
        Args:
            gloss_doc: Parsed gloss document
            
        Returns:
            Tuple of (active_subjects, passive_subjects) token lists
        """
        pass
    
    @abstractmethod
    def extract_objects_from_gloss(self, gloss_doc) -> List:
        """
        Extract various types of object tokens from a parsed gloss.
        Delegates to GlossParser instance.
        
        Args:
            gloss_doc: Parsed gloss document
            
        Returns:
            List of object tokens
        """
        pass
    
    @abstractmethod
    def extract_verbs_from_gloss(self, gloss_doc, include_passive: bool = False) -> List:
        """
        Extract verb tokens from a parsed gloss.
        Delegates to GlossParser instance.
        
        Args:
            gloss_doc: Parsed gloss document
            include_passive: Whether to include passive verbs
            
        Returns:
            List of verb tokens
        """
        pass
    
    @abstractmethod
    def find_instrumental_verbs(self, gloss_doc) -> List:
        """
        Find verbs associated with instrumental use (e.g., 'used for').
        Delegates to GlossParser instance.
        
        Args:
            gloss_doc: Parsed gloss document
            
        Returns:
            List of instrumental verb tokens
        """
        pass
    
    @abstractmethod
    def get_top_k_synset_branch_pairs(
        self,
        candidates: List[List],
        target_synset,
        beam_width: int = 3
    ) -> List[Tuple[Tuple, Tuple, float]]:
        """
        Get top-k synset branch pairs based on similarity to target.
        
        Args:
            candidates: List of candidate synset lists
            target_synset: Target synset for comparison
            beam_width: Maximum number of pairs to return
            
        Returns:
            List of ((synset, lexical_rel), (name, lexical_rel), relatedness) tuples
        """
        pass