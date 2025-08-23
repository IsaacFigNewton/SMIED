from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class IGlossParser(ABC):
    """
    Interface for parsing and extraction of semantic elements from WordNet glosses.
    
    Defines the contract for extracting subjects, objects, and verbs from parsed 
    gloss documents using dependency parsing information.
    """
    
    @abstractmethod
    def parse_gloss(self, gloss_text: str, nlp_func=None) -> Optional[Dict]:
        """
        Parse a gloss text and return a dictionary with extracted semantic elements.
        
        Args:
            gloss_text: The gloss definition text to parse
            nlp_func: Optional spaCy NLP function. Uses instance nlp_func if not provided.
            
        Returns:
            Dictionary with keys 'subjects', 'objects', 'predicates' containing lists of synsets,
            or None if parsing fails.
        """
        pass
    
    @abstractmethod
    def extract_subjects_from_gloss(self, gloss_doc) -> Tuple[List, List]:
        """
        Extract subject tokens from a parsed gloss.
        
        Args:
            gloss_doc: Parsed gloss document (e.g., spaCy Doc)
            
        Returns:
            Tuple of (active_subjects, passive_subjects) token lists
        """
        pass
    
    @abstractmethod
    def extract_objects_from_gloss(self, gloss_doc) -> List:
        """
        Extract various types of object tokens from a parsed gloss.
        
        Args:
            gloss_doc: Parsed gloss document (e.g., spaCy Doc)
            
        Returns:
            List of object tokens
        """
        pass
    
    @abstractmethod
    def extract_verbs_from_gloss(self, gloss_doc, include_passive: bool = False) -> List:
        """
        Extract verb tokens from a parsed gloss.
        
        Args:
            gloss_doc: Parsed gloss document (e.g., spaCy Doc)
            include_passive: Whether to include passive verbs
            
        Returns:
            List of verb tokens
        """
        pass
    
    @abstractmethod
    def find_instrumental_verbs(self, gloss_doc) -> List:
        """
        Find verbs associated with instrumental use (e.g., 'used for').
        
        Args:
            gloss_doc: Parsed gloss document (e.g., spaCy Doc)
            
        Returns:
            List of instrumental verb tokens
        """
        pass
    
    @abstractmethod
    def get_all_neighbors(self, synset, wn_module=None) -> List:
        """
        Get all lexically related neighbors of a synset.
        
        Args:
            synset: WordNet synset object
            wn_module: WordNet module (optional)
            
        Returns:
            List of neighbor synsets
        """
        pass
    
    @abstractmethod
    def path_syn_to_syn(self, start_synset, end_synset, max_depth: int = 6, wn_module=None) -> Optional[List[str]]:
        """
        Find shortest path between synsets of the same POS using bidirectional BFS.
        
        Args:
            start_synset: Starting synset
            end_synset: Target synset
            max_depth: Maximum search depth
            wn_module: WordNet module (optional)
            
        Returns:
            List of synset names forming the path, or None if no path found
        """
        pass