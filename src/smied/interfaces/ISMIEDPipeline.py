from typing import Optional, Tuple, List, Any, Dict
from abc import ABC, abstractmethod

class ISMIEDPipeline(ABC):
    """Interface for SMIED semantic analysis pipeline."""
    
    @abstractmethod
    def reinitialize(self, **kwargs) -> None:
        """Reinitialize the pipeline components with new settings."""
        pass
    
    @abstractmethod
    def analyze_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        **kwargs
    ) -> Tuple[Optional[List], Optional[List], Optional[Any]]:
        """
        Analyze a subject-predicate-object triple.
        
        Args:
            subject: Subject word
            predicate: Predicate word
            object: Object word
            **kwargs: Additional parameters for analysis
            
        Returns:
            Tuple of (subject_path, object_path, connecting_predicate)
        """
        pass
    
    @abstractmethod
    def get_synsets(self, word: str, pos: Optional[str] = None) -> List:
        """Get WordNet synsets for a word."""
        pass
    
    @abstractmethod
    def display_results(
        self,
        subject_path: Optional[List],
        object_path: Optional[List],
        connecting_predicate: Optional[Any],
        subject_word: str,
        predicate_word: str,
        object_word: str
    ) -> None:
        """Display analysis results."""
        pass