from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union


class IPatternLoader(ABC):
    """
    Interface for loading and managing JSON-based semantic patterns.
    
    Defines the contract for loading patterns from files, saving patterns to files,
    converting between JSON and pattern formats, and managing pattern collections.
    """
    
    @abstractmethod
    def load_patterns_from_file(self, file_path: str) -> None:
        """
        Load patterns from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing patterns
        """
        pass
    
    @abstractmethod
    def save_patterns_to_file(self, file_path: str) -> None:
        """
        Save current patterns to a JSON file.
        
        Args:
            file_path: Path where the patterns should be saved
        """
        pass
    
    @abstractmethod
    def json_to_pattern(self) -> None:
        """
        Reformat patterns to ensure they have the correct structure.
        This is useful if patterns were loaded from a file and need to be converted.
        """
        pass
    
    @abstractmethod
    def pattern_to_json(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert patterns to a JSON-serializable format.
        This is useful for saving patterns to a file.
        
        Returns:
            Dictionary containing patterns in JSON-serializable format
        """
        pass
    
    @abstractmethod
    def add_pattern(self, name: str, pattern: List[Dict[str, Any]], 
                   description: str = "", category: str = "custom") -> None:
        """
        Add a new pattern to the loader.
        
        Args:
            name: Name of the pattern
            pattern: Pattern definition as list of dictionaries
            description: Optional description of the pattern
            category: Category to store the pattern in (defaults to "custom")
        """
        pass