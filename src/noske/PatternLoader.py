import importlib.resources
import json
from typing import Dict, List, Any, Union
import networkx as nx

class PatternLoader:
    """
    Loader for JSON-based semantic patterns
    """
    
    def __init__(self,
                 patterns_file: str = None):
        self.patterns = {}
        if patterns_file:
            self.load_patterns_from_file(patterns_file)
        else:
            self.patterns = self._get_default_patterns()

        # Reformat patterns to ensure they have the correct structure
        self.reformat_patterns()
    
    def load_patterns_from_file(self,
                                file_path: str):
        """Load patterns from a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.patterns = json.load(f)
        except FileNotFoundError:
            print(f"File {file_path} not found. Loading default patterns instead.")
            self.patterns = self._get_default_patterns()
    
    def save_patterns_to_file(self,
                              file_path: str):
        """Save current patterns to a JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, indent=2, ensure_ascii=False)
    
    def reformat_patterns(self):
        """
        Reformat patterns to ensure they have the correct structure
        This is useful if patterns were loaded from a file and need to be converted
        """
        
        # Convert Lists in JSON patterns back to sets for faster access
        def convert_pattern_from_json(json_pattern: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            converted = []
            for item in json_pattern:
                converted_item = {}
                for key, value in item.items():
                    if isinstance(value, list) and key in ["root_type", "labels", "pos"]:
                        converted_item[key] = set(value)
                    else:
                        converted_item[key] = value
                converted.append(converted_item)
            return converted

        # Apply conversion to all patterns
        for category, patterns in self.patterns.items():
            for name, pattern in patterns.items():
                if isinstance(pattern, list):
                    self.patterns[category][name] = convert_pattern_from_json(pattern)
    

    def add_pattern(self,
                    name: str,
                    pattern: List[Dict[str, Any]], 
                    description: str = "",
                    category: str = "custom"):
        """Add a new pattern to the loader"""
        if category not in self.patterns:
            self.patterns[category] = {}
        
        self.patterns[category][name] = {
            "description": description,
            "pattern": pattern
        }
    
    
    def _get_default_patterns(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Default patterns as JSON-serializable dictionary"""
        patterns = ["lexical", "simple_semantic", "complex_semantic", "domain_specific"]
        default_patterns = dict()

        for pattern in patterns:
            try:
                with importlib.resources.open_text("noske", f"./patterns/{pattern}_patterns.json") as file:
                    default_patterns[pattern] = json.load(file)
            except FileNotFoundError:
                print(f"Default patterns file for {pattern} not found. No default patterns loaded for this category.")

        return default_patterns