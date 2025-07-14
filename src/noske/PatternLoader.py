import json
from typing import Dict, List, Any, Union
import networkx as nx

class PatternLoader:
    """
    Loader for JSON-based semantic patterns
    """
    
    def __init__(self, patterns_file: str = None):
        self.patterns = {}
        if patterns_file:
            self.load_patterns_from_file(patterns_file)
        else:
            self.patterns = self._get_default_patterns()
    
    def load_patterns_from_file(self, file_path: str):
        """Load patterns from a JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.patterns = json.load(f)
    
    def save_patterns_to_file(self, file_path: str):
        """Save current patterns to a JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, indent=2, ensure_ascii=False)
    
    def add_pattern(self, name: str, pattern: List[Dict[str, Any]], 
                   description: str = "", category: str = "custom"):
        """Add a new pattern"""
        if category not in self.patterns:
            self.patterns[category] = {}
        
        self.patterns[category][name] = {
            "description": description,
            "pattern": pattern
        }
    
    def get_pattern(self, category: str, name: str) -> List[Dict[str, Any]]:
        """Get a specific pattern by category and name"""
        pattern_data = self.patterns.get(category, {}).get(name, {})
        return self._convert_pattern_from_json(pattern_data.get("pattern", []))
    
    def get_all_patterns_in_category(self, category: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all patterns in a category"""
        category_patterns = self.patterns.get(category, {})
        result = {}
        for name, pattern_data in category_patterns.items():
            result[name] = self._convert_pattern_from_json(pattern_data["pattern"])
        return result
    
    def list_categories(self) -> List[str]:
        """List all available pattern categories"""
        return list(self.patterns.keys())
    
    def list_patterns_in_category(self, category: str) -> List[str]:
        """List all pattern names in a category"""
        return list(self.patterns.get(category, {}).keys())
    
    def _convert_pattern_from_json(self, json_pattern: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert JSON pattern to internal format (converting lists back to sets)"""
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
    
    def _get_default_patterns(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Default patterns as JSON-serializable dictionary"""
        return self.load_patterns_from_file("./semantic_patterns.json")