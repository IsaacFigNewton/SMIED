import networkx as nx
from noske.PatternLoader import PatternLoader
from noske.utils import match_chain
from typing import List, Dict

class PatternMatcher:
    """
    Enhanced pattern matcher with JSON-based configuration
    """
    
    def __init__(self, graph: nx.DiGraph, pattern_loader: PatternLoader = None):
        self.graph = graph
        self.pattern_loader = pattern_loader or PatternLoader()
    
    def find_pattern(self, category: str, pattern_name: str) -> List[List[str]]:
        """Find matches for a specific pattern"""
        pattern = self.pattern_loader.get_pattern(category, pattern_name)
        return match_chain(self.graph, pattern)
    
    def find_all_in_category(self, category: str) -> Dict[str, List[List[str]]]:
        """Find all patterns in a category"""
        results = {}
        patterns = self.pattern_loader.get_all_patterns_in_category(category)
        
        for pattern_name, pattern in patterns.items():
            results[pattern_name] = match_chain(self.graph, pattern)
        
        return results
    
    def find_all_patterns(self) -> Dict[str, Dict[str, List[List[str]]]]:
        """Find all patterns across all categories"""
        results = {}
        for category in self.pattern_loader.list_categories():
            results[category] = self.find_all_in_category(category)
        return results
    
    def get_pattern_summary(self) -> Dict[str, Dict[str, int]]:
        """Get a summary of match counts for all patterns"""
        results = {}
        for category in self.pattern_loader.list_categories():
            results[category] = {}
            patterns = self.pattern_loader.get_all_patterns_in_category(category)
            
            for pattern_name, pattern in patterns.items():
                matches = match_chain(self.graph, pattern)
                results[category][pattern_name] = len(matches)
        
        return results

# Utility functions for pattern analysis
def analyze_pattern_matches(matches: List[List[str]], pattern_name: str) -> int:
    """Analyze and summarize pattern matches"""
    print(f"\n=== {pattern_name} Analysis ===")
    print(f"Total matches found: {len(matches)}")
    
    if matches:
        print("Sample matches:")
        for i, match in enumerate(matches[:5]):  # Show first 5
            print(f"  {i+1}. {' -> '.join(match)}")
        
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more")
    
    return len(matches)

def extract_semantic_insights(graph: nx.DiGraph, pattern_loader: PatternLoader = None) -> Dict[str, int]:
    """Extract high-level semantic insights from the graph using patterns"""
    matcher = PatternMatcher(graph, pattern_loader)
    summary = matcher.get_pattern_summary()
    
    insights = {}
    for category, patterns in summary.items():
        for pattern_name, count in patterns.items():
            full_name = f"{category}.{pattern_name}"
            insights[full_name] = count
            if count > 0:
                print(f"Found {count} matches for {full_name}")
    
    return insights