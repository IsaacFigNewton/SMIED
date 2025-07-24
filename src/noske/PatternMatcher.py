from typing import List, Dict, Any

from noske.PatternLoader import PatternLoader
from noske.SemanticHypergraph import SemanticHypergraph

class PatternMatcher:
    """
    Pattern matcher with JSON-based configuration
    """
    
    def __init__(self,
                    semantic_graph: SemanticHypergraph,
                    pattern_loader: PatternLoader|None = None):
        self.semantic_graph = semantic_graph
        self.pattern_loader = pattern_loader or PatternLoader()
    
    
    def add_pattern(self,
                    name: str,
                    pattern: List[Dict[str, Any]], 
                    description: str = "",
                    category: str = "custom"):
        """Add a new pattern to the loader"""
        self.pattern_loader.add_pattern(
            name=name,
            pattern=pattern,
            description=description,
            category=category
        )


    # Enhanced helper functions with additional matching capabilities
    def node_matches(self, node_attrs: dict, pattern_attrs: dict) -> bool:
        """Node matching with additional pattern types"""
        for k, v in pattern_attrs.items():
            if k == "node_id_pattern":
                # Special pattern for matching node IDs with prefixes
                node_id = node_attrs.get("node_id", str(node_attrs.get("id", "")))
                if not node_id.startswith(v):
                    return False
            elif k == "semantic_type":
                # Match semantic types (for domain-specific patterns)
                if isinstance(v, set):
                    if node_attrs.get(k) not in v:
                        return False
                else:
                    if node_attrs.get(k) != v:
                        return False
            elif k == "lemma":
                # Match lemmatized forms
                if isinstance(v, set):
                    if node_attrs.get(k) not in v:
                        return False
                else:
                    if node_attrs.get(k) != v:
                        return False
            elif isinstance(v, set):
                if node_attrs.get(k) not in v:
                    return False
            else:
                if node_attrs.get(k) != v:
                    return False
        return True


    def edge_matches(self, edge_attrs, pattern_attrs):
        """Edge matching"""
        for k, v in pattern_attrs.items():
            if isinstance(v, set):
                if edge_attrs.get(k) not in v:
                    return False
            else:
                if edge_attrs.get(k) != v:
                    return False
        return True


    def match_chain(self, query: list[dict]) -> list[list]:
        """
        Find all paths in G matching the alternating node/edge attribute requirements
        Enhanced to work with JSON-loaded patterns
        """
        # Verify that query length is odd and > 0
        if len(query) < 1 or len(query) % 2 == 0:
            raise ValueError("Query must be non-empty and have odd length (node, edge, node, ...).")

        # Extract pattern requirements
        n = (len(query) + 1) // 2  # number of nodes in pattern
        node_patterns = [query[2*i] for i in range(n)]
        edge_patterns = [query[2*i + 1] for i in range(n-1)]

        # Find all matching paths using DFS
        results = []
        
        g = self.semantic_graph.G
        def dfs(current_path, pattern_idx):
            if pattern_idx == n:
                # We've matched all nodes in the pattern
                results.append(current_path[:])
                return
            
            if pattern_idx == 0:
                # First node - try all nodes in the graph
                for node in g.get_nodes():
                    node_data = g.get_nodes[node].copy()
                    # Add node_id for pattern matching
                    node_data["node_id"] = str(node)
                    if self.node_matches(node_data, node_patterns[0]):
                        current_path.append(node)
                        dfs(current_path, 1)
                        current_path.pop()
            else:
                # Not the first node - look for neighbors of the last node
                last_node = current_path[-1]
                edge_pattern = edge_patterns[pattern_idx - 1]
                node_pattern = node_patterns[pattern_idx]
                
                # Check all outgoing edges from the last node
                for neighbor in g.get_neighbors(last_node):
                    # Check if the edge matches the pattern
                    edge_data = g.get_edges[last_node, neighbor]
                    if self.edge_matches(edge_data, edge_pattern):
                        # Check if the neighbor node matches the pattern
                        neighbor_data = g.get_nodes[neighbor].copy()
                        neighbor_data["node_id"] = str(neighbor)
                        if self.node_matches(neighbor_data, node_pattern):
                            current_path.append(neighbor)
                            dfs(current_path, pattern_idx + 1)
                            current_path.pop()
        # Start DFS from an empty path
        dfs([], 0)
        return results
    
    
    def get_pattern_summary(self) -> Dict[str, Dict[str, int]]:
        """Get a summary of match counts for all patterns"""
        results = self()
        
        # Print insights
        insights = {}
        for category, patterns in results.items():
            for pattern_name, matches in patterns.items():
                full_name = f"{category}.{pattern_name}"
                if len(matches) > 0:
                    pattern_info = self.pattern_loader.patterns[category][pattern_name]
                    print(f"Found {len(matches)} matches for {full_name}:")
                    print(f"\tDescription: {pattern_info.get('description', 'No description')}")
                    print(f"\tPattern: {pattern_info['pattern']}")
                    for match in matches:
                        print(f"\t{match}")
                    print()

        return results
    

    # Overloaded call method for matching patterns at different granularities
    def __call__(self,
                 category:str|None = None,
                 name:str|None = None) -> Dict[str, Dict[str, List[List[str]]]]\
                                                | Dict[str, List[List[str]]]\
                                                | List[List[str]]:
        # Match all patterns across all categories
        if not category and not name:
            return {
                c: self(c)
                for c in self.pattern_loader.patterns.keys()
            }
        
        # Match all patterns in the specified category
        elif not name:
            # Check if the category exists
            if category not in self.pattern_loader.patterns.keys():
                raise KeyError(f"Category '{category}' does not exist.")
            return {
                n: self(category, n)
                for n in self.pattern_loader.patterns[category].keys()
            }
        
        # Match against a specified pattern in a specified category
        else:
            # Check if the category exists
            if category not in self.pattern_loader.patterns.keys():
                raise KeyError(f"Category '{category}' does not exist.")
            # Check if the pattern exists in the category
            if name not in self.pattern_loader.patterns[category].keys():
                raise KeyError(f"Pattern named '{name}' does not exist in category '{category}.")
            return self.match_chain(self.pattern_loader.patterns[category][name]["pattern"])