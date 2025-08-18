from typing import List, Dict, Any, Set, Tuple, Union

from noske.PatternLoader import PatternLoader
from noske.SemanticMetagraph import SemanticMetagraph

class PatternMatcher:
    """
    Pattern matcher with JSON-based configuration
    Enhanced to work with metavertex structure
    """
    
    def __init__(self,
                    semantic_graph: SemanticMetagraph,
                    pattern_loader: PatternLoader = None):
        self.semantic_graph = semantic_graph
        self.pattern_loader = pattern_loader or PatternLoader()
        self.use_metavertex_matching = True  # New flag for metavertex-based matching
    

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
    def metavertex_matches(self, mv_idx: int, pattern_attrs: dict) -> bool:
        """
        Check if a metavertex matches the given pattern attributes
        """
        if mv_idx >= len(self.semantic_graph.metaverts):
            return False
            
        mv = self.semantic_graph.metaverts[mv_idx]
        mv_content, mv_metadata = mv if len(mv) == 2 else (mv[0], {})
        
        # Extract attributes for matching
        mv_attrs = {
            "mv_idx": mv_idx,
            "content_type": type(mv_content).__name__,
            "content": mv_content
        }
        
        # Add metadata if present
        if mv_metadata:
            mv_attrs.update(mv_metadata)
        
        # Check for specific metavertex content types
        if isinstance(mv_content, str):
            mv_attrs["text"] = mv_content
            mv_attrs["is_atomic"] = True
        elif isinstance(mv_content, tuple) and len(mv_content) == 2:
            mv_attrs["is_directed_relation"] = True
            mv_attrs["source_idx"] = mv_content[0]
            mv_attrs["target_idx"] = mv_content[1]
            if mv_metadata and "relation" in mv_metadata:
                mv_attrs["relation"] = mv_metadata["relation"]
        elif isinstance(mv_content, list):
            mv_attrs["is_undirected_relation"] = True
            mv_attrs["component_indices"] = mv_content
            if mv_metadata and "relation" in mv_metadata:
                mv_attrs["relation"] = mv_metadata["relation"]
        
        return self.node_matches(mv_attrs, pattern_attrs)
    
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
            elif k == "mv_type":
                # Match metavertex type (atomic, directed_relation, undirected_relation)
                if v == "atomic" and not node_attrs.get("is_atomic", False):
                    return False
                elif v == "directed_relation" and not node_attrs.get("is_directed_relation", False):
                    return False
                elif v == "undirected_relation" and not node_attrs.get("is_undirected_relation", False):
                    return False
            elif k == "relation_type":
                # Match specific relation types
                if isinstance(v, set):
                    if node_attrs.get("relation") not in v:
                        return False
                else:
                    if node_attrs.get("relation") != v:
                        return False
            elif k == "references_mv":
                # Check if this metavertex references another specific metavertex
                target_idx = v
                if node_attrs.get("is_directed_relation"):
                    if target_idx not in [node_attrs.get("source_idx"), node_attrs.get("target_idx")]:
                        return False
                elif node_attrs.get("is_undirected_relation"):
                    if target_idx not in node_attrs.get("component_indices", []):
                        return False
                else:
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


    def match_metavertex_chain(self, query: list[dict]) -> list[list]:
        """
        Find sequences of metavertices matching the given pattern
        Works directly with metavertex indices and relationships
        """
        if len(query) == 0:
            return []
        
        # For single node patterns, find all matching metavertices
        if len(query) == 1:
            results = []
            for mv_idx in range(len(self.semantic_graph.metaverts)):
                if self.metavertex_matches(mv_idx, query[0]):
                    results.append([mv_idx])
            return results
        
        # For multi-node patterns, find chains of related metavertices
        results = []
        
        def find_chains(current_chain, remaining_query):
            if len(remaining_query) == 0:
                results.append(current_chain[:])
                return
            
            if len(current_chain) == 0:
                # Start with any matching metavertex
                for mv_idx in range(len(self.semantic_graph.metaverts)):
                    if self.metavertex_matches(mv_idx, remaining_query[0]):
                        find_chains([mv_idx], remaining_query[1:])
            else:
                # Look for metavertices related to the current chain
                last_mv_idx = current_chain[-1]
                next_pattern = remaining_query[0]
                
                # Find metavertices that reference the last one
                for mv_idx in range(len(self.semantic_graph.metaverts)):
                    if self.is_metavertex_related(last_mv_idx, mv_idx, next_pattern):
                        if self.metavertex_matches(mv_idx, next_pattern):
                            find_chains(current_chain + [mv_idx], remaining_query[1:])
        
        find_chains([], query)
        return results
    
    def is_metavertex_related(self, mv_idx1: int, mv_idx2: int, pattern: dict) -> bool:
        """
        Check if two metavertices are related according to the pattern
        """
        if mv_idx1 >= len(self.semantic_graph.metaverts) or mv_idx2 >= len(self.semantic_graph.metaverts):
            return False
        
        mv2 = self.semantic_graph.metaverts[mv_idx2]
        mv2_content = mv2[0] if len(mv2) > 0 else None
        
        # Check if mv2 references mv1
        if isinstance(mv2_content, tuple) and len(mv2_content) == 2:
            # Directed relation
            return mv_idx1 in mv2_content
        elif isinstance(mv2_content, list):
            # Undirected relation
            return mv_idx1 in mv2_content
        
        # Check pattern-specific relationships
        if "requires_reference" in pattern and pattern["requires_reference"]:
            return mv_idx2 > mv_idx1  # Later metavertices can reference earlier ones
        
        return True  # Default: allow any relationship
    
    def match_chain(self, query: list[dict]) -> list[list]:
        """
        Find all paths in G matching the alternating node/edge attribute requirements
        Enhanced to work with JSON-loaded patterns
        """
        # Use metavertex matching if enabled
        if self.use_metavertex_matching:
            return self.match_metavertex_chain(query)
        
        # Fallback to NetworkX-based matching
        g = self.semantic_graph.to_nx()
        
        # Verify that query length is odd and > 0
        if len(query) < 1 or len(query) % 2 == 0:
            raise ValueError("Query must be non-empty and have odd length (node, edge, node, ...).")

        # Extract pattern requirements
        n = (len(query) + 1) // 2  # number of nodes in pattern
        node_patterns = [query[2*i] for i in range(n)]
        edge_patterns = [query[2*i + 1] for i in range(n-1)]

        # Find all matching paths using DFS
        results = []
        
        def dfs(current_path, pattern_idx):
            if pattern_idx == n:
                # We've matched all nodes in the pattern
                results.append(current_path[:])
                return
            
            if pattern_idx == 0:
                # First node - try all nodes in the graph
                for node in g.nodes():
                    node_data = g.nodes[node].copy()
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
                for neighbor in g.neighbors(last_node):
                    # Check if the edge matches the pattern
                    edge_data = g.edges[last_node, neighbor]
                    if self.edge_matches(edge_data, edge_pattern):
                        # Check if the neighbor node matches the pattern
                        neighbor_data = g.nodes[neighbor].copy()
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

    def match_metavertex_pattern(self, pattern_dict: dict) -> List[List[int]]:
        """
        Match a pattern dictionary against metavertices
        Returns list of metavertex index sequences
        """
        if "pattern" not in pattern_dict:
            return []
        
        pattern = pattern_dict["pattern"]
        return self.match_metavertex_chain(pattern)
    
    def get_metavertex_context(self, mv_indices: List[int]) -> Dict[str, Any]:
        """
        Get context information for a sequence of metavertex indices
        """
        context = {
            "indices": mv_indices,
            "metaverts": [],
            "summary": ""
        }
        
        for mv_idx in mv_indices:
            if mv_idx < len(self.semantic_graph.metaverts):
                mv = self.semantic_graph.metaverts[mv_idx]
                context["metaverts"].append(mv)
                
                # Build summary
                if isinstance(mv[0], str):
                    context["summary"] += f"{mv[0]} "
                elif len(mv) > 1 and "relation" in mv[1]:
                    context["summary"] += f"[{mv[1]['relation']}] "
        
        context["summary"] = context["summary"].strip()
        return context

    # Overloaded call method for matching patterns at different granularities
    def __call__(self,
                 category: str = None,
                 pattern_name: str = None) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Enhanced pattern matching that returns metavertex information
        """
        # Match all patterns across all categories
        if not category and not pattern_name:
            results = {}
            for c in self.pattern_loader.patterns.keys():
                results[c] = self(c)
            return results
        
        # Match all patterns in the specified category
        elif not pattern_name:
            # Check if the category exists
            if category not in self.pattern_loader.patterns.keys():
                raise KeyError(f"Category '{category}' does not exist.")
            
            results = {}
            for n in self.pattern_loader.patterns[category].keys():
                results[n] = self(category, n)
            return results
        
        # Match against a specified pattern in a specified category
        else:
            # Check if the category exists
            if category not in self.pattern_loader.patterns.keys():
                raise KeyError(f"Category '{category}' does not exist.")
            # Check if the pattern exists in the category
            if pattern_name not in self.pattern_loader.patterns[category].keys():
                raise KeyError(f"Pattern named '{pattern_name}' does not exist in category '{category}'.")
            
            pattern_dict = self.pattern_loader.patterns[category][pattern_name]
            
            if self.use_metavertex_matching:
                # Return metavertex-based results with context
                mv_matches = self.match_metavertex_pattern(pattern_dict)
                return [self.get_metavertex_context(match) for match in mv_matches]
            else:
                # Return traditional NetworkX-based results
                return self.match_chain(pattern_dict["pattern"])

    def find_atomic_metavertices(self, **filters) -> List[int]:
        """
        Find atomic metavertices (string content) matching filters
        """
        results = []
        for mv_idx, mv in enumerate(self.semantic_graph.metaverts):
            if isinstance(mv[0], str):
                mv_attrs = {"text": mv[0], "mv_idx": mv_idx}
                if len(mv) > 1 and mv[1]:
                    mv_attrs.update(mv[1])
                
                # Check filters
                match = True
                for key, value in filters.items():
                    if key not in mv_attrs or mv_attrs[key] != value:
                        match = False
                        break
                
                if match:
                    results.append(mv_idx)
        
        return results
    
    def find_relation_metavertices(self, relation_type: str = None, **filters) -> List[int]:
        """
        Find relation metavertices matching criteria
        """
        results = []
        for mv_idx, mv in enumerate(self.semantic_graph.metaverts):
            mv_content, mv_metadata = mv if len(mv) == 2 else (mv[0], {})
            
            # Skip atomic metavertices
            if isinstance(mv_content, str):
                continue
            
            # Check relation type filter
            if relation_type and (not mv_metadata or mv_metadata.get("relation") != relation_type):
                continue
            
            # Check other filters
            mv_attrs = {"mv_idx": mv_idx}
            if mv_metadata:
                mv_attrs.update(mv_metadata)
            
            match = True
            for key, value in filters.items():
                if key not in mv_attrs or mv_attrs[key] != value:
                    match = False
                    break
            
            if match:
                results.append(mv_idx)
        
        return results
    
    def get_metavertex_chain(self, start_idx: int, max_depth: int = 3) -> List[List[int]]:
        """
        Get chains of metavertices starting from a given index
        """
        chains = []
        
        def build_chain(current_chain, depth):
            if depth <= 0:
                return
            
            last_idx = current_chain[-1]
            
            # Find metavertices that reference the last one
            for mv_idx in range(last_idx + 1, len(self.semantic_graph.metaverts)):
                mv = self.semantic_graph.metaverts[mv_idx]
                mv_content = mv[0]
                
                references_last = False
                if isinstance(mv_content, tuple) and len(mv_content) == 2:
                    references_last = last_idx in mv_content
                elif isinstance(mv_content, list):
                    references_last = last_idx in mv_content
                
                if references_last:
                    new_chain = current_chain + [mv_idx]
                    chains.append(new_chain[:])
                    build_chain(new_chain, depth - 1)
        
        build_chain([start_idx], max_depth)
        return chains
    
    def analyze_metavertex_patterns(self) -> Dict[str, Any]:
        """
        Analyze the metavertex structure and provide insights
        """
        analysis = {
            "total_metaverts": len(self.semantic_graph.metaverts),
            "atomic_count": 0,
            "directed_relation_count": 0,
            "undirected_relation_count": 0,
            "relation_types": {},
            "pos_distribution": {},
            "dependency_depth": 0
        }
        
        for mv_idx, mv in enumerate(self.semantic_graph.metaverts):
            mv_content, mv_metadata = mv if len(mv) == 2 else (mv[0], {})
            
            if isinstance(mv_content, str):
                analysis["atomic_count"] += 1
                if mv_metadata and "pos" in mv_metadata:
                    pos = mv_metadata["pos"]
                    analysis["pos_distribution"][pos] = analysis["pos_distribution"].get(pos, 0) + 1
            elif isinstance(mv_content, tuple) and len(mv_content) == 2:
                analysis["directed_relation_count"] += 1
                if mv_metadata and "relation" in mv_metadata:
                    rel = mv_metadata["relation"]
                    analysis["relation_types"][rel] = analysis["relation_types"].get(rel, 0) + 1
                # Calculate dependency depth
                max_ref = max(mv_content)
                analysis["dependency_depth"] = max(analysis["dependency_depth"], mv_idx - max_ref)
            elif isinstance(mv_content, list):
                analysis["undirected_relation_count"] += 1
                if mv_metadata and "relation" in mv_metadata:
                    rel = mv_metadata["relation"]
                    analysis["relation_types"][rel] = analysis["relation_types"].get(rel, 0) + 1
        
        return analysis