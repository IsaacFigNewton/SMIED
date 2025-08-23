"""
Mock classes for MetavertexPatternMatcher tests.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional, Set, Tuple
from tests.mocks.base.graph_pattern_mock import AbstractGraphPatternMock


class MetavertexPatternMatcherMockFactory:
    """Factory class for creating MetavertexPatternMatcher mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockMetavertexPatternMatcher': MockMetavertexPatternMatcher,
            'MockMetavertexPatternIntegration': MockMetavertexPatternIntegration,
            'MockMetagraphForMatcher': MockMetagraphForMatcher,
            'MockPatternLibraryForMatcher': MockPatternLibraryForMatcher,
            'MockMatchingEngine': MockMatchingEngine,
            'MockMetavertexPattern': MockMetavertexPattern,
            'MockPatternIndex': MockPatternIndex,
            'MockMatchingStatistics': MockMatchingStatistics,
            'MockRealMetagraphForMatcher': MockRealMetagraphForMatcher,
            'MockComplexPatternsForMatcher': MockComplexPatternsForMatcher,
            'MockPerformanceOptimizer': MockPerformanceOptimizer,
            'MockMetavertexForMatcher': MockMetavertexForMatcher,
            'MockMetaedgeForMatcher': MockMetaedgeForMatcher,
            'MockTreePattern': MockTreePattern,
            'MockCyclePattern': MockCyclePattern,
            'MockCliquePattern': MockCliquePattern,
            'MockChainPattern': MockChainPattern,
        }
    
    def __call__(self, mock_name: str, *args, **kwargs) -> Mock:
        """
        Create and return a mock instance by name.
        
        Args:
            mock_name: Name of the mock class to instantiate
            *args: Arguments to pass to the mock constructor
            **kwargs: Keyword arguments to pass to the mock constructor
            
        Returns:
            Mock instance of the specified type
            
        Raises:
            ValueError: If mock_name is not found
        """
        if mock_name not in self._mock_classes:
            available = ', '.join(self._mock_classes.keys())
            raise ValueError(f"Mock '{mock_name}' not found. Available mocks: {available}")
        
        mock_class = self._mock_classes[mock_name]
        return mock_class(*args, **kwargs)
    
    def get_available_mocks(self) -> List[str]:
        """Return list of available mock class names."""
        return list(self._mock_classes.keys())


class MockMetavertexPatternMatcher(Mock):
    """Mock for MetavertexPatternMatcher class testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize pattern matching components
        self.metagraph = MockMetagraphForMatcher()
        self.pattern_library = MockPatternLibraryForMatcher()
        self.matching_engine = MockMatchingEngine()
        
        # Pattern matching state
        self.active_patterns = []
        self.match_cache = {}
        self.statistics = MockMatchingStatistics()
        
        # Set up methods
        self.match_pattern = Mock(return_value=[])
        self.match_all_patterns = Mock(return_value=[])
        self.find_subgraph_matches = Mock(return_value=[])
        self.validate_match = Mock(return_value=True)
        self.score_match = Mock(return_value=1.0)
        
        # Pattern management
        self.load_pattern = Mock()
        self.save_pattern = Mock()
        self.create_pattern = Mock(return_value=MockMetavertexPattern())
        self.delete_pattern = Mock()
        
        # Advanced matching
        self.fuzzy_match = Mock(return_value=[])
        self.approximate_match = Mock(return_value=[])
        self.structural_match = Mock(return_value=[])
        self.semantic_match = Mock(return_value=[])


class MockMetavertexPatternIntegration(Mock):
    """Mock for MetavertexPatternMatcher integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integration components
        self.real_metagraph = MockRealMetagraphForMatcher()
        self.complex_patterns = MockComplexPatternsForMatcher()
        self.performance_optimizer = MockPerformanceOptimizer()


class MockMetagraphForMatcher(Mock):
    """Mock metagraph for MetavertexPatternMatcher."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Metagraph structure
        self.metavertices = set()
        self.metaedges = set()
        self.vertex_types = set()
        self.edge_types = set()
        
        # Graph operations
        self.get_metavertices = Mock(return_value=list(self.metavertices))
        self.get_metaedges = Mock(return_value=list(self.metaedges))
        self.get_neighbors = Mock(return_value=[])
        self.get_subgraph = Mock(return_value=Mock())
        
        # Query operations
        self.find_by_type = Mock(return_value=[])
        self.find_by_property = Mock(return_value=[])
        self.find_connected_components = Mock(return_value=[])
        self.get_vertex_degree = Mock(return_value=0)
        
        # Structural properties
        self.is_connected = Mock(return_value=True)
        self.has_cycles = Mock(return_value=False)
        self.get_diameter = Mock(return_value=1)


class MockPatternLibraryForMatcher(Mock):
    """Mock pattern library for MetavertexPatternMatcher."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pattern storage
        self.patterns = {}
        self.pattern_index = MockPatternIndex()
        
        # Library operations
        self.add_pattern = Mock()
        self.remove_pattern = Mock()
        self.get_pattern = Mock(return_value=None)
        self.list_patterns = Mock(return_value=[])
        self.search_patterns = Mock(return_value=[])
        
        # Pattern organization
        self.categorize_patterns = Mock(return_value={})
        self.get_similar_patterns = Mock(return_value=[])
        self.merge_patterns = Mock(return_value=Mock())


class MockMatchingEngine(Mock):
    """Mock matching engine for MetavertexPatternMatcher."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Matching algorithms
        self.exact_match = Mock(return_value=[])
        self.inexact_match = Mock(return_value=[])
        self.partial_match = Mock(return_value=[])
        self.template_match = Mock(return_value=[])
        
        # Matching strategies
        self.greedy_matching = Mock(return_value=[])
        self.optimal_matching = Mock(return_value=[])
        self.heuristic_matching = Mock(return_value=[])
        
        # Match validation
        self.validate_mapping = Mock(return_value=True)
        self.check_constraints = Mock(return_value=True)
        self.compute_confidence = Mock(return_value=0.9)


class MockMetavertexPattern(Mock):
    """Mock metavertex pattern for pattern matching."""
    
    def __init__(self, pattern_id="pattern1", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = pattern_id
        self.name = f"Pattern {pattern_id}"
        self.description = "Test pattern"
        self.category = "test"
        
        # Pattern structure
        self.vertices = []
        self.edges = []
        self.constraints = []
        self.variables = {}
        
        # Pattern properties
        self.complexity = 1
        self.selectivity = 0.5
        self.frequency = 0.1
        
        # Pattern methods
        self.match = Mock(return_value=[])
        self.instantiate = Mock(return_value=Mock())
        self.generalize = Mock(return_value=Mock())
        self.specialize = Mock(return_value=Mock())


class MockPatternIndex(Mock):
    """Mock pattern index for efficient pattern lookup."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Index structures
        self.type_index = {}
        self.property_index = {}
        self.structure_index = {}
        
        # Index operations
        self.build_index = Mock()
        self.update_index = Mock()
        self.query_index = Mock(return_value=[])
        self.rebuild_index = Mock()


class MockMatchingStatistics(Mock):
    """Mock statistics for pattern matching performance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Statistics tracking
        self.matches_found = 0
        self.patterns_tested = 0
        self.execution_time = 0.0
        self.memory_usage = 0
        
        # Statistics methods
        self.record_match = Mock()
        self.record_pattern_test = Mock()
        self.record_execution_time = Mock()
        self.get_statistics = Mock(return_value={})
        self.reset_statistics = Mock()


class MockRealMetagraphForMatcher(Mock):
    """Mock representing real metagraph behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Real metagraph complexity
        self.large_vertex_set = self._create_large_vertex_set()
        self.complex_edge_structure = self._create_complex_edges()
        self.hierarchical_organization = self._create_hierarchy()
    
    def _create_large_vertex_set(self):
        """Create large set of metavertices."""
        vertices = []
        for i in range(1000):
            vertex = MockMetavertexForMatcher(f"mv_{i}")
            vertices.append(vertex)
        return vertices
    
    def _create_complex_edges(self):
        """Create complex edge structure."""
        edges = []
        for i in range(2000):
            edge = MockMetaedgeForMatcher(f"me_{i}", f"mv_{i%100}", f"mv_{(i+1)%100}")
            edges.append(edge)
        return edges
    
    def _create_hierarchy(self):
        """Create hierarchical organization."""
        hierarchy = {}
        for level in range(5):
            hierarchy[level] = [f"mv_{level}_{i}" for i in range(20)]
        return hierarchy


class MockComplexPatternsForMatcher(Mock):
    """Mock complex patterns for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Complex pattern types
        self.tree_pattern = MockTreePattern()
        self.cycle_pattern = MockCyclePattern()
        self.clique_pattern = MockCliquePattern()
        self.chain_pattern = MockChainPattern()


class MockPerformanceOptimizer(Mock):
    """Mock performance optimizer for pattern matching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optimization strategies
        self.optimize_query_plan = Mock()
        self.prune_search_space = Mock()
        self.parallelize_matching = Mock()
        self.cache_results = Mock()
        
        # Performance monitoring
        self.monitor_performance = Mock()
        self.identify_bottlenecks = Mock(return_value=[])
        self.suggest_optimizations = Mock(return_value=[])


class MockMetavertexForMatcher(Mock):
    """Mock metavertex for pattern matching."""
    
    def __init__(self, vertex_id="mv1", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = vertex_id
        self.type = "concept"
        self.properties = {"label": f"Vertex {vertex_id}"}
        self.metadata = {}
        
        # Vertex operations
        self.get_property = Mock(return_value=None)
        self.set_property = Mock()
        self.has_property = Mock(return_value=False)
        self.matches = Mock(return_value=False)


class MockMetaedgeForMatcher(Mock):
    """Mock metaedge for pattern matching."""
    
    def __init__(self, edge_id="me1", source="mv1", target="mv2", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = edge_id
        self.source = source
        self.target = target
        self.type = "relates_to"
        self.properties = {}
        self.metadata = {}
        
        # Edge operations
        self.get_property = Mock(return_value=None)
        self.set_property = Mock()
        self.has_property = Mock(return_value=False)
        self.matches = Mock(return_value=False)


class MockTreePattern(AbstractGraphPatternMock):
    """Mock tree pattern for complex pattern matching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_type = "tree"
        self.root = "root"
        self.children = {"root": ["child1", "child2"]}
        self.depth = 3
        # Initialize tree structure
        self._setup_tree_structure()
    
    def _setup_tree_structure(self):
        """Set up the tree structure with nodes and edges."""
        # Add root node
        self.nodes.add(self.root)
        self.node_attributes[self.root] = {"level": 0, "is_root": True}
        self.adjacency_list[self.root] = set()
        
        # Add children and create edges
        for parent, child_list in self.children.items():
            for child in child_list:
                self.nodes.add(child)
                self.node_attributes[child] = {"level": 1, "parent": parent}
                edge = (parent, child)
                self.edges.add(edge)
                self.edge_attributes[edge] = {"type": "parent_child"}
                
                # Update adjacency list
                if parent not in self.adjacency_list:
                    self.adjacency_list[parent] = set()
                if child not in self.adjacency_list:
                    self.adjacency_list[child] = set()
                self.adjacency_list[parent].add(child)
                self.adjacency_list[child].add(parent)  # Undirected tree
        
        self.pattern_size = len(self.nodes)
    
    def generate_pattern(self, size: int, **kwargs) -> None:
        """Generate a tree pattern of specified size."""
        self.reset_pattern()
        max_depth = kwargs.get('max_depth', size)
        
        # Create root
        self.root = "root"
        self.nodes.add(self.root)
        self.node_attributes[self.root] = {"level": 0, "is_root": True}
        self.adjacency_list[self.root] = set()
        
        nodes_created = 1
        current_level_nodes = [self.root]
        level = 0
        
        while nodes_created < size and level < max_depth:
            next_level_nodes = []
            for parent in current_level_nodes:
                if nodes_created >= size:
                    break
                # Add 1-3 children per node
                num_children = min(3, size - nodes_created)
                if num_children <= 0:
                    break
                    
                for i in range(num_children):
                    child = f"node_{level+1}_{i}_{parent}"
                    self.nodes.add(child)
                    self.node_attributes[child] = {"level": level + 1, "parent": parent}
                    
                    # Create edge
                    edge = (parent, child)
                    self.edges.add(edge)
                    self.edge_attributes[edge] = {"type": "parent_child"}
                    
                    # Update adjacency
                    if parent not in self.adjacency_list:
                        self.adjacency_list[parent] = set()
                    if child not in self.adjacency_list:
                        self.adjacency_list[child] = set()
                    self.adjacency_list[parent].add(child)
                    self.adjacency_list[child].add(parent)
                    
                    next_level_nodes.append(child)
                    nodes_created += 1
                    
                    if nodes_created >= size:
                        break
            
            current_level_nodes = next_level_nodes
            level += 1
        
        self.depth = level
        self.pattern_size = len(self.nodes)
    
    def get_pattern_properties(self) -> Dict[str, Any]:
        """Get tree-specific properties."""
        base_props = self.calculate_basic_properties()
        tree_props = {
            "root_node": self.root,
            "tree_depth": self.depth,
            "is_binary_tree": all(len(children) <= 2 for children in self.children.values()),
            "leaf_nodes": [node for node in self.nodes 
                          if len(self.adjacency_list.get(node, set())) == 1 and node != self.root],
            "internal_nodes": [node for node in self.nodes 
                              if len(self.adjacency_list.get(node, set())) > 1]
        }
        return {**base_props, **tree_props}
    
    def validate_pattern_constraints(self) -> bool:
        """Validate tree constraints: connected, acyclic, single root."""
        if len(self.nodes) == 0:
            return True
        
        # Check if it's connected (all nodes reachable from root)
        if not self._is_connected_tree():
            return False
        
        # Check if it's acyclic (tree property)
        if self._has_cycles():
            return False
        
        # Check single root (node with no parent except itself)
        root_count = sum(1 for node, attrs in self.node_attributes.items() 
                        if attrs.get('is_root', False))
        return root_count == 1
    
    def _is_connected_tree(self) -> bool:
        """Check if the tree is connected."""
        if not self.nodes:
            return True
        
        visited = set()
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.adjacency_list.get(node, set()) - visited)
        
        return len(visited) == len(self.nodes)
    
    def _has_cycles(self) -> bool:
        """Check if the graph has cycles (should be False for trees)."""
        # For a tree: |edges| = |nodes| - 1
        return len(self.edges) != len(self.nodes) - 1


class MockCyclePattern(AbstractGraphPatternMock):
    """Mock cycle pattern for complex pattern matching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_type = "cycle"
        self.cycle_nodes = ["n1", "n2", "n3"]
        self.cycle_length = 3
        # Initialize cycle structure
        self._setup_cycle_structure()
    
    def _setup_cycle_structure(self):
        """Set up the cycle structure with nodes and edges."""
        # Add nodes
        for i, node in enumerate(self.cycle_nodes):
            self.nodes.add(node)
            self.node_attributes[node] = {"position": i}
            self.adjacency_list[node] = set()
        
        # Add edges to form cycle
        for i in range(len(self.cycle_nodes)):
            current = self.cycle_nodes[i]
            next_node = self.cycle_nodes[(i + 1) % len(self.cycle_nodes)]
            edge = (current, next_node)
            self.edges.add(edge)
            self.edge_attributes[edge] = {"type": "cycle_edge", "position": i}
            
            # Update adjacency list
            self.adjacency_list[current].add(next_node)
            self.adjacency_list[next_node].add(current)  # Undirected cycle
        
        self.pattern_size = len(self.nodes)
    
    def generate_pattern(self, size: int, **kwargs) -> None:
        """Generate a cycle pattern of specified size."""
        self.reset_pattern()
        
        if size < 3:
            raise ValueError("Cycle must have at least 3 nodes")
        
        # Create nodes
        self.cycle_nodes = [f"cycle_node_{i}" for i in range(size)]
        for i, node in enumerate(self.cycle_nodes):
            self.nodes.add(node)
            self.node_attributes[node] = {"position": i}
            self.adjacency_list[node] = set()
        
        # Create cycle edges
        for i in range(size):
            current = self.cycle_nodes[i]
            next_node = self.cycle_nodes[(i + 1) % size]
            edge = (current, next_node)
            self.edges.add(edge)
            self.edge_attributes[edge] = {"type": "cycle_edge", "position": i}
            
            # Update adjacency
            self.adjacency_list[current].add(next_node)
            self.adjacency_list[next_node].add(current)
        
        self.cycle_length = size
        self.pattern_size = size
    
    def get_pattern_properties(self) -> Dict[str, Any]:
        """Get cycle-specific properties."""
        base_props = self.calculate_basic_properties()
        cycle_props = {
            "cycle_length": self.cycle_length,
            "cycle_nodes": self.cycle_nodes.copy(),
            "is_simple_cycle": len(self.cycle_nodes) == self.cycle_length,
            "girth": self.cycle_length  # Length of shortest cycle
        }
        return {**base_props, **cycle_props}
    
    def validate_pattern_constraints(self) -> bool:
        """Validate cycle constraints: each node has degree 2, forms single cycle."""
        if len(self.nodes) < 3:
            return False
        
        # Check that each node has exactly degree 2
        for node in self.nodes:
            degree = len(self.adjacency_list.get(node, set()))
            if degree != 2:
                return False
        
        # Check that it forms exactly one cycle
        return self._forms_single_cycle()
    
    def _forms_single_cycle(self) -> bool:
        """Check if the graph forms exactly one cycle."""
        # For a single cycle: |edges| = |nodes| and connected
        if len(self.edges) != len(self.nodes):
            return False
        
        # Check connectivity by traversing from any node
        if not self.nodes:
            return True
        
        visited = set()
        start_node = next(iter(self.nodes))
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.adjacency_list.get(node, set()) - visited)
        
        return len(visited) == len(self.nodes)


class MockCliquePattern(AbstractGraphPatternMock):
    """Mock clique pattern for complex pattern matching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_type = "clique"
        self.clique_nodes = ["c1", "c2", "c3", "c4"]
        self.clique_size = 4
        # Initialize clique structure
        self._setup_clique_structure()
    
    def _setup_clique_structure(self):
        """Set up the clique structure with nodes and edges."""
        # Add nodes
        for node in self.clique_nodes:
            self.nodes.add(node)
            self.node_attributes[node] = {"clique_member": True}
            self.adjacency_list[node] = set()
        
        # Add all possible edges (complete graph)
        for i, node1 in enumerate(self.clique_nodes):
            for j, node2 in enumerate(self.clique_nodes):
                if i < j:  # Avoid duplicate edges in undirected graph
                    edge = (node1, node2)
                    self.edges.add(edge)
                    self.edge_attributes[edge] = {"type": "clique_edge"}
                    
                    # Update adjacency list
                    self.adjacency_list[node1].add(node2)
                    self.adjacency_list[node2].add(node1)
        
        self.pattern_size = len(self.nodes)
    
    def generate_pattern(self, size: int, **kwargs) -> None:
        """Generate a clique pattern of specified size."""
        self.reset_pattern()
        
        if size < 1:
            raise ValueError("Clique must have at least 1 node")
        
        # Create nodes
        self.clique_nodes = [f"clique_node_{i}" for i in range(size)]
        for node in self.clique_nodes:
            self.nodes.add(node)
            self.node_attributes[node] = {"clique_member": True}
            self.adjacency_list[node] = set()
        
        # Create all possible edges (complete graph)
        for i, node1 in enumerate(self.clique_nodes):
            for j, node2 in enumerate(self.clique_nodes):
                if i < j:
                    edge = (node1, node2)
                    self.edges.add(edge)
                    self.edge_attributes[edge] = {"type": "clique_edge"}
                    
                    # Update adjacency
                    self.adjacency_list[node1].add(node2)
                    self.adjacency_list[node2].add(node1)
        
        self.clique_size = size
        self.pattern_size = size
    
    def get_pattern_properties(self) -> Dict[str, Any]:
        """Get clique-specific properties."""
        base_props = self.calculate_basic_properties()
        clique_props = {
            "clique_size": self.clique_size,
            "clique_nodes": self.clique_nodes.copy(),
            "is_complete": self._is_complete_graph(),
            "expected_edges": self.clique_size * (self.clique_size - 1) // 2,
            "chromatic_number": self.clique_size  # Clique chromatic number equals size
        }
        return {**base_props, **clique_props}
    
    def validate_pattern_constraints(self) -> bool:
        """Validate clique constraints: complete graph, all nodes connected to all others."""
        if len(self.nodes) == 0:
            return True
        
        # Check that it's a complete graph
        if not self._is_complete_graph():
            return False
        
        # Check that each node is connected to all other nodes
        for node in self.nodes:
            expected_neighbors = self.nodes - {node}
            actual_neighbors = self.adjacency_list.get(node, set())
            if actual_neighbors != expected_neighbors:
                return False
        
        return True
    
    def _is_complete_graph(self) -> bool:
        """Check if the graph is complete (all possible edges exist)."""
        n = len(self.nodes)
        if n <= 1:
            return True
        
        expected_edges = n * (n - 1) // 2
        return len(self.edges) == expected_edges


class MockChainPattern(AbstractGraphPatternMock):
    """Mock chain pattern for complex pattern matching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_type = "chain"
        self.chain_nodes = ["start", "n1", "n2", "n3", "end"]
        self.chain_length = 5
        # Initialize chain structure
        self._setup_chain_structure()
    
    def _setup_chain_structure(self):
        """Set up the chain structure with nodes and edges."""
        # Add nodes
        for i, node in enumerate(self.chain_nodes):
            self.nodes.add(node)
            self.node_attributes[node] = {
                "position": i,
                "is_start": i == 0,
                "is_end": i == len(self.chain_nodes) - 1
            }
            self.adjacency_list[node] = set()
        
        # Add edges to form chain
        for i in range(len(self.chain_nodes) - 1):
            current = self.chain_nodes[i]
            next_node = self.chain_nodes[i + 1]
            edge = (current, next_node)
            self.edges.add(edge)
            self.edge_attributes[edge] = {"type": "chain_edge", "position": i}
            
            # Update adjacency list
            self.adjacency_list[current].add(next_node)
            self.adjacency_list[next_node].add(current)  # Undirected chain
        
        self.pattern_size = len(self.nodes)
    
    def generate_pattern(self, size: int, **kwargs) -> None:
        """Generate a chain pattern of specified size."""
        self.reset_pattern()
        
        if size < 1:
            raise ValueError("Chain must have at least 1 node")
        
        # Create nodes
        self.chain_nodes = [f"chain_node_{i}" for i in range(size)]
        for i, node in enumerate(self.chain_nodes):
            self.nodes.add(node)
            self.node_attributes[node] = {
                "position": i,
                "is_start": i == 0,
                "is_end": i == size - 1
            }
            self.adjacency_list[node] = set()
        
        # Create chain edges
        for i in range(size - 1):
            current = self.chain_nodes[i]
            next_node = self.chain_nodes[i + 1]
            edge = (current, next_node)
            self.edges.add(edge)
            self.edge_attributes[edge] = {"type": "chain_edge", "position": i}
            
            # Update adjacency
            self.adjacency_list[current].add(next_node)
            self.adjacency_list[next_node].add(current)
        
        self.chain_length = size
        self.pattern_size = size
    
    def get_pattern_properties(self) -> Dict[str, Any]:
        """Get chain-specific properties."""
        base_props = self.calculate_basic_properties()
        chain_props = {
            "chain_length": self.chain_length,
            "chain_nodes": self.chain_nodes.copy(),
            "start_node": self.chain_nodes[0] if self.chain_nodes else None,
            "end_node": self.chain_nodes[-1] if self.chain_nodes else None,
            "is_path": self._is_simple_path(),
            "diameter": len(self.chain_nodes) - 1 if len(self.chain_nodes) > 1 else 0
        }
        return {**base_props, **chain_props}
    
    def validate_pattern_constraints(self) -> bool:
        """Validate chain constraints: forms a simple path, no cycles."""
        if len(self.nodes) == 0:
            return True
        
        # Check that it forms a simple path
        if not self._is_simple_path():
            return False
        
        # Check connectivity
        if not self._is_connected_path():
            return False
        
        # Check no cycles (for chain: |edges| = |nodes| - 1)
        return len(self.edges) == len(self.nodes) - 1
    
    def _is_simple_path(self) -> bool:
        """Check if the graph forms a simple path."""
        if len(self.nodes) <= 1:
            return True
        
        # Count nodes by degree
        degree_count = {}
        for node in self.nodes:
            degree = len(self.adjacency_list.get(node, set()))
            if degree not in degree_count:
                degree_count[degree] = 0
            degree_count[degree] += 1
        
        # For a path: exactly 2 nodes with degree 1 (endpoints)
        # and all other nodes with degree 2
        if len(self.nodes) == 1:
            return degree_count.get(0, 0) == 1
        elif len(self.nodes) == 2:
            return degree_count.get(1, 0) == 2
        else:
            return (degree_count.get(1, 0) == 2 and 
                   degree_count.get(2, 0) == len(self.nodes) - 2)
    
    def _is_connected_path(self) -> bool:
        """Check if the chain is connected."""
        if not self.nodes:
            return True
        
        visited = set()
        start_node = next(iter(self.nodes))
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.adjacency_list.get(node, set()) - visited)
        
        return len(visited) == len(self.nodes)