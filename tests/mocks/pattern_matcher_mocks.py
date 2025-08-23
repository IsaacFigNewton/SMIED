"""
Mock classes for PatternMatcher tests.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional, Tuple
from tests.mocks.base.graph_pattern_mock import AbstractGraphPatternMock
from tests.mocks.base.edge_case_mock import AbstractEdgeCaseMock
from tests.mocks.base.integration_mock import AbstractIntegrationMock


class PatternMatcherMockFactory:
    """Factory class for creating PatternMatcher mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockPatternMatcher': MockPatternMatcher,
            'MockPatternMatcherEdgeCases': MockPatternMatcherEdgeCases,
            'MockPatternMatcherIntegration': MockPatternMatcherIntegration,
            'MockSemanticGraphForPattern': MockSemanticGraphForPattern,
            'MockPatternLoaderForPattern': MockPatternLoaderForPattern,
            'MockPattern': MockPattern,
            'MockPatternVertex': MockPatternVertex,
            'MockPatternEdge': MockPatternEdge,
            'MockVerticesCollection': MockVerticesCollection,
            'MockEdgesCollection': MockEdgesCollection,
            'MockNetworkXForPattern': MockNetworkXForPattern,
            'MockRealSemanticGraph': MockRealSemanticGraph,
            'MockRealPatternLoader': MockRealPatternLoader,
            'MockComplexPatterns': MockComplexPatterns,
            'MockMatchResult': MockMatchResult,
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


class MockPatternMatcher(Mock):
    """Mock for PatternMatcher class testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize components
        self.semantic_graph = MockSemanticGraphForPattern()
        self.pattern_loader = MockPatternLoaderForPattern()
        self.patterns = []
        
        # Set up methods
        self.load_patterns = Mock(return_value=[])
        self.match_pattern = Mock(return_value=[])
        self.find_all_matches = Mock(return_value=[])
        self.validate_pattern = Mock(return_value=True)
        self.apply_constraints = Mock(return_value=[])
        self.score_matches = Mock(return_value=[])
        
        # Internal methods
        self._match_vertices = Mock(return_value=[])
        self._match_edges = Mock(return_value=[])
        self._validate_constraints = Mock(return_value=True)
        self._compute_match_score = Mock(return_value=1.0)


class MockPatternMatcherEdgeCases(AbstractEdgeCaseMock):
    """Mock for PatternMatcher edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Preserve existing edge case scenarios
        self.empty_graph = Mock()
        self.empty_pattern = Mock()
        self.invalid_pattern = Mock(side_effect=ValueError("Invalid pattern"))
        self.no_matches = Mock(return_value=[])
        
        # Set up PatternMatcher-specific edge cases
        self._setup_pattern_matcher_edge_cases()
    
    def _setup_pattern_matcher_edge_cases(self):
        """Set up PatternMatcher-specific edge case scenarios."""
        # Pattern matching specific errors
        self.pattern_compilation_error = Mock(side_effect=RuntimeError("Pattern compilation failed"))
        self.graph_traversal_error = Mock(side_effect=RuntimeError("Graph traversal failed"))
        self.matching_timeout_error = Mock(side_effect=TimeoutError("Pattern matching timed out"))
        
        # Pattern matching specific empty/invalid responses
        self.empty_match_results = Mock(return_value=[])
        self.invalid_match_format = Mock(return_value={"invalid": "format"})
        self.malformed_pattern_data = Mock(return_value={"pattern": None})
        
        # Performance edge cases
        self.large_pattern_set = Mock(return_value=[MockPattern(f"large_{i}") for i in range(1000)])
        self.deep_recursion_pattern = Mock(side_effect=RecursionError("Pattern recursion limit exceeded"))
    
    def setup_edge_case_scenario(self, scenario_name: str) -> None:
        """Set up a specific edge case scenario for PatternMatcher."""
        self._track_scenario_execution(scenario_name)
        
        if scenario_name == "empty_graph":
            self.empty_graph.return_value = MockSemanticGraphForPattern()
            self.empty_graph.return_value.get_vertices.return_value = []
            self.empty_graph.return_value.get_edges.return_value = []
            
        elif scenario_name == "empty_pattern":
            self.empty_pattern.return_value = MockPattern("empty")
            self.empty_pattern.return_value.vertices = []
            self.empty_pattern.return_value.edges = []
            
        elif scenario_name == "invalid_pattern":
            # Already set up in __init__ with side_effect
            pass
            
        elif scenario_name == "no_matches":
            # Already set up in __init__
            pass
            
        elif scenario_name == "pattern_compilation_error":
            self.current_scenario = scenario_name
            
        elif scenario_name == "graph_traversal_error":
            self.current_scenario = scenario_name
            
        elif scenario_name == "matching_timeout":
            self.current_scenario = scenario_name
            
        elif scenario_name == "large_pattern_set":
            self.current_scenario = scenario_name
            
        elif scenario_name == "malformed_data":
            self.malformed_pattern_data.return_value = {"corrupted": True, "pattern": None}
            
        elif scenario_name == "memory_exhaustion":
            self.memory_error.side_effect = MemoryError("Pattern matching exhausted memory")
            
        else:
            # Handle unknown scenarios gracefully
            self.current_scenario = "unknown"
    
    def get_edge_case_scenarios(self) -> List[str]:
        """Get list of available edge case scenarios for PatternMatcher."""
        return [
            "empty_graph",
            "empty_pattern", 
            "invalid_pattern",
            "no_matches",
            "pattern_compilation_error",
            "graph_traversal_error",
            "matching_timeout",
            "large_pattern_set",
            "malformed_data",
            "memory_exhaustion",
            "file_not_found",
            "permission_denied",
            "json_decode_error",
            "value_error",
            "type_error"
        ]
    
    def create_failing_pattern_matcher(self, failure_type: str = "runtime_error") -> Mock:
        """Create a pattern matcher mock that fails in specific ways."""
        if failure_type == "empty_results":
            return self.create_conditional_mock(
                normal_return=[],
                edge_case_return=[],
                error_class=None
            )
        elif failure_type == "invalid_pattern":
            return self.create_conditional_mock(
                normal_return=[MockMatchResult()],
                edge_case_return=None,
                error_class=ValueError
            )
        elif failure_type == "timeout":
            return self.create_conditional_mock(
                normal_return=[MockMatchResult()],
                edge_case_return=None,
                error_class=TimeoutError
            )
        else:
            return self.create_conditional_mock(
                normal_return=[MockMatchResult()],
                edge_case_return=None,
                error_class=RuntimeError
            )
    
    def simulate_pattern_matching_failure(self, failure_count: int = 2) -> Mock:
        """Simulate intermittent pattern matching failures."""
        return self.simulate_intermittent_failure(
            success_return=[MockMatchResult(score=0.8)],
            failure_count=failure_count,
            error_class=RuntimeError
        )


class MockPatternMatcherIntegration(AbstractIntegrationMock):
    """Mock for PatternMatcher integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Preserve existing integration components
        self.real_semantic_graph = MockRealSemanticGraph()
        self.real_pattern_loader = MockRealPatternLoader()
        self.complex_patterns = MockComplexPatterns()
        
        # Set integration mode for PatternMatcher
        self.integration_mode = "full"
        
        # Initialize integration components
        self._setup_pattern_matcher_integration()
    
    def _setup_pattern_matcher_integration(self):
        """Set up PatternMatcher-specific integration components."""
        # Register core components
        self.register_component(
            "semantic_graph", 
            self.real_semantic_graph,
            dependencies=[],
            config={"graph_type": "semantic", "size": "medium"}
        )
        
        self.register_component(
            "pattern_loader",
            self.real_pattern_loader,
            dependencies=[],
            config={"loader_type": "file_based", "format": "json"}
        )
        
        self.register_component(
            "complex_patterns",
            self.complex_patterns,
            dependencies=["pattern_loader"],
            config={"pattern_complexity": "high", "variants": 3}
        )
        
        self.register_component(
            "pattern_matcher",
            MockPatternMatcher(),
            dependencies=["semantic_graph", "pattern_loader"],
            config={"matching_algorithm": "subgraph_isomorphism", "optimize": True}
        )
    
    def setup_integration_components(self) -> Dict[str, Any]:
        """Set up all components required for PatternMatcher integration testing."""
        components = {}
        
        # Set up semantic graph with realistic data
        semantic_graph = MockRealSemanticGraph()
        semantic_graph.complex_vertices = semantic_graph._create_complex_vertices()
        semantic_graph.complex_edges = semantic_graph._create_complex_edges()
        components["semantic_graph"] = semantic_graph
        
        # Set up pattern loader with complex patterns
        pattern_loader = MockRealPatternLoader()
        pattern_loader.complex_patterns = pattern_loader._create_complex_patterns()
        components["pattern_loader"] = pattern_loader
        
        # Set up complex patterns
        complex_patterns = MockComplexPatterns()
        complex_patterns.generate_pattern(10, pattern_type='mixed')
        components["complex_patterns"] = complex_patterns
        
        # Set up main pattern matcher
        pattern_matcher = MockPatternMatcher()
        pattern_matcher.semantic_graph = semantic_graph
        pattern_matcher.pattern_loader = pattern_loader
        pattern_matcher.patterns = pattern_loader.complex_patterns
        components["pattern_matcher"] = pattern_matcher
        
        # Set up supporting components
        components["match_results"] = [MockMatchResult(f"result_{i}", 0.8 - i*0.1) for i in range(5)]
        components["performance_monitor"] = Mock()
        components["validation_engine"] = Mock()
        
        return components
    
    def configure_component_interactions(self) -> None:
        """Configure how PatternMatcher components interact with each other."""
        # Configure semantic graph to pattern matcher interaction
        graph_to_matcher = self.create_component_interaction(
            "semantic_graph", "pattern_matcher", "data_flow"
        )
        graph_to_matcher.return_value = "graph_data_transferred"
        
        # Configure pattern loader to pattern matcher interaction
        loader_to_matcher = self.create_component_interaction(
            "pattern_loader", "pattern_matcher", "call"
        )
        loader_to_matcher.return_value = "patterns_loaded"
        
        # Configure complex patterns to pattern matcher interaction
        patterns_to_matcher = self.create_component_interaction(
            "complex_patterns", "pattern_matcher", "call"
        )
        patterns_to_matcher.return_value = "complex_patterns_applied"
        
        # Configure pattern matcher to results interaction
        matcher_to_results = self.create_component_interaction(
            "pattern_matcher", "match_results", "event"
        )
        matcher_to_results.return_value = "results_generated"
        
        # Set up interaction flows
        self.interactions["graph_to_matcher"] = graph_to_matcher
        self.interactions["loader_to_matcher"] = loader_to_matcher
        self.interactions["patterns_to_matcher"] = patterns_to_matcher
        self.interactions["matcher_to_results"] = matcher_to_results
    
    def validate_integration_state(self) -> bool:
        """Validate that the PatternMatcher integration is in a consistent state."""
        try:
            # Check that all required components are registered
            required_components = ["semantic_graph", "pattern_loader", "complex_patterns", "pattern_matcher"]
            for component_name in required_components:
                if component_name not in self.components:
                    return False
            
            # Validate component states
            for name, state in self.component_states.items():
                if state not in ["registered", "initialized"]:
                    return False
            
            # Validate semantic graph has data
            semantic_graph = self.get_component("semantic_graph")
            if not hasattr(semantic_graph, 'complex_vertices') or not semantic_graph.complex_vertices:
                return False
            
            # Validate pattern loader has patterns
            pattern_loader = self.get_component("pattern_loader")
            if not hasattr(pattern_loader, 'complex_patterns') or not pattern_loader.complex_patterns:
                return False
            
            # Validate complex patterns are properly configured
            complex_patterns = self.get_component("complex_patterns")
            if not complex_patterns.validate_pattern_constraints():
                return False
            
            # Validate pattern matcher has required components
            pattern_matcher = self.get_component("pattern_matcher")
            if not all(hasattr(pattern_matcher, attr) for attr in ['semantic_graph', 'pattern_loader', 'patterns']):
                return False
            
            return True
            
        except Exception:
            return False
    
    def run_integration_workflow(self) -> List[Any]:
        """Run a complete PatternMatcher integration workflow."""
        workflow_steps = [
            "initialize_semantic_graph",
            "load_patterns", 
            "configure_complex_patterns",
            "execute_pattern_matching",
            "validate_results",
            "generate_performance_report"
        ]
        
        return self.simulate_integration_workflow(workflow_steps)
    
    def initialize_semantic_graph(self) -> str:
        """Initialize the semantic graph component."""
        semantic_graph = self.get_component("semantic_graph")
        semantic_graph._create_complex_vertices()
        semantic_graph._create_complex_edges()
        return "semantic_graph_initialized"
    
    def load_patterns(self) -> str:
        """Load patterns using the pattern loader component."""
        pattern_loader = self.get_component("pattern_loader")
        pattern_loader._create_complex_patterns()
        return "patterns_loaded"
    
    def configure_complex_patterns(self) -> str:
        """Configure complex patterns component."""
        complex_patterns = self.get_component("complex_patterns")
        complex_patterns.generate_pattern(8, pattern_type='hierarchical')
        return "complex_patterns_configured"
    
    def execute_pattern_matching(self) -> str:
        """Execute pattern matching workflow."""
        pattern_matcher = self.get_component("pattern_matcher")
        # Simulate pattern matching execution
        pattern_matcher.find_all_matches.return_value = [MockMatchResult(f"match_{i}", 0.9-i*0.1) for i in range(3)]
        return "pattern_matching_executed"
    
    def validate_results(self) -> str:
        """Validate pattern matching results."""
        # Simulate result validation
        return "results_validated"
    
    def generate_performance_report(self) -> str:
        """Generate performance report for the integration."""
        summary = self.get_interaction_summary()
        return f"performance_report_generated: {summary['total_interactions']} interactions"


class MockSemanticGraphForPattern(Mock):
    """Mock SemanticGraph for PatternMatcher."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Graph structure
        self.vertices = MockVerticesCollection()
        self.edges = MockEdgesCollection()
        self.metadata = {"type": "semantic_graph"}
        
        # Methods
        self.get_vertices = Mock(return_value=[])
        self.get_edges = Mock(return_value=[])
        self.get_neighbors = Mock(return_value=[])
        self.to_nx = Mock(return_value=MockNetworkXForPattern())
        self.add_vertex = Mock()
        self.add_edge = Mock()
        self.remove_vertex = Mock()
        self.remove_edge = Mock()
        
        # Query methods
        self.find_vertices_by_type = Mock(return_value=[])
        self.find_edges_by_relation = Mock(return_value=[])
        self.get_subgraph = Mock(return_value=Mock())


class MockPatternLoaderForPattern(Mock):
    """Mock PatternLoader for PatternMatcher."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pattern loading
        self.load_patterns = Mock(return_value=[MockPattern()])
        self.load_from_file = Mock(return_value=[])
        self.load_from_string = Mock(return_value=[])
        self.validate_pattern = Mock(return_value=True)
        
        # Pattern management
        self.get_all_patterns = Mock(return_value=[])
        self.get_pattern_by_name = Mock(return_value=None)
        self.save_pattern = Mock()


class MockPattern(Mock):
    """Mock pattern object for PatternMatcher."""
    
    def __init__(self, name="test_pattern", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.vertices = [MockPatternVertex()]
        self.edges = [MockPatternEdge()]
        self.constraints = []
        self.metadata = {"priority": 1.0}
        
        # Pattern methods
        self.match = Mock(return_value=[])
        self.validate = Mock(return_value=True)
        self.get_vertex_count = Mock(return_value=len(self.vertices))
        self.get_edge_count = Mock(return_value=len(self.edges))


class MockPatternVertex(Mock):
    """Mock pattern vertex."""
    
    def __init__(self, vertex_id="v1", vertex_type="concept", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = vertex_id
        self.type = vertex_type
        self.properties = {"label": "test"}
        self.constraints = []
        
        # Vertex methods
        self.matches = Mock(return_value=True)
        self.get_property = Mock(return_value=None)
        self.set_property = Mock()


class MockPatternEdge(Mock):
    """Mock pattern edge."""
    
    def __init__(self, edge_id="e1", relation="relates_to", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = edge_id
        self.relation = relation
        self.source = "v1"
        self.target = "v2"
        self.properties = {}
        self.constraints = []
        
        # Edge methods
        self.matches = Mock(return_value=True)
        self.get_property = Mock(return_value=None)
        self.set_property = Mock()


class MockVerticesCollection(Mock):
    """Mock vertices collection for semantic graph."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vertices = []
        
        # Collection methods
        self.__iter__ = Mock(return_value=iter(self._vertices))
        self.__len__ = Mock(return_value=len(self._vertices))
        self.__getitem__ = Mock(side_effect=lambda i: self._vertices[i] if i < len(self._vertices) else None)
        self.add = Mock(side_effect=lambda v: self._vertices.append(v))
        self.remove = Mock(side_effect=lambda v: self._vertices.remove(v) if v in self._vertices else None)
        self.find_by_type = Mock(return_value=[])
        self.find_by_property = Mock(return_value=[])


class MockEdgesCollection(Mock):
    """Mock edges collection for semantic graph."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._edges = []
        
        # Collection methods
        self.__iter__ = Mock(return_value=iter(self._edges))
        self.__len__ = Mock(return_value=len(self._edges))
        self.__getitem__ = Mock(side_effect=lambda i: self._edges[i] if i < len(self._edges) else None)
        self.add = Mock(side_effect=lambda e: self._edges.append(e))
        self.remove = Mock(side_effect=lambda e: self._edges.remove(e) if e in self._edges else None)
        self.find_by_relation = Mock(return_value=[])
        self.find_by_vertices = Mock(return_value=[])


class MockNetworkXForPattern(Mock):
    """Mock NetworkX graph for PatternMatcher."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = Mock(return_value=[])
        self.edges = Mock(return_value=[])
        self.number_of_nodes = Mock(return_value=0)
        self.number_of_edges = Mock(return_value=0)
        
        # NetworkX methods
        self.subgraph = Mock(return_value=Mock())
        self.neighbors = Mock(return_value=[])
        self.predecessors = Mock(return_value=[])
        self.successors = Mock(return_value=[])


class MockRealSemanticGraph(Mock):
    """Mock representing real SemanticGraph behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # More complex structure for integration testing
        self.complex_vertices = self._create_complex_vertices()
        self.complex_edges = self._create_complex_edges()
    
    def _create_complex_vertices(self):
        """Create complex vertex structure."""
        vertices = []
        for i in range(5):
            vertex = MockPatternVertex(f"v{i}", "concept")
            vertices.append(vertex)
        return vertices
    
    def _create_complex_edges(self):
        """Create complex edge structure."""
        edges = []
        for i in range(3):
            edge = MockPatternEdge(f"e{i}", "relates_to")
            edges.append(edge)
        return edges


class MockRealPatternLoader(Mock):
    """Mock representing real PatternLoader behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complex_patterns = self._create_complex_patterns()
    
    def _create_complex_patterns(self):
        """Create complex patterns for testing."""
        patterns = []
        for i in range(3):
            pattern = MockPattern(f"complex_pattern_{i}")
            patterns.append(pattern)
        return patterns


class MockComplexPatterns(AbstractGraphPatternMock):
    """Mock complex patterns for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set pattern type for this mock
        self.pattern_type = "complex_patterns"
        
        # Complex pattern scenarios (preserve existing functionality)
        self.hierarchical_pattern = MockPattern("hierarchical")
        self.cyclic_pattern = MockPattern("cyclic")
        self.multi_constraint_pattern = MockPattern("multi_constraint")
        
        # Initialize with some complex patterns
        self._setup_complex_patterns()
    
    def _setup_complex_patterns(self):
        """Set up complex pattern structures."""
        # Add some nodes and edges to simulate complex patterns
        self.add_random_nodes(5, "complex")
        self.add_random_edges(7, allow_self_loops=True)
        
        # Set up hierarchical structure
        self.add_random_nodes(3, "hierarchical")
        
        # Set up cyclic structure  
        self.add_random_nodes(4, "cyclic")
        
        # Update pattern size
        self.pattern_size = len(self.nodes)
    
    def generate_pattern(self, size: int, **kwargs) -> None:
        """Generate complex pattern structure."""
        pattern_type = kwargs.get('pattern_type', 'mixed')
        
        # Clear existing patterns
        self.reset_pattern()
        
        if pattern_type == 'hierarchical':
            self._generate_hierarchical_pattern(size)
        elif pattern_type == 'cyclic':
            self._generate_cyclic_pattern(size)
        elif pattern_type == 'multi_constraint':
            self._generate_multi_constraint_pattern(size)
        else:
            self._generate_mixed_pattern(size)
            
        self.pattern_size = len(self.nodes)
    
    def _generate_hierarchical_pattern(self, size: int, levels: int = 3):
        """Generate hierarchical pattern structure."""
        nodes_per_level = max(1, size // levels)
        
        for level in range(levels):
            level_nodes = self.add_random_nodes(nodes_per_level, f"level_{level}")
            if level > 0:
                # Connect to previous level
                prev_level_nodes = [n for n in self.nodes if f"level_{level-1}" in n]
                for node in level_nodes:
                    if prev_level_nodes:
                        parent = prev_level_nodes[0]  # Simple hierarchy
                        self.edges.add((parent, node))
    
    def _generate_cyclic_pattern(self, size: int):
        """Generate cyclic pattern structure."""
        cycle_nodes = self.add_random_nodes(size, "cycle")
        
        # Create cycle
        for i in range(len(cycle_nodes)):
            current = cycle_nodes[i]
            next_node = cycle_nodes[(i + 1) % len(cycle_nodes)]
            self.edges.add((current, next_node))
    
    def _generate_multi_constraint_pattern(self, size: int):
        """Generate pattern with multiple constraints."""
        constraint_nodes = self.add_random_nodes(size, "constraint")
        
        # Add complex edge constraints
        for i, node in enumerate(constraint_nodes):
            self.node_attributes[node] = {'constraint_id': i, 'type': 'constraint'}
            
        # Add constraint edges
        self.add_random_edges(size * 2, allow_self_loops=False)
    
    def _generate_mixed_pattern(self, size: int):
        """Generate mixed complex pattern."""
        # Combine different pattern types
        third = size // 3
        self._generate_hierarchical_pattern(third, levels=2)
        self._generate_cyclic_pattern(third)
        self._generate_multi_constraint_pattern(size - 2 * third)
    
    def get_pattern_properties(self) -> Dict[str, Any]:
        """Get properties specific to complex patterns."""
        base_props = self.calculate_basic_properties()
        
        complex_props = {
            'has_hierarchical': any('hierarchical' in str(n) for n in self.nodes),
            'has_cyclic': any('cycle' in str(n) for n in self.nodes),
            'has_constraints': any('constraint' in str(n) for n in self.nodes),
            'complexity_score': len(self.edges) / len(self.nodes) if self.nodes else 0,
            'pattern_variants': ['hierarchical', 'cyclic', 'multi_constraint'],
            'hierarchical_pattern': self.hierarchical_pattern,
            'cyclic_pattern': self.cyclic_pattern,
            'multi_constraint_pattern': self.multi_constraint_pattern
        }
        
        return {**base_props, **complex_props}
    
    def validate_pattern_constraints(self) -> bool:
        """Validate complex pattern constraints."""
        # Check that we have at least one complex pattern type
        props = self.get_pattern_properties()
        
        has_complexity = (props['has_hierarchical'] or 
                         props['has_cyclic'] or 
                         props['has_constraints'])
        
        # Validate complexity score is reasonable
        complexity_valid = 0 <= props['complexity_score'] <= 10
        
        # Check that patterns are properly formed
        patterns_valid = all([
            hasattr(self.hierarchical_pattern, 'name'),
            hasattr(self.cyclic_pattern, 'name'),
            hasattr(self.multi_constraint_pattern, 'name')
        ])
        
        return has_complexity and complexity_valid and patterns_valid


class MockMatchResult(Mock):
    """Mock match result from pattern matching."""
    
    def __init__(self, pattern_name="test", score=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_name = pattern_name
        self.score = score
        self.vertex_mappings = {}
        self.edge_mappings = {}
        self.metadata = {}
        
        # Result methods
        self.get_mapping = Mock(return_value=None)
        self.get_score = Mock(return_value=self.score)
        self.is_valid = Mock(return_value=True)