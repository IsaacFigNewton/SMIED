"""
Mock classes for PairwiseBidirectionalAStar tests.
"""

from unittest.mock import Mock
import heapq
from typing import List, Dict, Any, Optional, Tuple, Set
from tests.mocks.base.algorithmic_function_mock import AbstractAlgorithmicFunctionMock
from tests.mocks.base.edge_case_mock import AbstractEdgeCaseMock
from tests.mocks.base.integration_mock import AbstractIntegrationMock


class PairwiseBidirectionalAStarMockFactory:
    """Factory class for creating PairwiseBidirectionalAStar mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockPairwiseBidirectionalAStar': MockPairwiseBidirectionalAStar,
            'MockPairwiseBidirectionalAStarEdgeCases': MockPairwiseBidirectionalAStarEdgeCases,
            'MockPairwiseBidirectionalAStarIntegration': MockPairwiseBidirectionalAStarIntegration,
            'MockGraphForPathfinding': MockGraphForPathfinding,
            'MockHeuristicFunction': MockHeuristicFunction,
            'MockCostFunction': MockCostFunction,
            'MockPriorityQueue': MockPriorityQueue,
            'MockSearchNode': MockSearchNode,
            'MockPathResult': MockPathResult,
            'MockRealGraphForPathfinding': MockRealGraphForPathfinding,
            'MockPerformanceMetrics': MockPerformanceMetrics,
            'MockMemoryManager': MockMemoryManager,
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


class MockPairwiseBidirectionalAStar(Mock):
    """Mock for PairwiseBidirectionalAStar class testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize pathfinding components
        self.graph = MockGraphForPathfinding()
        self.heuristic = MockHeuristicFunction()
        self.cost_function = MockCostFunction()
        
        # Pathfinding state
        self.forward_frontier = MockPriorityQueue()
        self.backward_frontier = MockPriorityQueue()
        self.forward_visited = set()
        self.backward_visited = set()
        self.forward_came_from = {}
        self.backward_came_from = {}
        
        # Set up methods
        self.find_path = Mock(return_value=[])
        self.find_paths = Mock(return_value=[])
        self.find_shortest_path = Mock(return_value=[])
        self.find_k_shortest_paths = Mock(return_value=[])
        
        # Algorithm components
        self.bidirectional_search = Mock(return_value=[])
        self.forward_search = Mock(return_value=[])
        self.backward_search = Mock(return_value=[])
        self.reconstruct_path = Mock(return_value=[])
        
        # Optimization methods
        self.prune_search_space = Mock()
        self.update_bounds = Mock()
        self.check_termination = Mock(return_value=False)


class MockPairwiseBidirectionalAStarEdgeCases(AbstractEdgeCaseMock):
    """Mock for PairwiseBidirectionalAStar edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Edge case scenarios specific to pathfinding
        self.no_path_exists = Mock(return_value=[])
        self.source_equals_target = Mock(return_value=["source"])
        self.disconnected_graph = Mock(return_value=[])
        self.infinite_cost = Mock(return_value=[])
        self.negative_weights = Mock(side_effect=ValueError("Negative weights not supported"))
    
    def setup_edge_case_scenario(self, scenario_name: str) -> None:
        """Set up specific edge case scenario for pathfinding."""
        if scenario_name == "no_path":
            self.find_path = self.no_path_exists
        elif scenario_name == "same_source_target":
            self.find_path = self.source_equals_target
        elif scenario_name == "disconnected_graph":
            self.find_path = self.disconnected_graph
        elif scenario_name == "infinite_cost":
            self.find_path = self.infinite_cost
        elif scenario_name == "negative_weights":
            self.find_path = self.negative_weights
        elif scenario_name == "empty_graph":
            self.find_path = self.return_empty_list
        elif scenario_name == "invalid_nodes":
            self.find_path = self.key_error
        else:
            raise ValueError(f"Unknown edge case scenario: {scenario_name}")
    
    def get_edge_case_scenarios(self) -> List[str]:
        """Get list of available pathfinding edge case scenarios."""
        return [
            "no_path",
            "same_source_target",
            "disconnected_graph", 
            "infinite_cost",
            "negative_weights",
            "empty_graph",
            "invalid_nodes"
        ]


class MockPairwiseBidirectionalAStarIntegration(AbstractIntegrationMock):
    """Mock for PairwiseBidirectionalAStar integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set integration mode for pathfinding
        self.integration_mode = "pathfinding"
        
        # Initialize integration components automatically
        if self.auto_setup:
            self.setup_integration_components()
    
    def setup_integration_components(self) -> Dict[str, Any]:
        """Set up all components required for pathfinding integration testing."""
        # Create pathfinding components
        graph_component = MockRealGraphForPathfinding()
        metrics_component = MockPerformanceMetrics()
        memory_component = MockMemoryManager()
        heuristic_component = MockHeuristicFunction()
        cost_component = MockCostFunction()
        
        # Register components with dependencies
        self.register_component(
            "graph", graph_component,
            dependencies=[],
            config={"topology": "complex", "weighted": True}
        )
        
        self.register_component(
            "heuristic", heuristic_component,
            dependencies=["graph"],
            config={"algorithm": "euclidean", "admissible": True}
        )
        
        self.register_component(
            "cost_function", cost_component,
            dependencies=["graph"],
            config={"optimization": True, "non_negative": True}
        )
        
        self.register_component(
            "performance_metrics", metrics_component,
            dependencies=["graph", "heuristic", "cost_function"],
            config={"tracking_enabled": True}
        )
        
        self.register_component(
            "memory_manager", memory_component,
            dependencies=[],
            config={"max_memory": 1000000, "gc_enabled": True}
        )
        
        # Set up legacy properties for backwards compatibility
        self.real_graph = graph_component
        self.performance_metrics = metrics_component
        self.memory_manager = memory_component
        
        return self.components
    
    def configure_component_interactions(self) -> None:
        """Configure how pathfinding components interact with each other."""
        # Set up graph-heuristic interaction
        self.create_component_interaction(
            "graph", "heuristic", "node_distance_calculation"
        )
        
        # Set up graph-cost interaction
        self.create_component_interaction(
            "graph", "cost_function", "edge_cost_calculation"
        )
        
        # Set up performance tracking interactions
        self.create_component_interaction(
            "heuristic", "performance_metrics", "heuristic_call_tracking"
        )
        
        self.create_component_interaction(
            "cost_function", "performance_metrics", "cost_call_tracking"
        )
        
        # Set up memory management interactions
        self.create_component_interaction(
            "performance_metrics", "memory_manager", "memory_usage_reporting"
        )
    
    def validate_integration_state(self) -> bool:
        """Validate that the pathfinding integration is in a consistent state."""
        # Check that all required components are registered
        required_components = [
            "graph", "heuristic", "cost_function", 
            "performance_metrics", "memory_manager"
        ]
        
        for component_name in required_components:
            if component_name not in self.components:
                return False
            
            # Check that component is initialized
            if self.component_states.get(component_name) != "initialized":
                return False
        
        # Validate component relationships
        graph = self.get_component("graph")
        heuristic = self.get_component("heuristic")
        cost_function = self.get_component("cost_function")
        
        # Ensure graph has nodes and edges for pathfinding
        if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
            return False
        
        # Ensure heuristic and cost functions are callable
        if not (hasattr(heuristic, "compute") and hasattr(cost_function, "compute")):
            return False
        
        return True


class MockGraphForPathfinding(Mock):
    """Mock graph for PairwiseBidirectionalAStar."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Graph structure
        self._nodes = set()
        self._edges = {}
        self._weights = {}
        
        # Graph methods
        self.nodes = Mock(return_value=list(self._nodes))
        self.edges = Mock(return_value=list(self._edges.keys()))
        self.neighbors = Mock(return_value=[])
        self.predecessors = Mock(return_value=[])
        self.successors = Mock(return_value=[])
        
        # Weight and cost methods
        self.get_edge_weight = Mock(return_value=1.0)
        self.has_edge = Mock(return_value=False)
        self.has_node = Mock(return_value=False)
        
        # Graph properties
        self.number_of_nodes = Mock(return_value=0)
        self.number_of_edges = Mock(return_value=0)
        self.is_directed = Mock(return_value=True)


class MockHeuristicFunction(AbstractAlgorithmicFunctionMock):
    """Mock heuristic function for A* algorithm."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set algorithm properties
        self.algorithm_name = "heuristic_function"
        self.algorithm_type = "heuristic"
        self.complexity_class = "O(1)"
        self.is_deterministic = True
        self.is_continuous = True
        
        # Heuristic methods
        self.estimate = Mock(return_value=0.0)
        self.is_admissible = Mock(return_value=True)
        self.is_consistent = Mock(return_value=True)
        
        # Distance methods
        self.euclidean_distance = Mock(return_value=1.0)
        self.manhattan_distance = Mock(return_value=1.0)
        self.semantic_distance = Mock(return_value=0.5)
    
    def compute(self, source_node, target_node, *args, **kwargs) -> float:
        """Compute heuristic estimate between two nodes."""
        return self.estimate(source_node, target_node)
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input nodes for heuristic calculation."""
        return len(args) >= 2  # At least source and target nodes
    
    def get_algorithm_properties(self) -> Dict[str, Any]:
        """Get heuristic function properties."""
        return {
            'is_admissible': True,
            'is_consistent': True,
            'distance_functions': ['euclidean', 'manhattan', 'semantic'],
            'domain': 'pathfinding',
            'complexity': self.complexity_class
        }


class MockCostFunction(AbstractAlgorithmicFunctionMock):
    """Mock cost function for pathfinding."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set algorithm properties
        self.algorithm_name = "cost_function"
        self.algorithm_type = "cost_calculator"
        self.complexity_class = "O(1)"
        self.is_deterministic = True
        self.codomain_bounds = (0, float('inf'))  # Non-negative costs
        
        # Cost methods
        self.edge_cost = Mock(return_value=1.0)
        self.path_cost = Mock(return_value=1.0)
        self.total_cost = Mock(return_value=1.0)
        
        # Cost optimization
        self.optimize_cost = Mock()
        self.validate_cost = Mock(return_value=True)
    
    def compute(self, source_node, target_node, *args, **kwargs) -> float:
        """Compute cost between two nodes."""
        return self.edge_cost(source_node, target_node)
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input nodes for cost calculation."""
        if len(args) < 2:
            return False
        # Check for non-negative costs if computed
        try:
            cost = self.compute(*args, **kwargs)
            return cost >= 0
        except:
            return False
    
    def get_algorithm_properties(self) -> Dict[str, Any]:
        """Get cost function properties."""
        return {
            'cost_types': ['edge', 'path', 'total'],
            'domain': 'pathfinding',
            'non_negative': True,
            'optimization_supported': True,
            'complexity': self.complexity_class
        }


class MockPriorityQueue(Mock):
    """Mock priority queue for pathfinding algorithms."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heap = []
        self._entries = {}
        
        # Queue operations
        self.push = Mock(side_effect=self._mock_push)
        self.pop = Mock(side_effect=self._mock_pop)
        self.empty = Mock(return_value=len(self._heap) == 0)
        self.size = Mock(return_value=len(self._heap))
        
        # Priority queue methods
        self.peek = Mock(return_value=None)
        self.update_priority = Mock()
        self.contains = Mock(return_value=False)
    
    def _mock_push(self, item, priority):
        heapq.heappush(self._heap, (priority, item))
        self._entries[item] = priority
    
    def _mock_pop(self):
        if self._heap:
            priority, item = heapq.heappop(self._heap)
            self._entries.pop(item, None)
            return item
        return None


class MockSearchNode(Mock):
    """Mock search node for pathfinding algorithms."""
    
    def __init__(self, node_id="node", g_cost=0.0, h_cost=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = node_id
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = None
        
        # Node methods
        self.get_f_cost = Mock(return_value=self.f_cost)
        self.get_g_cost = Mock(return_value=self.g_cost)
        self.get_h_cost = Mock(return_value=self.h_cost)
        self.set_parent = Mock()
        self.get_path = Mock(return_value=[])


class MockPathResult(Mock):
    """Mock path result from pathfinding."""
    
    def __init__(self, path=None, cost=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path or []
        self.cost = cost
        self.length = len(self.path)
        self.nodes_expanded = 0
        self.search_time = 0.0
        
        # Result methods
        self.get_path = Mock(return_value=self.path)
        self.get_cost = Mock(return_value=self.cost)
        self.get_length = Mock(return_value=self.length)
        self.is_valid = Mock(return_value=bool(self.path))


class MockRealGraphForPathfinding(Mock):
    """Mock representing real graph behavior for pathfinding."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Real graph structure
        self.complex_topology = self._create_complex_topology()
        self.weighted_edges = self._create_weighted_edges()
    
    def _create_complex_topology(self):
        """Create complex graph topology."""
        nodes = [f"node_{i}" for i in range(100)]
        edges = []
        for i in range(99):
            edges.append((f"node_{i}", f"node_{i+1}"))
            if i % 10 == 0:  # Add some cross connections
                edges.append((f"node_{i}", f"node_{min(i+10, 99)}"))
        return {"nodes": nodes, "edges": edges}
    
    def _create_weighted_edges(self):
        """Create weighted edge structure."""
        weights = {}
        for i in range(99):
            weights[(f"node_{i}", f"node_{i+1}")] = 1.0 + (i % 5) * 0.1
        return weights


class MockPerformanceMetrics(Mock):
    """Mock performance metrics for pathfinding."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Performance tracking
        self.nodes_expanded = 0
        self.nodes_visited = 0
        self.search_time = 0.0
        self.memory_usage = 0
        
        # Metrics methods
        self.start_timer = Mock()
        self.stop_timer = Mock()
        self.increment_nodes_expanded = Mock()
        self.increment_nodes_visited = Mock()
        self.get_metrics = Mock(return_value={})
        self.reset_metrics = Mock()


class MockMemoryManager(Mock):
    """Mock memory manager for pathfinding algorithms."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Memory management
        self.max_memory = 1000000  # 1MB
        self.current_usage = 0
        
        # Memory methods
        self.allocate = Mock()
        self.deallocate = Mock()
        self.check_memory_limit = Mock(return_value=True)
        self.gc_collect = Mock()
        self.get_usage = Mock(return_value=self.current_usage)