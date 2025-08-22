"""
Mock classes for PairwiseBidirectionalAStar tests.
"""

from unittest.mock import Mock
import heapq
from typing import List, Dict, Any, Optional, Tuple, Set


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


class MockPairwiseBidirectionalAStarEdgeCases(Mock):
    """Mock for PairwiseBidirectionalAStar edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Edge case scenarios
        self.no_path_exists = Mock(return_value=[])
        self.source_equals_target = Mock(return_value=["source"])
        self.disconnected_graph = Mock(return_value=[])
        self.infinite_cost = Mock(return_value=[])
        self.negative_weights = Mock(side_effect=ValueError("Negative weights not supported"))


class MockPairwiseBidirectionalAStarIntegration(Mock):
    """Mock for PairwiseBidirectionalAStar integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integration components
        self.real_graph = MockRealGraphForPathfinding()
        self.performance_metrics = MockPerformanceMetrics()
        self.memory_manager = MockMemoryManager()


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


class MockHeuristicFunction(Mock):
    """Mock heuristic function for A* algorithm."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Heuristic methods
        self.estimate = Mock(return_value=0.0)
        self.is_admissible = Mock(return_value=True)
        self.is_consistent = Mock(return_value=True)
        
        # Distance methods
        self.euclidean_distance = Mock(return_value=1.0)
        self.manhattan_distance = Mock(return_value=1.0)
        self.semantic_distance = Mock(return_value=0.5)


class MockCostFunction(Mock):
    """Mock cost function for pathfinding."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cost methods
        self.edge_cost = Mock(return_value=1.0)
        self.path_cost = Mock(return_value=1.0)
        self.total_cost = Mock(return_value=1.0)
        
        # Cost optimization
        self.optimize_cost = Mock()
        self.validate_cost = Mock(return_value=True)


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