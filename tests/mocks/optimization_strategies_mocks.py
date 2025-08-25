"""
Mock classes for optimization strategies tests.

This module provides mock implementations following the SMIED Testing Framework
Design Specifications with factory pattern and abstract base class hierarchy.
"""

from unittest.mock import Mock, MagicMock
import time
import sqlite3
import hashlib
import json
import pickle
import os
from typing import List, Optional, Any, Dict, Tuple, Set
from dataclasses import dataclass
import networkx as nx

# Import abstract base classes
from tests.mocks.base.entity_mock import AbstractEntityMock, EntityType
from tests.mocks.base.algorithmic_function_mock import AbstractAlgorithmicFunctionMock
from tests.mocks.base.operation_mock import AbstractOperationMock
from tests.mocks.base.collection_mock import AbstractCollectionMock


class OptimizationStrategiesMockFactory:
    """Factory class for creating optimization strategies mock instances.
    
    This factory follows the SMIED Testing Framework Design Specifications
    for mock creation using factory pattern with abstract base class hierarchy.
    """
    
    def __init__(self):
        self._mock_classes = {
            # Core optimization component mocks
            'MockPathCache': MockPathCache,
            'MockGraphOptimizer': MockGraphOptimizer,
            'MockPersistentCache': MockPersistentCache,
            'MockOptimizedSMIED': MockOptimizedSMIED,
            'MockOptimizationBenchmark': MockOptimizationBenchmark,
            
            # Validation specific mocks
            'MockPathCacheValidation': MockPathCacheValidation,
            'MockGraphOptimizerValidation': MockGraphOptimizerValidation,
            'MockPersistentCacheValidation': MockPersistentCacheValidation,
            'MockOptimizedSMIEDValidation': MockOptimizedSMIEDValidation,
            
            # Edge case specific mocks
            'MockPathCacheEdgeCases': MockPathCacheEdgeCases,
            'MockGraphOptimizerEdgeCases': MockGraphOptimizerEdgeCases,
            'MockPersistentCacheEdgeCases': MockPersistentCacheEdgeCases,
            'MockOptimizedSMIEDEdgeCases': MockOptimizedSMIEDEdgeCases,
            
            # Integration specific mocks
            'MockPathCacheIntegration': MockPathCacheIntegration,
            'MockGraphOptimizerIntegration': MockGraphOptimizerIntegration,
            'MockPersistentCacheIntegration': MockPersistentCacheIntegration,
            'MockOptimizedSMIEDIntegration': MockOptimizedSMIEDIntegration,
            
            # Utility and helper mocks
            'MockSMIEDForOptimization': MockSMIEDForOptimization,
            'MockOptimizationResult': MockOptimizationResult,
            'MockCacheStatistics': MockCacheStatistics,
            'MockPerformanceMetrics': MockPerformanceMetrics,
            'MockGraphStructure': MockGraphStructure,
            'MockDatabaseConnection': MockDatabaseConnection,
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


class MockPathCache(AbstractCollectionMock):
    """Mock implementation of PathCache for optimization strategies testing."""
    
    def __init__(self, max_size: int = 1000, *args, **kwargs):
        super().__init__(collection_type="cache", *args, **kwargs)
        
        # Core cache attributes
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self._hit_count = 0
        self._miss_count = 0
        
        # Set cache properties
        self.name = f"PathCache-{max_size}"
        self.label = f"LRU Cache with {max_size} capacity"
        
        # Add cache-specific tags (simple list instead of using add_tag method)
        self.tags = ["lru_cache", "path_cache", "optimization"]
        
        # Mock methods with realistic behavior
        self._make_key = Mock(side_effect=self._mock_make_key)
        self.get = Mock(side_effect=self._mock_get)
        self.put = Mock(side_effect=self._mock_put)
        self.clear = Mock(side_effect=self._mock_clear)
        self.size = Mock(side_effect=self._mock_size)
        self.hit_rate = Mock(side_effect=self._mock_hit_rate)
    
    def _mock_make_key(self, subject: str, predicate: str, object_term: str) -> str:
        """Create a cache key from triple components."""
        return f"{subject}||{predicate}||{object_term}"
    
    def _mock_get(self, subject: str, predicate: str, object_term: str) -> Optional[Any]:
        """Get cached result for a triple."""
        key = self._mock_make_key(subject, predicate, object_term)
        
        if key in self.cache:
            # Move to end (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            self._hit_count += 1
            return self.cache[key]
        
        self._miss_count += 1
        return None
    
    def _mock_put(self, subject: str, predicate: str, object_term: str, result: Any) -> None:
        """Cache a result for a triple."""
        key = self._mock_make_key(subject, predicate, object_term)
        
        if key in self.cache:
            # Update existing
            if key in self.access_order:
                self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            if self.access_order:
                lru_key = self.access_order.pop(0)
                if lru_key in self.cache:
                    del self.cache[lru_key]
        
        self.cache[key] = result
        self.access_order.append(key)
    
    def _mock_clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        self._hit_count = 0
        self._miss_count = 0
    
    def _mock_size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def _mock_hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self._hit_count + self._miss_count
        return (self._hit_count / total * 100) if total > 0 else 0.0
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this cache."""
        return self.max_size
    
    def validate_entity(self) -> bool:
        """Validate that the cache is consistent and valid."""
        if self.max_size <= 0:
            return False
        if len(self.cache) > self.max_size:
            return False
        if len(self.access_order) != len(self.cache):
            return False
        return True
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this cache."""
        return f"PathCache:{self.id}:{self.max_size}:{len(self.cache)}"
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about this cache collection."""
        return {
            'collection_type': 'lru_cache',
            'max_size': self.max_size,
            'current_size': len(self.cache),
            'hit_rate': self.hit_rate(),
            'total_accesses': self._hit_count + self._miss_count,
            'is_full': len(self.cache) >= self.max_size,
            'eviction_policy': 'LRU',
            'supports_persistence': False
        }
    
    def validate_item(self, item: Any) -> bool:
        """Validate that an item can be stored in this cache."""
        # For path cache, we accept any serializable result
        try:
            # Check if item is serializable (basic validation)
            import json
            json.dumps(str(item))
            return True
        except (TypeError, ValueError):
            return False
    
    def transform_item(self, item: Any) -> Any:
        """Transform an item before storing it in the cache."""
        # For path cache, store items as-is since they're already processed results
        return item


class MockGraphOptimizer(AbstractAlgorithmicFunctionMock):
    """Mock implementation of GraphOptimizer for optimization strategies testing."""
    
    def __init__(self, verbosity: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for graph optimization algorithm
        self.algorithm_name = "graph_optimizer"
        self.algorithm_type = "optimization"
        self.complexity_class = "O(V + E)"
        
        self.verbosity = verbosity
        self.distance_cache = {}
        self.centrality_cache = {}
        
        # Set optimizer properties
        self.name = f"GraphOptimizer-v{verbosity}"
        self.label = f"Graph optimization with verbosity {verbosity}"
        
        # Add optimizer-specific tags (simple list instead of using add_tag method)
        self.tags = ["graph_optimizer", "performance_optimization", "pathfinding_optimization"]
        
        # Mock methods with realistic behavior
        self.optimize_graph_structure = Mock(side_effect=self._mock_optimize_graph_structure)
        self.precompute_distances = Mock(side_effect=self._mock_precompute_distances)
        self.get_precomputed_distance = Mock(side_effect=self._mock_get_precomputed_distance)
        self.compute_node_importance = Mock(side_effect=self._mock_compute_node_importance)
    
    def _mock_optimize_graph_structure(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Mock implementation of optimize_graph_structure."""
        optimized_graph = graph.copy()
        
        # Simulate edge removal optimization
        initial_edges = optimized_graph.number_of_edges()
        edges_to_remove = max(0, int(initial_edges * 0.1))  # Remove 10% of edges
        
        # Mock edge removal
        all_edges = list(optimized_graph.edges())
        for i in range(min(edges_to_remove, len(all_edges))):
            if all_edges[i] in optimized_graph.edges():
                optimized_graph.remove_edge(*all_edges[i])
        
        # Simulate shortcut addition
        shortcuts_added = min(100, int(optimized_graph.number_of_nodes() * 0.05))
        for i in range(shortcuts_added):
            nodes = list(optimized_graph.nodes())
            if len(nodes) >= 2:
                source, target = nodes[i % len(nodes)], nodes[(i + 1) % len(nodes)]
                if not optimized_graph.has_edge(source, target):
                    optimized_graph.add_edge(source, target, relation='shortcut', weight=2.0)
        
        return optimized_graph
    
    def _mock_precompute_distances(self, graph: nx.DiGraph, key_nodes: List[str]) -> Dict[Tuple[str, str], float]:
        """Mock implementation of precompute_distances."""
        distances = {}
        limited_nodes = key_nodes[:100]  # Limit for performance
        
        for i, source in enumerate(limited_nodes):
            if source not in graph:
                continue
            
            for j, target in enumerate(limited_nodes):
                if target not in graph or source == target:
                    continue
                
                # Mock distance calculation
                mock_distance = abs(i - j) + 1.0  # Simple mock distance
                if mock_distance <= 6:  # Cutoff simulation
                    distances[(source, target)] = mock_distance
        
        self.distance_cache = distances
        return distances
    
    def _mock_get_precomputed_distance(self, source: str, target: str) -> Optional[float]:
        """Mock implementation of get_precomputed_distance."""
        return self.distance_cache.get((source, target))
    
    def _mock_compute_node_importance(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Mock implementation of compute_node_importance."""
        importance_scores = {}
        nodes = list(graph.nodes())
        
        for i, node in enumerate(nodes):
            # Mock importance calculation based on degree and position
            degree = graph.degree(node) if hasattr(graph, 'degree') else 1
            position_factor = (len(nodes) - i) / len(nodes)
            importance_scores[node] = 0.7 * (degree / 10.0) + 0.3 * position_factor
        
        self.centrality_cache = importance_scores
        return importance_scores
    
    def compute(self, *args, **kwargs) -> Any:
        """Compute graph optimization algorithm."""
        if 'graph' in kwargs:
            graph = kwargs['graph']
            operation = kwargs.get('operation', 'optimize')
            
            if operation == 'optimize':
                return self._mock_optimize_graph_structure(graph)
            elif operation == 'precompute_distances':
                key_nodes = kwargs.get('key_nodes', list(graph.nodes())[:50])
                return self._mock_precompute_distances(graph, key_nodes)
            elif operation == 'compute_importance':
                return self._mock_compute_node_importance(graph)
        
        return None
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input arguments for graph optimization."""
        if 'graph' in kwargs and not hasattr(kwargs['graph'], 'nodes'):
            return False
        return True
    
    def get_algorithm_properties(self) -> Dict[str, Any]:
        """Get properties specific to graph optimization algorithm."""
        return {
            "supports_optimization": True,
            "supports_distance_precomputation": True,
            "supports_importance_calculation": True,
            "supports_shortcuts": True,
            "edge_removal_optimization": True
        }


class MockPersistentCache(AbstractOperationMock):
    """Mock implementation of PersistentCache for optimization strategies testing."""
    
    def __init__(self, db_path: str = "test_smied_cache.db", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for cache operation
        self.operation_type = "caching"
        self.supports_persistence = True
        self.supports_transactions = True
        
        self.db_path = db_path
        self._mock_data = {
            'pathfinding_cache': {},
            'graph_cache': {}
        }
        
        # Set cache properties
        self.name = f"PersistentCache-{os.path.basename(db_path)}"
        self.label = f"SQLite persistent cache at {db_path}"
        
        # Add cache-specific tags (simple list instead of using add_tag method)
        self.tags = ["persistent_cache", "sqlite_cache", "database_cache"]
        
        # Mock methods with realistic behavior
        self.init_database = Mock(side_effect=self._mock_init_database)
        self.cache_pathfinding_result = Mock(side_effect=self._mock_cache_pathfinding_result)
        self.get_cached_result = Mock(side_effect=self._mock_get_cached_result)
        self.cache_graph = Mock(side_effect=self._mock_cache_graph)
        self.get_cached_graph = Mock(side_effect=self._mock_get_cached_graph)
        self.get_cache_stats = Mock(side_effect=self._mock_get_cache_stats)
        self.clear_cache = Mock(side_effect=self._mock_clear_cache)
    
    def _mock_init_database(self) -> None:
        """Mock implementation of init_database."""
        # Simulate database initialization
        pass
    
    def _mock_cache_pathfinding_result(self, subject: str, predicate: str, object_term: str, result: Any) -> None:
        """Mock implementation of cache_pathfinding_result."""
        key = f"{subject}||{predicate}||{object_term}"
        
        # Extract result data
        success = getattr(result, 'success', False)
        execution_time = getattr(result, 'execution_time', 0.0)
        subject_path = getattr(result, 'subject_path', None)
        object_path = getattr(result, 'object_path', None)
        connecting_predicate = getattr(result, 'connecting_predicate', None)
        
        self._mock_data['pathfinding_cache'][key] = {
            'subject': subject,
            'predicate': predicate,
            'object_term': object_term,
            'success': success,
            'execution_time': execution_time,
            'subject_path': subject_path,
            'object_path': object_path,
            'connecting_predicate': connecting_predicate
        }
    
    def _mock_get_cached_result(self, subject: str, predicate: str, object_term: str) -> Optional[Dict]:
        """Mock implementation of get_cached_result."""
        key = f"{subject}||{predicate}||{object_term}"
        return self._mock_data['pathfinding_cache'].get(key)
    
    def _mock_cache_graph(self, graph: nx.DiGraph) -> str:
        """Mock implementation of cache_graph."""
        # Create mock hash
        graph_str = f"nodes:{graph.number_of_nodes()},edges:{graph.number_of_edges()}"
        graph_hash = hashlib.md5(graph_str.encode()).hexdigest()[:8]
        
        self._mock_data['graph_cache'][graph_hash] = {
            'graph_hash': graph_hash,
            'node_count': graph.number_of_nodes(),
            'edge_count': graph.number_of_edges(),
            'graph_data': graph  # Store the actual graph for mock
        }
        
        return graph_hash
    
    def _mock_get_cached_graph(self, graph_hash: str) -> Optional[nx.DiGraph]:
        """Mock implementation of get_cached_graph."""
        cached_data = self._mock_data['graph_cache'].get(graph_hash)
        return cached_data['graph_data'] if cached_data else None
    
    def _mock_get_cache_stats(self) -> Dict[str, Any]:
        """Mock implementation of get_cache_stats."""
        pathfinding_count = len(self._mock_data['pathfinding_cache'])
        successful_count = sum(1 for data in self._mock_data['pathfinding_cache'].values() if data['success'])
        graph_count = len(self._mock_data['graph_cache'])
        
        return {
            'pathfinding_entries': pathfinding_count,
            'successful_pathfinding': successful_count,
            'success_rate': (successful_count / pathfinding_count * 100) if pathfinding_count > 0 else 0,
            'cached_graphs': graph_count
        }
    
    def _mock_clear_cache(self) -> None:
        """Mock implementation of clear_cache."""
        self._mock_data['pathfinding_cache'].clear()
        self._mock_data['graph_cache'].clear()
    
    def execute_operation(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute cache operation."""
        if operation_name == 'cache_result':
            return self._mock_cache_pathfinding_result(*args, **kwargs)
        elif operation_name == 'get_result':
            return self._mock_get_cached_result(*args, **kwargs)
        elif operation_name == 'cache_graph':
            return self._mock_cache_graph(*args, **kwargs)
        elif operation_name == 'get_graph':
            return self._mock_get_cached_graph(*args, **kwargs)
        elif operation_name == 'get_stats':
            return self._mock_get_cache_stats()
        elif operation_name == 'clear':
            return self._mock_clear_cache()
        
        return None
    
    def validate_operation_inputs(self, operation_name: str, *args, **kwargs) -> bool:
        """Validate inputs for cache operations."""
        if operation_name in ['cache_result', 'get_result']:
            return len(args) >= 3 or all(k in kwargs for k in ['subject', 'predicate', 'object_term'])
        elif operation_name in ['cache_graph', 'get_graph']:
            return len(args) >= 1 or 'graph' in kwargs or 'graph_hash' in kwargs
        return True
    
    def get_operation_metadata(self, operation_name: str) -> Dict[str, Any]:
        """Get metadata for cache operations."""
        operations = {
            'cache_result': {'type': 'write', 'persistent': True},
            'get_result': {'type': 'read', 'persistent': True},
            'cache_graph': {'type': 'write', 'persistent': True},
            'get_graph': {'type': 'read', 'persistent': True},
            'get_stats': {'type': 'read', 'computational': True},
            'clear': {'type': 'write', 'destructive': True}
        }
        return operations.get(operation_name, {})
    
    def execute(self, target: Any, *args, **kwargs) -> Any:
        """Execute the cache operation on the target."""
        # Delegate to execute_operation method
        if args:
            operation_name = args[0]
            return self.execute_operation(operation_name, *args[1:], **kwargs)
        elif 'operation' in kwargs:
            operation_name = kwargs.pop('operation')
            return self.execute_operation(operation_name, **kwargs)
        else:
            # Default operation is to initialize the cache
            return self._mock_init_database()
    
    def validate_target(self, target: Any) -> bool:
        """Validate that the target is suitable for cache operations."""
        # For persistent cache, we accept various types of targets
        if hasattr(target, '__dict__'):  # Objects with attributes
            return True
        if isinstance(target, (str, int, float, bool, dict, list, tuple)):
            return True
        return False


class MockOptimizedSMIED(AbstractOperationMock):
    """Mock implementation of OptimizedSMIED for optimization strategies testing."""
    
    def __init__(self, base_smied=None, enable_caching: bool = True, 
                 enable_graph_optimization: bool = True, verbosity: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for SMIED operation
        self.operation_type = "semantic_analysis"
        self.supports_caching = enable_caching
        self.supports_optimization = enable_graph_optimization
        
        self.base_smied = base_smied or MockSMIEDForOptimization()
        self.verbosity = verbosity
        
        # Initialize optimization components
        self.path_cache = MockPathCache(max_size=1000) if enable_caching else None
        self.persistent_cache = MockPersistentCache() if enable_caching else None
        self.graph_optimizer = MockGraphOptimizer(verbosity) if enable_graph_optimization else None
        
        self.cache_hits = 0
        self.cache_misses = 0
        self.optimized_graph = None
        
        # Set entity properties
        optimizations = []
        if enable_caching:
            optimizations.append("caching")
        if enable_graph_optimization:
            optimizations.append("graph_optimization")
        
        self.name = f"OptimizedSMIED-{'-'.join(optimizations)}"
        self.label = f"SMIED with {', '.join(optimizations)} optimizations"
        
        # Add optimization-specific tags (simple list instead of using add_tag method)
        self.tags = ["optimized_smied", "semantic_analysis"] + optimizations
        
        # Mock methods with realistic behavior
        self.build_optimized_graph = Mock(side_effect=self._mock_build_optimized_graph)
        self.analyze_triple_optimized = Mock(side_effect=self._mock_analyze_triple_optimized)
        self.get_optimization_stats = Mock(side_effect=self._mock_get_optimization_stats)
    
    def _mock_build_optimized_graph(self) -> nx.DiGraph:
        """Mock implementation of build_optimized_graph."""
        # Create a mock graph
        base_graph = nx.DiGraph()
        base_graph.add_nodes_from(['cat.n.01', 'animal.n.01', 'mammal.n.01', 'feline.n.01'])
        base_graph.add_edges_from([
            ('cat.n.01', 'feline.n.01'),
            ('feline.n.01', 'mammal.n.01'),
            ('mammal.n.01', 'animal.n.01')
        ])
        
        # Apply optimizations if available
        if self.graph_optimizer:
            self.optimized_graph = self.graph_optimizer._mock_optimize_graph_structure(base_graph)
            
            # Mock importance calculation and distance precomputation
            important_nodes = list(self.optimized_graph.nodes())[:10]
            if important_nodes:
                self.graph_optimizer._mock_precompute_distances(self.optimized_graph, important_nodes)
        else:
            self.optimized_graph = base_graph
        
        return self.optimized_graph
    
    def _mock_analyze_triple_optimized(self, subject: str, predicate: str, object_term: str,
                                     max_depth: int = 10, beam_width: int = 3, **kwargs) -> Tuple:
        """Mock implementation of analyze_triple_optimized."""
        # Check memory cache first
        if self.path_cache:
            cached_result = self.path_cache._mock_get(subject, predicate, object_term)
            if cached_result:
                self.cache_hits += 1
                return cached_result
        
        # Check persistent cache
        if self.persistent_cache:
            cached_result = self.persistent_cache._mock_get_cached_result(subject, predicate, object_term)
            if cached_result and cached_result.get('success'):
                self.cache_hits += 1
                result_tuple = (
                    cached_result['subject_path'],
                    cached_result['object_path'],
                    cached_result['connecting_predicate']
                )
                
                # Cache in memory for faster access
                if self.path_cache:
                    self.path_cache._mock_put(subject, predicate, object_term, result_tuple)
                
                return result_tuple
        
        # Cache miss - perform actual computation
        self.cache_misses += 1
        
        # Ensure we have an optimized graph
        if not self.optimized_graph:
            self._mock_build_optimized_graph()
        
        # Mock pathfinding result
        start_time = time.time()
        
        # Create realistic mock result
        if subject == "cat" and predicate == "chase" and object_term == "mouse":
            result = (
                ['cat.n.01', 'predator.n.01'],
                ['prey.n.01', 'mouse.n.01'],
                f'{predicate}.v.01'
            )
        else:
            # Generic mock result
            result = (
                [f'{subject}.n.01', 'intermediate.n.01'],
                ['intermediate.n.01', f'{object_term}.n.01'],
                f'{predicate}.v.01'
            )
        
        execution_time = time.time() - start_time
        
        # Cache the result
        if self.path_cache:
            self.path_cache._mock_put(subject, predicate, object_term, result)
        
        if self.persistent_cache:
            # Create a mock result object for persistent cache
            mock_result = type('MockResult', (), {
                'success': result and any([result[0], result[1], result[2]]),
                'execution_time': execution_time,
                'subject_path': result[0] if result else None,
                'object_path': result[1] if result else None,
                'connecting_predicate': result[2] if result else None
            })()
            
            self.persistent_cache._mock_cache_pathfinding_result(subject, predicate, object_term, mock_result)
        
        return result
    
    def _mock_get_optimization_stats(self) -> Dict[str, Any]:
        """Mock implementation of get_optimization_stats."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate
        }
        
        # Add cache-specific stats
        if self.path_cache:
            stats['memory_cache_size'] = self.path_cache._mock_size()
        
        if self.persistent_cache:
            persistent_stats = self.persistent_cache._mock_get_cache_stats()
            stats.update(persistent_stats)
        
        # Add graph optimization stats
        if self.optimized_graph:
            stats['optimized_graph_nodes'] = self.optimized_graph.number_of_nodes()
            stats['optimized_graph_edges'] = self.optimized_graph.number_of_edges()
        
        return stats
    
    def execute_operation(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute optimization operation."""
        if operation_name == 'build_graph':
            return self._mock_build_optimized_graph()
        elif operation_name == 'analyze_triple':
            return self._mock_analyze_triple_optimized(*args, **kwargs)
        elif operation_name == 'get_stats':
            return self._mock_get_optimization_stats()
        
        return None
    
    def validate_operation_inputs(self, operation_name: str, *args, **kwargs) -> bool:
        """Validate inputs for optimization operations."""
        if operation_name == 'analyze_triple':
            return len(args) >= 3 or all(k in kwargs for k in ['subject', 'predicate', 'object_term'])
        return True
    
    def get_operation_metadata(self, operation_name: str) -> Dict[str, Any]:
        """Get metadata for optimization operations."""
        operations = {
            'build_graph': {'type': 'initialization', 'expensive': True},
            'analyze_triple': {'type': 'analysis', 'cached': True},
            'get_stats': {'type': 'monitoring', 'lightweight': True}
        }
        return operations.get(operation_name, {})
    
    def execute(self, target: Any, *args, **kwargs) -> Any:
        """Execute the optimization operation on the target."""
        # Delegate to execute_operation method
        if args:
            operation_name = args[0]
            return self.execute_operation(operation_name, *args[1:], **kwargs)
        elif 'operation' in kwargs:
            operation_name = kwargs.pop('operation')
            return self.execute_operation(operation_name, **kwargs)
        else:
            # Default operation is to build the optimized graph
            return self._mock_build_optimized_graph()
    
    def validate_target(self, target: Any) -> bool:
        """Validate that the target is suitable for optimization operations."""
        # For optimized SMIED, we accept various types of analysis targets
        if isinstance(target, (str, tuple, list)):
            return True
        if hasattr(target, '__dict__'):  # Objects with attributes
            return True
        return False


class MockOptimizationBenchmark(AbstractAlgorithmicFunctionMock):
    """Mock implementation of OptimizationBenchmark for optimization strategies testing."""
    
    def __init__(self, base_smied=None, verbosity: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for benchmarking algorithm
        self.algorithm_name = "optimization_benchmark"
        self.algorithm_type = "benchmarking"
        self.complexity_class = "O(n*k)"  # n tests, k runs
        
        self.base_smied = base_smied or MockSMIEDForOptimization()
        self.optimized_smied = MockOptimizedSMIED(self.base_smied, verbosity=verbosity)
        self.verbosity = verbosity
        
        # Set benchmark properties
        self.name = f"OptimizationBenchmark-v{verbosity}"
        self.label = f"Optimization benchmark with verbosity {verbosity}"
        
        # Add benchmark-specific tags (simple list instead of using add_tag method)
        self.tags = ["optimization_benchmark", "performance_comparison", "timing_analysis"]
        
        # Mock methods
        self.run_optimization_benchmark = Mock(side_effect=self._mock_run_optimization_benchmark)
        self.print_benchmark_results = Mock(side_effect=self._mock_print_benchmark_results)
    
    def _mock_run_optimization_benchmark(self, test_cases: List, runs_per_test: int = 3) -> Dict[str, Any]:
        """Mock implementation of run_optimization_benchmark."""
        results = {
            'base_results': [],
            'optimized_results': [],
            'test_cases': []
        }
        
        for i, test_case in enumerate(test_cases):
            # Mock test case attributes
            subject = getattr(test_case, 'subject', f'subject_{i}')
            predicate = getattr(test_case, 'predicate', f'predicate_{i}')
            obj = getattr(test_case, 'object', f'object_{i}')
            description = getattr(test_case, 'description', f'test case {i}')
            
            # Mock base SMIED results
            base_times = [0.1 + i * 0.01 + run * 0.005 for run in range(runs_per_test)]
            base_successes = runs_per_test  # All succeed in mock
            
            # Mock optimized SMIED results (better performance)
            optimized_times = [t * 0.6 for t in base_times]  # 40% improvement
            optimized_successes = runs_per_test
            
            # Calculate statistics
            base_avg_time = sum(base_times) / len(base_times)
            optimized_avg_time = sum(optimized_times) / len(optimized_times)
            speedup = base_avg_time / optimized_avg_time if optimized_avg_time > 0 else 0
            
            results['base_results'].append({
                'avg_time': base_avg_time,
                'success_rate': base_successes / runs_per_test,
                'times': base_times
            })
            
            results['optimized_results'].append({
                'avg_time': optimized_avg_time,
                'success_rate': optimized_successes / runs_per_test,
                'times': optimized_times,
                'speedup': speedup
            })
            
            results['test_cases'].append({
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'description': description
            })
        
        # Calculate overall statistics
        base_avg_times = [r['avg_time'] for r in results['base_results']]
        optimized_avg_times = [r['avg_time'] for r in results['optimized_results']]
        speedups = [r['speedup'] for r in results['optimized_results'] if r['speedup'] > 0]
        
        results['summary'] = {
            'total_tests': len(test_cases),
            'base_avg_time': sum(base_avg_times) / len(base_avg_times) if base_avg_times else 0,
            'optimized_avg_time': sum(optimized_avg_times) / len(optimized_avg_times) if optimized_avg_times else 0,
            'average_speedup': sum(speedups) / len(speedups) if speedups else 0,
            'max_speedup': max(speedups) if speedups else 0,
            'optimization_stats': self.optimized_smied._mock_get_optimization_stats()
        }
        
        return results
    
    def _mock_print_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Mock implementation of print_benchmark_results."""
        # In a real implementation, this would print results
        # For mock, we just simulate the operation
        if self.verbosity >= 1:
            summary = results.get('summary', {})
            # Mock printing behavior - would normally print to console
            pass
    
    def compute(self, *args, **kwargs) -> Any:
        """Compute benchmarking algorithm."""
        if 'test_cases' in kwargs:
            test_cases = kwargs['test_cases']
            runs_per_test = kwargs.get('runs_per_test', 3)
            return self._mock_run_optimization_benchmark(test_cases, runs_per_test)
        
        # Default benchmark with mock test cases
        mock_test_cases = [
            type('TestCase', (), {'subject': 'cat', 'predicate': 'chase', 'object': 'mouse', 'description': 'cat chases mouse'}),
            type('TestCase', (), {'subject': 'dog', 'predicate': 'bark', 'object': 'stranger', 'description': 'dog barks at stranger'}),
            type('TestCase', (), {'subject': 'bird', 'predicate': 'fly', 'object': 'sky', 'description': 'bird flies in sky'})
        ]
        return self._mock_run_optimization_benchmark(mock_test_cases)
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input arguments for benchmarking."""
        if 'test_cases' in kwargs and not isinstance(kwargs['test_cases'], list):
            return False
        if 'runs_per_test' in kwargs and kwargs['runs_per_test'] <= 0:
            return False
        return True
    
    def get_algorithm_properties(self) -> Dict[str, Any]:
        """Get properties specific to benchmarking algorithm."""
        return {
            "compares_base_vs_optimized": True,
            "measures_execution_time": True,
            "calculates_speedup": True,
            "provides_statistical_analysis": True,
            "supports_multiple_runs": True
        }


# Utility and helper mocks

class MockSMIEDForOptimization(Mock):
    """Mock SMIED instance for optimization testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyze_triple = Mock(side_effect=self._mock_analyze_triple)
        self.build_synset_graph = Mock(side_effect=self._mock_build_synset_graph)
        self.nlp_model = None
        self.auto_download = False
        self.verbosity = 0
        self.synset_graph = None
    
    def _mock_analyze_triple(self, subject, predicate, obj, max_depth=10, beam_width=3, verbose=False):
        """Mock implementation of analyze_triple."""
        # Simulate realistic pathfinding behavior
        subject_path = [f"{subject}.n.01", "intermediate"]
        object_path = ["intermediate", f"{obj}.n.01"]
        connecting_predicate = f"{predicate}.v.01"
        
        # Sometimes fail based on input complexity
        if subject == "rock" and predicate == "sing":
            return None, None, None
        
        return subject_path, object_path, connecting_predicate
    
    def _mock_build_synset_graph(self, verbose=False):
        """Mock implementation of build_synset_graph."""
        graph = nx.DiGraph()
        graph.add_nodes_from(['cat.n.01', 'animal.n.01', 'mammal.n.01', 'feline.n.01'])
        graph.add_edges_from([
            ('cat.n.01', 'feline.n.01'),
            ('feline.n.01', 'mammal.n.01'),
            ('mammal.n.01', 'animal.n.01')
        ])
        return graph


class MockOptimizationResult(AbstractEntityMock):
    """Mock implementation of optimization results."""
    
    def __init__(self, success: bool = True, execution_time: float = 0.1, 
                 subject_path: Optional[List] = None, object_path: Optional[List] = None,
                 connecting_predicate: Optional[str] = None, *args, **kwargs):
        super().__init__(entity_type=EntityType.PATH, *args, **kwargs)
        
        # Core result attributes
        self.success = success
        self.execution_time = execution_time
        self.subject_path = subject_path or []
        self.object_path = object_path or []
        self.connecting_predicate = connecting_predicate
        
        # Set entity properties
        self.name = f"OptimizationResult-{success}"
        self.label = "success" if success else "failure"
        
        # Add result-specific tags
        self.add_tag("optimization_result")
        self.add_tag("success" if success else "failure")
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this result."""
        return self.success
    
    def validate_entity(self) -> bool:
        """Validate that the optimization result is consistent and valid."""
        if self.execution_time < 0:
            return False
        # For successful results, paths should exist (can be empty lists but not None)
        if self.success and (self.subject_path is None or self.object_path is None):
            return False
        return True
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this optimization result."""
        return f"OptimizationResult:{self.id}:{self.success}:{self.execution_time}"


# Specialized validation, edge case, and integration mocks

class MockPathCacheValidation(MockPathCache):
    """Mock PathCache for validation testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("validation_testing")
        else:
            self.tags = ["validation_testing"]
        
        # Add validation-specific methods
        self.validate_cache_consistency = Mock(return_value=True)
        self.check_lru_order = Mock(return_value=True)


class MockPathCacheEdgeCases(MockPathCache):
    """Mock PathCache for edge case testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("edge_case_testing")
        else:
            self.tags = ["edge_case_testing"]
        
        # Override methods to simulate edge cases
        self.get = Mock(side_effect=self._mock_edge_case_get)
        self.put = Mock(side_effect=self._mock_edge_case_put)
    
    def _mock_edge_case_get(self, subject: str, predicate: str, object_term: str) -> Optional[Any]:
        """Mock get method that simulates edge cases."""
        if subject == "error_test":
            raise ValueError("Cache error simulation")
        elif subject == "memory_test":
            raise MemoryError("Out of memory")
        return self._mock_get(subject, predicate, object_term)
    
    def _mock_edge_case_put(self, subject: str, predicate: str, object_term: str, result: Any) -> None:
        """Mock put method that simulates edge cases."""
        if subject == "error_test":
            raise ValueError("Cache error simulation")
        self._mock_put(subject, predicate, object_term, result)


class MockPathCacheIntegration(MockPathCache):
    """Mock PathCache for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("integration_testing")
        else:
            self.tags = ["integration_testing"]
        
        # Add integration-specific components
        self.external_cache_systems = Mock()
        self.cache_synchronization = Mock()


# Additional specialized mocks for other components follow similar patterns...

class MockGraphOptimizerValidation(MockGraphOptimizer):
    """Mock GraphOptimizer for validation testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("validation_testing")
        else:
            self.tags = ["validation_testing"]
        self.validate_optimization_results = Mock(return_value=True)


class MockGraphOptimizerEdgeCases(MockGraphOptimizer):
    """Mock GraphOptimizer for edge case testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("edge_case_testing")
        else:
            self.tags = ["edge_case_testing"]
        
        # Override to simulate edge cases
        self.optimize_graph_structure = Mock(side_effect=self._mock_edge_case_optimize)
    
    def _mock_edge_case_optimize(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Mock optimize that simulates edge cases."""
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot optimize empty graph")
        return self._mock_optimize_graph_structure(graph)


class MockGraphOptimizerIntegration(MockGraphOptimizer):
    """Mock GraphOptimizer for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("integration_testing")
        else:
            self.tags = ["integration_testing"]
        self.real_networkx_integration = Mock()


class MockPersistentCacheValidation(MockPersistentCache):
    """Mock PersistentCache for validation testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("validation_testing")
        else:
            self.tags = ["validation_testing"]
        self.validate_database_schema = Mock(return_value=True)


class MockPersistentCacheEdgeCases(MockPersistentCache):
    """Mock PersistentCache for edge case testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("edge_case_testing")
        else:
            self.tags = ["edge_case_testing"]
        
        # Override to simulate database errors
        self.init_database = Mock(side_effect=self._mock_edge_case_init)
    
    def _mock_edge_case_init(self):
        """Mock database init that can fail."""
        if self.db_path == "/nonexistent/path/cache.db":
            raise sqlite3.Error("Database connection failed")
        if self.db_path == "error_db.db":
            raise sqlite3.Error("Database connection failed")


# Shared data across integration test instances to simulate persistence
_INTEGRATION_CACHE_DATA = {
    'pathfinding_cache': {},
    'graph_cache': {}
}

class MockPersistentCacheIntegration(MockPersistentCache):
    """Mock PersistentCache for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("integration_testing")
        else:
            self.tags = ["integration_testing"]
        self.real_sqlite_connection = Mock()
        
        # Use shared data to simulate persistence across instances
        self._mock_data = _INTEGRATION_CACHE_DATA


class MockOptimizedSMIEDValidation(MockOptimizedSMIED):
    """Mock OptimizedSMIED for validation testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("validation_testing")
        else:
            self.tags = ["validation_testing"]
        self.validate_optimization_configuration = Mock(return_value=True)


class MockOptimizedSMIEDEdgeCases(MockOptimizedSMIED):
    """Mock OptimizedSMIED for edge case testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("edge_case_testing")
        else:
            self.tags = ["edge_case_testing"]
        
        # Override to simulate edge cases
        self.analyze_triple_optimized = Mock(side_effect=self._mock_edge_case_analyze)
    
    def _mock_edge_case_analyze(self, subject, predicate, object_term, **kwargs):
        """Mock analyze that simulates edge cases."""
        if subject == "timeout_test":
            raise TimeoutError("Analysis timeout")
        return self._mock_analyze_triple_optimized(subject, predicate, object_term, **kwargs)


class MockOptimizedSMIEDIntegration(MockOptimizedSMIED):
    """Mock OptimizedSMIED for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tags') and isinstance(self.tags, list):
            self.tags.append("integration_testing")
        else:
            self.tags = ["integration_testing"]
        self.real_smied_components = Mock()
        
        # Add integration-specific methods
        self.setup_integration_environment = Mock(side_effect=self._mock_setup_integration_environment)
        self.teardown_integration_environment = Mock(side_effect=self._mock_teardown_integration_environment)
        self.validate_component_interactions = Mock(return_value=True)
    
    def _mock_setup_integration_environment(self):
        """Mock setup for integration environment."""
        # Initialize integration components
        pass
    
    def _mock_teardown_integration_environment(self):
        """Mock teardown for integration environment."""
        # Clean up integration components
        pass


# Additional utility mocks

class MockCacheStatistics(AbstractEntityMock):
    """Mock implementation of cache statistics."""
    
    def __init__(self, hit_rate: float = 0.0, total_entries: int = 0, *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.hit_rate = hit_rate
        self.total_entries = total_entries
        self.name = f"CacheStats-{hit_rate:.1f}%-{total_entries}entries"
    
    def get_primary_attribute(self) -> Any:
        return self.hit_rate
    
    def validate_entity(self) -> bool:
        return 0.0 <= self.hit_rate <= 100.0 and self.total_entries >= 0
    
    def get_entity_signature(self) -> str:
        return f"CacheStatistics:{self.id}:{self.hit_rate}:{self.total_entries}"


class MockPerformanceMetrics(AbstractEntityMock):
    """Mock implementation of performance metrics."""
    
    def __init__(self, avg_time: float = 0.0, speedup: float = 1.0, *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.avg_time = avg_time
        self.speedup = speedup
        self.name = f"PerfMetrics-{avg_time:.3f}s-{speedup:.2f}x"
    
    def get_primary_attribute(self) -> Any:
        return self.speedup
    
    def validate_entity(self) -> bool:
        return self.avg_time >= 0.0 and self.speedup >= 0.0
    
    def get_entity_signature(self) -> str:
        return f"PerformanceMetrics:{self.id}:{self.avg_time}:{self.speedup}"


class MockGraphStructure(AbstractEntityMock):
    """Mock implementation of graph structure."""
    
    def __init__(self, node_count: int = 0, edge_count: int = 0, *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.node_count = node_count
        self.edge_count = edge_count
        self.name = f"Graph-{node_count}nodes-{edge_count}edges"
    
    def get_primary_attribute(self) -> Any:
        return self.node_count
    
    def validate_entity(self) -> bool:
        return self.node_count >= 0 and self.edge_count >= 0
    
    def get_entity_signature(self) -> str:
        return f"GraphStructure:{self.id}:{self.node_count}:{self.edge_count}"


class MockDatabaseConnection(Mock):
    """Mock implementation of database connection."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connect = Mock(return_value=self)
        self.cursor = Mock(return_value=Mock())
        self.commit = Mock()
        self.close = Mock()
        self.execute = Mock()
        self.fetchone = Mock(return_value=None)
        self.fetchall = Mock(return_value=[])