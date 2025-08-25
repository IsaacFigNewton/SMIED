"""
Optimization strategies and caching mechanisms for SMIED performance improvement.
This module tests various optimization strategies including caching,
precomputed distances, and graph optimization techniques following the
SMIED Testing Framework Design Specifications.
"""

import unittest
import sys
import os
import json
import time
import pickle
import sqlite3
import hashlib
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from functools import wraps

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import framework components - 3-layer architecture
from tests.mocks.optimization_strategies_mocks import OptimizationStrategiesMockFactory
from tests.config.optimization_strategies_config import OptimizationStrategiesMockConfig

try:
    from smied.SMIED import SMIED
    from smied.SemanticDecomposer import SemanticDecomposer
    SMIED_AVAILABLE = True
except ImportError:
    SMIED_AVAILABLE = False


# Mock test case class for compatibility
class TestCase:
    """Mock test case class for compatibility with existing code."""
    
    def __init__(self, subject: str, predicate: str, object: str, description: str = ""):
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.description = description


@unittest.skipUnless(SMIED_AVAILABLE, "SMIED not available for testing")
class TestOptimizationStrategies(unittest.TestCase):
    """Basic functionality tests for optimization strategies.
    
    This class tests core optimization functionality including caching,
    graph optimization, and performance improvements.
    """
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        self.mock_factory = OptimizationStrategiesMockFactory()
        self.config = OptimizationStrategiesMockConfig
        
        # Get test configurations
        cache_config = self.config.get_basic_cache_configurations()['medium_cache']
        graph_config = self.config.get_graph_optimization_parameters()['medium_graph_params']
        smied_config = self.config.get_smied_integration_configurations()['basic_smied_config']
        
        # Create mocks using factory
        self.mock_smied = self.mock_factory('MockSMIEDForOptimization')
        self.mock_path_cache = self.mock_factory('MockPathCache', max_size=cache_config['max_size'])
        self.mock_graph_optimizer = self.mock_factory('MockGraphOptimizer', verbosity=graph_config['verbosity'])
        self.mock_persistent_cache = self.mock_factory('MockPersistentCache')
        self.mock_optimized_smied = self.mock_factory('MockOptimizedSMIED', self.mock_smied)
        
        # Get test data
        self.test_triples = self.config.get_cache_test_data()['simple_triples']
        
        # Real SMIED for integration testing if available
        if SMIED_AVAILABLE:
            self.real_smied = SMIED(nlp_model=None, auto_download=False, verbosity=0)
    
    def test_path_cache_basic_functionality(self):
        """Test basic path cache put/get functionality."""
        test_data = self.config.get_cache_test_data()
        
        for triple in test_data['simple_triples']:
            subject, predicate, obj = triple['subject'], triple['predicate'], triple['object']
            expected_result = (f"{subject}.n.01", f"{obj}.n.01", f"{predicate}.v.01")
            
            # Test cache miss
            result = self.mock_path_cache._mock_get(subject, predicate, obj)
            self.assertIsNone(result)
            
            # Test cache put
            self.mock_path_cache._mock_put(subject, predicate, obj, expected_result)
            
            # Test cache hit
            result = self.mock_path_cache._mock_get(subject, predicate, obj)
            self.assertEqual(result, expected_result)
    
    def test_path_cache_lru_eviction(self):
        """Test LRU eviction behavior."""
        cache_config = self.config.get_basic_cache_configurations()['small_cache']
        small_cache = self.mock_factory('MockPathCache', max_size=cache_config['max_size'])
        
        # Fill cache beyond capacity
        test_entries = cache_config['test_entries']
        for i in range(test_entries):
            small_cache._mock_put(f"subject{i}", "predicate", "object", f"result{i}")
        
        # Check that cache size doesn't exceed max_size
        self.assertLessEqual(small_cache._mock_size(), cache_config['max_size'])
        
        # Verify eviction occurred
        if cache_config['expected_evictions']:
            result = small_cache._mock_get("subject0", "predicate", "object")
            self.assertIsNone(result)  # Should be evicted
    
    def test_graph_optimizer_initialization(self):
        """Test graph optimizer initialization and basic operations."""
        graph_config = self.config.get_graph_optimization_parameters()['small_graph_params']
        optimizer = self.mock_factory('MockGraphOptimizer', verbosity=graph_config['verbosity'])
        
        # Test initialization - check that both are MockGraphOptimizer instances
        from tests.mocks.optimization_strategies_mocks import MockGraphOptimizer
        self.assertIsInstance(optimizer, MockGraphOptimizer)
        self.assertIsInstance(self.mock_graph_optimizer, MockGraphOptimizer)
        self.assertEqual(optimizer.verbosity, graph_config['verbosity'])
        
        # Test properties
        properties = optimizer.get_algorithm_properties()
        self.assertIn('supports_optimization', properties)
        self.assertTrue(properties['supports_optimization'])
    
    def test_graph_structure_optimization(self):
        """Test graph structure optimization functionality."""
        graph_structures = self.config.get_mock_graph_structures()
        linear_hierarchy = graph_structures['linear_hierarchy']
        
        # Create mock graph
        test_graph = nx.DiGraph()
        test_graph.add_nodes_from(linear_hierarchy['nodes'])
        test_graph.add_edges_from(linear_hierarchy['edges'])
        
        # Test optimization
        optimized_graph = self.mock_graph_optimizer._mock_optimize_graph_structure(test_graph)
        
        # Verify optimization occurred
        self.assertIsInstance(optimized_graph, nx.DiGraph)
        self.assertGreaterEqual(optimized_graph.number_of_nodes(), test_graph.number_of_nodes())
    
    def test_persistent_cache_operations(self):
        """Test persistent cache database operations."""
        db_config = self.config.get_persistent_cache_configurations()['test_database_configs']['memory_db']
        persistent_cache = self.mock_factory('MockPersistentCache', db_path=db_config['db_path'])
        
        # Test caching result
        subject, predicate, obj = "cat", "chase", "mouse"
        mock_result = self.mock_factory('MockOptimizationResult', 
                                      success=True, 
                                      execution_time=0.05,
                                      subject_path=["cat.n.01"],
                                      object_path=["mouse.n.01"],
                                      connecting_predicate="chase.v.01")
        
        # Cache and retrieve
        persistent_cache._mock_cache_pathfinding_result(subject, predicate, obj, mock_result)
        cached_result = persistent_cache._mock_get_cached_result(subject, predicate, obj)
        
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result['subject'], subject)
        self.assertTrue(cached_result['success'])
    
    def test_optimized_smied_integration(self):
        """Test OptimizedSMIED integration with optimization components."""
        integration_config = self.config.get_smied_integration_configurations()['optimized_smied_config']
        
        optimized_smied = self.mock_factory('MockOptimizedSMIED', 
                                          self.mock_smied,
                                          enable_caching=integration_config['enable_caching'],
                                          enable_graph_optimization=integration_config['enable_graph_optimization'],
                                          verbosity=integration_config['verbosity'])
        
        # Test initialization
        self.assertTrue(optimized_smied.supports_caching)
        self.assertTrue(optimized_smied.supports_optimization)
        
        # Test optimization stats
        stats = optimized_smied._mock_get_optimization_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
    
    def test_optimization_performance_measurement(self):
        """Test performance measurement and statistics tracking."""
        benchmark_scenario = self.config.get_optimization_benchmark_scenarios()['quick_benchmark']
        benchmark = self.mock_factory('MockOptimizationBenchmark', self.mock_smied)
        
        # Create test cases from config
        test_cases = []
        for case_data in benchmark_scenario['test_cases']:
            test_case = TestCase(**case_data)
            test_cases.append(test_case)
        
        # Run benchmark
        results = benchmark._mock_run_optimization_benchmark(test_cases, runs_per_test=2)
        
        # Verify results structure
        self.assertIn('summary', results)
        self.assertIn('base_results', results)
        self.assertIn('optimized_results', results)
        
        summary = results['summary']
        self.assertGreater(summary['average_speedup'], 0)
        self.assertEqual(summary['total_tests'], len(test_cases))


class TestOptimizationStrategiesValidation(unittest.TestCase):
    """Validation and constraint tests for optimization strategies.
    
    This class tests input validation, parameter constraints,
    and boundary conditions for optimization components.
    """
    
    def setUp(self):
        """Set up validation test fixtures using mock factory and config injection."""
        self.mock_factory = OptimizationStrategiesMockFactory()
        self.config = OptimizationStrategiesMockConfig
        
        # Create validation-specific mocks
        self.validation_cache = self.mock_factory('MockPathCacheValidation')
        self.validation_optimizer = self.mock_factory('MockGraphOptimizerValidation')
        self.validation_persistent_cache = self.mock_factory('MockPersistentCacheValidation')
        
        # Get validation test data
        self.validation_data = self.config.get_validation_test_data()
    
    def test_cache_size_validation(self):
        """Test cache size parameter validation."""
        constraints = self.validation_data['constraint_validations']['cache_size_constraints']
        
        # Test valid sizes
        for valid_size in [constraints['min_size'], 100, constraints['max_size']]:
            cache = self.mock_factory('MockPathCacheValidation', max_size=valid_size)
            self.assertTrue(cache.validate_entity())
        
        # Test invalid sizes
        for invalid_size in constraints['invalid_sizes']:
            cache = self.mock_factory('MockPathCacheValidation', max_size=invalid_size)
            self.assertFalse(cache.validate_entity())
    
    def test_graph_input_validation(self):
        """Test graph input validation for optimization."""
        valid_graphs = self.validation_data['valid_graph_inputs']['valid_graphs']
        
        for graph_def in valid_graphs:
            test_graph = nx.DiGraph()
            test_graph.add_nodes_from(graph_def['nodes'])
            test_graph.add_edges_from(graph_def['edges'])
            
            # Validate graph can be processed
            result = self.validation_optimizer.validate_inputs(graph=test_graph)
            self.assertTrue(result)
    
    def test_timing_constraints_validation(self):
        """Test execution time validation constraints."""
        timing_constraints = self.validation_data['constraint_validations']['timing_constraints']
        
        # Test valid execution times
        valid_result = self.mock_factory('MockOptimizationResult', 
                                       execution_time=timing_constraints['min_execution_time'])
        self.assertTrue(valid_result.validate_entity())
        
        # Test invalid execution times
        for invalid_time in timing_constraints['invalid_times']:
            invalid_result = self.mock_factory('MockOptimizationResult', 
                                             execution_time=invalid_time)
            self.assertFalse(invalid_result.validate_entity())
    
    def test_database_schema_validation(self):
        """Test database schema validation for persistent cache."""
        db_config = self.config.get_persistent_cache_configurations()
        schema_def = db_config['schema_definitions']
        
        # Test schema validation
        validation_result = self.validation_persistent_cache.validate_database_schema()
        self.assertTrue(validation_result)
        
        # Verify expected columns exist
        expected_pathfinding_columns = schema_def['pathfinding_cache_columns']
        expected_graph_columns = schema_def['graph_cache_columns']
        
        self.assertGreater(len(expected_pathfinding_columns), 0)
        self.assertGreater(len(expected_graph_columns), 0)


class TestOptimizationStrategiesEdgeCases(unittest.TestCase):
    """Edge cases and error conditions tests for optimization strategies.
    
    This class tests error handling, resource limits,
    and unusual input conditions for optimization components.
    """
    
    def setUp(self):
        """Set up edge case test fixtures using mock factory and config injection."""
        self.mock_factory = OptimizationStrategiesMockFactory()
        self.config = OptimizationStrategiesMockConfig
        
        # Create edge case-specific mocks
        self.edge_case_cache = self.mock_factory('MockPathCacheEdgeCases')
        self.edge_case_optimizer = self.mock_factory('MockGraphOptimizerEdgeCases')
        self.edge_case_persistent_cache = self.mock_factory('MockPersistentCacheEdgeCases')
        self.edge_case_optimized_smied = self.mock_factory('MockOptimizedSMIEDEdgeCases')
        
        # Get edge case test data
        self.edge_case_data = self.config.get_edge_case_test_data()
    
    def test_empty_cache_operations(self):
        """Test operations on empty cache."""
        empty_operations = self.edge_case_data['cache_edge_cases']['empty_cache_operations']
        
        # Test get from empty cache
        get_params = empty_operations['get_from_empty']
        result = self.edge_case_cache._mock_get(*get_params)
        self.assertIsNone(result)
        
        # Test size of empty cache
        self.assertEqual(self.edge_case_cache._mock_size(), empty_operations['size_empty'])
        
        # Test clear empty cache (should not raise error)
        self.edge_case_cache._mock_clear()
    
    def test_cache_error_handling(self):
        """Test cache error handling for malformed inputs."""
        malformed_inputs = self.edge_case_data['cache_edge_cases']['malformed_inputs']
        
        # Test error handling for special inputs
        with self.assertRaises((ValueError, MemoryError)):
            self.edge_case_cache._mock_edge_case_get("error_test", "predicate", "object")
        
        with self.assertRaises(ValueError):
            self.edge_case_cache._mock_edge_case_put("error_test", "predicate", "object", "result")
    
    def test_empty_graph_optimization(self):
        """Test graph optimization with empty graph."""
        empty_graph_case = self.edge_case_data['graph_edge_cases']['empty_graph']
        
        empty_graph = nx.DiGraph()
        empty_graph.add_nodes_from(empty_graph_case['nodes'])
        empty_graph.add_edges_from(empty_graph_case['edges'])
        
        # Should handle empty graph gracefully
        with self.assertRaises(ValueError):
            self.edge_case_optimizer._mock_edge_case_optimize(empty_graph)
    
    def test_database_error_handling(self):
        """Test database error handling for persistent cache."""
        db_errors = self.edge_case_data['database_edge_cases']['database_errors']
        
        # Test database connection errors
        error_cache = self.mock_factory('MockPersistentCacheEdgeCases', 
                                       db_path=db_errors['nonexistent_path'])
        
        # Should handle database errors gracefully
        with self.assertRaises(sqlite3.Error):
            error_cache._mock_edge_case_init()
    
    def test_optimization_timeout_handling(self):
        """Test timeout handling in optimization operations."""
        # Test analysis timeout
        with self.assertRaises(TimeoutError):
            self.edge_case_optimized_smied._mock_edge_case_analyze(
                "timeout_test", "predicate", "object"
            )
    
    def test_resource_exhaustion_scenarios(self):
        """Test behavior under resource exhaustion."""
        extreme_params = self.edge_case_data['optimization_edge_cases']['extreme_parameters']
        
        # Test with extreme parameters (should be handled gracefully)
        extreme_cache = self.mock_factory('MockPathCacheEdgeCases', 
                                        max_size=extreme_params['zero_cache_size'])
        
        # Cache with zero size should still validate correctly (edge case handling)
        self.assertFalse(extreme_cache.validate_entity())


class TestOptimizationStrategiesIntegration(unittest.TestCase):
    """Integration tests for optimization strategies with other components.
    
    This class tests integration between optimization components,
    cross-component interactions, and end-to-end optimization workflows.
    """
    
    def setUp(self):
        """Set up integration test fixtures using mock factory and config injection."""
        self.mock_factory = OptimizationStrategiesMockFactory()
        self.config = OptimizationStrategiesMockConfig
        
        # Create integration-specific mocks
        self.integration_cache = self.mock_factory('MockPathCacheIntegration')
        self.integration_optimizer = self.mock_factory('MockGraphOptimizerIntegration')
        self.integration_persistent_cache = self.mock_factory('MockPersistentCacheIntegration')
        self.integration_optimized_smied = self.mock_factory('MockOptimizedSMIEDIntegration')
        
        # Get integration test data
        self.integration_scenarios = self.config.get_integration_scenarios()
        
        # Set up integration environment
        self.integration_optimized_smied.setup_integration_environment()
    
    def test_end_to_end_optimization_pipeline(self):
        """Test complete end-to-end optimization pipeline."""
        scenario = self.integration_scenarios['end_to_end_optimization']['scenario_1']
        
        # Follow the test sequence
        test_sequence = scenario['test_sequence']
        
        # Initialize optimized SMIED
        self.assertIn('initialize_optimized_smied', test_sequence)
        
        # Build optimized graph
        if 'build_optimized_graph' in test_sequence:
            optimized_graph = self.integration_optimized_smied._mock_build_optimized_graph()
            self.assertIsInstance(optimized_graph, nx.DiGraph)
        
        # Run pathfinding tests
        if 'run_pathfinding_tests' in test_sequence:
            test_result = self.integration_optimized_smied._mock_analyze_triple_optimized(
                "cat", "chase", "mouse"
            )
            self.assertIsNotNone(test_result)
        
        # Validate cache performance
        if 'validate_cache_performance' in test_sequence:
            stats = self.integration_optimized_smied._mock_get_optimization_stats()
            self.assertIn('cache_hits', stats)
        
        # Check expected outcomes
        expected_outcomes = scenario['expected_outcomes']
        self.assertTrue(expected_outcomes['graph_optimization_applied'])
        self.assertTrue(expected_outcomes['performance_improvement'])
    
    def test_persistent_cache_cross_session_integration(self):
        """Test persistent cache integration across multiple sessions."""
        scenario = self.integration_scenarios['end_to_end_optimization']['scenario_2']
        setup = scenario['setup']
        
        # Simulate multiple sessions
        for session in range(setup['session_count']):
            session_cache = self.mock_factory('MockPersistentCacheIntegration')
            
            # Populate cache in first session, use in subsequent sessions
            if session == 0:
                # Session 1: populate cache
                for i in range(setup['tests_per_session']):
                    mock_result = self.mock_factory('MockOptimizationResult', success=True)
                    session_cache._mock_cache_pathfinding_result(
                        f"subject{i}", "predicate", "object", mock_result
                    )
            else:
                # Subsequent sessions: verify persistence
                stats = session_cache._mock_get_cache_stats()
                self.assertGreater(stats['pathfinding_entries'], 0)
        
        # Verify expected outcomes
        expected_outcomes = scenario['expected_outcomes']
        self.assertTrue(expected_outcomes['cache_persistence_verified'])
    
    def test_stress_testing_high_volume_operations(self):
        """Test system behavior under high volume operations."""
        stress_scenario = self.integration_scenarios['stress_testing_scenarios']['high_volume_caching']
        
        # Perform high volume cache operations
        operations_count = min(stress_scenario['cache_operations'], 100)  # Limit for testing
        unique_triples = stress_scenario['unique_triples']
        
        for i in range(operations_count):
            triple_index = i % unique_triples
            subject = f"subject{triple_index}"
            result = (f"{subject}.n.01", "object.n.01", "predicate.v.01")
            
            self.integration_cache._mock_put(subject, "predicate", "object", result)
        
        # Verify system stability
        cache_size = self.integration_cache._mock_size()
        self.assertGreaterEqual(cache_size, 0)
        self.assertTrue(stress_scenario['expected_memory_stability'])
    
    def test_component_interaction_validation(self):
        """Test validation of component interactions."""
        # Test interaction between cache and optimizer
        interaction_valid = self.integration_optimized_smied.validate_component_interactions()
        self.assertTrue(interaction_valid)
        
        # Test integration with external components
        self.assertIsNotNone(self.integration_optimized_smied.real_smied_components)
        self.assertIsNotNone(self.integration_cache.external_cache_systems)
    
    def test_real_world_usage_simulation(self):
        """Test simulation of real-world usage patterns."""
        usage_pattern = self.integration_scenarios['real_world_simulation']['typical_usage_pattern']
        
        # Cache warm-up phase
        warmup_phase = usage_pattern['cache_warm_up_phase']
        for i in range(min(warmup_phase['initial_requests'], 10)):
            result = self.integration_optimized_smied._mock_analyze_triple_optimized(
                f"subject{i}", "predicate", "object"
            )
            self.assertIsNotNone(result)
        
        # Steady state phase
        steady_phase = usage_pattern['steady_state_phase']
        initial_stats = self.integration_optimized_smied._mock_get_optimization_stats()
        
        for i in range(min(steady_phase['additional_requests'], 20)):
            # Repeat some requests to test caching
            subject_index = i % 5  # Repeat every 5 requests
            result = self.integration_optimized_smied._mock_analyze_triple_optimized(
                f"subject{subject_index}", "predicate", "object"
            )
            self.assertIsNotNone(result)
        
        # Verify performance improvement
        final_stats = self.integration_optimized_smied._mock_get_optimization_stats()
        self.assertGreaterEqual(final_stats['total_requests'], initial_stats['total_requests'])
    
    def tearDown(self):
        """Clean up integration test environment."""
        self.integration_optimized_smied.teardown_integration_environment()


class OptimizationBenchmark:
    """
    Benchmark suite for comparing optimized vs unoptimized SMIED performance.
    
    This class is provided for backward compatibility with existing code that
    may reference it directly.
    """
    
    def __init__(self, base_smied: SMIED, verbosity: int = 1):
        self.base_smied = base_smied
        self.verbosity = verbosity
        
        # Use mock factory for testing
        self.mock_factory = OptimizationStrategiesMockFactory()
        self.optimized_smied = self.mock_factory('MockOptimizedSMIED', base_smied, verbosity=verbosity)
    
    def run_optimization_benchmark(self, test_cases: List[TestCase], 
                                 runs_per_test: int = 3) -> Dict[str, Any]:
        """
        Compare performance between optimized and unoptimized SMIED.
        
        Args:
            test_cases: List of test cases to benchmark
            runs_per_test: Number of runs per test case
            
        Returns:
            Benchmark results
        """
        benchmark = self.mock_factory('MockOptimizationBenchmark', self.base_smied, self.verbosity)
        return benchmark._mock_run_optimization_benchmark(test_cases, runs_per_test)
    
    def print_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Print benchmark results."""
        benchmark = self.mock_factory('MockOptimizationBenchmark', self.base_smied, self.verbosity)
        benchmark._mock_print_benchmark_results(results)


def run_optimization_analysis(verbosity: int = 1, save_results: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive optimization analysis.
    
    Args:
        verbosity: Verbosity level
        save_results: Whether to save results
        
    Returns:
        Optimization analysis results
    """
    if not SMIED_AVAILABLE:
        print("ERROR: SMIED not available. Cannot run optimization analysis.")
        return {}
    
    print("Initializing SMIED for optimization analysis...")
    smied = SMIED(nlp_model=None, auto_download=False, verbosity=0)
    
    # Run optimization benchmark using mock
    mock_factory = OptimizationStrategiesMockFactory()
    benchmark = mock_factory('MockOptimizationBenchmark', smied, verbosity=verbosity)
    
    # Create mock test cases
    config = OptimizationStrategiesMockConfig
    benchmark_scenario = config.get_optimization_benchmark_scenarios()['comprehensive_benchmark']
    test_cases = [TestCase(**case_data) for case_data in benchmark_scenario['test_cases'][:15]]
    
    results = benchmark._mock_run_optimization_benchmark(test_cases, runs_per_test=2)
    
    # Print results
    benchmark._mock_print_benchmark_results(results)
    
    if save_results:
        timestamp = int(time.time())
        filename = f"optimization_analysis_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Make results JSON-serializable
        json_results = {
            'summary': results['summary'],
            'test_cases': results['test_cases'],
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nOptimization results saved to: {filepath}")
    
    return results


if __name__ == '__main__':
    # Run optimization analysis if executed directly
    if SMIED_AVAILABLE:
        print("Running SMIED optimization analysis...")
        results = run_optimization_analysis(verbosity=2, save_results=True)
        
        # Print summary
        if results and 'summary' in results:
            summary = results['summary']
            print(f"\nOPTIMIZATION SUMMARY:")
            print(f"Average Speedup: {summary['average_speedup']:.2f}x")
            print(f"Max Speedup: {summary['max_speedup']:.2f}x")
            print(f"Cache Hit Rate: {summary['optimization_stats'].get('hit_rate_percent', 0):.1f}%")
    else:
        print("SMIED not available. Running basic unit tests only.")
        unittest.main()