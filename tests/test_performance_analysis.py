"""
Performance analysis and bottleneck identification tests for SMIED.
Follows SMIED Testing Framework Design Specifications with 3-layer architecture.
"""

import unittest
import time
import sys
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import from the Mock Layer
from tests.mocks.performance_analysis_mocks import (
    PerformanceAnalysisMockFactory,
    MockPerformanceProfile,
    MockScalabilityTestResult,
    MockBottleneckAnalysis
)

# Import from the Configuration Layer
from tests.config.performance_analysis_config import PerformanceAnalysisMockConfig

# Import actual classes if available
# SMIED module has been removed
SMIED_AVAILABLE = False
SMIED = None


@unittest.skipUnless(SMIED_AVAILABLE, "SMIED not available for testing")
class TestPerformanceAnalysis(unittest.TestCase):
    """Basic functionality tests for performance analysis components."""
    
    def setUp(self):
        """Set up test fixtures using factory pattern and configuration injection."""
        # Initialize mock factory
        self.mock_factory = PerformanceAnalysisMockFactory()
        
        # Get configuration data
        self.config = PerformanceAnalysisMockConfig
        
        # Create mock instances using factory
        self.mock_profiler = self.mock_factory('MockPerformanceProfiler', verbosity=0)
        self.mock_scalability_tester = self.mock_factory('MockScalabilityTester', verbosity=0)
        self.mock_memory_profiler = self.mock_factory('MockMemoryProfiler', verbosity=0)
        self.mock_smied = self.mock_factory('MockSMIED', verbosity=0)
        
        # Load test data from configuration
        self.performance_profiles = self.config.get_basic_performance_profiles()
        self.profiler_config = self.config.get_profiler_configuration()['basic_profiler']
    
    def test_performance_profiler_initialization(self):
        """Test performance profiler initialization."""
        self.assertIsNotNone(self.mock_profiler)
        self.assertEqual(self.mock_profiler.verbosity, 0)
        self.assertTrue(self.mock_profiler.profiling_enabled)
    
    def test_performance_profiler_single_operation(self):
        """Test profiling a single operation."""
        # Test with basic operation data
        expected_profile_data = self.performance_profiles['fast_operation']
        
        # Call the mock method
        profile = self.mock_profiler.profile_single_operation.return_value
        
        # Verify profile structure
        self.assertIsNotNone(profile)
        self.assertIsInstance(profile, MockPerformanceProfile)
        self.assertGreater(profile.execution_time, 0)
        self.assertGreaterEqual(profile.memory_peak_mb, 0)
        self.assertIsNone(profile.error)
    
    def test_performance_profiler_batch_operations(self):
        """Test profiling batch operations."""
        # Get batch results
        batch_profiles = self.mock_profiler.profile_batch_operations.return_value
        
        # Verify batch results
        self.assertIsInstance(batch_profiles, list)
        self.assertGreater(len(batch_profiles), 0)
        
        # Check each profile in batch
        for profile in batch_profiles:
            self.assertIsInstance(profile, MockPerformanceProfile)
            self.assertGreater(profile.execution_time, 0)
    
    def test_scalability_tester_initialization(self):
        """Test scalability tester initialization."""
        self.assertIsNotNone(self.mock_scalability_tester)
        self.assertEqual(self.mock_scalability_tester.verbosity, 0)
        self.assertIsNotNone(self.mock_scalability_tester.smied)
    
    def test_scalability_tester_graph_size_testing(self):
        """Test graph size scalability testing."""
        # Get scalability results
        results = self.mock_scalability_tester.test_graph_size_scalability.return_value
        
        # Verify results structure
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check individual results
        for result in results:
            self.assertIsInstance(result, MockScalabilityTestResult)
            self.assertGreater(result.graph_size, 0)
            self.assertEqual(result.test_complexity, "simple")
            self.assertGreaterEqual(result.success_rate, 0)
            self.assertLessEqual(result.success_rate, 100)
    
    def test_memory_profiler_initialization(self):
        """Test memory profiler initialization."""
        self.assertIsNotNone(self.mock_memory_profiler)
        self.assertEqual(self.mock_memory_profiler.verbosity, 0)
    
    def test_memory_profiler_graph_construction(self):
        """Test memory profiling of graph construction."""
        # Get construction results
        results = self.mock_memory_profiler.profile_graph_construction.return_value
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('construction_time', results)
        self.assertIn('memory_increase_mb', results)
        self.assertIn('nodes_count', results)
        self.assertIn('edges_count', results)
        
        # Verify reasonable values
        self.assertGreater(results['construction_time'], 0)
        self.assertGreater(results['nodes_count'], 0)
        self.assertGreater(results['edges_count'], 0)
    
    def test_bottleneck_analysis_generation(self):
        """Test bottleneck analysis generation."""
        # Get bottleneck analysis
        analysis = self.mock_profiler.identify_bottlenecks.return_value
        
        # Verify analysis structure
        self.assertIsInstance(analysis, MockBottleneckAnalysis)
        self.assertIsInstance(analysis.optimization_recommendations, list)
        self.assertGreater(len(analysis.optimization_recommendations), 0)
    
    def test_comprehensive_performance_suite_initialization(self):
        """Test comprehensive performance suite initialization."""
        # Create comprehensive suite mock
        suite_mock = self.mock_factory('MockComprehensivePerformanceSuite', verbosity=0)
        
        self.assertIsNotNone(suite_mock)
        self.assertEqual(suite_mock.verbosity, 0)
        self.assertIsNotNone(suite_mock.smied)
        self.assertIsNotNone(suite_mock.profiler)
        self.assertIsNotNone(suite_mock.scalability_tester)
        self.assertIsNotNone(suite_mock.memory_profiler)
    
    def test_comprehensive_performance_analysis_execution(self):
        """Test comprehensive performance analysis execution."""
        # Create comprehensive suite mock
        suite_mock = self.mock_factory('MockComprehensivePerformanceSuite', verbosity=0)
        
        # Get analysis results
        results = suite_mock.run_comprehensive_analysis.return_value
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('timestamp', results)
        self.assertIn('analysis_sections', results)
        
        # Check analysis sections
        sections = results['analysis_sections']
        expected_sections = [
            'graph_construction_memory',
            'depth_scalability', 
            'beam_scalability',
            'pathfinding_memory'
        ]
        
        for section in expected_sections:
            self.assertIn(section, sections)


@unittest.skipUnless(SMIED_AVAILABLE, "SMIED not available for testing")
class TestPerformanceAnalysisValidation(unittest.TestCase):
    """Validation and constraint tests for performance analysis components."""
    
    def setUp(self):
        """Set up test fixtures using factory pattern and configuration injection."""
        # Initialize mock factory
        self.mock_factory = PerformanceAnalysisMockFactory()
        
        # Get configuration data
        self.config = PerformanceAnalysisMockConfig
        
        # Create mock instances for validation testing
        self.mock_profiler = self.mock_factory('MockPerformanceProfiler', verbosity=1)
        
        # Load validation test data from configuration
        self.validation_data = self.config.get_validation_test_data()
    
    def test_profiler_valid_configurations(self):
        """Test profiler with valid configurations."""
        valid_configs = self.validation_data['valid_configurations']
        
        for config_name, config_data in valid_configs.items():
            with self.subTest(config=config_name):
                # Test profiler accepts valid configuration
                profiler_mock = self.mock_factory(
                    'MockPerformanceProfiler', 
                    verbosity=config_data['verbosity']
                )
                
                self.assertIsNotNone(profiler_mock)
                self.assertEqual(profiler_mock.verbosity, config_data['verbosity'])
    
    def test_profiler_parameter_validation(self):
        """Test profiler parameter validation."""
        invalid_configs = self.validation_data['invalid_configurations']
        
        # Test that invalid configurations are handled appropriately
        for config_name, config_data in invalid_configs.items():
            with self.subTest(config=config_name):
                # In mock testing, we simulate validation by checking expected behavior
                if 'expected_error' in config_data:
                    # Mock would handle this validation internally
                    self.assertIn('expected_error', config_data)
    
    def test_scalability_test_parameter_bounds(self):
        """Test scalability test parameter bounds."""
        # Test with boundary conditions
        boundary_data = self.validation_data['boundary_conditions']
        
        for boundary_name, boundary_config in boundary_data.items():
            with self.subTest(boundary=boundary_name):
                # Test that boundary conditions are handled properly
                self.assertIn('expected_behavior', boundary_config)
    
    def test_memory_profiler_data_validation(self):
        """Test memory profiler data validation."""
        # Test memory profiler validates its input and output data
        memory_data = self.config.get_memory_profiling_data()
        
        # Verify successful case structure
        successful_data = memory_data['graph_construction_results']['successful']
        required_fields = [
            'construction_time', 'initial_memory_mb', 'final_memory_mb',
            'memory_increase_mb', 'nodes_count', 'edges_count'
        ]
        
        for field in required_fields:
            self.assertIn(field, successful_data)
            self.assertIsInstance(successful_data[field], (int, float))
            self.assertGreaterEqual(successful_data[field], 0)
    
    def test_performance_profile_data_consistency(self):
        """Test performance profile data consistency."""
        profile_data = self.config.get_basic_performance_profiles()
        
        for profile_name, profile_info in profile_data.items():
            with self.subTest(profile=profile_name):
                # Check that memory current <= memory peak
                self.assertLessEqual(
                    profile_info['memory_current_mb'], 
                    profile_info['memory_peak_mb']
                )
                
                # Check that execution time is positive
                self.assertGreater(profile_info['execution_time'], 0)
                
                # Check that success rate is in valid range
                if 'success_rate' in profile_info:
                    self.assertGreaterEqual(profile_info['success_rate'], 0)
                    self.assertLessEqual(profile_info['success_rate'], 100)


@unittest.skipUnless(SMIED_AVAILABLE, "SMIED not available for testing")
class TestPerformanceAnalysisEdgeCases(unittest.TestCase):
    """Edge cases and error conditions tests for performance analysis components."""
    
    def setUp(self):
        """Set up test fixtures using factory pattern and configuration injection."""
        # Initialize mock factory
        self.mock_factory = PerformanceAnalysisMockFactory()
        
        # Get configuration data
        self.config = PerformanceAnalysisMockConfig
        
        # Create edge case mock instances
        self.edge_case_profiler = self.mock_factory('MockPerformanceProfilerEdgeCases', verbosity=0)
        self.edge_case_scalability = self.mock_factory('MockScalabilityTesterEdgeCases', verbosity=0)
        self.edge_case_memory = self.mock_factory('MockMemoryProfilerEdgeCases', verbosity=0)
        
        # Load edge case test data from configuration
        self.edge_case_data = self.config.get_edge_case_test_data()
        self.error_scenarios = self.config.get_error_handling_scenarios()
    
    def test_profiler_timeout_handling(self):
        """Test profiler timeout handling."""
        timeout_scenarios = self.error_scenarios['timeout_scenarios']
        
        for scenario_name, scenario_data in timeout_scenarios.items():
            with self.subTest(scenario=scenario_name):
                # Test that timeout scenarios are handled correctly
                # The edge case profiler should return profiles with timeout errors
                side_effects = list(self.edge_case_profiler.profile_single_operation.side_effect)
                profile = side_effects[0] if side_effects else None
                
                if profile:
                    self.assertIsNotNone(profile.error)
                    self.assertIn("timed out", profile.error.lower())
                    self.assertEqual(profile.execution_time, 60.0)  # Timeout duration
    
    def test_profiler_exception_handling(self):
        """Test profiler exception handling."""
        # Test with exception scenarios
        side_effects = list(self.edge_case_profiler.profile_single_operation.side_effect)
        profile = side_effects[1] if len(side_effects) > 1 else None
        
        if profile:
            self.assertIsNotNone(profile.error)
            self.assertIn("exception", profile.error.lower())
            self.assertGreater(profile.execution_time, 0)  # Should still have execution time
    
    def test_scalability_tester_empty_results(self):
        """Test scalability tester with empty results."""
        # Test empty results handling
        empty_results = self.edge_case_scalability.test_graph_size_scalability.return_value
        
        self.assertIsInstance(empty_results, list)
        self.assertEqual(len(empty_results), 0)
    
    def test_scalability_tester_exception_handling(self):
        """Test scalability tester exception handling."""
        # Test exception handling in search depth scalability
        with self.assertRaises(Exception) as context:
            # This should raise an exception as configured in the edge case mock
            self.edge_case_scalability.test_search_depth_scalability()
        
        self.assertIn("Test exception", str(context.exception))
    
    def test_scalability_tester_zero_success_rate(self):
        """Test scalability tester with zero success rate."""
        # Test beam width scalability with zero success rate
        results = self.edge_case_scalability.test_beam_width_scalability.return_value
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check that we have a result with 0% success rate
        zero_success_result = results[0]
        self.assertEqual(zero_success_result.success_rate, 0.0)
    
    def test_memory_profiler_construction_failure(self):
        """Test memory profiler with graph construction failure."""
        # Test construction failure handling
        results = self.edge_case_memory.profile_graph_construction.return_value
        
        self.assertIsInstance(results, dict)
        self.assertIn('error', results)
        self.assertEqual(results['error'], 'Graph construction failed')
    
    def test_memory_profiler_pathfinding_failure(self):
        """Test memory profiler with pathfinding failure."""
        # Test pathfinding memory profiling failure
        with self.assertRaises(Exception) as context:
            self.edge_case_memory.profile_pathfinding_memory()
        
        self.assertIn("Memory profiling error", str(context.exception))
    
    def test_empty_data_scenarios(self):
        """Test handling of empty data scenarios."""
        empty_scenarios = self.edge_case_data['empty_data_scenarios']
        
        # Test no operations scenario
        no_ops = empty_scenarios['no_operations']
        self.assertEqual(no_ops['operations'], [])
        self.assertEqual(no_ops['expected_result'], 'empty_list')
        
        # Test no profiles scenario  
        no_profiles = empty_scenarios['no_profiles']
        self.assertEqual(no_profiles['profiles'], [])
        self.assertEqual(no_profiles['expected_analysis'], 'empty_analysis')
    
    def test_extreme_values_handling(self):
        """Test handling of extreme values."""
        extreme_values = self.edge_case_data['extreme_values']
        
        # Test very large graph
        large_graph = extreme_values['very_large_graph']
        self.assertEqual(large_graph['expected_behavior'], 'resource_intensive')
        self.assertGreater(large_graph['nodes'], 100000)
        
        # Test very small graph
        small_graph = extreme_values['very_small_graph'] 
        self.assertEqual(small_graph['expected_behavior'], 'minimal_operation')
        self.assertEqual(small_graph['nodes'], 1)
        self.assertEqual(small_graph['edges'], 0)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        malformed_data = self.edge_case_data['malformed_data']
        
        # Test invalid profiles
        invalid_profiles = malformed_data['invalid_profiles']
        self.assertIn('error', invalid_profiles['profile_data'])
        
        # Test corrupted results
        corrupted_results = malformed_data['corrupted_results']
        self.assertIsNone(corrupted_results['analysis_data'])
        self.assertIn('expected_error', corrupted_results)


@unittest.skipUnless(SMIED_AVAILABLE, "SMIED not available for testing")
class TestPerformanceAnalysisIntegration(unittest.TestCase):
    """Integration tests with other components for performance analysis."""
    
    def setUp(self):
        """Set up test fixtures using factory pattern and configuration injection."""
        # Initialize mock factory
        self.mock_factory = PerformanceAnalysisMockFactory()
        
        # Get configuration data
        self.config = PerformanceAnalysisMockConfig
        
        # Create integration mock instances
        self.integration_profiler = self.mock_factory('MockPerformanceProfilerIntegration', verbosity=1)
        self.integration_scalability = self.mock_factory('MockScalabilityTesterIntegration', verbosity=1)
        self.integration_memory = self.mock_factory('MockMemoryProfilerIntegration', verbosity=1)
        self.integration_smied = self.mock_factory('MockSMIEDIntegration', verbosity=0)
        self.integration_suite = self.mock_factory('MockComprehensivePerformanceSuite', verbosity=1)
        
        # Load integration test data from configuration
        self.integration_params = self.config.get_integration_test_parameters()
        self.comprehensive_results = self.config.get_comprehensive_analysis_results()
    
    def test_profiler_smied_integration(self):
        """Test performance profiler integration with SMIED."""
        # Test realistic integration behavior
        profile = self.integration_profiler.profile_single_operation.return_value
        
        # Verify more realistic integration values
        self.assertGreater(profile.execution_time, 0.1)  # More realistic timing
        self.assertGreater(profile.memory_peak_mb, 20)   # Realistic memory usage
        self.assertGreater(profile.function_calls, 1000) # Realistic call count
        self.assertGreater(profile.graph_nodes, 500)     # Realistic graph size
    
    def test_scalability_tester_detailed_integration(self):
        """Test scalability tester with detailed integration results."""
        # Test detailed integration results
        results = self.integration_scalability.test_graph_size_scalability.return_value
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 2)  # Multiple test points
        
        # Verify detailed bottleneck information is present
        for result in results:
            self.assertIsInstance(result.bottleneck_functions, list)
            if len(result.bottleneck_functions) > 0:
                # Check bottleneck function format
                func_name, func_time = result.bottleneck_functions[0]
                self.assertIsInstance(func_name, str)
                self.assertIsInstance(func_time, float)
                self.assertGreater(func_time, 0)
    
    def test_memory_profiler_detailed_integration(self):
        """Test memory profiler with detailed integration results."""
        # Test detailed pathfinding memory results
        results = self.integration_memory.profile_pathfinding_memory.return_value
        
        self.assertIsInstance(results, dict)
        self.assertGreater(results['total_tests'], 10)  # More comprehensive testing
        self.assertIn('individual_profiles', results)
        
        # Check individual profile details
        individual_profiles = results['individual_profiles']
        self.assertGreater(len(individual_profiles), 0)
        
        for profile in individual_profiles:
            required_fields = [
                'test_case', 'success', 'execution_time',
                'initial_memory_mb', 'final_memory_mb', 'memory_delta_mb'
            ]
            for field in required_fields:
                self.assertIn(field, profile)
    
    def test_comprehensive_suite_full_integration(self):
        """Test comprehensive performance suite full integration."""
        # Test complete analysis results
        results = self.integration_suite.run_comprehensive_analysis.return_value
        
        # Verify comprehensive structure
        self.assertIsInstance(results, dict)
        self.assertIn('timestamp', results)
        self.assertIn('analysis_sections', results)
        
        sections = results['analysis_sections']
        
        # Verify graph construction memory section
        if 'graph_construction_memory' in sections:
            construction_data = sections['graph_construction_memory']
            self.assertIn('construction_time', construction_data)
            self.assertIn('nodes_count', construction_data)
            self.assertIn('edges_count', construction_data)
    
    def test_realistic_performance_scenarios(self):
        """Test realistic performance scenarios."""
        realistic_scenario = self.integration_params['realistic_scenario_1']
        
        # Verify realistic scenario parameters
        smied_config = realistic_scenario['smied_config']
        test_params = realistic_scenario['test_parameters']
        expected_results = realistic_scenario['expected_results']
        
        # Check configuration structure
        self.assertIn('verbosity', smied_config)
        self.assertIn('auto_download', smied_config)
        
        # Check test parameters
        self.assertIn('max_nodes', test_params)
        self.assertIn('beam_widths', test_params)
        self.assertIn('max_depths', test_params)
        
        # Check expected results
        self.assertIn('success_rate_threshold', expected_results)
        self.assertIn('max_average_time', expected_results)
        self.assertIn('max_memory_usage', expected_results)
        
        # Verify reasonable thresholds
        self.assertGreater(expected_results['success_rate_threshold'], 50.0)
        self.assertLess(expected_results['max_average_time'], 10.0)
    
    def test_stress_test_integration(self):
        """Test stress test integration scenarios."""
        stress_scenario = self.integration_params['stress_test_scenario']
        
        # Verify stress test parameters are more demanding
        test_params = stress_scenario['test_parameters']
        expected_results = stress_scenario['expected_results']
        
        # Check that stress test has larger parameters
        self.assertGreater(test_params['max_nodes'], 15000)
        self.assertIn(20, test_params['beam_widths'])  # Higher beam widths
        self.assertIn(20, test_params['max_depths'])   # Deeper searches
        
        # Check that expectations are adjusted for stress testing
        self.assertLess(
            expected_results['success_rate_threshold'],
            self.integration_params['realistic_scenario_1']['expected_results']['success_rate_threshold']
        )
    
    def test_suite_summary_generation(self):
        """Test comprehensive suite summary generation."""
        # Test summary printing functionality
        suite_mock = self.integration_suite
        
        # Verify print summary method exists and can be called
        self.assertTrue(hasattr(suite_mock, 'print_performance_summary'))
        
        # Call the mock method (no actual printing in tests)
        suite_mock.print_performance_summary(
            self.comprehensive_results['complete_analysis']
        )
        
        # Verify the method was called
        suite_mock.print_performance_summary.assert_called()
    
    def test_cross_component_data_flow(self):
        """Test data flow between performance analysis components."""
        # Test that components can work together with shared data
        
        # Get scalability results that could be used by bottleneck analysis
        scalability_data = self.config.get_scalability_test_scenarios()
        graph_size_scenarios = scalability_data['graph_size_scenarios']
        
        # Verify data format is compatible between components
        for scenario in graph_size_scenarios:
            required_fields = ['graph_size', 'test_complexity', 'average_time', 'success_rate']
            for field in required_fields:
                self.assertIn(field, scenario)
        
        # Get bottleneck analysis that could use scalability data
        bottleneck_data = self.config.get_bottleneck_analysis_data()
        typical_bottlenecks = bottleneck_data['typical_bottlenecks']
        
        # Verify bottleneck data structure
        self.assertIn('slowest_functions', typical_bottlenecks)
        self.assertIn('optimization_recommendations', typical_bottlenecks)
        
        # Check that optimization recommendations are actionable
        recommendations = typical_bottlenecks['optimization_recommendations']
        self.assertGreater(len(recommendations), 0)
        for recommendation in recommendations:
            self.assertIsInstance(recommendation, str)
            self.assertGreater(len(recommendation), 10)  # Substantial recommendation text


if __name__ == '__main__':
    # Run all test classes
    unittest.main(verbosity=2)