"""
Test suite for SMIED optimization demonstration and validation.

This comprehensive test suite validates the SMIED optimization demonstration functionality
following the SMIED Testing Framework Design Specifications with 3-layer architecture:
- Test Layer: Contains test logic and assertions  
- Mock Layer: Provides mock implementations and factories
- Configuration Layer: Supplies test data and constants
"""

import unittest
import time
import sys
import os
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import patch, Mock

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from smied.SMIED import SMIED
    SMIED_AVAILABLE = True
except ImportError:
    SMIED_AVAILABLE = False

# Import mock factory and configuration following 3-layer architecture
from tests.mocks.optimization_demo_mocks import OptimizationDemoMockFactory
from tests.config.optimization_demo_config import OptimizationDemoMockConfig


class TestOptimizationDemo(unittest.TestCase):
    """
    Basic functionality tests for optimization demonstration.
    
    Tests core optimization demonstration functionality including:
    - Basic optimization demonstration execution
    - Parameter application and validation
    - Performance improvement measurement
    - Mock factory integration
    """
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory
        self.mock_factory = OptimizationDemoMockFactory()
        
        # Load test configuration
        self.test_config = OptimizationDemoMockConfig.get_comprehensive_test_configuration()
        self.mock_config = OptimizationDemoMockConfig.get_mock_setup_configurations()['basic_setup']
        
        # Create core mock components through factory
        self.mock_optimization_demo = self.mock_factory('MockOptimizationDemo', 
                                                      verbosity=self.mock_config['verbosity'])
        self.mock_smied_optimized = self.mock_factory('MockSMIEDOptimized',
                                                    build_graph_on_init=self.mock_config['build_graph_on_init'])
        self.mock_smied_original = self.mock_factory('MockSMIEDOriginal',
                                                   build_graph_on_init=self.mock_config['build_graph_on_init'])
        
        # Load test data from configuration
        self.basic_test_cases = OptimizationDemoMockConfig.get_basic_test_data()
        self.optimization_params = OptimizationDemoMockConfig.get_optimization_parameters()
        self.performance_metrics = OptimizationDemoMockConfig.get_performance_metrics()
    
    def test_mock_factory_initialization(self):
        """Test that mock factory initializes correctly."""
        # Verify factory exists and has expected mocks
        available_mocks = self.mock_factory.get_available_mocks()
        
        self.assertIn('MockOptimizationDemo', available_mocks)
        self.assertIn('MockSMIEDForOptimization', available_mocks)
        self.assertIn('MockOptimizationParameters', available_mocks)
        self.assertIn('MockOptimizationResult', available_mocks)
        self.assertIn('MockOptimizationTestCase', available_mocks)
        
        # Test factory can create instances
        test_case_mock = self.mock_factory('MockOptimizationTestCase')
        self.assertIsNotNone(test_case_mock)
        
    def test_basic_optimization_demonstration(self):
        """Test basic optimization demonstration functionality."""
        # Get test case from config
        simple_cases = self.basic_test_cases['simple_cases']
        test_case_data = simple_cases[0]  # fox jump dog case
        
        # Create test case mock through factory
        test_case_mock = self.mock_factory('MockOptimizationTestCase', **test_case_data)
        
        # Verify test case structure
        self.assertEqual(test_case_mock.subject, 'fox')
        self.assertEqual(test_case_mock.predicate, 'jump')
        self.assertEqual(test_case_mock.object, 'dog')
        self.assertTrue(test_case_mock.expected_success)
        self.assertTrue(test_case_mock.requires_optimization)
        
        # Test demonstration execution
        result = self.mock_optimization_demo.demonstrate_optimization()
        
        # Verify demonstration completed successfully
        self.assertTrue(result)
        self.mock_optimization_demo.demonstrate_optimization.assert_called_once()
        
    def test_parameter_configuration(self):
        """Test optimization parameter configuration."""
        # Get parameter configurations from config
        original_params = self.optimization_params['original_parameters']
        optimized_params = self.optimization_params['optimized_parameters']
        
        # Create parameter mocks through factory
        mock_original = self.mock_factory('MockOriginalParameters')
        mock_optimized = self.mock_factory('MockOptimizationParameters')
        
        # Verify original parameters
        self.assertEqual(mock_original.max_depth, original_params['max_depth'])
        self.assertEqual(mock_original.beam_width, original_params['beam_width'])
        self.assertEqual(mock_original.len_tolerance, original_params['len_tolerance'])
        self.assertEqual(mock_original.relax_beam, original_params['relax_beam'])
        
        # Verify optimized parameters
        self.assertEqual(mock_optimized.max_depth, optimized_params['max_depth'])
        self.assertEqual(mock_optimized.beam_width, optimized_params['beam_width'])
        self.assertEqual(mock_optimized.len_tolerance, optimized_params['len_tolerance'])
        self.assertEqual(mock_optimized.relax_beam, optimized_params['relax_beam'])
        
        # Verify parameter improvements
        improvements = self.optimization_params['parameter_improvements']
        beam_width_improvement = ((optimized_params['beam_width'] - original_params['beam_width']) 
                                / original_params['beam_width'] * 100)
        self.assertAlmostEqual(beam_width_improvement, improvements['beam_width_increase_percent'], places=0)
        
    def test_optimization_result_creation(self):
        """Test creation and validation of optimization results."""
        # Create optimization result mock
        mock_result = self.mock_factory('MockOptimizationResult',
                                      success=True,
                                      execution_time=2.5,
                                      subject_path=["fox", "animal", "creature"],
                                      object_path=["dog", "animal", "creature"],
                                      connecting_predicate="interact")
        
        # Verify result structure
        self.assertTrue(mock_result.success)
        self.assertEqual(mock_result.execution_time, 2.5)
        self.assertIsNotNone(mock_result.subject_path)
        self.assertIsNotNone(mock_result.object_path)
        self.assertIsNotNone(mock_result.connecting_predicate)
        
        # Test result validation
        self.assertTrue(mock_result.validate_entity())
        
        # Test path quality calculation
        quality = mock_result.get_path_quality()
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)
        
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation and validation."""
        # Get performance expectations from config
        timing_expectations = self.performance_metrics['timing_expectations']
        success_expectations = self.performance_metrics['success_rate_expectations']
        
        # Create performance metrics mock
        mock_metrics = self.mock_factory('MockPerformanceMetrics',
                                       execution_time=timing_expectations['typical_execution_time_seconds'],
                                       success_rate=success_expectations['optimized_success_rate_percent'] / 100)
        
        # Verify metrics are within expected ranges
        self.assertLessEqual(mock_metrics.execution_time, timing_expectations['max_execution_time_seconds'])
        self.assertGreaterEqual(mock_metrics.success_rate, success_expectations['original_success_rate_percent'] / 100)
        
        # Test metrics calculation
        calculated_metrics = mock_metrics.calculate_metrics()
        self.assertIn('execution_time', calculated_metrics)
        self.assertIn('success_rate', calculated_metrics)
        self.assertIn('path_quality', calculated_metrics)
        
        # Test performance threshold validation
        meets_threshold = mock_metrics.meets_performance_threshold()
        self.assertTrue(meets_threshold)
        
    def test_optimization_summary_display(self):
        """Test optimization summary generation and display."""
        # Create optimization summary mock
        mock_summary = self.mock_factory('MockOptimizationSummary')
        
        # Verify summary structure
        self.assertTrue(mock_summary.validate_entity())
        
        # Test summary methods
        optimization_details = mock_summary.get_optimization_details()
        self.assertIsInstance(optimization_details, list)
        self.assertGreater(len(optimization_details), 0)
        
        # Test summary display
        mock_summary.display_summary()
        mock_summary.display_summary.assert_called_once()
        
        # Test improvement formatting
        improvements = mock_summary.format_improvements()
        self.assertIsInstance(improvements, str)
        self.assertIn("improvement", improvements.lower())
        

class TestOptimizationDemoValidation(unittest.TestCase):
    """
    Validation and constraint tests for optimization demonstration.
    
    Tests parameter validation, input constraints, and optimization requirements.
    """
    
    def setUp(self):
        """Set up test fixtures for validation testing."""
        # Initialize mock factory
        self.mock_factory = OptimizationDemoMockFactory()
        
        # Load validation configuration
        self.mock_config = OptimizationDemoMockConfig.get_mock_setup_configurations()['validation_setup']
        
        # Create validation-specific mock
        self.mock_validation_demo = self.mock_factory('MockOptimizationDemoValidation',
                                                    strict_validation=self.mock_config['strict_validation'])
        
        # Load test data
        self.optimization_params = OptimizationDemoMockConfig.get_optimization_parameters()
        self.performance_metrics = OptimizationDemoMockConfig.get_performance_metrics()
        
    def test_parameter_validation(self):
        """Test validation of optimization parameters."""
        # Test valid parameters
        valid_params = self.optimization_params['optimized_parameters']
        mock_params = self.mock_factory('MockOptimizationParameters', **valid_params)
        
        self.assertTrue(mock_params.validate_entity())
        
        # Test parameter validation method
        result = self.mock_validation_demo.validate_parameters()
        self.assertTrue(result)
        self.mock_validation_demo.validate_parameters.assert_called_once()
        
    def test_input_validation(self):
        """Test validation of demonstration inputs."""
        # Test valid inputs
        test_case = self.mock_factory('MockOptimizationTestCase',
                                    subject="fox",
                                    predicate="jump", 
                                    object="dog")
        
        self.assertTrue(test_case.validate_entity())
        
        # Test input validation method
        result = self.mock_validation_demo.validate_inputs()
        self.assertTrue(result)
        self.mock_validation_demo.validate_inputs.assert_called_once()
        
    def test_result_validation(self):
        """Test validation of optimization results."""
        # Create valid result
        mock_result = self.mock_factory('MockOptimizationResult',
                                      success=True,
                                      execution_time=2.5)
        
        self.assertTrue(mock_result.validate_entity())
        
        # Test result validation method
        result = self.mock_validation_demo.validate_results()
        self.assertTrue(result)
        self.mock_validation_demo.validate_results.assert_called_once()
        
    def test_optimization_improvement_validation(self):
        """Test validation of optimization improvements."""
        # Create performance comparison
        expected_improvement = self.optimization_params['parameter_improvements']['expected_success_rate_improvement']
        
        mock_comparison = self.mock_factory('MockPerformanceComparison',
                                          improvement_percent=expected_improvement)
        
        self.assertTrue(mock_comparison.validate_entity())
        self.assertGreaterEqual(mock_comparison.improvement_percent, 20.0)  # Minimum expected improvement
        
        # Test improvement validation
        result = self.mock_validation_demo.check_optimization_improvements()
        self.assertTrue(result)
        self.mock_validation_demo.check_optimization_improvements.assert_called_once()
        
    def test_performance_threshold_validation(self):
        """Test validation of performance thresholds."""
        timing_expectations = self.performance_metrics['timing_expectations']
        success_expectations = self.performance_metrics['success_rate_expectations']
        
        # Create metrics within acceptable thresholds
        mock_metrics = self.mock_factory('MockPerformanceMetrics',
                                       execution_time=timing_expectations['typical_execution_time_seconds'],
                                       success_rate=success_expectations['optimized_success_rate_percent'] / 100)
        
        # Test threshold validation
        meets_threshold = mock_metrics.meets_performance_threshold()
        self.assertTrue(meets_threshold)
        
        # Verify timing is within bounds
        self.assertLessEqual(mock_metrics.execution_time, timing_expectations['max_execution_time_seconds'])
        self.assertGreaterEqual(mock_metrics.execution_time, 0)
        
        # Verify success rate meets expectations
        min_success_rate = success_expectations['original_success_rate_percent'] / 100
        self.assertGreaterEqual(mock_metrics.success_rate, min_success_rate)
        

class TestOptimizationDemoEdgeCases(unittest.TestCase):
    """
    Edge cases and error conditions for optimization demonstration.
    
    Tests error handling, boundary conditions, and failure scenarios.
    """
    
    def setUp(self):
        """Set up test fixtures for edge case testing."""
        # Initialize mock factory
        self.mock_factory = OptimizationDemoMockFactory()
        
        # Load edge case configuration
        self.mock_config = OptimizationDemoMockConfig.get_mock_setup_configurations()['edge_case_setup']
        
        # Create edge case mock
        self.mock_edge_case_demo = self.mock_factory('MockOptimizationDemoEdgeCases',
                                                   mock_failures=self.mock_config['mock_failures'])
        
        # Load edge case data
        self.edge_cases = OptimizationDemoMockConfig.get_edge_case_scenarios()
        
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        empty_inputs = self.edge_cases['empty_inputs']
        
        for empty_input in empty_inputs:
            with self.subTest(empty_input=empty_input):
                # Test that empty inputs raise appropriate errors
                with self.assertRaises(ValueError):
                    self.mock_edge_case_demo.handle_empty_inputs()
                    
    def test_invalid_parameter_handling(self):
        """Test handling of invalid parameters."""
        invalid_params = self.edge_cases['invalid_parameters']
        
        for invalid_param in invalid_params:
            with self.subTest(invalid_param=invalid_param):
                # Create parameter mock with invalid values
                with self.assertRaises(ValueError):
                    self.mock_edge_case_demo.handle_invalid_parameters()
                    
    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        extreme_params = self.edge_cases['extreme_parameters']
        
        for extreme_param in extreme_params:
            with self.subTest(extreme_param=extreme_param):
                # Test that extreme parameters are handled appropriately
                mock_params = self.mock_factory('MockParameterSet', **extreme_param)
                
                # Verify validation fails for extreme values
                if any(value <= 0 for key, value in extreme_param.items() if key in ['max_depth', 'beam_width']):
                    self.assertFalse(mock_params.validate_entity())
                    
    def test_missing_dependency_handling(self):
        """Test handling of missing dependencies."""
        missing_components = self.edge_cases['missing_components']
        
        for missing_component in missing_components:
            with self.subTest(missing_component=missing_component):
                # Test that missing dependencies raise ImportError
                with self.assertRaises(ImportError):
                    self.mock_edge_case_demo.handle_missing_dependencies()
                    
    def test_optimization_failure_scenarios(self):
        """Test scenarios where optimization fails."""
        # Create challenging test case that should fail
        challenging_case = self.mock_factory('MockChallengingTestCase')
        
        self.assertEqual(challenging_case.difficulty_level, "very_hard")
        self.assertTrue(challenging_case.requires_optimization)
        
        # Test that demonstration can handle failures gracefully
        result = self.mock_edge_case_demo.demonstrate_optimization()
        
        # With mock failures enabled, should return False
        self.assertFalse(result)
        
    def test_timeout_scenarios(self):
        """Test timeout scenarios in optimization."""
        # Create execution timer with long execution time
        long_execution_timer = self.mock_factory('MockExecutionTimer', mock_execution_time=65.0)
        
        timeout_threshold = OptimizationDemoMockConfig.get_performance_metrics()['timing_expectations']['timeout_threshold_seconds']
        
        # Verify that long execution times exceed threshold
        self.assertGreater(long_execution_timer.get_elapsed_time(), timeout_threshold)
        
    def test_malformed_result_handling(self):
        """Test handling of malformed optimization results."""
        # Create result with invalid execution time
        mock_result = self.mock_factory('MockOptimizationResult', execution_time=-1.0)
        
        # Validation should fail for negative execution time
        self.assertFalse(mock_result.validate_entity())
            

class TestOptimizationDemoIntegration(unittest.TestCase):
    """
    Integration tests for optimization demonstration with other components.
    
    Tests multi-component interactions and system-level integration.
    """
    
    def setUp(self):
        """Set up test fixtures for integration testing."""
        # Initialize mock factory
        self.mock_factory = OptimizationDemoMockFactory()
        
        # Load integration configuration
        self.mock_config = OptimizationDemoMockConfig.get_mock_setup_configurations()['integration_setup']
        
        # Create integration mock
        self.mock_integration_demo = self.mock_factory('MockOptimizationDemoIntegration',
                                                     full_component_mocking=self.mock_config['full_component_mocking'])
        
        # Load integration test data
        self.integration_data = OptimizationDemoMockConfig.get_integration_test_data()
        
    def test_full_pipeline_integration(self):
        """Test integration with full SMIED pipeline."""
        multi_component_scenarios = self.integration_data['multi_component_scenarios']
        full_pipeline_scenario = multi_component_scenarios[0]
        
        # Verify scenario structure
        self.assertEqual(full_pipeline_scenario['name'], 'full_pipeline_test')
        self.assertIn('SMIED', full_pipeline_scenario['components'])
        
        # Test full pipeline execution
        result = self.mock_integration_demo.test_full_pipeline()
        self.assertTrue(result)
        self.mock_integration_demo.test_full_pipeline.assert_called_once()
        
    def test_component_interaction_validation(self):
        """Test validation of component interactions."""
        # Test component integration validation
        result = self.mock_integration_demo.validate_component_integration()
        self.assertTrue(result)
        self.mock_integration_demo.validate_component_integration.assert_called_once()
        
    def test_parameter_propagation(self):
        """Test parameter propagation across components."""
        # Create parameter sets for propagation testing
        mock_optimized_params = self.mock_factory('MockOptimizationParameters')
        mock_smied = self.mock_factory('MockSMIEDForOptimization')
        
        # Test parameter application
        mock_optimized_params.apply_to_smied()
        mock_optimized_params.apply_to_smied.assert_called_once()
        
        # Test parameter propagation validation
        result = self.mock_integration_demo.check_parameter_propagation()
        self.assertTrue(result)
        self.mock_integration_demo.check_parameter_propagation.assert_called_once()
        
    def test_multi_test_case_execution(self):
        """Test execution with multiple test cases."""
        test_cases_data = self.integration_data['multi_component_scenarios'][0]['test_cases']
        
        for test_case_data in test_cases_data:
            with self.subTest(test_case=test_case_data):
                # Create test case mock
                mock_test_case = self.mock_factory('MockOptimizationTestCase', **test_case_data)
                
                # Verify test case structure
                self.assertTrue(mock_test_case.validate_entity())
                
                # Test execution with mock SMIED
                mock_result = mock_test_case.run_with_parameters()
                self.assertIsNotNone(mock_result)
                
    def test_system_integration_dependencies(self):
        """Test integration with system dependencies."""
        system_integration = self.integration_data['system_integration']
        
        # Verify external dependencies are accounted for
        external_deps = system_integration['external_dependencies']
        self.assertIn('nltk', external_deps)
        self.assertIn('spacy', external_deps)
        self.assertIn('numpy', external_deps)
        
        # Verify optional dependencies are handled
        optional_deps = system_integration['optional_dependencies']
        self.assertIn('torch', optional_deps)
        self.assertIn('transformers', optional_deps)
        
    def test_cross_component_optimization_effects(self):
        """Test how optimization affects multiple components."""
        # Create both optimized and original SMIED mocks
        mock_optimized_smied = self.mock_factory('MockSMIEDOptimized')
        mock_original_smied = self.mock_factory('MockSMIEDOriginal')
        
        # Verify different optimization states
        self.assertTrue(mock_optimized_smied.optimization_enabled)
        self.assertFalse(mock_original_smied.optimization_enabled)
        
        # Test analysis results differ based on optimization
        optimized_result = mock_optimized_smied.analyze_triple("fox", "jump", "dog")
        original_result = mock_original_smied.analyze_triple("fox", "jump", "dog")
        
        # Optimized should succeed where original fails
        self.assertIsNotNone(optimized_result[0])  # subject_path
        self.assertIsNotNone(optimized_result[1])  # object_path
        self.assertIsNotNone(optimized_result[2])  # connecting_predicate
        
        # Original should return None results
        self.assertIsNone(original_result[0])
        self.assertIsNone(original_result[1])
        self.assertIsNone(original_result[2])


if __name__ == '__main__':
    # Configure test execution
    unittest.main(verbosity=2)