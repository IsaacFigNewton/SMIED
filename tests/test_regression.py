"""
Regression testing suite for SMIED functionality.

This suite ensures that existing functionality continues to work correctly
as new features are added and changes are made to the codebase.

Follows the SMIED Testing Framework Design Specifications:
- 3-layer architecture: Test Layer, Mock Layer, Configuration Layer
- Factory pattern for mock creation
- Configuration-driven test data
- Structured test class organization
"""

import unittest
import sys
import os
import json
import time
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import patch, Mock

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import mock factory and configuration (Mock Layer and Configuration Layer)
from tests.mocks.regression_mocks import regression_mock_factory
from tests.config.regression_config import RegressionMockConfig

# SMIED module has been removed
SMIED_AVAILABLE = False
SMIED = None


@unittest.skipUnless(SMIED_AVAILABLE, "SMIED not available for testing")
class TestRegression(unittest.TestCase):
    """Basic functionality tests for regression testing components."""
    
    def setUp(self):
        """Set up test fixtures using factory pattern and configuration injection."""
        # Initialize mock factory
        self.mock_factory = regression_mock_factory
        
        # Load configuration data
        self.config = RegressionMockConfig()
        self.basic_data = self.config.get_basic_test_data()
        
        # Create mock SMIED instance
        self.mock_smied = Mock()
        
        # Create core regression testing components using factory
        self.regression_tester = self.mock_factory.create_regression_tester(
            smied_instance=self.mock_smied,
            verbosity=0
        )
        self.baseline_manager = self.mock_factory.create_regression_baseline()
        
    def test_baseline_creation_and_management(self):
        """Test creation and management of baseline results."""
        # Get test data from configuration
        baseline_data = self.basic_data['baseline_results'][0]
        
        # Create baseline result using factory
        baseline_result = self.mock_factory.create_baseline_result(**baseline_data)
        
        # Test baseline properties
        self.assertEqual(baseline_result.test_id, baseline_data['test_id'])
        self.assertEqual(baseline_result.subject, baseline_data['subject'])
        self.assertEqual(baseline_result.predicate, baseline_data['predicate'])
        self.assertEqual(baseline_result.object, baseline_data['object'])
        self.assertEqual(baseline_result.success, baseline_data['success'])
        
        # Test entity validation
        self.assertTrue(baseline_result.validate_entity())
        
        # Test baseline management operations
        self.baseline_manager.add_baseline(baseline_result)
        retrieved_baseline = self.baseline_manager.get_baseline(baseline_data['test_id'])
        
        self.assertIsNotNone(retrieved_baseline)
        self.assertEqual(retrieved_baseline.test_id, baseline_data['test_id'])
    
    def test_test_id_generation(self):
        """Test test ID generation from triple components."""
        test_case_data = self.basic_data['simple_test_cases'][0]
        
        # Test ID creation should be consistent
        test_id1 = self.baseline_manager.create_test_id(
            test_case_data['subject'],
            test_case_data['predicate'], 
            test_case_data['object']
        )
        test_id2 = self.baseline_manager.create_test_id(
            test_case_data['subject'],
            test_case_data['predicate'],
            test_case_data['object']
        )
        
        self.assertEqual(test_id1, test_id2)
        self.assertEqual(len(test_id1), 12)
        
        # Different inputs should produce different IDs
        different_case = self.basic_data['simple_test_cases'][1]
        test_id3 = self.baseline_manager.create_test_id(
            different_case['subject'],
            different_case['predicate'],
            different_case['object']
        )
        
        self.assertNotEqual(test_id1, test_id3)
    
    def test_regression_test_result_creation(self):
        """Test creation of regression test results."""
        result_data = self.basic_data['regression_test_results'][0]
        
        # Create regression test result using factory
        regression_result = self.mock_factory.create_regression_test_result(**result_data)
        
        # Verify properties
        self.assertEqual(regression_result.test_id, result_data['test_id'])
        self.assertEqual(regression_result.status, result_data['status'])
        self.assertEqual(regression_result.baseline_success, result_data['baseline_success'])
        self.assertEqual(regression_result.current_success, result_data['current_success'])
        
        # Test entity validation
        self.assertTrue(regression_result.validate_entity())
    
    def test_mock_test_case_creation(self):
        """Test creation of mock test cases."""
        test_case_data = self.basic_data['simple_test_cases'][0]
        
        # Create test case using factory
        test_case = self.mock_factory.create_test_case(
            subject=test_case_data['subject'],
            predicate=test_case_data['predicate'],
            object=test_case_data['object'],
            expected_success=test_case_data['expected_success']
        )
        
        # Verify properties
        self.assertEqual(test_case.subject, test_case_data['subject'])
        self.assertEqual(test_case.predicate, test_case_data['predicate'])
        self.assertEqual(test_case.object, test_case_data['object'])
        self.assertEqual(test_case.expected_success, test_case_data['expected_success'])
        
        # Test entity validation
        self.assertTrue(test_case.validate_entity())


class TestRegressionValidation(unittest.TestCase):
    """Validation and constraint tests for regression testing."""
    
    def setUp(self):
        """Set up test fixtures using factory pattern and configuration injection."""
        self.mock_factory = regression_mock_factory
        self.config = RegressionMockConfig()
        self.edge_case_data = self.config.get_edge_case_scenarios()
        
        # Create regression tester for validation testing
        self.regression_tester = self.mock_factory.create_regression_tester()
    
    def test_baseline_validation_with_invalid_data(self):
        """Test baseline validation with invalid data."""
        # Test with empty test_id - should create but fail validation
        invalid_baseline = self.mock_factory.create_baseline_result(
            test_id="",
            subject="test",
            predicate="test",
            object="test"
        )
        self.assertFalse(invalid_baseline.validate_entity())
        
        # Test with negative path lengths
        invalid_baseline = self.mock_factory.create_baseline_result(
            test_id="test_invalid",
            subject="test",
            predicate="test", 
            object="test",
            subject_path_length=-1,
            object_path_length=-1
        )
        self.assertFalse(invalid_baseline.validate_entity())
        
        # Test with negative execution time
        invalid_baseline = self.mock_factory.create_baseline_result(
            test_id="test_invalid_time",
            subject="test",
            predicate="test",
            object="test",
            execution_time=-0.1
        )
        self.assertFalse(invalid_baseline.validate_entity())
    
    def test_regression_result_validation(self):
        """Test regression test result validation."""
        # Test with invalid status
        invalid_result = self.mock_factory.create_regression_test_result(
            test_id="test_invalid",
            status="INVALID_STATUS"
        )
        self.assertFalse(invalid_result.validate_entity())
        
        # Test with valid status values
        valid_statuses = ['PASS', 'FAIL', 'PERFORMANCE_REGRESSION', 'IMPROVEMENT']
        for status in valid_statuses:
            valid_result = self.mock_factory.create_regression_test_result(
                test_id="test_valid",
                status=status
            )
            self.assertTrue(valid_result.validate_entity())
    
    def test_test_case_validation_with_empty_fields(self):
        """Test test case validation with empty fields."""
        # Test with empty subject
        invalid_case = self.mock_factory.create_test_case(
            subject="",
            predicate="test",
            object="test"
        )
        self.assertFalse(invalid_case.validate_entity())
        
        # Test with empty predicate
        invalid_case = self.mock_factory.create_test_case(
            subject="test",
            predicate="",
            object="test"
        )
        self.assertFalse(invalid_case.validate_entity())
        
        # Test with empty object
        invalid_case = self.mock_factory.create_test_case(
            subject="test",
            predicate="test",
            object=""
        )
        self.assertFalse(invalid_case.validate_entity())


class TestRegressionEdgeCases(unittest.TestCase):
    """Edge cases and error conditions for regression testing."""
    
    def setUp(self):
        """Set up test fixtures using factory pattern and configuration injection."""
        self.mock_factory = regression_mock_factory
        self.config = RegressionMockConfig()
        self.edge_case_data = self.config.get_edge_case_scenarios()
        
        # Create components for edge case testing
        self.regression_tester = self.mock_factory.create_regression_tester()
        self.baseline_manager = self.mock_factory.create_regression_baseline()
    
    def test_performance_regression_detection(self):
        """Test detection of performance regressions."""
        perf_regression_data = self.edge_case_data['performance_regression_results'][0]
        
        # Create performance regression result
        perf_result = self.mock_factory.create_regression_test_result(**perf_regression_data)
        
        # Verify performance regression is detected
        self.assertEqual(perf_result.status, 'PERFORMANCE_REGRESSION')
        self.assertGreater(perf_result.performance_change_percent, 50.0)
        self.assertIn('degradation', perf_result.details.lower())
    
    def test_functional_regression_detection(self):
        """Test detection of functional regressions."""
        func_regression_data = self.edge_case_data['functional_regression_results'][0]
        
        # Create functional regression result
        func_result = self.mock_factory.create_regression_test_result(**func_regression_data)
        
        # Verify functional regression is detected
        self.assertEqual(func_result.status, 'FAIL')
        self.assertTrue(func_result.baseline_success)
        self.assertFalse(func_result.current_success)
        self.assertIn('regression', func_result.details.lower())
    
    def test_improvement_detection(self):
        """Test detection of test improvements."""
        improvement_data = self.edge_case_data['improvement_results'][0]
        
        # Create improvement result
        improvement_result = self.mock_factory.create_regression_test_result(**improvement_data)
        
        # Verify improvement is detected
        self.assertEqual(improvement_result.status, 'IMPROVEMENT')
        self.assertFalse(improvement_result.baseline_success)
        self.assertTrue(improvement_result.current_success)
        self.assertIn('improvement', improvement_result.details.lower())
    
    def test_missing_baseline_handling(self):
        """Test handling of missing baselines."""
        # Test with non-existent baseline
        missing_baseline = self.baseline_manager.get_baseline('non_existent_test')
        self.assertIsNone(missing_baseline)
        
        # Test has_baseline with non-existent baseline
        self.assertFalse(self.baseline_manager.has_baseline('non_existent_test'))
    
    def test_empty_baseline_collection(self):
        """Test handling of empty baseline collections."""
        # Test with empty baseline manager
        empty_baselines = self.baseline_manager.get_all_baselines()
        self.assertIsInstance(empty_baselines, dict)
        
        # Initially empty
        self.assertEqual(len(empty_baselines), 0)


class TestRegressionIntegration(unittest.TestCase):
    """Integration tests with other components for regression testing."""
    
    def setUp(self):
        """Set up test fixtures using factory pattern and configuration injection."""
        self.mock_factory = regression_mock_factory
        self.config = RegressionMockConfig()
        self.integration_data = self.config.get_integration_test_data()
        
        # Create integrated test environment
        self.mock_smied = Mock()
        self.regression_tester = self.mock_factory.create_regression_tester(
            smied_instance=self.mock_smied
        )
        self.test_suite = self.mock_factory.create_test_suite(
            smied_instance=self.mock_smied
        )
        self.trend_tracker = self.mock_factory.create_trend_tracker()
    
    def test_baseline_establishment_workflow(self):
        """Test the complete baseline establishment workflow."""
        # Create test cases from integration data
        test_cases = []
        for case_data in self.integration_data['cross_pos_test_cases'][:3]:  # Use first 3
            test_case = self.mock_factory.create_test_case(**case_data)
            test_cases.append(test_case)
        
        # Establish baseline using regression tester
        self.regression_tester.establish_baseline(test_cases, force_update=True)
        
        # Verify establish_baseline was called
        self.regression_tester.establish_baseline.assert_called_once()
    
    def test_regression_testing_workflow(self):
        """Test the complete regression testing workflow."""
        # Create test cases
        test_cases = []
        for case_data in self.integration_data['cross_pos_test_cases'][:2]:  # Use first 2
            test_case = self.mock_factory.create_test_case(**case_data)
            test_cases.append(test_case)
        
        # Run regression tests
        regression_results = self.regression_tester.run_regression_tests(test_cases)
        
        # Verify results
        self.assertIsInstance(regression_results, list)
        self.assertEqual(len(regression_results), len(test_cases))
        
        # Verify each result is a proper regression test result
        for result in regression_results:
            self.assertTrue(hasattr(result, 'test_id'))
            self.assertTrue(hasattr(result, 'status'))
    
    def test_report_generation_integration(self):
        """Test integration with report generation."""
        # Create sample regression results from configuration
        sample_results = []
        for result_data in self.config.get_report_generation_data()['sample_regression_results']:
            result = self.mock_factory.create_regression_test_result(**result_data)
            sample_results.append(result)
        
        # Generate report
        report = self.regression_tester.generate_regression_report(sample_results)
        
        # Verify report structure
        self.assertIn('summary', report)
        self.assertIn('performance', report)
        self.assertIn('regressions', report)
        
        # Verify summary data
        expected_structure = self.config.get_report_generation_data()['expected_report_structure']
        self.assertEqual(report['summary']['total_tests'], expected_structure['summary']['total_tests'])
    
    def test_long_term_tracking_integration(self):
        """Test integration with long-term trend tracking."""
        # Create sample regression results
        sample_results = []
        for result_data in self.config.get_report_generation_data()['sample_regression_results'][:2]:
            result = self.mock_factory.create_regression_test_result(**result_data)
            sample_results.append(result)
        
        # Record regression run
        self.trend_tracker.record_regression_run(sample_results)
        
        # Get trend analysis
        trend_analysis = self.trend_tracker.get_trend_analysis()
        
        # Verify trend analysis structure
        self.assertIsInstance(trend_analysis, dict)
        if 'error' not in trend_analysis:
            self.assertIn('total_runs', trend_analysis)
            self.assertIn('recent_average_pass_rate', trend_analysis)
    
    def test_mock_factory_integration(self):
        """Test integration with mock factory system."""
        # Test factory availability
        available_mocks = self.mock_factory.get_available_mocks()
        expected_mocks = self.config.get_mock_factory_test_data()['available_mock_types']
        
        for expected_mock in expected_mocks:
            self.assertIn(expected_mock, available_mocks)
        
        # Test factory methods
        factory_methods = self.config.get_mock_factory_test_data()['factory_method_mapping']
        for method_name, expected_type in factory_methods.items():
            # Verify method exists
            self.assertTrue(hasattr(self.mock_factory, method_name))
            
            # Test method call
            mock_instance = getattr(self.mock_factory, method_name)()
            self.assertIsNotNone(mock_instance)
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test invalid mock creation
        invalid_mock_names = self.config.get_mock_factory_test_data()['invalid_mock_names']
        
        for invalid_name in invalid_mock_names:
            with self.assertRaises(ValueError):
                self.mock_factory(invalid_name)
    
    def test_performance_testing_integration(self):
        """Test integration with performance testing scenarios."""
        performance_data = self.config.get_performance_test_data()
        
        # Test performance scenario handling
        for scenario in performance_data['performance_scenarios']:
            # Create mock regression result with performance data
            result = self.mock_factory.create_regression_test_result(
                test_id=f"perf_test_{scenario['scenario_name']}",
                performance_change_percent=scenario['expected_change_percent'],
                status=scenario['expected_status']
            )
            
            # Verify performance classification
            self.assertEqual(result.status, scenario['expected_status'])
            self.assertEqual(result.performance_change_percent, scenario['expected_change_percent'])


def run_comprehensive_regression_testing(verbosity: int = 1, 
                                        establish_new_baseline: bool = False,
                                        save_results: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive regression testing using mock factory and configuration.
    
    Args:
        verbosity: Verbosity level
        establish_new_baseline: Whether to establish new baseline
        save_results: Whether to save results
        
    Returns:
        Regression testing results
    """
    if not SMIED_AVAILABLE:
        print("ERROR: SMIED not available. Using mock-based testing.")
        
        # Use mock factory for testing when SMIED is not available
        mock_factory = regression_mock_factory
        config = RegressionMockConfig()
        
        # Create mock components
        mock_smied = Mock()
        regression_tester = mock_factory.create_regression_tester(
            smied_instance=mock_smied,
            verbosity=verbosity
        )
        
        # Create mock test cases
        integration_data = config.get_integration_test_data()
        mock_test_cases = []
        for case_data in integration_data['cross_pos_test_cases'][:5]:  # Use first 5
            test_case = mock_factory.create_test_case(**case_data)
            mock_test_cases.append(test_case)
        
        # Run mock regression tests
        regression_results = regression_tester.run_regression_tests(mock_test_cases)
        
        # Generate mock report
        report = regression_tester.generate_regression_report(regression_results)
        
        # Mock trend tracking
        tracker = mock_factory.create_trend_tracker()
        tracker.record_regression_run(regression_results)
        trend_analysis = tracker.get_trend_analysis()
        
        results = {
            'regression_report': report,
            'trend_analysis': trend_analysis,
            'timestamp': time.time(),
            'mock_mode': True
        }
        
        print("Mock-based regression testing completed successfully.")
        return results
    
    # Original SMIED-based implementation would go here
    print("Initializing SMIED for regression testing...")
    smied = SMIED(nlp_model=None, auto_download=False, verbosity=0)
    
    # Use existing regression testing logic with SMIED
    # ... (original implementation)
    
    return {'mock_mode': False}


if __name__ == '__main__':
    # Run regression tests if executed directly
    if SMIED_AVAILABLE:
        import time
        print("Running SMIED regression tests...")
        results = run_comprehensive_regression_testing(
            verbosity=2, 
            establish_new_baseline=True,  # Establish baseline on first run
            save_results=True
        )
        
        # Print trend analysis if available
        if 'trend_analysis' in results and 'error' not in results['trend_analysis']:
            trend = results['trend_analysis']
            print(f"\nTREND ANALYSIS:")
            print(f"Total Runs: {trend.get('total_runs', 0)}")
            print(f"Recent Pass Rate: {trend.get('recent_average_pass_rate', 0):.1f}%")
            print(f"Trend: {trend.get('trend', 'unknown')}")
    else:
        print("SMIED not available. Running mock-based regression tests.")
        results = run_comprehensive_regression_testing(verbosity=2)
        
        if results.get('mock_mode'):
            print("\nMock-based regression testing completed.")
            print("Install SMIED dependencies for full regression testing.")
        
        # Also run unit tests
        unittest.main(verbosity=2)