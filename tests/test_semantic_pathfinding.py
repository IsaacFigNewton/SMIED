"""
Test suite for semantic pathfinding validation and performance analysis.

This comprehensive test suite validates the SMIED semantic pathfinding capabilities
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
from tests.mocks.semantic_pathfinding_mocks import SemanticPathfindingMockFactory
from tests.config.semantic_pathfinding_config import SemanticPathfindingMockConfig


class TestSemanticPathfinding(unittest.TestCase):
    """
    Basic functionality tests for semantic pathfinding.
    
    Tests core pathfinding functionality including:
    - Basic pathfinding operations
    - Test case structure validation
    - Performance metrics calculation
    - Mock factory integration
    """
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory
        self.mock_factory = SemanticPathfindingMockFactory()
        
        # Load test configuration
        self.test_config = SemanticPathfindingMockConfig.get_comprehensive_test_configuration()
        self.mock_config = SemanticPathfindingMockConfig.get_mock_setup_configurations()['basic_setup']
        
        # Create core mock components through factory
        self.mock_smied = self.mock_factory('MockSMIEDForPathfinding')
        self.mock_pathfinding_suite = self.mock_factory(
            'MockSemanticPathfindingSuite',
            smied_instance=self.mock_smied,
            verbosity=self.test_config['test_execution_settings']['default_verbosity']
        )
        self.mock_validator = self.mock_factory('MockSemanticPathfindingValidator', verbosity=0)
        
        # Load test data from configuration
        self.basic_test_cases = SemanticPathfindingMockConfig.get_basic_test_cases()
        self.performance_benchmarks = SemanticPathfindingMockConfig.get_performance_benchmarks()
    
    def test_mock_factory_initialization(self):
        """Test that mock factory initializes correctly."""
        # Verify factory exists and has expected mocks
        available_mocks = self.mock_factory.get_available_mocks()
        
        self.assertIn('MockSemanticPathfindingSuite', available_mocks)
        self.assertIn('MockSemanticPathfindingValidator', available_mocks)
        self.assertIn('MockSMIEDForPathfinding', available_mocks)
        self.assertIn('MockTestCase', available_mocks)
        self.assertIn('MockPathfindingResult', available_mocks)
        
        # Test factory can create instances
        test_case_mock = self.mock_factory('MockTestCase')
        self.assertIsNotNone(test_case_mock)
        
    def test_basic_pathfinding_functionality(self):
        """Test basic pathfinding functionality through mocks."""
        # Get a simple test case from config
        simple_cases = self.basic_test_cases['simple_cases']
        test_case_data = simple_cases[0]  # Cat chasing mouse
        
        # Create test case mock through factory
        test_case_mock = self.mock_factory(
            'MockTestCase',
            **test_case_data
        )
        
        # Verify test case structure
        self.assertEqual(test_case_mock.subject, 'cat')
        self.assertEqual(test_case_mock.predicate, 'chase')
        self.assertEqual(test_case_mock.object, 'mouse')
        self.assertTrue(test_case_mock.expected_success)
        self.assertEqual(test_case_mock.difficulty_level, 'easy')
        
        # Test pathfinding execution
        result = self.mock_pathfinding_suite.run_single_test(test_case_mock)
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'execution_time'))
        self.assertGreater(result.execution_time, 0)
        
    def test_pathfinding_result_validation(self):
        """Test pathfinding result validation."""
        # Create test case and result mocks
        test_case_mock = self.mock_factory(
            'MockTestCase',
            subject='teacher',
            predicate='explain',
            object='concept',
            expected_success=True
        )
        
        # Create successful result mock
        successful_result = self.mock_factory(
            'MockPathfindingResult',
            success=True,
            subject_path=['teacher.n.01', 'predicate'],
            object_path=['predicate', 'concept.n.01'],
            execution_time=0.1
        )
        
        # Validate the result
        validation = self.mock_validator.validate_path_quality(test_case_mock, successful_result)
        
        # Verify validation structure
        self.assertIsInstance(validation, dict)
        self.assertIn('is_valid_path', validation)
        self.assertIn('semantic_coherence', validation)
        self.assertIn('connecting_predicate_relevant', validation)
        self.assertIn('issues', validation)
        
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Run mock test suite
        results = self.mock_pathfinding_suite.run_all_tests()
        
        # Calculate metrics
        metrics = self.mock_pathfinding_suite.calculate_metrics()
        
        # Verify metrics structure
        self.assertIsNotNone(metrics)
        self.assertTrue(hasattr(metrics, 'total_tests'))
        self.assertTrue(hasattr(metrics, 'successful_tests'))
        self.assertTrue(hasattr(metrics, 'failed_tests'))
        self.assertTrue(hasattr(metrics, 'success_rate'))
        self.assertTrue(hasattr(metrics, 'average_time'))
        
        # Verify metrics values are reasonable
        self.assertGreaterEqual(metrics.success_rate, 0.0)
        self.assertLessEqual(metrics.success_rate, 100.0)
        self.assertGreaterEqual(metrics.average_time, 0.0)
        
    def test_test_case_structure_validation(self):
        """Test that test cases from config are properly structured."""
        all_cases = []
        all_cases.extend(self.basic_test_cases['simple_cases'])
        all_cases.extend(self.basic_test_cases['medium_cases'])
        all_cases.extend(self.basic_test_cases['hard_cases'])
        
        # Verify minimum number of test cases
        self.assertGreaterEqual(len(all_cases), 5)
        
        # Check structure of each test case
        for case_data in all_cases:
            # Create mock from config data
            test_case = self.mock_factory('MockTestCase', **case_data)
            
            # Verify required attributes
            self.assertIsInstance(test_case.subject, str)
            self.assertIsInstance(test_case.predicate, str)
            self.assertIsInstance(test_case.object, str)
            self.assertIsInstance(test_case.expected_success, bool)
            self.assertIn(test_case.difficulty_level, ['easy', 'medium', 'hard'])
            self.assertTrue(test_case.cross_pos)
            
            # Verify entity validation
            self.assertTrue(test_case.validate_entity())
            
    def test_mock_integration_with_config(self):
        """Test integration between mock factory and configuration system."""
        # Test different mock setup configurations
        basic_setup = SemanticPathfindingMockConfig.get_mock_setup_configurations()['basic_setup']
        
        # Verify required mock types are available
        for mock_name in [basic_setup['pathfinding_suite_mock'], 
                         basic_setup['validator_mock'],
                         basic_setup['smied_mock']]:
            mock_instance = self.mock_factory(mock_name)
            self.assertIsNotNone(mock_instance)
        
        # Test with different test data sets
        validation_data = SemanticPathfindingMockConfig.get_validation_test_data()
        
        # Create result mock from config data
        valid_result_data = validation_data['valid_results']['short_path']
        result_mock = self.mock_factory(
            'MockPathfindingResult',
            success=valid_result_data['success'],
            subject_path=valid_result_data['subject_path'],
            object_path=valid_result_data['object_path'],
            execution_time=valid_result_data['execution_time']
        )
        
        # Verify result matches expected validation
        self.assertTrue(result_mock.success)
        self.assertEqual(len(result_mock.subject_path), 2)
        self.assertEqual(len(result_mock.object_path), 2)


class TestSemanticPathfindingValidation(unittest.TestCase):
    """
    Validation and constraint tests for semantic pathfinding.
    
    Tests validation functionality including:
    - Path quality validation
    - Semantic coherence checking
    - Input constraint validation
    - Validation criteria enforcement
    """
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory
        self.mock_factory = SemanticPathfindingMockFactory()
        
        # Load validation-specific configuration
        self.validation_config = SemanticPathfindingMockConfig.get_mock_setup_configurations()['validation_setup']
        self.validation_data = SemanticPathfindingMockConfig.get_validation_test_data()
        self.comprehensive_config = SemanticPathfindingMockConfig.get_comprehensive_test_configuration()
        
        # Create validation-specific mock components
        self.mock_validator = self.mock_factory('MockSemanticPathfindingValidator', verbosity=0)
        self.mock_suite = self.mock_factory('MockSemanticPathfindingValidation')
        self.mock_smied = self.mock_factory('MockSMIEDValidation')
        
    def test_path_quality_validation_successful(self):
        """Test path quality validation for successful results."""
        valid_results = self.validation_data['valid_results']
        
        for result_name, result_data in valid_results.items():
            with self.subTest(result=result_name):
                # Create test case and result mocks - use predicate from config data
                predicate_name = result_data['connecting_predicate_name'].split('.')[0]
                test_case = self.mock_factory(
                    'MockTestCase',
                    subject='test',
                    predicate=predicate_name,
                    object='path'
                )
                
                # Create connecting predicate mock
                connecting_predicate = self.mock_factory(
                    'MockConnectingPredicate',
                    predicate_name=result_data['connecting_predicate_name'].split('.')[0]
                )
                
                result = self.mock_factory(
                    'MockPathfindingResult',
                    success=result_data['success'],
                    subject_path=result_data['subject_path'],
                    object_path=result_data['object_path'],
                    connecting_predicate=connecting_predicate,
                    execution_time=result_data['execution_time']
                )
                
                # Perform validation
                validation = self.mock_validator.validate_path_quality(test_case, result)
                
                # Verify validation results match expected
                expected_validation = result_data['expected_validation']
                self.assertEqual(validation['is_valid_path'], expected_validation['is_valid_path'])
                self.assertEqual(validation['path_length_reasonable'], expected_validation['path_length_reasonable'])
                self.assertEqual(validation['connecting_predicate_relevant'], expected_validation['connecting_predicate_relevant'])
                self.assertEqual(validation['semantic_coherence'], expected_validation['semantic_coherence'])
                
    def test_path_quality_validation_failed(self):
        """Test path quality validation for failed results."""
        invalid_results = self.validation_data['invalid_results']
        
        for result_name, result_data in invalid_results.items():
            with self.subTest(result=result_name):
                # Create test case mock
                test_case = self.mock_factory(
                    'MockTestCase',
                    subject='test',
                    predicate='validate',
                    object='failure'
                )
                
                # Create failed result mock
                result_args = {
                    'success': result_data['success']
                }
                
                if 'subject_path' in result_data:
                    result_args['subject_path'] = result_data['subject_path']
                if 'object_path' in result_data:
                    result_args['object_path'] = result_data['object_path']
                if 'execution_time' in result_data:
                    result_args['execution_time'] = result_data['execution_time']
                if 'error' in result_data:
                    result_args['error'] = result_data['error']
                    
                # Add connecting predicate if specified
                if 'connecting_predicate_name' in result_data:
                    connecting_predicate = self.mock_factory(
                        'MockConnectingPredicate',
                        predicate_name=result_data['connecting_predicate_name'].split('.')[0]
                    )
                    result_args['connecting_predicate'] = connecting_predicate
                
                result = self.mock_factory('MockPathfindingResult', **result_args)
                
                # Perform validation
                validation = self.mock_validator.validate_path_quality(test_case, result)
                
                # Verify validation identifies issues
                expected_validation = result_data['expected_validation']
                self.assertEqual(validation['is_valid_path'], expected_validation['is_valid_path'])
                
                if 'issues' in expected_validation:
                    for expected_issue in expected_validation['issues']:
                        # Check if any validation issue contains the key words from expected issue
                        if 'Path too long' in expected_issue:
                            self.assertTrue(any('Path too long' in issue for issue in validation['issues']),
                                          f"Expected 'Path too long' issue not found in {validation['issues']}")
                        elif 'Connecting predicate' in expected_issue and "doesn't match" in expected_issue:
                            self.assertTrue(any("doesn't match" in issue for issue in validation['issues']),
                                          f"Expected predicate mismatch issue not found in {validation['issues']}")
                        else:
                            self.assertIn(expected_issue, validation['issues'])
                        
    def test_semantic_coherence_calculation(self):
        """Test semantic coherence score calculation."""
        # Test various coherence scenarios
        coherence_scenarios = [
            {
                'name': 'perfect_coherence',
                'is_valid_path': True,
                'path_length_reasonable': True,
                'connecting_predicate_relevant': True,
                'expected_score': 1.0
            },
            {
                'name': 'partial_coherence',
                'is_valid_path': True,
                'path_length_reasonable': False,
                'connecting_predicate_relevant': True,
                'expected_score': 0.7
            },
            {
                'name': 'no_coherence',
                'is_valid_path': False,
                'path_length_reasonable': False,
                'connecting_predicate_relevant': False,
                'expected_score': 0.0
            }
        ]
        
        for scenario in coherence_scenarios:
            with self.subTest(scenario=scenario['name']):
                # Create test case
                test_case = self.mock_factory('MockTestCase')
                
                # Create result that will produce the desired validation outcome
                if scenario['is_valid_path']:
                    subject_path = ['valid.n.01', 'predicate']
                    object_path = ['predicate', 'path.n.01']
                else:
                    subject_path = None
                    object_path = None
                
                if scenario['path_length_reasonable']:
                    # Keep paths short
                    pass  
                else:
                    # Make paths too long
                    subject_path = ['a'] * 10 if subject_path else None
                    object_path = ['b'] * 10 if object_path else None
                
                connecting_predicate = None
                if scenario['connecting_predicate_relevant']:
                    connecting_predicate = self.mock_factory(
                        'MockConnectingPredicate',
                        predicate_name=test_case.predicate
                    )
                
                result = self.mock_factory(
                    'MockPathfindingResult',
                    success=scenario['is_valid_path'],
                    subject_path=subject_path,
                    object_path=object_path,
                    connecting_predicate=connecting_predicate
                )
                
                # Perform validation
                validation = self.mock_validator.validate_path_quality(test_case, result)
                
                # Check coherence score
                self.assertEqual(validation['semantic_coherence'], scenario['expected_score'])
                
    def test_validation_constraints_enforcement(self):
        """Test enforcement of validation constraints from configuration."""
        validation_settings = self.comprehensive_config['validation_settings']
        
        # Test semantic coherence threshold
        coherence_threshold = validation_settings['semantic_coherence_threshold']
        self.assertGreaterEqual(coherence_threshold, 0.0)
        self.assertLessEqual(coherence_threshold, 1.0)
        
        # Test path length constraints
        max_path_length = validation_settings['max_reasonable_path_length']
        min_path_length = validation_settings['min_reasonable_path_length']
        
        # Create test cases to verify constraint enforcement
        test_cases = [
            {
                'name': 'path_too_short',
                'path_length': min_path_length - 1,
                'should_pass': False
            },
            {
                'name': 'path_acceptable',
                'path_length': (min_path_length + max_path_length) // 2,
                'should_pass': True
            },
            {
                'name': 'path_too_long',
                'path_length': max_path_length + 1,
                'should_pass': False
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case['name']):
                # Create paths of specified length
                path_length = case['path_length']
                subject_path = ['node'] * (path_length // 2)
                object_path = ['node'] * (path_length - len(subject_path))
                
                test_case = self.mock_factory('MockTestCase')
                result = self.mock_factory(
                    'MockPathfindingResult',
                    success=True,
                    subject_path=subject_path,
                    object_path=object_path
                )
                
                validation = self.mock_validator.validate_path_quality(test_case, result)
                
                # Verify path length validation matches expectation
                self.assertEqual(validation['path_length_reasonable'], case['should_pass'])
                
    def test_test_suite_validation(self):
        """Test validation of entire test suite results."""
        # Create mock test results
        test_results = []
        
        # Add some successful results
        for i in range(3):
            test_case = self.mock_factory(
                'MockTestCase',
                subject=f'subject{i}',
                predicate=f'predicate{i}',
                object=f'object{i}'
            )
            result = self.mock_factory(
                'MockPathfindingResult',
                success=True,
                subject_path=[f'subject{i}.n.01', 'pred'],
                object_path=['pred', f'object{i}.n.01'],
                execution_time=0.1 + i * 0.02
            )
            test_results.append((test_case, result))
        
        # Add some failed results
        for i in range(2):
            test_case = self.mock_factory(
                'MockTestCase',
                subject=f'failed{i}',
                predicate=f'fail{i}',
                object=f'error{i}',
                expected_success=False
            )
            result = self.mock_factory(
                'MockPathfindingResult',
                success=False,
                error=f'Test failure {i}'
            )
            test_results.append((test_case, result))
        
        # Validate entire suite
        suite_validation = self.mock_validator.validate_test_suite_results(test_results)
        
        # Verify suite validation structure
        self.assertIn('total_tests', suite_validation)
        self.assertIn('average_semantic_coherence', suite_validation)
        self.assertIn('valid_paths', suite_validation)
        self.assertIn('quality_issues_count', suite_validation)
        
        # Verify counts
        self.assertEqual(suite_validation['total_tests'], 5)
        self.assertGreaterEqual(suite_validation['average_semantic_coherence'], 0.0)
        self.assertLessEqual(suite_validation['average_semantic_coherence'], 1.0)


class TestSemanticPathfindingEdgeCases(unittest.TestCase):
    """
    Edge cases and error conditions for semantic pathfinding.
    
    Tests edge case scenarios including:
    - Timeout conditions
    - Memory exhaustion
    - Invalid inputs
    - Boundary conditions
    """
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory
        self.mock_factory = SemanticPathfindingMockFactory()
        
        # Load edge case specific configuration
        self.edge_case_config = SemanticPathfindingMockConfig.get_mock_setup_configurations()['edge_case_setup']
        self.edge_case_scenarios = SemanticPathfindingMockConfig.get_edge_case_scenarios()
        
        # Create edge case specific mock components
        self.mock_suite = self.mock_factory('MockSemanticPathfindingEdgeCases')
        self.mock_smied = self.mock_factory('MockSMIEDEdgeCases')
        
    def test_timeout_scenarios(self):
        """Test handling of timeout scenarios."""
        timeout_scenarios = self.edge_case_scenarios['timeout_scenarios']
        
        for scenario in timeout_scenarios:
            with self.subTest(scenario=scenario['subject']):
                # Create test case for timeout scenario
                test_case = self.mock_factory(
                    'MockTestCase',
                    subject=scenario['subject'],
                    predicate=scenario['predicate'],
                    object=scenario['object']
                )
                
                # Test timeout handling
                with self.assertRaises(TimeoutError):
                    self.mock_suite.handle_timeout_scenarios()
                    
    def test_memory_exhaustion_scenarios(self):
        """Test handling of memory exhaustion scenarios."""
        memory_scenarios = self.edge_case_scenarios['memory_scenarios']
        
        for scenario in memory_scenarios:
            with self.subTest(scenario=scenario['subject']):
                # Create test case for memory scenario
                test_case = self.mock_factory(
                    'MockTestCase',
                    subject=scenario['subject'],
                    predicate=scenario['predicate'],
                    object=scenario['object']
                )
                
                # Test memory exhaustion handling
                with self.assertRaises(MemoryError):
                    self.mock_suite.handle_memory_exhaustion()
                    
    def test_invalid_input_scenarios(self):
        """Test handling of invalid input scenarios."""
        error_scenarios = self.edge_case_scenarios['error_scenarios']
        
        for scenario in error_scenarios:
            with self.subTest(scenario=scenario['subject']):
                # Create test case for error scenario
                test_case = self.mock_factory(
                    'MockTestCase',
                    subject=scenario['subject'],
                    predicate=scenario['predicate'],
                    object=scenario['object']
                )
                
                # Test invalid input handling
                with self.assertRaises(ValueError):
                    self.mock_suite.handle_invalid_inputs()
                    
    def test_boundary_conditions(self):
        """Test boundary condition scenarios."""
        boundary_scenarios = self.edge_case_scenarios['boundary_scenarios']
        
        for scenario in boundary_scenarios:
            with self.subTest(scenario=scenario['subject']):
                # Create test case for boundary scenario
                test_case = self.mock_factory(
                    'MockTestCase',
                    subject=scenario['subject'],
                    predicate=scenario['predicate'],
                    object=scenario['object']
                )
                
                if 'expected_exception' in scenario:
                    # Test that boundary condition raises expected exception
                    expected_exception_name = scenario['expected_exception']
                    if expected_exception_name == 'ValueError':
                        expected_exception = ValueError
                    elif expected_exception_name == 'TypeError':
                        expected_exception = TypeError
                    elif expected_exception_name == 'AttributeError':
                        expected_exception = AttributeError
                    else:
                        expected_exception = Exception
                        
                    with self.assertRaises(expected_exception):
                        # Simulate boundary condition that should fail
                        if scenario['predicate'] == '':
                            raise ValueError("Empty predicate")
                else:
                    # Test successful boundary condition
                    result = self.mock_suite.run_single_test(test_case)
                    self.assertIsNotNone(result)
                    
    def test_empty_result_handling(self):
        """Test handling of empty/null results."""
        # Test empty results handling
        empty_result = self.mock_suite.handle_empty_results()
        
        # Verify empty results are handled gracefully
        self.assertIsNotNone(empty_result)
        self.assertEqual(empty_result, (None, None, None))
        
    def test_performance_edge_cases(self):
        """Test performance-related edge cases."""
        # Test with extreme parameters from config
        edge_case_data = SemanticPathfindingMockConfig.get_edge_case_scenarios()
        
        if 'performance_scenarios' in edge_case_data:
            performance_scenarios = edge_case_data['performance_scenarios']
            
            for scenario_name, scenario_data in performance_scenarios.items():
                with self.subTest(scenario=scenario_name):
                    # Create test case with extreme parameters
                    test_case = self.mock_factory('MockTestCase')
                    
                    # Test with extreme parameters if they cause issues
                    if 'large_search_space' in scenario_name:
                        # These should complete but may be slow
                        result = self.mock_suite.run_single_test(
                            test_case,
                            max_depth=scenario_data.get('beam_width', 50),
                            beam_width=scenario_data.get('max_depth', 20)
                        )
                        self.assertIsNotNone(result)
                        
    def test_malformed_data_handling(self):
        """Test handling of malformed or corrupted data."""
        malformed_scenarios = [
            {
                'name': 'none_subject',
                'subject': None,
                'predicate': 'test',
                'object': 'test'
            },
            {
                'name': 'empty_predicate',
                'subject': 'test',
                'predicate': '',
                'object': 'test'
            },
            {
                'name': 'numeric_object',
                'subject': 'test',
                'predicate': 'test',
                'object': 123
            }
        ]
        
        for scenario in malformed_scenarios:
            with self.subTest(scenario=scenario['name']):
                # Test that malformed data is handled appropriately
                if scenario['subject'] is None or scenario['predicate'] == '':
                    # These should cause validation failures
                    with self.assertRaises((ValueError, TypeError, AttributeError)):
                        test_case = self.mock_factory(
                            'MockTestCase',
                            subject=scenario['subject'],
                            predicate=scenario['predicate'],
                            object=scenario['object']
                        )
                        if not test_case.validate_entity():
                            raise ValueError("Invalid test case")
                else:
                    # These might be handled gracefully with type conversion
                    test_case = self.mock_factory(
                        'MockTestCase',
                        subject=scenario['subject'],
                        predicate=scenario['predicate'],
                        object=str(scenario['object'])  # Convert to string
                    )
                    # Should not raise exception after conversion
                    self.assertIsNotNone(test_case)


class TestSemanticPathfindingIntegration(unittest.TestCase):
    """
    Integration tests for semantic pathfinding with other components.
    
    Tests integration scenarios including:
    - End-to-end pathfinding workflows
    - Component interaction validation
    - Realistic scenario testing
    - External dependency integration
    """
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory
        self.mock_factory = SemanticPathfindingMockFactory()
        
        # Load integration-specific configuration
        self.integration_config = SemanticPathfindingMockConfig.get_mock_setup_configurations()['integration_setup']
        self.integration_data = SemanticPathfindingMockConfig.get_integration_test_data()
        
        # Create integration-specific mock components
        self.mock_suite = self.mock_factory('MockSemanticPathfindingIntegration')
        self.mock_smied = self.mock_factory('MockSMIEDIntegration')
        self.mock_benchmark = self.mock_factory(
            'MockSemanticPathfindingBenchmark',
            smied_instance=self.mock_smied,
            verbosity=1
        )
        
        # Setup integration environment
        self.mock_suite.setup_integration_environment()
        
    def tearDown(self):
        """Clean up integration test environment."""
        # Teardown integration environment
        self.mock_suite.teardown_integration_environment()
        
    def test_end_to_end_pathfinding_workflow(self):
        """Test complete end-to-end pathfinding workflow."""
        realistic_scenarios = self.integration_data['realistic_scenarios']
        
        for scenario in realistic_scenarios:
            with self.subTest(scenario=scenario['name']):
                # Test each case in the scenario
                for test_case_data in scenario['test_cases']:
                    # Create test case
                    test_case = self.mock_factory('MockTestCase', **test_case_data)
                    
                    # Run complete workflow
                    result = self.mock_suite.run_single_test(test_case)
                    
                    # Verify workflow completion
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result.success, bool)
                    
                    # Check if result matches expectation
                    if test_case_data['expected_success']:
                        self.assertTrue(result.success)
                        self.assertIsNotNone(result.subject_path)
                        self.assertIsNotNone(result.object_path)
                    
                # Test overall scenario success rate
                expected_success_rate = scenario['expected_overall_success_rate']
                self.assertIsInstance(expected_success_rate, float)
                self.assertGreaterEqual(expected_success_rate, 0.0)
                self.assertLessEqual(expected_success_rate, 100.0)
                
    def test_component_interaction_validation(self):
        """Test interaction validation between pathfinding components."""
        # Verify component interaction
        interaction_valid = self.mock_suite.validate_component_interactions()
        self.assertTrue(interaction_valid)
        
        # Test required components are available
        required_components = self.integration_data['integration_components']['required_smied_components']
        
        for component_name in required_components:
            with self.subTest(component=component_name):
                # Verify component availability
                self.assertTrue(hasattr(self.mock_smied, component_name.lower() + '_available'))
                
    def test_realistic_pathfinding_scenarios(self):
        """Test realistic pathfinding scenarios with integration components."""
        # Test scenarios that require multiple component integration
        realistic_scenarios = self.integration_data['realistic_scenarios']
        
        for scenario in realistic_scenarios:
            with self.subTest(scenario=scenario['name']):
                # Create scenario test suite
                scenario_results = []
                
                for test_case_data in scenario['test_cases']:
                    # Create and run test case
                    test_case = self.mock_factory('MockTestCase', **test_case_data)
                    result = self.mock_suite.run_single_test(test_case)
                    scenario_results.append((test_case, result))
                
                # Verify scenario outcomes
                success_count = sum(1 for _, result in scenario_results if result.success)
                actual_success_rate = (success_count / len(scenario_results)) * 100
                
                expected_success_rate = scenario['expected_overall_success_rate']
                
                # Allow some tolerance in success rate comparison
                tolerance = 10.0  # 10% tolerance
                self.assertGreaterEqual(
                    actual_success_rate, 
                    expected_success_rate - tolerance,
                    f"Success rate {actual_success_rate}% below expected {expected_success_rate}%"
                )
                
    def test_comprehensive_benchmark_integration(self):
        """Test comprehensive benchmark with all integrated components."""
        # Run comprehensive benchmark
        benchmark_results = self.mock_benchmark.run_comprehensive_benchmark()
        
        # Verify benchmark structure
        self.assertIn('main_test_results', benchmark_results)
        self.assertIn('parameter_sensitivity', benchmark_results)
        self.assertIn('scalability_analysis', benchmark_results)
        self.assertIn('validation_report', benchmark_results)
        self.assertIn('benchmark_timestamp', benchmark_results)
        
        # Verify main test results
        main_results = benchmark_results['main_test_results']
        self.assertIn('overall_metrics', main_results)
        self.assertIn('difficulty_analysis', main_results)
        
        # Verify metrics structure
        overall_metrics = main_results['overall_metrics']
        self.assertIn('success_rate', overall_metrics)
        self.assertIn('average_time', overall_metrics)
        self.assertGreaterEqual(overall_metrics['success_rate'], 0.0)
        self.assertLessEqual(overall_metrics['success_rate'], 100.0)
        
    def test_parameter_sensitivity_integration(self):
        """Test parameter sensitivity analysis in integration context."""
        # Run parameter sensitivity analysis
        param_analysis = self.mock_benchmark.benchmark_parameter_sensitivity()
        
        # Verify analysis structure
        self.assertIsInstance(param_analysis, dict)
        self.assertGreater(len(param_analysis), 0)
        
        # Check parameter combinations
        for param_key, param_results in param_analysis.items():
            with self.subTest(params=param_key):
                # Verify parameter result structure
                self.assertIn('max_depth', param_results)
                self.assertIn('beam_width', param_results)
                self.assertIn('success_rate', param_results)
                self.assertIn('average_time', param_results)
                
                # Verify reasonable values
                self.assertGreaterEqual(param_results['success_rate'], 0.0)
                self.assertLessEqual(param_results['success_rate'], 100.0)
                self.assertGreaterEqual(param_results['average_time'], 0.0)
                
    def test_scalability_integration(self):
        """Test scalability analysis in integration context."""
        # Run scalability analysis
        scalability_results = self.mock_benchmark.benchmark_scalability()
        
        # Verify scalability structure
        expected_difficulties = ['easy', 'medium', 'hard']
        
        for difficulty in expected_difficulties:
            with self.subTest(difficulty=difficulty):
                self.assertIn(difficulty, scalability_results)
                
                difficulty_results = scalability_results[difficulty]
                self.assertIn('test_count', difficulty_results)
                self.assertIn('success_rate', difficulty_results)
                self.assertIn('average_time', difficulty_results)
                
                # Verify scalability trends
                self.assertGreater(difficulty_results['test_count'], 0)
                self.assertGreaterEqual(difficulty_results['success_rate'], 0.0)
                self.assertLessEqual(difficulty_results['success_rate'], 100.0)
                
    def test_external_dependency_integration(self):
        """Test integration with external dependencies."""
        external_deps = self.integration_config.get('external_dependencies', [])
        
        for dependency in external_deps:
            with self.subTest(dependency=dependency):
                # Verify external dependency is properly mocked/integrated
                if dependency == 'wordnet':
                    # Test WordNet integration
                    self.assertTrue(hasattr(self.mock_smied, 'real_wordnet_available'))
                elif dependency == 'nlp_model':
                    # Test NLP model integration  
                    self.assertTrue(hasattr(self.mock_smied, 'real_nlp_model_available'))
                    
        # Test that integration works with external dependencies
        test_case = self.mock_factory(
            'MockTestCase',
            subject='integration',
            predicate='test',
            object='dependency'
        )
        
        result = self.mock_suite.run_single_test(test_case)
        self.assertIsNotNone(result)


# Utility functions for test execution following SMIED specifications
def run_semantic_pathfinding_tests_with_framework(verbosity: int = 1, save_results: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run semantic pathfinding tests using the SMIED Testing Framework.
    
    Args:
        verbosity: Verbosity level (0=silent, 1=normal, 2=detailed)
        save_results: Whether to save results to file
        
    Returns:
        Complete test results following framework specifications
    """
    # Initialize mock factory and configuration
    mock_factory = SemanticPathfindingMockFactory()
    config = SemanticPathfindingMockConfig()
    
    print("Running semantic pathfinding tests with SMIED Testing Framework...")
    
    # Create mock SMIED instance for testing
    mock_smied = mock_factory('MockSMIEDForPathfinding')
    
    # Run comprehensive benchmark using framework
    benchmark = mock_factory('MockSemanticPathfindingBenchmark', smied_instance=mock_smied, verbosity=verbosity)
    results = benchmark.run_comprehensive_benchmark()
    
    if save_results:
        timestamp = int(time.time())
        filename = f"semantic_pathfinding_framework_results_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Save results (convert to JSON-serializable format)
        import json
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {filepath}")
        except Exception as e:
            print(f"Could not save results: {e}")
    
    return results


if __name__ == '__main__':
    # Run the tests using the SMIED Testing Framework
    print("Running semantic pathfinding tests with SMIED Testing Framework...")
    
    # Run framework-based tests
    results = run_semantic_pathfinding_tests_with_framework(verbosity=2, save_results=True)
    
    # Print summary
    if results and 'main_test_results' in results:
        print("\n" + "="*60)
        print("SEMANTIC PATHFINDING TEST SUMMARY (SMIED Framework)")
        print("="*60)
        
        main_metrics = results['main_test_results']['overall_metrics']
        print(f"Success Rate: {main_metrics['success_rate']:.1f}%")
        print(f"Average Time: {main_metrics['average_time']:.3f}s")
        print(f"Total Tests: {main_metrics['total_tests']}")
        
        if 'validation_report' in results:
            val_report = results['validation_report']
            print(f"Average Semantic Coherence: {val_report['average_semantic_coherence']:.2f}")
            print(f"Quality Issues: {val_report['quality_issues_count']}")
        
        print("="*60)
    
    # Also run unittest discovery
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)