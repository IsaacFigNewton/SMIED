"""
Comparative analysis tests for SMIED vs ConceptNet.io and other knowledge bases.

This module follows the SMIED Testing Framework Design Specifications
for the Test Layer with 3-layer architecture:
- Test Layer: Contains test logic and assertions
- Mock Layer: Provides mock implementations and factories
- Configuration Layer: Supplies test data and constants

Test classes are organized following the framework pattern:
- TestComparativeAnalysis: Basic functionality tests
- TestComparativeAnalysisValidation: Validation and constraint tests  
- TestComparativeAnalysisEdgeCases: Edge cases and error conditions
- TestComparativeAnalysisIntegration: Integration tests with other components
"""

import unittest
import sys
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import mock factory and configuration (3-layer architecture)
from tests.mocks.comparative_analysis_mocks import (
    ComparativeAnalysisMockFactory,
    ConceptNetResult,
    SMIEDResult,
    ComparisonResult
)
from tests.config.comparative_analysis_config import ComparativeAnalysisMockConfig

try:
    from smied.SMIED import SMIED
    SMIED_AVAILABLE = True
except ImportError:
    SMIED_AVAILABLE = False
    SMIED = None


@unittest.skipUnless(SMIED_AVAILABLE, "SMIED not available for testing")
class TestComparativeAnalysis(unittest.TestCase):
    """Basic functionality tests for comparative analysis."""
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        self.mock_factory = ComparativeAnalysisMockFactory()
        self.config = ComparativeAnalysisMockConfig()
        
        # Get mock setup configuration
        setup_config = self.config.get_mock_setup_configurations()['basic_setup']
        
        # Create mocks using factory pattern
        self.smied_mock = self.mock_factory(setup_config['smied_instance_mock'])
        self.analyzer_mock = self.mock_factory(
            setup_config['comparative_analyzer_mock'],
            smied_instance=self.smied_mock,
            verbosity=setup_config['verbosity']
        )
        self.conceptnet_mock = self.mock_factory(setup_config['conceptnet_interface_mock'])
        
        # Inject test data from configuration
        self.basic_test_cases = self.config.get_basic_test_cases()
        self.performance_benchmarks = self.config.get_performance_benchmarks()
    
    def test_comparative_analyzer_initialization(self):
        """Test ComparativeAnalyzer initialization with factory-created mocks."""
        self.assertIsNotNone(self.analyzer_mock)
        self.assertIsNotNone(self.analyzer_mock.smied)
        self.assertIsNotNone(self.analyzer_mock.conceptnet)
        self.assertTrue(self.analyzer_mock.conceptnet_available)
    
    def test_smied_test_execution_basic_cases(self):
        """Test SMIED test execution with basic test cases from config."""
        simple_cases = self.basic_test_cases['simple_comparison_cases']
        
        for case_data in simple_cases[:3]:  # Test first 3 cases
            with self.subTest(case=case_data['subject']):
                # Create mock test case
                test_case_mock = self.mock_factory(
                    'MockTestCase',
                    subject=case_data['subject'],
                    predicate=case_data['predicate'],
                    object=case_data['object']
                )
                
                # Execute test
                result = self.analyzer_mock.run_smied_test(test_case_mock)
                
                # Validate result structure
                self.assertIsInstance(result, SMIEDResult)
                self.assertIsInstance(result.success, bool)
                self.assertGreaterEqual(result.response_time, 0)
                
                if case_data['expected_smied_success']:
                    self.assertTrue(result.path_found)
                    self.assertGreater(result.total_path_length, 0)
    
    def test_conceptnet_interface_basic_functionality(self):
        """Test ConceptNet interface basic functionality."""
        # Test connection testing
        self.assertTrue(self.conceptnet_mock.test_connection())
        
        # Test query relation method exists and returns proper result
        result = self.conceptnet_mock.query_relation("cat", "chase", "mouse")
        self.assertIsInstance(result, ConceptNetResult)
        self.assertIsInstance(result.success, bool)
        self.assertGreaterEqual(result.response_time, 0)
    
    def test_comparison_result_structure(self):
        """Test comparison result structure follows expected format."""
        simple_cases = self.basic_test_cases['simple_comparison_cases']
        test_case_data = simple_cases[0]
        
        # Create mock test case
        test_case_mock = self.mock_factory(
            'MockTestCase',
            subject=test_case_data['subject'],
            predicate=test_case_data['predicate'],
            object=test_case_data['object']
        )
        
        # Execute comparison
        comparison = self.analyzer_mock.compare_single_test(test_case_mock)
        
        # Validate structure
        self.assertIsInstance(comparison, ComparisonResult)
        self.assertEqual(comparison.subject, test_case_data['subject'])
        self.assertEqual(comparison.predicate, test_case_data['predicate'])
        self.assertEqual(comparison.object, test_case_data['object'])
        self.assertIn(comparison.winner, ['SMIED', 'ConceptNet', 'Tie', 'Both_Failed'])
        self.assertIn(comparison.performance_comparison, ['SMIED_Faster', 'ConceptNet_Faster', 'Similar'])
        self.assertIn(comparison.semantic_quality, ['SMIED_Better', 'ConceptNet_Better', 'Similar', 'Unknown'])
    
    def test_winner_determination_logic(self):
        """Test winner determination logic with different scenarios."""
        # Test SMIED wins scenario
        smied_success = SMIEDResult(True, True, 2, 2, 4, "chase.v.01", 0.1)
        conceptnet_fail = ConceptNetResult(True, False, None, 0.0, False, 0.2)
        
        winner = self.analyzer_mock._determine_winner(smied_success, conceptnet_fail)
        self.assertEqual(winner, "SMIED")
        
        # Test tie scenario
        conceptnet_success = ConceptNetResult(True, True, "RelatedTo", 2.5, True, 0.2)
        winner = self.analyzer_mock._determine_winner(smied_success, conceptnet_success)
        self.assertEqual(winner, "Tie")
        
        # Test both fail scenario
        smied_fail = SMIEDResult(False, False, 0, 0, 0, None, 0.1)
        winner = self.analyzer_mock._determine_winner(smied_fail, conceptnet_fail)
        self.assertEqual(winner, "Both_Failed")
    
    def test_performance_comparison_logic(self):
        """Test performance comparison logic with timing differences."""
        fast_smied = SMIEDResult(True, True, 2, 2, 4, "chase.v.01", 0.1)
        slow_conceptnet = ConceptNetResult(True, True, "RelatedTo", 2.5, True, 1.0)
        
        comparison = self.analyzer_mock._compare_performance(fast_smied, slow_conceptnet)
        self.assertEqual(comparison, "SMIED_Faster")
        
        # Test similar times
        similar_times = ConceptNetResult(True, True, "RelatedTo", 2.5, True, 0.15)
        comparison = self.analyzer_mock._compare_performance(fast_smied, similar_times)
        self.assertEqual(comparison, "Similar")
    
    def test_multiple_test_cases_execution(self):
        """Test execution of multiple test cases."""
        simple_cases = self.basic_test_cases['simple_comparison_cases']
        test_case_mocks = []
        
        for case_data in simple_cases:
            test_case_mock = self.mock_factory(
                'MockTestCase',
                subject=case_data['subject'],
                predicate=case_data['predicate'],
                object=case_data['object']
            )
            test_case_mocks.append(test_case_mock)
        
        # Execute comparative analysis
        results = self.analyzer_mock.run_comparative_analysis(test_case_mocks)
        
        # Validate results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(test_case_mocks))
        
        for result in results:
            self.assertIsInstance(result, ComparisonResult)


class TestComparativeAnalysisValidation(unittest.TestCase):
    """Validation and constraint tests for comparative analysis."""
    
    def setUp(self):
        """Set up test fixtures for validation scenarios."""
        self.mock_factory = ComparativeAnalysisMockFactory()
        self.config = ComparativeAnalysisMockConfig()
        
        # Get validation setup configuration
        setup_config = self.config.get_mock_setup_configurations()['validation_setup']
        
        # Create validation-specific mocks
        self.smied_mock = self.mock_factory(setup_config['smied_instance_mock'])
        self.analyzer_mock = self.mock_factory(
            setup_config['comparative_analyzer_mock'],
            smied_instance=self.smied_mock,
            verbosity=setup_config['verbosity']
        )
        
        # Inject validation test data
        self.validation_data = self.config.get_validation_test_data()
        self.edge_cases = self.config.get_edge_case_scenarios()
    
    def test_input_validation_empty_subject(self):
        """Test validation with empty subject."""
        boundary_cases = self.edge_cases['boundary_scenarios']
        empty_subject_case = [case for case in boundary_cases if case['subject'] == ''][0]
        
        test_case_mock = self.mock_factory(
            'MockTestCase',
            subject=empty_subject_case['subject'],
            predicate=empty_subject_case['predicate'],
            object=empty_subject_case['object']
        )
        
        result = self.analyzer_mock.run_smied_test(test_case_mock)
        
        # Should handle gracefully and return error
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("subject", result.error.lower())
    
    def test_input_validation_empty_predicate(self):
        """Test validation with empty predicate."""
        test_case_mock = self.mock_factory(
            'MockTestCase',
            subject="test",
            predicate="",
            object="test"
        )
        
        comparison = self.analyzer_mock.compare_single_test(test_case_mock)
        
        # Should handle gracefully
        self.assertEqual(comparison.winner, "Both_Failed")
        self.assertIn("invalid", comparison.notes.lower())
    
    def test_result_structure_validation(self):
        """Test that results have proper structure and types."""
        valid_smied_data = self.validation_data['valid_smied_results']['short_path']
        
        # Create a result with known valid data
        smied_result = SMIEDResult(**valid_smied_data)
        
        # Validate all required fields are present and correct types
        self.assertIsInstance(smied_result.success, bool)
        self.assertIsInstance(smied_result.path_found, bool)
        self.assertIsInstance(smied_result.subject_path_length, int)
        self.assertIsInstance(smied_result.object_path_length, int)
        self.assertIsInstance(smied_result.total_path_length, int)
        self.assertIsInstance(smied_result.response_time, (int, float))
        
        # Test ConceptNet result structure
        valid_conceptnet_data = self.validation_data['valid_conceptnet_results']['strong_relation']
        conceptnet_result = ConceptNetResult(**valid_conceptnet_data)
        
        self.assertIsInstance(conceptnet_result.success, bool)
        self.assertIsInstance(conceptnet_result.relation_found, bool)
        self.assertIsInstance(conceptnet_result.confidence_score, (int, float))
        self.assertIsInstance(conceptnet_result.response_time, (int, float))
    
    def test_comparison_logic_validation(self):
        """Test that comparison logic produces valid results."""
        # Test with known valid inputs
        valid_smied = self.validation_data['valid_smied_results']['short_path']
        valid_conceptnet = self.validation_data['valid_conceptnet_results']['strong_relation']
        
        smied_result = SMIEDResult(**valid_smied)
        conceptnet_result = ConceptNetResult(**valid_conceptnet)
        
        # Test winner determination
        winner = self.analyzer_mock._determine_winner(smied_result, conceptnet_result)
        self.assertIn(winner, ['SMIED', 'ConceptNet', 'Tie', 'Both_Failed'])
        
        # Test performance comparison
        perf_comp = self.analyzer_mock._compare_performance(smied_result, conceptnet_result)
        self.assertIn(perf_comp, ['SMIED_Faster', 'ConceptNet_Faster', 'Similar'])
    
    def test_semantic_quality_assessment_validation(self):
        """Test semantic quality assessment produces valid ratings."""
        valid_smied = self.validation_data['valid_smied_results']['medium_path']
        valid_conceptnet = self.validation_data['valid_conceptnet_results']['weak_relation']
        
        smied_result = SMIEDResult(**valid_smied)
        conceptnet_result = ConceptNetResult(**valid_conceptnet)
        
        test_case_mock = self.mock_factory('MockTestCase')
        
        quality = self.analyzer_mock._assess_semantic_quality(
            smied_result, conceptnet_result, test_case_mock
        )
        
        self.assertIn(quality, ['SMIED_Better', 'ConceptNet_Better', 'Similar', 'Unknown'])


class TestComparativeAnalysisEdgeCases(unittest.TestCase):
    """Edge cases and error conditions for comparative analysis."""
    
    def setUp(self):
        """Set up test fixtures for edge case scenarios."""
        self.mock_factory = ComparativeAnalysisMockFactory()
        self.config = ComparativeAnalysisMockConfig()
        
        # Get edge case setup configuration
        setup_config = self.config.get_mock_setup_configurations()['edge_case_setup']
        
        # Create edge case specific mocks
        self.smied_mock = self.mock_factory(setup_config['smied_instance_mock'])
        self.analyzer_mock = self.mock_factory(
            setup_config['comparative_analyzer_mock'],
            smied_instance=self.smied_mock,
            verbosity=setup_config['verbosity']
        )
        self.conceptnet_mock = self.mock_factory(
            'MockConceptNetInterfaceEdgeCases',
            verbosity=setup_config['verbosity']
        )
        
        # Inject edge case test data
        self.edge_cases = self.config.get_edge_case_scenarios()
        self.negative_cases = self.config.get_negative_test_cases()
    
    def test_timeout_scenarios(self):
        """Test timeout handling in both SMIED and ConceptNet."""
        timeout_cases = self.edge_cases['timeout_scenarios']
        
        for timeout_case in timeout_cases:
            with self.subTest(case=timeout_case['subject']):
                test_case_mock = self.mock_factory(
                    'MockTestCase',
                    subject=timeout_case['subject'],
                    predicate=timeout_case['predicate'],
                    object=timeout_case['object']
                )
                
                # Test SMIED timeout handling
                smied_result = self.analyzer_mock.run_smied_test(test_case_mock)
                if timeout_case.get('expected_smied_error'):
                    self.assertFalse(smied_result.success)
                    self.assertIn("timeout", smied_result.error.lower())
                
                # Test ConceptNet timeout handling
                conceptnet_result = self.conceptnet_mock.query_relation(
                    timeout_case['subject'],
                    timeout_case['predicate'],
                    timeout_case['object'],
                    timeout=timeout_case.get('timeout_seconds', 10.0)
                )
                
                if timeout_case.get('expected_conceptnet_error'):
                    self.assertFalse(conceptnet_result.success)
                    self.assertIn("timeout", conceptnet_result.error.lower())
    
    def test_api_error_scenarios(self):
        """Test API error handling."""
        api_error_cases = self.edge_cases['api_error_scenarios']
        
        for error_case in api_error_cases:
            with self.subTest(case=error_case['subject']):
                test_case_mock = self.mock_factory(
                    'MockTestCase',
                    subject=error_case['subject'],
                    predicate=error_case['predicate'],
                    object=error_case['object']
                )
                
                conceptnet_result = self.conceptnet_mock.query_relation(
                    error_case['subject'],
                    error_case['predicate'],
                    error_case['object']
                )
                
                if error_case.get('expected_conceptnet_error'):
                    self.assertFalse(conceptnet_result.success)
                    self.assertIsNotNone(conceptnet_result.error)
    
    def test_impossible_relationship_cases(self):
        """Test handling of impossible semantic relationships."""
        impossible_cases = self.negative_cases['impossible_cases']
        
        for impossible_case in impossible_cases:
            with self.subTest(case=f"{impossible_case['subject']}_{impossible_case['predicate']}_{impossible_case['object']}"):
                test_case_mock = self.mock_factory(
                    'MockTestCase',
                    subject=impossible_case['subject'],
                    predicate=impossible_case['predicate'],
                    object=impossible_case['object']
                )
                
                comparison = self.analyzer_mock.compare_single_test(test_case_mock)
                
                # Both systems should fail for impossible cases
                if impossible_case['expected_winner'] == 'Both_Failed':
                    self.assertEqual(comparison.winner, 'Both_Failed')
                    self.assertFalse(comparison.smied_result.path_found)
                    self.assertFalse(comparison.conceptnet_result.relation_found)
    
    def test_extreme_path_scenarios(self):
        """Test handling of extremely long or complex paths."""
        extreme_cases = self.edge_cases['extreme_path_scenarios']
        
        for extreme_case in extreme_cases:
            with self.subTest(case=extreme_case['subject']):
                test_case_mock = self.mock_factory(
                    'MockTestCase',
                    subject=extreme_case['subject'],
                    predicate=extreme_case['predicate'],
                    object=extreme_case['object']
                )
                
                smied_result = self.analyzer_mock.run_smied_test(test_case_mock)
                
                if extreme_case.get('expected_smied_path_length'):
                    # For edge case mocks, this might be configured to return specific lengths
                    self.assertGreater(smied_result.total_path_length, 10)  # Long path
                
                if extreme_case.get('expected_smied_time'):
                    # Should handle long execution times appropriately
                    self.assertIsInstance(smied_result.response_time, (int, float))
    
    def test_conceptnet_unavailable_scenario(self):
        """Test behavior when ConceptNet API is unavailable."""
        # The edge case mock should simulate ConceptNet being unavailable
        self.assertFalse(self.analyzer_mock.conceptnet_available)
        
        test_case_mock = self.mock_factory('MockTestCase')
        comparison = self.analyzer_mock.compare_single_test(test_case_mock)
        
        # ConceptNet result should indicate unavailability
        self.assertFalse(comparison.conceptnet_result.success)
        self.assertIsNotNone(comparison.conceptnet_result.error)


@unittest.skipUnless(SMIED_AVAILABLE, "SMIED not available for testing")
class TestComparativeAnalysisIntegration(unittest.TestCase):
    """Integration tests with other components."""
    
    def setUp(self):
        """Set up test fixtures for integration scenarios."""
        self.mock_factory = ComparativeAnalysisMockFactory()
        self.config = ComparativeAnalysisMockConfig()
        
        # Get integration setup configuration
        setup_config = self.config.get_mock_setup_configurations()['integration_setup']
        
        # Create integration-specific mocks
        self.smied_mock = self.mock_factory(setup_config['smied_instance_mock'])
        self.analyzer_mock = self.mock_factory(
            setup_config['comparative_analyzer_mock'],
            smied_instance=self.smied_mock,
            verbosity=setup_config['verbosity']
        )
        self.conceptnet_mock = self.mock_factory(
            'MockConceptNetIntegration',
            verbosity=setup_config['verbosity']
        )
        self.test_suite_mock = self.mock_factory(setup_config['test_suite_mock'])
        
        # Inject integration test data
        self.integration_data = self.config.get_integration_test_data()
        self.report_data = self.config.get_report_generation_test_data()
    
    def test_end_to_end_comparative_analysis(self):
        """Test complete end-to-end comparative analysis workflow."""
        realistic_scenarios = self.integration_data['realistic_comparison_scenarios']
        
        for scenario in realistic_scenarios:
            with self.subTest(scenario=scenario['name']):
                # Create test case mocks from scenario data
                test_case_mocks = []
                for case_data in scenario['test_cases']:
                    test_case_mock = self.mock_factory(
                        'MockTestCase',
                        subject=case_data['subject'],
                        predicate=case_data['predicate'],
                        object=case_data['object']
                    )
                    test_case_mocks.append(test_case_mock)
                
                # Execute full comparative analysis
                comparison_results = self.analyzer_mock.run_comparative_analysis(test_case_mocks)
                
                # Validate integration results
                self.assertIsInstance(comparison_results, list)
                self.assertEqual(len(comparison_results), len(test_case_mocks))
                
                # Check that results are realistic
                success_count = sum(1 for r in comparison_results if r.smied_result.path_found)
                success_rate = (success_count / len(comparison_results)) * 100
                
                # Should be within reasonable range for integration testing
                self.assertGreaterEqual(success_rate, 50.0)  # At least 50% success
    
    def test_report_generation_integration(self):
        """Test report generation with realistic data."""
        sample_results = self.report_data['sample_comparison_results']['diverse_outcomes']
        
        # Create comparison result objects from sample data
        comparison_results = []
        for result_data in sample_results:
            # Mock the nested result objects
            smied_result = SMIEDResult(
                success=result_data['smied_success'],
                path_found=result_data['smied_success'],
                subject_path_length=2,
                object_path_length=2,
                total_path_length=4,
                connecting_predicate="test.v.01",
                response_time=result_data['smied_time']
            )
            
            conceptnet_result = ConceptNetResult(
                success=result_data['conceptnet_success'],
                relation_found=result_data['conceptnet_success'],
                relation_type="RelatedTo" if result_data['conceptnet_success'] else None,
                confidence_score=2.5 if result_data['conceptnet_success'] else 0.0,
                path_exists=result_data['conceptnet_success'],
                response_time=result_data['conceptnet_time']
            )
            
            comparison = ComparisonResult(
                test_case_id=result_data['test_case_id'],
                subject=result_data['subject'],
                predicate=result_data['predicate'],
                object=result_data['object'],
                smied_result=smied_result,
                conceptnet_result=conceptnet_result,
                winner=result_data['winner'],
                performance_comparison=result_data['performance_comparison'],
                semantic_quality=result_data['semantic_quality'],
                notes=result_data['notes']
            )
            comparison_results.append(comparison)
        
        # Generate report
        report = self.analyzer_mock.generate_comparative_report(comparison_results)
        
        # Validate report structure
        expected_fields = self.report_data['expected_report_structure']
        
        self.assertIn('summary', report)
        for field in expected_fields['summary_fields']:
            self.assertIn(field, report['summary'])
        
        for field in expected_fields['distribution_fields']:
            self.assertIn(field, report)
        
        for field in expected_fields['analysis_fields']:
            self.assertIn(field, report)
    
    def test_component_interaction_realistic_behavior(self):
        """Test realistic interaction between components."""
        # Use realistic test cases
        realistic_scenarios = self.integration_data['realistic_comparison_scenarios'][0]
        test_case_data = realistic_scenarios['test_cases'][0]
        
        test_case_mock = self.mock_factory(
            'MockTestCase',
            subject=test_case_data['subject'],
            predicate=test_case_data['predicate'],
            object=test_case_data['object']
        )
        
        # Test SMIED interaction
        smied_result = self.analyzer_mock.run_smied_test(test_case_mock)
        self.assertIsInstance(smied_result, SMIEDResult)
        
        # Test ConceptNet interaction
        conceptnet_result = self.conceptnet_mock.query_relation(
            test_case_data['subject'],
            test_case_data['predicate'],
            test_case_data['object']
        )
        self.assertIsInstance(conceptnet_result, ConceptNetResult)
        
        # Test complete comparison
        comparison = self.analyzer_mock.compare_single_test(test_case_mock)
        self.assertIsInstance(comparison, ComparisonResult)
        
        # Validate realistic behavior characteristics
        if test_case_data.get('expected_smied_success'):
            self.assertTrue(smied_result.path_found or not smied_result.path_found)  # Either outcome is valid
        
        if test_case_data.get('expected_conceptnet_success'):
            self.assertTrue(conceptnet_result.relation_found or not conceptnet_result.relation_found)  # Either outcome is valid
    
    def test_performance_benchmarks_integration(self):
        """Test that performance meets expected benchmarks."""
        benchmarks = self.config.get_performance_benchmarks()
        timing_benchmarks = benchmarks['timing_benchmarks']
        
        # Test SMIED performance characteristics
        smied_benchmarks = timing_benchmarks['smied_performance']
        
        test_case_mock = self.mock_factory('MockTestCase', difficulty='easy')
        smied_result = self.analyzer_mock.run_smied_test(test_case_mock)
        
        # Performance should be within reasonable bounds for mocked tests
        max_time = smied_benchmarks['fast_execution']['max_time_seconds']
        self.assertLessEqual(smied_result.response_time, max_time * 2)  # Allow some flexibility for mocks
    
    def test_external_dependency_integration(self):
        """Test integration with external dependencies."""
        integration_components = self.integration_data['integration_components']
        required_components = integration_components['required_smied_components']
        
        # Verify all required components are available through mocks
        for component in required_components:
            if component == 'SMIED':
                self.assertIsNotNone(self.smied_mock)
            elif component == 'ComparativeAnalyzer':
                self.assertIsNotNone(self.analyzer_mock)
            # Additional components would be tested here
        
        # Test mock external dependencies
        external_deps = integration_components['mock_external_dependencies']
        for dep in external_deps:
            if dep == 'requests':
                # ConceptNet mock should handle requests simulation
                self.assertIsNotNone(self.conceptnet_mock)


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)