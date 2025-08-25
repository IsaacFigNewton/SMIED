"""
Configuration data for regression testing.

This module provides structured test data for regression testing components
following the SMIED Testing Framework design specifications. All test data
is organized by testing scenario and accessed through static methods.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time


class RegressionMockConfig:
    """Configuration class for regression testing mock data."""
    
    @staticmethod
    def get_basic_test_data() -> Dict[str, Any]:
        """Get basic regression test data for standard scenarios."""
        return {
            'simple_test_cases': [
                {
                    'subject': 'cat',
                    'predicate': 'chase',
                    'object': 'mouse',
                    'expected_success': True,
                    'expected_subject_path_length': 2,
                    'expected_object_path_length': 2,
                    'expected_connecting_predicate': 'chase.v.01'
                },
                {
                    'subject': 'dog',
                    'predicate': 'bark',
                    'object': 'loudly',
                    'expected_success': True,
                    'expected_subject_path_length': 1,
                    'expected_object_path_length': 1,
                    'expected_connecting_predicate': 'bark.v.01'
                },
                {
                    'subject': 'bird',
                    'predicate': 'fly',
                    'object': 'sky',
                    'expected_success': True,
                    'expected_subject_path_length': 2,
                    'expected_object_path_length': 3,
                    'expected_connecting_predicate': 'fly.v.01'
                }
            ],
            'baseline_results': [
                {
                    'test_id': 'baseline_001',
                    'subject': 'cat',
                    'predicate': 'chase',
                    'object': 'mouse',
                    'success': True,
                    'subject_path_length': 2,
                    'object_path_length': 2,
                    'connecting_predicate': 'chase.v.01',
                    'execution_time': 0.1,
                    'version': '1.0'
                },
                {
                    'test_id': 'baseline_002',
                    'subject': 'dog',
                    'predicate': 'bark',
                    'object': 'loudly',
                    'success': True,
                    'subject_path_length': 1,
                    'object_path_length': 1,
                    'connecting_predicate': 'bark.v.01',
                    'execution_time': 0.05,
                    'version': '1.0'
                }
            ],
            'regression_test_results': [
                {
                    'test_id': 'reg_test_001',
                    'baseline_success': True,
                    'current_success': True,
                    'baseline_path_length': 4,
                    'current_path_length': 4,
                    'performance_change_percent': 0.0,
                    'status': 'PASS',
                    'details': 'No significant changes detected'
                },
                {
                    'test_id': 'reg_test_002',
                    'baseline_success': True,
                    'current_success': True,
                    'baseline_path_length': 4,
                    'current_path_length': 5,
                    'performance_change_percent': 5.0,
                    'status': 'PASS',
                    'details': 'Path length changed: 4 -> 5'
                }
            ]
        }
    
    @staticmethod
    def get_edge_case_scenarios() -> Dict[str, Any]:
        """Get edge case test data for regression testing."""
        return {
            'failure_scenarios': [
                {
                    'subject': 'nonexistent_word_xyz',
                    'predicate': 'impossible_relation',
                    'object': 'another_nonexistent',
                    'expected_success': False,
                    'expected_subject_path_length': 0,
                    'expected_object_path_length': 0,
                    'expected_connecting_predicate': None
                }
            ],
            'performance_regression_results': [
                {
                    'test_id': 'perf_reg_001',
                    'baseline_success': True,
                    'current_success': True,
                    'baseline_path_length': 4,
                    'current_path_length': 4,
                    'performance_change_percent': 75.0,  # Significant slowdown
                    'status': 'PERFORMANCE_REGRESSION',
                    'details': 'Performance degradation: +75.0%'
                }
            ],
            'functional_regression_results': [
                {
                    'test_id': 'func_reg_001',
                    'baseline_success': True,
                    'current_success': False,
                    'baseline_path_length': 4,
                    'current_path_length': 0,
                    'performance_change_percent': 0.0,
                    'status': 'FAIL',
                    'details': 'Functional regression: test now fails'
                }
            ],
            'improvement_results': [
                {
                    'test_id': 'improvement_001',
                    'baseline_success': False,
                    'current_success': True,
                    'baseline_path_length': 0,
                    'current_path_length': 4,
                    'performance_change_percent': -20.0,  # Performance improvement
                    'status': 'IMPROVEMENT',
                    'details': 'Functional improvement: test now passes; Performance improvement: -20.0%'
                }
            ],
            'empty_baselines': [],
            'invalid_test_ids': [
                '', None, 'test_with_special_chars!@#'
            ]
        }
    
    @staticmethod
    def get_integration_test_data() -> Dict[str, Any]:
        """Get integration test data for multi-component testing."""
        return {
            'cross_pos_test_cases': [
                # Noun-Verb-Noun patterns
                {'subject': 'cat', 'predicate': 'chase', 'object': 'mouse'},
                {'subject': 'dog', 'predicate': 'bark', 'object': 'loudly'},
                {'subject': 'child', 'predicate': 'play', 'object': 'game'},
                
                # Verb-Adverb patterns
                {'subject': 'run', 'predicate': 'quickly', 'object': 'fast'},
                {'subject': 'speak', 'predicate': 'softly', 'object': 'quiet'},
                
                # Complex semantic relationships
                {'subject': 'scientist', 'predicate': 'discover', 'object': 'truth'},
                {'subject': 'artist', 'predicate': 'create', 'object': 'beauty'},
                {'subject': 'teacher', 'predicate': 'explain', 'object': 'knowledge'}
            ],
            'mixed_success_baselines': [
                {
                    'test_id': 'mixed_001',
                    'subject': 'successful_case',
                    'predicate': 'works',
                    'object': 'well',
                    'success': True,
                    'subject_path_length': 3,
                    'object_path_length': 2,
                    'connecting_predicate': 'works.v.01',
                    'execution_time': 0.15
                },
                {
                    'test_id': 'mixed_002',
                    'subject': 'failing_case',
                    'predicate': 'fails',
                    'object': 'badly',
                    'success': False,
                    'subject_path_length': 0,
                    'object_path_length': 0,
                    'connecting_predicate': None,
                    'execution_time': 0.02
                }
            ],
            'large_dataset_simulation': {
                'test_case_count': 100,
                'expected_pass_rate': 85.0,
                'expected_performance_regression_rate': 5.0,
                'expected_improvement_rate': 10.0
            }
        }
    
    @staticmethod
    def get_performance_test_data() -> Dict[str, Any]:
        """Get performance-related test data for regression testing."""
        return {
            'performance_thresholds': {
                'acceptable_performance_change': 50.0,  # 50% threshold
                'significant_performance_change': 10.0,  # Report at 10%
                'execution_time_baseline': 0.1,  # 100ms baseline
                'timeout_threshold': 5.0  # 5 second timeout
            },
            'performance_scenarios': [
                {
                    'scenario_name': 'baseline_fast',
                    'baseline_time': 0.05,
                    'current_time': 0.05,
                    'expected_change_percent': 0.0,
                    'expected_status': 'PASS'
                },
                {
                    'scenario_name': 'minor_slowdown',
                    'baseline_time': 0.1,
                    'current_time': 0.11,
                    'expected_change_percent': 10.0,
                    'expected_status': 'PASS'
                },
                {
                    'scenario_name': 'major_slowdown',
                    'baseline_time': 0.1,
                    'current_time': 0.2,
                    'expected_change_percent': 100.0,
                    'expected_status': 'PERFORMANCE_REGRESSION'
                },
                {
                    'scenario_name': 'performance_improvement',
                    'baseline_time': 0.2,
                    'current_time': 0.1,
                    'expected_change_percent': -50.0,
                    'expected_status': 'IMPROVEMENT'
                }
            ],
            'trend_analysis_data': {
                'stable_trend': {
                    'recent_pass_rates': [95.0, 94.0, 96.0, 95.0, 95.0],
                    'expected_trend': 'stable'
                },
                'improving_trend': {
                    'recent_pass_rates': [85.0, 87.0, 90.0, 93.0, 95.0],
                    'expected_trend': 'improving'
                },
                'declining_trend': {
                    'recent_pass_rates': [95.0, 92.0, 88.0, 85.0, 80.0],
                    'expected_trend': 'declining'
                }
            }
        }
    
    @staticmethod
    def get_long_term_tracking_data() -> Dict[str, Any]:
        """Get long-term regression tracking test data."""
        return {
            'historical_runs': [
                {
                    'timestamp': time.time() - 86400 * 7,  # 7 days ago
                    'total_tests': 50,
                    'passed': 45,
                    'failed': 3,
                    'performance_regressions': 1,
                    'improvements': 1,
                    'metadata': {'version': '1.0', 'branch': 'main'}
                },
                {
                    'timestamp': time.time() - 86400 * 3,  # 3 days ago
                    'total_tests': 52,
                    'passed': 48,
                    'failed': 2,
                    'performance_regressions': 1,
                    'improvements': 1,
                    'metadata': {'version': '1.1', 'branch': 'main'}
                },
                {
                    'timestamp': time.time() - 86400,  # 1 day ago
                    'total_tests': 55,
                    'passed': 52,
                    'failed': 1,
                    'performance_regressions': 0,
                    'improvements': 2,
                    'metadata': {'version': '1.2', 'branch': 'main'}
                }
            ],
            'trend_expectations': {
                'total_runs': 3,
                'recent_average_pass_rate': 94.5,
                'trend': 'improving',
                'performance_regression_frequency': 0.33
            }
        }
    
    @staticmethod
    def get_report_generation_data() -> Dict[str, Any]:
        """Get test data for regression report generation."""
        return {
            'sample_regression_results': [
                # Mixed results for comprehensive reporting
                {
                    'test_id': 'report_test_001',
                    'baseline_success': True,
                    'current_success': True,
                    'baseline_path_length': 4,
                    'current_path_length': 4,
                    'performance_change_percent': 0.0,
                    'status': 'PASS',
                    'details': 'No significant changes detected'
                },
                {
                    'test_id': 'report_test_002',
                    'baseline_success': True,
                    'current_success': False,
                    'baseline_path_length': 4,
                    'current_path_length': 0,
                    'performance_change_percent': 0.0,
                    'status': 'FAIL',
                    'details': 'Functional regression: test now fails'
                },
                {
                    'test_id': 'report_test_003',
                    'baseline_success': True,
                    'current_success': True,
                    'baseline_path_length': 4,
                    'current_path_length': 4,
                    'performance_change_percent': 75.0,
                    'status': 'PERFORMANCE_REGRESSION',
                    'details': 'Performance degradation: +75.0%'
                },
                {
                    'test_id': 'report_test_004',
                    'baseline_success': False,
                    'current_success': True,
                    'baseline_path_length': 0,
                    'current_path_length': 4,
                    'performance_change_percent': -25.0,
                    'status': 'IMPROVEMENT',
                    'details': 'Functional improvement: test now passes'
                }
            ],
            'expected_report_structure': {
                'summary': {
                    'total_tests': 4,
                    'passed': 1,
                    'failed': 1,
                    'performance_regressions': 1,
                    'improvements': 1,
                    'pass_rate': 25.0
                },
                'performance': {
                    'average_performance_change': 12.5,
                    'tests_with_perf_changes': 2
                },
                'regressions': {
                    'functional_regressions': 1,
                    'new_failures': 1
                }
            }
        }
    
    @staticmethod
    def get_file_system_mock_data() -> Dict[str, Any]:
        """Get file system related mock data for testing."""
        return {
            'baseline_files': {
                'default': 'regression_baseline.json',
                'test': 'test_baseline.json',
                'mock': 'mock_baseline.json'
            },
            'history_files': {
                'default': 'regression_history.json',
                'test': 'test_history.json',
                'mock': 'mock_history.json'
            },
            'mock_file_paths': {
                'baseline_path': '/mock/path/regression_baseline.json',
                'history_path': '/mock/path/regression_history.json',
                'results_path': '/mock/path/regression_results_123456.json'
            },
            'sample_json_content': {
                'baseline_json': {
                    'test_001': {
                        'test_id': 'test_001',
                        'subject': 'cat',
                        'predicate': 'chase',
                        'object': 'mouse',
                        'success': True,
                        'subject_path_length': 2,
                        'object_path_length': 2,
                        'connecting_predicate': 'chase.v.01',
                        'execution_time': 0.1,
                        'version': '1.0'
                    }
                },
                'history_json': {
                    'runs': [
                        {
                            'timestamp': 1640995200.0,
                            'total_tests': 10,
                            'passed': 9,
                            'failed': 1,
                            'performance_regressions': 0,
                            'improvements': 0
                        }
                    ],
                    'metadata': {
                        'created': 1640995200.0
                    }
                }
            }
        }
    
    @staticmethod
    def get_mock_factory_test_data() -> Dict[str, Any]:
        """Get test data for mock factory testing."""
        return {
            'available_mock_types': [
                'BaselineResultMock',
                'RegressionTestResultMock', 
                'MockRegressionBaseline',
                'MockRegressionTester',
                'MockLongTermRegressionTracker',
                'MockTestCase',
                'MockSemanticPathfindingTestSuite'
            ],
            'factory_method_mapping': {
                'create_baseline_result': 'BaselineResultMock',
                'create_regression_test_result': 'RegressionTestResultMock',
                'create_regression_baseline': 'MockRegressionBaseline',
                'create_regression_tester': 'MockRegressionTester',
                'create_trend_tracker': 'MockLongTermRegressionTracker',
                'create_test_case': 'MockTestCase',
                'create_test_suite': 'MockSemanticPathfindingTestSuite'
            },
            'invalid_mock_names': [
                'InvalidMock',
                'NonExistentMock',
                'UndefinedMock'
            ]
        }