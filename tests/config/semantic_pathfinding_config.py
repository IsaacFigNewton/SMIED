"""
Configuration class containing mock constants and test data for semantic pathfinding tests.

This module follows the SMIED Testing Framework Design Specifications
for configuration-driven test data management with static methods.
"""


class SemanticPathfindingMockConfig:
    """Configuration class containing mock constants and test data for semantic pathfinding tests."""
    
    @staticmethod
    def get_basic_test_cases():
        """Get basic semantic pathfinding test cases."""
        return {
            'simple_cases': [
                {
                    'subject': 'cat',
                    'predicate': 'chase', 
                    'object': 'mouse',
                    'expected_success': True,
                    'description': 'Cat chasing mouse - classic predator-prey',
                    'semantic_relationship': 'predator_prey',
                    'difficulty_level': 'easy',
                    'cross_pos': True
                },
                {
                    'subject': 'dog',
                    'predicate': 'bark',
                    'object': 'intruder',
                    'expected_success': True,
                    'description': 'Dog barking at intruder - protective behavior',
                    'semantic_relationship': 'protective_action',
                    'difficulty_level': 'easy',
                    'cross_pos': True
                },
                {
                    'subject': 'bird',
                    'predicate': 'build',
                    'object': 'nest',
                    'expected_success': True,
                    'description': 'Bird building nest - natural behavior',
                    'semantic_relationship': 'natural_behavior',
                    'difficulty_level': 'easy',
                    'cross_pos': True
                }
            ],
            'medium_cases': [
                {
                    'subject': 'chef',
                    'predicate': 'cook',
                    'object': 'meal',
                    'expected_success': True,
                    'description': 'Chef cooking meal - occupational function',
                    'semantic_relationship': 'occupational_function',
                    'difficulty_level': 'medium',
                    'cross_pos': True
                },
                {
                    'subject': 'teacher',
                    'predicate': 'explain',
                    'object': 'concept',
                    'expected_success': True,
                    'description': 'Teacher explaining concept - educational role',
                    'semantic_relationship': 'educational_role',
                    'difficulty_level': 'medium',
                    'cross_pos': True
                }
            ],
            'hard_cases': [
                {
                    'subject': 'student',
                    'predicate': 'understand',
                    'object': 'mathematics',
                    'expected_success': True,
                    'description': 'Student understanding mathematics - cognitive process',
                    'semantic_relationship': 'cognitive_process',
                    'difficulty_level': 'hard',
                    'cross_pos': True
                },
                {
                    'subject': 'scientist',
                    'predicate': 'discover',
                    'object': 'truth',
                    'expected_success': True,
                    'description': 'Scientist discovering truth - research goal',
                    'semantic_relationship': 'research_goal',
                    'difficulty_level': 'hard',
                    'cross_pos': True
                }
            ]
        }
    
    @staticmethod
    def get_negative_test_cases():
        """Get negative test cases that should fail."""
        return {
            'impossible_cases': [
                {
                    'subject': 'rock',
                    'predicate': 'sing',
                    'object': 'opera',
                    'expected_success': False,
                    'description': 'Rock singing opera - impossible action',
                    'semantic_relationship': 'impossible_action',
                    'difficulty_level': 'hard',
                    'cross_pos': True
                },
                {
                    'subject': 'color',
                    'predicate': 'calculate',
                    'object': 'mathematics',
                    'expected_success': False,
                    'description': 'Color calculating mathematics - nonsensical',
                    'semantic_relationship': 'nonsensical',
                    'difficulty_level': 'hard',
                    'cross_pos': True
                },
                {
                    'subject': 'silence',
                    'predicate': 'eat',
                    'object': 'sound',
                    'expected_success': False,
                    'description': 'Silence eating sound - abstract impossibility',
                    'semantic_relationship': 'abstract_impossibility',
                    'difficulty_level': 'hard',
                    'cross_pos': True
                }
            ]
        }
    
    @staticmethod
    def get_edge_case_scenarios():
        """Get edge case scenarios for pathfinding."""
        return {
            'timeout_scenarios': [
                {
                    'subject': 'timeout_test',
                    'predicate': 'trigger',
                    'object': 'timeout',
                    'max_depth': 20,
                    'beam_width': 50,
                    'expected_exception': 'TimeoutError'
                }
            ],
            'memory_scenarios': [
                {
                    'subject': 'memory_test',
                    'predicate': 'exhaust',
                    'object': 'memory',
                    'max_depth': 30,
                    'beam_width': 100,
                    'expected_exception': 'MemoryError'
                }
            ],
            'error_scenarios': [
                {
                    'subject': 'error_test',
                    'predicate': 'cause',
                    'object': 'error',
                    'expected_exception': 'ValueError'
                }
            ],
            'boundary_scenarios': [
                {
                    'subject': 'valid',
                    'predicate': 'test',
                    'object': 'boundary',
                    'max_depth': 1,
                    'beam_width': 1,
                    'expected_success': True
                },
                {
                    'subject': 'empty',
                    'predicate': '',
                    'object': 'test',
                    'expected_exception': 'ValueError'
                }
            ]
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get test data for validation scenarios."""
        return {
            'valid_results': {
                'short_path': {
                    'success': True,
                    'subject_path': ['cat.n.01', 'predicate'],
                    'object_path': ['predicate', 'mouse.n.01'],
                    'connecting_predicate_name': 'chase.v.01',
                    'execution_time': 0.1,
                    'expected_validation': {
                        'is_valid_path': True,
                        'path_length_reasonable': True,
                        'connecting_predicate_relevant': True,
                    }
                },
                'medium_path': {
                    'success': True,
                    'subject_path': ['teacher.n.01', 'person.n.01', 'intermediate'],
                    'object_path': ['intermediate', 'knowledge.n.01', 'concept.n.01'],
                    'connecting_predicate_name': 'explain.v.01',
                    'execution_time': 0.15,
                    'expected_validation': {
                        'is_valid_path': True,
                        'path_length_reasonable': True,
                        'connecting_predicate_relevant': True,
                    }
                }
            },
            'invalid_results': {
                'no_path': {
                    'success': False,
                    'subject_path': None,
                    'object_path': None,
                    'connecting_predicate': None,
                    'execution_time': 0.05,
                    'error': 'No path found',
                    'expected_validation': {
                        'is_valid_path': False,
                        'issues': ['No path found']
                    }
                },
                'too_long_path': {
                    'success': True,
                    'subject_path': ['a'] * 10,
                    'object_path': ['b'] * 10,
                    'connecting_predicate_name': 'unrelated.v.01',
                    'execution_time': 0.5,
                    'expected_validation': {
                        'is_valid_path': True,
                        'path_length_reasonable': False,
                        'connecting_predicate_relevant': False,
                        'issues': ['Path too long: 20 steps', 'Connecting predicate \'unrelated.v.01\' doesn\'t match \'test\'']
                    }
                }
            }
        }
    
    @staticmethod
    def get_integration_test_data():
        """Get test data for integration scenarios."""
        return {
            'realistic_scenarios': [
                {
                    'name': 'animal_action_scenario',
                    'test_cases': [
                        {
                            'subject': 'cat',
                            'predicate': 'hunt',
                            'object': 'prey',
                            'expected_success': True,
                            'semantic_relationship': 'predator_behavior'
                        },
                        {
                            'subject': 'dog',
                            'predicate': 'guard',
                            'object': 'house',
                            'expected_success': True,
                            'semantic_relationship': 'protective_behavior'
                        }
                    ],
                    'expected_overall_success_rate': 90.0
                },
                {
                    'name': 'human_activity_scenario',
                    'test_cases': [
                        {
                            'subject': 'artist',
                            'predicate': 'create',
                            'object': 'painting',
                            'expected_success': True,
                            'semantic_relationship': 'creative_activity'
                        },
                        {
                            'subject': 'musician',
                            'predicate': 'perform',
                            'object': 'concert',
                            'expected_success': True,
                            'semantic_relationship': 'artistic_performance'
                        }
                    ],
                    'expected_overall_success_rate': 85.0
                }
            ],
            'integration_components': {
                'required_smied_components': ['SemanticDecomposer', 'PairwiseBidirectionalAStar', 'SMIED'],
                'optional_components': ['EmbeddingHelper', 'BeamBuilder'],
                'mock_external_dependencies': ['nlp_model', 'wordnet']
            }
        }
    
    @staticmethod
    def get_performance_benchmarks():
        """Get performance benchmarks for testing."""
        return {
            'timing_benchmarks': {
                'fast_execution': {
                    'max_time_seconds': 0.1,
                    'typical_cases': ['easy_difficulty'],
                    'beam_width': 3,
                    'max_depth': 5
                },
                'medium_execution': {
                    'max_time_seconds': 0.3,
                    'typical_cases': ['medium_difficulty'],
                    'beam_width': 5,
                    'max_depth': 10
                },
                'slow_execution': {
                    'max_time_seconds': 1.0,
                    'typical_cases': ['hard_difficulty'],
                    'beam_width': 10,
                    'max_depth': 15
                }
            },
            'memory_benchmarks': {
                'low_memory': {
                    'max_memory_mb': 10,
                    'typical_cases': ['small_search_space'],
                    'beam_width': 3,
                    'max_depth': 5
                },
                'medium_memory': {
                    'max_memory_mb': 50,
                    'typical_cases': ['medium_search_space'],
                    'beam_width': 7,
                    'max_depth': 10
                },
                'high_memory': {
                    'max_memory_mb': 200,
                    'typical_cases': ['large_search_space'],
                    'beam_width': 15,
                    'max_depth': 20
                }
            },
            'success_rate_benchmarks': {
                'easy_cases': {
                    'minimum_success_rate': 90.0,
                    'target_success_rate': 95.0
                },
                'medium_cases': {
                    'minimum_success_rate': 70.0,
                    'target_success_rate': 80.0
                },
                'hard_cases': {
                    'minimum_success_rate': 50.0,
                    'target_success_rate': 65.0
                }
            }
        }
    
    @staticmethod
    def get_parameter_sensitivity_test_data():
        """Get test data for parameter sensitivity analysis."""
        return {
            'beam_width_variations': [1, 3, 5, 7, 10, 15, 20],
            'max_depth_variations': [3, 5, 8, 10, 12, 15, 20],
            'test_subset_size': 10,  # Number of test cases to use for parameter testing
            'expected_trends': {
                'beam_width': {
                    'success_rate': 'increases_then_plateaus',
                    'execution_time': 'increases_linearly',
                    'memory_usage': 'increases_exponentially'
                },
                'max_depth': {
                    'success_rate': 'increases_then_plateaus',
                    'execution_time': 'increases_exponentially',
                    'memory_usage': 'increases_linearly'
                }
            }
        }
    
    @staticmethod
    def get_scalability_test_data():
        """Get test data for scalability analysis."""
        return {
            'difficulty_groups': {
                'easy': {
                    'expected_min_cases': 8,
                    'expected_success_rate_range': [90, 100],
                    'expected_avg_time_range': [0.01, 0.1],
                    'expected_memory_usage_range': [1, 10]
                },
                'medium': {
                    'expected_min_cases': 6,
                    'expected_success_rate_range': [60, 85],
                    'expected_avg_time_range': [0.05, 0.2],
                    'expected_memory_usage_range': [5, 25]
                },
                'hard': {
                    'expected_min_cases': 4,
                    'expected_success_rate_range': [40, 70],
                    'expected_avg_time_range': [0.1, 0.5],
                    'expected_memory_usage_range': [10, 50]
                }
            }
        }
    
    @staticmethod
    def get_comprehensive_test_configuration():
        """Get configuration for comprehensive testing."""
        return {
            'test_execution_settings': {
                'default_max_depth': 10,
                'default_beam_width': 3,
                'default_measure_memory': True,
                'default_verbosity': 0,
                'timeout_seconds': 30
            },
            'report_generation_settings': {
                'include_detailed_analysis': True,
                'include_difficulty_breakdown': True,
                'include_semantic_relationship_analysis': True,
                'max_failed_tests_shown': 10,
                'max_quality_issues_shown': 20
            },
            'validation_settings': {
                'max_reasonable_path_length': 15,
                'min_reasonable_path_length': 2,
                'require_connecting_predicate_relevance': True
            }
        }
    
    @staticmethod
    def get_mock_setup_configurations():
        """Get configurations for mock setup in different test scenarios."""
        return {
            'basic_setup': {
                'pathfinding_suite_mock': 'MockSemanticPathfindingSuite',
                'validator_mock': 'MockSemanticPathfindingValidator',
                'benchmark_mock': 'MockSemanticPathfindingBenchmark',
                'smied_mock': 'MockSMIEDForPathfinding',
                'enable_memory_tracking': True
            },
            'validation_setup': {
                'pathfinding_suite_mock': 'MockSemanticPathfindingValidation',
                'validator_mock': 'MockSemanticPathfindingValidator',
                'smied_mock': 'MockSMIEDValidation',
                'enable_strict_validation': True,
                'validation_criteria': 'strict'
            },
            'edge_case_setup': {
                'pathfinding_suite_mock': 'MockSemanticPathfindingEdgeCases',
                'smied_mock': 'MockSMIEDEdgeCases',
                'simulate_failures': True,
                'enable_timeout_testing': True,
                'enable_memory_exhaustion_testing': True
            },
            'integration_setup': {
                'pathfinding_suite_mock': 'MockSemanticPathfindingIntegration',
                'smied_mock': 'MockSMIEDIntegration',
                'use_realistic_behavior': True,
                'enable_component_interaction_testing': True,
                'external_dependencies': ['wordnet', 'nlp_model']
            }
        }
    
    @staticmethod
    def get_expected_test_outcomes():
        """Get expected outcomes for different test scenarios."""
        return {
            'basic_functionality': {
                'test_count_range': [25, 35],
                'overall_success_rate_range': [75, 95],
                'average_execution_time_range': [0.01, 0.2],
                'expected_test_types': ['easy', 'medium', 'hard', 'negative']
            },
            'validation_tests': {
                'validation_coverage': ['path_quality', 'predicate_relevance'],
                'expected_validation_issues_count': [0, 5],
            },
            'edge_case_tests': {
                'expected_exceptions': ['TimeoutError', 'MemoryError', 'ValueError'],
                'boundary_condition_coverage': ['empty_inputs', 'extreme_parameters', 'malformed_data'],
                'graceful_failure_required': True
            },
            'integration_tests': {
                'component_interaction_required': ['SMIED', 'SemanticDecomposer', 'PairwiseBidirectionalAStar'],
                'realistic_behavior_verification': True,
                'end_to_end_functionality': True
            }
        }