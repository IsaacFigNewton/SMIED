"""
Configuration class containing mock constants and test data for comparative analysis tests.

This module follows the SMIED Testing Framework Design Specifications
for configuration-driven test data management with static methods.
"""

from typing import Dict, List, Any, Optional


class ComparativeAnalysisMockConfig:
    """Configuration class containing mock constants and test data for comparative analysis tests."""
    
    @staticmethod
    def get_basic_test_cases():
        """Get basic comparative analysis test cases."""
        return {
            'simple_comparison_cases': [
                {
                    'subject': 'cat',
                    'predicate': 'chase',
                    'object': 'mouse',
                    'expected_smied_success': True,
                    'expected_conceptnet_success': True,
                    'description': 'Cat chasing mouse - classic predator-prey relationship',
                    'semantic_relationship': 'predator_prey',
                    'difficulty_level': 'easy',
                    'expected_winner': 'Tie'
                },
                {
                    'subject': 'dog',
                    'predicate': 'bark',
                    'object': 'intruder',
                    'expected_smied_success': True,
                    'expected_conceptnet_success': True,
                    'description': 'Dog barking at intruder - protective behavior',
                    'semantic_relationship': 'protective_action',
                    'difficulty_level': 'easy',
                    'expected_winner': 'Tie'
                },
                {
                    'subject': 'bird',
                    'predicate': 'fly',
                    'object': 'sky',
                    'expected_smied_success': True,
                    'expected_conceptnet_success': True,
                    'description': 'Bird flying in sky - natural behavior',
                    'semantic_relationship': 'natural_behavior',
                    'difficulty_level': 'easy',
                    'expected_winner': 'Tie'
                },
                {
                    'subject': 'fish',
                    'predicate': 'swim',
                    'object': 'water',
                    'expected_smied_success': True,
                    'expected_conceptnet_success': True,
                    'description': 'Fish swimming in water - natural environment',
                    'semantic_relationship': 'natural_environment',
                    'difficulty_level': 'easy',
                    'expected_winner': 'Tie'
                },
                {
                    'subject': 'person',
                    'predicate': 'read',
                    'object': 'book',
                    'expected_smied_success': True,
                    'expected_conceptnet_success': True,
                    'description': 'Person reading book - human activity',
                    'semantic_relationship': 'human_activity',
                    'difficulty_level': 'easy',
                    'expected_winner': 'Tie'
                }
            ],
            'medium_comparison_cases': [
                {
                    'subject': 'scientist',
                    'predicate': 'research',
                    'object': 'hypothesis',
                    'expected_smied_success': True,
                    'expected_conceptnet_success': True,
                    'description': 'Scientist researching hypothesis - professional activity',
                    'semantic_relationship': 'professional_activity',
                    'difficulty_level': 'medium',
                    'expected_winner': 'SMIED'
                },
                {
                    'subject': 'artist',
                    'predicate': 'create',
                    'object': 'masterpiece',
                    'expected_smied_success': True,
                    'expected_conceptnet_success': False,
                    'description': 'Artist creating masterpiece - creative process',
                    'semantic_relationship': 'creative_process',
                    'difficulty_level': 'medium',
                    'expected_winner': 'SMIED'
                }
            ],
            'complex_comparison_cases': [
                {
                    'subject': 'algorithm',
                    'predicate': 'optimize',
                    'object': 'solution',
                    'expected_smied_success': True,
                    'expected_conceptnet_success': False,
                    'description': 'Algorithm optimizing solution - computational process',
                    'semantic_relationship': 'computational_process',
                    'difficulty_level': 'hard',
                    'expected_winner': 'SMIED'
                }
            ]
        }
    
    @staticmethod
    def get_negative_test_cases():
        """Get negative test cases where both systems should fail."""
        return {
            'impossible_cases': [
                {
                    'subject': 'rock',
                    'predicate': 'compose',
                    'object': 'symphony',
                    'expected_smied_success': False,
                    'expected_conceptnet_success': False,
                    'description': 'Rock composing symphony - impossible action',
                    'semantic_relationship': 'impossible_action',
                    'difficulty_level': 'hard',
                    'expected_winner': 'Both_Failed'
                },
                {
                    'subject': 'color',
                    'predicate': 'calculate',
                    'object': 'mathematics',
                    'expected_smied_success': False,
                    'expected_conceptnet_success': False,
                    'description': 'Color calculating mathematics - nonsensical relationship',
                    'semantic_relationship': 'nonsensical',
                    'difficulty_level': 'hard',
                    'expected_winner': 'Both_Failed'
                }
            ],
            'abstract_impossibilities': [
                {
                    'subject': 'silence',
                    'predicate': 'eat',
                    'object': 'sound',
                    'expected_smied_success': False,
                    'expected_conceptnet_success': False,
                    'description': 'Silence eating sound - abstract impossibility',
                    'semantic_relationship': 'abstract_impossibility',
                    'difficulty_level': 'hard',
                    'expected_winner': 'Both_Failed'
                }
            ]
        }
    
    @staticmethod
    def get_edge_case_scenarios():
        """Get edge case scenarios for comparative analysis."""
        return {
            'timeout_scenarios': [
                {
                    'subject': 'timeout_test',
                    'predicate': 'trigger',
                    'object': 'timeout',
                    'timeout_seconds': 0.1,
                    'expected_smied_error': 'Request timeout',
                    'expected_conceptnet_error': 'Request timeout',
                    'expected_winner': 'Both_Failed'
                }
            ],
            'api_error_scenarios': [
                {
                    'subject': 'error_test',
                    'predicate': 'cause',
                    'object': 'error',
                    'expected_smied_error': None,
                    'expected_conceptnet_error': 'HTTP 500 Server Error',
                    'expected_winner': 'SMIED'
                }
            ],
            'network_failure_scenarios': [
                {
                    'subject': 'network_test',
                    'predicate': 'fail',
                    'object': 'connection',
                    'simulate_network_failure': True,
                    'expected_conceptnet_error': 'Connection failed',
                    'expected_winner': 'SMIED'
                }
            ],
            'extreme_path_scenarios': [
                {
                    'subject': 'long_path_test',
                    'predicate': 'traverse',
                    'object': 'complex_path',
                    'expected_smied_path_length': 22,
                    'expected_smied_time': 5.0,
                    'expected_conceptnet_confidence': 0.5,
                    'expected_winner': 'ConceptNet'
                }
            ],
            'boundary_scenarios': [
                {
                    'subject': '',
                    'predicate': 'test',
                    'object': 'boundary',
                    'expected_validation_error': True,
                    'expected_winner': 'Both_Failed'
                },
                {
                    'subject': 'test',
                    'predicate': '',
                    'object': 'boundary',
                    'expected_validation_error': True,
                    'expected_winner': 'Both_Failed'
                }
            ]
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get test data for validation scenarios."""
        return {
            'valid_smied_results': {
                'short_path': {
                    'success': True,
                    'path_found': True,
                    'subject_path_length': 2,
                    'object_path_length': 2,
                    'total_path_length': 4,
                    'connecting_predicate': 'chase.v.01',
                    'response_time': 0.1,
                    'error': None
                },
                'medium_path': {
                    'success': True,
                    'path_found': True,
                    'subject_path_length': 5,
                    'object_path_length': 6,
                    'total_path_length': 11,
                    'connecting_predicate': 'research.v.01',
                    'response_time': 0.25,
                    'error': None
                }
            },
            'invalid_smied_results': {
                'no_path': {
                    'success': False,
                    'path_found': False,
                    'subject_path_length': 0,
                    'object_path_length': 0,
                    'total_path_length': 0,
                    'connecting_predicate': None,
                    'response_time': 0.05,
                    'error': 'No path found'
                },
                'timeout_result': {
                    'success': False,
                    'path_found': False,
                    'subject_path_length': 0,
                    'object_path_length': 0,
                    'total_path_length': 0,
                    'connecting_predicate': None,
                    'response_time': 10.0,
                    'error': 'Request timeout'
                }
            },
            'valid_conceptnet_results': {
                'strong_relation': {
                    'success': True,
                    'relation_found': True,
                    'relation_type': 'RelatedTo',
                    'confidence_score': 3.5,
                    'path_exists': True,
                    'response_time': 0.2,
                    'error': None,
                    'raw_response': {'edges': [{'rel': {'label': 'RelatedTo'}, 'weight': 3.5}]}
                },
                'weak_relation': {
                    'success': True,
                    'relation_found': True,
                    'relation_type': 'RelatedTo',
                    'confidence_score': 1.2,
                    'path_exists': True,
                    'response_time': 0.3,
                    'error': None,
                    'raw_response': {'edges': [{'rel': {'label': 'RelatedTo'}, 'weight': 1.2}]}
                }
            },
            'invalid_conceptnet_results': {
                'no_relation': {
                    'success': True,
                    'relation_found': False,
                    'relation_type': None,
                    'confidence_score': 0.0,
                    'path_exists': False,
                    'response_time': 0.25,
                    'error': None,
                    'raw_response': {'edges': []}
                },
                'api_error': {
                    'success': False,
                    'relation_found': False,
                    'relation_type': None,
                    'confidence_score': 0.0,
                    'path_exists': False,
                    'response_time': 0.1,
                    'error': 'HTTP 500 Server Error',
                    'raw_response': None
                }
            }
        }
    
    @staticmethod
    def get_integration_test_data():
        """Get test data for integration scenarios."""
        return {
            'realistic_comparison_scenarios': [
                {
                    'name': 'animal_behavior_comparison',
                    'test_cases': [
                        {
                            'subject': 'cat',
                            'predicate': 'hunt',
                            'object': 'prey',
                            'expected_smied_success': True,
                            'expected_conceptnet_success': True,
                            'expected_winner': 'Tie',
                            'semantic_relationship': 'predator_behavior'
                        },
                        {
                            'subject': 'dog',
                            'predicate': 'guard',
                            'object': 'house',
                            'expected_smied_success': True,
                            'expected_conceptnet_success': True,
                            'expected_winner': 'Tie',
                            'semantic_relationship': 'protective_behavior'
                        }
                    ],
                    'expected_smied_success_rate': 85.0,
                    'expected_conceptnet_success_rate': 75.0
                },
                {
                    'name': 'human_activity_comparison',
                    'test_cases': [
                        {
                            'subject': 'teacher',
                            'predicate': 'educate',
                            'object': 'student',
                            'expected_smied_success': True,
                            'expected_conceptnet_success': True,
                            'expected_winner': 'Tie',
                            'semantic_relationship': 'educational_relationship'
                        },
                        {
                            'subject': 'doctor',
                            'predicate': 'heal',
                            'object': 'patient',
                            'expected_smied_success': True,
                            'expected_conceptnet_success': True,
                            'expected_winner': 'SMIED',
                            'semantic_relationship': 'professional_care'
                        }
                    ],
                    'expected_smied_success_rate': 90.0,
                    'expected_conceptnet_success_rate': 70.0
                }
            ],
            'integration_components': {
                'required_smied_components': ['SMIED', 'ComparativeAnalyzer', 'SemanticPathfindingTestSuite'],
                'required_conceptnet_components': ['ConceptNetInterface', 'RequestsSession'],
                'mock_external_dependencies': ['requests', 'time', 'json']
            },
            'external_api_simulation': {
                'conceptnet_api_responses': {
                    'successful_response': {
                        'status_code': 200,
                        'json_data': {
                            'edges': [
                                {
                                    'rel': {'label': 'RelatedTo'},
                                    'start': {'label': 'cat'},
                                    'end': {'label': 'mouse'},
                                    'weight': 2.8
                                }
                            ]
                        }
                    },
                    'not_found_response': {
                        'status_code': 200,
                        'json_data': {'edges': []}
                    },
                    'server_error_response': {
                        'status_code': 500,
                        'json_data': {'error': 'Internal server error'}
                    }
                }
            }
        }
    
    @staticmethod
    def get_performance_benchmarks():
        """Get performance benchmarks for comparative analysis."""
        return {
            'timing_benchmarks': {
                'smied_performance': {
                    'fast_execution': {
                        'max_time_seconds': 0.15,
                        'typical_cases': ['easy_difficulty'],
                        'expected_success_rate': 90.0
                    },
                    'medium_execution': {
                        'max_time_seconds': 0.5,
                        'typical_cases': ['medium_difficulty'],
                        'expected_success_rate': 75.0
                    },
                    'slow_execution': {
                        'max_time_seconds': 2.0,
                        'typical_cases': ['hard_difficulty'],
                        'expected_success_rate': 60.0
                    }
                },
                'conceptnet_performance': {
                    'fast_execution': {
                        'max_time_seconds': 0.3,
                        'typical_cases': ['common_relations'],
                        'expected_success_rate': 85.0
                    },
                    'medium_execution': {
                        'max_time_seconds': 1.0,
                        'typical_cases': ['uncommon_relations'],
                        'expected_success_rate': 65.0
                    },
                    'slow_execution': {
                        'max_time_seconds': 5.0,
                        'typical_cases': ['complex_relations'],
                        'expected_success_rate': 45.0
                    }
                }
            },
            'comparison_benchmarks': {
                'winner_distribution': {
                    'expected_smied_wins': 40,
                    'expected_conceptnet_wins': 30,
                    'expected_ties': 25,
                    'expected_both_failed': 5
                },
                'semantic_quality_distribution': {
                    'smied_better': 35,
                    'conceptnet_better': 25,
                    'similar': 35,
                    'unknown': 5
                }
            }
        }
    
    @staticmethod
    def get_report_generation_test_data():
        """Get test data for report generation scenarios."""
        return {
            'sample_comparison_results': {
                'diverse_outcomes': [
                    {
                        'test_case_id': 'cat_chase_mouse',
                        'subject': 'cat',
                        'predicate': 'chase',
                        'object': 'mouse',
                        'winner': 'Tie',
                        'performance_comparison': 'SMIED_Faster',
                        'semantic_quality': 'Similar',
                        'smied_success': True,
                        'conceptnet_success': True,
                        'smied_time': 0.1,
                        'conceptnet_time': 0.25,
                        'notes': 'Both systems found valid relationships'
                    },
                    {
                        'test_case_id': 'scientist_research_hypothesis',
                        'subject': 'scientist',
                        'predicate': 'research',
                        'object': 'hypothesis',
                        'winner': 'SMIED',
                        'performance_comparison': 'Similar',
                        'semantic_quality': 'SMIED_Better',
                        'smied_success': True,
                        'conceptnet_success': False,
                        'smied_time': 0.3,
                        'conceptnet_time': 0.35,
                        'notes': 'SMIED found sophisticated semantic path'
                    },
                    {
                        'test_case_id': 'rock_compose_symphony',
                        'subject': 'rock',
                        'predicate': 'compose',
                        'object': 'symphony',
                        'winner': 'Both_Failed',
                        'performance_comparison': 'Similar',
                        'semantic_quality': 'Similar',
                        'smied_success': False,
                        'conceptnet_success': False,
                        'smied_time': 0.05,
                        'conceptnet_time': 0.1,
                        'notes': 'Impossible relationship correctly rejected by both systems'
                    }
                ]
            },
            'expected_report_structure': {
                'summary_fields': [
                    'total_tests',
                    'smied_success_rate',
                    'conceptnet_success_rate',
                    'conceptnet_available'
                ],
                'distribution_fields': [
                    'winner_distribution',
                    'performance_comparison',
                    'semantic_quality_distribution'
                ],
                'analysis_fields': [
                    'response_times',
                    'smied_path_analysis',
                    'detailed_failures'
                ]
            }
        }
    
    @staticmethod
    def get_mock_setup_configurations():
        """Get configurations for mock setup in different test scenarios."""
        return {
            'basic_setup': {
                'comparative_analyzer_mock': 'MockComparativeAnalyzer',
                'conceptnet_interface_mock': 'MockConceptNetInterface',
                'smied_instance_mock': 'MockSMIEDInstance',
                'test_suite_mock': 'MockSemanticPathfindingTestSuite',
                'enable_conceptnet_api': True,
                'verbosity': 0
            },
            'validation_setup': {
                'comparative_analyzer_mock': 'MockComparativeAnalyzerValidation',
                'conceptnet_interface_mock': 'MockConceptNetInterface',
                'smied_instance_mock': 'MockSMIEDInstance',
                'enable_strict_validation': True,
                'validation_criteria': 'strict',
                'verbosity': 1
            },
            'edge_case_setup': {
                'comparative_analyzer_mock': 'MockComparativeAnalyzerEdgeCases',
                'conceptnet_interface_mock': 'MockConceptNetInterfaceEdgeCases',
                'smied_instance_mock': 'MockSMIEDInstance',
                'simulate_failures': True,
                'enable_timeout_testing': True,
                'enable_api_error_testing': True,
                'conceptnet_available': False,
                'verbosity': 0
            },
            'integration_setup': {
                'comparative_analyzer_mock': 'MockComparativeAnalyzerIntegration',
                'conceptnet_interface_mock': 'MockConceptNetIntegration',
                'smied_instance_mock': 'MockSMIEDInstance',
                'test_suite_mock': 'MockSemanticPathfindingTestSuite',
                'use_realistic_behavior': True,
                'enable_component_interaction_testing': True,
                'external_dependencies': ['requests', 'conceptnet_api'],
                'verbosity': 1
            }
        }
    
    @staticmethod
    def get_expected_test_outcomes():
        """Get expected outcomes for different test scenarios."""
        return {
            'basic_functionality': {
                'test_count_range': [20, 30],
                'overall_smied_success_rate_range': [70, 90],
                'overall_conceptnet_success_rate_range': [60, 85],
                'expected_test_types': ['simple', 'medium', 'complex', 'negative'],
                'expected_winners': ['SMIED', 'ConceptNet', 'Tie', 'Both_Failed']
            },
            'validation_tests': {
                'validation_coverage': [
                    'input_validation',
                    'result_structure_validation',
                    'comparison_logic_validation'
                ],
                'expected_validation_errors': [
                    'missing_subject',
                    'missing_predicate',
                    'missing_object',
                    'invalid_result_structure'
                ],
                'validation_success_rate_range': [85, 100]
            },
            'edge_case_tests': {
                'expected_exceptions': [
                    'TimeoutError',
                    'ConnectionError',
                    'HTTPError',
                    'ValueError'
                ],
                'boundary_condition_coverage': [
                    'empty_inputs',
                    'network_failures',
                    'api_errors',
                    'extreme_response_times'
                ],
                'graceful_failure_required': True,
                'edge_case_handling_success_rate': [80, 100]
            },
            'integration_tests': {
                'component_interaction_required': [
                    'SMIED',
                    'ConceptNet',
                    'ComparativeAnalyzer',
                    'SemanticPathfindingTestSuite'
                ],
                'realistic_behavior_verification': True,
                'end_to_end_functionality': True,
                'integration_success_rate_range': [75, 95]
            }
        }
    
    @staticmethod
    def get_comprehensive_test_configuration():
        """Get configuration for comprehensive comparative analysis testing."""
        return {
            'test_execution_settings': {
                'default_verbosity': 0,
                'default_timeout_seconds': 10.0,
                'default_max_tests': 50,
                'enable_conceptnet_api': True,
                'rate_limit_delay_seconds': 0.1,
                'save_results': False  # Default to False for testing
            },
            'comparison_settings': {
                'performance_threshold_seconds': 0.1,
                'semantic_quality_threshold': 0.6,
                'confidence_score_threshold': 2.0,
                'max_reasonable_path_length': 15,
                'min_reasonable_path_length': 2
            },
            'report_generation_settings': {
                'include_detailed_analysis': True,
                'include_performance_comparison': True,
                'include_semantic_quality_assessment': True,
                'max_detailed_failures_shown': 10,
                'max_comparison_results_shown': 100
            },
            'mock_behavior_settings': {
                'simulate_realistic_response_times': True,
                'simulate_network_variations': True,
                'simulate_api_failures': True,
                'failure_simulation_probability': 0.1
            }
        }