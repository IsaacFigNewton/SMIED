"""
Configuration class containing mock constants and test data for optimization demo tests.

This module follows the SMIED Testing Framework Design Specifications
for configuration-driven test data management with static methods.
"""


class OptimizationDemoMockConfig:
    """Configuration class containing mock constants and test data for optimization demo tests."""
    
    @staticmethod
    def get_basic_test_data():
        """Get basic optimization demonstration test data."""
        return {
            'simple_cases': [
                {
                    'subject': 'fox',
                    'predicate': 'jump',
                    'object': 'dog',
                    'expected_success': True,
                    'description': 'Classic challenging semantic decomposition case',
                    'difficulty_level': 'hard',
                    'cross_pos': True,
                    'requires_optimization': True
                },
                {
                    'subject': 'cat',
                    'predicate': 'chase',
                    'object': 'mouse',
                    'expected_success': True,
                    'description': 'Simple predator-prey relationship',
                    'difficulty_level': 'easy',
                    'cross_pos': True,
                    'requires_optimization': False
                },
                {
                    'subject': 'bird',
                    'predicate': 'fly',
                    'object': 'sky',
                    'expected_success': True,
                    'description': 'Natural action-location relationship',
                    'difficulty_level': 'medium',
                    'cross_pos': True,
                    'requires_optimization': False
                }
            ],
            'challenging_cases': [
                {
                    'subject': 'person',
                    'predicate': 'think',
                    'object': 'idea',
                    'expected_success': False,
                    'description': 'Abstract cognitive relationship',
                    'difficulty_level': 'very_hard',
                    'cross_pos': True,
                    'requires_optimization': True
                },
                {
                    'subject': 'scientist',
                    'predicate': 'discover',
                    'object': 'truth',
                    'expected_success': False,
                    'description': 'Complex epistemic relationship',
                    'difficulty_level': 'very_hard',
                    'cross_pos': True,
                    'requires_optimization': True
                }
            ]
        }
    
    @staticmethod
    def get_optimization_parameters():
        """Get optimization parameter configurations."""
        return {
            'original_parameters': {
                'max_depth': 6,
                'beam_width': 3,
                'len_tolerance': 0,
                'relax_beam': False,
                'heuristic': 'embedding'
            },
            'optimized_parameters': {
                'max_depth': 10,
                'beam_width': 10,
                'len_tolerance': 3,
                'relax_beam': True,
                'heuristic': 'hybrid'
            },
            'parameter_improvements': {
                'beam_width_increase_percent': 233,
                'max_depth_increase_percent': 67,
                'expected_success_rate_improvement': 25
            }
        }
    
    @staticmethod
    def get_optimization_summary_data():
        """Get data for optimization summary demonstrations."""
        return {
            'parameter_optimization': [
                "beam_width: 3 -> 10 (233% increase in search space)",
                "max_depth: 6 -> 10 (67% deeper exploration)", 
                "relax_beam: False -> True (removes embedding constraints)",
                "len_tolerance: 0 -> 3 (accepts longer valid paths)"
            ],
            'heuristic_enhancement': [
                "Hybrid heuristic: 70% embedding + 30% WordNet distance",
                "WordNet distance estimator for fast heuristic calculation",
                "Alternative heuristics: uniform, pure WordNet, pure embedding",
                "Cross-POS penalty for noun-verb connections"
            ],
            'missing_relations_integration': [
                "Added WordNet attributes() relations (noun-adjective)",
                "Integrated lemma-level antonym relations",
                "Enhanced semantic connectivity through new relation types"
            ],
            'analysis_validation': [
                "Parameter sensitivity analysis framework",
                "Performance benchmarking system",
                "Comprehensive validation suite",
                "Parameter tuning guidelines document"
            ]
        }
    
    @staticmethod
    def get_performance_metrics():
        """Get expected performance metrics for optimization validation."""
        return {
            'timing_expectations': {
                'max_execution_time_seconds': 30.0,
                'typical_execution_time_seconds': 5.0,
                'timeout_threshold_seconds': 60.0
            },
            'success_rate_expectations': {
                'original_success_rate_percent': 60,
                'optimized_success_rate_percent': 85,
                'minimum_improvement_percent': 20
            },
            'path_quality_metrics': {
                'max_acceptable_path_length': 15,
                'typical_path_length': 8,
                'quality_threshold': 0.7
            }
        }
    
    @staticmethod
    def get_edge_case_scenarios():
        """Get edge case scenarios for optimization testing."""
        return {
            'empty_inputs': [
                {'subject': '', 'predicate': 'test', 'object': 'test'},
                {'subject': 'test', 'predicate': '', 'object': 'test'},
                {'subject': 'test', 'predicate': 'test', 'object': ''}
            ],
            'invalid_parameters': [
                {'max_depth': -1, 'beam_width': 5},
                {'max_depth': 5, 'beam_width': 0},
                {'max_depth': 0, 'beam_width': -1}
            ],
            'extreme_parameters': [
                {'max_depth': 100, 'beam_width': 100},
                {'max_depth': 1, 'beam_width': 1},
                {'len_tolerance': 50}
            ],
            'missing_components': [
                'wordnet_not_available',
                'spacy_model_not_loaded',
                'embedding_model_missing'
            ]
        }
    
    @staticmethod
    def get_integration_test_data():
        """Get integration test data for optimization demo testing."""
        return {
            'multi_component_scenarios': [
                {
                    'name': 'full_pipeline_test',
                    'components': ['SMIED', 'SemanticDecomposer', 'PairwiseBidirectionalAStar'],
                    'test_cases': [
                        {'subject': 'student', 'predicate': 'learn', 'object': 'knowledge'},
                        {'subject': 'teacher', 'predicate': 'educate', 'object': 'student'}
                    ],
                    'expected_interactions': 5
                },
                {
                    'name': 'parameter_propagation_test',
                    'components': ['SMIED'],
                    'parameter_sets': ['original', 'optimized'],
                    'validation_points': ['initialization', 'execution', 'results']
                }
            ],
            'system_integration': {
                'external_dependencies': ['nltk', 'spacy', 'numpy'],
                'optional_dependencies': ['torch', 'transformers'],
                'configuration_files': ['wordnet_config', 'embedding_config']
            }
        }
    
    @staticmethod
    def get_mock_setup_configurations():
        """Get mock setup configurations for different test scenarios."""
        return {
            'basic_setup': {
                'verbosity': 1,
                'build_graph_on_init': False,
                'mock_successful_analysis': True,
                'mock_execution_time': 2.5
            },
            'validation_setup': {
                'verbosity': 0,
                'strict_validation': True,
                'mock_parameter_validation': True,
                'mock_edge_case_handling': True
            },
            'edge_case_setup': {
                'verbosity': 2,
                'mock_failures': True,
                'mock_exceptions': True,
                'mock_timeout_scenarios': True
            },
            'integration_setup': {
                'verbosity': 1,
                'full_component_mocking': True,
                'cross_component_validation': True,
                'mock_external_dependencies': True
            }
        }
    
    @staticmethod
    def get_test_execution_settings():
        """Get test execution settings and constants."""
        return {
            'timeout_settings': {
                'default_test_timeout': 30,
                'long_running_test_timeout': 120,
                'integration_test_timeout': 180
            },
            'verbosity_levels': {
                'silent': 0,
                'normal': 1,
                'verbose': 2,
                'debug': 3
            },
            'assertion_tolerances': {
                'timing_tolerance_percent': 20,
                'success_rate_tolerance_percent': 5,
                'path_length_tolerance': 2
            }
        }
    
    @staticmethod
    def get_comprehensive_test_configuration():
        """Get comprehensive configuration combining all test settings."""
        return {
            'test_data': OptimizationDemoMockConfig.get_basic_test_data(),
            'optimization_params': OptimizationDemoMockConfig.get_optimization_parameters(),
            'performance_metrics': OptimizationDemoMockConfig.get_performance_metrics(),
            'edge_cases': OptimizationDemoMockConfig.get_edge_case_scenarios(),
            'integration_data': OptimizationDemoMockConfig.get_integration_test_data(),
            'mock_configs': OptimizationDemoMockConfig.get_mock_setup_configurations(),
            'execution_settings': OptimizationDemoMockConfig.get_test_execution_settings()
        }