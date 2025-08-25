"""
Configuration class containing mock constants and test data for PerformanceAnalysis tests.
"""


class PerformanceAnalysisMockConfig:
    """Configuration class containing mock constants and test data for PerformanceAnalysis tests."""
    
    @staticmethod
    def get_basic_performance_profiles():
        """Get basic performance profile data for testing."""
        return {
            'fast_operation': {
                'execution_time': 0.05,
                'memory_peak_mb': 5.0,
                'memory_current_mb': 3.0,
                'cpu_usage_percent': 25.0,
                'function_calls': 500,
                'graph_nodes': 50,
                'graph_edges': 100,
                'search_depth_reached': 3,
                'beam_expansions': 5,
                'error': None
            },
            'medium_operation': {
                'execution_time': 0.25,
                'memory_peak_mb': 15.0,
                'memory_current_mb': 10.0,
                'cpu_usage_percent': 55.0,
                'function_calls': 2500,
                'graph_nodes': 250,
                'graph_edges': 500,
                'search_depth_reached': 6,
                'beam_expansions': 15,
                'error': None
            },
            'slow_operation': {
                'execution_time': 1.0,
                'memory_peak_mb': 50.0,
                'memory_current_mb': 35.0,
                'cpu_usage_percent': 85.0,
                'function_calls': 10000,
                'graph_nodes': 1000,
                'graph_edges': 2500,
                'search_depth_reached': 10,
                'beam_expansions': 50,
                'error': None
            }
        }
    
    @staticmethod
    def get_scalability_test_scenarios():
        """Get scalability test scenarios for different graph sizes and parameters."""
        return {
            'graph_size_scenarios': [
                {
                    'graph_size': 1000,
                    'test_complexity': 'simple',
                    'average_time': 0.1,
                    'memory_usage_mb': 10.0,
                    'success_rate': 100.0,
                    'bottleneck_functions': [('func_a', 0.05), ('func_b', 0.03)]
                },
                {
                    'graph_size': 5000,
                    'test_complexity': 'simple',
                    'average_time': 0.3,
                    'memory_usage_mb': 35.0,
                    'success_rate': 95.0,
                    'bottleneck_functions': [('func_a', 0.18), ('func_b', 0.09)]
                },
                {
                    'graph_size': 10000,
                    'test_complexity': 'simple',
                    'average_time': 0.7,
                    'memory_usage_mb': 75.0,
                    'success_rate': 85.0,
                    'bottleneck_functions': [('func_a', 0.45), ('func_b', 0.15)]
                }
            ],
            'search_depth_scenarios': [
                {
                    'graph_size': 0,
                    'test_complexity': 'depth_3',
                    'average_time': 0.08,
                    'memory_usage_mb': 8.0,
                    'success_rate': 100.0,
                    'bottleneck_functions': [('search_func', 0.05)]
                },
                {
                    'graph_size': 0,
                    'test_complexity': 'depth_8',
                    'average_time': 0.35,
                    'memory_usage_mb': 25.0,
                    'success_rate': 90.0,
                    'bottleneck_functions': [('search_func', 0.25), ('path_func', 0.08)]
                },
                {
                    'graph_size': 0,
                    'test_complexity': 'depth_15',
                    'average_time': 0.95,
                    'memory_usage_mb': 65.0,
                    'success_rate': 70.0,
                    'bottleneck_functions': [('search_func', 0.70), ('path_func', 0.20)]
                }
            ],
            'beam_width_scenarios': [
                {
                    'graph_size': 0,
                    'test_complexity': 'beam_1',
                    'average_time': 0.05,
                    'memory_usage_mb': 5.0,
                    'success_rate': 100.0,
                    'bottleneck_functions': [('beam_func', 0.03)]
                },
                {
                    'graph_size': 0,
                    'test_complexity': 'beam_5',
                    'average_time': 0.25,
                    'memory_usage_mb': 20.0,
                    'success_rate': 95.0,
                    'bottleneck_functions': [('beam_func', 0.18), ('expand_func', 0.05)]
                },
                {
                    'graph_size': 0,
                    'test_complexity': 'beam_10',
                    'average_time': 0.55,
                    'memory_usage_mb': 45.0,
                    'success_rate': 85.0,
                    'bottleneck_functions': [('beam_func', 0.40), ('expand_func', 0.12)]
                }
            ]
        }
    
    @staticmethod
    def get_memory_profiling_data():
        """Get memory profiling test data."""
        return {
            'graph_construction_results': {
                'successful': {
                    'construction_time': 2.5,
                    'initial_memory_mb': 100.0,
                    'final_memory_mb': 175.0,
                    'peak_traced_mb': 200.0,
                    'memory_increase_mb': 75.0,
                    'nodes_count': 15000,
                    'edges_count': 35000,
                    'memory_per_node_kb': 5.0,
                    'memory_per_edge_kb': 2.14
                },
                'failed': {
                    'error': 'Graph construction failed due to memory limit'
                }
            },
            'pathfinding_memory_results': {
                'successful': {
                    'total_tests': 15,
                    'successful_tests': 12,
                    'failed_tests': 3,
                    'average_memory_delta_mb': 3.2,
                    'max_memory_delta_mb': 8.5,
                    'average_peak_memory_mb': 18.7,
                    'max_peak_memory_mb': 32.1,
                    'individual_profiles': [
                        {
                            'test_case': 'cat->chase->mouse',
                            'success': True,
                            'execution_time': 0.12,
                            'initial_memory_mb': 120.0,
                            'final_memory_mb': 123.2,
                            'peak_traced_mb': 28.5,
                            'memory_delta_mb': 3.2
                        },
                        {
                            'test_case': 'dog->bark->intruder',
                            'success': True,
                            'execution_time': 0.18,
                            'initial_memory_mb': 123.2,
                            'final_memory_mb': 127.1,
                            'peak_traced_mb': 32.1,
                            'memory_delta_mb': 3.9
                        }
                    ]
                },
                'failed': {
                    'error': 'Memory profiling failed'
                }
            }
        }
    
    @staticmethod
    def get_bottleneck_analysis_data():
        """Get bottleneck analysis test data."""
        return {
            'typical_bottlenecks': {
                'slowest_functions': [
                    ('pathfinding_algorithm', 0.45),
                    ('graph_traversal', 0.25),
                    ('similarity_calculation', 0.15),
                    ('beam_expansion', 0.10)
                ],
                'memory_hotspots': [
                    ('graph_storage', 45.0),
                    ('node_cache', 25.0),
                    ('path_tracking', 15.0),
                    ('temporary_objects', 10.0)
                ],
                'io_operations': [
                    ('wordnet_lookup', 0.08),
                    ('file_cache_read', 0.05),
                    ('log_writing', 0.02)
                ],
                'optimization_recommendations': [
                    'Consider reducing search depth or beam width for better performance',
                    'Implement graph caching to reduce memory allocation overhead',
                    'Consider streaming or lazy evaluation for large graphs',
                    'High function call count detected - consider algorithm optimization'
                ]
            },
            'minimal_bottlenecks': {
                'slowest_functions': [('main_algorithm', 0.15)],
                'memory_hotspots': [('core_data', 20.0)],
                'io_operations': [('config_read', 0.01)],
                'optimization_recommendations': ['Performance is acceptable']
            }
        }
    
    @staticmethod
    def get_comprehensive_analysis_results():
        """Get comprehensive analysis test results."""
        return {
            'complete_analysis': {
                'timestamp': 1234567890.0,
                'analysis_sections': {
                    'graph_construction_memory': {
                        'construction_time': 3.2,
                        'memory_increase_mb': 85.0,
                        'nodes_count': 18000,
                        'edges_count': 42000,
                        'memory_per_node_kb': 4.7
                    },
                    'depth_scalability': [
                        {
                            'complexity': 'depth_3',
                            'average_time': 0.09,
                            'memory_usage_mb': 12.0,
                            'success_rate': 100.0
                        },
                        {
                            'complexity': 'depth_8',
                            'average_time': 0.42,
                            'memory_usage_mb': 35.0,
                            'success_rate': 85.0
                        },
                        {
                            'complexity': 'depth_12',
                            'average_time': 0.88,
                            'memory_usage_mb': 68.0,
                            'success_rate': 75.0
                        }
                    ],
                    'beam_scalability': [
                        {
                            'complexity': 'beam_1',
                            'average_time': 0.06,
                            'memory_usage_mb': 8.0,
                            'success_rate': 100.0
                        },
                        {
                            'complexity': 'beam_5',
                            'average_time': 0.28,
                            'memory_usage_mb': 25.0,
                            'success_rate': 92.0
                        },
                        {
                            'complexity': 'beam_10',
                            'average_time': 0.65,
                            'memory_usage_mb': 55.0,
                            'success_rate': 88.0
                        }
                    ],
                    'pathfinding_memory': {
                        'total_tests': 20,
                        'successful_tests': 17,
                        'average_memory_delta_mb': 4.1,
                        'max_peak_memory_mb': 38.5
                    }
                }
            },
            'partial_analysis_with_errors': {
                'timestamp': 1234567890.0,
                'analysis_sections': {
                    'graph_construction_memory': {'error': 'Construction failed'},
                    'scalability_error': 'Scalability testing failed',
                    'pathfinding_memory': {'error': 'Memory profiling failed'}
                }
            }
        }
    
    @staticmethod
    def get_test_operation_data():
        """Get test operation data for profiling."""
        return {
            'simple_operations': [
                {
                    'name': 'fast_op',
                    'function': lambda: 42,
                    'expected_time_range': (0.0, 0.01),
                    'timeout': 5.0
                }
            ],
            'complex_operations': [
                {
                    'name': 'pathfinding_op',
                    'description': 'Semantic pathfinding operation',
                    'expected_time_range': (0.1, 2.0),
                    'timeout': 30.0
                }
            ],
            'timeout_operations': [
                {
                    'name': 'slow_op',
                    'description': 'Operation that exceeds timeout',
                    'timeout': 0.1,
                    'expected_error': 'Operation timed out'
                }
            ]
        }
    
    @staticmethod
    def get_test_case_data():
        """Get test case data for semantic pathfinding."""
        return {
            'simple_test_cases': [
                {
                    'subject': 'cat',
                    'predicate': 'chase',
                    'object': 'mouse',
                    'expected': True,
                    'description': 'Simple predator-prey',
                    'category': 'predator_prey',
                    'difficulty_level': 'easy'
                },
                {
                    'subject': 'dog',
                    'predicate': 'bark',
                    'object': 'intruder',
                    'expected': True,
                    'description': 'Simple protective action',
                    'category': 'protective_action',
                    'difficulty_level': 'easy'
                },
                {
                    'subject': 'bird',
                    'predicate': 'build',
                    'object': 'nest',
                    'expected': True,
                    'description': 'Simple natural behavior',
                    'category': 'natural_behavior',
                    'difficulty_level': 'easy'
                }
            ],
            'medium_test_cases': [
                {
                    'subject': 'student',
                    'predicate': 'study',
                    'object': 'mathematics',
                    'expected': True,
                    'description': 'Educational activity',
                    'category': 'education',
                    'difficulty_level': 'medium'
                },
                {
                    'subject': 'chef',
                    'predicate': 'cook',
                    'object': 'meal',
                    'expected': True,
                    'description': 'Professional activity',
                    'category': 'profession',
                    'difficulty_level': 'medium'
                }
            ],
            'complex_test_cases': [
                {
                    'subject': 'scientist',
                    'predicate': 'research',
                    'object': 'hypothesis',
                    'expected': True,
                    'description': 'Complex intellectual activity',
                    'category': 'research',
                    'difficulty_level': 'hard'
                }
            ]
        }
    
    @staticmethod
    def get_profiler_configuration():
        """Get profiler configuration options."""
        return {
            'basic_profiler': {
                'verbosity': 0,
                'timeout': 30.0,
                'enable_memory_tracking': True,
                'enable_cpu_tracking': True
            },
            'detailed_profiler': {
                'verbosity': 2,
                'timeout': 60.0,
                'enable_memory_tracking': True,
                'enable_cpu_tracking': True,
                'enable_call_tracking': True
            },
            'performance_profiler': {
                'verbosity': 1,
                'timeout': 120.0,
                'enable_memory_tracking': True,
                'enable_cpu_tracking': False,
                'batch_size': 10
            }
        }
    
    @staticmethod
    def get_error_handling_scenarios():
        """Get error handling scenarios for testing."""
        return {
            'timeout_scenarios': {
                'short_timeout': {
                    'timeout': 0.01,
                    'expected_error': 'Operation timed out after 0.01s'
                },
                'medium_timeout': {
                    'timeout': 1.0,
                    'expected_error': 'Operation timed out after 1.0s'
                }
            },
            'exception_scenarios': {
                'generic_exception': {
                    'exception': Exception('Test exception'),
                    'expected_error': 'Test exception'
                },
                'memory_error': {
                    'exception': MemoryError('Out of memory'),
                    'expected_error': 'Out of memory'
                },
                'value_error': {
                    'exception': ValueError('Invalid value'),
                    'expected_error': 'Invalid value'
                }
            },
            'resource_scenarios': {
                'no_graph': {
                    'graph': None,
                    'expected_behavior': 'graceful_failure'
                },
                'empty_graph': {
                    'nodes': 0,
                    'edges': 0,
                    'expected_behavior': 'minimal_performance'
                }
            }
        }
    
    @staticmethod
    def get_integration_test_parameters():
        """Get parameters for integration testing."""
        return {
            'realistic_scenario_1': {
                'smied_config': {
                    'nlp_model': None,
                    'auto_download': False,
                    'verbosity': 0
                },
                'test_parameters': {
                    'max_nodes': 5000,
                    'beam_widths': [3, 5, 7],
                    'max_depths': [5, 8, 10],
                    'timeout_per_operation': 30.0
                },
                'expected_results': {
                    'success_rate_threshold': 80.0,
                    'max_average_time': 2.0,
                    'max_memory_usage': 100.0
                }
            },
            'stress_test_scenario': {
                'smied_config': {
                    'nlp_model': None,
                    'auto_download': False,
                    'verbosity': 0
                },
                'test_parameters': {
                    'max_nodes': 20000,
                    'beam_widths': [10, 15, 20],
                    'max_depths': [12, 15, 20],
                    'timeout_per_operation': 120.0
                },
                'expected_results': {
                    'success_rate_threshold': 60.0,
                    'max_average_time': 10.0,
                    'max_memory_usage': 500.0
                }
            }
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get test data for validation tests."""
        return {
            'valid_configurations': {
                'basic_valid': {
                    'verbosity': 1,
                    'timeout': 30.0,
                    'profiling_enabled': True
                },
                'minimal_valid': {
                    'verbosity': 0,
                    'timeout': 1.0,
                    'profiling_enabled': False
                }
            },
            'invalid_configurations': {
                'negative_verbosity': {
                    'verbosity': -1,
                    'expected_error': 'Invalid verbosity level'
                },
                'zero_timeout': {
                    'timeout': 0,
                    'expected_error': 'Timeout must be positive'
                },
                'negative_timeout': {
                    'timeout': -5.0,
                    'expected_error': 'Timeout must be positive'
                }
            },
            'boundary_conditions': {
                'max_verbosity': {
                    'verbosity': 10,
                    'expected_behavior': 'very_verbose'
                },
                'very_long_timeout': {
                    'timeout': 3600.0,
                    'expected_behavior': 'patient_waiting'
                }
            }
        }
    
    @staticmethod
    def get_edge_case_test_data():
        """Get test data for edge case testing."""
        return {
            'empty_data_scenarios': {
                'no_operations': {
                    'operations': [],
                    'expected_result': 'empty_list'
                },
                'no_profiles': {
                    'profiles': [],
                    'expected_analysis': 'empty_analysis'
                }
            },
            'extreme_values': {
                'very_large_graph': {
                    'nodes': 1000000,
                    'edges': 5000000,
                    'expected_behavior': 'resource_intensive'
                },
                'very_small_graph': {
                    'nodes': 1,
                    'edges': 0,
                    'expected_behavior': 'minimal_operation'
                }
            },
            'malformed_data': {
                'invalid_profiles': {
                    'profile_data': {
                        'execution_time': 'invalid',
                        'memory_peak_mb': None,
                        'error': 'Invalid profile data'
                    }
                },
                'corrupted_results': {
                    'analysis_data': None,
                    'expected_error': 'Data corruption detected'
                }
            }
        }
    
    @staticmethod
    def get_mock_setup_configurations():
        """Get configurations for mock setup in different test scenarios."""
        return {
            'basic_setup': {
                'profiler_mock': 'MockPerformanceProfiler',
                'scalability_mock': 'MockScalabilityTester',
                'memory_mock': 'MockMemoryProfiler',
                'smied_mock': 'MockSMIED',
                'enable_realistic_timing': False
            },
            'validation_setup': {
                'profiler_mock': 'MockPerformanceProfiler',
                'validation_mock': 'MockPerformanceProfilerValidation',
                'enable_validation': True,
                'strict_mode': True
            },
            'edge_case_setup': {
                'profiler_mock': 'MockPerformanceProfilerEdgeCases',
                'scalability_mock': 'MockScalabilityTesterEdgeCases',
                'memory_mock': 'MockMemoryProfilerEdgeCases',
                'simulate_failures': True,
                'enable_timeouts': True,
                'enable_exceptions': True
            },
            'integration_setup': {
                'profiler_mock': 'MockPerformanceProfilerIntegration',
                'scalability_mock': 'MockScalabilityTesterIntegration',
                'memory_mock': 'MockMemoryProfilerIntegration',
                'smied_mock': 'MockSMIEDIntegration',
                'use_realistic_data': True,
                'enable_comprehensive_analysis': True
            }
        }
    
    @staticmethod
    def get_expected_test_outcomes():
        """Get expected outcomes for different test scenarios."""
        return {
            'successful_profiling': {
                'profile_created': True,
                'execution_time_valid': True,
                'memory_data_present': True,
                'no_errors': True,
                'performance_acceptable': True
            },
            'failed_profiling': {
                'profile_created': True,
                'has_error_message': True,
                'error_type_expected': ['timeout', 'exception', 'resource_limit'],
                'graceful_failure': True
            },
            'scalability_analysis': {
                'results_generated': True,
                'trend_identified': True,
                'bottlenecks_found': True,
                'recommendations_provided': True
            },
            'comprehensive_analysis': {
                'all_sections_present': True,
                'summary_generated': True,
                'actionable_insights': True,
                'performance_trends': True
            }
        }