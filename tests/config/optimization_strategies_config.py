"""
Configuration class containing mock constants and test data for optimization strategies tests.
"""


class OptimizationStrategiesMockConfig:
    """Configuration class containing mock constants and test data for optimization strategies tests."""
    
    @staticmethod
    def get_basic_cache_configurations():
        """Get basic cache configurations for testing."""
        return {
            'small_cache': {
                'max_size': 10,
                'expected_evictions': True,
                'test_entries': 15
            },
            'medium_cache': {
                'max_size': 100,
                'expected_evictions': False,
                'test_entries': 50
            },
            'large_cache': {
                'max_size': 1000,
                'expected_evictions': False,
                'test_entries': 100
            }
        }
    
    @staticmethod
    def get_cache_test_data():
        """Get test data for cache functionality testing."""
        return {
            'simple_triples': [
                {'subject': 'cat', 'predicate': 'chase', 'object': 'mouse'},
                {'subject': 'dog', 'predicate': 'bark', 'object': 'stranger'},
                {'subject': 'bird', 'predicate': 'fly', 'object': 'sky'},
                {'subject': 'fish', 'predicate': 'swim', 'object': 'water'},
                {'subject': 'car', 'predicate': 'drive', 'object': 'road'}
            ],
            'complex_triples': [
                {'subject': 'scientist', 'predicate': 'study', 'object': 'organism'},
                {'subject': 'teacher', 'predicate': 'educate', 'object': 'student'},
                {'subject': 'artist', 'predicate': 'create', 'object': 'masterpiece'},
                {'subject': 'doctor', 'predicate': 'treat', 'object': 'patient'}
            ],
            'cached_results': [
                {
                    'key': 'cat||chase||mouse',
                    'result': (['cat.n.01', 'predator.n.01'], ['prey.n.01', 'mouse.n.01'], 'chase.v.01'),
                    'success': True,
                    'execution_time': 0.05
                },
                {
                    'key': 'dog||bark||stranger',
                    'result': (['dog.n.01', 'guardian.n.01'], ['threat.n.01', 'stranger.n.01'], 'bark.v.01'),
                    'success': True,
                    'execution_time': 0.03
                },
                {
                    'key': 'bird||fly||sky',
                    'result': (['bird.n.01', 'flyer.n.01'], ['atmosphere.n.01', 'sky.n.01'], 'fly.v.01'),
                    'success': True,
                    'execution_time': 0.04
                }
            ]
        }
    
    @staticmethod
    def get_graph_optimization_parameters():
        """Get parameters for graph optimization testing."""
        return {
            'small_graph_params': {
                'node_count': 50,
                'edge_count': 80,
                'verbosity': 0,
                'expected_shortcuts': 5,
                'expected_removals': 8
            },
            'medium_graph_params': {
                'node_count': 200,
                'edge_count': 350,
                'verbosity': 1,
                'expected_shortcuts': 25,
                'expected_removals': 35
            },
            'large_graph_params': {
                'node_count': 1000,
                'edge_count': 1800,
                'verbosity': 2,
                'expected_shortcuts': 100,
                'expected_removals': 180
            },
            'optimization_settings': {
                'max_shortcuts': 100,
                'centrality_sample_size': 1000,
                'distance_precompute_limit': 100,
                'path_cutoff': 6
            }
        }
    
    @staticmethod
    def get_mock_graph_structures():
        """Get mock graph structures for testing."""
        return {
            'linear_hierarchy': {
                'nodes': ['entity.n.01', 'physical_entity.n.01', 'object.n.01', 'living_thing.n.01', 'organism.n.01', 'animal.n.01', 'vertebrate.n.01', 'mammal.n.01', 'carnivore.n.01', 'feline.n.01', 'cat.n.01'],
                'edges': [
                    ('cat.n.01', 'feline.n.01'),
                    ('feline.n.01', 'carnivore.n.01'),
                    ('carnivore.n.01', 'mammal.n.01'),
                    ('mammal.n.01', 'vertebrate.n.01'),
                    ('vertebrate.n.01', 'animal.n.01'),
                    ('animal.n.01', 'organism.n.01'),
                    ('organism.n.01', 'living_thing.n.01'),
                    ('living_thing.n.01', 'object.n.01'),
                    ('object.n.01', 'physical_entity.n.01'),
                    ('physical_entity.n.01', 'entity.n.01')
                ],
                'properties': {
                    'is_dag': True,
                    'is_tree': True,
                    'max_depth': 10
                }
            },
            'branching_taxonomy': {
                'root': 'vehicle.n.01',
                'branches': {
                    'land_vehicle.n.01': ['car.n.01', 'truck.n.01', 'bicycle.n.01', 'motorcycle.n.01'],
                    'water_vehicle.n.01': ['boat.n.01', 'ship.n.01', 'submarine.n.01'],
                    'air_vehicle.n.01': ['airplane.n.01', 'helicopter.n.01', 'balloon.n.01']
                },
                'cross_connections': [
                    ('car.n.01', 'boat.n.01', 'amphibious_vehicle'),
                    ('airplane.n.01', 'boat.n.01', 'seaplane')
                ]
            },
            'dense_network': {
                'node_count': 20,
                'connection_probability': 0.3,
                'expected_edges': 60,
                'clustering_coefficient': 0.4
            }
        }
    
    @staticmethod
    def get_persistent_cache_configurations():
        """Get configurations for persistent cache testing."""
        return {
            'test_database_configs': {
                'memory_db': {
                    'db_path': ':memory:',
                    'is_temporary': True,
                    'cleanup_required': False
                },
                'file_db': {
                    'db_path': 'test_cache.db',
                    'is_temporary': True,
                    'cleanup_required': True
                },
                'persistent_db': {
                    'db_path': 'persistent_test_cache.db',
                    'is_temporary': False,
                    'cleanup_required': False
                }
            },
            'schema_definitions': {
                'pathfinding_cache_columns': [
                    'id', 'subject', 'predicate', 'object_term', 'success',
                    'subject_path', 'object_path', 'connecting_predicate',
                    'execution_time', 'created_at'
                ],
                'graph_cache_columns': [
                    'id', 'graph_hash', 'node_count', 'edge_count',
                    'graph_data', 'created_at'
                ]
            }
        }
    
    @staticmethod
    def get_optimization_benchmark_scenarios():
        """Get scenarios for optimization benchmark testing."""
        return {
            'quick_benchmark': {
                'test_case_count': 5,
                'runs_per_test': 2,
                'expected_duration_ms': 500,
                'test_cases': [
                    {'subject': 'cat', 'predicate': 'chase', 'object': 'mouse', 'description': 'cat chases mouse'},
                    {'subject': 'dog', 'predicate': 'bark', 'object': 'stranger', 'description': 'dog barks at stranger'},
                    {'subject': 'bird', 'predicate': 'fly', 'object': 'sky', 'description': 'bird flies in sky'},
                    {'subject': 'fish', 'predicate': 'swim', 'object': 'water', 'description': 'fish swims in water'},
                    {'subject': 'car', 'predicate': 'drive', 'object': 'road', 'description': 'car drives on road'}
                ]
            },
            'comprehensive_benchmark': {
                'test_case_count': 15,
                'runs_per_test': 3,
                'expected_duration_ms': 2000,
                'test_cases': [
                    {'subject': 'scientist', 'predicate': 'study', 'object': 'organism', 'description': 'scientist studies organism'},
                    {'subject': 'teacher', 'predicate': 'educate', 'object': 'student', 'description': 'teacher educates student'},
                    {'subject': 'artist', 'predicate': 'create', 'object': 'masterpiece', 'description': 'artist creates masterpiece'},
                    {'subject': 'doctor', 'predicate': 'treat', 'object': 'patient', 'description': 'doctor treats patient'},
                    {'subject': 'engineer', 'predicate': 'design', 'object': 'system', 'description': 'engineer designs system'},
                    {'subject': 'musician', 'predicate': 'compose', 'object': 'symphony', 'description': 'musician composes symphony'},
                    {'subject': 'writer', 'predicate': 'author', 'object': 'novel', 'description': 'writer authors novel'},
                    {'subject': 'chef', 'predicate': 'prepare', 'object': 'meal', 'description': 'chef prepares meal'},
                    {'subject': 'architect', 'predicate': 'plan', 'object': 'building', 'description': 'architect plans building'},
                    {'subject': 'gardener', 'predicate': 'cultivate', 'object': 'garden', 'description': 'gardener cultivates garden'},
                    {'subject': 'photographer', 'predicate': 'capture', 'object': 'moment', 'description': 'photographer captures moment'},
                    {'subject': 'dancer', 'predicate': 'perform', 'object': 'routine', 'description': 'dancer performs routine'},
                    {'subject': 'pilot', 'predicate': 'navigate', 'object': 'aircraft', 'description': 'pilot navigates aircraft'},
                    {'subject': 'sailor', 'predicate': 'steer', 'object': 'vessel', 'description': 'sailor steers vessel'},
                    {'subject': 'driver', 'predicate': 'operate', 'object': 'vehicle', 'description': 'driver operates vehicle'}
                ]
            },
            'performance_benchmark': {
                'test_case_count': 50,
                'runs_per_test': 5,
                'expected_duration_ms': 10000,
                'performance_metrics': {
                    'expected_base_avg_time': 0.1,
                    'expected_optimized_avg_time': 0.06,
                    'expected_min_speedup': 1.2,
                    'expected_max_speedup': 3.0,
                    'expected_cache_hit_rate': 70.0
                }
            }
        }
    
    @staticmethod
    def get_smied_integration_configurations():
        """Get configurations for SMIED integration testing."""
        return {
            'basic_smied_config': {
                'nlp_model': None,
                'auto_download': False,
                'verbosity': 0,
                'expected_components': ['synset_graph', 'analyze_triple', 'build_synset_graph']
            },
            'optimized_smied_config': {
                'enable_caching': True,
                'enable_graph_optimization': True,
                'verbosity': 1,
                'cache_max_size': 1000,
                'optimization_settings': {
                    'max_shortcuts': 100,
                    'precompute_distances': True,
                    'compute_importance': True
                }
            },
            'integration_test_scenarios': {
                'cache_integration': {
                    'test_triples': [
                        ('cat', 'chase', 'mouse'),
                        ('dog', 'bark', 'stranger'),
                        ('bird', 'fly', 'sky')
                    ],
                    'expected_cache_behavior': 'first_miss_then_hit',
                    'validate_cache_content': True
                },
                'graph_optimization_integration': {
                    'base_graph_properties': {
                        'min_nodes': 10,
                        'min_edges': 15
                    },
                    'optimization_expectations': {
                        'shortcuts_added': True,
                        'redundant_edges_removed': True,
                        'distances_precomputed': True
                    }
                }
            }
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get test data for validation tests."""
        return {
            'valid_cache_inputs': {
                'max_sizes': [1, 10, 100, 1000],
                'valid_keys': [
                    ('cat', 'chase', 'mouse'),
                    ('dog', 'bark', 'stranger'),
                    ('bird', 'fly', 'sky')
                ],
                'valid_results': [
                    (['cat.n.01'], ['mouse.n.01'], 'chase.v.01'),
                    (['dog.n.01'], ['stranger.n.01'], 'bark.v.01'),
                    (['bird.n.01'], ['sky.n.01'], 'fly.v.01')
                ]
            },
            'valid_graph_inputs': {
                'verbosity_levels': [0, 1, 2],
                'valid_graphs': [
                    {'nodes': ['a', 'b', 'c'], 'edges': [('a', 'b'), ('b', 'c')]},
                    {'nodes': ['x', 'y', 'z', 'w'], 'edges': [('x', 'y'), ('y', 'z'), ('z', 'w'), ('x', 'z')]},
                    {'nodes': list(range(10)), 'edges': [(i, i+1) for i in range(9)]}
                ]
            },
            'valid_database_inputs': {
                'db_paths': ['test1.db', 'test2.db', ':memory:'],
                'valid_queries': [
                    "SELECT * FROM pathfinding_cache LIMIT 5",
                    "SELECT COUNT(*) FROM graph_cache",
                    "SELECT success FROM pathfinding_cache WHERE success = 1"
                ]
            },
            'constraint_validations': {
                'cache_size_constraints': {
                    'min_size': 1,
                    'max_size': 10000,
                    'invalid_sizes': [-1, 0, -100]
                },
                'timing_constraints': {
                    'min_execution_time': 0.0,
                    'max_reasonable_time': 60.0,
                    'invalid_times': [-1.0, -0.001]
                },
                'graph_constraints': {
                    'min_nodes': 0,
                    'min_edges': 0,
                    'max_reasonable_nodes': 100000
                }
            }
        }
    
    @staticmethod
    def get_edge_case_test_data():
        """Get test data for edge case tests."""
        return {
            'cache_edge_cases': {
                'empty_cache_operations': {
                    'get_from_empty': ('nonexistent', 'predicate', 'object'),
                    'clear_empty': True,
                    'size_empty': 0
                },
                'cache_overflow': {
                    'max_size': 3,
                    'entries_to_add': [
                        ('a', 'b', 'c'),
                        ('d', 'e', 'f'),
                        ('g', 'h', 'i'),
                        ('j', 'k', 'l'),  # Should evict first entry
                        ('m', 'n', 'o')   # Should evict second entry
                    ],
                    'expected_evictions': 2
                },
                'malformed_inputs': {
                    'empty_strings': ('', '', ''),
                    'none_values': (None, None, None),
                    'mixed_invalid': ('valid', None, ''),
                    'unicode_strings': ('café', 'naïve', 'résumé'),
                    'very_long_strings': ('a' * 1000, 'b' * 1000, 'c' * 1000)
                }
            },
            'graph_edge_cases': {
                'empty_graph': {
                    'nodes': [],
                    'edges': [],
                    'expected_behavior': 'handle_gracefully'
                },
                'single_node_graph': {
                    'nodes': ['single'],
                    'edges': [],
                    'expected_shortcuts': 0
                },
                'disconnected_graph': {
                    'components': [
                        {'nodes': ['a', 'b'], 'edges': [('a', 'b')]},
                        {'nodes': ['c', 'd'], 'edges': [('c', 'd')]}
                    ],
                    'expected_behavior': 'optimize_separately'
                },
                'self_loop_graph': {
                    'nodes': ['a', 'b', 'c'],
                    'edges': [('a', 'a'), ('a', 'b'), ('b', 'c')],
                    'expected_self_loops_handled': True
                },
                'very_large_graph': {
                    'node_count': 10000,
                    'edge_count': 50000,
                    'expected_timeout_handling': True
                }
            },
            'database_edge_cases': {
                'database_errors': {
                    'nonexistent_path': '/nonexistent/path/cache.db',
                    'readonly_path': '/readonly/cache.db',
                    'invalid_permissions': True
                },
                'corrupted_data': {
                    'invalid_json_paths': ['[invalid', '{"incomplete": ', None],
                    'binary_data': b'\x00\x01\x02\x03',
                    'very_large_data': 'x' * (10 * 1024 * 1024)  # 10MB string
                },
                'concurrent_access': {
                    'multiple_writers': 5,
                    'multiple_readers': 10,
                    'expected_behavior': 'handle_locks'
                }
            },
            'optimization_edge_cases': {
                'extreme_parameters': {
                    'zero_cache_size': 0,
                    'negative_verbosity': -1,
                    'huge_beam_width': 10000,
                    'zero_max_depth': 0
                },
                'resource_exhaustion': {
                    'memory_limit_mb': 1,
                    'time_limit_seconds': 0.001,
                    'expected_graceful_degradation': True
                },
                'invalid_smied_instance': {
                    'none_smied': None,
                    'missing_methods': 'incomplete_mock',
                    'exception_throwing_smied': 'error_mock'
                }
            }
        }
    
    @staticmethod
    def get_integration_scenarios():
        """Get comprehensive integration test scenarios."""
        return {
            'end_to_end_optimization': {
                'scenario_1': {
                    'description': 'Complete optimization pipeline with caching and graph optimization',
                    'setup': {
                        'enable_all_optimizations': True,
                        'cache_size': 100,
                        'verbosity': 1
                    },
                    'test_sequence': [
                        'initialize_optimized_smied',
                        'build_optimized_graph',
                        'run_pathfinding_tests',
                        'validate_cache_performance',
                        'check_optimization_stats'
                    ],
                    'expected_outcomes': {
                        'cache_hit_rate_min': 50.0,
                        'graph_optimization_applied': True,
                        'performance_improvement': True
                    }
                },
                'scenario_2': {
                    'description': 'Optimization with persistent cache across sessions',
                    'setup': {
                        'persistent_cache_enabled': True,
                        'session_count': 3,
                        'tests_per_session': 10
                    },
                    'test_sequence': [
                        'session_1_populate_cache',
                        'session_2_use_cached_results',
                        'session_3_verify_persistence'
                    ],
                    'expected_outcomes': {
                        'cache_persistence_verified': True,
                        'cross_session_speedup': True
                    }
                }
            },
            'stress_testing_scenarios': {
                'high_volume_caching': {
                    'cache_operations': 10000,
                    'unique_triples': 1000,
                    'expected_memory_stability': True,
                    'expected_performance_degradation': 'minimal'
                },
                'large_graph_optimization': {
                    'graph_nodes': 5000,
                    'graph_edges': 10000,
                    'optimization_timeout_seconds': 30,
                    'expected_completion': True
                },
                'concurrent_optimization': {
                    'concurrent_threads': 5,
                    'operations_per_thread': 100,
                    'expected_thread_safety': True
                }
            },
            'real_world_simulation': {
                'typical_usage_pattern': {
                    'cache_warm_up_phase': {
                        'initial_requests': 100,
                        'cache_hit_rate_expected': 0.0
                    },
                    'steady_state_phase': {
                        'additional_requests': 500,
                        'cache_hit_rate_expected': 60.0,
                        'performance_improvement_expected': 2.0
                    },
                    'burst_phase': {
                        'burst_requests': 1000,
                        'time_window_seconds': 10,
                        'expected_stability': True
                    }
                }
            }
        }
    
    @staticmethod
    def get_mock_setup_configurations():
        """Get configurations for mock setup in different test scenarios."""
        return {
            'basic_setup': {
                'path_cache_mock': 'MockPathCache',
                'graph_optimizer_mock': 'MockGraphOptimizer',
                'persistent_cache_mock': 'MockPersistentCache',
                'optimized_smied_mock': 'MockOptimizedSMIED',
                'benchmark_mock': 'MockOptimizationBenchmark'
            },
            'validation_setup': {
                'path_cache_mock': 'MockPathCacheValidation',
                'graph_optimizer_mock': 'MockGraphOptimizerValidation',
                'persistent_cache_mock': 'MockPersistentCacheValidation',
                'optimized_smied_mock': 'MockOptimizedSMIEDValidation',
                'enable_validation': True
            },
            'edge_case_setup': {
                'path_cache_mock': 'MockPathCacheEdgeCases',
                'graph_optimizer_mock': 'MockGraphOptimizerEdgeCases',
                'persistent_cache_mock': 'MockPersistentCacheEdgeCases',
                'optimized_smied_mock': 'MockOptimizedSMIEDEdgeCases',
                'simulate_failures': True
            },
            'integration_setup': {
                'path_cache_mock': 'MockPathCacheIntegration',
                'graph_optimizer_mock': 'MockGraphOptimizerIntegration',
                'persistent_cache_mock': 'MockPersistentCacheIntegration',
                'optimized_smied_mock': 'MockOptimizedSMIEDIntegration',
                'smied_mock': 'MockSMIEDForOptimization',
                'enable_real_components': False
            }
        }
    
    @staticmethod
    def get_performance_expectations():
        """Get performance expectations for different optimization strategies."""
        return {
            'caching_performance': {
                'memory_cache': {
                    'hit_rate_after_warmup': 70.0,
                    'access_time_improvement': 10.0,  # 10x faster
                    'memory_overhead_mb': 10.0
                },
                'persistent_cache': {
                    'hit_rate_cross_session': 80.0,
                    'startup_time_penalty_ms': 100.0,
                    'disk_space_overhead_mb': 50.0
                }
            },
            'graph_optimization_performance': {
                'structure_optimization': {
                    'edge_reduction_percentage': 10.0,
                    'shortcut_addition_percentage': 5.0,
                    'pathfinding_speedup': 1.5
                },
                'distance_precomputation': {
                    'precompute_coverage_percentage': 10.0,
                    'distance_lookup_speedup': 100.0,
                    'memory_overhead_mb': 20.0
                }
            },
            'combined_optimization_performance': {
                'overall_speedup_range': (1.2, 3.0),
                'memory_overhead_mb_max': 100.0,
                'initialization_time_penalty_ms_max': 1000.0,
                'cache_hit_rate_min': 60.0
            }
        }
    
    @staticmethod
    def get_expected_test_outcomes():
        """Get expected outcomes for different test scenarios."""
        return {
            'successful_optimization': {
                'cache_initialization': True,
                'graph_optimization_applied': True,
                'performance_improvement_measured': True,
                'no_functionality_regression': True,
                'resource_usage_reasonable': True
            },
            'failed_optimization': {
                'graceful_fallback_to_base': True,
                'error_handling_proper': True,
                'no_data_corruption': True,
                'expected_exceptions': ['ValueError', 'MemoryError', 'TimeoutError', 'sqlite3.Error']
            },
            'partial_optimization_success': {
                'some_optimizations_applied': True,
                'degraded_performance_acceptable': True,
                'fallback_strategies_used': ['disable_caching', 'disable_graph_optimization', 'reduce_cache_size']
            },
            'benchmark_outcomes': {
                'comparative_analysis_generated': True,
                'performance_metrics_calculated': True,
                'speedup_measurements_valid': True,
                'statistical_significance_considered': True
            }
        }