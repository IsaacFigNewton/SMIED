"""
Configuration class containing mock constants and test data for PairwiseBidirectionalAStar tests.
"""


class PairwiseBidirectionalAStarMockConfig:
    """Configuration class containing mock constants and test data for PairwiseBidirectionalAStar tests.
    
    This class follows the SMIED Testing Framework Design Specifications by providing
    centralized test data management with static methods for different test scenarios.
    """
    
    @staticmethod
    def get_basic_graph_structures():
        """Get basic graph structures for pathfinding tests."""
        return {
            'linear_graph': {
                'nodes': ["start", "middle1", "middle2", "end"],
                'edges': [
                    ("start", "middle1", {"weight": 1.0}),
                    ("middle1", "middle2", {"weight": 1.0}),
                    ("middle2", "end", {"weight": 1.0})
                ]
            },
            'branching_graph': {
                'nodes': ["start", "branch1", "branch2", "end"],
                'edges': [
                    ("start", "branch1", {"weight": 1.0}),
                    ("start", "branch2", {"weight": 2.0}),
                    ("branch1", "end", {"weight": 1.0}),
                    ("branch2", "end", {"weight": 1.0})
                ]
            },
            'diamond_graph': {
                'nodes': ["A", "B", "C", "D"],
                'edges': [
                    ("A", "B", {"weight": 1.0}),
                    ("A", "C", {"weight": 2.0}),
                    ("B", "D", {"weight": 2.0}),
                    ("C", "D", {"weight": 1.0})
                ]
            }
        }
    
    @staticmethod
    def get_algorithm_parameters():
        """Get algorithm parameters for testing."""
        return {
            'default_params': {
                'beam_width': 3,
                'max_depth': 6,
                'similarity_threshold': 0.5,
                'use_gloss_bonus': True
            },
            'performance_params': {
                'beam_width': 5,
                'max_depth': 10,
                'similarity_threshold': 0.7,
                'use_gloss_bonus': False
            },
            'comprehensive_params': {
                'beam_width': 10,
                'max_depth': 15,
                'similarity_threshold': 0.3,
                'use_gloss_bonus': True
            },
            'minimal_params': {
                'beam_width': 1,
                'max_depth': 3,
                'similarity_threshold': 0.9,
                'use_gloss_bonus': False
            }
        }
    
    @staticmethod
    def get_complex_graph_scenarios():
        """Get complex graph scenarios for advanced testing."""
        return {
            'grid_graph': {
                'dimensions': (5, 5),
                'nodes': [(i, j) for i in range(5) for j in range(5)],
                'edge_pattern': 'four_connected',  # Each node connects to 4 neighbors
                'start': (0, 0),
                'goal': (4, 4),
                'expected_path_length': 8
            },
            'circular_graph': {
                'nodes': [f"node_{i}" for i in range(10)],
                'edges': [(f"node_{i}", f"node_{(i+1)%10}", {"weight": 1.0}) for i in range(10)],
                'shortcuts': [("node_0", "node_5", {"weight": 1.5})],
                'start': "node_0",
                'goal': "node_7"
            },
            'weighted_graph': {
                'nodes': ["A", "B", "C", "D", "E", "F"],
                'edges': [
                    ("A", "B", {"weight": 4.0}),
                    ("A", "C", {"weight": 2.0}),
                    ("B", "E", {"weight": 3.0}),
                    ("C", "D", {"weight": 1.0}),
                    ("C", "F", {"weight": 5.0}),
                    ("D", "E", {"weight": 1.0}),
                    ("E", "F", {"weight": 2.0})
                ],
                'optimal_path': ["A", "C", "D", "E", "F"]
            }
        }
    
    @staticmethod
    def get_gloss_bonus_test_data():
        """Get test data for GLOSS_BONUS constant testing."""
        return {
            'gloss_bonus_scenarios': [
                {
                    'node': 'cat.n.01',
                    'gloss_score': 0.8,
                    'bonus_multiplier': 1.2,
                    'expected_final_score': 0.96
                },
                {
                    'node': 'run.v.01',
                    'gloss_score': 0.6,
                    'bonus_multiplier': 1.1,
                    'expected_final_score': 0.66
                },
                {
                    'node': 'fast.a.01',
                    'gloss_score': 0.9,
                    'bonus_multiplier': 1.15,
                    'expected_final_score': 1.035
                }
            ],
            'bonus_threshold_tests': [
                {
                    'description': 'High gloss relevance',
                    'gloss_relevance': 0.9,
                    'should_apply_bonus': True
                },
                {
                    'description': 'Medium gloss relevance',
                    'gloss_relevance': 0.5,
                    'should_apply_bonus': True
                },
                {
                    'description': 'Low gloss relevance',
                    'gloss_relevance': 0.1,
                    'should_apply_bonus': False
                }
            ]
        }
    
    @staticmethod
    def get_wordnet_taxonomy_structures():
        """Get realistic WordNet taxonomy structures."""
        return {
            'animal_taxonomy': {
                'root': 'entity.n.01',
                'hierarchy': {
                    'entity.n.01': {
                        'physical_entity.n.01': {
                            'object.n.01': {
                                'living_thing.n.01': {
                                    'organism.n.01': {
                                        'animal.n.01': {
                                            'chordate.n.01': {
                                                'vertebrate.n.01': {
                                                    'mammal.n.01': {
                                                        'carnivore.n.01': {
                                                            'feline.n.01': {
                                                                'cat.n.01': {}
                                                            }
                                                        },
                                                        'canine.n.01': {
                                                            'dog.n.01': {}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                'test_paths': [
                    {
                        'source': 'cat.n.01',
                        'target': 'animal.n.01',
                        'expected_hops': 6
                    },
                    {
                        'source': 'cat.n.01',
                        'target': 'dog.n.01',
                        'expected_hops': 4  # Through mammal.n.01
                    }
                ]
            },
            'verb_taxonomy': {
                'root': 'verb.v.01',
                'hierarchy': {
                    'verb.v.01': {
                        'move.v.01': {
                            'locomote.v.01': {
                                'walk.v.01': {},
                                'run.v.01': {
                                    'sprint.v.01': {},
                                    'jog.v.01': {}
                                },
                                'swim.v.01': {}
                            }
                        },
                        'communicate.v.01': {
                            'speak.v.01': {
                                'whisper.v.01': {},
                                'shout.v.01': {}
                            }
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def get_pathfinding_edge_cases():
        """Get edge cases for pathfinding testing."""
        return {
            'no_path_cases': [
                {
                    'description': 'Disconnected components',
                    'nodes': ["A", "B", "C", "D"],
                    'edges': [("A", "B", {}), ("C", "D", {})],
                    'start': "A",
                    'goal': "C",
                    'expected_result': None
                },
                {
                    'description': 'Isolated target',
                    'nodes': ["start", "middle", "isolated"],
                    'edges': [("start", "middle", {})],
                    'start': "start",
                    'goal': "isolated",
                    'expected_result': None
                }
            ],
            'trivial_cases': [
                {
                    'description': 'Same start and goal',
                    'start': "node_1",
                    'goal': "node_1",
                    'expected_path': ["node_1"]
                },
                {
                    'description': 'Direct connection',
                    'nodes': ["A", "B"],
                    'edges': [("A", "B", {})],
                    'start': "A",
                    'goal': "B",
                    'expected_path': ["A", "B"]
                }
            ],
            'maximum_depth_cases': [
                {
                    'description': 'Path longer than max_depth',
                    'max_depth': 3,
                    'path_length': 5,
                    'expected_result': 'timeout_or_failure'
                }
            ]
        }
    
    @staticmethod
    def get_bidirectional_search_scenarios():
        """Get bidirectional search specific scenarios."""
        return {
            'meeting_in_middle': {
                'description': 'Forward and backward search meet',
                'nodes': ["A", "B", "C", "D", "E"],
                'edges': [("A", "B", {}), ("B", "C", {}), ("C", "D", {}), ("D", "E", {})],
                'start': "A",
                'goal': "E",
                'expected_meeting_point': "C",
                'expected_forward_steps': 2,
                'expected_backward_steps': 2
            },
            'asymmetric_meeting': {
                'description': 'Meeting point closer to one end',
                'graph_structure': 'asymmetric_tree',
                'expected_behavior': 'adaptive_search_expansion'
            },
            'early_termination': {
                'description': 'Search terminates early when paths meet',
                'termination_condition': 'path_intersection',
                'expected_efficiency_gain': 0.4  # 40% reduction in nodes explored
            }
        }
    
    @staticmethod
    def get_heuristic_function_tests():
        """Get heuristic function test data."""
        return {
            'distance_heuristics': [
                {
                    'name': 'wordnet_path_similarity',
                    'node_pairs': [
                        ('cat.n.01', 'dog.n.01', 0.8),
                        ('cat.n.01', 'vehicle.n.01', 0.1),
                        ('run.v.01', 'walk.v.01', 0.7)
                    ]
                },
                {
                    'name': 'embedding_cosine_similarity',
                    'requires_embeddings': True,
                    'similarity_threshold': 0.5
                },
                {
                    'name': 'gloss_semantic_similarity',
                    'requires_gloss_parsing': True,
                    'weight_factor': 1.2
                }
            ],
            'heuristic_admissibility': {
                'description': 'Test that heuristics never overestimate',
                'test_cases': [
                    {
                        'start': 'cat.n.01',
                        'goal': 'animal.n.01',
                        'true_distance': 6,
                        'heuristic_estimate': 5.8,  # Should be <= true_distance
                        'is_admissible': True
                    }
                ]
            }
        }
    
    @staticmethod
    def get_performance_benchmarks():
        """Get performance benchmarks for algorithm testing."""
        return {
            'small_graph': {
                'nodes': 100,
                'edges': 200,
                'max_search_time_ms': 100,
                'max_memory_mb': 10
            },
            'medium_graph': {
                'nodes': 1000,
                'edges': 2000,
                'max_search_time_ms': 1000,
                'max_memory_mb': 50
            },
            'large_graph': {
                'nodes': 10000,
                'edges': 20000,
                'max_search_time_ms': 5000,
                'max_memory_mb': 200
            },
            'wordnet_scale': {
                'nodes': 117000,  # Approximate WordNet size
                'edges': 200000,
                'max_search_time_ms': 30000,
                'max_memory_mb': 1000
            }
        }
    
    @staticmethod
    def get_beam_search_configurations():
        """Get beam search specific configurations."""
        return {
            'narrow_beam': {
                'beam_width': 1,
                'description': 'Greedy search behavior',
                'expected_characteristics': ['fast', 'possibly_suboptimal']
            },
            'medium_beam': {
                'beam_width': 5,
                'description': 'Balanced search',
                'expected_characteristics': ['moderate_speed', 'good_quality']
            },
            'wide_beam': {
                'beam_width': 20,
                'description': 'Comprehensive search',
                'expected_characteristics': ['thorough', 'memory_intensive']
            },
            'adaptive_beam': {
                'initial_width': 3,
                'expansion_factor': 1.5,
                'max_width': 15,
                'description': 'Dynamically adjusting beam'
            }
        }
    
    @staticmethod
    def get_heuristic_type_configurations():
        """Get heuristic type configurations from implementation."""
        return {
            'uniform': {
                'description': 'Uniform heuristic (constant value)',
                'expected_value': 1.0,
                'use_case': 'baseline_testing'
            },
            'wordnet': {
                'description': 'WordNet path-based heuristic',
                'dependencies': ['nltk', 'wordnet'],
                'expected_range': (1.0, 10.0),
                'use_case': 'taxonomic_similarity'
            },
            'embedding': {
                'description': 'Embedding-based similarity heuristic',
                'dependencies': ['embedding_helper'],
                'expected_range': (0.0, 1.0),
                'use_case': 'semantic_similarity'
            },
            'hybrid': {
                'description': 'Combined WordNet and embedding heuristic',
                'dependencies': ['embedding_helper', 'nltk', 'wordnet'],
                'weight_distribution': {'embedding': 0.7, 'wordnet': 0.3},
                'cross_pos_penalty': 0.2,
                'use_case': 'comprehensive_similarity'
            }
        }
    
    @staticmethod
    def get_optimization_parameters():
        """Get optimization parameters from implementation updates."""
        return {
            'original_defaults': {
                'beam_width': 3,
                'max_depth': 6,
                'relax_beam': False
            },
            'optimized_defaults': {
                'beam_width': 10,
                'max_depth': 10,
                'relax_beam': True
            },
            'optimization_rationale': {
                'beam_width_increase': 'Better coverage of search space',
                'max_depth_increase': 'Handle deeper taxonomic structures',
                'relax_beam_enabled': 'Allow exploration beyond initial beam'
            }
        }
    
    @staticmethod
    def get_wordnet_distance_test_cases():
        """Get test cases for WordNet distance estimation."""
        return {
            'same_synset_pairs': [
                {
                    'synset1': 'cat.n.01',
                    'synset2': 'cat.n.01',
                    'expected_distance': 0.0,
                    'description': 'Identical synsets should have zero distance'
                }
            ],
            'related_animal_pairs': [
                {
                    'synset1': 'cat.n.01',
                    'synset2': 'dog.n.01',
                    'expected_min_distance': 2.0,
                    'expected_max_distance': 8.0,
                    'description': 'Related animals through mammal hierarchy'
                }
            ],
            'cross_pos_pairs': [
                {
                    'synset1': 'cat.n.01',
                    'synset2': 'run.v.01',
                    'expected_distance': 8.0,
                    'description': 'Cross-POS should have higher penalty'
                }
            ],
            'unrelated_pairs': [
                {
                    'synset1': 'computer.n.01',
                    'synset2': 'tree.n.01',
                    'expected_distance': 6.0,
                    'description': 'Unrelated synsets default distance'
                }
            ]
        }
    
    @staticmethod
    def get_gloss_bonus_constant_tests():
        """Get test data specifically for GLOSS_BONUS constant."""
        return {
            'gloss_bonus_value': 0.15,
            'expected_type': float,
            'expected_range': (0.0, 1.0),
            'description': 'Bonus applied to heuristic for gloss seed nodes'
        }
    
    @staticmethod
    def get_initialization_test_data():
        """Get test data for initialization scenarios."""
        return {
            'basic_initialization': {
                'src': 'start',
                'tgt': 'end',
                'expected_defaults': {
                    'beam_width': 10,
                    'max_depth': 10,
                    'relax_beam': True,
                    'heuristic_type': 'hybrid'
                }
            },
            'custom_initialization': {
                'src': 'cat.n.01',
                'tgt': 'dog.n.01',
                'custom_params': {
                    'beam_width': 5,
                    'max_depth': 8,
                    'relax_beam': False,
                    'heuristic_type': 'embedding'
                }
            }
        }
    
    @staticmethod
    def get_search_state_validation_data():
        """Get data for validating search state initialization and management."""
        return {
            'initial_state_checks': {
                'priority_queue_sizes': {
                    'forward': 1,
                    'backward': 1
                },
                'g_score_initialization': {
                    'source_g_forward': 0.0,
                    'target_g_backward': 0.0
                },
                'depth_initialization': {
                    'source_depth_forward': 0,
                    'target_depth_backward': 0
                },
                'parent_initialization': {
                    'source_parent_forward': None,
                    'target_parent_backward': None
                },
                'closed_set_initialization': {
                    'forward_size': 0,
                    'backward_size': 0
                }
            },
            'queue_entry_structure': {
                'tuple_length': 3,
                'components': ['f_score', 'counter', 'node'],
                'f_score_type': 'float',
                'counter_type': 'int',
                'node_type': 'str'
            }
        }
    
    @staticmethod
    def get_path_reconstruction_scenarios():
        """Get scenarios for testing path reconstruction."""
        return {
            'simple_linear_path': {
                'meeting_node': 'middle',
                'forward_parents': {
                    'start': None,
                    'middle': 'start'
                },
                'backward_parents': {
                    'end': None,
                    'middle': 'end'
                },
                'expected_path': ['start', 'middle', 'end']
            },
            'complex_branching_path': {
                'meeting_node': 'junction',
                'forward_parents': {
                    'start': None,
                    'branch1': 'start',
                    'junction': 'branch1'
                },
                'backward_parents': {
                    'end': None,
                    'branch2': 'end',
                    'junction': 'branch2'
                },
                'expected_path_pattern': ['start', '*', 'junction', '*', 'end']
            }
        }
    
    @staticmethod
    def get_integration_test_scenarios():
        """Get integration test scenarios."""
        return {
            'wordnet_integration': {
                'description': 'Integration with WordNet taxonomy',
                'test_pairs': [
                    ('cat.n.01', 'mammal.n.01'),
                    ('dog.n.01', 'animal.n.01')
                ],
                'expected_path_properties': {
                    'min_length': 2,
                    'max_length': 8,
                    'contains_hypernyms': True
                }
            },
            'embedding_integration': {
                'description': 'Integration with embedding helper',
                'requires_embedding_helper': True,
                'test_parameters': {
                    'heuristic_type': 'embedding',
                    'similarity_threshold': 0.7
                }
            },
            'gloss_seed_integration': {
                'description': 'Integration with gloss seed nodes',
                'seed_nodes': ['seed1', 'seed2'],
                'expected_bonus_application': True,
                'expected_allowed_sets_inclusion': True
            }
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get validation test data for parameter validation."""
        return {
            'valid_parameters': {
                'beam_width_range': (1, 100),
                'max_depth_range': (1, 50),
                'heuristic_types': ['uniform', 'wordnet', 'embedding', 'hybrid']
            },
            'invalid_parameters': {
                'negative_beam_width': -1,
                'zero_max_depth': 0,
                'invalid_heuristic_type': 'unknown_type'
            },
            'boundary_conditions': {
                'min_beam_width': 1,
                'max_reasonable_depth': 20,
                'empty_graph': True,
                'single_node_graph': True
            }
        }