"""
Configuration class containing mock constants and test data for SemanticDecomposer tests.
"""


class SemanticDecomposerMockConfig:
    """Configuration class containing mock constants and test data for SemanticDecomposer tests."""
    
    @staticmethod
    def get_wordnet_synset_names():
        """Get WordNet synset names for testing."""
        return {
            'animal_synsets': ['cat.n.01', 'dog.n.01', 'bird.n.01', 'fish.n.01'],
            'action_synsets': ['run.v.01', 'walk.v.01', 'jump.v.01', 'swim.v.01'],
            'location_synsets': ['park.n.01', 'house.n.01', 'street.n.01', 'forest.n.01'],
            'object_synsets': ['ball.n.01', 'book.n.01', 'car.n.01', 'tree.n.01'],
            'quality_synsets': ['fast.a.01', 'slow.a.01', 'big.a.01', 'small.a.01']
        }
    
    @staticmethod
    def get_pathfinding_test_scenarios():
        """Get pathfinding test scenarios."""
        return {
            'simple_path': {
                'source': 'cat.n.01',
                'target': 'animal.n.01',
                'expected_length': 2,
                'max_depth': 5
            },
            'complex_path': {
                'source': 'kitten.n.01',
                'target': 'vertebrate.n.01', 
                'expected_length': 4,
                'max_depth': 10
            },
            'no_path': {
                'source': 'rock.n.01',
                'target': 'emotion.n.01',
                'expected_length': None,
                'max_depth': 5
            },
            'long_path': {
                'source': 'specific_breed.n.01',
                'target': 'entity.n.01',
                'expected_length': 8,
                'max_depth': 15
            }
        }
    
    @staticmethod
    def get_graph_building_parameters():
        """Get graph building parameters for testing."""
        return {
            'basic_params': {
                'beam_width': 3,
                'max_depth': 6,
                'similarity_threshold': 0.5
            },
            'performance_params': {
                'beam_width': 5,
                'max_depth': 10,
                'similarity_threshold': 0.7
            },
            'comprehensive_params': {
                'beam_width': 10,
                'max_depth': 15,
                'similarity_threshold': 0.3
            }
        }
    
    @staticmethod
    def get_expected_graph_structures():
        """Get expected graph structures for validation."""
        return {
            'simple_taxonomy': {
                'nodes': ['cat.n.01', 'feline.n.01', 'carnivore.n.01', 'mammal.n.01', 'animal.n.01'],
                'edges': [
                    ('cat.n.01', 'feline.n.01'),
                    ('feline.n.01', 'carnivore.n.01'),
                    ('carnivore.n.01', 'mammal.n.01'),
                    ('mammal.n.01', 'animal.n.01')
                ]
            },
            'branching_taxonomy': {
                'root': 'vehicle.n.01',
                'branches': {
                    'land_vehicle.n.01': ['car.n.01', 'truck.n.01', 'bicycle.n.01'],
                    'water_vehicle.n.01': ['boat.n.01', 'ship.n.01'],
                    'air_vehicle.n.01': ['airplane.n.01', 'helicopter.n.01']
                }
            }
        }
    
    @staticmethod
    def get_mock_gloss_parsing_results():
        """Get mock gloss parsing results."""
        return {
            'cat.n.01': {
                'gloss': 'feline mammal usually having thick soft fur',
                'parsed_tokens': ['feline', 'mammal', 'thick', 'soft', 'fur'],
                'key_terms': ['feline', 'mammal', 'fur'],
                'relations': [('mammal', 'feline'), ('mammal', 'fur')]
            },
            'run.v.01': {
                'gloss': 'move fast by using legs',
                'parsed_tokens': ['move', 'fast', 'using', 'legs'],
                'key_terms': ['move', 'fast', 'legs'],
                'relations': [('move', 'fast'), ('move', 'legs')]
            },
            'park.n.01': {
                'gloss': 'a large area of land preserved in its natural state',
                'parsed_tokens': ['large', 'area', 'land', 'preserved', 'natural', 'state'],
                'key_terms': ['area', 'land', 'natural'],
                'relations': [('area', 'large'), ('area', 'land'), ('land', 'natural')]
            }
        }
    
    @staticmethod
    def get_integration_test_parameters():
        """Get parameters for integration testing."""
        return {
            'realistic_scenario_1': {
                'source_synset': 'domestic_cat.n.01',
                'target_synset': 'vertebrate.n.01',
                'embedding_model': 'test_model',
                'expected_components': ['pathfinder', 'beam_builder', 'gloss_parser'],
                'beam_width': 4,
                'max_depth': 8
            },
            'realistic_scenario_2': {
                'source_synset': 'sprint.v.01',
                'target_synset': 'locomotion.n.01',
                'embedding_model': 'test_model',
                'expected_components': ['pathfinder', 'beam_builder', 'gloss_parser'],
                'beam_width': 3,
                'max_depth': 6
            }
        }
    
    @staticmethod
    def get_error_handling_scenarios():
        """Get scenarios for error handling testing."""
        return {
            'invalid_synsets': {
                'nonexistent_synset': 'invalid.n.01',
                'malformed_synset': 'not_a_synset',
                'empty_synset': '',
                'none_synset': None
            },
            'invalid_parameters': {
                'negative_beam_width': -1,
                'zero_beam_width': 0,
                'negative_max_depth': -5,
                'zero_max_depth': 0,
                'invalid_threshold': 1.5
            },
            'missing_components': {
                'no_pathfinder': {'beam_builder': True, 'gloss_parser': True},
                'no_beam_builder': {'pathfinder': True, 'gloss_parser': True},
                'no_gloss_parser': {'pathfinder': True, 'beam_builder': True}
            }
        }
    
    @staticmethod
    def get_performance_benchmarks():
        """Get performance benchmarks for testing."""
        return {
            'small_graph': {
                'max_nodes': 100,
                'max_edges': 200,
                'expected_time_ms': 1000,
                'memory_limit_mb': 50
            },
            'medium_graph': {
                'max_nodes': 1000,
                'max_edges': 2000,
                'expected_time_ms': 5000,
                'memory_limit_mb': 200
            },
            'large_graph': {
                'max_nodes': 10000,
                'max_edges': 20000,
                'expected_time_ms': 30000,
                'memory_limit_mb': 500
            }
        }
    
    @staticmethod
    def get_semantic_relation_types():
        """Get semantic relation types for testing."""
        return {
            'hypernym_relations': ['hypernyms', 'instance_hypernyms'],
            'hyponym_relations': ['hyponyms', 'instance_hyponyms'],
            'meronym_relations': ['part_meronyms', 'member_meronyms', 'substance_meronyms'],
            'holonym_relations': ['part_holonyms', 'member_holonyms', 'substance_holonyms'],
            'similarity_relations': ['similar_tos', 'also_sees'],
            'verbal_relations': ['entailments', 'causes', 'verb_groups']
        }
    
    @staticmethod
    def get_wordnet_hierarchy_samples():
        """Get WordNet hierarchy samples for testing."""
        return {
            'animal_hierarchy': {
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
            }
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get test data for validation tests."""
        return {
            'valid_inputs': {
                'subject': 'cat',
                'predicate': 'run', 
                'object': 'park',
                'beam_width': 3,
                'max_depth': 6,
                'similarity_threshold': 0.5
            },
            'invalid_inputs': {
                'empty_subject': '',
                'none_predicate': None,
                'invalid_beam_width': -1,
                'invalid_max_depth': 0,
                'invalid_threshold': 1.5
            },
            'boundary_inputs': {
                'min_beam_width': 1,
                'max_beam_width': 100,
                'min_max_depth': 1,
                'max_max_depth': 50,
                'min_threshold': 0.0,
                'max_threshold': 1.0
            }
        }
    
    @staticmethod
    def get_edge_case_test_data():
        """Get test data for edge case tests."""
        return {
            'empty_scenarios': {
                'no_synsets_found': {
                    'subject': 'nonexistent_word_xyz',
                    'predicate': 'nonexistent_action_abc', 
                    'object': 'nonexistent_location_def'
                },
                'empty_graph': {
                    'nodes': [],
                    'edges': []
                },
                'no_paths_found': {
                    'source': 'emotion.n.01',
                    'target': 'rock.n.01'
                }
            },
            'malformed_scenarios': {
                'invalid_synset_names': ['invalid', 'not.a.synset', ''],
                'malformed_graph_data': {
                    'nodes': [None, '', 123],
                    'edges': [(None, 'valid'), ('', None)]
                }
            },
            'performance_scenarios': {
                'large_search_space': {
                    'beam_width': 50,
                    'max_depth': 20,
                    'expected_timeout_ms': 10000
                },
                'memory_intensive': {
                    'large_graph_nodes': 10000,
                    'large_graph_edges': 50000
                }
            }
        }
    
    @staticmethod
    def get_integration_scenarios():
        """Get comprehensive integration test scenarios."""
        return {
            'realistic_semantic_decomposition': {
                'triple_1': {
                    'subject': 'scientist',
                    'predicate': 'study',
                    'object': 'organism',
                    'expected_frame': 'Research',
                    'expected_roles': ['Researcher', 'Topic'],
                    'expected_path_length': [2, 4]
                },
                'triple_2': {
                    'subject': 'student',
                    'predicate': 'learn',
                    'object': 'mathematics',
                    'expected_frame': 'Education_teaching',
                    'expected_roles': ['Student', 'Subject'],
                    'expected_path_length': [1, 3]
                }
            },
            'multi_strategy_scenarios': {
                'frame_connection': {
                    'synsets': ['teach.v.01', 'learn.v.01'],
                    'strategy': 'framenet_srl',
                    'expected_success': True
                },
                'derivational_fallback': {
                    'synsets': ['teacher.n.01', 'teaching.n.01'], 
                    'strategy': 'derivational_morphology',
                    'expected_success': True
                },
                'hypernym_fallback': {
                    'synsets': ['cat.n.01', 'mammal.n.01'],
                    'strategy': 'hypernym_hyponym',
                    'expected_success': True
                }
            }
        }
    
    @staticmethod
    def get_mock_setup_configurations():
        """Get configurations for mock setup in different test scenarios."""
        return {
            'basic_setup': {
                'wordnet_mock': 'MockWordNetForDecomposer',
                'nlp_mock': 'MockNLPForDecomposer',
                'embedding_mock': 'MockEmbeddingModelForDecomposer',
                'additional_components': ['BeamBuilder', 'EmbeddingHelper', 'GlossParser']
            },
            'validation_setup': {
                'wordnet_mock': 'MockWordNetForDecomposer',
                'nlp_mock': 'MockNLPForDecomposer',
                'validation_mock': 'MockSemanticDecomposerValidation',
                'enable_validation': True
            },
            'edge_case_setup': {
                'wordnet_mock': 'MockWordNetForDecomposer',
                'nlp_mock': 'MockNLPForDecomposer',
                'edge_case_mock': 'MockSemanticDecomposerEdgeCases',
                'simulate_failures': True
            },
            'integration_setup': {
                'wordnet_mock': 'MockRealWordNet',
                'nlp_mock': 'MockNLPForDecomposer',
                'integration_mock': 'MockSemanticDecomposerIntegration',
                'use_real_components': ['FrameNetSpacySRL', 'DerivationalMorphology']
            }
        }
    
    @staticmethod
    def get_expected_test_outcomes():
        """Get expected outcomes for different test scenarios."""
        return {
            'successful_pathfinding': {
                'has_subject_path': True,
                'has_object_path': True,
                'has_predicate': True,
                'path_length_range': [1, 10],
                'similarity_score_range': [0.1, 1.0]
            },
            'failed_pathfinding': {
                'has_subject_path': False,
                'has_object_path': False,
                'has_predicate': False,
                'expected_exceptions': ['ValueError', 'TimeoutError']
            },
            'partial_success': {
                'has_subject_path': True,
                'has_object_path': False,
                'has_predicate': True,
                'fallback_strategies_used': ['hypernym_hyponym']
            }
        }