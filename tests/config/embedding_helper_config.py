"""
Configuration class containing mock constants and test data for EmbeddingHelper tests.

This module provides structured test data following the SMIED Testing Framework 
Design Specifications with:
- Static methods for test data access
- Hierarchically organized test scenarios
- Consistent data formats
- Scenario-based organization
"""

import numpy as np


class EmbeddingHelperMockConfig:
    """Configuration class containing mock constants and test data for EmbeddingHelper tests.
    
    Provides centralized test data management with static methods for:
    - Basic functionality testing
    - Validation and constraint testing  
    - Edge case and error condition testing
    - Integration testing scenarios
    """
    
    @staticmethod
    def get_embedding_dimensions():
        """Get embedding dimension constants for testing."""
        return {
            'small_dim': 50,
            'medium_dim': 100,
            'large_dim': 300,
            'standard_dim': 200
        }
    
    @staticmethod
    def get_test_embedding_vectors():
        """Get test embedding vectors with consistent formats."""
        dimensions = EmbeddingHelperMockConfig.get_embedding_dimensions()
        return {
            'cat_vector': np.random.rand(dimensions['standard_dim']).tolist(),
            'dog_vector': np.random.rand(dimensions['standard_dim']).tolist(),
            'animal_vector': np.random.rand(dimensions['standard_dim']).tolist(),
            'run_vector': np.random.rand(dimensions['standard_dim']).tolist(),
            'jump_vector': np.random.rand(dimensions['standard_dim']).tolist(),
            'zero_vector': [0.0] * dimensions['standard_dim'],
            'unit_vector': [1.0] + [0.0] * (dimensions['standard_dim'] - 1),
            'normalized_vector': (np.random.rand(dimensions['standard_dim']) / np.linalg.norm(np.random.rand(dimensions['standard_dim']))).tolist()
        }
    
    @staticmethod
    def get_synset_embedding_test_data():
        """Get synset embedding test data with consistent vector formats."""
        return {
            'synset_vectors': {
                'cat.n.01': [0.1, 0.2, 0.3, 0.4, 0.5],
                'dog.n.01': [0.2, 0.3, 0.4, 0.5, 0.6],
                'animal.n.01': [0.15, 0.25, 0.35, 0.45, 0.55],
                'run.v.01': [0.5, 0.4, 0.3, 0.2, 0.1],
                'walk.v.01': [0.4, 0.5, 0.2, 0.3, 0.1]
            },
            'expected_similarities': {
                ('cat.n.01', 'dog.n.01'): 0.87,
                ('cat.n.01', 'animal.n.01'): 0.92,
                ('dog.n.01', 'animal.n.01'): 0.89,
                ('run.v.01', 'walk.v.01'): 0.76
            }
        }
    
    @staticmethod
    def get_lexical_relation_test_structures():
        """Get lexical relation test structures."""
        return {
            'hypernym_relations': {
                'cat.n.01': [('animal.n.01', 0.9), ('mammal.n.01', 0.85)],
                'dog.n.01': [('animal.n.01', 0.88), ('mammal.n.01', 0.83)],
                'bird.n.01': [('animal.n.01', 0.91), ('vertebrate.n.01', 0.87)]
            },
            'hyponym_relations': {
                'animal.n.01': [('cat.n.01', 0.9), ('dog.n.01', 0.88), ('bird.n.01', 0.91)],
                'mammal.n.01': [('cat.n.01', 0.85), ('dog.n.01', 0.83)],
                'vehicle.n.01': [('car.n.01', 0.92), ('truck.n.01', 0.89)]
            },
            'similarity_relations': {
                'cat.n.01': [('feline.n.01', 0.95), ('kitten.n.01', 0.78)],
                'run.v.01': [('sprint.v.01', 0.88), ('jog.v.01', 0.72)],
                'happy.a.01': [('joyful.a.01', 0.91), ('glad.a.01', 0.85)]
            }
        }
    
    @staticmethod
    def get_similarity_calculation_test_matrices():
        """Get similarity calculation test matrices."""
        return {
            'cosine_similarity_tests': [
                {
                    'vector_a': [1.0, 0.0, 0.0],
                    'vector_b': [0.0, 1.0, 0.0],
                    'expected_similarity': 0.0
                },
                {
                    'vector_a': [1.0, 1.0, 0.0],
                    'vector_b': [1.0, 1.0, 0.0],
                    'expected_similarity': 1.0
                },
                {
                    'vector_a': [1.0, 0.0, 0.0],
                    'vector_b': [1.0, 1.0, 0.0],
                    'expected_similarity': 0.707  # approximately sqrt(2)/2
                }
            ],
            'euclidean_distance_tests': [
                {
                    'vector_a': [0.0, 0.0, 0.0],
                    'vector_b': [1.0, 1.0, 1.0],
                    'expected_distance': 1.732  # approximately sqrt(3)
                },
                {
                    'vector_a': [1.0, 2.0, 3.0],
                    'vector_b': [1.0, 2.0, 3.0],
                    'expected_distance': 0.0
                }
            ]
        }
    
    @staticmethod
    def get_realistic_embedding_model_mock_data():
        """Get realistic embedding model mock data."""
        return {
            'model_vocabulary': [
                'cat', 'dog', 'animal', 'run', 'walk', 'jump', 'house', 'car',
                'tree', 'book', 'computer', 'phone', 'happy', 'sad', 'fast', 'slow'
            ],
            'model_parameters': {
                'vector_size': 200,
                'window': 5,
                'min_count': 1,
                'workers': 4,
                'sg': 1  # skip-gram
            },
            'most_similar_results': {
                'cat': [('kitten', 0.85), ('feline', 0.82), ('pet', 0.79)],
                'dog': [('puppy', 0.87), ('canine', 0.84), ('pet', 0.81)],
                'run': [('sprint', 0.83), ('jog', 0.79), ('dash', 0.77)],
                'happy': [('joyful', 0.89), ('glad', 0.86), ('cheerful', 0.84)]
            }
        }
    
    @staticmethod
    def get_embedding_alignment_test_data():
        """Get embedding alignment test data."""
        return {
            'alignment_scenarios': [
                {
                    'source_embeddings': {
                        'hypernyms': [('animal.n.01', 0.9), ('mammal.n.01', 0.8)]
                    },
                    'target_embeddings': {
                        'hypernyms': [('animal.n.01', 0.85), ('mammal.n.01', 0.75)]
                    },
                    'expected_pairs': [
                        (('source', 'hypernyms'), ('target', 'hypernyms'), 0.875)
                    ]
                },
                {
                    'source_embeddings': {
                        'similar_tos': [('feline.n.01', 0.95)]
                    },
                    'target_embeddings': {
                        'similar_tos': [('canine.n.01', 0.92)]
                    },
                    'expected_pairs': [
                        (('source', 'similar_tos'), ('target', 'similar_tos'), 0.935)
                    ]
                }
            ]
        }
    
    @staticmethod
    def get_performance_test_configurations():
        """Get performance test configurations."""
        return {
            'small_scale': {
                'num_synsets': 100,
                'num_relations': 500,
                'vector_dimension': 50,
                'expected_time_ms': 1000
            },
            'medium_scale': {
                'num_synsets': 1000,
                'num_relations': 5000,
                'vector_dimension': 100,
                'expected_time_ms': 5000
            },
            'large_scale': {
                'num_synsets': 10000,
                'num_relations': 50000,
                'vector_dimension': 200,
                'expected_time_ms': 30000
            }
        }
    
    @staticmethod
    def get_edge_case_test_data():
        """Get edge case test data for error handling."""
        return {
            'empty_vectors': {
                'zero_length_vector': [],
                'all_zeros_vector': [0.0, 0.0, 0.0, 0.0, 0.0],
                'single_element_vector': [1.0]
            },
            'invalid_synsets': {
                'nonexistent_synset': 'fake.n.01',
                'malformed_synset': 'not_a_synset',
                'empty_synset': '',
                'none_synset': None
            },
            'mismatched_dimensions': {
                'vector_a': [1.0, 2.0, 3.0],
                'vector_b': [1.0, 2.0, 3.0, 4.0, 5.0],  # Different dimensions
                'expected_error': 'DimensionMismatchError'
            }
        }
    
    @staticmethod
    def get_batch_processing_test_data():
        """Get batch processing test data."""
        return {
            'small_batch': {
                'synsets': ['cat.n.01', 'dog.n.01', 'bird.n.01'],
                'batch_size': 10,
                'expected_processing_time_ms': 100
            },
            'large_batch': {
                'synsets': [f'synset_{i}.n.01' for i in range(1000)],
                'batch_size': 100,
                'expected_processing_time_ms': 5000
            },
            'streaming_batch': {
                'total_synsets': 10000,
                'batch_size': 50,
                'expected_memory_mb': 100
            }
        }
    
    @staticmethod
    def get_relation_type_mappings():
        """Get relation type mappings for testing."""
        return {
            'symmetric_relations': ['similar_tos', 'also_sees', 'verb_groups'],
            'asymmetric_relations': ['hypernyms', 'hyponyms', 'meronyms', 'holonyms'],
            'transitive_relations': ['hypernyms', 'hyponyms'],
            'part_relations': ['part_meronyms', 'part_holonyms'],
            'member_relations': ['member_meronyms', 'member_holonyms'],
            'substance_relations': ['substance_meronyms', 'substance_holonyms'],
            'all_relation_types': [
                'part_holonyms', 'substance_holonyms', 'member_holonyms',
                'part_meronyms', 'substance_meronyms', 'member_meronyms',
                'hypernyms', 'hyponyms', 'entailments', 'causes',
                'also_sees', 'verb_groups'
            ]
        }

    @staticmethod
    def get_test_lemma_names():
        """Get test lemma names for consistent testing."""
        return {
            'simple_lemmas': ["cat", "feline"],
            'compound_lemmas': ["ice_cream", "hot_dog"],
            'space_separated': ["ice cream", "hot dog"], 
            'multi_word_examples': [
                {"lemma": "hot_dog", "components": ["hot", "dog"]},
                {"lemma": "ice_cream", "components": ["ice", "cream"]},
                {"lemma": "multi word", "components": ["multi", "word"]},
                {"lemma": "partial multi", "components": ["partial", "multi"]}
            ],
            'mixed_availability_lemmas': ["available", "not_available", "multi word", "partial multi"]
        }

    @staticmethod
    def get_mock_synset_names():
        """Get mock synset names for testing."""
        return {
            'animal_synsets': {
                'cat': 'cat.n.01',
                'dog': 'dog.n.01', 
                'animal': 'animal.n.01',
                'mammal': 'mammal.n.01',
                'kitten': 'kitten.n.01',
                'puppy': 'puppy.n.01',
                'feline': 'feline.n.01',
                'canine': 'canine.n.01'
            },
            'action_synsets': {
                'run': 'run.v.01',
                'walk': 'walk.v.01',
                'jump': 'jump.v.01',
                'sprint': 'sprint.v.01'
            },
            'test_synsets': {
                'test': 'test.n.01',
                'related': 'related.n.01',
                'target': 'target.n.01'
            }
        }

    @staticmethod
    def get_embedding_test_vectors():
        """Get specific embedding vectors for testing scenarios."""
        return {
            'simple_vectors': {
                'cat': np.array([1.0, 2.0, 3.0]),
                'feline': np.array([2.0, 3.0, 4.0]),
                'ice cream': np.array([1.0, 2.0, 3.0]),
                'ice_cream': np.array([1.0, 2.0, 3.0])
            },
            'multi_word_vectors': {
                'hot': np.array([1.0, 2.0, 3.0]),
                'dog': np.array([3.0, 4.0, 5.0]),
                'ice': np.array([2.0, 3.0, 4.0]),
                'cream': np.array([1.0, 2.0, 3.0])
            },
            'similarity_test_vectors': {
                'synset1': np.array([1.0, 0.0, 0.0]),
                'synset2': np.array([1.0, 1.0, 0.0]),
                'synset3': np.array([0.0, 1.0, 0.0]),
                'synset4': np.array([1.0, 1.0, 0.0])
            },
            'zero_norm_vectors': {
                'zero': np.array([0.0, 0.0]),
                'normal': np.array([1.0, 1.0])
            },
            'mixed_availability_vectors': {
                'available': np.array([1.0, 1.0]),
                'multi': np.array([2.0, 2.0]),
                'word': np.array([3.0, 3.0]),
                'partial': np.array([4.0, 4.0])
            }
        }

    @staticmethod
    def get_relation_mapping_test_data():
        """Get relation mapping test data for alignment tests."""
        return {
            'basic_mapping': {
                "hypernyms": "hyponyms",
                "hyponyms": "hypernyms"
            },
            'extended_mapping': {
                "hypernyms": "hyponyms",
                "hyponyms": "hypernyms",
                "meronyms": "holonyms",
                "holonyms": "meronyms"
            },
            'missing_relation_mapping': {
                "hypernyms": "hyponyms",
                "missing_relation": "also_missing"
            }
        }

    @staticmethod
    def get_beam_test_parameters():
        """Get beam test parameters for various scenarios."""
        return {
            'small_beam': {'width': 2, 'expected_results': 2},
            'medium_beam': {'width': 3, 'expected_results': 3},
            'large_beam': {'width': 10, 'expected_results': 10},
            'zero_beam': {'width': 0, 'expected_behavior': 'return_all'},
            'negative_beam': {'width': -1, 'expected_behavior': 'return_all'}
        }

    @staticmethod
    def get_gloss_test_data():
        """Get gloss test data for predicate testing."""
        return {
            'test_definitions': [
                "test definition",
                "a small carnivorous mammal",
                "move fast by using legs"
            ],
            'wordnet_pos_constants': {
                'NOUN': 'n',
                'VERB': 'v',
                'ADJ': 'a',
                'ADV': 'r'
            },
            'extraction_modes': ['subjects', 'objects', 'predicates'],
            'sample_tokens': {
                'subject': 'subject',
                'object': 'object', 
                'predicate': 'predicate'
            },
            'max_sample_sizes': [3, 5, 10, 20],
            'target_synset_examples': ["target", "subject.n.01", "object.n.01"]
        }

    @staticmethod
    def get_integration_test_scenarios():
        """Get integration test scenarios for realistic testing."""
        return {
            'cat_synset_scenario': {
                'main_synset': 'cat.n.01',
                'lemmas': ['cat', 'feline'],
                'hypernym': 'mammal.n.01',
                'hyponym': 'kitten.n.01',
                'expected_relations': ['hypernyms', 'hyponyms']
            },
            'dog_synset_scenario': {
                'main_synset': 'dog.n.01',
                'lemmas': ['dog', 'canine'],
                'hypernym': 'mammal.n.01', 
                'hyponym': 'puppy.n.01',
                'expected_relations': ['hypernyms', 'hyponyms']
            },
            'workflow_parameters': {
                'embedding_dims': 100,
                'similarity_threshold': 0.5,
                'max_relations': 10
            }
        }

    # ===== TEST LAYER-SPECIFIC DATA ORGANIZATION =====
    
    @staticmethod
    def get_basic_test_data():
        """Get test data for basic functionality tests (TestEmbeddingHelper)."""
        return {
            'synset_centroid_tests': {
                'simple_synset': {
                    'synset_name': 'cat.n.01',
                    'lemmas': ['cat', 'feline'],
                    'expected_centroid_shape': (3,),
                    'mock_vectors': {
                        'cat': np.array([1.0, 2.0, 3.0]),
                        'feline': np.array([2.0, 3.0, 4.0])
                    },
                    'expected_centroid': np.array([1.5, 2.5, 3.5])
                },
                'compound_lemma': {
                    'synset_name': 'ice_cream.n.01',
                    'lemmas': ['ice_cream'],
                    'space_separated_lookup': 'ice cream',
                    'mock_vectors': {
                        'ice cream': np.array([1.0, 2.0, 3.0]),
                        'ice_cream': np.array([1.0, 2.0, 3.0])
                    }
                },
                'multi_word_fallback': {
                    'synset_name': 'hot_dog.n.01',
                    'lemmas': ['hot_dog'],
                    'component_words': ['hot', 'dog'],
                    'mock_vectors': {
                        'hot': np.array([1.0, 2.0, 3.0]),
                        'dog': np.array([3.0, 4.0, 5.0])
                    },
                    'expected_centroid': np.array([2.0, 3.0, 4.0])
                }
            },
            'lexical_relations_tests': {
                'basic_relations': {
                    'synset_name': 'cat.n.01',
                    'relations': {
                        'hypernyms': ['animal.n.01'],
                        'hyponyms': ['kitten.n.01'],
                        'empty_relations': ['part_holonyms', 'substance_holonyms', 'member_holonyms',
                                          'part_meronyms', 'substance_meronyms', 'member_meronyms',
                                          'entailments', 'causes', 'also_sees', 'verb_groups']
                    },
                    'expected_embeddings': {
                        'hypernyms': [('animal.n.01', np.array([1.0, 2.0]))],
                        'hyponyms': [('kitten.n.01', np.array([3.0, 4.0]))]
                    }
                }
            },
            'similarity_calculation_tests': {
                'basic_similarity': {
                    'rel_embs_1': [
                        ('synset1', np.array([1.0, 0.0, 0.0])),
                        ('synset2', np.array([1.0, 1.0, 0.0]))
                    ],
                    'rel_embs_2': [
                        ('synset3', np.array([0.0, 1.0, 0.0])),
                        ('synset4', np.array([1.0, 1.0, 0.0]))
                    ],
                    'expected_similarities': np.array([[0.0, 0.7071067811865476], [0.7071067811865476, 1.0]])
                }
            }
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get test data for validation tests (TestEmbeddingHelperValidation)."""
        return {
            'input_validation_scenarios': {
                'valid_synset_input': {
                    'synset_name': 'cat.n.01',
                    'model_type': 'dict',
                    'expected_validation': True
                },
                'invalid_synset_input': {
                    'synset_name': None,
                    'model_type': 'dict',
                    'expected_error': ValueError,
                    'error_message': 'Invalid synset input'
                },
                'empty_model_input': {
                    'synset_name': 'cat.n.01',
                    'model_type': 'empty_dict',
                    'expected_error': KeyError,
                    'error_message': 'Empty embedding model'
                }
            },
            'parameter_constraint_tests': {
                'beam_width_constraints': {
                    'valid_beam_widths': [1, 3, 5, 10],
                    'invalid_beam_widths': [-1, 0],
                    'edge_case_beam_widths': [1000, 0]  # 0 should return all results
                },
                'dimension_constraints': {
                    'valid_dimensions': [50, 100, 200, 300],
                    'invalid_dimensions': [0, -1],
                    'mismatched_dimensions': {
                        'vector_a': np.array([1.0, 2.0, 3.0]),
                        'vector_b': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                        'expected_error': ValueError
                    }
                }
            }
        }
    
    @staticmethod
    def get_edge_case_test_data():
        """Get test data for edge case tests (TestEmbeddingHelperEdgeCases)."""
        return {
            'empty_input_scenarios': {
                'empty_synset_lemmas': {
                    'synset_name': 'empty.n.01',
                    'lemmas': [],
                    'expected_centroid': np.array([])
                },
                'empty_embedding_lists': {
                    'rel_embs_1': [],
                    'rel_embs_2': [('synset', np.array([1.0]))],
                    'expected_similarity_shape': (0, 0)
                }
            },
            'zero_vector_scenarios': {
                'zero_norm_vectors': {
                    'vector_a': np.array([0.0, 0.0]),
                    'vector_b': np.array([1.0, 1.0]),
                    'expected_behavior': 'no_divide_by_zero_error'
                }
            },
            'exception_handling_scenarios': {
                'synset_lemmas_exception': {
                    'side_effect': Exception('Test exception'),
                    'expected_result': np.array([]),
                    'expected_print_call': True
                },
                'relation_method_exception': {
                    'relation': 'hypernyms',
                    'side_effect': Exception('Test error'),
                    'expected_empty_relation': True
                }
            },
            'large_scale_scenarios': {
                'large_matrices': {
                    'matrix_size_1': 50,
                    'matrix_size_2': 30,
                    'vector_dimension': 10,
                    'expected_shape': (50, 30),
                    'similarity_bounds': (-1.0, 1.0)
                }
            }
        }
    
    @staticmethod
    def get_integration_test_data():
        """Get test data for integration tests (TestEmbeddingHelperIntegration)."""
        return {
            'workflow_scenarios': {
                'cat_to_animal_workflow': {
                    'source_synset': {
                        'name': 'cat.n.01',
                        'lemmas': ['cat', 'feline'],
                        'hypernyms': ['mammal.n.01'],
                        'hyponyms': ['kitten.n.01']
                    },
                    'target_synset': {
                        'name': 'dog.n.01', 
                        'lemmas': ['dog', 'canine'],
                        'hypernyms': ['mammal.n.01'],
                        'hyponyms': ['puppy.n.01']
                    },
                    'embedding_model': {
                        'cat': np.array([1.0, 0.0, 0.0]),
                        'feline': np.array([0.9, 0.1, 0.0]),
                        'dog': np.array([0.8, 0.2, 0.0]),
                        'canine': np.array([0.7, 0.3, 0.0]),
                        'mammal': np.array([0.5, 0.5, 0.0]),
                        'kitten': np.array([0.8, 0.0, 0.2]),
                        'puppy': np.array([0.6, 0.2, 0.2])
                    },
                    'expected_beam_results': {
                        'min_similarity': 0.5,
                        'max_results': 5
                    }
                }
            },
            'realistic_beam_generation': {
                'multi_synset_scenario': {
                    'synsets': ['cat.n.01', 'dog.n.01', 'bird.n.01'],
                    'beam_width': 5,
                    'expected_result_structure': {
                        'result_type': list,
                        'item_structure': tuple,
                        'item_length': 3,
                        'similarity_index': 2
                    }
                }
            }
        }