"""
Configuration class containing mock constants and test data for EmbeddingHelper tests.
"""

import numpy as np


class EmbeddingHelperMockConfig:
    """Configuration class containing mock constants and test data for EmbeddingHelper tests."""
    
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
            'substance_relations': ['substance_meronyms', 'substance_holonyms']
        }