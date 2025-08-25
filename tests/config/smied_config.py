"""
Configuration class containing mock constants and test data for SMIED tests.
"""


class SMIEDMockConfig:
    """Configuration class containing mock constants and test data for SMIED tests."""
    
    @staticmethod
    def get_model_name_constants():
        """Get model name constants for testing."""
        return {
            'default_spacy_model': 'en_core_web_sm',
            'large_spacy_model': 'en_core_web_lg',
            'medium_spacy_model': 'en_core_web_md',
            'test_spacy_model': 'en_core_web_test',
            'embedding_model': 'word2vec_model',
            'test_embedding_model': 'test_model'
        }
    
    @staticmethod
    def get_triple_analysis_test_cases():
        """Get triple analysis test cases."""
        return {
            'simple_triples': [
                ("cat", "chase", "mouse"),
                ("dog", "run", "park"),
                ("bird", "fly", "sky"),
                ("fish", "swim", "water")
            ],
            'complex_triples': [
                ("student", "study", "mathematics"),
                ("scientist", "discover", "phenomenon"),
                ("artist", "create", "masterpiece"),
                ("teacher", "explain", "concept")
            ],
            'entity_triples': [
                ("John", "work", "Microsoft"),
                ("Apple", "manufacture", "iPhone"),
                ("Einstein", "develop", "relativity"),
                ("Shakespeare", "write", "Hamlet")
            ],
            'abstract_triples': [
                ("happiness", "bring", "joy"),
                ("knowledge", "lead", "wisdom"),
                ("practice", "improve", "skill"),
                ("time", "heal", "wound")
            ]
        }
    
    @staticmethod
    def get_mock_synset_structures():
        """Get mock synset structures with consistent format."""
        return {
            'animal_synsets': {
                'cat.n.01': {
                    'name': 'cat.n.01',
                    'definition': 'feline mammal usually having thick soft fur and no ability to roar: domestic cats; wildcats',
                    'pos': 'n',
                    'examples': ['cats are often kept as pets'],
                    'lemma_names': ['cat', 'true_cat'],
                    'hypernyms': ['feline.n.01'],
                    'hyponyms': ['wildcat.n.03', 'domestic_cat.n.01']
                },
                'dog.n.01': {
                    'name': 'dog.n.01',
                    'definition': 'a member of the genus Canis',
                    'pos': 'n',
                    'examples': ['the dog barked all night'],
                    'lemma_names': ['dog', 'domestic_dog', 'Canis_familiaris'],
                    'hypernyms': ['canine.n.01'],
                    'hyponyms': ['puppy.n.01', 'hound.n.01']
                },
                'animal.n.01': {
                    'name': 'animal.n.01',
                    'definition': 'a living organism characterized by voluntary movement',
                    'pos': 'n',
                    'examples': ['animals in the zoo'],
                    'lemma_names': ['animal', 'animate_being', 'beast', 'brute', 'creature', 'fauna'],
                    'hypernyms': ['organism.n.01'],
                    'hyponyms': ['mammal.n.01', 'bird.n.01', 'fish.n.01']
                }
            },
            'action_synsets': {
                'run.v.01': {
                    'name': 'run.v.01',
                    'definition': 'move fast by using legs',
                    'pos': 'v',
                    'examples': ['The children ran to the store'],
                    'lemma_names': ['run'],
                    'hypernyms': ['locomote.v.01'],
                    'hyponyms': ['sprint.v.01', 'jog.v.01']
                },
                'walk.v.01': {
                    'name': 'walk.v.01', 
                    'definition': 'use legs to move from one place to another',
                    'pos': 'v',
                    'examples': ['We walked for miles'],
                    'lemma_names': ['walk'],
                    'hypernyms': ['locomote.v.01'],
                    'hyponyms': ['stroll.v.01', 'hike.v.01']
                }
            },
            'location_synsets': {
                'park.n.01': {
                    'name': 'park.n.01',
                    'definition': 'a large area of land preserved in its natural state',
                    'pos': 'n',
                    'examples': ['they went for a walk in the park'],
                    'lemma_names': ['park', 'parkland'],
                    'hypernyms': ['tract.n.01'],
                    'hyponyms': ['national_park.n.01', 'theme_park.n.01']
                }
            }
        }
    
    @staticmethod
    def get_similarity_calculation_test_data():
        """Get similarity calculation test data."""
        return {
            'wordnet_similarity_pairs': [
                {
                    'synset1': 'cat.n.01',
                    'synset2': 'dog.n.01',
                    'path_similarity': 0.2,
                    'wup_similarity': 0.8571,
                    'lch_similarity': 1.072
                },
                {
                    'synset1': 'cat.n.01',
                    'synset2': 'animal.n.01',
                    'path_similarity': 0.125,
                    'wup_similarity': 0.4,
                    'lch_similarity': 0.693
                },
                {
                    'synset1': 'run.v.01',
                    'synset2': 'walk.v.01',
                    'path_similarity': 0.333,
                    'wup_similarity': 0.8,
                    'lch_similarity': 1.386
                }
            ],
            'embedding_similarity_pairs': [
                {
                    'word1': 'cat',
                    'word2': 'dog',
                    'cosine_similarity': 0.75,
                    'euclidean_distance': 0.8
                },
                {
                    'word1': 'run',
                    'word2': 'walk',
                    'cosine_similarity': 0.68,
                    'euclidean_distance': 0.9
                },
                {
                    'word1': 'happy',
                    'word2': 'joyful',
                    'cosine_similarity': 0.87,
                    'euclidean_distance': 0.6
                }
            ]
        }
    
    @staticmethod
    def get_integration_test_scenarios():
        """Get integration test scenarios with expected results."""
        return {
            'semantic_analysis_scenario': {
                'input_triple': ("cat", "chase", "mouse"),
                'expected_components': {
                    'semantic_metagraph': 'initialized',
                    'semantic_decomposer': 'initialized',
                    'gloss_parser': 'initialized',
                    'embedding_helper': 'initialized'
                },
                'expected_analysis_steps': [
                    'parse_input_triple',
                    'create_semantic_graph',
                    'find_wordnet_synsets',
                    'compute_semantic_paths',
                    'calculate_similarities',
                    'generate_analysis_report'
                ],
                'expected_output_format': {
                    'source_synsets': ['cat.n.01'],
                    'target_synsets': ['mouse.n.01'],
                    'relation_synsets': ['chase.v.01'],
                    'semantic_paths': 'list_of_paths',
                    'similarity_scores': 'dict_of_scores'
                }
            },
            'end_to_end_scenario': {
                'input_text': "The dog ran quickly through the park.",
                'processing_pipeline': [
                    'text_to_semantic_graph',
                    'extract_key_relations',
                    'find_semantic_connections',
                    'compute_relation_strengths',
                    'generate_insights'
                ],
                'expected_entities': ['dog', 'park'],
                'expected_actions': ['ran'],
                'expected_modifiers': ['quickly', 'through']
            }
        }
    
    @staticmethod
    def get_component_initialization_data():
        """Get component initialization data for testing."""
        return {
            'required_components': [
                'semantic_metagraph',
                'semantic_decomposer', 
                'gloss_parser',
                'embedding_helper'
            ],
            'optional_components': [
                'pattern_matcher',
                'beam_builder',
                'pathfinder'
            ],
            'component_dependencies': {
                'semantic_decomposer': ['gloss_parser', 'embedding_helper'],
                'semantic_metagraph': ['spacy_nlp'],
                'pattern_matcher': ['semantic_metagraph'],
                'beam_builder': ['embedding_helper']
            },
            'initialization_parameters': {
                'semantic_metagraph': {'spacy_model': 'en_core_web_sm'},
                'embedding_helper': {'model_path': 'test_model'},
                'gloss_parser': {'spacy_model': 'en_core_web_sm'},
                'semantic_decomposer': {'beam_width': 3, 'max_depth': 6}
            }
        }
    
    @staticmethod
    def get_error_handling_scenarios():
        """Get error handling scenarios for testing."""
        return {
            'invalid_input_scenarios': [
                {
                    'input': None,
                    'expected_error': 'ValueError',
                    'error_message': 'Input cannot be None'
                },
                {
                    'input': '',
                    'expected_error': 'ValueError', 
                    'error_message': 'Input cannot be empty'
                },
                {
                    'input': ('incomplete_triple',),
                    'expected_error': 'ValueError',
                    'error_message': 'Triple must have exactly 3 elements'
                }
            ],
            'component_failure_scenarios': [
                {
                    'failing_component': 'spacy_model',
                    'failure_type': 'model_not_found',
                    'expected_behavior': 'graceful_fallback'
                },
                {
                    'failing_component': 'embedding_model',
                    'failure_type': 'model_load_error',
                    'expected_behavior': 'continue_without_embeddings'
                },
                {
                    'failing_component': 'wordnet',
                    'failure_type': 'synset_not_found',
                    'expected_behavior': 'skip_unknown_synsets'
                }
            ],
            'resource_limitation_scenarios': [
                {
                    'scenario': 'memory_exhaustion',
                    'trigger': 'very_large_graph',
                    'expected_behavior': 'implement_streaming'
                },
                {
                    'scenario': 'processing_timeout',
                    'trigger': 'complex_analysis',
                    'expected_behavior': 'return_partial_results'
                },
                {
                    'scenario': 'disk_space_exhaustion',
                    'trigger': 'large_cache_files',
                    'expected_behavior': 'clean_cache_and_retry'
                }
            ],
            'concurrent_access_scenarios': [
                {
                    'scenario': 'multiple_threads_same_instance',
                    'expected_behavior': 'thread_safe_access'
                },
                {
                    'scenario': 'concurrent_graph_building',
                    'expected_behavior': 'prevent_race_conditions'
                }
            ],
            'malformed_input_scenarios': [
                {
                    'input': ("", "", ""),
                    'expected_error': 'ValueError',
                    'error_message': 'All triple elements must be non-empty'
                },
                {
                    'input': (None, "valid", "valid"),
                    'expected_error': 'TypeError',
                    'error_message': 'Triple elements cannot be None'
                },
                {
                    'input': ("valid", 123, "valid"),
                    'expected_error': 'TypeError',
                    'error_message': 'Triple elements must be strings'
                }
            ]
        }
    
    @staticmethod
    def get_performance_expectations():
        """Get performance expectations for different scenarios."""
        return {
            'single_triple_analysis': {
                'input_size': '3 words',
                'max_processing_time_ms': 1000,
                'max_memory_mb': 50,
                'expected_output_size': 'analysis_dict'
            },
            'batch_triple_analysis': {
                'input_size': '100 triples',
                'max_processing_time_ms': 30000,
                'max_memory_mb': 200,
                'expected_throughput': '3.3 triples/second'
            },
            'large_text_analysis': {
                'input_size': '10000 words',
                'max_processing_time_ms': 60000,
                'max_memory_mb': 500,
                'expected_entities': '> 100 entities'
            }
        }
    
    @staticmethod
    def get_configuration_options():
        """Get configuration options for SMIED system."""
        return {
            'analysis_modes': [
                'basic_similarity',
                'deep_semantic_analysis',
                'relation_extraction',
                'concept_mapping'
            ],
            'output_formats': [
                'json',
                'xml',
                'graph_format',
                'text_summary'
            ],
            'similarity_metrics': [
                'path_similarity',
                'wup_similarity',
                'lch_similarity',
                'cosine_similarity',
                'jaccard_similarity'
            ],
            'processing_options': {
                'use_caching': True,
                'parallel_processing': False,
                'verbose_logging': False,
                'save_intermediate_results': False
            }
        }
    
    @staticmethod
    def get_validation_test_cases():
        """Get validation test cases for input/output verification."""
        return {
            'input_validation': [
                {
                    'input': ("valid", "input", "triple"),
                    'is_valid': True,
                    'validation_notes': 'Proper triple format'
                },
                {
                    'input': ["list", "instead", "of", "tuple"],
                    'is_valid': True,
                    'validation_notes': 'Should convert list to tuple'
                },
                {
                    'input': ("too", "few"),
                    'is_valid': False,
                    'validation_notes': 'Insufficient elements'
                },
                {
                    'input': ("too", "many", "elements", "here"),
                    'is_valid': False,
                    'validation_notes': 'Too many elements'
                }
            ],
            'output_validation': [
                {
                    'description': 'Complete analysis output',
                    'required_keys': ['source_analysis', 'relation_analysis', 'target_analysis', 'similarity_scores'],
                    'optional_keys': ['metadata', 'processing_time', 'confidence_scores']
                },
                {
                    'description': 'Error output format',
                    'required_keys': ['error_type', 'error_message', 'timestamp'],
                    'optional_keys': ['debug_info', 'suggested_actions']
                }
            ]
        }
    
    @staticmethod
    def get_edge_case_test_scenarios():
        """Get edge case test scenarios for comprehensive testing."""
        return {
            'synset_edge_cases': [
                {
                    'edge_case_type': 'empty_synset',
                    'expected_behavior': 'handle_gracefully',
                    'should_raise_error': True
                },
                {
                    'edge_case_type': 'malformed_synset',
                    'expected_behavior': 'validate_and_reject',
                    'should_raise_error': True
                },
                {
                    'edge_case_type': 'circular_relationships',
                    'expected_behavior': 'detect_and_handle_cycles',
                    'should_raise_error': False
                },
                {
                    'edge_case_type': 'missing_relationships',
                    'expected_behavior': 'accept_isolated_synsets',
                    'should_raise_error': False
                }
            ],
            'analysis_edge_cases': [
                {
                    'scenario': 'very_long_words',
                    'input': ("supercalifragilisticexpialidocious", "analyze", "antidisestablishmentarianism"),
                    'expected_behavior': 'handle_long_strings'
                },
                {
                    'scenario': 'unicode_characters',
                    'input': ("café", "naïve", "résumé"),
                    'expected_behavior': 'handle_unicode'
                },
                {
                    'scenario': 'special_characters',
                    'input': ("@user", "#hashtag", "$money"),
                    'expected_behavior': 'handle_special_chars'
                }
            ],
            'memory_edge_cases': [
                {
                    'scenario': 'large_graph_analysis',
                    'graph_size': 100000,
                    'expected_behavior': 'use_memory_efficiently'
                },
                {
                    'scenario': 'repeated_analysis_calls',
                    'call_count': 1000,
                    'expected_behavior': 'maintain_performance'
                }
            ]
        }
    
    @staticmethod  
    def get_mock_factory_test_cases():
        """Get test cases for mock factory validation."""
        return {
            'valid_mock_types': [
                'MockISMIEDPipeline',
                'MockSMIED', 
                'MockSMIEDIntegration',
                'MockSMIEDEdgeCases',
                'MockNLTK',
                'MockSpacy',
                'MockWordNet',
                'MockSynset',
                'MockSynsetEdgeCases',
                'MockSemanticDecomposer',
                'MockGraph'
            ],
            'invalid_mock_types': [
                'NonexistentMock',
                'InvalidMockType',
                '',
                None
            ],
            'mock_creation_scenarios': [
                {
                    'mock_type': 'MockSynset',
                    'args': [],
                    'kwargs': {'name': 'test.n.01', 'definition': 'test'},
                    'expected_success': True
                },
                {
                    'mock_type': 'MockSMIEDEdgeCases',
                    'args': [],
                    'kwargs': {'failure_mode': 'analysis_error'},
                    'expected_success': True
                }
            ]
        }