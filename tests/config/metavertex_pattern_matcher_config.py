"""
Configuration class containing mock constants and test data for MetavertexPatternMatcher tests.
"""


class MetavertexPatternMatcherMockConfig:
    """Configuration class containing mock constants and test data for MetavertexPatternMatcher tests."""
    
    @staticmethod
    def get_test_sentence_patterns():
        """Get test sentence patterns for metavertex matching."""
        return {
            'simple_sentences': [
                "The cat runs fast.",
                "Dogs bark loudly.",
                "Birds fly high.",
                "Fish swim underwater."
            ],
            'complex_sentences': [
                "John gives Mary a book because she studies hard.",
                "The quick brown fox jumps over the lazy dog.",
                "Although it was raining, we decided to go hiking.",
                "The student who works hard will succeed in life."
            ],
            'compound_sentences': [
                "The sun is shining, and the birds are singing.",
                "She wanted to go to the party, but she had to work.",
                "You can take the bus, or you can walk to school.",
                "He studied all night, so he passed the exam."
            ],
            'interrogative_sentences': [
                "Where did the cat go?",
                "What time does the meeting start?", 
                "How fast can the car drive?",
                "Why are the birds singing?"
            ],
            'passive_sentences': [
                "The book was written by the author.",
                "The cake was eaten by the children.",
                "The house was built by skilled workers.",
                "The song was sung by the choir."
            ]
        }
    
    @staticmethod
    def get_spacy_model_setup_constants():
        """Get spaCy model setup constants."""
        return {
            'default_model': 'en_core_web_sm',
            'large_model': 'en_core_web_lg',
            'test_model': 'en_core_web_test',
            'model_components': ['tagger', 'parser', 'ner'],
            'disable_components': ['lemmatizer', 'textcat'],
            'model_config': {
                'max_length': 1000000,
                'disable': []
            }
        }
    
    @staticmethod
    def get_pattern_matching_test_structures():
        """Get pattern matching test structures."""
        return {
            'svo_pattern': {
                'description': 'Subject-Verb-Object pattern',
                'structure': {
                    'subject': {'pos': 'NOUN', 'dep': 'nsubj'},
                    'verb': {'pos': 'VERB', 'dep': 'ROOT'},
                    'object': {'pos': 'NOUN', 'dep': 'dobj'}
                },
                'expected_matches': [
                    {'subject': 'cat', 'verb': 'chases', 'object': 'mouse'},
                    {'subject': 'dog', 'verb': 'fetches', 'object': 'ball'},
                    {'subject': 'bird', 'verb': 'builds', 'object': 'nest'}
                ]
            },
            'prepositional_phrase_pattern': {
                'description': 'Verb with prepositional phrase',
                'structure': {
                    'verb': {'pos': 'VERB', 'dep': 'ROOT'},
                    'preposition': {'pos': 'ADP', 'dep': 'prep'},
                    'object': {'pos': 'NOUN', 'dep': 'pobj'}
                },
                'expected_matches': [
                    {'verb': 'runs', 'preposition': 'through', 'object': 'park'},
                    {'verb': 'sits', 'preposition': 'on', 'object': 'chair'},
                    {'verb': 'looks', 'preposition': 'at', 'object': 'picture'}
                ]
            },
            'modifier_pattern': {
                'description': 'Noun with modifiers',
                'structure': {
                    'determiner': {'pos': 'DET', 'dep': 'det'},
                    'adjective': {'pos': 'ADJ', 'dep': 'amod'},
                    'noun': {'pos': 'NOUN', 'dep': 'nsubj'}
                },
                'expected_matches': [
                    {'determiner': 'the', 'adjective': 'quick', 'noun': 'fox'},
                    {'determiner': 'a', 'adjective': 'large', 'noun': 'dog'},
                    {'determiner': 'the', 'adjective': 'beautiful', 'noun': 'flower'}
                ]
            }
        }
    
    @staticmethod
    def get_expected_matching_results():
        """Get expected matching results for test sentences."""
        return {
            'the_cat_runs_fast': {
                'sentence': "The cat runs fast.",
                'expected_patterns': [
                    {
                        'pattern_type': 'svo_with_modifier',
                        'matches': {
                            'subject': 'cat',
                            'verb': 'runs',
                            'modifier': 'fast'
                        },
                        'confidence': 0.95
                    }
                ],
                'expected_metavertices': [
                    {'text': 'The', 'pos': 'DET', 'dep': 'det'},
                    {'text': 'cat', 'pos': 'NOUN', 'dep': 'nsubj'},
                    {'text': 'runs', 'pos': 'VERB', 'dep': 'ROOT'},
                    {'text': 'fast', 'pos': 'ADV', 'dep': 'advmod'}
                ]
            },
            'john_gives_mary_book': {
                'sentence': "John gives Mary a book because she studies hard.",
                'expected_patterns': [
                    {
                        'pattern_type': 'ditransitive',
                        'matches': {
                            'subject': 'John',
                            'verb': 'gives',
                            'indirect_object': 'Mary',
                            'direct_object': 'book'
                        }
                    },
                    {
                        'pattern_type': 'causal_clause',
                        'matches': {
                            'conjunction': 'because',
                            'subject': 'she',
                            'verb': 'studies',
                            'modifier': 'hard'
                        }
                    }
                ]
            }
        }
    
    @staticmethod
    def get_complex_sentence_examples():
        """Get complex sentence examples for advanced testing."""
        return {
            'relative_clause_sentence': {
                'text': "The student who studies hard will succeed.",
                'expected_structure': {
                    'main_clause': {
                        'subject': 'student',
                        'verb': 'will succeed'
                    },
                    'relative_clause': {
                        'relative_pronoun': 'who',
                        'verb': 'studies',
                        'modifier': 'hard'
                    }
                },
                'expected_dependencies': [
                    ('student', 'succeed', 'nsubj'),
                    ('who', 'studies', 'nsubj'),
                    ('studies', 'student', 'relcl'),
                    ('hard', 'studies', 'advmod')
                ]
            },
            'conditional_sentence': {
                'text': "If it rains, we will stay inside.",
                'expected_structure': {
                    'condition_clause': {
                        'marker': 'If',
                        'subject': 'it',
                        'verb': 'rains'
                    },
                    'main_clause': {
                        'subject': 'we',
                        'verb': 'will stay',
                        'location': 'inside'
                    }
                },
                'pattern_type': 'conditional'
            },
            'coordination_sentence': {
                'text': "The cat sleeps and the dog plays.",
                'expected_structure': {
                    'coordinated_clauses': [
                        {
                            'subject': 'cat',
                            'verb': 'sleeps'
                        },
                        {
                            'subject': 'dog', 
                            'verb': 'plays'
                        }
                    ],
                    'coordinator': 'and'
                },
                'pattern_type': 'coordination'
            }
        }
    
    @staticmethod
    def get_entity_recognition_scenarios():
        """Get entity recognition scenarios for testing."""
        return {
            'person_entities': {
                'sentences': [
                    "Barack Obama was the president.",
                    "Einstein developed the theory of relativity.",
                    "Shakespeare wrote many famous plays."
                ],
                'expected_entities': [
                    {'text': 'Barack Obama', 'label': 'PERSON'},
                    {'text': 'Einstein', 'label': 'PERSON'},
                    {'text': 'Shakespeare', 'label': 'PERSON'}
                ]
            },
            'organization_entities': {
                'sentences': [
                    "Apple released a new iPhone.",
                    "Microsoft develops software products.",
                    "Google processes billions of searches."
                ],
                'expected_entities': [
                    {'text': 'Apple', 'label': 'ORG'},
                    {'text': 'Microsoft', 'label': 'ORG'},
                    {'text': 'Google', 'label': 'ORG'}
                ]
            },
            'location_entities': {
                'sentences': [
                    "Paris is the capital of France.",
                    "The meeting is in New York.",
                    "Tokyo is a large city in Japan."
                ],
                'expected_entities': [
                    {'text': 'Paris', 'label': 'GPE'},
                    {'text': 'France', 'label': 'GPE'},
                    {'text': 'New York', 'label': 'GPE'},
                    {'text': 'Tokyo', 'label': 'GPE'},
                    {'text': 'Japan', 'label': 'GPE'}
                ]
            }
        }
    
    @staticmethod
    def get_pattern_validation_scenarios():
        """Get pattern validation scenarios."""
        return {
            'valid_patterns': [
                {
                    'pattern_id': 'valid_svo',
                    'vertices': [
                        {'id': 0, 'pos': 'NOUN', 'dep': 'nsubj'},
                        {'id': 1, 'pos': 'VERB', 'dep': 'ROOT'},
                        {'id': 2, 'pos': 'NOUN', 'dep': 'dobj'}
                    ],
                    'edges': [
                        {'source': 0, 'target': 1, 'relation': 'subject'},
                        {'source': 1, 'target': 2, 'relation': 'object'}
                    ],
                    'validation_result': True
                }
            ],
            'invalid_patterns': [
                {
                    'pattern_id': 'missing_vertices',
                    'edges': [{'source': 0, 'target': 1}],
                    'validation_error': 'Missing vertices field'
                },
                {
                    'pattern_id': 'invalid_edge_reference',
                    'vertices': [{'id': 0, 'pos': 'NOUN'}],
                    'edges': [{'source': 0, 'target': 5}],  # Invalid target
                    'validation_error': 'Edge references non-existent vertex'
                }
            ]
        }
    
    @staticmethod
    def get_performance_test_parameters():
        """Get performance test parameters."""
        return {
            'small_text_test': {
                'text_length': 100,  # characters
                'num_sentences': 5,
                'expected_processing_time_ms': 100,
                'max_memory_mb': 20
            },
            'medium_text_test': {
                'text_length': 1000,
                'num_sentences': 20,
                'expected_processing_time_ms': 500,
                'max_memory_mb': 100
            },
            'large_text_test': {
                'text_length': 10000,
                'num_sentences': 100,
                'expected_processing_time_ms': 2000,
                'max_memory_mb': 300
            },
            'stress_test': {
                'text_length': 100000,
                'num_sentences': 1000,
                'expected_processing_time_ms': 10000,
                'max_memory_mb': 1000
            }
        }
    
    @staticmethod
    def get_linguistic_feature_tests():
        """Get linguistic feature test cases."""
        return {
            'pos_tagging_tests': [
                {
                    'sentence': "The quick brown fox jumps.",
                    'expected_pos': ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB']
                },
                {
                    'sentence': "She happily sings beautiful songs.",
                    'expected_pos': ['PRON', 'ADV', 'VERB', 'ADJ', 'NOUN']
                }
            ],
            'dependency_parsing_tests': [
                {
                    'sentence': "The cat sits on the mat.",
                    'expected_dependencies': [
                        ('cat', 'sits', 'nsubj'),
                        ('The', 'cat', 'det'),
                        ('on', 'sits', 'prep'),
                        ('mat', 'on', 'pobj'),
                        ('the', 'mat', 'det')
                    ]
                }
            ],
            'named_entity_tests': [
                {
                    'sentence': "Apple Inc. is located in Cupertino, California.",
                    'expected_entities': [
                        {'text': 'Apple Inc.', 'label': 'ORG'},
                        {'text': 'Cupertino', 'label': 'GPE'},
                        {'text': 'California', 'label': 'GPE'}
                    ]
                }
            ]
        }
    
    @staticmethod
    def get_error_handling_test_cases():
        """Get error handling test cases."""
        return {
            'empty_input_tests': [
                {'input': '', 'expected_behavior': 'return_empty_result'},
                {'input': None, 'expected_error': 'ValueError'},
                {'input': '   ', 'expected_behavior': 'return_empty_result'}
            ],
            'malformed_input_tests': [
                {
                    'input': 'This is not a complete...',
                    'expected_behavior': 'best_effort_processing'
                },
                {
                    'input': 'Text with émojis 🚀 and spéciàl chars',
                    'expected_behavior': 'handle_unicode'
                }
            ],
            'model_failure_tests': [
                {
                    'scenario': 'spacy_model_not_found',
                    'expected_error': 'ModelLoadError',
                    'fallback_behavior': 'use_default_model'
                },
                {
                    'scenario': 'parsing_timeout',
                    'expected_behavior': 'return_partial_results'
                }
            ]
        }
    
    @staticmethod
    def get_integration_test_scenarios():
        """Get integration test scenarios."""
        return {
            'full_pipeline_test': {
                'input_text': "The researchers at MIT developed a new algorithm that improves machine learning accuracy.",
                'expected_steps': [
                    'text_preprocessing',
                    'spacy_processing',
                    'metavertex_creation',
                    'pattern_matching',
                    'result_compilation'
                ],
                'expected_outputs': {
                    'entities': ['researchers', 'MIT', 'algorithm', 'machine learning'],
                    'patterns': ['research_activity', 'institutional_affiliation', 'improvement_relation'],
                    'metavertices_count': 15
                }
            },
            'batch_processing_test': {
                'input_texts': [
                    "Sentence one for testing.",
                    "Another sentence with different structure.",
                    "The third sentence contains named entities like Google."
                ],
                'expected_behavior': 'process_all_successfully',
                'performance_requirement': 'complete_within_1_second'
            }
        }
    
    @staticmethod
    def get_configuration_options():
        """Get configuration options for testing."""
        return {
            'processing_options': {
                'enable_entity_recognition': True,
                'enable_dependency_parsing': True,
                'enable_pos_tagging': True,
                'max_sentence_length': 1000,
                'timeout_seconds': 30
            },
            'pattern_matching_options': {
                'fuzzy_matching': False,
                'case_sensitive': False,
                'ignore_punctuation': True,
                'similarity_threshold': 0.8
            },
            'output_options': {
                'include_confidence_scores': True,
                'include_processing_time': False,
                'verbose_logging': False,
                'return_intermediate_results': False
            }
        }