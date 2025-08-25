"""
Configuration class containing mock constants and test data for SemanticMetagraph tests.
Follows SMIED Testing Framework Design Specifications for configuration-driven test data management.
"""

from typing import Dict, Any, List, Tuple


class SemanticMetagraphMockConfig:
    """Configuration class containing mock constants and test data for SemanticMetagraph tests."""
    
    @staticmethod
    def get_test_texts():
        """Get test texts for semantic analysis."""
        return {
            'simple_sentence': "The cat runs fast.",
            'complex_sentence': "Apple is a technology company based in California.",
            'multi_sentence': "The quick brown fox jumps over the lazy dog. It was a beautiful day.",
            'entity_rich': "John works at Microsoft in Seattle and drives a Tesla.",
            'temporal_text': "Yesterday, Mary visited the museum and saw ancient artifacts.",
            'question_text': "Where did the bird fly when it left the tree?",
            'passive_voice': "The book was written by the famous author.",
            'compound_sentence': "The sun was shining, and the birds were singing happily.",
            'conditional_text': "If it rains tomorrow, we will stay inside.",
            'negation_text': "The dog did not run in the park today."
        }
    
    @staticmethod
    def get_spacy_model_constants():
        """Get spaCy model name constants."""
        return {
            'default_model': 'en_core_web_sm',
            'large_model': 'en_core_web_lg',
            'medium_model': 'en_core_web_md',
            'test_model': 'en_core_web_test'  # For testing purposes
        }
    
    @staticmethod
    def get_expected_vertex_structures():
        """Get expected vertex structures for different text types."""
        return {
            'simple_sentence_vertices': [
                ("The", {"pos": "DET", "dep": "det", "ent_type": ""}),
                ("cat", {"pos": "NOUN", "dep": "nsubj", "ent_type": ""}),
                ("runs", {"pos": "VERB", "dep": "ROOT", "ent_type": ""}),
                ("fast", {"pos": "ADV", "dep": "advmod", "ent_type": ""}),
                (".", {"pos": "PUNCT", "dep": "punct", "ent_type": ""})
            ],
            'entity_rich_vertices': [
                ("John", {"pos": "PROPN", "dep": "nsubj", "ent_type": "PERSON"}),
                ("works", {"pos": "VERB", "dep": "ROOT", "ent_type": ""}),
                ("at", {"pos": "ADP", "dep": "prep", "ent_type": ""}),
                ("Microsoft", {"pos": "PROPN", "dep": "pobj", "ent_type": "ORG"}),
                ("in", {"pos": "ADP", "dep": "prep", "ent_type": ""}),
                ("Seattle", {"pos": "PROPN", "dep": "pobj", "ent_type": "GPE"}),
                ("Tesla", {"pos": "PROPN", "dep": "dobj", "ent_type": "ORG"})
            ]
        }
    
    @staticmethod
    def get_expected_semantic_structures():
        """Get expected semantic structures for validation."""
        return {
            'apple_company_structure': {
                'entities': [
                    {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
                    {"text": "California", "label": "GPE", "start": 40, "end": 50}
                ],
                'relations': [
                    {"source": "Apple", "relation": "is", "target": "company"},
                    {"source": "company", "relation": "based_in", "target": "California"}
                ],
                'pos_tags': ["PROPN", "AUX", "DET", "NOUN", "NOUN", "VERB", "ADP", "PROPN", "PUNCT"]
            },
            'fox_sentence_structure': {
                'adjectives': ["quick", "brown", "lazy"],
                'nouns': ["fox", "dog"],
                'verbs': ["jumps"],
                'prepositions': ["over"],
                'determiners': ["The", "the"]
            }
        }
    
    @staticmethod
    def get_entity_handling_scenarios():
        """Get entity handling scenarios for testing."""
        return {
            'person_entities': {
                'text': "Barack Obama was the president of the United States.",
                'expected_entities': [
                    {"text": "Barack Obama", "label": "PERSON"},
                    {"text": "United States", "label": "GPE"}
                ]
            },
            'organization_entities': {
                'text': "Google and Microsoft compete in the technology sector.",
                'expected_entities': [
                    {"text": "Google", "label": "ORG"},
                    {"text": "Microsoft", "label": "ORG"}
                ]
            },
            'location_entities': {
                'text': "The meeting will be held in New York at the UN building.",
                'expected_entities': [
                    {"text": "New York", "label": "GPE"},
                    {"text": "UN", "label": "ORG"}
                ]
            },
            'temporal_entities': {
                'text': "The conference starts on Monday, January 15th, 2024.",
                'expected_entities': [
                    {"text": "Monday", "label": "DATE"},
                    {"text": "January 15th, 2024", "label": "DATE"}
                ]
            }
        }
    
    @staticmethod
    def get_punctuation_handling_scenarios():
        """Get punctuation handling scenarios."""
        return {
            'basic_punctuation': {
                'text': "Hello, world! How are you?",
                'expected_tokens': ["Hello", ",", "world", "!", "How", "are", "you", "?"],
                'punctuation_tokens': [",", "!", "?"]
            },
            'complex_punctuation': {
                'text': 'He said, "This is amazing!" Then he left.',
                'expected_tokens': ["He", "said", ",", '"', "This", "is", "amazing", "!", '"', "Then", "he", "left", "."],
                'punctuation_tokens': [",", '"', "!", '"', "."]
            },
            'abbreviations': {
                'text': "Dr. Smith works at U.S.A. Corp.",
                'expected_tokens': ["Dr.", "Smith", "works", "at", "U.S.A.", "Corp", "."],
                'abbreviation_tokens': ["Dr.", "U.S.A."]
            }
        }
    
    @staticmethod
    def get_dependency_parsing_scenarios():
        """Get dependency parsing scenarios."""
        return {
            'simple_dependencies': {
                'text': "The cat sits on the mat.",
                'expected_dependencies': [
                    ("cat", "sits", "nsubj"),
                    ("sits", "sits", "ROOT"),
                    ("mat", "sits", "pobj"),
                    ("The", "cat", "det"),
                    ("on", "sits", "prep"),
                    ("the", "mat", "det")
                ]
            },
            'complex_dependencies': {
                'text': "The student who studies hard will succeed.",
                'expected_dependencies': [
                    ("student", "succeed", "nsubj"),
                    ("succeed", "succeed", "ROOT"),
                    ("studies", "student", "relcl"),
                    ("who", "studies", "nsubj"),
                    ("hard", "studies", "advmod"),
                    ("will", "succeed", "aux")
                ]
            }
        }
    
    @staticmethod
    def get_integration_test_scenarios():
        """Get integration test scenarios for realistic testing."""
        return {
            'news_article_excerpt': {
                'text': "The tech giant announced its quarterly earnings yesterday. Shares rose by 5% in after-hours trading.",
                'expected_analysis': {
                    'entities': ["tech giant", "quarterly earnings", "yesterday", "5%"],
                    'key_verbs': ["announced", "rose"],
                    'temporal_expressions': ["yesterday", "quarterly", "after-hours"]
                }
            },
            'scientific_abstract': {
                'text': "Researchers at MIT developed a new algorithm that improves machine learning accuracy by 15%.",
                'expected_analysis': {
                    'entities': ["Researchers", "MIT", "15%"],
                    'technical_terms': ["algorithm", "machine learning", "accuracy"],
                    'key_actions': ["developed", "improves"]
                }
            },
            'narrative_text': {
                'text': "Once upon a time, there was a brave knight who lived in a castle near the forest.",
                'expected_analysis': {
                    'entities': ["knight", "castle", "forest"],
                    'descriptors': ["brave"],
                    'temporal_markers': ["Once upon a time"],
                    'spatial_relations': ["in", "near"]
                }
            }
        }
    
    @staticmethod
    def get_error_handling_scenarios():
        """Get error handling scenarios for testing."""
        return {
            'empty_text': {
                'text': "",
                'expected_behavior': "return_empty_graph"
            },
            'whitespace_only': {
                'text': "   \n\t  ",
                'expected_behavior': "return_empty_graph"
            },
            'very_long_text': {
                'text': "This is a sentence. " * 1000,  # Very long text
                'expected_behavior': "handle_gracefully"
            },
            'special_characters': {
                'text': "Text with émojis 🚀 and spéciàl characters ñoño",
                'expected_behavior': "process_unicode_correctly"
            },
            'malformed_sentences': {
                'text': "This is not a complete... and this one too...",
                'expected_behavior': "best_effort_parsing"
            }
        }
    
    @staticmethod
    def get_performance_benchmarks():
        """Get performance benchmarks for testing."""
        return {
            'small_text': {
                'text_length': 100,  # characters
                'max_processing_time_ms': 100,
                'max_memory_mb': 10
            },
            'medium_text': {
                'text_length': 1000,
                'max_processing_time_ms': 500,
                'max_memory_mb': 50
            },
            'large_text': {
                'text_length': 10000,
                'max_processing_time_ms': 2000,
                'max_memory_mb': 200
            }
        }
    
    @staticmethod
    def get_linguistic_feature_examples():
        """Get examples of various linguistic features."""
        return {
            'coordination': {
                'text': "John and Mary went to the store and bought apples and oranges.",
                'expected_coordinations': [
                    ("John", "Mary", "and"),
                    ("apples", "oranges", "and"),
                    ("went", "bought", "and")
                ]
            },
            'subordination': {
                'text': "Although it was raining, we decided to go hiking.",
                'expected_subordination': {
                    'main_clause': "we decided to go hiking",
                    'subordinate_clause': "it was raining",
                    'subordinator': "Although"
                }
            },
            'modification': {
                'text': "The extremely tall basketball player scored easily.",
                'expected_modifications': [
                    ("tall", "extremely", "advmod"),
                    ("player", "tall", "amod"),
                    ("player", "basketball", "compound"),
                    ("scored", "easily", "advmod")
                ]
            }
        }

    @staticmethod
    def get_basic_test_vertex_structures():
        """Get basic vertex structures for testing."""
        return {
            'simple_vertices': [
                ("word1", {"pos": "NOUN"}),
                ("word2", {"pos": "VERB"}),
                ("word3", {"pos": "ADJ"})
            ],
            'complex_vertices': [
                ("Apple", {"pos": "PROPN", "ent_type": "ORG"}),
                ("technology", {"pos": "NOUN", "ent_type": ""}),
                ("company", {"pos": "NOUN", "ent_type": ""})
            ]
        }

    @staticmethod
    def get_basic_test_edge_structures():
        """Get basic edge structures for testing."""
        return {
            'simple_edges': [
                ((0, 1), {"relation": "subject"}),
                ((1, 2), {"relation": "object"}),
                ((0, 2), {"relation": "modifier"})
            ],
            'complex_edges': [
                ((0, 1), {"relation": "nsubj", "weight": 1.0}),
                ((1, 2), {"relation": "compound", "weight": 0.8}),
                ((2, 3), {"relation": "dobj", "weight": 0.9})
            ]
        }
    
    @staticmethod
    def get_validation_test_scenarios():
        """Get validation test scenarios for testing graph and vertex validation."""
        return {
            'valid_graphs': {
                'simple_valid': [
                    ("word1", {"pos": "NOUN"}),
                    ("word2", {"pos": "VERB"}),
                    ((0, 1), {"relation": "subject"})
                ],
                'complex_valid': [
                    ("Apple", {"pos": "PROPN", "ent_type": "ORG"}),
                    ("is", {"pos": "AUX"}),
                    ("company", {"pos": "NOUN"}),
                    ((0, 1), {"relation": "nsubj"}),
                    ((1, 2), {"relation": "attr"})
                ]
            },
            'invalid_graphs': {
                'invalid_vertex_reference': [
                    ("word1", {"pos": "NOUN"}),
                    ((0, 2), {"relation": "subject"})  # References non-existent vertex 2
                ],
                'missing_relation_metadata': [
                    ("word1", {"pos": "NOUN"}),
                    ("word2", {"pos": "VERB"}),
                    ((0, 1), {})  # Missing relation metadata
                ],
                'invalid_vertex_structure': [
                    ("word1", {"pos": "NOUN"}),
                    (123, {"pos": "VERB"})  # Invalid vertex type
                ]
            },
            'edge_case_graphs': {
                'empty_graph': [],
                'single_vertex': [("single", {"pos": "NOUN"})],
                'undirected_relations': [
                    ("word1", {"pos": "NOUN"}),
                    ("word2", {"pos": "VERB"}),
                    ([0, 1], {"relation": "coordination"})
                ]
            }
        }
    
    @staticmethod
    def get_edge_case_test_scenarios():
        """Get edge case test scenarios for robust testing."""
        return {
            'empty_inputs': {
                'empty_doc': None,
                'empty_vert_list': [],
                'empty_text': "",
                'whitespace_only_text': "   \n\t  "
            },
            'malformed_data': {
                'invalid_json': '{"malformed": json}',
                'incomplete_metaverts': [
                    ("word1",),  # Missing required metadata
                    ((0,), {"relation": "incomplete"})  # Incomplete tuple relation
                ],
                'circular_references': [
                    ("word1", {"pos": "NOUN"}),
                    ("word2", {"pos": "VERB"}),
                    ((1, 0), {"relation": "dep1"}),
                    ((0, 1), {"relation": "dep2"})
                ]
            },
            'boundary_conditions': {
                'very_long_text': "This is a sentence." * 100,
                'special_characters': "Text with émojis 🚀 and spéciàl characters ñoño",
                'unicode_text': "Test with unicode: α β γ δ ε 中文 العربية",
                'large_graph': [("word_{}".format(i), {"pos": "NOUN"}) for i in range(1000)]
            },
            'memory_intensive': {
                'deep_nesting': {
                    'description': 'Deeply nested graph structures',
                    'max_depth': 50,
                    'node_count': 500
                },
                'wide_branching': {
                    'description': 'Wide branching graph structures',
                    'branch_factor': 100,
                    'levels': 5
                }
            }
        }
    
    @staticmethod
    def get_integration_test_data():
        """Get integration test data for multi-component testing."""
        return {
            'spacy_integration': {
                'sample_texts': [
                    "Apple Inc. is based in Cupertino, California.",
                    "The quick brown fox jumps over the lazy dog.",
                    "Natural language processing is a fascinating field."
                ],
                'expected_token_counts': [8, 9, 7],
                'expected_entity_counts': [3, 0, 0]
            },
            'networkx_integration': {
                'expected_node_attributes': ['label', 'text', 'pos', 'ent_type'],
                'expected_edge_attributes': ['relation', 'type', 'weight'],
                'graph_properties': {
                    'is_directed': True,
                    'is_multigraph': False,
                    'allows_self_loops': True
                }
            },
            'json_serialization': {
                'required_fields': ['metaverts'],
                'metavert_types': ['atomic', 'directed', 'undirected'],
                'expected_structure': {
                    'atomic': ['id', 'type', 'value', 'metadata'],
                    'directed': ['id', 'type', 'source', 'target', 'metadata'],
                    'undirected': ['id', 'type', 'nodes', 'metadata']
                }
            },
            'full_pipeline': {
                'input_text': "Microsoft develops software products.",
                'expected_pipeline_stages': [
                    'spacy_processing',
                    'metavert_creation',
                    'relation_extraction',
                    'entity_linking',
                    'graph_construction'
                ],
                'expected_outputs': {
                    'tokens': 4,
                    'entities': 1,
                    'relations': 3,
                    'metaverts': 8
                }
            }
        }
    
    @staticmethod
    def get_mock_factory_configurations():
        """Get configurations for mock factory usage."""
        return {
            'basic_mocks': {
                'MockSemanticMetagraph': {
                    'default_args': [],
                    'default_kwargs': {'doc': None, 'vert_list': None},
                    'common_configurations': {
                        'with_doc': {'doc': 'mock_spacy_doc'},
                        'with_vert_list': {'vert_list': []},
                        'empty': {}
                    }
                },
                'MockSpacyDoc': {
                    'default_args': [],
                    'default_kwargs': {'text': "Apple is a technology company."},
                    'common_configurations': {
                        'simple_text': {'text': "The cat runs fast."},
                        'complex_text': {'text': "Apple Inc. is based in California."},
                        'empty_text': {'text': ""}
                    }
                }
            },
            'validation_mocks': {
                'MockSemanticMetagraphValidation': {
                    'default_behavior': 'pass_validation',
                    'failure_modes': ['validation_error', 'vertex_error', 'canonicalization_error'],
                    'configurations': {
                        'strict_validation': {'strict_mode': True},
                        'lenient_validation': {'strict_mode': False}
                    }
                }
            },
            'edge_case_mocks': {
                'MockSemanticMetagraphEdgeCases': {
                    'available_cases': ['empty_input', 'invalid_data', 'memory_limit', 'malformed_json'],
                    'default_case': 'empty_input',
                    'configurations': {
                        'empty_scenario': {'edge_case_type': 'empty_input'},
                        'error_scenario': {'edge_case_type': 'invalid_data'},
                        'memory_scenario': {'edge_case_type': 'memory_limit'}
                    }
                }
            },
            'integration_mocks': {
                'MockSemanticMetagraphIntegration': {
                    'required_components': ['wordnet_integration', 'knowledge_base', 'reasoning_engine', 'spacy_nlp'],
                    'pipeline_stages': ['preprocessing', 'analysis', 'enrichment', 'postprocessing'],
                    'configurations': {
                        'minimal_integration': {'components': ['spacy_nlp']},
                        'full_integration': {'components': ['wordnet_integration', 'knowledge_base', 'reasoning_engine', 'spacy_nlp']}
                    }
                }
            }
        }
    
    @staticmethod
    def get_test_assertion_data():
        """Get data for test assertions and expected outcomes."""
        return {
            'basic_functionality': {
                'initialization': {
                    'from_doc_assertions': ['has_metaverts', 'doc_stored', 'proper_structure'],
                    'from_vert_list_assertions': ['correct_count', 'proper_indexing', 'no_doc'],
                    'empty_initialization_assertions': ['empty_metaverts', 'no_doc', 'zero_index']
                },
                'vertex_operations': {
                    'add_vert_assertions': ['increased_count', 'proper_indexing', 'correct_structure'],
                    'remove_vert_assertions': ['decreased_count', 'removed_relations', 'maintained_integrity']
                },
                'conversion_operations': {
                    'to_json_assertions': ['has_metaverts_field', 'valid_json', 'proper_structure'],
                    'from_json_assertions': ['equivalent_structure', 'proper_reconstruction'],
                    'to_nx_assertions': ['is_digraph', 'has_nodes', 'has_edges', 'proper_attributes']
                }
            },
            'validation_testing': {
                'graph_validation': {
                    'valid_graph_assertions': ['validation_passes', 'no_exceptions'],
                    'invalid_graph_assertions': ['validation_fails', 'proper_exception', 'descriptive_message']
                },
                'vertex_validation': {
                    'valid_vertex_assertions': ['proper_structure', 'correct_metadata', 'valid_references'],
                    'invalid_vertex_assertions': ['validation_error', 'specific_error_type']
                }
            },
            'edge_case_testing': {
                'empty_input_assertions': ['handles_gracefully', 'returns_empty_structure', 'no_crashes'],
                'malformed_data_assertions': ['proper_error_handling', 'informative_messages', 'safe_failure'],
                'boundary_condition_assertions': ['maintains_performance', 'memory_efficient', 'scalable_behavior']
            },
            'integration_testing': {
                'spacy_integration_assertions': ['proper_token_extraction', 'entity_recognition', 'dependency_parsing'],
                'networkx_integration_assertions': ['graph_structure', 'node_attributes', 'edge_attributes'],
                'full_pipeline_assertions': ['end_to_end_functionality', 'component_interaction', 'data_flow_integrity']
            }
        }
    
    @staticmethod
    def get_performance_benchmarks():
        """Get performance benchmarks for testing with updated thresholds."""
        return {
            'small_text_processing': {
                'text_length': 100,
                'max_processing_time_ms': 50,
                'max_memory_mb': 5,
                'expected_metavert_count': 25
            },
            'medium_text_processing': {
                'text_length': 1000,
                'max_processing_time_ms': 200,
                'max_memory_mb': 20,
                'expected_metavert_count': 250
            },
            'large_text_processing': {
                'text_length': 10000,
                'max_processing_time_ms': 1000,
                'max_memory_mb': 100,
                'expected_metavert_count': 2500
            },
            'graph_operations': {
                'vertex_addition_time_us': 10,
                'vertex_removal_time_us': 20,
                'json_serialization_time_ms': 50,
                'networkx_conversion_time_ms': 100
            }
        }