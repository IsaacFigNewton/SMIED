"""
Configuration class containing mock constants and test data for SemanticMetagraph tests.
"""


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