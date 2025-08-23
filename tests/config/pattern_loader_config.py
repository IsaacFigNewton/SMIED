"""
Configuration class containing mock constants and test data for PatternLoader tests.
"""

import json


class PatternLoaderMockConfig:
    """Configuration class containing mock constants and test data for PatternLoader tests."""
    
    @staticmethod
    def get_sample_pattern_structures():
        """Get sample pattern structures with consistent format."""
        return {
            'simple_pattern': {
                "pattern_id": "simple_svo",
                "description": "Subject-Verb-Object pattern",
                "vertices": [
                    {"id": 0, "pos": "NOUN", "dep": "nsubj"},
                    {"id": 1, "pos": "VERB", "dep": "ROOT"},
                    {"id": 2, "pos": "NOUN", "dep": "dobj"}
                ],
                "edges": [
                    {"source": 0, "target": 1, "relation": "subject"},
                    {"source": 1, "target": 2, "relation": "object"}
                ]
            },
            'complex_pattern': {
                "pattern_id": "complex_clause",
                "description": "Complex clause with modifiers",
                "vertices": [
                    {"id": 0, "pos": "NOUN", "dep": "nsubj", "labels": ["PERSON"]},
                    {"id": 1, "pos": "VERB", "dep": "ROOT", "tense": "PAST"},
                    {"id": 2, "pos": "NOUN", "dep": "dobj", "labels": ["OBJECT"]},
                    {"id": 3, "pos": "ADV", "dep": "advmod"},
                    {"id": 4, "pos": "ADP", "dep": "prep"},
                    {"id": 5, "pos": "NOUN", "dep": "pobj", "labels": ["LOCATION"]}
                ],
                "edges": [
                    {"source": 0, "target": 1, "relation": "agent"},
                    {"source": 1, "target": 2, "relation": "patient"},
                    {"source": 1, "target": 3, "relation": "manner"},
                    {"source": 1, "target": 4, "relation": "prep"},
                    {"source": 4, "target": 5, "relation": "location"}
                ]
            },
            'nested_pattern': {
                "pattern_id": "nested_clause",
                "description": "Pattern with nested clauses",
                "vertices": [
                    {"id": 0, "pos": "NOUN", "dep": "nsubj"},
                    {"id": 1, "pos": "VERB", "dep": "ROOT"},
                    {"id": 2, "pos": "SCONJ", "dep": "mark"},
                    {"id": 3, "pos": "NOUN", "dep": "nsubj", "clause": "subordinate"},
                    {"id": 4, "pos": "VERB", "dep": "advcl", "clause": "subordinate"}
                ],
                "edges": [
                    {"source": 0, "target": 1, "relation": "agent"},
                    {"source": 2, "target": 4, "relation": "marker"},
                    {"source": 3, "target": 4, "relation": "sub_agent"},
                    {"source": 1, "target": 4, "relation": "adverbial"}
                ]
            }
        }
    
    @staticmethod
    def get_file_operation_test_data():
        """Get file operation test data."""
        return {
            'valid_json_content': {
                "patterns": [
                    {
                        "pattern_id": "test_pattern_1",
                        "vertices": [{"id": 0, "pos": "NOUN"}],
                        "edges": []
                    },
                    {
                        "pattern_id": "test_pattern_2", 
                        "vertices": [{"id": 0, "pos": "VERB"}],
                        "edges": []
                    }
                ]
            },
            'invalid_json_content': '{"patterns": [{"pattern_id": "broken", "vertices": [}',
            'empty_file_content': '',
            'malformed_structure': {
                "not_patterns": [{"wrong_key": "value"}]
            },
            'missing_required_fields': {
                "patterns": [
                    {"vertices": [{"id": 0}]},  # missing pattern_id
                    {"pattern_id": "incomplete"}  # missing vertices
                ]
            }
        }
    
    @staticmethod
    def get_pattern_validation_test_cases():
        """Get pattern validation test cases."""
        return {
            'valid_patterns': [
                {
                    "pattern_id": "valid_1",
                    "vertices": [{"id": 0, "pos": "NOUN"}, {"id": 1, "pos": "VERB"}],
                    "edges": [{"source": 0, "target": 1, "relation": "subject"}]
                },
                {
                    "pattern_id": "valid_2",
                    "vertices": [{"id": 0, "pos": "ADJ"}],
                    "edges": []
                }
            ],
            'invalid_patterns': [
                {
                    "pattern_id": "missing_vertices"
                    # Missing vertices field
                },
                {
                    "vertices": [{"id": 0, "pos": "NOUN"}],
                    "edges": []
                    # Missing pattern_id
                },
                {
                    "pattern_id": "invalid_edge",
                    "vertices": [{"id": 0, "pos": "NOUN"}],
                    "edges": [{"source": 0, "target": 1, "relation": "invalid"}]
                    # Edge references non-existent vertex
                },
                {
                    "pattern_id": "duplicate_vertex_id",
                    "vertices": [
                        {"id": 0, "pos": "NOUN"},
                        {"id": 0, "pos": "VERB"}  # Duplicate id
                    ],
                    "edges": []
                }
            ]
        }
    
    @staticmethod
    def get_convertible_key_lists():
        """Get convertible key lists for testing."""
        return {
            'standard_keys': ["pos", "root_type", "labels", "relation_type"],
            'extended_keys': ["pos", "dep", "ent_type", "lemma", "shape"],
            'minimal_keys': ["pos"],
            'custom_keys': ["semantic_role", "syntactic_function", "discourse_marker"]
        }
    
    @staticmethod
    def get_large_pattern_generation_parameters():
        """Get parameters for large pattern generation."""
        return {
            'small_pattern': {
                'num_vertices': 10,
                'num_edges': 15,
                'complexity': 'simple'
            },
            'medium_pattern': {
                'num_vertices': 50,
                'num_edges': 100,
                'complexity': 'moderate'
            },
            'large_pattern': {
                'num_vertices': 200,
                'num_edges': 500,
                'complexity': 'complex'
            },
            'stress_test_pattern': {
                'num_vertices': 1000,
                'num_edges': 2000,
                'complexity': 'maximum'
            }
        }
    
    @staticmethod
    def get_realistic_semantic_patterns():
        """Get realistic semantic patterns with SVO structures."""
        return {
            'basic_transitive': {
                "pattern_id": "basic_transitive",
                "description": "Basic transitive sentence",
                "vertices": [
                    {"id": 0, "pos": "NOUN", "semantic_role": "agent", "labels": ["PERSON", "ANIMAL"]},
                    {"id": 1, "pos": "VERB", "semantic_role": "action", "transitivity": "transitive"},
                    {"id": 2, "pos": "NOUN", "semantic_role": "patient", "labels": ["OBJECT", "PERSON"]}
                ],
                "edges": [
                    {"source": 0, "target": 1, "relation": "agent", "weight": 1.0},
                    {"source": 1, "target": 2, "relation": "patient", "weight": 1.0}
                ]
            },
            'ditransitive': {
                "pattern_id": "ditransitive",
                "description": "Ditransitive sentence with recipient",
                "vertices": [
                    {"id": 0, "pos": "NOUN", "semantic_role": "agent"},
                    {"id": 1, "pos": "VERB", "semantic_role": "action", "transitivity": "ditransitive"},
                    {"id": 2, "pos": "NOUN", "semantic_role": "recipient"},
                    {"id": 3, "pos": "NOUN", "semantic_role": "theme"}
                ],
                "edges": [
                    {"source": 0, "target": 1, "relation": "agent"},
                    {"source": 1, "target": 2, "relation": "recipient"},
                    {"source": 1, "target": 3, "relation": "theme"}
                ]
            },
            'causative': {
                "pattern_id": "causative",
                "description": "Causative construction",
                "vertices": [
                    {"id": 0, "pos": "NOUN", "semantic_role": "causer"},
                    {"id": 1, "pos": "VERB", "semantic_role": "causative_action"},
                    {"id": 2, "pos": "NOUN", "semantic_role": "causee"},
                    {"id": 3, "pos": "VERB", "semantic_role": "caused_action"}
                ],
                "edges": [
                    {"source": 0, "target": 1, "relation": "causer"},
                    {"source": 1, "target": 2, "relation": "causee"},
                    {"source": 1, "target": 3, "relation": "caused_event"}
                ]
            }
        }
    
    @staticmethod
    def get_file_path_test_scenarios():
        """Get file path test scenarios."""
        return {
            'valid_paths': [
                "patterns/simple_patterns.json",
                "patterns/complex_patterns.json",
                "data/semantic_patterns.json",
                "/absolute/path/to/patterns.json"
            ],
            'invalid_paths': [
                "nonexistent/file.json",
                "",  # Empty path
                None,  # None path
                "not_json_file.txt",
                "patterns/",  # Directory instead of file
            ],
            'special_characters': [
                "patterns/spéciàl_chärs.json",
                "patterns/with spaces.json",
                "patterns/with-dashes.json",
                "patterns/with_underscores.json"
            ]
        }
    
    @staticmethod
    def get_performance_benchmarks():
        """Get performance benchmarks for pattern loading."""
        return {
            'small_file': {
                'patterns_count': 10,
                'max_load_time_ms': 100,
                'max_memory_mb': 10
            },
            'medium_file': {
                'patterns_count': 100,
                'max_load_time_ms': 1000,
                'max_memory_mb': 50
            },
            'large_file': {
                'patterns_count': 1000,
                'max_load_time_ms': 5000,
                'max_memory_mb': 200
            }
        }
    
    @staticmethod
    def get_pattern_category_examples():
        """Get pattern category examples."""
        return {
            'grammatical_patterns': {
                'svo': "Subject-Verb-Object",
                'svc': "Subject-Verb-Complement",
                'svoo': "Subject-Verb-Object-Object",
                'passive': "Passive construction"
            },
            'semantic_patterns': {
                'agent_action': "Agent performing action",
                'possession': "Possessive relationship",
                'location': "Locative relationship",
                'temporal': "Temporal relationship"
            },
            'discourse_patterns': {
                'question': "Question pattern",
                'conditional': "Conditional statement",
                'comparison': "Comparative structure",
                'contrast': "Contrastive structure"
            }
        }
    
    @staticmethod
    def get_error_recovery_scenarios():
        """Get error recovery scenarios for testing."""
        return {
            'partial_corruption': {
                'description': 'File with some corrupted patterns',
                'valid_patterns': 5,
                'corrupted_patterns': 2,
                'expected_behavior': 'load_valid_skip_invalid'
            },
            'encoding_issues': {
                'description': 'File with encoding problems',
                'encoding': 'latin-1',
                'expected_encoding': 'utf-8',
                'expected_behavior': 'attempt_encoding_detection'
            },
            'large_file_timeout': {
                'description': 'Very large file causing timeout',
                'file_size_mb': 100,
                'timeout_ms': 30000,
                'expected_behavior': 'timeout_gracefully'
            }
        }

    @staticmethod
    def get_sample_patterns():
        """Get sample pattern structure for basic testing."""
        return {
            "test_category": {
                "test_pattern": {
                    "description": "Test pattern",
                    "pattern": [
                        {"text": ["cat"], "pos": ["NOUN"]},
                        {"relation": ["subject"]}
                    ]
                }
            }
        }

    @staticmethod
    def get_json_conversion_test_data():
        """Get test data for JSON pattern conversion."""
        return {
            'basic_conversion': {
                "test_category": {
                    "test_pattern": {
                        "description": "Basic conversion test",
                        "pattern": [
                            {"pos": ["NOUN"], "labels": ["PERSON"]},
                            {"pos": ["VERB"], "root_type": ["LEXICAL"]},
                            {"relation": ["agent"], "relation_type": ["SEMANTIC"]}
                        ]
                    }
                }
            },
            'complex_conversion': {
                "complex_category": {
                    "complex_pattern": {
                        "description": "Complex pattern conversion",
                        "pattern": [
                            {"pos": ["NOUN", "PROPN"], "labels": ["PERSON", "ORG"]},
                            {"pos": ["VERB"], "tense": ["PAST"], "root_type": ["ACTION"]},
                            {"pos": ["NOUN"], "labels": ["OBJECT"]},
                            {"relation": ["nsubj"], "relation_type": ["GRAMMATICAL"]},
                            {"relation": ["dobj"], "relation_type": ["GRAMMATICAL"]}
                        ]
                    }
                }
            }
        }