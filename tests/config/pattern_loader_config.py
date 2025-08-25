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
    def get_pattern_management_test_data():
        """Get test data for pattern management operations."""
        return {
            'add_pattern_scenarios': {
                'new_category': {
                    'name': 'new_pattern',
                    'pattern': [{"text": "test"}],
                    'description': 'New pattern',
                    'category': 'new_category'
                },
                'existing_category': {
                    'name': 'another_pattern',
                    'pattern': [{"pos": ["NOUN"]}],
                    'description': 'Another pattern',
                    'category': 'existing'
                },
                'default_category': {
                    'name': 'default_pattern',
                    'pattern': [{"text": "default"}],
                    'description': '',
                    'category': 'custom'  # Default category
                }
            },
            'string_representation_data': {
                "converted": "data"
            }
        }
    
    @staticmethod
    def get_error_scenarios():
        """Get error scenario test data."""
        return {
            'file_errors': {
                'file_not_found': 'nonexistent.json',
                'permission_error': 'readonly.json',
                'malformed_json': '{"incomplete": json',
                'io_error': 'io_error_file.json'
            },
            'pattern_errors': {
                'invalid_structure': 'not_a_dict',
                'missing_fields': {'pattern': []},  # missing description
                'wrong_type': 123
            },
            'conversion_errors': {
                'deep_nesting': 100,  # levels of nesting
                'circular_reference': True,
                'memory_limit': '1GB'
            }
        }
    
    @staticmethod
    def get_default_pattern_test_data():
        """Get test data for default pattern operations."""
        return {
            'mock_resource_patterns': {
                "lexical": {"basic_word": {"description": "Basic word", "pattern": []}},
                "simple_semantic": {"action": {"description": "Action", "pattern": []}}
            },
            'pattern_files': ["lexical", "simple_semantic", "complex_semantic", "domain_specific", 
                           "metavertex_basic", "metavertex_semantic", "metavertex_complex"],
            'expected_fallback': {"default": {"pattern": {}}}
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
    def get_basic_test_data():
        """Get basic test data for PatternLoader functionality tests."""
        return {
            'simple_patterns': {
                "basic": {
                    "simple_svo": {
                        "description": "Simple SVO pattern",
                        "pattern": [
                            {"pos": ["NOUN"], "dep": ["nsubj"]},
                            {"pos": ["VERB"], "dep": ["ROOT"]},
                            {"pos": ["NOUN"], "dep": ["dobj"]}
                        ]
                    }
                }
            },
            'json_test_data': {
                "category": {
                    "pattern1": [
                        {"pos": ["NOUN", "VERB"], "text": "test"},
                        {"relation": ["subject"], "other": "value"}
                    ]
                }
            },
            'set_conversion_data': {
                "category": {
                    "pattern1": {
                        "description": "Test description",
                        "pattern": [
                            {"pos": {"NOUN", "VERB"}, "text": "test"},
                            {"relation": {"subject"}, "other": "value"}
                        ]
                    }
                }
            }
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get validation test data."""
        return {
            'valid_pattern_data': {
                "test_category": {
                    "valid_pattern": {
                        "description": "Valid pattern",
                        "pattern": [
                            {"pos": ["NOUN"], "text": "word"}
                        ]
                    }
                }
            },
            'invalid_pattern_data': {
                "test_category": {
                    "invalid_pattern": "not_a_dict"
                }
            },
            'convertible_keys_test': {
                "category": {
                    "pattern1": [
                        {
                            "pos": ["NOUN", "VERB"],         # Should convert
                            "root_type": ["entity"],         # Should convert  
                            "labels": ["person", "animal"],  # Should convert
                            "relation_type": ["subject"],    # Should convert
                            "other_list": ["keep", "as", "list"],  # Should NOT convert
                            "string_value": "unchanged"      # Should remain unchanged
                        }
                    ]
                }
            }
        }
    
    @staticmethod
    def get_edge_case_scenarios():
        """Get edge case test data."""
        return {
            'empty_patterns': {},
            'malformed_json': '{"incomplete": json',
            'missing_description': {
                "category": {
                    "pattern1": {
                        "pattern": [{"text": "test"}]
                        # Missing "description" key
                    }
                }
            },
            'non_list_patterns': {
                "category": {
                    "pattern1": "not_a_list"
                }
            },
            'large_pattern_params': {
                'num_categories': 10,
                'patterns_per_category': 20,
                'items_per_pattern': 5,
                'words_per_item': 10
            },
            'nested_patterns': {
                "complex": {
                    "nested_pattern": {
                        "description": "Complex nested pattern",
                        "pattern": [
                            {
                                "text": ["surface_text"],
                                "nested": {
                                    "deep": {
                                        "pos": ["NOUN"],  # This should NOT be converted (not at top level)
                                        "very_deep": ["value"]
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
    
    @staticmethod
    def get_integration_test_data():
        """Get integration test data."""
        return {
            'workflow_patterns': {
                "workflow": {
                    "test_pattern": {
                        "description": "Workflow test",
                        "pattern": [{"pos": ["NOUN"], "text": "test"}]
                    }
                }
            },
            'realistic_semantic_patterns': {
                "semantic": {
                    "subject_verb_object": {
                        "description": "Basic SVO pattern",
                        "pattern": [
                            {"pos": ["NOUN", "PROPN"], "dep": ["nsubj"]},
                            {"pos": ["VERB"], "dep": ["ROOT"]},
                            {"pos": ["NOUN", "PROPN"], "dep": ["dobj", "pobj"]}
                        ]
                    },
                    "entity_relation": {
                        "description": "Entity relation pattern",
                        "pattern": [
                            {"ent_type": ["PERSON", "ORG"], "pos": ["NOUN"]},
                            {"relation_type": ["works_for", "part_of"]},
                            {"ent_type": ["ORG", "PLACE"], "pos": ["NOUN"]}
                        ]
                    }
                },
                "syntactic": {
                    "noun_phrase": {
                        "description": "Noun phrase pattern",
                        "pattern": [
                            {"pos": ["DET"], "optional": True},
                            {"pos": ["ADJ"], "optional": True, "repeat": True},
                            {"pos": ["NOUN", "PROPN"]}
                        ]
                    }
                }
            },
            'temp_filenames': [
                "temp_test_patterns.json",
                "test_workflow_patterns.json",
                "integration_test_output.json"
            ]
        }

    @staticmethod
    def get_file_operation_test_data():
        """Get file operation test data."""
        return {
            'valid_json_content': {
                "test_category": {
                    "test_pattern": {
                        "description": "File test pattern",
                        "pattern": [{"pos": ["NOUN"], "text": "test"}]
                    }
                }
            },
            'mock_file_paths': {
                'existing_file': 'test_patterns.json',
                'nonexistent_file': 'nonexistent.json',
                'output_file': 'output.json',
                'malformed_file': 'malformed.json'
            },
            'default_patterns_data': {
                "default": {
                    "default_pattern": {
                        "description": "Default pattern",
                        "pattern": [{"text": "default"}]
                    }
                }
            }
        }