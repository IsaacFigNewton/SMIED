"""
Configuration class containing mock constants and test data for PatternMatcher tests.
"""


class PatternMatcherMockConfig:
    """Configuration class containing mock constants and test data for PatternMatcher tests."""
    
    @staticmethod
    def get_mock_semantic_graph_structures():
        """Get mock semantic graph structures with consistent vertex patterns."""
        return {
            'simple_svo_graph': {
                'vertices': [
                    ("cat", {"pos": "NOUN", "dep": "nsubj", "ent_type": ""}),
                    ("chases", {"pos": "VERB", "dep": "ROOT", "ent_type": ""}),
                    ("mouse", {"pos": "NOUN", "dep": "dobj", "ent_type": ""})
                ],
                'edges': [
                    (("cat", "chases"), {"relation": "subject"}),
                    (("chases", "mouse"), {"relation": "object"})
                ]
            },
            'complex_sentence_graph': {
                'vertices': [
                    ("John", {"pos": "PROPN", "dep": "nsubj", "ent_type": "PERSON"}),
                    ("quickly", {"pos": "ADV", "dep": "advmod", "ent_type": ""}),
                    ("runs", {"pos": "VERB", "dep": "ROOT", "ent_type": ""}),
                    ("through", {"pos": "ADP", "dep": "prep", "ent_type": ""}),
                    ("the", {"pos": "DET", "dep": "det", "ent_type": ""}),
                    ("park", {"pos": "NOUN", "dep": "pobj", "ent_type": "LOCATION"})
                ],
                'edges': [
                    (("John", "runs"), {"relation": "subject"}),
                    (("quickly", "runs"), {"relation": "modifier"}),
                    (("runs", "through"), {"relation": "preposition"}),
                    (("through", "park"), {"relation": "object"}),
                    (("the", "park"), {"relation": "determiner"})
                ]
            },
            'nested_clause_graph': {
                'vertices': [
                    ("Mary", {"pos": "PROPN", "dep": "nsubj", "ent_type": "PERSON"}),
                    ("believes", {"pos": "VERB", "dep": "ROOT", "ent_type": ""}),
                    ("that", {"pos": "SCONJ", "dep": "mark", "ent_type": ""}),
                    ("cats", {"pos": "NOUN", "dep": "nsubj", "ent_type": ""}),
                    ("are", {"pos": "AUX", "dep": "cop", "ent_type": ""}),
                    ("intelligent", {"pos": "ADJ", "dep": "acomp", "ent_type": ""})
                ],
                'edges': [
                    (("Mary", "believes"), {"relation": "subject"}),
                    (("believes", "intelligent"), {"relation": "complement"}),
                    (("that", "intelligent"), {"relation": "marker"}),
                    (("cats", "intelligent"), {"relation": "subject"}),
                    (("are", "intelligent"), {"relation": "copula"})
                ]
            }
        }
    
    @staticmethod
    def get_pattern_loader_test_patterns():
        """Get pattern loader test patterns with categories and descriptions."""
        return {
            'basic_patterns': [
                {
                    "pattern_id": "svo_basic",
                    "description": "Basic Subject-Verb-Object",
                    "category": "grammatical",
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
                {
                    "pattern_id": "prepositional_phrase",
                    "description": "Prepositional phrase attachment",
                    "category": "syntactic",
                    "vertices": [
                        {"id": 0, "pos": "VERB", "dep": "ROOT"},
                        {"id": 1, "pos": "ADP", "dep": "prep"},
                        {"id": 2, "pos": "NOUN", "dep": "pobj"}
                    ],
                    "edges": [
                        {"source": 0, "target": 1, "relation": "preposition"},
                        {"source": 1, "target": 2, "relation": "object"}
                    ]
                }
            ],
            'semantic_patterns': [
                {
                    "pattern_id": "agent_action_patient",
                    "description": "Agent performs action on patient",
                    "category": "semantic",
                    "vertices": [
                        {"id": 0, "semantic_role": "agent", "labels": ["PERSON", "ANIMAL"]},
                        {"id": 1, "semantic_role": "action", "pos": "VERB"},
                        {"id": 2, "semantic_role": "patient", "labels": ["OBJECT", "PERSON"]}
                    ],
                    "edges": [
                        {"source": 0, "target": 1, "relation": "agent"},
                        {"source": 1, "target": 2, "relation": "patient"}
                    ]
                },
                {
                    "pattern_id": "location_relation", 
                    "description": "Entity at location",
                    "category": "semantic",
                    "vertices": [
                        {"id": 0, "labels": ["PERSON", "OBJECT"]},
                        {"id": 1, "pos": "ADP", "lemma": ["at", "in", "on"]},
                        {"id": 2, "labels": ["LOCATION", "PLACE"]}
                    ],
                    "edges": [
                        {"source": 0, "target": 1, "relation": "location_prep"},
                        {"source": 1, "target": 2, "relation": "location"}
                    ]
                }
            ],
            'complex_patterns': [
                {
                    "pattern_id": "causative_construction",
                    "description": "X causes Y to Z",
                    "category": "causative",
                    "vertices": [
                        {"id": 0, "semantic_role": "causer"},
                        {"id": 1, "pos": "VERB", "semantic_type": "causative"},
                        {"id": 2, "semantic_role": "causee"},
                        {"id": 3, "pos": "VERB", "semantic_type": "result"}
                    ],
                    "edges": [
                        {"source": 0, "target": 1, "relation": "agent"},
                        {"source": 1, "target": 2, "relation": "causee"},
                        {"source": 1, "target": 3, "relation": "result"}
                    ]
                }
            ]
        }
    
    @staticmethod
    def get_realistic_test_scenarios():
        """Get realistic test scenarios with large metavertex structures."""
        return {
            'news_article_scenario': {
                'text': "Apple Inc. announced quarterly earnings yesterday. The tech giant reported record profits.",
                'expected_patterns': [
                    "entity_announcement",
                    "temporal_reference",
                    "corporate_reporting"
                ],
                'expected_entities': [
                    {"text": "Apple Inc.", "label": "ORG"},
                    {"text": "yesterday", "label": "DATE"},
                    {"text": "tech giant", "label": "ORG"}
                ],
                'expected_relations': [
                    ("Apple Inc.", "announced", "earnings"),
                    ("tech giant", "reported", "profits")
                ]
            },
            'academic_text_scenario': {
                'text': "Researchers at Stanford University developed a new machine learning algorithm that improves accuracy.",
                'expected_patterns': [
                    "research_development",
                    "institutional_affiliation",
                    "improvement_relation"
                ],
                'expected_entities': [
                    {"text": "Researchers", "label": "PERSON"},
                    {"text": "Stanford University", "label": "ORG"},
                    {"text": "machine learning algorithm", "label": "PRODUCT"}
                ]
            },
            'narrative_scenario': {
                'text': "The brave knight rescued the princess from the dragon in the castle.",
                'expected_patterns': [
                    "hero_rescue",
                    "agent_action_patient",
                    "location_relation"
                ],
                'expected_semantic_roles': {
                    'knight': 'agent',
                    'rescued': 'action', 
                    'princess': 'patient',
                    'dragon': 'source',
                    'castle': 'location'
                }
            }
        }
    
    @staticmethod
    def get_pos_tag_patterns():
        """Get common POS tag patterns and relation types."""
        return {
            'noun_phrase_patterns': [
                ["DET", "ADJ", "NOUN"],
                ["DET", "NOUN", "NOUN"],  # Compound nouns
                ["PROPN", "PROPN"],      # Proper noun sequences
                ["NUM", "NOUN"]          # Quantified nouns
            ],
            'verb_phrase_patterns': [
                ["AUX", "VERB"],         # Auxiliary + main verb
                ["ADV", "VERB"],         # Adverb + verb
                ["VERB", "PART"],        # Phrasal verbs
                ["VERB", "ADP", "NOUN"]  # Verb + prepositional phrase
            ],
            'clause_patterns': [
                ["NOUN", "VERB", "NOUN"],           # SVO
                ["NOUN", "VERB", "ADJ"],            # SVC
                ["NOUN", "VERB", "NOUN", "NOUN"],   # SVOO
                ["NOUN", "AUX", "VERB", "ADP", "NOUN"]  # Passive with agent
            ],
            'dependency_relations': {
                'core_relations': ['nsubj', 'dobj', 'iobj', 'ccomp', 'xcomp'],
                'modifier_relations': ['amod', 'advmod', 'nmod', 'acl'],
                'function_relations': ['det', 'aux', 'cop', 'mark'],
                'coordination': ['conj', 'cc'],
                'clausal': ['advcl', 'relcl', 'csubj', 'csubjpass']
            }
        }
    
    @staticmethod
    def get_pattern_matching_algorithms():
        """Get different pattern matching algorithm configurations."""
        return {
            'exact_matching': {
                'algorithm': 'exact_match',
                'parameters': {
                    'case_sensitive': False,
                    'ignore_determiners': True,
                    'match_threshold': 1.0
                }
            },
            'fuzzy_matching': {
                'algorithm': 'fuzzy_match',
                'parameters': {
                    'similarity_threshold': 0.8,
                    'edit_distance_max': 2,
                    'semantic_similarity': True
                }
            },
            'structural_matching': {
                'algorithm': 'graph_isomorphism',
                'parameters': {
                    'node_match_function': 'attribute_similarity',
                    'edge_match_function': 'relation_similarity',
                    'allow_partial_match': True
                }
            },
            'semantic_matching': {
                'algorithm': 'semantic_similarity',
                'parameters': {
                    'use_word_embeddings': True,
                    'use_wordnet_similarity': True,
                    'similarity_weight': 0.6
                }
            }
        }
    
    @staticmethod
    def get_performance_test_configurations():
        """Get performance test configurations."""
        return {
            'small_scale_test': {
                'num_patterns': 10,
                'num_graphs': 100,
                'max_graph_size': 20,
                'expected_time_ms': 1000
            },
            'medium_scale_test': {
                'num_patterns': 50,
                'num_graphs': 500,
                'max_graph_size': 100,
                'expected_time_ms': 10000
            },
            'large_scale_test': {
                'num_patterns': 200,
                'num_graphs': 1000,
                'max_graph_size': 500,
                'expected_time_ms': 60000
            }
        }
    
    @staticmethod
    def get_edge_case_scenarios():
        """Get edge case scenarios for robust testing."""
        return {
            'empty_inputs': {
                'empty_graph': {'vertices': [], 'edges': []},
                'empty_patterns': [],
                'expected_behavior': 'return_empty_results'
            },
            'malformed_patterns': [
                {
                    'pattern': {'vertices': [{'id': 0}]},  # Missing required fields
                    'expected_error': 'PatternValidationError'
                },
                {
                    'pattern': {'vertices': [], 'edges': [{'source': 0, 'target': 1}]},  # Edge without vertices
                    'expected_error': 'PatternValidationError'
                }
            ],
            'complex_nested_structures': {
                'deeply_nested_clauses': 5,
                'recursive_patterns': True,
                'coordination_chains': 10,
                'expected_behavior': 'handle_gracefully'
            },
            'ambiguous_matches': {
                'multiple_valid_matches': True,
                'overlapping_patterns': True,
                'ranking_criteria': ['pattern_specificity', 'match_confidence'],
                'expected_behavior': 'return_ranked_results'
            }
        }
    
    @staticmethod
    def get_validation_criteria():
        """Get validation criteria for pattern matching results."""
        return {
            'match_quality_metrics': [
                'precision',
                'recall', 
                'f1_score',
                'pattern_coverage',
                'false_positive_rate'
            ],
            'structural_validation': {
                'vertex_count_match': True,
                'edge_count_match': True,
                'graph_connectivity': True,
                'attribute_consistency': True
            },
            'semantic_validation': {
                'role_consistency': True,
                'type_compatibility': True,
                'relation_validity': True,
                'context_appropriateness': True
            }
        }
    
    @staticmethod
    def get_integration_test_data():
        """Get integration test data for end-to-end testing."""
        return {
            'pattern_loader_integration': {
                'pattern_files': ['basic_patterns.json', 'semantic_patterns.json'],
                'expected_loaded_patterns': 25,
                'validation_required': True
            },
            'semantic_graph_integration': {
                'input_texts': [
                    "The cat sits on the mat.",
                    "John gives Mary a book.",
                    "The students study hard for the exam."
                ],
                'expected_graph_features': ['pos_tags', 'dependencies', 'entities'],
                'pattern_matching_required': True
            },
            'full_pipeline_integration': {
                'input': "Complex sentence with multiple clauses that should match various patterns.",
                'expected_steps': [
                    'text_parsing',
                    'graph_construction',
                    'pattern_loading',
                    'pattern_matching',
                    'result_ranking',
                    'output_formatting'
                ]
            }
        }