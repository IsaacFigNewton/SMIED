"""
Configuration class containing mock constants and test data for DirectedMetagraph tests.
"""


class DirectedMetagraphMockConfig:
    """Configuration class containing mock constants and test data for DirectedMetagraph tests."""
    
    @staticmethod
    def get_basic_vertex_structures():
        """Get basic vertex structures for testing."""
        return {
            'simple_vertices': [
                ("word1", {"pos": "NOUN"}),
                ("word2", {"pos": "VERB"}),
                ("word3", {"pos": "ADJ"})
            ],
            'person_action_vertices': [
                ("John", {"type": "person", "pos": "NOUN"}),
                ("runs", {"type": "action", "pos": "VERB"}),
                ("quickly", {"type": "manner", "pos": "ADV"})
            ],
            'svo_structure': [
                ("cat", {"pos": "NOUN", "role": "subject"}),
                ("chases", {"pos": "VERB", "role": "predicate"}),
                ("mouse", {"pos": "NOUN", "role": "object"})
            ]
        }
    
    @staticmethod
    def get_edge_structures():
        """Get edge structures for testing."""
        return {
            'simple_edges': [
                ((0, 1), {"relation": "subject"}),
                ((1, 2), {"relation": "object"}),
                ((0, 2), {"relation": "modifier"})
            ],
            'semantic_edges': [
                (("John", "runs"), {"relation": "agent", "weight": 0.9}),
                (("runs", "quickly"), {"relation": "manner", "weight": 0.7}),
                (("cat", "mouse"), {"relation": "target", "weight": 0.8})
            ],
            'complex_relations': [
                ((0, 1), {"relation": "nsubj", "dependency": "grammatical"}),
                ((1, 2), {"relation": "dobj", "dependency": "grammatical"}),
                ((2, 3), {"relation": "prep", "dependency": "prepositional"})
            ]
        }
    
    @staticmethod
    def get_graph_manipulation_scenarios():
        """Get scenarios for graph manipulation testing."""
        return {
            'addition_scenario': {
                'initial_vertices': [("A", {}), ("B", {})],
                'initial_edges': [((0, 1), {})],
                'vertices_to_add': [("C", {"new": True})],
                'edges_to_add': [((1, 2), {"relation": "new_edge"})]
            },
            'removal_scenario': {
                'initial_vertices': [("A", {}), ("B", {}), ("C", {}), ("D", {})],
                'initial_edges': [((0, 1), {}), ((1, 2), {}), ((2, 3), {})],
                'vertices_to_remove': ["C"],
                'edges_to_remove': [((1, 2), {})]
            },
            'update_scenario': {
                'initial_vertices': [("word", {"pos": "NOUN"})],
                'updated_attributes': {"pos": "VERB", "updated": True}
            }
        }
    
    @staticmethod
    def get_entity_relation_patterns():
        """Get entity-relation patterns for testing."""
        return {
            'person_organization': [
                ("John", {"type": "PERSON", "pos": "NOUN"}),
                ("works", {"type": "RELATION", "pos": "VERB"}),
                ("Apple", {"type": "ORG", "pos": "NOUN"})
            ],
            'location_relation': [
                ("Paris", {"type": "GPE", "pos": "NOUN"}),
                ("is_located_in", {"type": "RELATION", "pos": "VERB"}),
                ("France", {"type": "GPE", "pos": "NOUN"})
            ],
            'temporal_relation': [
                ("meeting", {"type": "EVENT", "pos": "NOUN"}),
                ("occurs_on", {"type": "TEMPORAL", "pos": "VERB"}),
                ("Monday", {"type": "DATE", "pos": "NOUN"})
            ]
        }
    
    @staticmethod
    def get_complex_graph_structures():
        """Get complex graph structures for advanced testing."""
        return {
            'hierarchical_structure': {
                'vertices': [
                    ("animal", {"level": 0, "type": "category"}),
                    ("mammal", {"level": 1, "type": "subcategory"}),
                    ("carnivore", {"level": 1, "type": "subcategory"}),
                    ("cat", {"level": 2, "type": "instance"}),
                    ("dog", {"level": 2, "type": "instance"})
                ],
                'edges': [
                    (("animal", "mammal"), {"relation": "hypernym"}),
                    (("animal", "carnivore"), {"relation": "hypernym"}),
                    (("mammal", "cat"), {"relation": "instance_of"}),
                    (("carnivore", "cat"), {"relation": "instance_of"}),
                    (("mammal", "dog"), {"relation": "instance_of"})
                ]
            },
            'cyclic_structure': {
                'vertices': [("A", {}), ("B", {}), ("C", {}), ("D", {})],
                'edges': [
                    (("A", "B"), {"relation": "next"}),
                    (("B", "C"), {"relation": "next"}),
                    (("C", "D"), {"relation": "next"}),
                    (("D", "A"), {"relation": "cycle_back"})
                ]
            }
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get data for validation testing."""
        return {
            'valid_vertices': [
                ("valid_word", {"pos": "NOUN", "lemma": "valid"}),
                ("another_valid", {"pos": "VERB", "tense": "present"})
            ],
            'invalid_vertices': [
                (None, {"pos": "NOUN"}),  # None vertex
                ("", {"pos": "VERB"}),    # Empty string vertex
                ("word", None)            # None attributes
            ],
            'valid_edges': [
                (("word1", "word2"), {"relation": "subject"}),
                (("word2", "word3"), {"relation": "object", "weight": 0.5})
            ],
            'invalid_edges': [
                ((None, "word"), {"relation": "invalid"}),  # None source
                (("word", None), {"relation": "invalid"}),  # None target
                (("word1", "word2"), None)                  # None attributes
            ]
        }
    
    @staticmethod
    def get_graph_traversal_scenarios():
        """Get scenarios for graph traversal testing."""
        return {
            'linear_path': {
                'vertices': [("start", {}), ("middle1", {}), ("middle2", {}), ("end", {})],
                'edges': [
                    (("start", "middle1"), {"step": 1}),
                    (("middle1", "middle2"), {"step": 2}),
                    (("middle2", "end"), {"step": 3})
                ],
                'expected_path': ["start", "middle1", "middle2", "end"]
            },
            'branching_path': {
                'vertices': [
                    ("root", {}), ("branch1", {}), ("branch2", {}), 
                    ("leaf1", {}), ("leaf2", {}), ("leaf3", {})
                ],
                'edges': [
                    (("root", "branch1"), {"type": "branch"}),
                    (("root", "branch2"), {"type": "branch"}),
                    (("branch1", "leaf1"), {"type": "leaf"}),
                    (("branch1", "leaf2"), {"type": "leaf"}),
                    (("branch2", "leaf3"), {"type": "leaf"})
                ]
            }
        }
    
    @staticmethod
    def get_performance_test_data():
        """Get data for performance testing with large graphs."""
        return {
            'large_graph_params': {
                'num_vertices': 1000,
                'num_edges': 2000,
                'vertex_prefix': 'vertex_',
                'edge_relation': 'connects'
            },
            'stress_test_params': {
                'num_operations': 10000,
                'operation_types': ['add_vertex', 'add_edge', 'remove_vertex', 'update_vertex']
            }
        }