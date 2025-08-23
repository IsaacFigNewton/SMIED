"""
Configuration class containing mock constants and test data for BeamBuilder tests.
"""


class BeamBuilderMockConfig:
    """Configuration class containing mock constants and test data."""
    
    @staticmethod
    def get_expected_asymmetric_pairs():
        """Get expected asymmetric pairs mapping."""
        return {
            "part_holonyms": "part_meronyms",
            "substance_holonyms": "substance_meronyms", 
            "member_holonyms": "member_meronyms",
            "part_meronyms": "part_holonyms",
            "substance_meronyms": "substance_holonyms",
            "member_meronyms": "member_holonyms",
            "hypernyms": "hyponyms",
            "hyponyms": "hypernyms"
        }
    
    @staticmethod
    def get_expected_symmetric_relations():
        """Get expected symmetric relations list."""
        return [
            "part_holonyms", "substance_holonyms", "member_holonyms",
            "part_meronyms", "substance_meronyms", "member_meronyms",
            "hypernyms", "hyponyms", "entailments", "causes", 
            "also_sees", "verb_groups"
        ]
    
    @staticmethod
    def get_basic_test_graph():
        """Get basic test graph for get_new_beams tests."""
        import networkx as nx
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        mock_graph.add_node("animal.n.01")
        mock_graph.add_edge("cat.n.01", "animal.n.01")
        mock_graph.add_edge("dog.n.01", "animal.n.01")
        return mock_graph
    
    @staticmethod
    def get_basic_test_embeddings():
        """Get basic test embeddings configuration."""
        return {
            'src_embeddings': {
                "hypernyms": [("animal.n.01", 0.9)]
            },
            'tgt_embeddings': {
                "hypernyms": [("animal.n.01", 0.8)]
            }
        }
    
    @staticmethod
    def get_basic_test_pairs():
        """Get basic test aligned pairs."""
        return {
            'asymm_pairs': [
                (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.85)
            ],
            'symm_pairs': [
                (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.75)
            ]
        }
    
    @staticmethod
    def get_empty_embeddings():
        """Get empty embeddings for testing edge cases."""
        return {
            'src_embeddings': {},
            'tgt_embeddings': {}
        }
    
    @staticmethod
    def get_validation_error_embeddings():
        """Get embeddings that will cause validation errors."""
        return {
            'src_embeddings': {
                "hypernyms": [("animal.n.01", 0.9)]  # Not a neighbor in graph
            },
            'tgt_embeddings': {}
        }
    
    @staticmethod
    def get_beam_width_test_pairs():
        """Get multiple pairs for beam width testing."""
        return {
            'asymm_pairs': [
                (("cat.n.01", "rel1"), ("dog.n.01", "rel1"), 0.9),
                (("cat.n.01", "rel2"), ("dog.n.01", "rel2"), 0.8),
                (("cat.n.01", "rel3"), ("dog.n.01", "rel3"), 0.7),
                (("cat.n.01", "rel4"), ("dog.n.01", "rel4"), 0.6),
            ],
            'symm_pairs': [
                (("cat.n.01", "relA"), ("dog.n.01", "relA"), 0.85),
                (("cat.n.01", "relB"), ("dog.n.01", "relB"), 0.75),
            ]
        }
    
    @staticmethod
    def get_identical_scores_pairs():
        """Get pairs with identical scores for testing."""
        return [
            (("cat.n.01", "rel1"), ("dog.n.01", "rel1"), 0.8),
            (("cat.n.01", "rel2"), ("dog.n.01", "rel2"), 0.8),
            (("cat.n.01", "rel3"), ("dog.n.01", "rel3"), 0.8),
        ]
    
    @staticmethod
    def get_realistic_wordnet_graph():
        """Get realistic WordNet-like graph structure."""
        import networkx as nx
        graph = nx.DiGraph()
        
        # Add nodes for a simple taxonomy
        nodes = ["cat.n.01", "dog.n.01", "animal.n.01", "mammal.n.01", "pet.n.01"]
        for node in nodes:
            graph.add_node(node)
        
        # Add hierarchical relationships
        graph.add_edge("cat.n.01", "mammal.n.01")    # cat -> mammal (hypernym)
        graph.add_edge("dog.n.01", "mammal.n.01")    # dog -> mammal (hypernym)
        graph.add_edge("mammal.n.01", "animal.n.01")  # mammal -> animal (hypernym)
        graph.add_edge("cat.n.01", "pet.n.01")       # cat -> pet (also)
        graph.add_edge("dog.n.01", "pet.n.01")       # dog -> pet (also)
        
        return graph
    
    @staticmethod
    def get_realistic_embeddings():
        """Get realistic embeddings for integration testing."""
        return {
            'cat_embeddings': {
                "hypernyms": [("mammal.n.01", 0.9)],
                "also_sees": [("pet.n.01", 0.7)]
            },
            'dog_embeddings': {
                "hypernyms": [("mammal.n.01", 0.85)], 
                "also_sees": [("pet.n.01", 0.75)]
            }
        }
    
    @staticmethod
    def get_realistic_aligned_pairs():
        """Get realistic aligned pairs for integration testing."""
        return {
            'asymm_pairs': [
                (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.87)
            ],
            'symm_pairs': [
                (("cat.n.01", "also_sees"), ("dog.n.01", "also_sees"), 0.72)
            ]
        }
    
    @staticmethod
    def get_complex_graph():
        """Get complex graph with many relation types."""
        import networkx as nx
        graph = nx.DiGraph()
        
        # Central synsets
        nodes = ["cat.n.01", "dog.n.01", "mammal.n.01", "animal.n.01", "vertebrate.n.01"]
        for node in nodes:
            graph.add_node(node)
        
        # Add various types of edges to simulate different WordNet relations
        edges = [
            ("cat.n.01", "mammal.n.01"),      # hypernym
            ("dog.n.01", "mammal.n.01"),      # hypernym
            ("mammal.n.01", "vertebrate.n.01"), # hypernym
            ("vertebrate.n.01", "animal.n.01"), # hypernym
        ]
        
        for src, tgt in edges:
            graph.add_edge(src, tgt)
        
        return graph
    
    @staticmethod
    def get_complex_embeddings():
        """Get complex embeddings with multiple relation types."""
        return {
            'cat_embeddings': {
                "hypernyms": [("mammal.n.01", 0.95)],
                "part_holonyms": [],  # No part relations for cat
                "similar_tos": []     # No similar relations
            },
            'dog_embeddings': {
                "hypernyms": [("mammal.n.01", 0.92)],
                "part_holonyms": [],  # No part relations for dog
                "similar_tos": []     # No similar relations
            }
        }
    
    @staticmethod
    def get_complex_aligned_pairs():
        """Get complex aligned pairs for testing."""
        return {
            'asymm_pairs': [
                (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.93)
            ],
            'symm_pairs': [
                (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.88)
            ]
        }
    
    @staticmethod
    def get_partial_neighbor_validation_graph():
        """Get graph for partial neighbor validation testing."""
        import networkx as nx
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        mock_graph.add_node("animal.n.01")
        mock_graph.add_edge("cat.n.01", "animal.n.01")  # Only cat has animal as neighbor
        return mock_graph
    
    @staticmethod
    def get_partial_neighbor_embeddings():
        """Get embeddings for partial neighbor validation testing."""
        return {
            'src_embeddings': {"hypernyms": [("animal.n.01", 0.9)]},  # Valid
            'tgt_embeddings': {"hypernyms": [("animal.n.01", 0.8)]}   # Invalid - dog doesn't have animal as neighbor
        }