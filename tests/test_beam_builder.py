import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.BeamBuilder import BeamBuilder


class TestBeamBuilder(unittest.TestCase):
    """Test the BeamBuilder class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_embedding_helper = Mock()
        self.beam_builder = BeamBuilder(self.mock_embedding_helper)

    def test_initialization(self):
        """Test BeamBuilder initialization"""
        self.assertEqual(self.beam_builder.embedding_helper, self.mock_embedding_helper)
        
        # Check that the relation maps are properly initialized
        self.assertIsInstance(self.beam_builder.asymmetric_pairs_map, dict)
        self.assertIsInstance(self.beam_builder.symmetric_pairs_map, dict)
        
        # Check some expected mappings
        self.assertEqual(self.beam_builder.asymmetric_pairs_map["hypernyms"], "hyponyms")
        self.assertEqual(self.beam_builder.asymmetric_pairs_map["hyponyms"], "hypernyms")
        self.assertEqual(self.beam_builder.symmetric_pairs_map["hypernyms"], "hypernyms")

    def test_asymmetric_pairs_map_content(self):
        """Test asymmetric pairs mapping contains expected relations"""
        expected_pairs = {
            "part_holonyms": "part_meronyms",
            "substance_holonyms": "substance_meronyms", 
            "member_holonyms": "member_meronyms",
            "part_meronyms": "part_holonyms",
            "substance_meronyms": "substance_holonyms",
            "member_meronyms": "member_holonyms",
            "hypernyms": "hyponyms",
            "hyponyms": "hypernyms"
        }
        
        for key, value in expected_pairs.items():
            self.assertEqual(self.beam_builder.asymmetric_pairs_map[key], value)

    def test_symmetric_pairs_map_content(self):
        """Test symmetric pairs mapping contains expected relations"""
        expected_symmetric = [
            "part_holonyms", "substance_holonyms", "member_holonyms",
            "part_meronyms", "substance_meronyms", "member_meronyms",
            "hypernyms", "hyponyms", "entailments", "causes", 
            "also_sees", "verb_groups"
        ]
        
        for relation in expected_symmetric:
            self.assertEqual(self.beam_builder.symmetric_pairs_map[relation], relation)

    def test_get_new_beams_basic(self):
        """Test basic functionality of get_new_beams"""
        # Mock graph
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        mock_graph.add_node("animal.n.01")
        mock_graph.add_edge("cat.n.01", "animal.n.01")
        mock_graph.add_edge("dog.n.01", "animal.n.01")
        
        # Mock synsets
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        # Mock embedding helper responses
        mock_src_embeddings = {
            "hypernyms": [("animal.n.01", 0.9)]
        }
        mock_tgt_embeddings = {
            "hypernyms": [("animal.n.01", 0.8)]
        }
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            mock_src_embeddings, mock_tgt_embeddings
        ]
        
        mock_asymm_pairs = [
            (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.85)
        ]
        mock_symm_pairs = [
            (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.75)
        ]
        
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            mock_asymm_pairs, mock_symm_pairs
        ]
        
        # Mock WordNet synset creation
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            result = self.beam_builder.get_new_beams(
                mock_graph, "cat.n.01", "dog.n.01", mock_model, beam_width=3
            )
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)  # Combined asymm and symm results
            
            # Check that results are sorted by relatedness (descending)
            relatedness_scores = [item[2] for item in result]
            self.assertEqual(relatedness_scores, sorted(relatedness_scores, reverse=True))

    def test_get_new_beams_empty_embeddings(self):
        """Test get_new_beams with empty embeddings"""
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        # Empty embeddings
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [{}, {}]
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.return_value = []
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            result = self.beam_builder.get_new_beams(
                mock_graph, "cat.n.01", "dog.n.01", mock_model
            )
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 0)

    def test_get_new_beams_validation_error(self):
        """Test get_new_beams raises error when graph neighbors don't match embeddings"""
        # Graph with limited neighbors
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        # No edges, so no neighbors
        
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        # Embeddings claim to have neighbors that aren't in the graph
        mock_src_embeddings = {
            "hypernyms": [("animal.n.01", 0.9)]  # Not a neighbor in graph
        }
        mock_tgt_embeddings = {}
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            mock_src_embeddings, mock_tgt_embeddings
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            with self.assertRaises(ValueError) as context:
                self.beam_builder.get_new_beams(
                    mock_graph, "cat.n.01", "dog.n.01", mock_model
                )
            
            self.assertIn("Not all lexical properties", str(context.exception))
            self.assertIn("in graph", str(context.exception))

    def test_get_new_beams_beam_width_limiting(self):
        """Test get_new_beams respects beam_width parameter"""
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        mock_graph.add_node("animal.n.01")
        mock_graph.add_edge("cat.n.01", "animal.n.01")
        mock_graph.add_edge("dog.n.01", "animal.n.01")
        
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        mock_src_embeddings = {"hypernyms": [("animal.n.01", 0.9)]}
        mock_tgt_embeddings = {"hypernyms": [("animal.n.01", 0.8)]}
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            mock_src_embeddings, mock_tgt_embeddings
        ]
        
        # Return many pairs (more than beam_width)
        many_asymm_pairs = [
            (("cat.n.01", "rel1"), ("dog.n.01", "rel1"), 0.9),
            (("cat.n.01", "rel2"), ("dog.n.01", "rel2"), 0.8),
            (("cat.n.01", "rel3"), ("dog.n.01", "rel3"), 0.7),
            (("cat.n.01", "rel4"), ("dog.n.01", "rel4"), 0.6),
        ]
        many_symm_pairs = [
            (("cat.n.01", "relA"), ("dog.n.01", "relA"), 0.85),
            (("cat.n.01", "relB"), ("dog.n.01", "relB"), 0.75),
        ]
        
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            many_asymm_pairs, many_symm_pairs
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            beam_width = 3
            result = self.beam_builder.get_new_beams(
                mock_graph, "cat.n.01", "dog.n.01", mock_model, beam_width=beam_width
            )
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), beam_width)
            
            # Should be sorted by relatedness (highest first)
            relatedness_scores = [item[2] for item in result]
            self.assertEqual(relatedness_scores, sorted(relatedness_scores, reverse=True))

    def test_get_new_beams_result_structure(self):
        """Test get_new_beams returns properly structured results"""
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        mock_graph.add_node("animal.n.01")
        mock_graph.add_edge("cat.n.01", "animal.n.01")
        mock_graph.add_edge("dog.n.01", "animal.n.01")
        
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        mock_src_embeddings = {"hypernyms": [("animal.n.01", 0.9)]}
        mock_tgt_embeddings = {"hypernyms": [("animal.n.01", 0.8)]}
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            mock_src_embeddings, mock_tgt_embeddings
        ]
        
        mock_pair = (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.85)
        
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            [mock_pair], []  # asymm, symm
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            result = self.beam_builder.get_new_beams(
                mock_graph, "cat.n.01", "dog.n.01", mock_model
            )
            
            self.assertEqual(len(result), 1)
            beam_item = result[0]
            
            # Should be tuple of ((synset1, rel), (synset2, rel), relatedness)
            self.assertIsInstance(beam_item, tuple)
            self.assertEqual(len(beam_item), 3)
            
            # First two elements should be tuples of (synset, relation)
            self.assertIsInstance(beam_item[0], tuple)
            self.assertIsInstance(beam_item[1], tuple)
            self.assertEqual(len(beam_item[0]), 2)
            self.assertEqual(len(beam_item[1]), 2)
            
            # Third element should be float (relatedness score)
            self.assertIsInstance(beam_item[2], (int, float))

    def test_get_new_beams_calls_embedding_helper_correctly(self):
        """Test get_new_beams calls embedding helper methods with correct parameters"""
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        mock_graph.add_edge("cat.n.01", "animal.n.01")
        mock_graph.add_edge("dog.n.01", "animal.n.01")
        
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        mock_model = Mock()
        
        mock_src_embeddings = {"hypernyms": [("animal.n.01", 0.9)]}
        mock_tgt_embeddings = {"hypernyms": [("animal.n.01", 0.8)]}
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            mock_src_embeddings, mock_tgt_embeddings
        ]
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.return_value = []
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            beam_width = 5
            result = self.beam_builder.get_new_beams(
                mock_graph, "cat.n.01", "dog.n.01", mock_model, beam_width=beam_width
            )
            
            # Check that embedding helper methods were called correctly
            self.assertEqual(self.mock_embedding_helper.embed_lexical_relations.call_count, 2)
            self.mock_embedding_helper.embed_lexical_relations.assert_any_call(mock_cat_synset, mock_model)
            self.mock_embedding_helper.embed_lexical_relations.assert_any_call(mock_dog_synset, mock_model)
            
            # Check alignment calls
            self.assertEqual(self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.call_count, 2)
            
            # First call should be for asymmetric pairs
            first_call_args = self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.call_args_list[0]
            self.assertEqual(first_call_args[0][0], self.beam_builder.asymmetric_pairs_map)
            
            # Second call should be for symmetric pairs
            second_call_args = self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.call_args_list[1]
            self.assertEqual(second_call_args[0][0], self.beam_builder.symmetric_pairs_map)


class TestBeamBuilderEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for BeamBuilder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_embedding_helper = Mock()
        self.beam_builder = BeamBuilder(self.mock_embedding_helper)

    def test_get_new_beams_with_zero_beam_width(self):
        """Test get_new_beams with beam_width=0"""
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [{}, {}]
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.return_value = []
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            result = self.beam_builder.get_new_beams(
                mock_graph, "cat.n.01", "dog.n.01", mock_model, beam_width=0
            )
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 0)

    def test_get_new_beams_identical_scores(self):
        """Test get_new_beams when multiple pairs have identical scores"""
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        mock_graph.add_node("animal.n.01")
        mock_graph.add_edge("cat.n.01", "animal.n.01")
        mock_graph.add_edge("dog.n.01", "animal.n.01")
        
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        mock_src_embeddings = {"hypernyms": [("animal.n.01", 0.9)]}
        mock_tgt_embeddings = {"hypernyms": [("animal.n.01", 0.8)]}
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            mock_src_embeddings, mock_tgt_embeddings
        ]
        
        # Multiple pairs with same score
        identical_score_pairs = [
            (("cat.n.01", "rel1"), ("dog.n.01", "rel1"), 0.8),
            (("cat.n.01", "rel2"), ("dog.n.01", "rel2"), 0.8),
            (("cat.n.01", "rel3"), ("dog.n.01", "rel3"), 0.8),
        ]
        
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            identical_score_pairs, []
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            result = self.beam_builder.get_new_beams(
                mock_graph, "cat.n.01", "dog.n.01", mock_model, beam_width=2
            )
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            # All returned items should have the same score
            scores = [item[2] for item in result]
            self.assertTrue(all(score == 0.8 for score in scores))

    def test_get_new_beams_partial_neighbor_validation_error(self):
        """Test get_new_beams when only some embeddings fail validation"""
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        mock_graph.add_node("animal.n.01")
        mock_graph.add_edge("cat.n.01", "animal.n.01")  # Only cat has animal as neighbor
        
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        # Both synsets claim animal as neighbor, but only cat has it in graph
        mock_src_embeddings = {"hypernyms": [("animal.n.01", 0.9)]}  # Valid
        mock_tgt_embeddings = {"hypernyms": [("animal.n.01", 0.8)]}  # Invalid - dog doesn't have animal as neighbor
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            mock_src_embeddings, mock_tgt_embeddings
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            with self.assertRaises(ValueError) as context:
                self.beam_builder.get_new_beams(
                    mock_graph, "cat.n.01", "dog.n.01", mock_model
                )
            
            error_msg = str(context.exception)
            self.assertIn("Not all lexical properties", error_msg)
            self.assertIn("dog.n.01", error_msg)
            self.assertIn("in graph", error_msg)


class TestBeamBuilderIntegration(unittest.TestCase):
    """Integration tests for BeamBuilder with realistic scenarios"""
    
    def setUp(self):
        """Set up for integration testing"""
        self.mock_embedding_helper = Mock()
        self.beam_builder = BeamBuilder(self.mock_embedding_helper)

    def test_realistic_wordnet_scenario(self):
        """Test BeamBuilder with a realistic WordNet-like graph structure"""
        # Create a more realistic graph structure
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
        
        # Mock realistic embeddings
        cat_embeddings = {
            "hypernyms": [("mammal.n.01", 0.9)],
            "also_sees": [("pet.n.01", 0.7)]
        }
        dog_embeddings = {
            "hypernyms": [("mammal.n.01", 0.85)], 
            "also_sees": [("pet.n.01", 0.75)]
        }
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            cat_embeddings, dog_embeddings
        ]
        
        # Mock realistic pair alignments
        asymm_pairs = [
            (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.87)
        ]
        symm_pairs = [
            (("cat.n.01", "also_sees"), ("dog.n.01", "also_sees"), 0.72)
        ]
        
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            asymm_pairs, symm_pairs
        ]
        
        # Mock synsets
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            result = self.beam_builder.get_new_beams(
                graph, "cat.n.01", "dog.n.01", mock_model, beam_width=5
            )
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            
            # Check that hypernym relationship scored higher than also_sees
            self.assertEqual(result[0][2], 0.87)  # hypernyms score (should be first)
            self.assertEqual(result[1][2], 0.72)  # also_sees score (should be second)

    def test_complex_graph_with_many_relations(self):
        """Test BeamBuilder with a complex graph having many relation types"""
        # Create a complex graph
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
        
        # Mock complex embeddings with multiple relation types
        cat_embeddings = {
            "hypernyms": [("mammal.n.01", 0.95)],
            "part_holonyms": [],  # No part relations for cat
            "similar_tos": []     # No similar relations
        }
        
        dog_embeddings = {
            "hypernyms": [("mammal.n.01", 0.92)],
            "part_holonyms": [],  # No part relations for dog
            "similar_tos": []     # No similar relations
        }
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            cat_embeddings, dog_embeddings
        ]
        
        # Mock alignment with various relation types
        strong_asymm_pairs = [
            (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.93)
        ]
        strong_symm_pairs = [
            (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.88)
        ]
        
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            strong_asymm_pairs, strong_symm_pairs
        ]
        
        mock_cat_synset = Mock()
        mock_dog_synset = Mock()
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = Mock()
            result = self.beam_builder.get_new_beams(
                graph, "cat.n.01", "dog.n.01", mock_model, beam_width=3
            )
            
            self.assertIsInstance(result, list)
            self.assertLessEqual(len(result), 3)  # Respects beam width
            
            # Verify results are properly sorted
            if len(result) > 1:
                scores = [item[2] for item in result]
                self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == '__main__':
    unittest.main()