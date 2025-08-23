import unittest
from unittest.mock import patch
import networkx as nx
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.BeamBuilder import BeamBuilder
from tests.mocks.beam_builder_mocks import BeamBuilderMockFactory
from tests.config.beam_builder_config import BeamBuilderMockConfig


class TestBeamBuilder(unittest.TestCase):
    """Test the BeamBuilder class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize mock factory and config
        self.mock_factory = BeamBuilderMockFactory()
        self.mock_config = BeamBuilderMockConfig()
        
        # Create embedding helper using factory
        self.mock_embedding_helper = self.mock_factory('MockEmbeddingHelper')
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
        expected_pairs = self.mock_config.get_expected_asymmetric_pairs()
        
        for key, value in expected_pairs.items():
            self.assertEqual(self.beam_builder.asymmetric_pairs_map[key], value)

    def test_symmetric_pairs_map_content(self):
        """Test symmetric pairs mapping contains expected relations"""
        expected_symmetric = self.mock_config.get_expected_symmetric_relations()
        
        for relation in expected_symmetric:
            self.assertEqual(self.beam_builder.symmetric_pairs_map[relation], relation)

    def test_get_new_beams_basic(self):
        """Test basic functionality of get_new_beams"""
        # Get test data from mock config
        mock_graph = self.mock_config.get_basic_test_graph()
        embeddings = self.mock_config.get_basic_test_embeddings()
        pairs = self.mock_config.get_basic_test_pairs()
        
        # Mock synsets using factory
        mock_cat_synset = self.mock_factory('MockSynsetForBeam', "cat.n.01", "feline mammal")
        mock_dog_synset = self.mock_factory('MockSynsetForBeam', "dog.n.01", "canine mammal")
        
        # Set up embedding helper responses using config data
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['src_embeddings'], embeddings['tgt_embeddings']
        ]
        
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            pairs['asymm_pairs'], pairs['symm_pairs']
        ]
        
        # Mock WordNet synset creation
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = self.mock_factory('MockEmbeddingModel')
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
        
        mock_cat_synset = self.mock_factory('MockSynsetForBeam', "cat.n.01", "feline mammal")
        mock_dog_synset = self.mock_factory('MockSynsetForBeam', "dog.n.01", "canine mammal")
        
        # Get empty embeddings from config
        embeddings = self.mock_config.get_empty_embeddings()
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['src_embeddings'], embeddings['tgt_embeddings']
        ]
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.return_value = []
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = self.mock_factory('MockEmbeddingModel')
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
        
        mock_cat_synset = self.mock_factory('MockSynsetForBeam', "cat.n.01", "feline mammal")
        mock_dog_synset = self.mock_factory('MockSynsetForBeam', "dog.n.01", "canine mammal")
        
        # Get validation error embeddings from config
        embeddings = self.mock_config.get_validation_error_embeddings()
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['src_embeddings'], embeddings['tgt_embeddings']
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = self.mock_factory('MockEmbeddingModel')
            with self.assertRaises(ValueError) as context:
                self.beam_builder.get_new_beams(
                    mock_graph, "cat.n.01", "dog.n.01", mock_model
                )
            
            self.assertIn("Not all lexical properties", str(context.exception))
            self.assertIn("in graph", str(context.exception))

    def test_get_new_beams_beam_width_limiting(self):
        """Test get_new_beams respects beam_width parameter"""
        mock_graph = self.mock_config.get_basic_test_graph()
        
        mock_cat_synset = self.mock_factory('MockSynsetForBeam', "cat.n.01", "feline mammal")
        mock_dog_synset = self.mock_factory('MockSynsetForBeam', "dog.n.01", "canine mammal")
        
        embeddings = self.mock_config.get_basic_test_embeddings()
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['src_embeddings'], embeddings['tgt_embeddings']
        ]
        
        # Get test pairs for beam width testing
        pairs = self.mock_config.get_beam_width_test_pairs()
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            pairs['asymm_pairs'], pairs['symm_pairs']
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = self.mock_factory('MockEmbeddingModel')
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
        mock_graph = self.mock_config.get_basic_test_graph()
        
        mock_cat_synset = self.mock_factory('MockSynsetForBeam', "cat.n.01", "feline mammal")
        mock_dog_synset = self.mock_factory('MockSynsetForBeam', "dog.n.01", "canine mammal")
        
        embeddings = self.mock_config.get_basic_test_embeddings()
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['src_embeddings'], embeddings['tgt_embeddings']
        ]
        
        mock_pair = (("cat.n.01", "hypernyms"), ("dog.n.01", "hypernyms"), 0.85)
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            [mock_pair], []  # asymm, symm
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = self.mock_factory('MockEmbeddingModel')
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
        mock_graph = self.mock_config.get_basic_test_graph()
        
        mock_cat_synset = self.mock_factory('MockSynsetForBeam', "cat.n.01", "feline mammal")
        mock_dog_synset = self.mock_factory('MockSynsetForBeam', "dog.n.01", "canine mammal")
        mock_model = self.mock_factory('MockEmbeddingModel')
        
        embeddings = self.mock_config.get_basic_test_embeddings()
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['src_embeddings'], embeddings['tgt_embeddings']
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
        # Initialize mock factory and config
        self.mock_factory = BeamBuilderMockFactory()
        self.mock_config = BeamBuilderMockConfig()
        
        # Create mocks using factory
        self.mock_embedding_helper = self.mock_factory('MockEmbeddingHelper')
        self.beam_builder = BeamBuilder(self.mock_embedding_helper)
        
        # Set up edge cases mock
        self.edge_case_mock = self.mock_factory('MockBeamBuilderEdgeCases')

    def test_get_new_beams_with_zero_beam_width(self):
        """Test get_new_beams with beam_width=0"""
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("dog.n.01")
        
        mock_cat_synset = self.mock_factory('MockSynsetForBeam', "cat.n.01", "feline mammal")
        mock_dog_synset = self.mock_factory('MockSynsetForBeam', "dog.n.01", "canine mammal")
        
        embeddings = self.mock_config.get_empty_embeddings()
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['src_embeddings'], embeddings['tgt_embeddings']
        ]
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.return_value = []
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = self.mock_factory('MockEmbeddingModel')
            result = self.beam_builder.get_new_beams(
                mock_graph, "cat.n.01", "dog.n.01", mock_model, beam_width=0
            )
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 0)

    def test_get_new_beams_identical_scores(self):
        """Test get_new_beams when multiple pairs have identical scores"""
        mock_graph = self.mock_config.get_basic_test_graph()
        
        mock_cat_synset = self.mock_factory('MockSynsetForBeam', "cat.n.01", "feline mammal")
        mock_dog_synset = self.mock_factory('MockSynsetForBeam', "dog.n.01", "canine mammal")
        
        embeddings = self.mock_config.get_basic_test_embeddings()
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['src_embeddings'], embeddings['tgt_embeddings']
        ]
        
        # Get pairs with identical scores from config
        identical_score_pairs = self.mock_config.get_identical_scores_pairs()
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            identical_score_pairs, []
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = self.mock_factory('MockEmbeddingModel')
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
        mock_graph = self.mock_config.get_partial_neighbor_validation_graph()
        
        mock_cat_synset = self.mock_factory('MockSynsetForBeam', "cat.n.01", "feline mammal")
        mock_dog_synset = self.mock_factory('MockSynsetForBeam', "dog.n.01", "canine mammal")
        
        # Get embeddings that will cause partial validation error
        embeddings = self.mock_config.get_partial_neighbor_embeddings()
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['src_embeddings'], embeddings['tgt_embeddings']
        ]
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            mock_model = self.mock_factory('MockEmbeddingModel')
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
        # Initialize mock factory and config for integration tests
        self.mock_factory = BeamBuilderMockFactory()
        self.mock_config = BeamBuilderMockConfig()
        self.integration_mock = self.mock_factory('MockBeamBuilderIntegration')
        
        # Create embedding helper using factory
        self.mock_embedding_helper = self.mock_factory('MockEmbeddingHelper')
        self.beam_builder = BeamBuilder(self.mock_embedding_helper)

    def test_realistic_wordnet_scenario(self):
        """Test BeamBuilder with a realistic WordNet-like graph structure"""
        # Get realistic graph and data from config
        graph = self.mock_config.get_realistic_wordnet_graph()
        embeddings = self.mock_config.get_realistic_embeddings()
        pairs = self.mock_config.get_realistic_aligned_pairs()
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['cat_embeddings'], embeddings['dog_embeddings']
        ]
        
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            pairs['asymm_pairs'], pairs['symm_pairs']
        ]
        
        # Use synset mocks from integration mock
        mock_cat_synset, mock_dog_synset = self.integration_mock.create_integration_synset_mocks()
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            mock_synset.side_effect = [mock_cat_synset, mock_dog_synset]
            
            # Use embedding model mock from integration mock
            mock_model = self.integration_mock.create_realistic_embedding_model_mock()
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
        # Get complex test data from config
        graph = self.mock_config.get_complex_graph()
        embeddings = self.mock_config.get_complex_embeddings()
        pairs = self.mock_config.get_complex_aligned_pairs()
        
        self.mock_embedding_helper.embed_lexical_relations.side_effect = [
            embeddings['cat_embeddings'], embeddings['dog_embeddings']
        ]
        
        self.mock_embedding_helper.get_top_k_aligned_lex_rel_pairs.side_effect = [
            pairs['asymm_pairs'], pairs['symm_pairs']
        ]
        
        # Use complex synset mocks from integration mock
        synset_mocks = self.integration_mock.create_complex_synset_mocks()
        mock_cat_synset = synset_mocks['cat']
        mock_dog_synset = synset_mocks['dog']
        
        with patch('nltk.corpus.wordnet.synset') as mock_synset:
            # Use setup_patch_side_effects from integration mock
            mock_synset.side_effect = self.integration_mock.setup_patch_side_effects(synset_mocks)
            
            # Use embedding model mock from integration mock
            mock_model = self.integration_mock.create_realistic_embedding_model_mock()
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