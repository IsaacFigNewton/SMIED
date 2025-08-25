import unittest
from unittest.mock import patch, Mock
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.EmbeddingHelper import EmbeddingHelper
from tests.mocks.embedding_helper_mocks import EmbeddingHelperMockFactory
from tests.config.embedding_helper_config import EmbeddingHelperMockConfig


class TestEmbeddingHelper(unittest.TestCase):
    """Test the EmbeddingHelper class basic functionality.
    
    Focuses on core operations and expected behavior under normal conditions.
    Tests the primary methods and typical use cases of EmbeddingHelper.
    """
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory and config
        self.mock_factory = EmbeddingHelperMockFactory()
        self.mock_config = EmbeddingHelperMockConfig()
        
        # Create embedding helper instance
        self.embedding_helper = EmbeddingHelper()
        
        # Get basic test data from config
        self.basic_test_data = self.mock_config.get_basic_test_data()

    def test_initialization(self):
        """Test EmbeddingHelper initialization."""
        self.assertIsInstance(self.embedding_helper, EmbeddingHelper)

    def test_get_synset_embedding_centroid_simple(self):
        """Test get_synset_embedding_centroid with simple synset."""
        # Get test data from config
        simple_test = self.basic_test_data['synset_centroid_tests']['simple_synset']
        
        # Create synset using factory and config data
        mock_synset = self.mock_factory('MockSynset', simple_test['synset_name'])
        mock_lemma1 = self.mock_factory('MockLemma', simple_test['lemmas'][0])  # "cat"
        mock_lemma2 = self.mock_factory('MockLemma', simple_test['lemmas'][1])  # "feline"
        mock_synset.lemmas.return_value = [mock_lemma1, mock_lemma2]
        
        # Create embedding model using config vectors
        mock_model = simple_test['mock_vectors']
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, simple_test['expected_centroid_shape'])
        np.testing.assert_array_equal(result, simple_test['expected_centroid'])

    def test_get_synset_embedding_centroid_underscore_replacement(self):
        """Test get_synset_embedding_centroid with underscore replacement."""
        # Get test data from config
        compound_test = self.basic_test_data['synset_centroid_tests']['compound_lemma']
        
        mock_synset = self.mock_factory('MockSynset', compound_test['synset_name'])
        mock_lemma = self.mock_factory('MockLemma', compound_test['lemmas'][0])  # "ice_cream"
        mock_synset.lemmas.return_value = [mock_lemma]
        
        # Use config vectors that include space-separated version
        mock_model = compound_test['mock_vectors']
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, compound_test['mock_vectors']['ice cream'])

    def test_get_synset_embedding_centroid_space_to_underscore(self):
        """Test get_synset_embedding_centroid with space to underscore conversion."""
        # Get test data from config
        compound_test = self.basic_test_data['synset_centroid_tests']['compound_lemma']
        
        mock_synset = self.mock_factory('MockSynset', compound_test['synset_name'])
        mock_lemma = self.mock_factory('MockLemma', compound_test['lemmas'][0])  # "ice_cream"
        mock_synset.lemmas.return_value = [mock_lemma]
        
        # Use underscore version from config
        mock_model = {
            compound_test['lemmas'][0]: compound_test['mock_vectors']['ice_cream']  # "ice_cream" vector
        }
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, compound_test['mock_vectors']['ice_cream'])

    def test_get_synset_embedding_centroid_multi_word_fallback(self):
        """Test get_synset_embedding_centroid with multi-word fallback."""
        # Get test data from config
        multi_word_test = self.basic_test_data['synset_centroid_tests']['multi_word_fallback']
        
        mock_synset = self.mock_factory('MockSynset', multi_word_test['synset_name'])
        mock_lemma = self.mock_factory('MockLemma', multi_word_test['lemmas'][0])  # "hot_dog"
        mock_synset.lemmas.return_value = [mock_lemma]
        
        # Use multi-word vectors from config (contains "hot" and "dog" but not "hot_dog")
        mock_model = multi_word_test['mock_vectors']
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, multi_word_test['expected_centroid'])

    def test_embed_lexical_relations(self):
        """Test embed_lexical_relations method."""
        # Get test data from config
        relations_test = self.basic_test_data['lexical_relations_tests']['basic_relations']
        
        mock_synset = self.mock_factory('MockSynset', relations_test['synset_name'])
        
        # Mock related synsets using config data
        mock_hypernym = self.mock_factory('MockSynset', relations_test['relations']['hypernyms'][0])  # "animal.n.01"
        mock_hyponym = self.mock_factory('MockSynset', relations_test['relations']['hyponyms'][0])  # "kitten.n.01"
        
        # Set up relation methods using config data
        mock_synset.hypernyms.return_value = [mock_hypernym]
        mock_synset.hyponyms.return_value = [mock_hyponym]
        
        # Set empty relations for other types from config
        for rel_type in relations_test['relations']['empty_relations']:
            getattr(mock_synset, rel_type).return_value = []
        
        mock_model = self.mock_factory('MockEmbeddingModelForHelper')
        
        with patch.object(self.embedding_helper, 'get_synset_embedding_centroid') as mock_centroid:
            mock_centroid.side_effect = [
                relations_test['expected_embeddings']['hypernyms'][0][1],  # For hypernym
                relations_test['expected_embeddings']['hyponyms'][0][1]    # For hyponym
            ]
            
            result = self.embedding_helper.embed_lexical_relations(mock_synset, mock_model)
            
            self.assertIsInstance(result, dict)
            self.assertIn('hypernyms', result)
            self.assertIn('hyponyms', result)
            
            # Check hypernyms using config values
            self.assertEqual(len(result['hypernyms']), 1)
            self.assertEqual(result['hypernyms'][0][0], relations_test['relations']['hypernyms'][0])
            np.testing.assert_array_equal(result['hypernyms'][0][1], relations_test['expected_embeddings']['hypernyms'][0][1])
            
            # Check hyponyms using config values
            self.assertEqual(len(result['hyponyms']), 1)
            self.assertEqual(result['hyponyms'][0][0], relations_test['relations']['hyponyms'][0])
            np.testing.assert_array_equal(result['hyponyms'][0][1], relations_test['expected_embeddings']['hyponyms'][0][1])

    def test_get_embedding_similarities_basic(self):
        """Test get_embedding_similarities with basic inputs."""
        # Get similarity test data from config
        similarity_test = self.basic_test_data['similarity_calculation_tests']['basic_similarity']
        
        rel_embs_1 = similarity_test['rel_embs_1']
        rel_embs_2 = similarity_test['rel_embs_2']
        
        result = self.embedding_helper.get_embedding_similarities(rel_embs_1, rel_embs_2)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        
        # Compare with expected similarities from config
        expected = similarity_test['expected_similarities']
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_top_k_aligned_lex_rel_pairs(self):
        """Test get_top_k_aligned_lex_rel_pairs method."""
        # Get relation mapping and synset data from config
        relation_mappings = self.mock_config.get_relation_mapping_test_data()
        synset_names = self.mock_config.get_mock_synset_names()
        
        src_tgt_rel_map = relation_mappings['basic_mapping']
        
        src_emb_dict = {
            "hypernyms": [(synset_names['animal_synsets']['animal'], np.array([1.0, 0.0]))],
            "hyponyms": [(synset_names['animal_synsets']['kitten'], np.array([0.0, 1.0]))]
        }
        
        tgt_emb_dict = {
            "hyponyms": [(synset_names['animal_synsets']['puppy'], np.array([1.0, 0.0]))],
            "hypernyms": [(synset_names['animal_synsets']['mammal'], np.array([0.0, 1.0]))]
        }
        
        with patch.object(self.embedding_helper, 'get_embedding_similarities') as mock_similarities:
            mock_similarities.side_effect = [
                np.array([[0.9]]),  # hypernyms -> hyponyms similarity
                np.array([[0.8]])   # hyponyms -> hypernyms similarity
            ]
            
            # Get beam width from config
            beam_params = self.mock_config.get_beam_test_parameters()
            beam_width = beam_params['small_beam']['width']
            
            result = self.embedding_helper.get_top_k_aligned_lex_rel_pairs(
                src_tgt_rel_map, src_emb_dict, tgt_emb_dict, beam_width=beam_width
            )
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            
            # Results should be sorted by similarity (descending)
            self.assertGreaterEqual(result[0][2], result[1][2])
            
            # Check structure of results
            for item in result:
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 3)
                self.assertIsInstance(item[0], tuple)  # (synset_name, relation)
                self.assertIsInstance(item[1], tuple)  # (synset_name, relation)
                self.assertIsInstance(item[2], (int, float))  # similarity

    def test_get_new_beams_from_embeddings(self):
        """Test get_new_beams_from_embeddings method."""
        mock_graph = Mock()
        mock_wn_module = Mock()
        
        # Mock synsets
        mock_src_synset = Mock()
        mock_tgt_synset = Mock()
        mock_wn_module.synset.side_effect = [mock_src_synset, mock_tgt_synset]
        
        mock_model = {}
        
        # Mock embedding results
        mock_src_emb = {"hypernyms": [("animal.n.01", np.array([1.0]))]}
        mock_tgt_emb = {"hyponyms": [("puppy.n.01", np.array([1.0]))]}
        
        with patch.object(self.embedding_helper, 'embed_lexical_relations') as mock_embed, \
             patch.object(self.embedding_helper, 'get_top_k_aligned_lex_rel_pairs') as mock_pairs:
            
            mock_embed.side_effect = [mock_src_emb, mock_tgt_emb]
            mock_pairs.side_effect = [
                [(("animal.n.01", "hypernyms"), ("puppy.n.01", "hyponyms"), 0.9)],  # asymm
                [(("animal.n.01", "hypernyms"), ("puppy.n.01", "hypernyms"), 0.8)]   # symm
            ]
            
            result = self.embedding_helper.get_new_beams_from_embeddings(
                mock_graph, "cat.n.01", "dog.n.01", mock_wn_module, mock_model, beam_width=3
            )
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            
            # Should be sorted by similarity (descending)
            self.assertEqual(result[0][2], 0.9)
            self.assertEqual(result[1][2], 0.8)

    def test_build_gloss_seed_nodes_from_predicate_subjects_mode(self):
        """Test build_gloss_seed_nodes_from_predicate with subjects mode."""
        mock_pred_syn = Mock()
        mock_pred_syn.definition.return_value = "test definition"
        
        mock_wn_module = Mock()
        mock_wn_module.NOUN = 'n'
        mock_wn_module.VERB = 'v'
        
        mock_nlp_func = Mock()
        mock_doc = Mock()
        mock_nlp_func.return_value = mock_doc
        
        mock_token = Mock()
        mock_token.text = "subject"
        mock_subject_synset = Mock()
        mock_subject_synset.name.return_value = "subject.n.01"
        mock_wn_module.synsets.return_value = [mock_subject_synset]
        
        mock_extract_subjects_fn = Mock()
        mock_extract_subjects_fn.return_value = ([mock_token], [])
        
        result = self.embedding_helper.build_gloss_seed_nodes_from_predicate(
            mock_pred_syn, mock_wn_module, mock_nlp_func,
            mode="subjects", extract_subjects_fn=mock_extract_subjects_fn
        )
        
        self.assertIsInstance(result, set)
        # Without top_k_branch_fn, should add first few candidate synsets
        self.assertIn("subject.n.01", result)


class TestEmbeddingHelperValidation(unittest.TestCase):
    """Test validation and constraint checking for EmbeddingHelper.
    
    Focuses on input validation, parameter constraints, and proper 
    error handling for invalid inputs and edge conditions.
    """
    
    def setUp(self):
        """Set up validation test fixtures using mock factory and config injection."""
        # Initialize mock factory and config
        self.mock_factory = EmbeddingHelperMockFactory()
        self.mock_config = EmbeddingHelperMockConfig()
        
        # Create embedding helper instance
        self.embedding_helper = EmbeddingHelper()
        
        # Get validation test data from config
        self.validation_test_data = self.mock_config.get_validation_test_data()
        
        # Create validation mock for specialized testing
        self.validation_mock = self.mock_factory('MockEmbeddingHelperValidation')

    def test_get_embedding_similarities_empty_inputs(self):
        """Test get_embedding_similarities with empty inputs."""
        result1 = self.embedding_helper.get_embedding_similarities([], [("test", np.array([1.0]))])
        result2 = self.embedding_helper.get_embedding_similarities([("test", np.array([1.0]))], [])
        result3 = self.embedding_helper.get_embedding_similarities([], [])
        
        self.assertEqual(result1.shape, (0, 0))
        self.assertEqual(result2.shape, (0, 0))
        self.assertEqual(result3.shape, (0, 0))

    def test_get_top_k_aligned_lex_rel_pairs_empty_relations(self):
        """Test get_top_k_aligned_lex_rel_pairs with missing relations."""
        # Get missing relation mapping from config
        relation_mappings = self.mock_config.get_relation_mapping_test_data()
        src_tgt_rel_map = relation_mappings['missing_relation_mapping']
        
        # Get synset names from config
        synset_names = self.mock_config.get_mock_synset_names()
        src_emb_dict = {"hypernyms": [(synset_names['animal_synsets']['animal'], np.array([1.0]))]}
        tgt_emb_dict = {"hyponyms": [(synset_names['animal_synsets']['puppy'], np.array([1.0]))]}
        
        with patch.object(self.embedding_helper, 'get_embedding_similarities') as mock_similarities:
            mock_similarities.return_value = np.array([[0.9]])
            
            result = self.embedding_helper.get_top_k_aligned_lex_rel_pairs(
                src_tgt_rel_map, src_emb_dict, tgt_emb_dict, beam_width=3
            )
            
            # Should only have one result (missing relations are skipped)
            self.assertEqual(len(result), 1)

    def test_get_top_k_aligned_lex_rel_pairs_zero_beam_width(self):
        """Test get_top_k_aligned_lex_rel_pairs with beam_width <= 0."""
        src_tgt_rel_map = {"hypernyms": "hyponyms"}
        src_emb_dict = {"hypernyms": [("animal.n.01", np.array([1.0]))]}
        tgt_emb_dict = {"hyponyms": [("puppy.n.01", np.array([1.0]))]}
        
        with patch.object(self.embedding_helper, 'get_embedding_similarities') as mock_similarities:
            mock_similarities.return_value = np.array([[0.9]])
            
            result = self.embedding_helper.get_top_k_aligned_lex_rel_pairs(
                src_tgt_rel_map, src_emb_dict, tgt_emb_dict, beam_width=0
            )
            
            # Should return all results when beam_width <= 0
            self.assertEqual(len(result), 1)

    def test_beam_width_parameter_validation(self):
        """Test validation of beam width parameters."""
        constraints = self.validation_test_data['parameter_constraint_tests']['beam_width_constraints']
        
        # Test valid beam widths
        for valid_width in constraints['valid_beam_widths']:
            self.assertTrue(self.validation_mock.validate_beam_width(valid_width))
        
        # Test edge case beam widths (0 should be valid - returns all results)
        for edge_width in constraints['edge_case_beam_widths']:
            if edge_width == 0:
                self.assertTrue(self.validation_mock.validate_beam_width(edge_width))

    def test_dimension_mismatch_validation(self):
        """Test validation of vector dimension mismatches."""
        dimension_test = self.validation_test_data['parameter_constraint_tests']['dimension_constraints']
        mismatch_data = dimension_test['mismatched_dimensions']
        
        # Should detect dimension mismatch
        with self.assertRaises(mismatch_data['expected_error']):
            self.validation_mock.simulate_dimension_mismatch(
                mismatch_data['vector_a'], 
                mismatch_data['vector_b']
            )

    def test_input_validation_scenarios(self):
        """Test various input validation scenarios."""
        validation_scenarios = self.validation_test_data['input_validation_scenarios']
        
        # Test valid inputs
        valid_scenario = validation_scenarios['valid_synset_input']
        self.assertTrue(self.validation_mock.validate_synset_input(valid_scenario['synset_name']))
        
        # Test invalid inputs
        invalid_scenario = validation_scenarios['invalid_synset_input']
        with self.assertRaises(invalid_scenario['expected_error']):
            self.validation_mock.simulate_invalid_synset(invalid_scenario['synset_name'])
        
        # Test empty model input
        empty_model_scenario = validation_scenarios['empty_model_input']
        with self.assertRaises(empty_model_scenario['expected_error']):
            self.validation_mock.simulate_empty_model(empty_model_scenario['synset_name'])


class TestEmbeddingHelperEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for EmbeddingHelper.
    
    Handles boundary conditions, exceptional inputs, and error scenarios
    that may occur during embedding operations.
    """
    
    def setUp(self):
        """Set up edge case test fixtures using mock factory and config injection."""
        # Initialize mock factory and config
        self.mock_factory = EmbeddingHelperMockFactory()
        self.mock_config = EmbeddingHelperMockConfig()
        
        # Create embedding helper and edge case mock
        self.embedding_helper = EmbeddingHelper()
        self.edge_case_mock = self.mock_factory('MockEmbeddingHelperEdgeCases')
        
        # Get edge case test data from config
        self.edge_case_data = self.mock_config.get_edge_case_test_data()

    def test_get_synset_embedding_centroid_empty_result(self):
        """Test get_synset_embedding_centroid when no embeddings found."""
        # Get edge case data from config
        empty_scenario = self.edge_case_data['empty_input_scenarios']['empty_synset_lemmas']
        
        mock_synset = self.mock_factory('MockSynset', empty_scenario['synset_name'])
        mock_lemma = self.mock_factory('MockLemma', empty_scenario['synset_name'])
        mock_synset.lemmas.return_value = []  # Empty lemmas list
        
        mock_model = {}  # Empty model
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 0)  # Empty array

    def test_get_synset_embedding_centroid_exception_handling(self):
        """Test get_synset_embedding_centroid handles exceptions gracefully."""
        # Get exception scenario from config
        exception_scenario = self.edge_case_data['exception_handling_scenarios']['synset_lemmas_exception']
        
        mock_synset = self.mock_factory('MockSynset', 'test_synset')
        mock_synset.lemmas.side_effect = exception_scenario['side_effect']
        
        mock_model = {}
        
        with patch('builtins.print') as mock_print:
            result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 0)
        if exception_scenario['expected_print_call']:
            mock_print.assert_called()  # Should print error message

    def test_get_embedding_similarities_zero_norm_handling(self):
        """Test get_embedding_similarities handles zero-norm vectors."""
        # Get zero norm test data from config
        zero_norm_scenario = self.edge_case_data['zero_vector_scenarios']['zero_norm_vectors']
        
        rel_embs_1 = [("synset1", zero_norm_scenario['vector_a'])]    # Zero vector
        rel_embs_2 = [("synset2", zero_norm_scenario['vector_b'])]    # Normal vector
        
        result = self.embedding_helper.get_embedding_similarities(rel_embs_1, rel_embs_2)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 1))
        # Should not raise divide-by-zero error and should return finite value
        self.assertTrue(np.isfinite(result[0, 0]))

    def test_get_embedding_similarities_large_matrices(self):
        """Test get_embedding_similarities with larger input matrices."""
        # Get large scale scenario from config
        large_scenario = self.edge_case_data['large_scale_scenarios']['large_matrices']
        
        # Create larger embedding lists using config parameters
        rel_embs_1 = [(f"synset{i}", np.random.rand(large_scenario['vector_dimension'])) 
                     for i in range(large_scenario['matrix_size_1'])]
        rel_embs_2 = [(f"target{i}", np.random.rand(large_scenario['vector_dimension'])) 
                     for i in range(large_scenario['matrix_size_2'])]
        
        result = self.embedding_helper.get_embedding_similarities(rel_embs_1, rel_embs_2)
        
        self.assertEqual(result.shape, large_scenario['expected_shape'])
        self.assertTrue(np.all(np.isfinite(result)))
        # Check similarity bounds from config
        lower_bound, upper_bound = large_scenario['similarity_bounds']
        self.assertTrue(np.all(result >= lower_bound))  # Cosine similarity >= -1
        self.assertTrue(np.all(result <= upper_bound))   # Cosine similarity <= 1

    def test_embed_lexical_relations_empty_centroids_filtered(self):
        """Test embed_lexical_relations filters out empty centroids."""
        mock_synset = self.mock_factory('MockSynset', 'test.n.01')
        
        mock_related = self.mock_factory('MockSynset', 'related.n.01')
        
        mock_synset.hypernyms.return_value = [mock_related]
        # Set all other relations to empty using config relation types
        relation_mappings = self.mock_config.get_relation_type_mappings()
        all_relations = relation_mappings['all_relation_types']
        
        for rel in ['part_holonyms', 'substance_holonyms', 'member_holonyms', 
                   'part_meronyms', 'substance_meronyms', 'member_meronyms',
                   'hyponyms', 'entailments', 'causes', 'also_sees', 'verb_groups']:
            getattr(mock_synset, rel).return_value = []
        
        mock_model = {}
        
        with patch.object(self.embedding_helper, 'get_synset_embedding_centroid') as mock_centroid:
            mock_centroid.return_value = np.array([])  # Empty centroid
            
            result = self.embedding_helper.embed_lexical_relations(mock_synset, mock_model)
            
            # All relations should be empty since centroids are empty
            for rel_name, rel_items in result.items():
                self.assertEqual(len(rel_items), 0)

    def test_embed_lexical_relations_exception_handling(self):
        """Test embed_lexical_relations handles exceptions in relation methods."""
        # Get exception scenario from config
        exception_scenario = self.edge_case_data['exception_handling_scenarios']['relation_method_exception']
        
        mock_synset = self.mock_factory('MockSynset', 'test.n.01')
        
        # Make hypernyms raise an exception
        mock_synset.hypernyms.side_effect = exception_scenario['side_effect']
        
        # Set other relations to empty using config relation types
        for rel in ['part_holonyms', 'substance_holonyms', 'member_holonyms', 
                   'part_meronyms', 'substance_meronyms', 'member_meronyms',
                   'hyponyms', 'entailments', 'causes', 'also_sees', 'verb_groups']:
            getattr(mock_synset, rel).return_value = []
        
        mock_model = {}
        
        with patch('builtins.print') as mock_print:
            result = self.embedding_helper.embed_lexical_relations(mock_synset, mock_model)
        
        # Should handle exception and return empty for hypernyms
        if exception_scenario['expected_empty_relation']:
            self.assertEqual(len(result['hypernyms']), 0)
        mock_print.assert_called()

    def test_get_top_k_aligned_lex_rel_pairs_index_error_handling(self):
        """Test get_top_k_aligned_lex_rel_pairs handles IndexError properly."""
        src_tgt_rel_map = {"hypernyms": "hyponyms"}
        src_emb_dict = {"hypernyms": [("animal.n.01", np.array([1.0]))]}
        tgt_emb_dict = {"hyponyms": [("puppy.n.01", np.array([1.0]))]}
        
        # Mock get_embedding_similarities to return mismatched dimensions
        with patch.object(self.embedding_helper, 'get_embedding_similarities') as mock_similarities:
            # Return matrix that doesn't match the input list sizes
            mock_similarities.return_value = np.array([[0.9, 0.8]])  # 1x2 but should be 1x1
            
            with self.assertRaises(IndexError) as context:
                self.embedding_helper.get_top_k_aligned_lex_rel_pairs(
                    src_tgt_rel_map, src_emb_dict, tgt_emb_dict
                )
            
            self.assertIn("Index error in get_top_k_aligned_lex_rel_pairs", str(context.exception))

    def test_embed_lexical_relations_all_relation_types(self):
        """Test embed_lexical_relations covers all expected relation types."""
        mock_synset = Mock()
        
        # Get all relation types from config
        relation_types = self.mock_config.get_relation_type_mappings()['all_relation_types']
        
        for i, rel_type in enumerate(relation_types):
            mock_related = Mock()
            mock_related.name.return_value = f"{rel_type}.n.01"
            getattr(mock_synset, rel_type).return_value = [mock_related]
        
        mock_model = {}
        
        with patch.object(self.embedding_helper, 'get_synset_embedding_centroid') as mock_centroid:
            mock_centroid.return_value = np.array([1.0, 2.0])
            
            result = self.embedding_helper.embed_lexical_relations(mock_synset, mock_model)
            
            # Should have all relation types in result
            for rel_type in relation_types:
                self.assertIn(rel_type, result)
                self.assertEqual(len(result[rel_type]), 1)
                self.assertEqual(result[rel_type][0][0], f"{rel_type}.n.01")

    def test_get_new_beams_from_embeddings_invalid_synsets(self):
        """Test get_new_beams_from_embeddings with invalid synset names."""
        mock_graph = Mock()
        mock_wn_module = Mock()
        mock_wn_module.synset.side_effect = Exception("Invalid synset")
        
        mock_model = {}
        
        result = self.embedding_helper.get_new_beams_from_embeddings(
            mock_graph, "invalid.synset", "another.invalid", mock_wn_module, mock_model
        )
        
        self.assertEqual(result, [])


class TestEmbeddingHelperIntegration(unittest.TestCase):
    """Integration tests for EmbeddingHelper with realistic scenarios.
    
    Tests complete workflows and interactions between EmbeddingHelper
    components using realistic data and scenarios.
    """
    
    def setUp(self):
        """Set up for integration testing using mock factory and config injection."""
        # Initialize mock factory for integration tests
        self.mock_factory = EmbeddingHelperMockFactory()
        self.mock_config = EmbeddingHelperMockConfig()
        self.integration_mock = self.mock_factory('MockEmbeddingHelperIntegration')
        
        self.embedding_helper = EmbeddingHelper()
        
        # Get integration test data from config
        self.integration_test_data = self.mock_config.get_integration_test_data()

    def test_full_embedding_workflow(self):
        """Test complete embedding workflow from synset to similarity pairs."""
        # Get workflow scenario from config
        workflow_scenario = self.integration_test_data['workflow_scenarios']['cat_to_animal_workflow']
        
        # Mock a realistic synset using config data
        source_synset_config = workflow_scenario['source_synset']
        mock_synset = Mock()
        mock_synset.name.return_value = source_synset_config['name']
        
        # Mock lemmas using config data
        mock_lemmas = []
        for lemma_name in source_synset_config['lemmas']:
            mock_lemma = Mock()
            mock_lemma.name.return_value = lemma_name
            mock_lemmas.append(mock_lemma)
        mock_synset.lemmas.return_value = mock_lemmas
        
        # Mock related synsets using config data
        mock_hypernym = Mock()
        mock_hypernym.name.return_value = source_synset_config['hypernyms'][0]
        mock_hypernym.lemmas.return_value = [Mock()]
        mock_hypernym.lemmas.return_value[0].name.return_value = "mammal"
        
        mock_hyponym = Mock()
        mock_hyponym.name.return_value = source_synset_config['hyponyms'][0]
        mock_hyponym.lemmas.return_value = [Mock()]
        mock_hyponym.lemmas.return_value[0].name.return_value = "kitten"
        
        # Set up relations
        mock_synset.hypernyms.return_value = [mock_hypernym]
        mock_synset.hyponyms.return_value = [mock_hyponym]
        # Set all other relations to empty
        for rel in ['part_holonyms', 'substance_holonyms', 'member_holonyms', 
                   'part_meronyms', 'substance_meronyms', 'member_meronyms',
                   'entailments', 'causes', 'also_sees', 'verb_groups']:
            getattr(mock_synset, rel).return_value = []
        
        # Mock embedding model using config data
        mock_model = workflow_scenario['embedding_model']
        
        # Test embedding lexical relations
        result = self.embedding_helper.embed_lexical_relations(mock_synset, mock_model)
        
        self.assertIn('hypernyms', result)
        self.assertIn('hyponyms', result)
        self.assertEqual(len(result['hypernyms']), 1)
        self.assertEqual(len(result['hyponyms']), 1)
        
        # Test similarity computation between relations
        hypernym_embs = result['hypernyms']
        hyponym_embs = result['hyponyms']
        
        similarities = self.embedding_helper.get_embedding_similarities(hypernym_embs, hyponym_embs)
        
        self.assertEqual(similarities.shape, (1, 1))
        
        # Validate against expected results from config
        expected_results = workflow_scenario['expected_beam_results']
        self.assertTrue(similarities[0, 0] >= expected_results['min_similarity'])

    def test_realistic_beam_generation_scenario(self):
        """Test realistic beam generation with multiple synsets and relations."""
        # Get beam generation scenario from config
        beam_scenario = self.integration_test_data['realistic_beam_generation']['multi_synset_scenario']
        
        # Mock WordNet module and synsets
        mock_wn = Mock()
        
        mock_cat = Mock()
        mock_cat.name.return_value = beam_scenario['synsets'][0]  # "cat.n.01"
        mock_dog = Mock()
        mock_dog.name.return_value = beam_scenario['synsets'][1]  # "dog.n.01"
        
        mock_wn.synset.side_effect = [mock_cat, mock_dog]
        
        # Mock embedding model with realistic word vectors from workflow config
        workflow_scenario = self.integration_test_data['workflow_scenarios']['cat_to_animal_workflow']
        mock_model = workflow_scenario['embedding_model']
        
        # Mock lexical relations with realistic structure
        mock_cat_emb = {
            "hypernyms": [("mammal.n.01", mock_model['mammal'])],
            "hyponyms": []
        }
        
        mock_dog_emb = {
            "hypernyms": [("mammal.n.01", mock_model['mammal'])],
            "hyponyms": [("puppy.n.01", mock_model['puppy'])]
        }
        
        with patch.object(self.embedding_helper, 'embed_lexical_relations') as mock_embed:
            mock_embed.side_effect = [mock_cat_emb, mock_dog_emb]
            
            result = self.embedding_helper.get_new_beams_from_embeddings(
                Mock(), beam_scenario['synsets'][0], beam_scenario['synsets'][1], 
                mock_wn, mock_model, beam_width=beam_scenario['beam_width']
            )
            
            # Validate result structure using config expectations
            expected_structure = beam_scenario['expected_result_structure']
            self.assertIsInstance(result, expected_structure['result_type'])
            self.assertLessEqual(len(result), beam_scenario['beam_width'])  # Should respect beam width
            
            # Check result structure
            for beam_item in result:
                self.assertIsInstance(beam_item, expected_structure['item_structure'])
                self.assertEqual(len(beam_item), expected_structure['item_length'])
                self.assertIsInstance(beam_item[expected_structure['similarity_index']], (int, float))

    def test_build_gloss_seed_nodes_integration(self):
        """Test build_gloss_seed_nodes_from_predicate with complete integration."""
        # Get gloss test data from config
        gloss_test_data = self.mock_config.get_gloss_test_data()
        
        mock_pred_syn = Mock()
        mock_pred_syn.definition.return_value = gloss_test_data['test_definitions'][0]
        
        mock_wn_module = Mock()
        mock_wn_module.NOUN = gloss_test_data['wordnet_pos_constants']['NOUN']
        
        mock_nlp_func = Mock()
        mock_doc = Mock()
        mock_nlp_func.return_value = mock_doc
        
        mock_token = Mock()
        mock_token.text = gloss_test_data['sample_tokens']['subject']
        
        mock_subject_synset = Mock()
        mock_subject_synset.name.return_value = gloss_test_data['target_synset_examples'][1]  # "subject.n.01"
        mock_wn_module.synsets.return_value = [mock_subject_synset]
        
        mock_extract_subjects_fn = Mock()
        mock_extract_subjects_fn.return_value = ([mock_token], [])
        
        mock_top_k_branch_fn = Mock()
        mock_top_k_branch_fn.return_value = [
            ((mock_subject_synset, "relation"), ("target", "relation"), 0.9)
        ]
        
        result = self.embedding_helper.build_gloss_seed_nodes_from_predicate(
            mock_pred_syn, mock_wn_module, mock_nlp_func,
            mode=gloss_test_data['extraction_modes'][0],  # "subjects"
            extract_subjects_fn=mock_extract_subjects_fn,
            top_k_branch_fn=mock_top_k_branch_fn, 
            target_synsets=gloss_test_data['target_synset_examples'],
            max_sample_size=gloss_test_data['max_sample_sizes'][0],  # 3
            beam_width=3
        )
        
        self.assertIsInstance(result, set)
        self.assertIn(gloss_test_data['target_synset_examples'][1], result)  # "subject.n.01"


if __name__ == '__main__':
    unittest.main()