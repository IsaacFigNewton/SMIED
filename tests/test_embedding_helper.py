import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.EmbeddingHelper import EmbeddingHelper
from tests.mocks.embedding_helper_mocks import EmbeddingHelperMockFactory
from tests.config.embedding_helper_config import EmbeddingHelperMockConfig


class TestEmbeddingHelper(unittest.TestCase):
    """Test the EmbeddingHelper class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize mock factory and config
        self.mock_factory = EmbeddingHelperMockFactory()
        self.mock_config = EmbeddingHelperMockConfig()
        
        # Create embedding helper instance
        self.embedding_helper = EmbeddingHelper()

    def test_initialization(self):
        """Test EmbeddingHelper initialization"""
        self.assertIsInstance(self.embedding_helper, EmbeddingHelper)

    def test_get_synset_embedding_centroid_simple(self):
        """Test get_synset_embedding_centroid with simple synset"""
        # Get test embedding data from config
        embedding_data = self.mock_config.get_synset_embedding_test_data()
        similarity_tests = self.mock_config.get_similarity_calculation_test_matrices()
        
        # Create synset using mock factory
        mock_synset = Mock()
        mock_lemma1 = Mock()
        mock_lemma1.name.return_value = "cat"
        mock_lemma2 = Mock()
        mock_lemma2.name.return_value = "feline"
        mock_synset.lemmas.return_value = [mock_lemma1, mock_lemma2]
        
        # Create embedding model using factory and config data
        mock_model = self.mock_factory('MockEmbeddingModelForHelper')
        mock_model.__getitem__ = lambda self, key: {
            "cat": np.array([1.0, 2.0, 3.0]),
            "feline": np.array([2.0, 3.0, 4.0])
        }[key]
        mock_model.__contains__ = lambda self, key: key in ["cat", "feline"]
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        expected = np.array([1.5, 2.5, 3.5])  # Mean of the two embeddings
        np.testing.assert_array_equal(result, expected)

    def test_get_synset_embedding_centroid_underscore_replacement(self):
        """Test get_synset_embedding_centroid with underscore replacement"""
        # Get embedding vectors from config
        test_vectors = self.mock_config.get_test_embedding_vectors()
        
        mock_synset = Mock()
        mock_lemma = Mock()
        mock_lemma.name.return_value = "ice_cream"
        mock_synset.lemmas.return_value = [mock_lemma]
        
        mock_model = self.mock_factory('MockEmbeddingModelForHelper')
        mock_model.__getitem__ = lambda self, key: {
            "ice cream": np.array([1.0, 2.0, 3.0])  # Space instead of underscore
        }[key]
        mock_model.__contains__ = lambda self, key: key in ["ice cream"]
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_get_synset_embedding_centroid_space_to_underscore(self):
        """Test get_synset_embedding_centroid with space to underscore conversion"""
        # Get test vectors from config
        test_vectors = self.mock_config.get_test_embedding_vectors()
        
        mock_synset = Mock()
        mock_lemma = Mock()
        mock_lemma.name.return_value = "ice_cream"
        mock_synset.lemmas.return_value = [mock_lemma]
        
        mock_model = {
            "ice_cream": np.array([1.0, 2.0, 3.0])  # Underscore version exists
        }
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_get_synset_embedding_centroid_multi_word_fallback(self):
        """Test get_synset_embedding_centroid with multi-word fallback"""
        # Get vocabulary from config
        model_data = self.mock_config.get_realistic_embedding_model_mock_data()
        vocabulary = model_data['model_vocabulary']
        
        mock_synset = Mock()
        mock_lemma = Mock()
        mock_lemma.name.return_value = "hot_dog"
        mock_synset.lemmas.return_value = [mock_lemma]
        
        mock_model = {
            "hot": np.array([1.0, 2.0, 3.0]),
            "dog": np.array([3.0, 4.0, 5.0])
            # Note: "hot_dog" and "hot dog" are not in model
        }
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        expected = np.array([2.0, 3.0, 4.0])  # Mean of "hot" and "dog"
        np.testing.assert_array_equal(result, expected)

    def test_get_synset_embedding_centroid_empty_result(self):
        """Test get_synset_embedding_centroid when no embeddings found"""
        # Get edge case data from config
        edge_cases = self.mock_config.get_edge_case_test_data()
        invalid_synsets = edge_cases['invalid_synsets']
        
        mock_synset = Mock()
        mock_lemma = Mock()
        mock_lemma.name.return_value = invalid_synsets['nonexistent_synset']  # "fake.n.01"
        mock_synset.lemmas.return_value = [mock_lemma]
        
        mock_model = {}  # Empty model
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 0)  # Empty array

    def test_get_synset_embedding_centroid_exception_handling(self):
        """Test get_synset_embedding_centroid handles exceptions gracefully"""
        mock_synset = Mock()
        mock_synset.name.return_value = "test_synset"
        mock_synset.lemmas.side_effect = Exception("Test exception")
        
        mock_model = {}
        
        with patch('builtins.print') as mock_print:
            result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 0)
        mock_print.assert_called()  # Should print error message

    def test_embed_lexical_relations(self):
        """Test embed_lexical_relations method"""
        # Get synset data from config
        synset_data = self.mock_config.get_synset_embedding_test_data()
        relation_types = self.mock_config.get_relation_type_mappings()
        
        mock_synset = Mock()
        mock_synset.name.return_value = "cat.n.01"
        
        # Mock related synsets using config data
        mock_hypernym = Mock()
        mock_hypernym.name.return_value = "animal.n.01"
        mock_hyponym = Mock()
        mock_hyponym.name.return_value = "kitten.n.01"
        
        # Set up relation methods using config relation types
        mock_synset.hypernyms.return_value = [mock_hypernym]
        mock_synset.hyponyms.return_value = [mock_hyponym]
        
        # Set empty relations for other types from config
        for rel_type in ['part_holonyms', 'substance_holonyms', 'member_holonyms',
                        'part_meronyms', 'substance_meronyms', 'member_meronyms',
                        'entailments', 'causes', 'also_sees', 'verb_groups']:
            getattr(mock_synset, rel_type).return_value = []
        
        mock_model = self.mock_factory('MockEmbeddingModelForHelper')
        
        with patch.object(self.embedding_helper, 'get_synset_embedding_centroid') as mock_centroid:
            mock_centroid.side_effect = [
                np.array([1.0, 2.0]),  # For hypernym
                np.array([3.0, 4.0])   # For hyponym
            ]
            
            result = self.embedding_helper.embed_lexical_relations(mock_synset, mock_model)
            
            self.assertIsInstance(result, dict)
            self.assertIn('hypernyms', result)
            self.assertIn('hyponyms', result)
            
            # Check hypernyms using expected values
            self.assertEqual(len(result['hypernyms']), 1)
            self.assertEqual(result['hypernyms'][0][0], "animal.n.01")
            np.testing.assert_array_equal(result['hypernyms'][0][1], np.array([1.0, 2.0]))
            
            # Check hyponyms using expected values
            self.assertEqual(len(result['hyponyms']), 1)
            self.assertEqual(result['hyponyms'][0][0], "kitten.n.01")
            np.testing.assert_array_equal(result['hyponyms'][0][1], np.array([3.0, 4.0]))

    def test_embed_lexical_relations_empty_centroids_filtered(self):
        """Test embed_lexical_relations filters out empty centroids"""
        mock_synset = Mock()
        mock_synset.name.return_value = "test.n.01"
        
        mock_related = Mock()
        mock_related.name.return_value = "related.n.01"
        
        mock_synset.hypernyms.return_value = [mock_related]
        # Set all other relations to empty using config relation types
        relation_mappings = self.mock_config.get_relation_type_mappings()
        all_relations = (relation_mappings['asymmetric_relations'] + 
                        relation_mappings['symmetric_relations'] + 
                        relation_mappings['part_relations'])
        
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
        """Test embed_lexical_relations handles exceptions in relation methods"""
        mock_synset = Mock()
        mock_synset.name.return_value = "test.n.01"
        
        # Make hypernyms raise an exception
        mock_synset.hypernyms.side_effect = Exception("Test error")
        
        # Set other relations to empty using config relation types
        for rel in ['part_holonyms', 'substance_holonyms', 'member_holonyms', 
                   'part_meronyms', 'substance_meronyms', 'member_meronyms',
                   'hyponyms', 'entailments', 'causes', 'also_sees', 'verb_groups']:
            getattr(mock_synset, rel).return_value = []
        
        mock_model = {}
        
        with patch('builtins.print') as mock_print:
            result = self.embedding_helper.embed_lexical_relations(mock_synset, mock_model)
        
        # Should handle exception and return empty for hypernyms
        self.assertEqual(len(result['hypernyms']), 0)
        mock_print.assert_called()

    def test_get_embedding_similarities_basic(self):
        """Test get_embedding_similarities with basic inputs"""
        # Get similarity test data from config
        similarity_tests = self.mock_config.get_similarity_calculation_test_matrices()
        cosine_tests = similarity_tests['cosine_similarity_tests']
        
        rel_embs_1 = [
            ("synset1", np.array(cosine_tests[0]['vector_a'])),  # [1.0, 0.0, 0.0]
            ("synset2", np.array(cosine_tests[1]['vector_a']))   # [1.0, 1.0, 0.0] 
        ]
        rel_embs_2 = [
            ("synset3", np.array(cosine_tests[0]['vector_b'])),  # [0.0, 1.0, 0.0]
            ("synset4", np.array(cosine_tests[1]['vector_b']))   # [1.0, 1.0, 0.0]
        ]
        
        result = self.embedding_helper.get_embedding_similarities(rel_embs_1, rel_embs_2)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        
        # Expected similarities:
        # synset1 vs synset3: cosine([1,0], [1,0]) = 1.0
        # synset1 vs synset4: cosine([1,0], [0,1]) = 0.0
        # synset2 vs synset3: cosine([0,1], [1,0]) = 0.0
        # synset2 vs synset4: cosine([0,1], [0,1]) = 1.0
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_embedding_similarities_empty_inputs(self):
        """Test get_embedding_similarities with empty inputs"""
        result1 = self.embedding_helper.get_embedding_similarities([], [("test", np.array([1.0]))])
        result2 = self.embedding_helper.get_embedding_similarities([("test", np.array([1.0]))], [])
        result3 = self.embedding_helper.get_embedding_similarities([], [])
        
        self.assertEqual(result1.shape, (0, 0))
        self.assertEqual(result2.shape, (0, 0))
        self.assertEqual(result3.shape, (0, 0))

    def test_get_embedding_similarities_zero_norm_handling(self):
        """Test get_embedding_similarities handles zero-norm vectors"""
        rel_embs_1 = [("synset1", np.array([0.0, 0.0]))]  # Zero vector
        rel_embs_2 = [("synset2", np.array([1.0, 1.0]))]
        
        result = self.embedding_helper.get_embedding_similarities(rel_embs_1, rel_embs_2)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 1))
        # Should not raise divide-by-zero error and should return finite value
        self.assertTrue(np.isfinite(result[0, 0]))

    def test_get_top_k_aligned_lex_rel_pairs(self):
        """Test get_top_k_aligned_lex_rel_pairs method"""
        src_tgt_rel_map = {
            "hypernyms": "hyponyms",
            "hyponyms": "hypernyms"
        }
        
        src_emb_dict = {
            "hypernyms": [("animal.n.01", np.array([1.0, 0.0]))],
            "hyponyms": [("kitten.n.01", np.array([0.0, 1.0]))]
        }
        
        tgt_emb_dict = {
            "hyponyms": [("puppy.n.01", np.array([1.0, 0.0]))],
            "hypernyms": [("mammal.n.01", np.array([0.0, 1.0]))]
        }
        
        with patch.object(self.embedding_helper, 'get_embedding_similarities') as mock_similarities:
            mock_similarities.side_effect = [
                np.array([[0.9]]),  # hypernyms -> hyponyms similarity
                np.array([[0.8]])   # hyponyms -> hypernyms similarity
            ]
            
            result = self.embedding_helper.get_top_k_aligned_lex_rel_pairs(
                src_tgt_rel_map, src_emb_dict, tgt_emb_dict, beam_width=2
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

    def test_get_top_k_aligned_lex_rel_pairs_empty_relations(self):
        """Test get_top_k_aligned_lex_rel_pairs with missing relations"""
        src_tgt_rel_map = {
            "hypernyms": "hyponyms",
            "missing_relation": "also_missing"
        }
        
        src_emb_dict = {"hypernyms": [("animal.n.01", np.array([1.0]))]}
        tgt_emb_dict = {"hyponyms": [("puppy.n.01", np.array([1.0]))]}
        
        with patch.object(self.embedding_helper, 'get_embedding_similarities') as mock_similarities:
            mock_similarities.return_value = np.array([[0.9]])
            
            result = self.embedding_helper.get_top_k_aligned_lex_rel_pairs(
                src_tgt_rel_map, src_emb_dict, tgt_emb_dict, beam_width=3
            )
            
            # Should only have one result (missing relations are skipped)
            self.assertEqual(len(result), 1)

    def test_get_top_k_aligned_lex_rel_pairs_zero_beam_width(self):
        """Test get_top_k_aligned_lex_rel_pairs with beam_width <= 0"""
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

    def test_get_new_beams_from_embeddings(self):
        """Test get_new_beams_from_embeddings method"""
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

    def test_get_new_beams_from_embeddings_invalid_synsets(self):
        """Test get_new_beams_from_embeddings with invalid synset names"""
        mock_graph = Mock()
        mock_wn_module = Mock()
        mock_wn_module.synset.side_effect = Exception("Invalid synset")
        
        mock_model = {}
        
        result = self.embedding_helper.get_new_beams_from_embeddings(
            mock_graph, "invalid.synset", "another.invalid", mock_wn_module, mock_model
        )
        
        self.assertEqual(result, [])

    def test_build_gloss_seed_nodes_from_predicate_subjects_mode(self):
        """Test build_gloss_seed_nodes_from_predicate with subjects mode"""
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

    def test_build_gloss_seed_nodes_from_predicate_with_top_k_branch_fn(self):
        """Test build_gloss_seed_nodes_from_predicate with top_k_branch_fn"""
        mock_pred_syn = Mock()
        mock_pred_syn.definition.return_value = "test definition"
        
        mock_wn_module = Mock()
        mock_wn_module.NOUN = 'n'
        
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
        
        mock_top_k_branch_fn = Mock()
        mock_top_k_branch_fn.return_value = [
            ((mock_subject_synset, "relation"), ("target", "relation"), 0.9)
        ]
        
        result = self.embedding_helper.build_gloss_seed_nodes_from_predicate(
            mock_pred_syn, mock_wn_module, mock_nlp_func,
            mode="subjects", extract_subjects_fn=mock_extract_subjects_fn,
            top_k_branch_fn=mock_top_k_branch_fn, target_synsets=["target"],
            max_sample_size=5, beam_width=3
        )
        
        self.assertIsInstance(result, set)
        self.assertIn("subject.n.01", result)

    def test_build_gloss_seed_nodes_from_predicate_fallback_mode(self):
        """Test build_gloss_seed_nodes_from_predicate with fallback to nouns"""
        mock_pred_syn = Mock()
        mock_pred_syn.definition.return_value = "test definition"
        
        mock_wn_module = Mock()
        mock_nlp_func = Mock()
        
        mock_noun_token = Mock()
        mock_noun_token.pos_ = "NOUN"
        mock_noun_token.text = "noun"
        
        mock_verb_token = Mock()
        mock_verb_token.pos_ = "VERB"
        
        mock_doc = [mock_noun_token, mock_verb_token]
        mock_nlp_func.return_value = mock_doc
        
        mock_wn_module.synsets.return_value = []
        
        result = self.embedding_helper.build_gloss_seed_nodes_from_predicate(
            mock_pred_syn, mock_wn_module, mock_nlp_func,
            mode="unknown_mode"  # Will fallback to noun extraction
        )
        
        self.assertIsInstance(result, set)
        # Should use fallback noun extraction
        mock_wn_module.synsets.assert_called()


class TestEmbeddingHelperEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for EmbeddingHelper"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize mock factory
        self.mock_factory = EmbeddingHelperMockFactory()
        
        # Create embedding helper and edge case mock
        self.embedding_helper = EmbeddingHelper()
        self.edge_case_mock = self.mock_factory('MockEmbeddingHelperEdgeCases')

    def test_get_embedding_similarities_large_matrices(self):
        """Test get_embedding_similarities with larger input matrices"""
        # Create larger embedding lists
        rel_embs_1 = [(f"synset{i}", np.random.rand(10)) for i in range(50)]
        rel_embs_2 = [(f"target{i}", np.random.rand(10)) for i in range(30)]
        
        result = self.embedding_helper.get_embedding_similarities(rel_embs_1, rel_embs_2)
        
        self.assertEqual(result.shape, (50, 30))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result >= -1.0))  # Cosine similarity >= -1
        self.assertTrue(np.all(result <= 1.0))   # Cosine similarity <= 1

    def test_get_top_k_aligned_lex_rel_pairs_index_error_handling(self):
        """Test get_top_k_aligned_lex_rel_pairs handles IndexError properly"""
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
        """Test embed_lexical_relations covers all expected relation types"""
        mock_synset = Mock()
        
        # Create mock related synsets for each relation type
        relation_types = [
            'part_holonyms', 'substance_holonyms', 'member_holonyms',
            'part_meronyms', 'substance_meronyms', 'member_meronyms',
            'hypernyms', 'hyponyms', 'entailments', 'causes',
            'also_sees', 'verb_groups'
        ]
        
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

    def test_get_synset_embedding_centroid_mixed_availability(self):
        """Test get_synset_embedding_centroid with mixed lemma availability"""
        mock_synset = Mock()
        
        mock_lemmas = []
        for lemma_text in ["available", "not_available", "multi word", "partial multi"]:
            mock_lemma = Mock()
            mock_lemma.name.return_value = lemma_text
            mock_lemmas.append(mock_lemma)
        
        mock_synset.lemmas.return_value = mock_lemmas
        
        mock_model = {
            "available": np.array([1.0, 1.0]),
            "multi": np.array([2.0, 2.0]),
            "word": np.array([3.0, 3.0]),
            "partial": np.array([4.0, 4.0])
            # "not_available" and "multi" are missing
        }
        
        result = self.embedding_helper.get_synset_embedding_centroid(mock_synset, mock_model)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))
        # Should be mean of: [1,1] (available), [1,1] (not available->available), 
        # [2.5,2.5] (mean of multi+word), [3,3] (mean of partial+multi)
        # = ([1+1+2.5+3]/4, [1+1+2.5+3]/4) = (1.875, 1.875)
        expected = np.array([1.875, 1.875])
        np.testing.assert_array_almost_equal(result, expected)


class TestEmbeddingHelperIntegration(unittest.TestCase):
    """Integration tests for EmbeddingHelper with realistic scenarios"""
    
    def setUp(self):
        """Set up for integration testing"""
        # Initialize mock factory for integration tests
        self.mock_factory = EmbeddingHelperMockFactory()
        self.integration_mock = self.mock_factory('MockEmbeddingHelperIntegration')
        
        self.embedding_helper = EmbeddingHelper()

    def test_full_embedding_workflow(self):
        """Test complete embedding workflow from synset to similarity pairs"""
        # Mock a realistic synset
        mock_synset = Mock()
        mock_synset.name.return_value = "cat.n.01"
        
        # Mock lemmas
        mock_lemma1 = Mock()
        mock_lemma1.name.return_value = "cat"
        mock_lemma2 = Mock()
        mock_lemma2.name.return_value = "feline"
        mock_synset.lemmas.return_value = [mock_lemma1, mock_lemma2]
        
        # Mock related synsets
        mock_hypernym = Mock()
        mock_hypernym.name.return_value = "mammal.n.01"
        mock_hypernym.lemmas.return_value = [Mock()]
        mock_hypernym.lemmas.return_value[0].name.return_value = "mammal"
        
        mock_hyponym = Mock()
        mock_hyponym.name.return_value = "kitten.n.01"
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
        
        # Mock embedding model
        mock_model = {
            "cat": np.array([1.0, 0.0, 0.0]),
            "feline": np.array([0.9, 0.1, 0.0]),
            "mammal": np.array([0.5, 0.5, 0.0]),
            "kitten": np.array([0.8, 0.0, 0.2])
        }
        
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
        self.assertTrue(0.0 <= similarities[0, 0] <= 1.0)  # Valid cosine similarity

    def test_realistic_beam_generation_scenario(self):
        """Test realistic beam generation with multiple synsets and relations"""
        # Mock WordNet module and synsets
        mock_wn = Mock()
        
        mock_cat = Mock()
        mock_cat.name.return_value = "cat.n.01"
        mock_dog = Mock()
        mock_dog.name.return_value = "dog.n.01"
        
        mock_wn.synset.side_effect = [mock_cat, mock_dog]
        
        # Mock embedding model with realistic word vectors
        mock_model = {
            "cat": np.array([0.1, 0.2, 0.3]),
            "dog": np.array([0.2, 0.1, 0.3]),
            "animal": np.array([0.0, 0.0, 0.5]),
            "mammal": np.array([0.1, 0.1, 0.4])
        }
        
        # Mock lexical relations with realistic structure
        mock_cat_emb = {
            "hypernyms": [("mammal.n.01", np.array([0.1, 0.1, 0.4]))],
            "hyponyms": []
        }
        
        mock_dog_emb = {
            "hypernyms": [("mammal.n.01", np.array([0.1, 0.1, 0.4]))],
            "hyponyms": [("puppy.n.01", np.array([0.2, 0.05, 0.25]))]
        }
        
        with patch.object(self.embedding_helper, 'embed_lexical_relations') as mock_embed:
            mock_embed.side_effect = [mock_cat_emb, mock_dog_emb]
            
            result = self.embedding_helper.get_new_beams_from_embeddings(
                Mock(), "cat.n.01", "dog.n.01", mock_wn, mock_model, beam_width=5
            )
            
            self.assertIsInstance(result, list)
            self.assertLessEqual(len(result), 5)  # Should respect beam width
            
            # Check result structure
            for beam_item in result:
                self.assertIsInstance(beam_item, tuple)
                self.assertEqual(len(beam_item), 3)
                self.assertIsInstance(beam_item[2], (int, float))


if __name__ == '__main__':
    unittest.main()