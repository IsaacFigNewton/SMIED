"""
Unit tests for the SMIED class.
"""

import unittest
from unittest.mock import patch, call
import sys
import os
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.SMIED import SMIED, ISMIEDPipeline
from tests.mocks.smied_mocks import SMIEDMockFactory


class TestISMIEDPipeline(unittest.TestCase):
    """Test the ISMIEDPipeline interface."""
    
    def test_interface_cannot_be_instantiated(self):
        """Test that the abstract interface cannot be instantiated."""
        with self.assertRaises(TypeError):
            ISMIEDPipeline()
    
    def test_interface_methods_defined(self):
        """Test that all interface methods are defined."""
        # Check that abstract methods are defined
        self.assertTrue(hasattr(ISMIEDPipeline, 'reinitialize'))
        self.assertTrue(hasattr(ISMIEDPipeline, 'analyze_triple'))
        self.assertTrue(hasattr(ISMIEDPipeline, 'get_synsets'))
        self.assertTrue(hasattr(ISMIEDPipeline, 'display_results'))


class TestSMIED(unittest.TestCase):
    """Test the SMIED class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize mock factory
        self.mock_factory = SMIEDMockFactory()
        # Create SMIED instance with mocked components
        with patch('smied.SMIED.nltk'), \
             patch('smied.SMIED.wn'), \
             patch('smied.SMIED.SemanticDecomposer'):
            self.smied = SMIED(nlp_model=None, auto_download=False)
    
    def test_initialization(self):
        """Test SMIED initialization."""
        smied = SMIED(
            nlp_model="test_model",
            embedding_model="test_embedding",
            auto_download=False
        )
        
        self.assertEqual(smied.nlp_model_name, "test_model")
        self.assertEqual(smied.embedding_model, "test_embedding")
        self.assertFalse(smied.auto_download)
        # NLP will be None because test_model doesn't exist
        self.assertIsNone(smied.nlp)
        # Decomposer should be initialized immediately
        self.assertIsNotNone(smied.decomposer)
        # Synset graph is None by default (build_graph_on_init=False)
        self.assertIsNone(smied.synset_graph)
    
    def test_initialization_on_construct(self):
        """Test that initialization happens in constructor."""
        with patch('smied.SMIED.nltk') as mock_nltk, \
             patch('spacy.load') as mock_spacy_load, \
             patch('smied.SMIED.SemanticDecomposer') as mock_decomposer:
            
            mock_spacy_load.return_value = self.mock_factory('MockSpacy')
            
            # Initialization happens during construction
            smied = SMIED(nlp_model="en_core_web_sm", auto_download=True, build_graph_on_init=False)
            
            # Check NLTK downloads were called during __init__
            mock_nltk.download.assert_any_call('wordnet', quiet=True)
            mock_nltk.download.assert_any_call('omw-1.4', quiet=True)
            
            # Check spaCy model was loaded during __init__
            mock_spacy_load.assert_called_with("en_core_web_sm")
            
            # Check decomposer was initialized during __init__
            mock_decomposer.assert_called_once()
            
            # Check components are set
            self.assertIsNotNone(smied.decomposer)
    
    def test_reinitialize(self):
        """Test reinitialize method."""
        with patch('smied.SMIED.nltk'), \
             patch('spacy.load') as mock_spacy_load, \
             patch('smied.SMIED.SemanticDecomposer') as mock_decomposer:
            
            mock_spacy_load.return_value = self.mock_factory('MockSpacy')
            
            smied = SMIED(auto_download=False)
            initial_decomposer_call_count = mock_decomposer.call_count
            
            # Reinitialize with new model
            smied.reinitialize(nlp_model="new_model")
            
            # Check that decomposer was created again
            self.assertEqual(mock_decomposer.call_count, initial_decomposer_call_count + 1)
            
            # Check synset graph was cleared
            self.assertIsNone(smied.synset_graph)
    
    def test_setup_nlp_success(self):
        """Test successful NLP setup."""
        with patch('spacy.load') as mock_spacy_load:
            mock_nlp = self.mock_factory('MockSpacy')
            mock_spacy_load.return_value = mock_nlp
            
            smied = SMIED(nlp_model="test_model", auto_download=False)
            
            # spacy.load should have been called during initialization
            mock_spacy_load.assert_called_with("test_model")
            self.assertEqual(smied.nlp, mock_nlp)
    
    def test_setup_nlp_failure(self):
        """Test NLP setup when model not found."""
        with patch('spacy.load') as mock_spacy_load:
            mock_spacy_load.side_effect = OSError("Model not found")
            
            smied = SMIED(nlp_model="test_model")
            with patch('builtins.print') as mock_print:
                result = smied._setup_nlp()
                
                self.assertIsNone(result)
                # Check warning was printed
                mock_print.assert_any_call("Warning: spaCy 'test_model' model not found.")
    
    def test_setup_nlp_no_model(self):
        """Test NLP setup with no model specified."""
        smied = SMIED(nlp_model=None)
        result = smied._setup_nlp()
        self.assertIsNone(result)
    
    def test_build_synset_graph(self):
        """Test synset graph building."""
        with patch('smied.SMIED.SemanticDecomposer') as mock_decomposer_class:
            mock_decomposer = self.mock_factory('MockSemanticDecomposer')
            mock_graph = self.mock_factory('MockGraph')
            mock_graph.number_of_nodes.return_value = 100
            mock_graph.number_of_edges.return_value = 200
            mock_decomposer.build_synset_graph.return_value = mock_graph
            mock_decomposer_class.return_value = mock_decomposer
            
            smied = SMIED(nlp_model=None, auto_download=False)
            
            with patch('builtins.print'):
                result = smied.build_synset_graph()
            
            self.assertEqual(result, mock_graph)
            self.assertEqual(smied.synset_graph, mock_graph)
            mock_decomposer.build_synset_graph.assert_called_once()
    
    def test_build_synset_graph_cached(self):
        """Test that graph is cached after first build."""
        with patch('smied.SMIED.SemanticDecomposer'):
            smied = SMIED(nlp_model=None, auto_download=False)
            
            mock_graph = self.mock_factory('MockGraph')
            smied.synset_graph = mock_graph
            
            result = smied.build_synset_graph(verbose=False)
            
            self.assertEqual(result, mock_graph)
    
    def test_get_synsets_with_pos(self):
        """Test getting synsets with POS specified."""
        with patch('smied.SMIED.wn') as mock_wn:
            mock_synsets = [self.mock_factory('MockSynset'), self.mock_factory('MockSynset')]
            mock_wn.synsets.return_value = mock_synsets
            mock_wn.NOUN = 'n'
            
            smied = SMIED()
            result = smied.get_synsets("cat", pos=mock_wn.NOUN)
            
            self.assertEqual(result, mock_synsets)
            mock_wn.synsets.assert_called_once_with("cat", pos='n')
    
    def test_get_synsets_without_pos(self):
        """Test getting synsets without POS specified."""
        with patch('smied.SMIED.wn') as mock_wn:
            mock_synsets = [self.mock_factory('MockSynset'), self.mock_factory('MockSynset')]
            mock_wn.synsets.return_value = mock_synsets
            
            smied = SMIED()
            result = smied.get_synsets("cat")
            
            self.assertEqual(result, mock_synsets)
            mock_wn.synsets.assert_called_once_with("cat")
    
    def test_analyze_triple_success(self):
        """Test successful triple analysis."""
        with patch('smied.SMIED.SemanticDecomposer') as mock_decomposer_class, \
             patch('smied.SMIED.wn') as mock_wn, \
             patch('builtins.print'):
            
            # Set up mocks
            mock_decomposer = self.mock_factory('MockSemanticDecomposer')
            mock_decomposer_class.return_value = mock_decomposer
            
            mock_graph = self.mock_factory('MockGraph')
            mock_graph.number_of_nodes.return_value = 100
            mock_graph.number_of_edges.return_value = 200
            mock_decomposer.build_synset_graph.return_value = mock_graph
            
            mock_path = [self.mock_factory('MockSynset'), self.mock_factory('MockSynset')]
            mock_predicate = self.mock_factory('MockSynset')
            mock_decomposer.find_connected_shortest_paths.return_value = (
                mock_path, mock_path, mock_predicate
            )
            
            # Create SMIED instance
            smied = SMIED(nlp_model=None)
            
            # Analyze triple
            result = smied.analyze_triple("cat", "jump", "mat", verbose=False)
            
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)
            mock_decomposer.find_connected_shortest_paths.assert_called_once()
    
    def test_analyze_triple_with_verbose(self):
        """Test triple analysis with verbose output."""
        with patch('smied.SMIED.SemanticDecomposer') as mock_decomposer_class, \
             patch('smied.SMIED.wn') as mock_wn, \
             patch('builtins.print') as mock_print:
            
            # Set up mocks
            mock_decomposer = self.mock_factory('MockSemanticDecomposer')
            mock_decomposer_class.return_value = mock_decomposer
            
            mock_graph = self.mock_factory('MockGraph')
            mock_graph.number_of_nodes.return_value = 100
            mock_graph.number_of_edges.return_value = 200
            mock_decomposer.build_synset_graph.return_value = mock_graph
            
            mock_decomposer.find_connected_shortest_paths.return_value = (
                None, None, None
            )
            
            mock_wn.synsets.return_value = []
            mock_wn.NOUN = 'n'
            mock_wn.VERB = 'v'
            
            # Create SMIED instance
            smied = SMIED(nlp_model=None)
            
            # Analyze triple with verbose
            smied.analyze_triple("cat", "jump", "mat", verbose=True)
            
            # Check that output was printed
            mock_print.assert_called()
            print_calls = [str(call) for call in mock_print.call_args_list]
            self.assertTrue(any("SEMANTIC DECOMPOSITION ANALYSIS" in str(call) 
                              for call in print_calls))
    
    def test_analyze_triple_exception(self):
        """Test triple analysis with exception."""
        with patch('smied.SMIED.SemanticDecomposer') as mock_decomposer_class:
            mock_decomposer = self.mock_factory('MockSemanticDecomposer')
            mock_decomposer_class.return_value = mock_decomposer
            mock_decomposer.build_synset_graph.side_effect = Exception("Test error")
            
            smied = SMIED(nlp_model=None)
            
            with self.assertRaises(Exception):
                smied.analyze_triple("cat", "jump", "mat", verbose=False)
    
    def test_display_results_with_path(self):
        """Test displaying results when path is found."""
        with patch('smied.SMIED.SemanticDecomposer') as mock_decomposer_class, \
             patch('builtins.print') as mock_print:
            
            from unittest.mock import Mock
            mock_decomposer_class.show_connected_paths = Mock()
            
            # Create mock paths and predicate
            mock_synset = self.mock_factory('MockSynset')
            mock_synset.name.return_value = "test.n.01"
            mock_synset.definition.return_value = "test definition"
            
            subject_path = [mock_synset]
            object_path = [mock_synset]
            predicate = mock_synset
            
            smied = SMIED()
            smied.display_results(
                subject_path, object_path, predicate,
                "cat", "jump", "mat"
            )
            
            # Check success message was printed
            mock_print.assert_any_call("SUCCESS: Found connected semantic paths!")
            mock_print.assert_any_call()
            
            # Check show_connected_paths was called
            mock_decomposer_class.show_connected_paths.assert_called_once()
    
    def test_display_results_no_path(self):
        """Test displaying results when no path is found."""
        with patch('builtins.print') as mock_print, \
             patch.object(SMIED, '_show_fallback_relationships') as mock_fallback:
            
            smied = SMIED()
            smied.display_results(None, None, None, "cat", "jump", "mat")
            
            # Check no path message was printed
            mock_print.assert_any_call("No connected semantic path found.")
            
            # Check fallback relationships were shown
            mock_fallback.assert_called_once_with("cat", "jump", "mat")
    
    def test_calculate_similarity_path(self):
        """Test calculating path similarity between words."""
        with patch('smied.SMIED.wn') as mock_wn:
            mock_synset1 = self.mock_factory('MockSynset')
            mock_synset2 = self.mock_factory('MockSynset')
            mock_synset1.path_similarity.return_value = 0.5
            
            mock_wn.synsets.side_effect = [[mock_synset1], [mock_synset2]]
            
            smied = SMIED()
            result = smied.calculate_similarity("cat", "dog", method="path")
            
            self.assertEqual(result, 0.5)
            mock_synset1.path_similarity.assert_called_once_with(mock_synset2)
    
    def test_calculate_similarity_wu_palmer(self):
        """Test calculating Wu-Palmer similarity."""
        with patch('smied.SMIED.wn') as mock_wn:
            mock_synset1 = self.mock_factory('MockSynset')
            mock_synset2 = self.mock_factory('MockSynset')
            mock_synset1.wup_similarity.return_value = 0.8
            
            mock_wn.synsets.side_effect = [[mock_synset1], [mock_synset2]]
            
            smied = SMIED()
            result = smied.calculate_similarity("cat", "dog", method="wu_palmer")
            
            self.assertEqual(result, 0.8)
            mock_synset1.wup_similarity.assert_called_once_with(mock_synset2)
    
    def test_calculate_similarity_lch(self):
        """Test calculating LCH similarity."""
        with patch('smied.SMIED.wn') as mock_wn:
            mock_synset1 = self.mock_factory('MockSynset')
            mock_synset2 = self.mock_factory('MockSynset')
            mock_synset1.lch_similarity.return_value = 2.5
            
            mock_wn.synsets.side_effect = [[mock_synset1], [mock_synset2]]
            
            smied = SMIED()
            result = smied.calculate_similarity("cat", "dog", method="lch")
            
            self.assertEqual(result, 2.5)
            mock_synset1.lch_similarity.assert_called_once_with(mock_synset2)
    
    def test_calculate_similarity_no_synsets(self):
        """Test similarity calculation when synsets not found."""
        with patch('smied.SMIED.wn') as mock_wn:
            mock_wn.synsets.side_effect = [[], [self.mock_factory('MockSynset')]]
            
            smied = SMIED()
            result = smied.calculate_similarity("xyz", "dog")
            
            self.assertIsNone(result)
    
    def test_calculate_similarity_invalid_method(self):
        """Test similarity calculation with invalid method."""
        with patch('smied.SMIED.wn') as mock_wn:
            mock_synset1 = self.mock_factory('MockSynset')
            mock_synset2 = self.mock_factory('MockSynset')
            mock_wn.synsets.side_effect = [[mock_synset1], [mock_synset2]]
            
            smied = SMIED()
            result = smied.calculate_similarity("cat", "dog", method="invalid")
            
            self.assertIsNone(result)
    
    def test_get_word_info(self):
        """Test getting comprehensive word information."""
        with patch('smied.SMIED.wn') as mock_wn:
            # Create mock synset
            mock_synset = self.mock_factory('MockSynset')
            mock_synset.name.return_value = "cat.n.01"
            mock_synset.definition.return_value = "feline mammal"
            mock_synset.pos.return_value = "n"
            mock_synset.examples.return_value = ["The cat sat on the mat"]
            
            mock_lemma = self.mock_factory('MockSynset')
            mock_lemma.name.return_value = "cat"
            mock_synset.lemmas.return_value = [mock_lemma]
            
            mock_hypernym = self.mock_factory('MockSynset')
            mock_hypernym.name.return_value = "feline.n.01"
            mock_synset.hypernyms.return_value = [mock_hypernym]
            
            mock_hyponym = self.mock_factory('MockSynset')
            mock_hyponym.name.return_value = "kitten.n.01"
            mock_synset.hyponyms.return_value = [mock_hyponym]
            
            mock_wn.synsets.return_value = [mock_synset]
            
            smied = SMIED()
            result = smied.get_word_info("cat")
            
            self.assertEqual(result["word"], "cat")
            self.assertEqual(result["total_senses"], 1)
            self.assertEqual(len(result["synsets"]), 1)
            
            synset_info = result["synsets"][0]
            self.assertEqual(synset_info["name"], "cat.n.01")
            self.assertEqual(synset_info["definition"], "feline mammal")
            self.assertEqual(synset_info["pos"], "n")
            self.assertEqual(synset_info["examples"], ["The cat sat on the mat"])
            self.assertEqual(synset_info["lemmas"], ["cat"])
            self.assertEqual(synset_info["hypernyms"], ["feline.n.01"])
            self.assertEqual(synset_info["hyponyms"], ["kitten.n.01"])
    
    def test_demonstrate_alternative_approaches(self):
        """Test demonstrating alternative approaches."""
        with patch('smied.SMIED.wn') as mock_wn, \
             patch('builtins.print') as mock_print:
            
            # Create mock synsets
            mock_synset = self.mock_factory('MockSynset')
            mock_synset.name.return_value = "test.n.01"
            mock_synset.definition.return_value = "test definition"
            mock_synset.hypernyms.return_value = []
            mock_synset.entailments.return_value = []
            mock_synset.causes.return_value = []
            mock_synset.verb_groups.return_value = []
            
            mock_wn.synsets.return_value = [mock_synset]
            mock_wn.NOUN = 'n'
            mock_wn.VERB = 'v'
            
            smied = SMIED()
            smied.demonstrate_alternative_approaches("cat", "jump", "mat")
            
            # Check that alternative analysis header was printed
            mock_print.assert_any_call("ALTERNATIVE SEMANTIC ANALYSIS")
    
    def test_show_hypernym_path(self):
        """Test showing hypernym path."""
        with patch('builtins.print') as mock_print:
            # Create mock synset chain
            mock_synset1 = self.mock_factory('MockSynset')
            mock_synset1.name.return_value = "cat.n.01"
            mock_synset1.definition.return_value = "feline mammal"
            
            mock_synset2 = self.mock_factory('MockSynset')
            mock_synset2.name.return_value = "feline.n.01"
            mock_synset2.definition.return_value = "cat family"
            mock_synset2.hypernyms.return_value = []
            
            mock_synset1.hypernyms.return_value = [mock_synset2]
            
            smied = SMIED()
            smied._show_hypernym_path(mock_synset1, max_depth=2)
            
            # Check both synsets were printed
            mock_print.assert_any_call("  cat.n.01: feline mammal")
            mock_print.assert_any_call("    feline.n.01: cat family")
    
    def test_show_verb_relations(self):
        """Test showing verb relations."""
        with patch('builtins.print') as mock_print:
            # Create mock synset with relations
            mock_synset = self.mock_factory('MockSynset')
            mock_synset.name.return_value = "jump.v.01"
            
            mock_entailment = self.mock_factory('MockSynset')
            mock_entailment.name.return_value = "move.v.01"
            mock_entailment.definition.return_value = "change position"
            mock_synset.entailments.return_value = [mock_entailment]
            
            mock_synset.causes.return_value = []
            mock_synset.verb_groups.return_value = []
            
            smied = SMIED()
            smied._show_verb_relations(mock_synset, "jump")
            
            # Check verb relations were printed
            mock_print.assert_any_call("\nJump (jump.v.01) verb relations:")
            mock_print.assert_any_call("  Entailments (what it necessarily involves):")
    
    def test_show_fallback_relationships(self):
        """Test showing fallback relationships."""
        with patch('smied.SMIED.wn') as mock_wn, \
             patch('builtins.print') as mock_print:
            
            # Create mock synsets
            mock_synset1 = self.mock_factory('MockSynset')
            mock_synset1.name.return_value = "cat.n.01"
            mock_synset1.path_similarity.return_value = 0.5
            mock_synset1.hypernyms.return_value = []
            
            mock_synset2 = self.mock_factory('MockSynset')
            mock_synset2.name.return_value = "dog.n.01"
            
            mock_wn.synsets.side_effect = [
                [mock_synset1],  # subject
                [],  # predicate
                [mock_synset2]   # object
            ]
            mock_wn.NOUN = 'n'
            mock_wn.VERB = 'v'
            
            smied = SMIED()
            smied._show_fallback_relationships("cat", "jump", "dog")
            
            # Check fallback header was printed
            mock_print.assert_any_call("Individual synset relationships:")


class TestSMIEDIntegration(unittest.TestCase):
    """Integration tests for SMIED class."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        from tests.mocks.smied_mocks import SMIEDMockFactory
        self.mock_factory = SMIEDMockFactory()
        self.integration_mock = self.mock_factory('MockSMIEDIntegration')
    
    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked components."""
        # Create full pipeline mocks using integration mock
        test_mocks = self.integration_mock.create_full_pipeline_mocks()
        
        with patch('smied.SMIED.nltk', test_mocks['mock_nltk']), \
             patch('spacy.load', return_value=test_mocks['mock_nlp']), \
             patch('smied.SMIED.wn', test_mocks['mock_wn']), \
             patch('smied.SMIED.SemanticDecomposer', return_value=test_mocks['mock_decomposer']), \
             patch('builtins.print'):
            
            # Create and use SMIED
            smied = SMIED(nlp_model="en_core_web_sm", auto_download=True, build_graph_on_init=False)
            result = smied.analyze_triple("cat", "jump", "mat", verbose=False)
            
            # Verify result
            self.assertIsNotNone(result)
            subject_path, object_path, predicate = result
            self.assertIsNotNone(subject_path)
            self.assertIsNotNone(object_path)
            self.assertIsNotNone(predicate)
            
            # Verify calls using test_mocks
            test_mocks['mock_nltk'].download.assert_any_call('wordnet', quiet=True)
            test_mocks['mock_decomposer'].build_synset_graph.assert_called_once()
            test_mocks['mock_decomposer'].find_connected_shortest_paths.assert_called_once()
    
    def test_multiple_analyses(self):
        """Test multiple analyses with same instance."""
        with patch('smied.SMIED.SemanticDecomposer') as mock_decomposer_class, \
             patch('smied.SMIED.wn') as mock_wn, \
             patch('builtins.print'):
            
            mock_decomposer = self.mock_factory('MockSemanticDecomposer')
            mock_decomposer_class.return_value = mock_decomposer
            
            mock_graph = self.mock_factory('MockGraph')
            mock_graph.number_of_nodes.return_value = 100
            mock_graph.number_of_edges.return_value = 200
            mock_decomposer.build_synset_graph.return_value = mock_graph
            
            mock_decomposer.find_connected_shortest_paths.return_value = (
                None, None, None
            )
            
            mock_wn.synsets.return_value = []
            mock_wn.NOUN = 'n'
            mock_wn.VERB = 'v'
            
            smied = SMIED(nlp_model=None)
            
            # Perform multiple analyses
            smied.analyze_triple("cat", "jump", "mat", verbose=False)
            smied.analyze_triple("dog", "run", "park", verbose=False)
            smied.analyze_triple("bird", "fly", "tree", verbose=False)
            
            # Graph should only be built once
            mock_decomposer.build_synset_graph.assert_called_once()
            
            # But analysis should be called three times
            self.assertEqual(mock_decomposer.find_connected_shortest_paths.call_count, 3)


if __name__ == '__main__':
    unittest.main()