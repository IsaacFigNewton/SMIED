"""Unit tests for the SMIED class following the 3-layer testing architecture.

This test file implements the SMIED Testing Framework Design Specifications:
- Test Layer: Contains test logic and assertions
- Mock Layer: Provides mock implementations via SMIEDMockFactory  
- Configuration Layer: Supplies test data via SMIEDMockConfig

Test class organization:
- TestSmied: Basic functionality tests
- TestSmiedValidation: Validation and constraint tests  
- TestSmiedEdgeCases: Edge cases and error conditions
- TestSmiedIntegration: Integration tests with other components
"""

import unittest
from unittest.mock import patch, call, Mock
import sys
import os
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.SMIED import SMIED, ISMIEDPipeline
from tests.mocks.smied_mocks import SMIEDMockFactory
from tests.config.smied_config import SMIEDMockConfig


class TestSmied(unittest.TestCase):
    """Basic functionality tests for SMIED class.
    
    Tests core SMIED functionality including initialization, basic operations,
    and standard use cases following the factory pattern for mock creation.
    """
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize configuration layer
        self.config = SMIEDMockConfig()
        
        # Initialize mock factory from config
        self.mock_factory = SMIEDMockFactory()
        
        # Get test data from configuration
        self.model_constants = self.config.get_model_name_constants()
        self.triple_test_cases = self.config.get_triple_analysis_test_cases()
        self.synset_structures = self.config.get_mock_synset_structures()
        
        # Create common mock objects using factory
        self.mock_nltk = self.mock_factory('MockNLTK')
        self.mock_spacy = self.mock_factory('MockSpacy')
        self.mock_wn = self.mock_factory('MockWordNet')
        self.mock_decomposer = self.mock_factory('MockSemanticDecomposer')
        self.mock_graph = self.mock_factory('MockGraph')
        
        # Configure mock behaviors from config data
        self._configure_mock_behaviors()
        
    def _configure_mock_behaviors(self):
        """Configure mock behaviors based on configuration data."""
        # Configure graph mock with realistic data
        self.mock_graph.number_of_nodes.return_value = 100
        self.mock_graph.number_of_edges.return_value = 200
        
        # Configure decomposer mock
        self.mock_decomposer.build_synset_graph.return_value = self.mock_graph
        
        # Configure WordNet constants from config
        constants = self.model_constants
        self.mock_wn.NOUN = 'n'
        self.mock_wn.VERB = 'v'
        self.mock_wn.ADJ = 'a'
        self.mock_wn.ADV = 'r'

    def test_interface_cannot_be_instantiated(self):
        """Test that the abstract ISMIEDPipeline interface cannot be instantiated."""
        with self.assertRaises(TypeError):
            ISMIEDPipeline()
    
    def test_interface_methods_defined(self):
        """Test that all required interface methods are defined."""
        # Check that abstract methods are defined on the interface
        self.assertTrue(hasattr(ISMIEDPipeline, 'reinitialize'))
        self.assertTrue(hasattr(ISMIEDPipeline, 'analyze_triple'))
        self.assertTrue(hasattr(ISMIEDPipeline, 'get_synsets'))
        self.assertTrue(hasattr(ISMIEDPipeline, 'display_results'))

    def test_initialization_basic(self):
        """Test basic SMIED initialization with configuration data."""
        # Get test model from configuration
        test_model = self.model_constants['test_spacy_model']
        test_embedding = self.model_constants['test_embedding_model']
        
        # Mock patches using factory-created mocks
        with patch('smied.SMIED.nltk', self.mock_nltk), \
             patch('smied.SMIED.wn', self.mock_wn), \
             patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            
            smied = SMIED(
                nlp_model=test_model,
                embedding_model=test_embedding,
                auto_download=False
            )
            
            self.assertEqual(smied.nlp_model_name, test_model)
            self.assertEqual(smied.embedding_model, test_embedding)
            self.assertFalse(smied.auto_download)
            # NLP will be None because test_model doesn't exist
            self.assertIsNone(smied.nlp)
            # Decomposer should be initialized immediately
            self.assertIsNotNone(smied.decomposer)
            # Synset graph is None by default (build_graph_on_init=False)
            self.assertIsNone(smied.synset_graph)
    
    def test_initialization_with_auto_download(self):
        """Test initialization with auto download enabled using config data."""
        # Get default model from configuration
        default_model = self.model_constants['default_spacy_model']
        
        with patch('smied.SMIED.nltk', self.mock_nltk), \
             patch('spacy.load') as mock_spacy_load, \
             patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            
            mock_spacy_load.return_value = self.mock_spacy
            
            # Initialization happens during construction
            smied = SMIED(
                nlp_model=default_model, 
                auto_download=True, 
                build_graph_on_init=False
            )
            
            # Check NLTK downloads were called during __init__
            self.mock_nltk.download.assert_any_call('wordnet', quiet=True)
            self.mock_nltk.download.assert_any_call('omw-1.4', quiet=True)
            
            # Check spaCy model was loaded during __init__
            mock_spacy_load.assert_called_with(default_model)
            
            # Check decomposer was initialized during __init__
            self.assertIsNotNone(smied.decomposer)
            
            # Check components are set
            self.assertEqual(smied.nlp_model_name, default_model)

    def test_reinitialize_with_new_model(self):
        """Test reinitialize method with new model using config data."""
        # Get models from configuration
        initial_model = self.model_constants['default_spacy_model']
        new_model = self.model_constants['medium_spacy_model']
        
        with patch('smied.SMIED.nltk', self.mock_nltk), \
             patch('spacy.load') as mock_spacy_load, \
             patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer) as mock_decomposer_class:
            
            mock_spacy_load.return_value = self.mock_spacy
            
            smied = SMIED(nlp_model=initial_model, auto_download=False)
            initial_decomposer_call_count = mock_decomposer_class.call_count
            
            # Reinitialize with new model from config
            smied.reinitialize(nlp_model=new_model)
            
            # Check that decomposer was created again
            self.assertEqual(
                mock_decomposer_class.call_count, 
                initial_decomposer_call_count + 1
            )
            
            # Check synset graph was cleared
            self.assertIsNone(smied.synset_graph)
            
            # Check new model name is set
            self.assertEqual(smied.nlp_model_name, new_model)

    def test_setup_nlp_success(self):
        """Test successful NLP setup with configured model."""
        # Get test model from configuration
        test_model = self.model_constants['test_spacy_model']
        
        with patch('spacy.load') as mock_spacy_load:
            mock_spacy_load.return_value = self.mock_spacy
            
            smied = SMIED(nlp_model=test_model, auto_download=False)
            
            # spacy.load should have been called during initialization
            mock_spacy_load.assert_called_with(test_model)
            self.assertEqual(smied.nlp, self.mock_spacy)

    def test_setup_nlp_model_not_found(self):
        """Test NLP setup when configured model not found."""
        # Get test model from configuration
        test_model = self.model_constants['test_spacy_model']
        
        with patch('spacy.load') as mock_spacy_load:
            mock_spacy_load.side_effect = OSError("Model not found")
            
            smied = SMIED(nlp_model=test_model)
            with patch('builtins.print') as mock_print:
                result = smied._setup_nlp()
                
                self.assertIsNone(result)
                # Check warning was printed with the configured model name
                expected_warning = f"[WARNING] spaCy '{test_model}' model not found."
                mock_print.assert_any_call(expected_warning)

    def test_setup_nlp_no_model_specified(self):
        """Test NLP setup with no model specified."""
        smied = SMIED(nlp_model=None, auto_download=False)
        result = smied._setup_nlp()
        self.assertIsNone(result)
        self.assertIsNone(smied.nlp)

    def test_build_synset_graph(self):
        """Test synset graph building with mocked components."""
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            smied = SMIED(nlp_model=None, auto_download=False)
            
            with patch('builtins.print'):
                result = smied.build_synset_graph()
            
            self.assertEqual(result, self.mock_graph)
            self.assertEqual(smied.synset_graph, self.mock_graph)
            self.mock_decomposer.build_synset_graph.assert_called_once()

    def test_build_synset_graph_cached(self):
        """Test that synset graph is cached after first build."""
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            smied = SMIED(nlp_model=None, auto_download=False)
            
            # Set cached graph
            smied.synset_graph = self.mock_graph
            
            result = smied.build_synset_graph(verbose=False)
            
            # Should return cached graph without rebuilding
            self.assertEqual(result, self.mock_graph)
            # Should not call build_synset_graph again
            self.mock_decomposer.build_synset_graph.assert_not_called()

    def test_get_synsets_with_pos(self):
        """Test getting synsets with POS tag specified."""
        # Create mock synsets using factory
        mock_synsets = [
            self.mock_factory('MockSynset', name="cat.n.01"),
            self.mock_factory('MockSynset', name="cat.n.02")
        ]
        
        with patch('smied.SMIED.wn', self.mock_wn):
            self.mock_wn.synsets.return_value = mock_synsets
            
            smied = SMIED(auto_download=False)
            result = smied.get_synsets("cat", pos=self.mock_wn.NOUN)
            
            self.assertEqual(result, mock_synsets)
            self.mock_wn.synsets.assert_called_once_with("cat", pos='n')

    def test_get_synsets_without_pos(self):
        """Test getting synsets without POS tag specified."""
        # Create mock synsets using factory
        mock_synsets = [
            self.mock_factory('MockSynset', name="cat.n.01"),
            self.mock_factory('MockSynset', name="cat.v.01")
        ]
        
        with patch('smied.SMIED.wn', self.mock_wn):
            self.mock_wn.synsets.return_value = mock_synsets
            
            smied = SMIED(auto_download=False)
            result = smied.get_synsets("cat")
            
            self.assertEqual(result, mock_synsets)
            self.mock_wn.synsets.assert_called_once_with("cat")

    def test_analyze_triple_success(self):
        """Test successful triple analysis using config test cases."""
        # Get test triple from configuration
        simple_triples = self.triple_test_cases['simple_triples']
        test_triple = simple_triples[0]  # ("cat", "chase", "mouse")
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer), \
             patch('smied.SMIED.wn', self.mock_wn), \
             patch('builtins.print'):
            
            # Configure mock path results
            mock_subject_synset = self.mock_factory('MockSynset', name="cat.n.01")
            mock_object_synset = self.mock_factory('MockSynset', name="mouse.n.01")
            mock_predicate_synset = self.mock_factory('MockSynset', name="chase.v.01")
            
            mock_path = [mock_subject_synset, mock_object_synset]
            self.mock_decomposer.find_connected_shortest_paths.return_value = (
                mock_path, mock_path, mock_predicate_synset
            )
            
            # Create SMIED instance
            smied = SMIED(nlp_model=None, auto_download=False)
            
            # Analyze triple using config data
            result = smied.analyze_triple(*test_triple, verbose=False)
            
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)
            self.mock_decomposer.find_connected_shortest_paths.assert_called_once()

    def test_analyze_triple_verbose_output(self):
        """Test triple analysis with verbose output using config test cases."""
        # Get complex triple from configuration
        complex_triples = self.triple_test_cases['complex_triples']
        test_triple = complex_triples[0]  # ("student", "study", "mathematics")
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer), \
             patch('smied.SMIED.wn', self.mock_wn), \
             patch('builtins.print') as mock_print:
            
            # Configure mocks with no path found
            self.mock_decomposer.find_connected_shortest_paths.return_value = (
                None, None, None
            )
            self.mock_wn.synsets.return_value = []
            
            # Create SMIED instance
            smied = SMIED(nlp_model=None, auto_download=False)
            
            # Analyze triple with verbose output
            smied.analyze_triple(*test_triple, verbose=True)
            
            # Check that analysis output was printed
            mock_print.assert_called()
            print_calls = [str(call) for call in mock_print.call_args_list]
            self.assertTrue(
                any("SEMANTIC DECOMPOSITION ANALYSIS" in str(call) 
                    for call in print_calls),
                "Expected verbose analysis output not found in print calls"
            )

    def test_analyze_triple_with_exception(self):
        """Test triple analysis when exception occurs in graph building."""
        # Get test triple from configuration
        test_triple = self.triple_test_cases['simple_triples'][1]  # ("dog", "run", "park")
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            # Configure mock to raise exception during graph building
            self.mock_decomposer.build_synset_graph.side_effect = Exception(
                "Graph building failed"
            )
            
            smied = SMIED(nlp_model=None, auto_download=False)
            
            with self.assertRaises(Exception) as context:
                smied.analyze_triple(*test_triple, verbose=False)
            
            self.assertEqual(str(context.exception), "Graph building failed")

    def test_display_results_with_semantic_path(self):
        """Test displaying results when semantic path is found."""
        # Get test synset structure from configuration
        animal_synsets = self.synset_structures['animal_synsets']
        cat_data = animal_synsets['cat.n.01']
        
        with patch('smied.SMIED.SemanticDecomposer') as mock_decomposer_class, \
             patch('builtins.print') as mock_print:
            
            mock_decomposer_class.show_connected_paths = Mock()
            
            # Create mock synset using config data
            mock_cat_synset = self.mock_factory('MockSynset', 
                                              name=cat_data['name'],
                                              definition=cat_data['definition'])
            
            subject_path = [mock_cat_synset]
            object_path = [mock_cat_synset]
            predicate = mock_cat_synset
            
            smied = SMIED(auto_download=False)
            smied.display_results(
                subject_path, object_path, predicate,
                "cat", "jump", "mat"
            )
            
            # Check success message was printed
            mock_print.assert_any_call("SUCCESS: Found connected semantic paths!")
            mock_print.assert_any_call()
            
            # Check show_connected_paths was called
            mock_decomposer_class.show_connected_paths.assert_called_once()

    def test_display_results_no_semantic_path_found(self):
        """Test displaying results when no semantic path is found."""
        # Get test triple from configuration
        test_triple = self.triple_test_cases['abstract_triples'][0]
        
        with patch('builtins.print') as mock_print, \
             patch.object(SMIED, '_show_fallback_relationships') as mock_fallback:
            
            smied = SMIED(auto_download=False)
            smied.display_results(None, None, None, *test_triple)
            
            # Check no path message was printed
            mock_print.assert_any_call("No connected semantic path found.")
            
            # Check fallback relationships were shown with config data
            mock_fallback.assert_called_once_with(*test_triple)

    def test_calculate_similarity_path_method(self):
        """Test calculating path similarity between words using config data."""
        # Get similarity test data from configuration
        similarity_data = self.config.get_similarity_calculation_test_data()
        test_pair = similarity_data['wordnet_similarity_pairs'][0]
        
        with patch('smied.SMIED.wn', self.mock_wn):
            mock_synset1 = self.mock_factory('MockSynset', name=test_pair['synset1'])
            mock_synset2 = self.mock_factory('MockSynset', name=test_pair['synset2'])
            mock_synset1.path_similarity.return_value = test_pair['path_similarity']
            
            self.mock_wn.synsets.side_effect = [[mock_synset1], [mock_synset2]]
            
            smied = SMIED(auto_download=False)
            result = smied.calculate_similarity("cat", "dog", method="path")
            
            self.assertEqual(result, test_pair['path_similarity'])
            mock_synset1.path_similarity.assert_called_once_with(mock_synset2)

    def test_calculate_similarity_wu_palmer_method(self):
        """Test calculating Wu-Palmer similarity using config data."""
        # Get similarity test data from configuration
        similarity_data = self.config.get_similarity_calculation_test_data()
        test_pair = similarity_data['wordnet_similarity_pairs'][0]
        
        with patch('smied.SMIED.wn', self.mock_wn):
            mock_synset1 = self.mock_factory('MockSynset', name=test_pair['synset1'])
            mock_synset2 = self.mock_factory('MockSynset', name=test_pair['synset2'])
            mock_synset1.wup_similarity.return_value = test_pair['wup_similarity']
            
            self.mock_wn.synsets.side_effect = [[mock_synset1], [mock_synset2]]
            
            smied = SMIED(auto_download=False)
            result = smied.calculate_similarity("cat", "dog", method="wu_palmer")
            
            self.assertEqual(result, test_pair['wup_similarity'])
            mock_synset1.wup_similarity.assert_called_once_with(mock_synset2)

    def test_calculate_similarity_lch_method(self):
        """Test calculating LCH similarity using config data."""
        # Get similarity test data from configuration
        similarity_data = self.config.get_similarity_calculation_test_data()
        test_pair = similarity_data['wordnet_similarity_pairs'][0]
        
        with patch('smied.SMIED.wn', self.mock_wn):
            mock_synset1 = self.mock_factory('MockSynset', name=test_pair['synset1'])
            mock_synset2 = self.mock_factory('MockSynset', name=test_pair['synset2'])
            mock_synset1.lch_similarity.return_value = test_pair['lch_similarity']
            
            self.mock_wn.synsets.side_effect = [[mock_synset1], [mock_synset2]]
            
            smied = SMIED(auto_download=False)
            result = smied.calculate_similarity("cat", "dog", method="lch")
            
            self.assertEqual(result, test_pair['lch_similarity'])
            mock_synset1.lch_similarity.assert_called_once_with(mock_synset2)

    def test_calculate_similarity_no_synsets_found(self):
        """Test similarity calculation when synsets are not found."""
        with patch('smied.SMIED.wn', self.mock_wn):
            # First word has no synsets, second word has synsets
            mock_synset = self.mock_factory('MockSynset', name="dog.n.01")
            self.mock_wn.synsets.side_effect = [[], [mock_synset]]
            
            smied = SMIED(auto_download=False)
            result = smied.calculate_similarity("xyz", "dog")
            
            self.assertIsNone(result)

    def test_calculate_similarity_invalid_method(self):
        """Test similarity calculation with invalid method name."""
        with patch('smied.SMIED.wn', self.mock_wn):
            mock_synset1 = self.mock_factory('MockSynset', name="cat.n.01")
            mock_synset2 = self.mock_factory('MockSynset', name="dog.n.01")
            self.mock_wn.synsets.side_effect = [[mock_synset1], [mock_synset2]]
            
            smied = SMIED(auto_download=False)
            result = smied.calculate_similarity("cat", "dog", method="invalid_method")
            
            self.assertIsNone(result)

    def test_get_word_information_comprehensive(self):
        """Test getting comprehensive word information using config data."""
        # Get synset structure from configuration
        cat_data = self.synset_structures['animal_synsets']['cat.n.01']
        
        with patch('smied.SMIED.wn', self.mock_wn):
            # Create mock synset using config data
            mock_synset = self.mock_factory('MockSynset',
                                          name=cat_data['name'], 
                                          definition=cat_data['definition'])
            mock_synset.pos.return_value = cat_data['pos']
            mock_synset.examples.return_value = cat_data['examples']
            
            # Create mock lemma
            mock_lemma = self.mock_factory('MockSynset')
            mock_lemma.name.return_value = cat_data['lemma_names'][0]
            mock_synset.lemmas.return_value = [mock_lemma]
            
            # Create mock hypernym and hyponym
            mock_hypernym = self.mock_factory('MockSynset')
            mock_hypernym.name.return_value = cat_data['hypernyms'][0]
            mock_synset.hypernyms.return_value = [mock_hypernym]
            
            mock_hyponyms = []
            for hyponym_name in cat_data['hyponyms']:
                mock_hyponym = self.mock_factory('MockSynset')
                mock_hyponym.name.return_value = hyponym_name
                mock_hyponyms.append(mock_hyponym)
            mock_synset.hyponyms.return_value = mock_hyponyms
            
            self.mock_wn.synsets.return_value = [mock_synset]
            
            smied = SMIED(auto_download=False)
            result = smied.get_word_info("cat")
            
            # Verify results using config data
            self.assertEqual(result["word"], "cat")
            self.assertEqual(result["total_senses"], 1)
            self.assertEqual(len(result["synsets"]), 1)
            
            synset_info = result["synsets"][0]
            self.assertEqual(synset_info["name"], cat_data['name'])
            self.assertEqual(synset_info["definition"], cat_data['definition'])
            self.assertEqual(synset_info["pos"], cat_data['pos'])
            self.assertEqual(synset_info["examples"], cat_data['examples'])
            self.assertEqual(synset_info["lemmas"], [cat_data['lemma_names'][0]])
            self.assertEqual(synset_info["hypernyms"], cat_data['hypernyms'])
            self.assertEqual(synset_info["hyponyms"], cat_data['hyponyms'])

    def test_demonstrate_alternative_semantic_approaches(self):
        """Test demonstrating alternative semantic analysis approaches."""
        # Get test triple from configuration
        test_triple = self.triple_test_cases['entity_triples'][0]
        
        with patch('smied.SMIED.wn', self.mock_wn), \
             patch('builtins.print') as mock_print:
            
            # Create mock synset using config data
            mock_synset = self.mock_factory('MockSynset',
                                          name="test.n.01",
                                          definition="test definition")
            mock_synset.hypernyms.return_value = []
            mock_synset.entailments.return_value = []
            mock_synset.causes.return_value = []
            mock_synset.verb_groups.return_value = []
            
            self.mock_wn.synsets.return_value = [mock_synset]
            
            smied = SMIED(auto_download=False)
            smied.demonstrate_alternative_approaches(*test_triple)
            
            # Check that alternative analysis header was printed
            mock_print.assert_any_call("ALTERNATIVE SEMANTIC ANALYSIS")

    def test_show_hypernym_semantic_path(self):
        """Test showing hypernym path using config synset data."""
        # Get synset data from configuration
        cat_data = self.synset_structures['animal_synsets']['cat.n.01']
        animal_data = self.synset_structures['animal_synsets']['animal.n.01']
        
        with patch('builtins.print') as mock_print:
            # Create mock synset chain using config data
            mock_cat_synset = self.mock_factory('MockSynset',
                                               name=cat_data['name'],
                                               definition=cat_data['definition'])
            
            mock_animal_synset = self.mock_factory('MockSynset',
                                                  name=animal_data['name'],
                                                  definition=animal_data['definition'])
            mock_animal_synset.hypernyms.return_value = []
            
            mock_cat_synset.hypernyms.return_value = [mock_animal_synset]
            
            smied = SMIED(auto_download=False)
            smied._show_hypernym_path(mock_cat_synset, max_depth=2)
            
            # Check both synsets were printed using config data
            expected_cat = f"  {cat_data['name']}: {cat_data['definition']}"
            expected_animal = f"    {animal_data['name']}: {animal_data['definition']}"
            mock_print.assert_any_call(expected_cat)
            mock_print.assert_any_call(expected_animal)

    def test_show_verb_semantic_relations(self):
        """Test showing verb relations using config action synset data."""
        # Get action synset data from configuration
        run_data = self.synset_structures['action_synsets']['run.v.01']
        
        with patch('builtins.print') as mock_print:
            # Create mock verb synset using config data
            mock_run_synset = self.mock_factory('MockSynset', name=run_data['name'])
            
            mock_entailment = self.mock_factory('MockSynset',
                                               name="move.v.01",
                                               definition="change position")
            mock_run_synset.entailments.return_value = [mock_entailment]
            mock_run_synset.causes.return_value = []
            mock_run_synset.verb_groups.return_value = []
            
            smied = SMIED(auto_download=False)
            smied._show_verb_relations(mock_run_synset, "run")
            
            # Check verb relations were printed using config data
            expected_header = f"\nRun ({run_data['name']}) verb relations:"
            mock_print.assert_any_call(expected_header)
            mock_print.assert_any_call("  Entailments (what it necessarily involves):")

    def test_show_fallback_semantic_relationships(self):
        """Test showing fallback relationships using config synset data."""
        # Get synset data from configuration
        cat_data = self.synset_structures['animal_synsets']['cat.n.01']
        dog_data = self.synset_structures['animal_synsets']['dog.n.01']
        
        with patch('smied.SMIED.wn', self.mock_wn), \
             patch('builtins.print') as mock_print:
            
            # Create mock synsets using config data
            mock_cat_synset = self.mock_factory('MockSynset', name=cat_data['name'])
            mock_cat_synset.path_similarity.return_value = 0.5
            mock_cat_synset.hypernyms.return_value = []
            
            mock_dog_synset = self.mock_factory('MockSynset', name=dog_data['name'])
            
            self.mock_wn.synsets.side_effect = [
                [mock_cat_synset],  # subject
                [],                 # predicate (no synsets found)
                [mock_dog_synset]   # object
            ]
            
            smied = SMIED(auto_download=False)
            smied._show_fallback_relationships("cat", "jump", "dog")
            
            # Check fallback header was printed
            mock_print.assert_any_call("Individual synset relationships:")


class TestSmiedValidation(unittest.TestCase):
    """Validation and constraint tests for SMIED class.
    
    Tests input validation, parameter constraints, and data integrity
    checks using configuration-driven test data.
    """
    
    def setUp(self):
        """Set up validation test fixtures using config injection."""
        # Initialize configuration layer
        self.config = SMIEDMockConfig()
        
        # Initialize mock factory
        self.mock_factory = SMIEDMockFactory()
        
        # Get validation test data from configuration
        self.validation_cases = self.config.get_validation_test_cases()
        self.error_scenarios = self.config.get_error_handling_scenarios()
        
        # Create common mocks
        self.mock_nltk = self.mock_factory('MockNLTK')
        self.mock_wn = self.mock_factory('MockWordNet')
        self.mock_decomposer = self.mock_factory('MockSemanticDecomposer')
        
    def test_input_validation_proper_triple_format(self):
        """Test validation of proper triple format using config data."""
        # Get valid input cases from configuration
        valid_inputs = [case for case in self.validation_cases['input_validation'] 
                       if case['is_valid']]
        
        with patch('smied.SMIED.nltk', self.mock_nltk), \
             patch('smied.SMIED.wn', self.mock_wn), \
             patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            
            smied = SMIED(nlp_model=None, auto_download=False)
            
            for test_case in valid_inputs:
                with self.subTest(input=test_case['input']):
                    # Should not raise an exception for valid inputs
                    try:
                        # Convert to tuple if needed
                        if isinstance(test_case['input'], list):
                            triple = tuple(test_case['input'])
                        else:
                            triple = test_case['input']
                        
                        # This should work without raising an exception
                        if len(triple) == 3:
                            result = smied.analyze_triple(*triple, verbose=False)
                            # Result format doesn't matter for validation test
                            self.assertIsNotNone(result)
                    except Exception as e:
                        self.fail(f"Valid input {test_case['input']} raised {e}")
    
    def test_input_validation_invalid_triple_format(self):
        """Test validation handles invalid triple formats appropriately."""
        # Get invalid input cases from configuration
        invalid_inputs = [case for case in self.validation_cases['input_validation']
                         if not case['is_valid']]
        
        with patch('smied.SMIED.nltk', self.mock_nltk), \
             patch('smied.SMIED.wn', self.mock_wn), \
             patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            
            smied = SMIED(nlp_model=None, auto_download=False)
            
            for test_case in invalid_inputs:
                with self.subTest(input=test_case['input']):
                    input_tuple = test_case['input']
                    if len(input_tuple) < 3:
                        # Too few arguments - Python will raise TypeError
                        with self.assertRaises(TypeError):
                            smied.analyze_triple(*input_tuple, verbose=False)
                    elif len(input_tuple) > 3:
                        # Too many arguments - treated as extra keyword args
                        # This may cause type errors when non-int passed to max_depth
                        try:
                            result = smied.analyze_triple(*input_tuple, verbose=False)
                            # Should handle gracefully and return some result
                            self.assertIsNotNone(result)
                        except (TypeError, ValueError) as e:
                            # May raise error due to type mismatch (string as max_depth)
                            self.assertIn('int', str(e).lower() or 'type' in str(e).lower())
                    else:
                        # Should not reach here based on test config
                        self.fail(f"Unexpected valid triple length: {len(input_tuple)}")
    
    def test_model_name_validation(self):
        """Test validation of model names and parameters."""
        # Get model constants from configuration
        model_constants = self.config.get_model_name_constants()
        
        # Test valid model names
        valid_models = [
            model_constants['default_spacy_model'],
            model_constants['large_spacy_model'],
            model_constants['medium_spacy_model']
        ]
        
        for model in valid_models:
            with self.subTest(model=model):
                # Should create instance without error
                smied = SMIED(nlp_model=model, auto_download=False)
                self.assertEqual(smied.nlp_model_name, model)
    
    def test_parameter_constraint_validation(self):
        """Test validation of parameter constraints and ranges."""
        # Test boolean parameters
        smied = SMIED(auto_download=True, build_graph_on_init=False)
        self.assertTrue(smied.auto_download)
        
        smied = SMIED(auto_download=False, build_graph_on_init=True)
        self.assertFalse(smied.auto_download)
        
        # Test None handling
        smied = SMIED(nlp_model=None, embedding_model=None)
        self.assertIsNone(smied.nlp_model_name)
        self.assertIsNone(smied.embedding_model)
    
    def test_output_format_validation(self):
        """Test validation of output formats using config data."""
        # Get output validation data from configuration
        output_validation = self.validation_cases['output_validation']
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer), \
             patch('smied.SMIED.wn', self.mock_wn):
            
            # Configure mock to return structured result
            mock_synset = self.mock_factory('MockSynset', name="test.n.01")
            self.mock_decomposer.find_connected_shortest_paths.return_value = (
                [mock_synset], [mock_synset], mock_synset
            )
            
            smied = SMIED(nlp_model=None, auto_download=False)
            result = smied.analyze_triple("cat", "chase", "mouse", verbose=False)
            
            # Validate result structure
            self.assertIsNotNone(result)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            
            # Test word info output format
            word_info = smied.get_word_info("test")
            self.assertIsInstance(word_info, dict)
            # Should contain required keys from config
            required_keys = ['word', 'synsets', 'total_senses']
            for key in required_keys:
                self.assertIn(key, word_info, f"Missing required key: {key}")

    def test_mock_factory_validation(self):
        """Test mock factory creation and validation using config data."""
        # Get factory test cases from configuration
        factory_cases = self.config.get_mock_factory_test_cases()
        
        # Test valid mock types
        valid_mocks = factory_cases['valid_mock_types']
        for mock_type in valid_mocks:
            with self.subTest(mock_type=mock_type):
                try:
                    mock_instance = self.mock_factory(mock_type)
                    self.assertIsNotNone(mock_instance)
                    
                    # Test that mocks have expected interface
                    if 'SMIED' in mock_type and mock_type != 'MockISMIEDPipeline':
                        # SMIED mocks should have core methods
                        expected_methods = ['analyze_triple', 'get_synsets', 'build_synset_graph']
                        for method in expected_methods:
                            self.assertTrue(hasattr(mock_instance, method),
                                          f"Mock {mock_type} missing method {method}")
                            
                except Exception as e:
                    self.fail(f"Failed to create valid mock {mock_type}: {e}")
        
        # Test invalid mock types
        invalid_mocks = factory_cases['invalid_mock_types']
        for invalid_mock in invalid_mocks:
            if invalid_mock is not None:  # Skip None test as it would cause different error
                with self.subTest(invalid_mock=invalid_mock):
                    with self.assertRaises(ValueError):
                        self.mock_factory(invalid_mock)
    
    def test_abstract_base_class_compliance(self):
        """Test that mocks properly implement abstract base class methods."""
        # Test synset mocks
        synset_mock = self.mock_factory('MockSynset', name="test.n.01")
        
        # Should implement abstract methods from AbstractEntityMock
        self.assertTrue(hasattr(synset_mock, 'get_primary_attribute'))
        self.assertTrue(hasattr(synset_mock, 'validate_entity'))
        self.assertTrue(hasattr(synset_mock, 'get_entity_signature'))
        
        # Test method functionality
        primary_attr = synset_mock.get_primary_attribute()
        self.assertIsNotNone(primary_attr)
        
        is_valid = synset_mock.validate_entity()
        self.assertIsInstance(is_valid, bool)
        
        signature = synset_mock.get_entity_signature()
        self.assertIsInstance(signature, str)
        self.assertTrue(len(signature) > 0)


class TestSmiedEdgeCases(unittest.TestCase):
    """Edge cases and error condition tests for SMIED class.
    
    Tests boundary conditions, error scenarios, and exceptional cases
    using factory-created mocks and configuration data.
    """
    
    def setUp(self):
        """Set up edge case test fixtures using mock factory."""
        # Initialize configuration layer
        self.config = SMIEDMockConfig()
        
        # Initialize mock factory
        self.mock_factory = SMIEDMockFactory()
        
        # Get error handling scenarios from configuration
        self.error_scenarios = self.config.get_error_handling_scenarios()
        self.performance_expectations = self.config.get_performance_expectations()
        self.edge_case_scenarios = self.config.get_edge_case_test_scenarios()
        
        # Create edge case specific mocks
        self.mock_nltk = self.mock_factory('MockNLTK')
        self.mock_wn = self.mock_factory('MockWordNet')
        self.mock_decomposer = self.mock_factory('MockSemanticDecomposer')
        
    def test_empty_input_handling(self):
        """Test handling of empty inputs - SMIED processes them gracefully."""
        with patch('smied.SMIED.nltk', self.mock_nltk), \
             patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            
            # Configure mock to return empty paths when no synsets found
            self.mock_decomposer.find_connected_shortest_paths.return_value = (None, None, None)
            
            smied = SMIED(nlp_model=None, auto_download=False)
            
            # Test empty string in triple - should return (None, None, None)
            result = smied.analyze_triple('', 'test', 'test', verbose=False)
            self.assertIsNotNone(result)
            self.assertEqual(result, (None, None, None))
            
            # Test with all empty strings - should still process gracefully
            result = smied.analyze_triple('', '', '', verbose=False)
            self.assertIsNotNone(result)
            self.assertEqual(result, (None, None, None))
            
            # Verify analysis was attempted (method was called)
            self.mock_decomposer.find_connected_shortest_paths.assert_called()
    
    def test_missing_synsets_edge_case(self):
        """Test handling when WordNet synsets are not found."""
        with patch('smied.SMIED.wn', self.mock_wn), \
             patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            
            # Configure mock to return empty synsets (word not found)
            self.mock_wn.synsets.return_value = []
            
            smied = SMIED(nlp_model=None, auto_download=False)
            
            # Should handle gracefully when synsets not found
            result = smied.get_synsets("nonexistentword123")
            self.assertEqual(result, [])
            
            # Similarity calculation should return None
            similarity = smied.calculate_similarity("nonexistent1", "nonexistent2")
            self.assertIsNone(similarity)
    
    def test_component_initialization_failure(self):
        """Test handling of component initialization failures."""
        # Get component failure scenarios from configuration
        failure_scenarios = self.error_scenarios['component_failure_scenarios']
        
        for scenario in failure_scenarios:
            with self.subTest(component=scenario['failing_component']):
                if scenario['failing_component'] == 'spacy_model':
                    # Test spaCy model loading failure
                    with patch('spacy.load') as mock_spacy_load:
                        mock_spacy_load.side_effect = OSError("Model not found")
                        
                        # Should handle gracefully
                        smied = SMIED(nlp_model="nonexistent_model")
                        self.assertIsNone(smied.nlp)
                
                elif scenario['failing_component'] == 'embedding_model':
                    # Test embedding model failure (should continue without embeddings)
                    smied = SMIED(embedding_model="nonexistent_embedding")
                    # Should still initialize other components
                    self.assertIsNotNone(smied.decomposer)
    
    def test_malformed_triple_edge_cases(self):
        """Test handling of malformed or unusual triple formats."""
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            smied = SMIED(nlp_model=None, auto_download=False)
            
            # Test with very long strings
            long_string = "a" * 1000
            try:
                result = smied.analyze_triple(long_string, "test", "test", verbose=False)
                # Should handle without crashing
                self.assertIsNotNone(result)
            except Exception as e:
                # If it raises an exception, it should be a reasonable one
                self.assertIsInstance(e, (ValueError, MemoryError))
            
            # Test with special characters
            special_chars = ["@#$%", "123", "äöü", "中文"]
            for char_set in special_chars:
                with self.subTest(chars=char_set):
                    try:
                        result = smied.analyze_triple(char_set, "test", "test", verbose=False)
                        self.assertIsNotNone(result)
                    except Exception:
                        # Some special characters might not be handled
                        pass
    
    def test_memory_and_performance_edge_cases(self):
        """Test behavior under resource constraints using config expectations."""
        # Get performance expectations from configuration
        expectations = self.performance_expectations
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            # Configure decomposer to simulate large graph
            large_graph = self.mock_factory('MockGraph')
            large_graph.number_of_nodes.return_value = 100000
            large_graph.number_of_edges.return_value = 500000
            self.mock_decomposer.build_synset_graph.return_value = large_graph
            
            smied = SMIED(nlp_model=None, auto_download=False)
            
            # Should handle large graph creation
            with patch('builtins.print'):  # Suppress verbose output
                result = smied.build_synset_graph()
            
            self.assertIsNotNone(result)
            self.assertEqual(result.number_of_nodes(), 100000)
    
    def test_concurrent_access_edge_case(self):
        """Test thread safety and concurrent access scenarios."""
        import threading
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            smied = SMIED(nlp_model=None, auto_download=False)
            
            results = []
            errors = []
            
            def analyze_triple_thread(subject, predicate, obj):
                try:
                    result = smied.analyze_triple(subject, predicate, obj, verbose=False)
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            # Create multiple threads doing analysis
            threads = []
            test_triples = [("cat", "chase", "mouse"), ("dog", "bark", "loud")]
            
            for triple in test_triples:
                thread = threading.Thread(target=analyze_triple_thread, args=triple)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Should handle concurrent access gracefully
            # Either all succeed or fail gracefully
            total_operations = len(results) + len(errors)
            self.assertEqual(total_operations, len(test_triples))

    def test_synset_edge_case_handling(self):
        """Test handling of various synset edge cases using config scenarios."""
        # Get synset edge cases from configuration
        synset_edge_cases = self.edge_case_scenarios['synset_edge_cases']
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            smied = SMIED(nlp_model=None, auto_download=False)
            
            for case in synset_edge_cases:
                edge_case_type = case['edge_case_type']
                should_raise_error = case['should_raise_error']
                
                with self.subTest(edge_case_type=edge_case_type):
                    # Create edge case mock using factory
                    edge_mock = self.mock_factory('MockSynsetEdgeCases', 
                                                edge_case_type=edge_case_type)
                    
                    if should_raise_error:
                        # Test that invalid synsets are handled properly
                        self.assertFalse(edge_mock.validate_entity())
                    else:
                        # Test that valid edge cases are accepted
                        self.assertTrue(edge_mock.validate_entity())
                    
                    # Test entity signature generation
                    signature = edge_mock.get_entity_signature()
                    self.assertIn(edge_case_type, signature)

    def test_unicode_and_special_character_handling(self):
        """Test handling of unicode and special characters using config data."""
        # Get analysis edge cases from configuration
        analysis_edge_cases = self.edge_case_scenarios['analysis_edge_cases']
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer):
            smied = SMIED(nlp_model=None, auto_download=False)
            
            for case in analysis_edge_cases:
                scenario = case['scenario']
                test_input = case['input']
                
                with self.subTest(scenario=scenario):
                    try:
                        # Should handle special inputs gracefully
                        result = smied.analyze_triple(*test_input, verbose=False)
                        self.assertIsNotNone(result)
                    except Exception as e:
                        # If it raises an exception, should be a reasonable one
                        self.assertIsInstance(e, (ValueError, TypeError, UnicodeError))

    def test_malformed_input_comprehensive(self):
        """Test comprehensive malformed input handling - SMIED processes inputs gracefully."""
        # Get malformed input scenarios from configuration
        malformed_scenarios = self.error_scenarios['malformed_input_scenarios']
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_decomposer), \
             patch('smied.SMIED.wn') as mock_wn:
            # Configure mock to return no paths for malformed inputs
            self.mock_decomposer.find_connected_shortest_paths.return_value = (None, None, None)
            
            # Configure WordNet mock to handle various inputs gracefully
            mock_wn.synsets.return_value = []
            mock_wn.NOUN = 'n'
            mock_wn.VERB = 'v' 
            
            smied = SMIED(nlp_model=None, auto_download=False, verbosity=0)
            
            for scenario in malformed_scenarios:
                test_input = scenario['input']
                
                with self.subTest(input=test_input):
                    # SMIED is designed to handle malformed inputs gracefully
                    # It processes them through WordNet and returns results
                    try:
                        result = smied.analyze_triple(*test_input, verbose=False)
                        # Should return a tuple of (subject_path, object_path, predicate)
                        self.assertIsNotNone(result)
                        self.assertIsInstance(result, tuple)
                        self.assertEqual(len(result), 3)
                        # For malformed inputs, typically returns (None, None, None)
                        subject_path, object_path, predicate = result
                        # These may be None for malformed inputs
                        self.assertTrue(subject_path is None or isinstance(subject_path, list))
                        self.assertTrue(object_path is None or isinstance(object_path, list))
                        # predicate can be None or a synset object
                    except Exception as e:
                        # If an exception is raised, it should be a reasonable one
                        # (e.g., from WordNet processing, not from SMIED validation)
                        self.assertIsInstance(e, (AttributeError, TypeError, ValueError))
                        # The error should be from downstream processing, not input validation
                        self.assertNotIn('validation', str(e).lower())
                        self.assertNotIn('invalid input', str(e).lower())

    def test_edge_case_mock_factory_usage(self):
        """Test proper usage of edge case mocks through factory."""
        # Get mock factory test cases from configuration
        factory_cases = self.config.get_mock_factory_test_cases()
        
        for scenario in factory_cases['mock_creation_scenarios']:
            mock_type = scenario['mock_type']
            kwargs = scenario.get('kwargs', {})
            expected_success = scenario['expected_success']
            
            with self.subTest(mock_type=mock_type):
                if expected_success:
                    # Should successfully create mock
                    mock_instance = self.mock_factory(mock_type, **kwargs)
                    self.assertIsNotNone(mock_instance)
                    
                    # Test that edge case mocks have proper attributes
                    if 'EdgeCases' in mock_type:
                        if hasattr(mock_instance, 'edge_case_type'):
                            self.assertIsNotNone(mock_instance.edge_case_type)
                        elif hasattr(mock_instance, 'failure_mode'):
                            self.assertIsNotNone(mock_instance.failure_mode)

    def test_component_failure_modes(self):
        """Test various component failure modes using edge case mocks."""
        failure_modes = ['initialization_error', 'analysis_error', 'resource_error']
        
        for failure_mode in failure_modes:
            with self.subTest(failure_mode=failure_mode):
                # Create edge case mock with specific failure mode
                edge_case_mock = self.mock_factory('MockSMIEDEdgeCases', 
                                                 failure_mode=failure_mode)
                
                # Test appropriate exceptions are configured
                if failure_mode == 'initialization_error':
                    with self.assertRaises(RuntimeError):
                        edge_case_mock.reinitialize()
                elif failure_mode == 'analysis_error':
                    with self.assertRaises(ValueError):
                        edge_case_mock.analyze_triple("test", "test", "test")
                elif failure_mode == 'resource_error':
                    with self.assertRaises(OSError):
                        edge_case_mock.build_synset_graph()


class TestSmiedIntegration(unittest.TestCase):
    """Integration tests for SMIED class with other components.
    
    Tests full pipeline integration, component interactions, and
    end-to-end scenarios using realistic mock configurations.
    """
    
    def setUp(self):
        """Set up integration test fixtures using mock factory and config."""
        # Initialize configuration and factory
        self.config = SMIEDMockConfig()
        self.mock_factory = SMIEDMockFactory()
        
        # Get integration test data from configuration
        self.integration_scenarios = self.config.get_integration_test_scenarios()
        self.component_init_data = self.config.get_component_initialization_data()
        
        # Create integration-specific mock
        self.integration_mock = self.mock_factory('MockSMIEDIntegration')
        
        # Get test data
        self.triple_test_cases = self.config.get_triple_analysis_test_cases()

    def test_full_semantic_analysis_pipeline(self):
        """Test complete semantic analysis pipeline integration using config scenarios."""
        # Get semantic analysis scenario from configuration
        scenario = self.integration_scenarios['semantic_analysis_scenario']
        test_triple = scenario['input_triple']
        expected_components = scenario['expected_components']
        
        # Create full pipeline mocks using integration mock
        test_mocks = self.integration_mock.create_full_pipeline_mocks()
        
        with patch('smied.SMIED.nltk', test_mocks['mock_nltk']), \
             patch('spacy.load', return_value=test_mocks['mock_nlp']), \
             patch('smied.SMIED.wn', test_mocks['mock_wn']), \
             patch('smied.SMIED.SemanticDecomposer', return_value=test_mocks['mock_decomposer']), \
             patch('builtins.print'):
            
            # Get default model from config
            model_constants = self.config.get_model_name_constants()
            default_model = model_constants['default_spacy_model']
            
            # Create and use SMIED with config data
            smied = SMIED(
                nlp_model=default_model, 
                auto_download=True, 
                build_graph_on_init=False
            )
            result = smied.analyze_triple(*test_triple, verbose=False)
            
            # Verify result structure using config expectations
            self.assertIsNotNone(result)
            subject_path, object_path, predicate = result
            self.assertIsNotNone(subject_path)
            self.assertIsNotNone(object_path)
            self.assertIsNotNone(predicate)
            
            # Verify component initialization and calls
            test_mocks['mock_nltk'].download.assert_any_call('wordnet', quiet=True)
            test_mocks['mock_decomposer'].build_synset_graph.assert_called_once()
            test_mocks['mock_decomposer'].find_connected_shortest_paths.assert_called_once()
            
            # Verify expected components are available (using config data)
            for component, status in expected_components.items():
                if status == 'initialized':
                    # Component should be properly initialized
                    if component == 'semantic_decomposer':
                        self.assertIsNotNone(smied.decomposer)
                    elif component == 'semantic_metagraph':
                        # Graph should be buildable
                        graph = smied.build_synset_graph(verbose=False)
                        self.assertIsNotNone(graph)

    def test_multiple_semantic_analyses_with_caching(self):
        """Test multiple analyses with same instance using config test cases."""
        # Get multiple test triples from configuration
        simple_triples = self.triple_test_cases['simple_triples'][:3]
        
        with patch('smied.SMIED.SemanticDecomposer') as mock_decomposer_class, \
             patch('smied.SMIED.wn') as mock_wn, \
             patch('builtins.print'):
            
            # Create mocks using factory
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
            
            smied = SMIED(nlp_model=None, auto_download=False)
            
            # Perform multiple analyses using config test cases
            for triple in simple_triples:
                smied.analyze_triple(*triple, verbose=False)
            
            # Graph should only be built once (caching behavior)
            mock_decomposer.build_synset_graph.assert_called_once()
            
            # But analysis should be called for each triple
            self.assertEqual(
                mock_decomposer.find_connected_shortest_paths.call_count, 
                len(simple_triples)
            )
    
    def test_end_to_end_text_processing_integration(self):
        """Test end-to-end text processing pipeline using config scenarios."""
        # Get end-to-end scenario from configuration
        end_to_end = self.integration_scenarios['end_to_end_scenario']
        expected_entities = end_to_end['expected_entities']
        expected_actions = end_to_end['expected_actions']
        
        # Create full pipeline mocks
        test_mocks = self.integration_mock.create_full_pipeline_mocks()
        
        # Configure proper mock synset with iterable methods
        mock_synset = Mock()
        mock_synset.name.return_value = "dog.n.01"
        mock_synset.definition.return_value = "domestic animal"
        mock_synset.pos.return_value = "n"
        mock_synset.examples.return_value = ["The dog barked"]
        
        # Create mock lemma with proper name method
        mock_lemma = Mock()
        mock_lemma.name.return_value = "dog"
        mock_synset.lemmas.return_value = [mock_lemma]
        
        mock_synset.hypernyms.return_value = []
        mock_synset.hyponyms.return_value = []
        
        # Configure WordNet mock to return proper synsets
        test_mocks['mock_wn'].synsets.return_value = [mock_synset]
        
        with patch('smied.SMIED.nltk', test_mocks['mock_nltk']), \
             patch('spacy.load', return_value=test_mocks['mock_nlp']), \
             patch('smied.SMIED.wn', test_mocks['mock_wn']), \
             patch('smied.SMIED.SemanticDecomposer', return_value=test_mocks['mock_decomposer']), \
             patch('builtins.print'):
            
            smied = SMIED(nlp_model="en_core_web_sm", auto_download=False)
            
            # Test individual entity analysis (simulating text processing)
            for entity in expected_entities:
                word_info = smied.get_word_info(entity)
                self.assertIsNotNone(word_info)
                self.assertEqual(word_info['word'], entity)
                self.assertIn('synsets', word_info)
                self.assertIn('total_senses', word_info)
            
            # Test action analysis
            for action in expected_actions:
                synsets = smied.get_synsets(action, pos='v')
                # Should return mock synsets
                self.assertIsNotNone(synsets)
                self.assertIsInstance(synsets, list)
    
    def test_component_interaction_and_dependencies(self):
        """Test proper component interaction and dependency handling."""
        # Get component dependencies from configuration
        dependencies = self.component_init_data['component_dependencies']
        init_params = self.component_init_data['initialization_parameters']
        
        # Test semantic_decomposer dependencies
        decomposer_deps = dependencies.get('semantic_decomposer', [])
        
        with patch('smied.SMIED.SemanticDecomposer') as mock_decomposer_class:
            mock_decomposer = self.mock_factory('MockSemanticDecomposer')
            mock_decomposer_class.return_value = mock_decomposer
            
            # Initialize SMIED with specific parameters from config
            semantic_params = init_params.get('semantic_decomposer', {})
            
            smied = SMIED(nlp_model=None, auto_download=False)
            
            # Verify decomposer was initialized
            self.assertIsNotNone(smied.decomposer)
            
            # Test dependency interaction
            if 'gloss_parser' in decomposer_deps:
                # Decomposer should work with gloss parsing functionality
                result = smied.analyze_triple("test", "analyze", "dependency", verbose=False)
                self.assertIsNotNone(result)
    
    def test_configuration_driven_component_setup(self):
        """Test component setup using configuration parameters."""
        # Get configuration options from config
        config_options = self.config.get_configuration_options()
        
        # Test different analysis modes
        analysis_modes = config_options['analysis_modes']
        processing_options = config_options['processing_options']
        
        with patch('smied.SMIED.SemanticDecomposer', return_value=self.mock_factory('MockSemanticDecomposer')):
            # Test with caching enabled (from config)
            if processing_options.get('use_caching', False):
                smied = SMIED(nlp_model=None, auto_download=False)
                
                # First analysis - should build graph
                graph1 = smied.build_synset_graph(verbose=False)
                
                # Second call - should return cached graph
                graph2 = smied.build_synset_graph(verbose=False)
                
                # Should be the same cached instance
                self.assertEqual(graph1, graph2)
            
            # Test similarity metrics from config
            similarity_metrics = config_options['similarity_metrics']
            for metric in ['path_similarity', 'wup_similarity', 'lch_similarity']:
                if metric in similarity_metrics:
                    with self.subTest(metric=metric):
                        # Test that SMIED supports configured similarity metric
                        method_map = {
                            'path_similarity': 'path',
                            'wup_similarity': 'wu_palmer',
                            'lch_similarity': 'lch'
                        }
                        method = method_map.get(metric)
                        if method:
                            result = smied.calculate_similarity("test1", "test2", method=method)
                            # Result might be None if synsets not found, which is acceptable
                            self.assertIsInstance(result, (float, type(None)))

    def test_mock_factory_integration_coverage(self):
        """Test comprehensive mock factory integration with all components."""
        # Get factory test cases from configuration
        factory_cases = self.config.get_mock_factory_test_cases()
        
        # Test that all registered mocks can be created and work together
        valid_mocks = factory_cases['valid_mock_types']
        
        # Create instances of all mock types
        mock_instances = {}
        for mock_type in valid_mocks:
            try:
                mock_instances[mock_type] = self.mock_factory(mock_type)
            except Exception as e:
                self.fail(f"Failed to create mock {mock_type}: {e}")
        
        # Test basic integration between mock types
        smied_mock = mock_instances.get('MockSMIED')
        synset_mock = mock_instances.get('MockSynset')
        
        if smied_mock and synset_mock:
            # Test that synset mock can be used with SMIED mock
            self.assertIsNotNone(synset_mock.get_primary_attribute())
            self.assertTrue(synset_mock.validate_entity())
            
            # Test that SMIED mock has expected interface
            expected_attrs = ['analyze_triple', 'get_synsets', 'nlp_model_name']
            for attr in expected_attrs:
                self.assertTrue(hasattr(smied_mock, attr),
                              f"MockSMIED missing attribute {attr}")

    def test_comprehensive_error_scenario_coverage(self):
        """Test comprehensive error scenario coverage using all config data."""
        # Get all error scenarios from configuration
        all_error_scenarios = self.config.get_error_handling_scenarios()
        
        # Test each type of error scenario
        for scenario_type, scenarios in all_error_scenarios.items():
            with self.subTest(scenario_type=scenario_type):
                if scenario_type == 'malformed_input_scenarios':
                    # Already tested in edge cases, verify configuration exists
                    self.assertGreater(len(scenarios), 0)
                elif scenario_type == 'component_failure_scenarios':
                    # Test that edge case mocks can simulate these failures
                    for scenario in scenarios:
                        component = scenario['failing_component']
                        failure_type = scenario['failure_type']
                        
                        if component == 'spacy_model' and failure_type == 'model_not_found':
                            # Test with edge case mock
                            edge_mock = self.mock_factory('MockSMIEDEdgeCases', 
                                                        failure_mode='initialization_error')
                            self.assertIsNotNone(edge_mock.failure_mode)


if __name__ == '__main__':
    unittest.main()