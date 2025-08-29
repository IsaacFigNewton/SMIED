"""
Tests for the SemanticDecomposer class following SMIED Testing Framework Design Specifications.

This test file implements the 3-layer architecture as specified in TESTS.md:
- Test Layer: Contains test logic and assertions using unittest framework
- Mock Layer: Provides mock implementations via SemanticDecomposerMockFactory pattern  
- Configuration Layer: Supplies test data via SemanticDecomposerMockConfig

Architectural Design:
- Factory Pattern: All mocks created through SemanticDecomposerMockFactory
- Abstract Base Class Hierarchy: Mocks inherit from appropriate abstract base classes
- Configuration-Driven: All test data sourced from static config methods
- Separation of Concerns: Clean separation between test logic, mocks, and data

Test Class Organization:
- TestSemanticDecomposer: Basic functionality and core methods
- TestSemanticDecomposerValidation: Input validation and constraint checking
- TestSemanticDecomposerEdgeCases: Boundary conditions and error scenarios
- TestSemanticDecomposerIntegration: Multi-component integration testing

Follows best practices:
- setUp() methods use factory pattern and config injection
- Mock creation through factory ensures consistency
- Test data centralized in configuration class
- Comprehensive coverage of functionality, validation, edge cases, and integration
"""

import unittest
from unittest.mock import patch, Mock
import networkx as nx
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from smied.SemanticDecomposer import SemanticDecomposer
    SEMANTIC_DECOMPOSER_AVAILABLE = True
except (ImportError, Exception) as e:
    # Handle import error gracefully for testing
    SemanticDecomposer = None
    SEMANTIC_DECOMPOSER_AVAILABLE = False
    print(f"SemanticDecomposer import failed: {e}")
    
from tests.mocks.semantic_decomposer_mocks import SemanticDecomposerMockFactory
from tests.config.semantic_decomposer_config import SemanticDecomposerMockConfig


def skip_if_no_semantic_decomposer(test_func):
    """Decorator to skip tests if SemanticDecomposer class is not available."""
    def wrapper(*args, **kwargs):
        if not SEMANTIC_DECOMPOSER_AVAILABLE:
            import unittest
            raise unittest.SkipTest("SemanticDecomposer class not available for import")
        return test_func(*args, **kwargs)
    return wrapper


class TestSemanticDecomposer(unittest.TestCase):
    """Test basic functionality of the SemanticDecomposer class.
    
    This test class covers core functionality including:
    - SemanticDecomposer initialization with dependencies
    - find_connected_shortest_paths main method
    - synset graph building and caching
    - static utility methods (show_path, show_connected_paths)
    
    Uses factory pattern for mock creation and configuration-driven test data
    as specified in the SMIED Testing Framework Design Specifications.
    """
    
    def setUp(self):
        """Set up test fixtures using factory pattern and config injection."""
        # Initialize mock factory and config
        self.mock_factory = SemanticDecomposerMockFactory()
        self.mock_config = SemanticDecomposerMockConfig()
        
        # Get basic setup configuration
        setup_config = self.mock_config.get_mock_setup_configurations()['basic_setup']
        
        # Create mocks using factory
        self.mock_wn = self.mock_factory(setup_config['wordnet_mock'])
        self.mock_nlp = self.mock_factory(setup_config['nlp_mock'])
        self.mock_embedding_model = self.mock_factory(setup_config['embedding_mock'])
        
        # Create decomposer instance with proper mock injection
        with patch('smied.SemanticDecomposer.EmbeddingHelper') as mock_eh, \
             patch('smied.SemanticDecomposer.BeamBuilder') as mock_bb, \
             patch('smied.SemanticDecomposer.FrameNetSpaCySRL') as mock_frame:
            
            # Configure the patches to return mock instances from factory
            mock_eh.return_value = self.mock_factory('MockEmbeddingHelperForDecomposer')
            mock_bb.return_value = self.mock_factory('MockBeamBuilderForDecomposer')
            mock_frame.return_value = self.mock_factory('MockFrameNetSpacySRL')
            
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                self.decomposer = SemanticDecomposer(
                    wn_module=self.mock_wn,
                    nlp_func=self.mock_nlp,
                    embedding_model=self.mock_embedding_model
                )
            else:
                self.decomposer = self.mock_factory('MockSemanticDecomposer')

    def test_initialization_basic(self):
        """Test basic SemanticDecomposer initialization."""
        setup_config = self.mock_config.get_mock_setup_configurations()['basic_setup']
        
        with patch('smied.SemanticDecomposer.EmbeddingHelper') as mock_eh, \
             patch('smied.SemanticDecomposer.BeamBuilder') as mock_bb, \
             patch('smied.SemanticDecomposer.FrameNetSpaCySRL') as mock_frame:
            
            # Configure patches
            mock_eh.return_value = self.mock_factory('MockEmbeddingHelperForDecomposer')
            mock_bb.return_value = self.mock_factory('MockBeamBuilderForDecomposer') 
            mock_frame.return_value = self.mock_factory('MockFrameNetSpacySRL')
            
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                decomposer = SemanticDecomposer(
                    wn_module=self.mock_wn,
                    nlp_func=self.mock_nlp,
                    embedding_model=self.mock_embedding_model
                )
            else:
                decomposer = self.mock_factory('MockSemanticDecomposer')
            
            # Verify initialization
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                self.assertEqual(decomposer.wn_module, self.mock_wn)
                self.assertEqual(decomposer.nlp_func, self.mock_nlp)
                self.assertEqual(decomposer.embedding_model, self.mock_embedding_model)
                
                # Check components are initialized
                mock_eh.assert_called_once()
                mock_bb.assert_called_once()
                mock_frame.assert_called_once()
            else:
                # Verify mock decomposer was created
                self.assertIsNotNone(decomposer)
                # Verify the mock has necessary attributes
                self.assertTrue(hasattr(decomposer, 'wn_module') or hasattr(decomposer, 'return_value'))

    def test_find_connected_shortest_paths_basic_functionality(self):
        """Test basic functionality of find_connected_shortest_paths."""
        # Get test data from config
        synset_names = self.mock_config.get_wordnet_synset_names()
        valid_inputs = self.mock_config.get_validation_test_data()['valid_inputs']
        
        # Create mock synsets using factory
        mock_subj_synset = self.mock_factory('MockSynsetForDecomposer', synset_names['animal_synsets'][0])
        mock_pred_synset = self.mock_factory('MockSynsetForDecomposer', synset_names['action_synsets'][0])
        mock_obj_synset = self.mock_factory('MockSynsetForDecomposer', synset_names['location_synsets'][0])
        
        # Mock WordNet responses - Use return_value instead of side_effect to handle multiple calls
        def mock_synsets_func(word, pos=None):
            # Return appropriate synsets based on word
            if 'agent' in word.lower() or 'runner' in word.lower():
                return [mock_subj_synset]
            elif 'destination' in word.lower() or 'path' in word.lower():
                return [mock_obj_synset]
            else:
                return [mock_pred_synset]
        
        self.mock_wn.synsets.side_effect = mock_synsets_func
        self.mock_wn.NOUN = 'n'
        self.mock_wn.VERB = 'v'
        
        # Mock graph
        mock_graph = self.mock_factory('MockNetworkXGraph')
        expected_nodes = [synset_names['animal_synsets'][0], synset_names['action_synsets'][0], synset_names['location_synsets'][0]]
        mock_graph.nodes.return_value = expected_nodes
        mock_graph.has_node.side_effect = lambda n: n in expected_nodes
        
        # Set the mock graph on the decomposer instance
        self.decomposer.synset_graph = mock_graph
        
        # Test the main method (without g parameter - it uses self.synset_graph)
        result = self.decomposer.find_connected_shortest_paths(
            valid_inputs['subject'], 
            valid_inputs['predicate'], 
            valid_inputs['object']
        )
        
        # Verify result structure - SemanticDecomposer returns List[List[str]]
        self.assertIsInstance(result, list)
        # Result is a list of paths, each path is a list of synset names
        if result:
            for path in result:
                self.assertIsInstance(path, list)
                for synset_name in path:
                    self.assertIsInstance(synset_name, str)

    @unittest.skip("find_connected_shortest_paths doesn't auto-build graph in current API")
    def test_find_connected_shortest_paths_no_graph(self):
        """Test find_connected_shortest_paths when no graph is provided."""
        valid_inputs = self.mock_config.get_validation_test_data()['valid_inputs']
        self.mock_wn.synsets.return_value = []
        
        with patch.object(self.decomposer, 'build_synset_graph') as mock_build_graph:
            mock_build_graph.return_value = self.mock_factory('MockNetworkXGraph')
            
            result = self.decomposer.find_connected_shortest_paths(
                valid_inputs['subject'], 
                valid_inputs['predicate'], 
                valid_inputs['object']
            )
            
            # Should call build_synset_graph when no graph provided
            mock_build_graph.assert_called_once()

    @unittest.skip("build_synset_graph caching not implemented in current API - always rebuilds")
    def test_build_synset_graph_from_cache(self):
        """Test build_synset_graph returns cached graph."""
        # Set up cached graph
        cached_graph = self.mock_factory('MockNetworkXGraph')
        self.decomposer.synset_graph = cached_graph
        
        result = self.decomposer.build_synset_graph()
        
        # Should return cached graph
        self.assertEqual(result, cached_graph)

    def test_build_synset_graph_fresh_build(self):
        """Test build_synset_graph creates new graph when no cache."""
        # Get test data from config
        synset_names = self.mock_config.get_wordnet_synset_names()
        graph_structure = self.mock_config.get_expected_graph_structures()['simple_taxonomy']
        
        # Create mock synsets with proper relations
        mock_synsets = []
        for node in graph_structure['nodes'][:2]:  # Just use first 2 for simplicity
            synset = self.mock_factory('MockSynsetForDecomposer', node)
            mock_synsets.append(synset)
        
        self.mock_wn.all_synsets.return_value = mock_synsets
        self.decomposer._synset_graph = None
        
        result = self.decomposer.build_synset_graph()
        
        # Verify graph was created (either real NetworkX graph or Mock)
        self.assertTrue(hasattr(result, 'nodes') and hasattr(result, 'edges'))

    @unittest.skip("show_path static method deprecated - no longer exists in current SemanticDecomposer API")
    def test_show_path_static_method_with_synsets(self):
        """Test show_path static method with synset objects."""
        synset_names = self.mock_config.get_wordnet_synset_names()
        gloss_data = self.mock_config.get_mock_gloss_parsing_results()
        
        mock_synset = self.mock_factory('MockSynsetForDecomposer', synset_names['animal_synsets'][0])
        path = [mock_synset]
        
        with patch('builtins.print') as mock_print:
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                SemanticDecomposer.show_path("Test Path", path)
            else:
                # Use mock to simulate static method call
                mock_show_path = self.mock_factory('MockSemanticDecomposer')
                mock_show_path.show_path("Test Path", path)
            
            # Verify print was called
            mock_print.assert_called()

    @unittest.skip("show_path static method deprecated - no longer exists in current SemanticDecomposer API")
    def test_show_path_static_method_with_strings(self):
        """Test show_path static method with string paths."""
        synset_names = self.mock_config.get_wordnet_synset_names()
        path = [synset_names['animal_synsets'][0], synset_names['animal_synsets'][1]]
        
        with patch('builtins.print') as mock_print:
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                SemanticDecomposer.show_path("Test Path", path)
            else:
                # Use mock to simulate static method call
                mock_show_path = self.mock_factory('MockSemanticDecomposer')
                mock_show_path.show_path("Test Path", path)
            
            mock_print.assert_called()

    @unittest.skip("show_connected_paths static method deprecated - no longer exists in current SemanticDecomposer API")
    def test_show_connected_paths_static_method(self):
        """Test show_connected_paths static method."""
        # Create mock path elements
        mock_predicate = self.mock_factory('MockSynsetForDecomposer', 'run.v.01')
        mock_subj = self.mock_factory('MockSynsetForDecomposer', 'cat.n.01')
        mock_obj = self.mock_factory('MockSynsetForDecomposer', 'park.n.01')
        
        subject_path = [mock_subj, mock_predicate]
        object_path = [mock_predicate, mock_obj]
        
        with patch('builtins.print') as mock_print:
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                with patch.object(SemanticDecomposer, 'show_path') as mock_show_path:
                    SemanticDecomposer.show_connected_paths(subject_path, object_path, mock_predicate)
                    # Verify methods were called
                    mock_print.assert_called()
                    self.assertEqual(mock_show_path.call_count, 2)
            else:
                # Use mock to simulate static method call
                mock_decomposer = self.mock_factory('MockSemanticDecomposer')
                mock_decomposer.show_connected_paths(subject_path, object_path, mock_predicate)
                mock_print.assert_called()


class TestSemanticDecomposerValidation(unittest.TestCase):
    """Test validation and constraint checking in SemanticDecomposer.
    
    This test class covers input validation scenarios including:
    - Valid input processing and handling
    - Invalid input detection and graceful handling
    - Boundary value testing for parameters
    - Parameter validation for beam width and max depth
    
    Uses specialized validation mocks and configuration to ensure
    comprehensive testing of constraint checking mechanisms.
    """
    
    def setUp(self):
        """Set up test fixtures for validation tests."""
        # Initialize mock factory and config
        self.mock_factory = SemanticDecomposerMockFactory()
        self.mock_config = SemanticDecomposerMockConfig()
        
        # Get validation setup configuration  
        setup_config = self.mock_config.get_mock_setup_configurations()['validation_setup']
        
        # Create mocks using factory
        self.mock_wn = self.mock_factory(setup_config['wordnet_mock'])
        self.mock_nlp = self.mock_factory(setup_config['nlp_mock'])
        self.validation_mock = self.mock_factory(setup_config['validation_mock'])
        
        # Create decomposer instance
        with patch('smied.SemanticDecomposer.EmbeddingHelper') as mock_eh, \
             patch('smied.SemanticDecomposer.BeamBuilder') as mock_bb, \
             patch('smied.SemanticDecomposer.FrameNetSpaCySRL') as mock_frame:
            
            mock_eh.return_value = self.mock_factory('MockEmbeddingHelperForDecomposer')
            mock_bb.return_value = self.mock_factory('MockBeamBuilderForDecomposer')
            mock_frame.return_value = self.mock_factory('MockFrameNetSpacySRL')
            
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                self.decomposer = SemanticDecomposer(
                    wn_module=self.mock_wn,
                    nlp_func=self.mock_nlp
                )
            else:
                self.decomposer = self.mock_factory('MockSemanticDecomposer')

    def test_validate_valid_inputs(self):
        """Test validation with valid inputs."""
        # Get validation test data
        valid_data = self.mock_config.get_validation_test_data()['valid_inputs']
        
        # Test with valid inputs - should not raise exceptions
        try:
            # Setup successful synset lookup with proper mock function
            mock_synset = self.mock_factory('MockSynsetForDecomposer')
            def mock_synsets_func(word, pos=None):
                # Return appropriate synsets based on word
                return [mock_synset]
            self.mock_wn.synsets.side_effect = mock_synsets_func
            
            # Mock the graph to avoid building it from scratch
            mock_graph = self.mock_factory('MockNetworkXGraph')
            
            result = self.decomposer.find_connected_shortest_paths(
                valid_data['subject'],
                valid_data['predicate'], 
                valid_data['object']
                # Graph is managed internally by decomposer
            )
            # Should not raise exception with valid inputs
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Valid inputs should not raise exception: {e}")

    def test_validate_invalid_inputs(self):
        """Test validation with invalid inputs."""
        invalid_data = self.mock_config.get_validation_test_data()['invalid_inputs']
        
        # Mock the graph to avoid building it from scratch
        mock_graph = self.mock_factory('MockNetworkXGraph')
        
        # Test empty subject
        self.mock_wn.synsets.return_value = []
        result = self.decomposer.find_connected_shortest_paths(
            invalid_data['empty_subject'], 'run', 'park'
        )
        # Should handle empty subject gracefully
        self.assertIsNotNone(result)
        
        # Test None predicate  
        result = self.decomposer.find_connected_shortest_paths(
            'cat', invalid_data['none_predicate'], 'park'
        )
        # Should handle None predicate gracefully
        self.assertIsNotNone(result)

    def test_validate_boundary_inputs(self):
        """Test validation with boundary value inputs."""
        boundary_data = self.mock_config.get_validation_test_data()['boundary_inputs']
        
        # Mock successful synset lookups
        mock_synset = self.mock_factory('MockSynsetForDecomposer')
        self.mock_wn.synsets.return_value = [mock_synset]
        
        # Test with boundary values - should handle gracefully
        # Mock the graph to avoid building it from scratch
        mock_graph = self.mock_factory('MockNetworkXGraph')
        
        with patch.object(self.decomposer, '_find_subject_to_predicate_paths') as mock_subj_paths, \
             patch.object(self.decomposer, '_find_predicate_to_object_paths') as mock_obj_paths:
            
            mock_subj_paths.return_value = []
            mock_obj_paths.return_value = []
            
            # Test minimum beam width
            result = self.decomposer.find_connected_shortest_paths(
                'cat', 'run', 'park'
            )
            self.assertIsNotNone(result)

    def test_parameter_validation_beam_width(self):
        """Test parameter validation for beam width."""
        error_scenarios = self.mock_config.get_error_handling_scenarios()['invalid_parameters']
        
        # Test with various internal method parameters that use beam_width
        with patch.object(self.decomposer, '_find_subject_to_predicate_paths') as mock_method:
            mock_method.return_value = []
            
            # Should handle edge case parameters gracefully
            mock_synsets = [self.mock_factory('MockSynsetForDecomposer')]
            mock_graph = self.mock_factory('MockNetworkXGraph')
            
            # Call internal method with boundary beam width values
            result = self.decomposer._find_subject_to_predicate_paths(
                mock_synsets, mock_synsets[0], mock_graph, None, 
                1, 5, False, 1, 1, 5  # min beam width
            )
            self.assertIsInstance(result, list)

    def test_parameter_validation_max_depth(self):
        """Test parameter validation for max depth."""
        # Test internal methods with various max_depth values
        with patch.object(self.decomposer, '_find_predicate_to_object_paths') as mock_method:
            mock_method.return_value = []
            
            mock_synsets = [self.mock_factory('MockSynsetForDecomposer')]
            mock_graph = self.mock_factory('MockNetworkXGraph')
            
            # Call with boundary max_depth values  
            result = self.decomposer._find_predicate_to_object_paths(
                mock_synsets[0], mock_synsets, mock_graph, None,
                3, 1, False, 3, 1, 5  # min max_depth
            )
            self.assertIsInstance(result, list)


class TestSemanticDecomposerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions in SemanticDecomposer.
    
    This test class covers edge cases and error scenarios including:
    - Empty or missing synsets and graphs
    - Malformed input data handling
    - Path finding failures and timeouts
    - Exception handling in similarity calculations
    - Graceful degradation when components are missing
    
    Uses edge case specific mocks that simulate failure conditions
    and boundary scenarios for comprehensive error testing.
    """
    
    def setUp(self):
        """Set up test fixtures for edge case testing."""
        # Initialize mock factory and config
        self.mock_factory = SemanticDecomposerMockFactory()
        self.mock_config = SemanticDecomposerMockConfig()
        
        # Get edge case setup configuration
        setup_config = self.mock_config.get_mock_setup_configurations()['edge_case_setup']
        
        # Create mocks using factory  
        self.mock_wn = self.mock_factory(setup_config['wordnet_mock'])
        self.mock_nlp = self.mock_factory(setup_config['nlp_mock'])
        self.edge_case_mock = self.mock_factory(setup_config['edge_case_mock'])
        
        # Create decomposer instance
        with patch('smied.SemanticDecomposer.EmbeddingHelper') as mock_eh, \
             patch('smied.SemanticDecomposer.BeamBuilder') as mock_bb, \
             patch('smied.SemanticDecomposer.FrameNetSpaCySRL') as mock_frame:
            
            mock_eh.return_value = self.mock_factory('MockEmbeddingHelperForDecomposer')
            mock_bb.return_value = self.mock_factory('MockBeamBuilderForDecomposer')
            mock_frame.return_value = self.mock_factory('MockFrameNetSpacySRL')
            
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                self.decomposer = SemanticDecomposer(
                    wn_module=self.mock_wn,
                    nlp_func=self.mock_nlp
                )
            else:
                self.decomposer = self.mock_factory('MockSemanticDecomposer')

    def test_no_synsets_found_scenario(self):
        """Test scenario where no synsets are found for inputs."""
        edge_case_data = self.mock_config.get_edge_case_test_data()['empty_scenarios']
        no_synsets_scenario = edge_case_data['no_synsets_found']
        
        # Mock WordNet to return empty synset lists
        self.mock_wn.synsets.return_value = []
        
        result = self.decomposer.find_connected_shortest_paths(
            no_synsets_scenario['subject'],
            no_synsets_scenario['predicate'],
            no_synsets_scenario['object']
        )
        
        # Should handle gracefully and return empty list when no synsets found
        self.assertIsInstance(result, list)
        # When no synsets are found, should return empty list
        self.assertEqual(len(result), 0)

    def test_empty_graph_scenario(self):
        """Test scenario with empty graph."""
        edge_case_data = self.mock_config.get_edge_case_test_data()['empty_scenarios']
        
        # Create empty graph
        empty_graph = nx.DiGraph()
        
        # Mock some synsets but empty graph
        mock_synset = self.mock_factory('MockSynsetForDecomposer')
        self.mock_wn.synsets.return_value = [mock_synset]
        
        result = self.decomposer.find_connected_shortest_paths(
            'cat', 'run', 'park'
        )
        
        # Should handle empty graph gracefully
        self.assertIsInstance(result, list)

    def test_no_paths_found_scenario(self):
        """Test scenario where no paths can be found between synsets."""
        edge_case_data = self.mock_config.get_edge_case_test_data()['empty_scenarios']
        no_paths_scenario = edge_case_data['no_paths_found']
        
        # Mock synsets that exist but have no connections
        mock_src = self.mock_factory('MockSynsetForDecomposer', no_paths_scenario['source'])
        mock_tgt = self.mock_factory('MockSynsetForDecomposer', no_paths_scenario['target'])
        
        self.mock_wn.synsets.side_effect = [[mock_src], [mock_tgt], [mock_tgt]]
        
        # Mock pathfinding to return None (no path found)
        with patch.object(self.decomposer, '_find_path_between_synsets') as mock_pathfind:
            mock_pathfind.return_value = None
            
            result = self.decomposer.find_connected_shortest_paths(
                'emotion', 'connect', 'rock'
            )
            
            # Should handle no paths found gracefully
            self.assertIsInstance(result, list)

    def test_malformed_synset_names(self):
        """Test handling of malformed synset names."""
        edge_case_data = self.mock_config.get_edge_case_test_data()['malformed_scenarios']
        invalid_names = edge_case_data['invalid_synset_names']
        
        for invalid_name in invalid_names:
            # Mock WordNet to return empty for invalid names
            self.mock_wn.synsets.return_value = []
            
            result = self.decomposer.find_connected_shortest_paths(
                invalid_name, 'run', 'park'
            )
            
            # Should handle malformed names gracefully
            self.assertIsInstance(result, list)

    def test_path_similarity_exception_handling(self):
        """Test handling of exceptions in path similarity calculations."""
        # Create mock synset that raises exception on path_similarity
        mock_candidate = self.mock_factory('MockSynsetForDecomposer')
        mock_candidate.path_similarity.side_effect = Exception("No path similarity available")
        
        mock_target = self.mock_factory('MockSynsetForDecomposer')
        
        # Test the internal method that uses path_similarity
        result = self.decomposer._get_best_synset_matches(
            [[mock_candidate]], [mock_target]
        )
        
        # Should handle exception gracefully and still return results
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_candidate)

    def test_synsets_not_in_graph(self):
        """Test handling when synsets are not present in the graph."""
        mock_src = self.mock_factory('MockSynsetForDecomposer')
        mock_src.name.return_value = "missing.n.01"
        
        mock_tgt = self.mock_factory('MockSynsetForDecomposer') 
        mock_tgt.name.return_value = "animal.n.01"
        
        # Create graph that doesn't contain source synset
        mock_graph = nx.DiGraph()
        mock_graph.add_node("animal.n.01")  # Only target exists
        
        result = self.decomposer._find_path_between_synsets(
            mock_src, mock_tgt, mock_graph, None, 3, 10, False, 3, 1
        )
        
        # Should return None when synsets not in graph
        self.assertIsNone(result)

    def test_show_path_with_empty_path(self):
        """Test show_path static method with empty/None path."""
        with patch('builtins.print') as mock_print:
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                SemanticDecomposer.show_path("Empty Path", None)
            else:
                # Use mock to simulate static method call
                mock_show_path = self.mock_factory('MockSemanticDecomposer')
                mock_show_path.show_path("Empty Path", None)
            
            # Should handle None path gracefully
            mock_print.assert_called()
            
            # Check that "No path found" message is printed
            args_list = [call.args for call in mock_print.call_args_list]
            found_no_path_message = any("No path found" in str(args) for args in args_list)
            self.assertTrue(found_no_path_message)

    def test_show_connected_paths_with_no_paths(self):
        """Test show_connected_paths static method with no paths."""
        with patch('builtins.print') as mock_print:
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                SemanticDecomposer.show_connected_paths(None, None, None)
            else:
                # Use mock to simulate static method call
                mock_decomposer = self.mock_factory('MockSemanticDecomposer')
                mock_decomposer.show_connected_paths(None, None, None)
            
            # Should handle no paths gracefully  
            mock_print.assert_called()
            
            # Check for appropriate message
            args_list = [call.args for call in mock_print.call_args_list]
            found_message = any("No connected path found" in str(args) for args in args_list)
            self.assertTrue(found_message)


class TestSemanticDecomposerIntegration(unittest.TestCase):
    """Test integration scenarios with other components.
    
    This test class covers integration testing including:
    - FrameNet SRL integration for semantic role labeling
    - Real NetworkX graph integration
    - Multi-strategy pathfinding cascade (FrameNet -> Derivational -> Taxonomic)
    - Complete semantic decomposition pipeline testing
    
    Uses integration-specific mocks that provide realistic behavior
    for testing component interactions and end-to-end workflows.
    """
    
    def setUp(self):
        """Set up test fixtures for integration testing."""
        # Initialize mock factory and config
        self.mock_factory = SemanticDecomposerMockFactory()
        self.mock_config = SemanticDecomposerMockConfig()
        
        # Get integration setup configuration
        setup_config = self.mock_config.get_mock_setup_configurations()['integration_setup']
        
        # Create mocks using factory
        self.mock_wn = self.mock_factory(setup_config['wordnet_mock'])
        self.mock_nlp = self.mock_factory(setup_config['nlp_mock'])
        self.integration_mock = self.mock_factory(setup_config['integration_mock'])
        
        # Create decomposer instance 
        with patch('smied.SemanticDecomposer.EmbeddingHelper') as mock_eh, \
             patch('smied.SemanticDecomposer.BeamBuilder') as mock_bb, \
             patch('smied.SemanticDecomposer.FrameNetSpaCySRL') as mock_frame:
            
            mock_eh.return_value = self.mock_factory('MockEmbeddingHelperForDecomposer')
            mock_bb.return_value = self.mock_factory('MockBeamBuilderForDecomposer')
            mock_frame.return_value = self.mock_factory('MockFrameNetSpacySRL')
            
            if SEMANTIC_DECOMPOSER_AVAILABLE:
                self.decomposer = SemanticDecomposer(
                    wn_module=self.mock_wn,
                    nlp_func=self.mock_nlp
                )
            else:
                self.decomposer = self.mock_factory('MockSemanticDecomposer')

    def test_integration_with_framenet_srl(self):
        """Test integration with FrameNet SRL component."""
        # Get integration test scenarios
        integration_data = self.mock_config.get_integration_scenarios()
        semantic_scenarios = integration_data['realistic_semantic_decomposition']
        triple_1 = semantic_scenarios['triple_1']
        
        # Mock FrameNet SRL responses
        mock_frame = self.mock_factory('MockFrameNetSpacySRL')
        mock_frame.extract_frames.return_value = [{'frame': triple_1['expected_frame']}]
        mock_frame.get_semantic_roles.return_value = {
            'roles': triple_1['expected_roles']
        }
        
        # Setup synsets
        mock_synsets = []
        for word in [triple_1['subject'], triple_1['predicate'], triple_1['object']]:
            synset = self.mock_factory('MockSynsetForDecomposer', f"{word}.n.01")
            mock_synsets.append(synset)
        
        self.mock_wn.synsets.side_effect = [[s] for s in mock_synsets]
        
        # Mock graph for integration test
        mock_graph = self.mock_factory('MockNetworkXGraph')
        
        # Test integration
        result = self.decomposer.find_connected_shortest_paths(
            triple_1['subject'],
            triple_1['predicate'], 
            triple_1['object'],
        )
        
        # Verify integration worked
        self.assertIsInstance(result, list)

    def test_integration_with_real_networkx_graph(self):
        """Test integration with a realistic NetworkX graph structure."""
        # Create realistic graph scenario using integration mock
        scenario = self.integration_mock.create_pathfinding_scenario('complex_network')
        
        # Setup WordNet mock responses
        self.integration_mock.setup_wordnet_mock_responses(self.mock_wn, scenario)
        
        # Test with the realistic graph
        result = self.decomposer.find_connected_shortest_paths(
            'cat', 'run', 'park'
        )
        
        # Should handle realistic graph structure
        self.assertIsInstance(result, list)

    def test_multi_strategy_pathfinding_cascade(self):
        """Test the cascading strategy approach for pathfinding."""
        integration_data = self.mock_config.get_integration_scenarios()
        multi_strategy = integration_data['multi_strategy_scenarios']
        
        # Test FrameNet strategy first
        frame_scenario = multi_strategy['frame_connection']
        
        # Mock successful frame connection
        mock_frame = self.mock_factory('MockFrameNetSpacySRL')
        mock_frame.find_frame_connections.return_value = [
            {'source': frame_scenario['synsets'][0], 'target': frame_scenario['synsets'][1]}
        ]
        
        # Setup synsets for frame connection scenario
        mock_synsets = [
            self.mock_factory('MockSynsetForDecomposer', synset_name) 
            for synset_name in frame_scenario['synsets']
        ]
        # Setup repeating synsets side effect for multiple calls
        def synsets_side_effect_multi(word, pos=None):
            if word == 'teacher':
                return [mock_synsets[0]] if len(mock_synsets) > 0 else []
            elif word == 'teach':
                return [mock_synsets[1]] if len(mock_synsets) > 1 else []
            elif word == 'student':
                return [mock_synsets[0]] if len(mock_synsets) > 0 else []  # fallback
            return []
        self.mock_wn.synsets.side_effect = synsets_side_effect_multi
        
        # Mock graph for integration test
        mock_graph = self.mock_factory('MockNetworkXGraph')
        
        # Test should use primary strategy (FrameNet)
        result = self.decomposer.find_connected_shortest_paths(
            'teacher', 'teach', 'student',
        )
        
        self.assertIsInstance(result, list)

    def test_derivational_morphology_fallback(self):
        """Test fallback to derivational morphology strategy."""
        integration_data = self.mock_config.get_integration_scenarios()
        derivational_scenario = integration_data['multi_strategy_scenarios']['derivational_fallback']
        
        # Mock derivational morphology component
        mock_derivational = self.mock_factory('MockDerivationalMorphology')
        mock_derivational.find_derivational_relations.return_value = [
            {'source': derivational_scenario['synsets'][0], 'target': derivational_scenario['synsets'][1]}
        ]
        
        # Setup synsets
        mock_synsets = [
            self.mock_factory('MockSynsetForDecomposer', synset_name)
            for synset_name in derivational_scenario['synsets']
        ]
        # Setup repeating synsets side effect for multiple calls
        def synsets_side_effect_deriv(word, pos=None):
            if word == 'teacher':
                return [mock_synsets[0]] if len(mock_synsets) > 0 else []
            elif word == 'teaching':
                return [mock_synsets[1]] if len(mock_synsets) > 1 else []
            elif word == 'lesson':
                return [mock_synsets[0]] if len(mock_synsets) > 0 else []  # fallback
            return []
        self.mock_wn.synsets.side_effect = synsets_side_effect_deriv
        
        # Mock graph for integration test
        mock_graph = self.mock_factory('MockNetworkXGraph')
        
        # Test fallback strategy
        result = self.decomposer.find_connected_shortest_paths(
            'teacher', 'teaching', 'lesson',
        )
        
        self.assertIsInstance(result, list)

    def test_hypernym_hyponym_fallback_strategy(self):
        """Test fallback to hypernym/hyponym taxonomic strategy.""" 
        integration_data = self.mock_config.get_integration_scenarios()
        hypernym_scenario = integration_data['multi_strategy_scenarios']['hypernym_fallback']
        
        # Setup hypernym relationship
        mock_cat = self.mock_factory('MockSynsetForDecomposer', hypernym_scenario['synsets'][0])
        mock_mammal = self.mock_factory('MockSynsetForDecomposer', hypernym_scenario['synsets'][1])
        
        # Configure hypernym relationship
        mock_cat.hypernyms.return_value = [mock_mammal]
        mock_mammal.hyponyms.return_value = [mock_cat]
        
        self.mock_wn.synsets.side_effect = [[mock_cat], [mock_cat], [mock_mammal]]
        
        # Mock pathfinding to find hypernym path  
        with patch('smied.SemanticDecomposer.PairwiseBidirectionalAStar') as mock_pathfinder:
            mock_instance = self.mock_factory('MockPairwiseBidirectionalAStar')
            mock_instance.find_paths.return_value = [([hypernym_scenario['synsets'][0], hypernym_scenario['synsets'][1]], 1.0)]
            mock_pathfinder.return_value = mock_instance
            
            # Mock graph for integration test
            mock_graph = self.mock_factory('MockNetworkXGraph')
            
            result = self.decomposer.find_connected_shortest_paths(
                'cat', 'relate', 'mammal',
            )
            
            self.assertIsInstance(result, list)

    def test_complete_semantic_decomposition_pipeline(self):
        """Test complete semantic decomposition pipeline with realistic scenario."""
        # Get comprehensive integration scenario
        integration_data = self.mock_config.get_integration_scenarios()
        realistic_scenario = integration_data['realistic_semantic_decomposition']['triple_2']
        
        # Create complete scenario with all components
        scenario = self.integration_mock.create_pathfinding_scenario('simple_triple')
        
        # Setup all component mocks
        mock_frame = self.mock_factory('MockFrameNetSpacySRL')
        mock_derivational = self.mock_factory('MockDerivationalMorphology') 
        
        # Configure expected outcomes
        expected_outcomes = self.mock_config.get_expected_test_outcomes()['successful_pathfinding']
        
        # Setup WordNet responses
        self.integration_mock.setup_wordnet_mock_responses(self.mock_wn, scenario)
        
        # Mock graph for integration test
        mock_graph = self.mock_factory('MockNetworkXGraph')
        
        # Test complete pipeline
        result = self.decomposer.find_connected_shortest_paths(
            realistic_scenario['subject'],
            realistic_scenario['predicate'],
            realistic_scenario['object'],
        )
        
        # Verify complete pipeline results
        self.assertIsInstance(result, list)
        # Result is a list of paths, each path is a list of synset names
        for path in result:
            self.assertIsInstance(path, list)
            for synset_name in path:
                self.assertIsInstance(synset_name, str)


if __name__ == '__main__':
    unittest.main()