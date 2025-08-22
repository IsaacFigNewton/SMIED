import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.SemanticDecomposer import SemanticDecomposer


class TestSemanticDecomposer(unittest.TestCase):
    """Test the SemanticDecomposer class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_wn = Mock()
        self.mock_nlp = Mock()
        self.mock_embedding_model = Mock()
        
        # Create decomposer instance
        with patch('smied.SemanticDecomposer.EmbeddingHelper'), \
             patch('smied.SemanticDecomposer.BeamBuilder'), \
             patch('smied.SemanticDecomposer.GlossParser'):
            self.decomposer = SemanticDecomposer(
                wn_module=self.mock_wn,
                nlp_func=self.mock_nlp,
                embedding_model=self.mock_embedding_model
            )

    def test_initialization(self):
        """Test SemanticDecomposer initialization"""
        with patch('smied.SemanticDecomposer.EmbeddingHelper') as mock_eh, \
             patch('smied.SemanticDecomposer.BeamBuilder') as mock_bb, \
             patch('smied.SemanticDecomposer.GlossParser') as mock_gp:
            
            decomposer = SemanticDecomposer(
                wn_module=self.mock_wn,
                nlp_func=self.mock_nlp,
                embedding_model=self.mock_embedding_model
            )
            
            self.assertEqual(decomposer.wn_module, self.mock_wn)
            self.assertEqual(decomposer.nlp_func, self.mock_nlp)
            self.assertEqual(decomposer.embedding_model, self.mock_embedding_model)
            self.assertIsNone(decomposer._synset_graph)
            
            # Check components are initialized
            mock_eh.assert_called_once()
            mock_bb.assert_called_once()
            mock_gp.assert_called_once_with(nlp_func=self.mock_nlp)

    def test_find_connected_shortest_paths_basic(self):
        """Test basic functionality of find_connected_shortest_paths"""
        # Mock synsets
        mock_subj_synset = Mock()
        mock_subj_synset.name.return_value = "cat.n.01"
        
        mock_pred_synset = Mock()
        mock_pred_synset.name.return_value = "run.v.01"
        
        mock_obj_synset = Mock()
        mock_obj_synset.name.return_value = "park.n.01"
        
        # Mock WordNet synsets calls
        self.mock_wn.synsets.side_effect = [
            [mock_subj_synset],  # subject synsets
            [mock_pred_synset],  # predicate synsets
            [mock_obj_synset]    # object synsets
        ]
        self.mock_wn.NOUN = 'n'
        self.mock_wn.VERB = 'v'
        
        # Mock graph
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("run.v.01")
        mock_graph.add_node("park.n.01")
        
        # Mock path finding methods
        with patch.object(self.decomposer, '_find_subject_to_predicate_paths') as mock_subj_paths, \
             patch.object(self.decomposer, '_find_predicate_to_object_paths') as mock_obj_paths:
            
            mock_subj_paths.return_value = [[mock_subj_synset, mock_pred_synset]]
            mock_obj_paths.return_value = [[mock_pred_synset, mock_obj_synset]]
            
            result = self.decomposer.find_connected_shortest_paths(
                "cat", "run", "park", g=mock_graph
            )
            
            subj_path, obj_path, predicate = result
            self.assertIsNotNone(subj_path)
            self.assertIsNotNone(obj_path)
            self.assertIsNotNone(predicate)

    def test_find_connected_shortest_paths_no_graph(self):
        """Test find_connected_shortest_paths when no graph is provided"""
        self.mock_wn.synsets.return_value = []
        
        with patch.object(self.decomposer, 'build_synset_graph') as mock_build_graph:
            mock_build_graph.return_value = nx.DiGraph()
            
            result = self.decomposer.find_connected_shortest_paths("cat", "run", "park")
            
            mock_build_graph.assert_called_once()

    def test_find_connected_shortest_paths_no_embedding_model(self):
        """Test find_connected_shortest_paths without embedding model"""
        self.decomposer.embedding_model = None
        self.mock_wn.synsets.return_value = []
        
        with patch.object(self.decomposer, 'build_synset_graph') as mock_build_graph:
            mock_build_graph.return_value = nx.DiGraph()
            
            result = self.decomposer.find_connected_shortest_paths("cat", "run", "park")
            
            # Should still work without embedding model
            self.assertIsNotNone(result)

    def test_find_subject_to_predicate_paths(self):
        """Test _find_subject_to_predicate_paths method"""
        mock_subj_synset = Mock()
        mock_pred_synset = Mock()
        mock_graph = nx.DiGraph()
        
        # Mock gloss parser
        mock_parsed_gloss = {
            'subjects': [Mock()],
            'predicates': [Mock()]
        }
        self.decomposer.gloss_parser.parse_gloss.return_value = mock_parsed_gloss
        
        with patch.object(self.decomposer, '_get_best_synset_matches') as mock_matches, \
             patch.object(self.decomposer, '_find_path_between_synsets') as mock_path, \
             patch.object(self.decomposer, '_explore_hypernym_paths') as mock_hypernyms:
            
            mock_matches.return_value = [Mock()]
            mock_path.return_value = [mock_subj_synset, mock_pred_synset]
            mock_hypernyms.return_value = []
            
            result = self.decomposer._find_subject_to_predicate_paths(
                [mock_subj_synset], mock_pred_synset, mock_graph,
                None, 3, 10, False, 3, 1, 5
            )
            
            self.assertIsInstance(result, list)

    def test_find_predicate_to_object_paths(self):
        """Test _find_predicate_to_object_paths method"""
        mock_pred_synset = Mock()
        mock_obj_synset = Mock()
        mock_graph = nx.DiGraph()
        
        # Mock gloss parser
        mock_parsed_gloss = {
            'objects': [Mock()],
            'predicates': [Mock()]
        }
        self.decomposer.gloss_parser.parse_gloss.return_value = mock_parsed_gloss
        
        with patch.object(self.decomposer, '_get_best_synset_matches') as mock_matches, \
             patch.object(self.decomposer, '_find_path_between_synsets') as mock_path, \
             patch.object(self.decomposer, '_explore_hypernym_paths') as mock_hypernyms:
            
            mock_matches.return_value = [Mock()]
            mock_path.return_value = [mock_pred_synset, mock_obj_synset]
            mock_hypernyms.return_value = []
            
            result = self.decomposer._find_predicate_to_object_paths(
                mock_pred_synset, [mock_obj_synset], mock_graph,
                None, 3, 10, False, 3, 1, 5
            )
            
            self.assertIsInstance(result, list)

    def test_get_best_synset_matches(self):
        """Test _get_best_synset_matches method"""
        mock_candidate1 = Mock()
        mock_candidate1.path_similarity.return_value = 0.8
        
        mock_candidate2 = Mock()
        mock_candidate2.path_similarity.return_value = 0.6
        
        mock_target = Mock()
        
        result = self.decomposer._get_best_synset_matches(
            [[mock_candidate1, mock_candidate2]], [mock_target], top_k=2
        )
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 2)
        # Should be sorted by similarity (highest first)
        if len(result) >= 2:
            self.assertEqual(result[0], mock_candidate1)
            self.assertEqual(result[1], mock_candidate2)

    def test_get_best_synset_matches_no_path_similarity(self):
        """Test _get_best_synset_matches when path_similarity fails"""
        mock_candidate = Mock()
        mock_candidate.path_similarity.side_effect = Exception("No path similarity")
        
        mock_target = Mock()
        
        result = self.decomposer._get_best_synset_matches(
            [[mock_candidate]], [mock_target]
        )
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_candidate)

    def test_find_path_between_synsets(self):
        """Test _find_path_between_synsets method"""
        mock_src = Mock()
        mock_src.name.return_value = "cat.n.01"
        
        mock_tgt = Mock()
        mock_tgt.name.return_value = "animal.n.01"
        
        mock_graph = nx.DiGraph()
        mock_graph.add_node("cat.n.01")
        mock_graph.add_node("animal.n.01")
        
        with patch('smied.SemanticDecomposer.PairwiseBidirectionalAStar') as mock_pathfinder:
            mock_instance = Mock()
            mock_instance.find_paths.return_value = [(["cat.n.01", "animal.n.01"], 1.0)]
            mock_pathfinder.return_value = mock_instance
            
            result = self.decomposer._find_path_between_synsets(
                mock_src, mock_tgt, mock_graph, None, 3, 10, False, 3, 1
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)

    def test_find_path_between_synsets_not_in_graph(self):
        """Test _find_path_between_synsets when synsets not in graph"""
        mock_src = Mock()
        mock_src.name.return_value = "missing.n.01"
        
        mock_tgt = Mock()
        mock_tgt.name.return_value = "animal.n.01"
        
        mock_graph = nx.DiGraph()
        mock_graph.add_node("animal.n.01")
        
        result = self.decomposer._find_path_between_synsets(
            mock_src, mock_tgt, mock_graph, None, 3, 10, False, 3, 1
        )
        
        self.assertIsNone(result)

    def test_explore_hypernym_paths(self):
        """Test _explore_hypernym_paths method"""
        mock_src = Mock()
        mock_src.hypernyms.return_value = [Mock()]
        
        mock_tgt = Mock()
        mock_tgt.hypernyms.return_value = [Mock()]
        
        mock_graph = nx.DiGraph()
        
        with patch.object(self.decomposer, '_find_path_between_synsets') as mock_path:
            mock_path.return_value = [Mock(), Mock()]
            
            result = self.decomposer._explore_hypernym_paths(
                [mock_src], [mock_tgt], mock_graph, None, 3, 10, False, 3, 1
            )
            
            self.assertIsInstance(result, list)

    def test_build_synset_graph_cached(self):
        """Test build_synset_graph with caching"""
        # Set up cached graph
        cached_graph = nx.DiGraph()
        self.decomposer._synset_graph = cached_graph
        
        result = self.decomposer.build_synset_graph()
        
        self.assertEqual(result, cached_graph)

    def test_build_synset_graph_fresh(self):
        """Test build_synset_graph creating new graph"""
        mock_synset1 = Mock()
        mock_synset1.name.return_value = "cat.n.01"
        mock_synset1.hypernyms.return_value = []
        mock_synset1.hyponyms.return_value = []
        mock_synset1.holonyms.return_value = []
        mock_synset1.meronyms.return_value = []
        mock_synset1.similar_tos.return_value = []
        mock_synset1.also_sees.return_value = []
        mock_synset1.verb_groups.return_value = []
        mock_synset1.entailments.return_value = []
        mock_synset1.causes.return_value = []
        
        mock_synset2 = Mock()
        mock_synset2.name.return_value = "animal.n.01"
        mock_synset2.hypernyms.return_value = []
        mock_synset2.hyponyms.return_value = [mock_synset1]  # cat is hyponym of animal
        mock_synset2.holonyms.return_value = []
        mock_synset2.meronyms.return_value = []
        mock_synset2.similar_tos.return_value = []
        mock_synset2.also_sees.return_value = []
        mock_synset2.verb_groups.return_value = []
        mock_synset2.entailments.return_value = []
        mock_synset2.causes.return_value = []
        
        self.mock_wn.all_synsets.return_value = [mock_synset1, mock_synset2]
        
        # Clear cached graph
        self.decomposer._synset_graph = None
        
        result = self.decomposer.build_synset_graph()
        
        self.assertIsInstance(result, nx.DiGraph)
        self.assertIn("cat.n.01", result.nodes())
        self.assertIn("animal.n.01", result.nodes())
        # Check that edge was added from animal to cat (hyponym relationship)
        self.assertIn(("animal.n.01", "cat.n.01"), result.edges())

    def test_show_path_with_synsets(self):
        """Test show_path static method with synset objects"""
        mock_synset = Mock()
        mock_synset.name.return_value = "cat.n.01"
        mock_synset.definition.return_value = "a small domesticated carnivorous mammal"
        
        path = [mock_synset]
        
        with patch('builtins.print') as mock_print:
            SemanticDecomposer.show_path("Test Path", path)
            
            mock_print.assert_called()
            # Check that it printed the label
            args_list = [call.args for call in mock_print.call_args_list]
            self.assertTrue(any("Test Path" in str(args) for args in args_list))

    def test_show_path_with_strings(self):
        """Test show_path static method with string paths"""
        path = ["cat.n.01", "animal.n.01"]
        
        with patch('builtins.print') as mock_print:
            SemanticDecomposer.show_path("Test Path", path)
            
            mock_print.assert_called()

    def test_show_path_empty(self):
        """Test show_path static method with empty path"""
        with patch('builtins.print') as mock_print:
            SemanticDecomposer.show_path("Empty Path", None)
            
            mock_print.assert_called()
            # Should print "No path found"
            args_list = [call.args for call in mock_print.call_args_list]
            self.assertTrue(any("No path found" in str(args) for args in args_list))

    def test_show_connected_paths(self):
        """Test show_connected_paths static method"""
        mock_predicate = Mock()
        mock_predicate.name.return_value = "run.v.01"
        
        subject_path = [Mock(), mock_predicate]
        object_path = [mock_predicate, Mock()]
        
        with patch('builtins.print') as mock_print, \
             patch.object(SemanticDecomposer, 'show_path') as mock_show_path:
            
            SemanticDecomposer.show_connected_paths(subject_path, object_path, mock_predicate)
            
            mock_print.assert_called()
            self.assertEqual(mock_show_path.call_count, 2)

    def test_show_connected_paths_no_path(self):
        """Test show_connected_paths static method with no path"""
        with patch('builtins.print') as mock_print:
            SemanticDecomposer.show_connected_paths(None, None, None)
            
            mock_print.assert_called()
            # Should print "No connected path found"
            args_list = [call.args for call in mock_print.call_args_list]
            self.assertTrue(any("No connected path found" in str(args) for args in args_list))


class TestSemanticDecomposerIntegration(unittest.TestCase):
    """Integration tests for SemanticDecomposer"""
    
    def setUp(self):
        """Set up for integration testing"""
        self.mock_wn = Mock()
        self.mock_nlp = Mock()
        
        with patch('smied.SemanticDecomposer.EmbeddingHelper'), \
             patch('smied.SemanticDecomposer.BeamBuilder'), \
             patch('smied.SemanticDecomposer.GlossParser'):
            self.decomposer = SemanticDecomposer(
                wn_module=self.mock_wn,
                nlp_func=self.mock_nlp
            )

    def test_integration_with_real_graph(self):
        """Test integration with a real NetworkX graph"""
        # Create a small test graph
        graph = nx.DiGraph()
        graph.add_node("cat.n.01")
        graph.add_node("animal.n.01")
        graph.add_node("run.v.01")
        graph.add_edge("cat.n.01", "animal.n.01", relation="hypernym")
        
        # Mock synsets
        self.mock_wn.synsets.side_effect = [
            [],  # No synsets found for subject
            [],  # No synsets found for predicate
            []   # No synsets found for object
        ]
        self.mock_wn.NOUN = 'n'
        self.mock_wn.VERB = 'v'
        
        result = self.decomposer.find_connected_shortest_paths(
            "cat", "run", "park", g=graph
        )
        
        # Should handle empty synset lists gracefully
        self.assertIsNotNone(result)
        subj_path, obj_path, predicate = result
        self.assertIsNone(subj_path)
        self.assertIsNone(obj_path)
        self.assertIsNone(predicate)

    def test_complex_pathfinding_scenario(self):
        """Test a complex pathfinding scenario with multiple strategies"""
        # Mock synsets with proper relationships
        mock_cat = Mock()
        mock_cat.name.return_value = "cat.n.01"
        mock_cat.definition.return_value = "a small carnivorous mammal"
        
        mock_run = Mock()
        mock_run.name.return_value = "run.v.01"
        mock_run.definition.return_value = "move fast by using legs"
        
        mock_park = Mock()
        mock_park.name.return_value = "park.n.01"
        mock_park.definition.return_value = "a large public garden"
        
        # Set up WordNet mock responses
        self.mock_wn.synsets.side_effect = [
            [mock_cat],     # cat synsets
            [mock_run],     # run synsets
            [mock_park]     # park synsets
        ]
        self.mock_wn.NOUN = 'n'
        self.mock_wn.VERB = 'v'
        self.mock_wn.synset.side_effect = lambda name: {
            "cat.n.01": mock_cat,
            "run.v.01": mock_run,
            "park.n.01": mock_park
        }.get(name)
        
        # Create graph with connections
        graph = nx.DiGraph()
        graph.add_node("cat.n.01")
        graph.add_node("run.v.01")
        graph.add_node("park.n.01")
        graph.add_edge("cat.n.01", "run.v.01")
        graph.add_edge("run.v.01", "park.n.01")
        
        # Mock gloss parsing to return some results
        mock_gloss_result = {
            'subjects': [Mock()],
            'objects': [Mock()],
            'predicates': [Mock()]
        }
        self.decomposer.gloss_parser.parse_gloss.return_value = mock_gloss_result
        
        # Mock path finding
        with patch('smied.SemanticDecomposer.PairwiseBidirectionalAStar') as mock_pathfinder:
            mock_instance = Mock()
            mock_instance.find_paths.return_value = [(["cat.n.01", "run.v.01"], 1.0)]
            mock_pathfinder.return_value = mock_instance
            
            with patch.object(self.decomposer, '_get_best_synset_matches') as mock_matches:
                mock_matches.return_value = [mock_cat]  # Return matching synsets
                
                result = self.decomposer.find_connected_shortest_paths(
                    "cat", "run", "park", g=graph
                )
                
                self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()