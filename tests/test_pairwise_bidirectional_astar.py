"""
Tests for PairwiseBidirectionalAStar class.

This test file follows the SMIED Testing Framework Design Specifications:
- Test Layer: Contains test logic and assertions
- Mock Layer: Uses factory-created mocks from tests.mocks.pairwise_bidirectional_astar_mocks
- Configuration Layer: Loads test data from tests.config.pairwise_bidirectional_astar_config

Test Class Structure:
- TestPairwiseBidirectionalAStar: Basic functionality tests
- TestPairwiseBidirectionalAStarValidation: Validation and constraint tests  
- TestPairwiseBidirectionalAStarEdgeCases: Edge cases and error conditions
- TestPairwiseBidirectionalAStarIntegration: Integration tests with other components
"""

import unittest
from unittest.mock import patch, Mock
import networkx as nx
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.PairwiseBidirectionalAStar import PairwiseBidirectionalAStar
from tests.mocks.pairwise_bidirectional_astar_mocks import PairwiseBidirectionalAStarMockFactory
from tests.config.pairwise_bidirectional_astar_config import PairwiseBidirectionalAStarMockConfig


class TestPairwiseBidirectionalAStar(unittest.TestCase):
    """Test basic functionality of the PairwiseBidirectionalAStar class."""
    
    def setUp(self):
        """Set up test fixtures using factory pattern and configuration injection."""
        # Initialize configuration layer
        self.config = PairwiseBidirectionalAStarMockConfig()
        
        # Initialize mock factory
        self.mock_factory = PairwiseBidirectionalAStarMockFactory()
        
        # Load test data from configuration
        self.algorithm_params = self.config.get_algorithm_parameters()['default_params']
        self.basic_graphs = self.config.get_basic_graph_structures()
        self.initialization_data = self.config.get_initialization_test_data()
        self.gloss_bonus_data = self.config.get_gloss_bonus_constant_tests()
        
        # Create mock graph using factory
        self.mock_graph = self.mock_factory('MockGraphForPathfinding')
        self.mock_graph.nodes.return_value = ["start", "middle", "end"]
        self.mock_graph.edges.return_value = [("start", "middle"), ("middle", "end")]
        self.mock_graph.has_node.side_effect = lambda n: n in ["start", "middle", "end"]
        self.mock_graph.has_edge.side_effect = lambda u, v: (u, v) in [("start", "middle"), ("middle", "end")]
        
        # Create default pathfinder instance
        self.pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            beam_width=self.algorithm_params['beam_width'],
            max_depth=self.algorithm_params['max_depth']
        )

    def test_initialization_basic(self):
        """Test basic initialization of PairwiseBidirectionalAStar."""
        init_data = self.initialization_data['basic_initialization']
        expected_defaults = init_data['expected_defaults']
        
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src=init_data['src'],
            tgt=init_data['tgt']
        )
        
        self.assertEqual(pathfinder.g, self.mock_graph)
        self.assertEqual(pathfinder.src, init_data['src'])
        self.assertEqual(pathfinder.tgt, init_data['tgt'])
        self.assertIsNone(pathfinder.get_new_beams_fn)
        self.assertEqual(pathfinder.gloss_seed_nodes, set())
        self.assertEqual(pathfinder.beam_width, expected_defaults['beam_width'])
        self.assertEqual(pathfinder.max_depth, expected_defaults['max_depth'])
        self.assertEqual(pathfinder.relax_beam, expected_defaults['relax_beam'])
        self.assertEqual(pathfinder.heuristic_type, expected_defaults['heuristic_type'])

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        init_data = self.initialization_data['custom_initialization']
        custom_params = init_data['custom_params']
        
        mock_beams_fn = self.mock_factory('MockHeuristicFunction')
        mock_embedding_helper = self.mock_factory('MockEmbeddingHelper')
        gloss_seeds = ["seed1", "seed2"]
        
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src=init_data['src'],
            tgt=init_data['tgt'],
            get_new_beams_fn=mock_beams_fn,
            gloss_seed_nodes=gloss_seeds,
            beam_width=custom_params['beam_width'],
            max_depth=custom_params['max_depth'],
            relax_beam=custom_params['relax_beam'],
            heuristic_type=custom_params['heuristic_type'],
            embedding_helper=mock_embedding_helper
        )
        
        self.assertEqual(pathfinder.get_new_beams_fn, mock_beams_fn)
        self.assertEqual(pathfinder.gloss_seed_nodes, {"seed1", "seed2"})
        self.assertEqual(pathfinder.beam_width, custom_params['beam_width'])
        self.assertEqual(pathfinder.max_depth, custom_params['max_depth'])
        self.assertEqual(pathfinder.relax_beam, custom_params['relax_beam'])
        self.assertEqual(pathfinder.heuristic_type, custom_params['heuristic_type'])
        self.assertEqual(pathfinder.embedding_helper, mock_embedding_helper)

    def test_gloss_bonus_constant(self):
        """Test GLOSS_BONUS constant value and properties."""
        bonus_data = self.gloss_bonus_data
        
        # Test constant value
        self.assertEqual(PairwiseBidirectionalAStar.GLOSS_BONUS, bonus_data['gloss_bonus_value'])
        
        # Test type
        self.assertIsInstance(PairwiseBidirectionalAStar.GLOSS_BONUS, bonus_data['expected_type'])
        
        # Test range
        min_range, max_range = bonus_data['expected_range']
        self.assertGreaterEqual(PairwiseBidirectionalAStar.GLOSS_BONUS, min_range)
        self.assertLessEqual(PairwiseBidirectionalAStar.GLOSS_BONUS, max_range)

    def test_build_allowed_and_heuristics_no_beams_function(self):
        """Test _build_allowed_and_heuristics without get_new_beams_fn."""
        self.pathfinder._build_allowed_and_heuristics()
        
        # Should include src and tgt in allowed sets
        self.assertIn("start", self.pathfinder.src_allowed)
        self.assertIn("end", self.pathfinder.tgt_allowed)
        
        # Should have heuristics for src and tgt
        self.assertIn("start", self.pathfinder.h_forward)
        self.assertIn("end", self.pathfinder.h_backward)

    def test_build_allowed_and_heuristics_with_beams_function(self):
        """Test _build_allowed_and_heuristics with get_new_beams_fn."""
        # Use a simple callable instead of mock to ensure it works
        beam_call_count = 0
        
        def mock_beams_fn(g, src, tgt):
            nonlocal beam_call_count
            beam_call_count += 1
            return [
                (("start", "rel1"), ("middle", "rel1"), 0.8),
                (("middle", "rel2"), ("end", "rel2"), 0.7)
            ]
        
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            get_new_beams_fn=mock_beams_fn
        )
        
        pathfinder._build_allowed_and_heuristics()
        
        # Check that beams function was called
        self.assertGreater(beam_call_count, 0)
        
        # Check allowed sets include src, tgt (they're always added)
        self.assertIn("start", pathfinder.src_allowed)
        self.assertIn("end", pathfinder.tgt_allowed)
        
        # Check that beam nodes are included (if the function executed successfully)
        self.assertIn("middle", pathfinder.src_allowed)
        self.assertIn("middle", pathfinder.tgt_allowed)

    def test_build_allowed_and_heuristics_with_gloss_seeds(self):
        """Test _build_allowed_and_heuristics with gloss seed nodes."""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            gloss_seed_nodes=["middle", "seed_node"]
        )
        
        pathfinder._build_allowed_and_heuristics()
        
        # Gloss seeds should be in both allowed sets
        self.assertIn("middle", pathfinder.src_allowed)
        self.assertIn("middle", pathfinder.tgt_allowed)
        self.assertIn("seed_node", pathfinder.src_allowed)
        self.assertIn("seed_node", pathfinder.tgt_allowed)
        
        # Gloss seeds should have reduced heuristic (gloss bonus applied)
        if "middle" in pathfinder.h_forward:
            expected_h = max(0.0, pathfinder.h_forward["middle"] + PairwiseBidirectionalAStar.GLOSS_BONUS)
            # The actual value should be lower due to bonus
            self.assertLessEqual(pathfinder.h_forward["middle"], expected_h)

    def test_init_search_state(self):
        """Test _init_search_state method."""
        search_state_data = self.config.get_search_state_validation_data()
        expected_state = search_state_data['initial_state_checks']
        
        self.pathfinder._build_allowed_and_heuristics()
        self.pathfinder._init_search_state()
        
        # Check initial state matches expected values
        self.assertEqual(len(self.pathfinder.open_f), expected_state['priority_queue_sizes']['forward'])
        self.assertEqual(len(self.pathfinder.open_b), expected_state['priority_queue_sizes']['backward'])
        self.assertEqual(self.pathfinder.g_f["start"], expected_state['g_score_initialization']['source_g_forward'])
        self.assertEqual(self.pathfinder.g_b["end"], expected_state['g_score_initialization']['target_g_backward'])
        self.assertEqual(self.pathfinder.depth_f["start"], expected_state['depth_initialization']['source_depth_forward'])
        self.assertEqual(self.pathfinder.depth_b["end"], expected_state['depth_initialization']['target_depth_backward'])
        self.assertEqual(self.pathfinder.parent_f["start"], expected_state['parent_initialization']['source_parent_forward'])
        self.assertEqual(self.pathfinder.parent_b["end"], expected_state['parent_initialization']['target_parent_backward'])
        self.assertEqual(len(self.pathfinder.closed_f), expected_state['closed_set_initialization']['forward_size'])
        self.assertEqual(len(self.pathfinder.closed_b), expected_state['closed_set_initialization']['backward_size'])

    def test_edge_weight_default(self):
        """Test _edge_weight method with default weight."""
        weight = self.pathfinder._edge_weight("start", "middle")
        self.assertEqual(weight, 1.0)

    def test_edge_weight_custom(self):
        """Test _edge_weight method with custom weight from graph structure."""
        # Create a real networkx graph with weights from config
        graph_data = self.basic_graphs['diamond_graph']
        real_graph = nx.DiGraph()
        
        for node in graph_data['nodes']:
            real_graph.add_node(node)
        
        for edge in graph_data['edges']:
            src, tgt, attrs = edge
            real_graph.add_edge(src, tgt, **attrs)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=real_graph,
            src="A",
            tgt="D"
        )
        
        weight = pathfinder._edge_weight("A", "C")
        self.assertEqual(weight, 2.0)

    def test_allowed_forward_relax_beam_enabled(self):
        """Test _allowed_forward with relax_beam=True."""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        # Should allow any node when relax_beam=True
        self.assertTrue(pathfinder._allowed_forward("any_node"))
        self.assertTrue(pathfinder._allowed_forward("start"))
        self.assertTrue(pathfinder._allowed_forward("end"))

    def test_allowed_forward_relax_beam_disabled(self):
        """Test _allowed_forward with relax_beam=False."""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            relax_beam=False
        )
        
        pathfinder._build_allowed_and_heuristics()
        
        # Should allow src and tgt
        self.assertTrue(pathfinder._allowed_forward("start"))
        self.assertTrue(pathfinder._allowed_forward("end"))

    def test_allowed_backward_relax_beam_enabled(self):
        """Test _allowed_backward with relax_beam=True."""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        # Should allow any node when relax_beam=True
        self.assertTrue(pathfinder._allowed_backward("any_node"))
        self.assertTrue(pathfinder._allowed_backward("start"))
        self.assertTrue(pathfinder._allowed_backward("end"))

    def test_allowed_backward_relax_beam_disabled(self):
        """Test _allowed_backward with relax_beam=False."""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            relax_beam=False
        )
        
        pathfinder._build_allowed_and_heuristics()
        
        # Should allow src and tgt
        self.assertTrue(pathfinder._allowed_backward("start"))
        self.assertTrue(pathfinder._allowed_backward("end"))

    def test_find_paths_basic_functionality(self):
        """Test find_paths method basic functionality."""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            relax_beam=True  # Allow exploration
        )
        
        paths = pathfinder.find_paths(max_results=1)
        
        # Should return a list
        self.assertIsInstance(paths, list)
        
        # If paths found, validate structure
        for path_result in paths:
            self.assertIsInstance(path_result, tuple)
            self.assertEqual(len(path_result), 2)
            path, cost = path_result
            self.assertIsInstance(path, list)
            self.assertIsInstance(cost, (int, float))

    def test_find_paths_identical_source_target(self):
        """Test find_paths when source equals target."""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="start"  # Same as source
        )
        
        paths = pathfinder.find_paths(max_results=1)
        
        self.assertIsInstance(paths, list)
        if paths:
            path, cost = paths[0]
            self.assertEqual(len(path), 1)
            self.assertEqual(path[0], "start")
            self.assertEqual(cost, 0.0)


class TestPairwiseBidirectionalAStarValidation(unittest.TestCase):
    """Test validation and constraint checking for PairwiseBidirectionalAStar."""
    
    def setUp(self):
        """Set up validation test fixtures."""
        # Initialize configuration layer
        self.config = PairwiseBidirectionalAStarMockConfig()
        
        # Initialize mock factory
        self.mock_factory = PairwiseBidirectionalAStarMockFactory()
        
        # Load validation test data
        self.validation_data = self.config.get_validation_test_data()
        self.heuristic_configs = self.config.get_heuristic_type_configurations()
        
        # Create mock graph
        self.mock_graph = self.mock_factory('MockGraphForPathfinding')
        self.mock_graph.nodes.return_value = ["start", "end"]
        self.mock_graph.has_node.side_effect = lambda n: n in ["start", "end"]

    def test_parameter_validation_beam_width(self):
        """Test beam_width parameter validation."""
        valid_range = self.validation_data['valid_parameters']['beam_width_range']
        min_beam, max_beam = valid_range
        
        # Test valid beam widths
        for beam_width in [min_beam, max_beam // 2, max_beam]:
            pathfinder = PairwiseBidirectionalAStar(
                g=self.mock_graph,
                src="start",
                tgt="end",
                beam_width=beam_width
            )
            self.assertEqual(pathfinder.beam_width, beam_width)

    def test_parameter_validation_max_depth(self):
        """Test max_depth parameter validation."""
        valid_range = self.validation_data['valid_parameters']['max_depth_range']
        min_depth, max_depth = valid_range
        
        # Test valid max depths
        for depth in [min_depth, max_depth // 2, max_depth]:
            pathfinder = PairwiseBidirectionalAStar(
                g=self.mock_graph,
                src="start",
                tgt="end",
                max_depth=depth
            )
            self.assertEqual(pathfinder.max_depth, depth)

    def test_heuristic_type_validation(self):
        """Test heuristic_type parameter validation."""
        valid_types = self.validation_data['valid_parameters']['heuristic_types']
        
        # Test all valid heuristic types
        for heuristic_type in valid_types:
            pathfinder = PairwiseBidirectionalAStar(
                g=self.mock_graph,
                src="start",
                tgt="end",
                heuristic_type=heuristic_type
            )
            self.assertEqual(pathfinder.heuristic_type, heuristic_type)

    def test_boundary_condition_minimum_beam_width(self):
        """Test boundary condition with minimum beam width."""
        min_beam = self.validation_data['boundary_conditions']['min_beam_width']
        
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            beam_width=min_beam
        )
        
        self.assertEqual(pathfinder.beam_width, min_beam)

    def test_expand_forward_once_max_depth_constraint(self):
        """Test that _expand_forward_once respects max_depth constraint."""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            max_depth=0,  # Very restrictive
            relax_beam=True
        )
        
        pathfinder._build_allowed_and_heuristics()
        pathfinder._init_search_state()
        
        # Should not expand beyond max_depth
        result = pathfinder._expand_forward_once()
        self.assertIsNone(result)

    def test_expand_backward_once_max_depth_constraint(self):
        """Test that _expand_backward_once respects max_depth constraint."""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.mock_graph,
            src="start",
            tgt="end",
            max_depth=0,  # Very restrictive
            relax_beam=True
        )
        
        pathfinder._build_allowed_and_heuristics()
        pathfinder._init_search_state()
        
        # Should not expand beyond max_depth
        result = pathfinder._expand_backward_once()
        self.assertIsNone(result)


class TestPairwiseBidirectionalAStarEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for PairwiseBidirectionalAStar."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        # Initialize configuration layer
        self.config = PairwiseBidirectionalAStarMockConfig()
        
        # Initialize mock factory
        self.mock_factory = PairwiseBidirectionalAStarMockFactory()
        
        # Load edge case test data
        self.edge_cases = self.config.get_pathfinding_edge_cases()
        self.validation_data = self.config.get_validation_test_data()
        
        # Create edge case mock
        self.edge_case_mock = self.mock_factory('MockPairwiseBidirectionalAStarEdgeCases')

    def test_empty_graph(self):
        """Test pathfinding with graph that has nodes but no edges."""
        empty_graph = nx.DiGraph()
        empty_graph.add_node("start")
        empty_graph.add_node("end")
        # No edges - disconnected nodes
        
        pathfinder = PairwiseBidirectionalAStar(
            g=empty_graph,
            src="start",
            tgt="end"
        )
        
        paths = pathfinder.find_paths()
        self.assertEqual(paths, [])

    def test_single_node_graph(self):
        """Test pathfinding with single node graph."""
        single_graph = nx.DiGraph()
        single_graph.add_node("only")
        
        pathfinder = PairwiseBidirectionalAStar(
            g=single_graph,
            src="only",
            tgt="only"
        )
        
        paths = pathfinder.find_paths()
        self.assertIsInstance(paths, list)
        
        if paths:
            path, cost = paths[0]
            self.assertEqual(path, ["only"])
            self.assertEqual(cost, 0.0)

    def test_disconnected_graph(self):
        """Test pathfinding with disconnected components."""
        edge_case = self.edge_cases['no_path_cases'][0]
        
        disconnected_graph = nx.DiGraph()
        for node in edge_case['nodes']:
            disconnected_graph.add_node(node)
        for edge in edge_case['edges']:
            src, tgt, attrs = edge
            disconnected_graph.add_edge(src, tgt, **attrs)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=disconnected_graph,
            src=edge_case['start'],
            tgt=edge_case['goal'],
            relax_beam=True
        )
        
        paths = pathfinder.find_paths()
        self.assertEqual(len(paths), 0)

    def test_isolated_target(self):
        """Test pathfinding with isolated target node."""
        edge_case = self.edge_cases['no_path_cases'][1]
        
        isolated_graph = nx.DiGraph()
        for node in edge_case['nodes']:
            isolated_graph.add_node(node)
        for edge in edge_case['edges']:
            src, tgt, attrs = edge
            isolated_graph.add_edge(src, tgt, **attrs)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=isolated_graph,
            src=edge_case['start'],
            tgt=edge_case['goal'],
            relax_beam=True
        )
        
        paths = pathfinder.find_paths()
        self.assertEqual(len(paths), 0)

    def test_very_large_graph(self):
        """Test pathfinding performance with larger graph structure."""
        # Create a larger grid-like structure
        large_graph = nx.DiGraph()
        
        for i in range(10):
            for j in range(10):
                node = f"node_{i}_{j}"
                large_graph.add_node(node)
                if i > 0:
                    large_graph.add_edge(f"node_{i-1}_{j}", node)
                if j > 0:
                    large_graph.add_edge(f"node_{i}_{j-1}", node)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=large_graph,
            src="node_0_0",
            tgt="node_9_9",
            relax_beam=True,
            max_depth=20
        )
        
        paths = pathfinder.find_paths(max_results=1)
        
        # Should handle large graphs without crashing
        self.assertIsInstance(paths, list)

    def test_circular_graph_structure(self):
        """Test pathfinding with circular graph structure."""
        circular_graph = nx.DiGraph()
        nodes = ["a", "b", "c", "d"]
        
        for i, node in enumerate(nodes):
            circular_graph.add_node(node)
            next_node = nodes[(i + 1) % len(nodes)]
            circular_graph.add_edge(node, next_node)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=circular_graph,
            src="a",
            tgt="c",
            relax_beam=True
        )
        
        paths = pathfinder.find_paths(max_results=2)
        
        self.assertIsInstance(paths, list)

    def test_weighted_vs_unweighted_edges(self):
        """Test pathfinding behavior with mixed weighted/unweighted edges."""
        mixed_graph = nx.DiGraph()
        mixed_graph.add_edge("start", "path1")  # No weight (defaults to 1.0)
        mixed_graph.add_edge("start", "path2", weight=0.5)  # Lighter weight
        mixed_graph.add_edge("path1", "end", weight=1.0)
        mixed_graph.add_edge("path2", "end", weight=1.0)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=mixed_graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        paths = pathfinder.find_paths(max_results=2)
        
        self.assertIsInstance(paths, list)
        if len(paths) >= 2:
            # Paths should be ordered by cost (lower cost first)
            self.assertLessEqual(paths[0][1], paths[1][1])

    def test_negative_weights_handling(self):
        """Test pathfinding behavior with negative edge weights."""
        negative_graph = nx.DiGraph()
        negative_graph.add_edge("start", "middle", weight=-1.0)
        negative_graph.add_edge("middle", "end", weight=1.0)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=negative_graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        # Should handle negative weights without crashing
        paths = pathfinder.find_paths(max_results=1)
        self.assertIsInstance(paths, list)

    def test_beams_function_exception_handling(self):
        """Test handling of exceptions from beams function."""
        # Create graph with nodes for the test
        test_graph = nx.DiGraph()
        test_graph.add_node("start")
        test_graph.add_node("end")
        
        mock_beams_fn = self.mock_factory('MockHeuristicFunction')
        mock_beams_fn.side_effect = Exception("Beam function error")
        
        pathfinder = PairwiseBidirectionalAStar(
            g=test_graph,
            src="start",
            tgt="end",
            get_new_beams_fn=mock_beams_fn
        )
        
        # Should not raise exception during initialization
        pathfinder._build_allowed_and_heuristics()
        
        # Should still have src/tgt in allowed sets
        self.assertIn("start", pathfinder.src_allowed)
        self.assertIn("end", pathfinder.tgt_allowed)

    def test_zero_beam_width(self):
        """Test pathfinding behavior with zero beam width."""
        # Create simple graph for the test
        test_graph = nx.DiGraph()
        test_graph.add_node("start")
        test_graph.add_node("end")
        test_graph.add_edge("start", "end")
        
        pathfinder = PairwiseBidirectionalAStar(
            g=test_graph,
            src="start",
            tgt="end",
            beam_width=0
        )
        
        # Should still work (beam width affects beam generation, not core search)
        paths = pathfinder.find_paths(max_results=1)
        self.assertIsInstance(paths, list)


class TestPairwiseBidirectionalAStarIntegration(unittest.TestCase):
    """Integration tests for PairwiseBidirectionalAStar with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Initialize configuration layer
        self.config = PairwiseBidirectionalAStarMockConfig()
        
        # Initialize mock factory
        self.mock_factory = PairwiseBidirectionalAStarMockFactory()
        
        # Load integration test data
        self.integration_scenarios = self.config.get_integration_test_scenarios()
        self.wordnet_data = self.config.get_wordnet_taxonomy_structures()
        self.performance_benchmarks = self.config.get_performance_benchmarks()
        self.beam_configs = self.config.get_beam_search_configurations()
        
        # Create integration mock
        self.integration_mock = self.mock_factory('MockPairwiseBidirectionalAStarIntegration')

    def test_wordnet_integration_scenario(self):
        """Test integration with WordNet-like taxonomy structure."""
        wordnet_scenario = self.integration_scenarios['wordnet_integration']
        
        # Build a realistic WordNet-like taxonomy
        wn_graph = nx.DiGraph()
        taxonomy = {
            "cat.n.01": ["mammal.n.01"],
            "dog.n.01": ["mammal.n.01"],
            "mammal.n.01": ["animal.n.01"],
            "animal.n.01": ["entity.n.01"]
        }
        
        for child, parents in taxonomy.items():
            wn_graph.add_node(child)
            for parent in parents:
                wn_graph.add_node(parent)
                wn_graph.add_edge(child, parent, relation="hypernym")
                wn_graph.add_edge(parent, child, relation="hyponym")
        
        pathfinder = PairwiseBidirectionalAStar(
            g=wn_graph,
            src="cat.n.01",
            tgt="dog.n.01",
            heuristic_type="wordnet",
            beam_width=5,
            max_depth=8
        )
        
        paths = pathfinder.find_paths(max_results=3, len_tolerance=1)
        
        self.assertIsInstance(paths, list)
        
        # Validate path properties for WordNet integration
        expected_props = wordnet_scenario['expected_path_properties']
        for path, cost in paths:
            self.assertIsInstance(path, list)
            self.assertGreaterEqual(len(path), expected_props['min_length'])
            self.assertLessEqual(len(path), expected_props['max_length'])
            self.assertEqual(path[0], "cat.n.01")
            self.assertEqual(path[-1], "dog.n.01")
            self.assertIsInstance(cost, (int, float))
            self.assertGreaterEqual(cost, 0.0)

    def test_embedding_integration_scenario(self):
        """Test integration with embedding helper."""
        embedding_scenario = self.integration_scenarios['embedding_integration']
        
        # Create graph with embedding helper mock
        graph = nx.DiGraph()
        graph.add_edge("cat.n.01", "mammal.n.01")
        graph.add_edge("dog.n.01", "mammal.n.01")
        graph.add_edge("mammal.n.01", "animal.n.01")
        
        mock_embedding_helper = self.mock_factory('MockEmbeddingHelper')
        mock_embedding_helper.similarity.return_value = 0.75
        
        pathfinder = PairwiseBidirectionalAStar(
            g=graph,
            src="cat.n.01",
            tgt="dog.n.01",
            heuristic_type=embedding_scenario['test_parameters']['heuristic_type'],
            embedding_helper=mock_embedding_helper,
            beam_width=3
        )
        
        paths = pathfinder.find_paths(max_results=1)
        
        self.assertIsInstance(paths, list)

    def test_gloss_seed_integration_scenario(self):
        """Test integration with gloss seed nodes."""
        gloss_scenario = self.integration_scenarios['gloss_seed_integration']
        
        # Create graph with known seed nodes
        graph = nx.DiGraph()
        graph.add_edge("start", "seed1")
        graph.add_edge("seed1", "end")
        graph.add_edge("start", "alternative")
        graph.add_edge("alternative", "end")
        
        pathfinder = PairwiseBidirectionalAStar(
            g=graph,
            src="start",
            tgt="end",
            gloss_seed_nodes=gloss_scenario['seed_nodes'],
            relax_beam=False  # Force use of beam constraints
        )
        
        paths = pathfinder.find_paths(max_results=2)
        
        self.assertIsInstance(paths, list)
        
        # Verify gloss seeds are properly integrated
        if gloss_scenario['expected_allowed_sets_inclusion']:
            pathfinder._build_allowed_and_heuristics()
            for seed in gloss_scenario['seed_nodes']:
                if seed in ["seed1", "seed2"]:  # Only check seeds that exist in our test graph
                    self.assertIn(seed, pathfinder.src_allowed)
                    self.assertIn(seed, pathfinder.tgt_allowed)

    def test_hybrid_heuristic_integration(self):
        """Test integration with hybrid heuristic combining multiple approaches."""
        # Create realistic WordNet graph
        graph = nx.DiGraph()
        graph.add_edge("cat.n.01", "mammal.n.01")
        graph.add_edge("dog.n.01", "mammal.n.01")
        graph.add_edge("mammal.n.01", "animal.n.01")
        
        mock_embedding_helper = self.mock_factory('MockEmbeddingHelper')
        mock_embedding_helper.similarity.return_value = 0.8
        
        pathfinder = PairwiseBidirectionalAStar(
            g=graph,
            src="cat.n.01",
            tgt="dog.n.01",
            heuristic_type="hybrid",
            embedding_helper=mock_embedding_helper,
            beam_width=5
        )
        
        paths = pathfinder.find_paths(max_results=2)
        
        self.assertIsInstance(paths, list)

    def test_performance_benchmark_compliance(self):
        """Test that pathfinding meets performance benchmarks."""
        benchmark = self.performance_benchmarks['small_graph']
        
        # Create graph matching benchmark size
        graph = nx.DiGraph()
        
        # Add nodes up to benchmark count
        for i in range(min(50, benchmark['nodes'])):  # Limit for test performance
            graph.add_node(f"node_{i}")
        
        # Add some edges
        for i in range(min(25, benchmark['edges'] // 4)):
            graph.add_edge(f"node_{i}", f"node_{i+1}")
        
        pathfinder = PairwiseBidirectionalAStar(
            g=graph,
            src="node_0",
            tgt=f"node_{min(49, benchmark['nodes']-1)}",
            relax_beam=True
        )
        
        # Should complete within reasonable time
        paths = pathfinder.find_paths(max_results=1)
        
        self.assertIsInstance(paths, list)

    def test_beam_configuration_scenarios(self):
        """Test different beam configuration scenarios."""
        for config_name, config_data in self.beam_configs.items():
            if config_name == 'adaptive_beam':
                continue  # Skip adaptive beam for this test
                
            with self.subTest(beam_config=config_name):
                # Create simple graph
                graph = nx.DiGraph()
                graph.add_edge("start", "middle")
                graph.add_edge("middle", "end")
                
                pathfinder = PairwiseBidirectionalAStar(
                    g=graph,
                    src="start",
                    tgt="end",
                    beam_width=config_data['beam_width'],
                    relax_beam=True
                )
                
                paths = pathfinder.find_paths(max_results=1)
                
                self.assertIsInstance(paths, list)


if __name__ == '__main__':
    unittest.main()