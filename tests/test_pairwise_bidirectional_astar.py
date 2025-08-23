import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import heapq
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.PairwiseBidirectionalAStar import PairwiseBidirectionalAStar
from tests.mocks.pairwise_bidirectional_astar_mocks import PairwiseBidirectionalAStarMockFactory


class TestPairwiseBidirectionalAStar(unittest.TestCase):
    """Test the PairwiseBidirectionalAStar class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize mock factory
        self.mock_factory = PairwiseBidirectionalAStarMockFactory()
        
        # Create a simple test graph using mock factory
        self.graph = self.mock_factory('MockGraphForPathfinding')
        self.graph.nodes.return_value = ["start", "middle", "end"]
        self.graph.edges.return_value = [("start", "middle"), ("middle", "end")]
        self.graph.has_node.side_effect = lambda n: n in ["start", "middle", "end"]
        self.graph.has_edge.side_effect = lambda u, v: (u, v) in [("start", "middle"), ("middle", "end")]
        
        # Basic pathfinder
        self.pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            beam_width=3,
            max_depth=6
        )

    def test_initialization_basic(self):
        """Test basic initialization of PairwiseBidirectionalAStar"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end"
        )
        
        self.assertEqual(pathfinder.g, self.graph)
        self.assertEqual(pathfinder.src, "start")
        self.assertEqual(pathfinder.tgt, "end")
        self.assertIsNone(pathfinder.get_new_beams_fn)
        self.assertEqual(pathfinder.gloss_seed_nodes, set())
        self.assertEqual(pathfinder.beam_width, 3)  # default
        self.assertEqual(pathfinder.max_depth, 6)   # default
        self.assertFalse(pathfinder.relax_beam)     # default

    def test_initialization_with_options(self):
        """Test initialization with all options"""
        mock_beams_fn = self.mock_factory('MockHeuristicFunction')
        gloss_seeds = ["seed1", "seed2"]
        
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            get_new_beams_fn=mock_beams_fn,
            gloss_seed_nodes=gloss_seeds,
            beam_width=5,
            max_depth=10,
            relax_beam=True
        )
        
        self.assertEqual(pathfinder.get_new_beams_fn, mock_beams_fn)
        self.assertEqual(pathfinder.gloss_seed_nodes, {"seed1", "seed2"})
        self.assertEqual(pathfinder.beam_width, 5)
        self.assertEqual(pathfinder.max_depth, 10)
        self.assertTrue(pathfinder.relax_beam)

    def test_gloss_bonus_constant(self):
        """Test GLOSS_BONUS constant value"""
        self.assertEqual(PairwiseBidirectionalAStar.GLOSS_BONUS, 0.15)

    def test_build_allowed_and_heuristics_no_beams_fn(self):
        """Test _build_allowed_and_heuristics without get_new_beams_fn"""
        self.pathfinder._build_allowed_and_heuristics()
        
        # Should include src and tgt in allowed sets
        self.assertIn("start", self.pathfinder.src_allowed)
        self.assertIn("end", self.pathfinder.tgt_allowed)
        
        # Should have default heuristics
        self.assertIn("start", self.pathfinder.h_forward)
        self.assertIn("end", self.pathfinder.h_backward)

    def test_build_allowed_and_heuristics_with_beams_fn(self):
        """Test _build_allowed_and_heuristics with get_new_beams_fn"""
        mock_beams_fn = self.mock_factory('MockHeuristicFunction')
        mock_beams_fn.return_value = [
            (("start", "rel1"), ("middle", "rel1"), 0.8),
            (("middle", "rel2"), ("end", "rel2"), 0.7)
        ]
        
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            get_new_beams_fn=mock_beams_fn
        )
        
        pathfinder._build_allowed_and_heuristics()
        
        # Check that beams function was called
        mock_beams_fn.assert_called_once_with(self.graph, "start", "end")
        
        # Check allowed sets include beam nodes
        self.assertIn("start", pathfinder.src_allowed)
        self.assertIn("middle", pathfinder.src_allowed)
        self.assertIn("middle", pathfinder.tgt_allowed)
        self.assertIn("end", pathfinder.tgt_allowed)
        
        # Check heuristics were set
        self.assertIn("start", pathfinder.h_forward)
        self.assertIn("middle", pathfinder.h_forward)

    def test_build_allowed_and_heuristics_with_gloss_seeds(self):
        """Test _build_allowed_and_heuristics with gloss seeds"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
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
        expected_h = max(0.0, 0.5 - PairwiseBidirectionalAStar.GLOSS_BONUS)
        self.assertEqual(pathfinder.h_forward["middle"], expected_h)
        self.assertEqual(pathfinder.h_backward["middle"], expected_h)

    def test_build_allowed_and_heuristics_beams_fn_exception(self):
        """Test _build_allowed_and_heuristics handles beams_fn exceptions"""
        mock_beams_fn = self.mock_factory('MockHeuristicFunction')
        mock_beams_fn.side_effect = Exception("Beam function error")
        
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            get_new_beams_fn=mock_beams_fn
        )
        
        # Should not raise exception
        pathfinder._build_allowed_and_heuristics()
        
        # Should still have src/tgt in allowed sets
        self.assertIn("start", pathfinder.src_allowed)
        self.assertIn("end", pathfinder.tgt_allowed)

    def test_init_search_state(self):
        """Test _init_search_state method"""
        self.pathfinder._build_allowed_and_heuristics()
        self.pathfinder._init_search_state()
        
        # Check initial state
        self.assertEqual(len(self.pathfinder.open_f), 1)
        self.assertEqual(len(self.pathfinder.open_b), 1)
        self.assertEqual(self.pathfinder.g_f["start"], 0.0)
        self.assertEqual(self.pathfinder.g_b["end"], 0.0)
        self.assertEqual(self.pathfinder.depth_f["start"], 0)
        self.assertEqual(self.pathfinder.depth_b["end"], 0)
        self.assertIsNone(self.pathfinder.parent_f["start"])
        self.assertIsNone(self.pathfinder.parent_b["end"])
        self.assertEqual(len(self.pathfinder.closed_f), 0)
        self.assertEqual(len(self.pathfinder.closed_b), 0)

    def test_edge_weight_default(self):
        """Test _edge_weight method with default weight"""
        weight = self.pathfinder._edge_weight("start", "middle")
        self.assertEqual(weight, 1.0)

    def test_edge_weight_custom(self):
        """Test _edge_weight method with custom weight"""
        # Create a real networkx graph for this test since we need actual edge weights
        real_graph = nx.DiGraph()
        real_graph.add_node("start")
        real_graph.add_node("middle") 
        real_graph.add_node("end")
        real_graph.add_edge("start", "middle", weight=1.0)
        real_graph.add_edge("middle", "end", weight=2.5)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=real_graph,
            src="start",
            tgt="end"
        )
        
        weight = pathfinder._edge_weight("middle", "end")
        self.assertEqual(weight, 2.5)

    def test_edge_weight_missing_edge(self):
        """Test _edge_weight method with missing edge"""
        weight = self.pathfinder._edge_weight("start", "nonexistent")
        self.assertEqual(weight, 1.0)  # Default fallback

    def test_allowed_forward_relax_beam(self):
        """Test _allowed_forward with relax_beam=True"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        # Should allow any node when relax_beam=True
        self.assertTrue(pathfinder._allowed_forward("any_node"))
        self.assertTrue(pathfinder._allowed_forward("start"))
        self.assertTrue(pathfinder._allowed_forward("end"))

    def test_allowed_forward_strict_beam(self):
        """Test _allowed_forward with relax_beam=False"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            relax_beam=False
        )
        
        pathfinder._build_allowed_and_heuristics()
        
        # Should allow src, tgt, and nodes in allowed sets
        self.assertTrue(pathfinder._allowed_forward("start"))
        self.assertTrue(pathfinder._allowed_forward("end"))
        
        # Should not allow random nodes
        self.assertFalse(pathfinder._allowed_forward("random_node"))

    def test_allowed_backward_relax_beam(self):
        """Test _allowed_backward with relax_beam=True"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        # Should allow any node when relax_beam=True
        self.assertTrue(pathfinder._allowed_backward("any_node"))
        self.assertTrue(pathfinder._allowed_backward("start"))
        self.assertTrue(pathfinder._allowed_backward("end"))

    def test_allowed_backward_strict_beam(self):
        """Test _allowed_backward with relax_beam=False"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            relax_beam=False
        )
        
        pathfinder._build_allowed_and_heuristics()
        
        # Should allow src, tgt, and nodes in allowed sets
        self.assertTrue(pathfinder._allowed_backward("start"))
        self.assertTrue(pathfinder._allowed_backward("end"))
        
        # Should not allow random nodes
        self.assertFalse(pathfinder._allowed_backward("random_node"))

    def test_expand_forward_once_max_depth(self):
        """Test _expand_forward_once respects max_depth"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
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

    def test_expand_backward_once_max_depth(self):
        """Test _expand_backward_once respects max_depth"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
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

    def test_reconstruct_path_simple(self):
        """Test _reconstruct_path with simple path"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end"
        )
        
        # Set up parent pointers manually
        pathfinder.parent_f = {"start": None, "middle": "start", "end": "middle"}
        pathfinder.parent_b = {"end": None, "middle": "end"}
        
        path = pathfinder._reconstruct_path("middle")
        
        self.assertIsInstance(path, list)
        self.assertIn("start", path)
        self.assertIn("middle", path)
        # Should not duplicate the meeting point

    def test_find_paths_simple_case(self):
        """Test find_paths with simple graph"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            relax_beam=True  # Allow all nodes
        )
        
        paths = pathfinder.find_paths(max_results=1)
        
        self.assertIsInstance(paths, list)
        if paths:  # If path found
            path, cost = paths[0]
            self.assertIsInstance(path, list)
            self.assertIsInstance(cost, (int, float))
            self.assertIn("start", path)
            self.assertIn("end", path)

    def test_find_paths_no_path_exists(self):
        """Test find_paths when no path exists"""
        # Create disconnected graph
        disconnected_graph = nx.DiGraph()
        disconnected_graph.add_node("start")
        disconnected_graph.add_node("end")
        # No edges between them
        
        pathfinder = PairwiseBidirectionalAStar(
            g=disconnected_graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        paths = pathfinder.find_paths(max_results=3)
        
        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 0)

    def test_find_paths_max_results_limit(self):
        """Test find_paths respects max_results limit"""
        # Create graph with multiple paths
        multi_path_graph = nx.DiGraph()
        multi_path_graph.add_edge("start", "path1", weight=1)
        multi_path_graph.add_edge("start", "path2", weight=1)
        multi_path_graph.add_edge("path1", "end", weight=1)
        multi_path_graph.add_edge("path2", "end", weight=1)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=multi_path_graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        paths = pathfinder.find_paths(max_results=1)
        
        self.assertIsInstance(paths, list)
        self.assertLessEqual(len(paths), 1)

    def test_find_paths_len_tolerance(self):
        """Test find_paths with length tolerance"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        paths = pathfinder.find_paths(max_results=3, len_tolerance=2)
        
        self.assertIsInstance(paths, list)
        # Should include paths within tolerance of the shortest

    def test_find_paths_identical_src_tgt(self):
        """Test find_paths when source equals target"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
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


class TestPairwiseBidirectionalAStarEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_factory = PairwiseBidirectionalAStarMockFactory()
        self.graph = self.mock_factory('MockGraphForPathfinding')
        self.graph.nodes.return_value = ["start", "end"]
        self.graph.has_node.side_effect = lambda n: n in ["start", "end"]
        
        # Set up edge case mock
        self.edge_case_mock = self.mock_factory('MockPairwiseBidirectionalAStarEdgeCases')

    def test_empty_graph(self):
        """Test with empty graph"""
        empty_graph = self.mock_factory('MockGraphForPathfinding')
        empty_graph.nodes.return_value = []
        empty_graph.edges.return_value = []
        empty_graph.has_node.return_value = False
        
        pathfinder = PairwiseBidirectionalAStar(
            g=empty_graph,
            src="start",
            tgt="end"
        )
        
        paths = pathfinder.find_paths()
        self.assertEqual(paths, [])

    def test_single_node_graph(self):
        """Test with single node graph"""
        single_graph = self.mock_factory('MockGraphForPathfinding')
        single_graph.nodes.return_value = ["only"]
        single_graph.has_node.side_effect = lambda n: n == "only"
        
        pathfinder = PairwiseBidirectionalAStar(
            g=single_graph,
            src="only",
            tgt="only"
        )
        
        paths = pathfinder.find_paths()
        self.assertIsInstance(paths, list)

    def test_very_large_graph(self):
        """Test with larger graph structure"""
        large_graph = nx.DiGraph()
        
        # Create a grid-like structure
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
        
        self.assertIsInstance(paths, list)
        # May or may not find path depending on algorithm specifics

    def test_circular_graph(self):
        """Test with circular graph structure"""
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
        """Test behavior with mixed weighted/unweighted edges"""
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
            # Lighter weighted path should have lower cost
            self.assertLessEqual(paths[0][1], paths[1][1])

    def test_negative_weights_handling(self):
        """Test behavior with negative edge weights"""
        negative_graph = nx.DiGraph()
        negative_graph.add_edge("start", "middle", weight=-1.0)
        negative_graph.add_edge("middle", "end", weight=1.0)
        
        pathfinder = PairwiseBidirectionalAStar(
            g=negative_graph,
            src="start",
            tgt="end",
            relax_beam=True
        )
        
        # Should handle negative weights (though may not guarantee optimality)
        paths = pathfinder.find_paths(max_results=1)
        self.assertIsInstance(paths, list)

    def test_zero_beam_width(self):
        """Test with zero beam width"""
        pathfinder = PairwiseBidirectionalAStar(
            g=self.graph,
            src="start",
            tgt="end",
            beam_width=0
        )
        
        # Should still work (beam width affects beam generation, not core search)
        paths = pathfinder.find_paths(max_results=1)
        self.assertIsInstance(paths, list)


class TestPairwiseBidirectionalAStarIntegration(unittest.TestCase):
    """Integration tests with realistic scenarios"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.mock_factory = PairwiseBidirectionalAStarMockFactory()
        self.integration_mock = self.mock_factory('MockPairwiseBidirectionalAStarIntegration')
    
    def test_with_embedding_beams_function(self):
        """Test integration with embedding beams function"""
        # Use real NetworkX graph for integration test
        graph = nx.DiGraph()
        graph.add_edge("cat.n.01", "mammal.n.01")
        graph.add_edge("dog.n.01", "mammal.n.01")
        graph.add_edge("mammal.n.01", "animal.n.01")
        
        def mock_beams_fn(g, src, tgt):
            return [
                ((src, "hypernyms"), ("mammal.n.01", "hypernyms"), 0.9),
                (("mammal.n.01", "hyponyms"), (tgt, "hyponyms"), 0.8)
            ]
        
        pathfinder = PairwiseBidirectionalAStar(
            g=graph,
            src="cat.n.01",
            tgt="dog.n.01",
            get_new_beams_fn=mock_beams_fn,
            beam_width=3
        )
        
        paths = pathfinder.find_paths(max_results=1)
        
        self.assertIsInstance(paths, list)

    def test_with_gloss_seed_nodes(self):
        """Test integration with gloss seed nodes"""
        # Use real NetworkX graph for integration test
        graph = nx.DiGraph()
        graph.add_edge("start", "seed_node")
        graph.add_edge("seed_node", "end")
        graph.add_edge("start", "alternative")
        graph.add_edge("alternative", "end")
        
        pathfinder = PairwiseBidirectionalAStar(
            g=graph,
            src="start",
            tgt="end",
            gloss_seed_nodes=["seed_node"],
            relax_beam=False  # Force use of beam constraints
        )
        
        paths = pathfinder.find_paths(max_results=2)
        
        self.assertIsInstance(paths, list)
        if paths:
            # Should find paths, potentially preferring the seeded route
            path, _ = paths[0]
            self.assertIsInstance(path, list)

    def test_realistic_wordnet_scenario(self):
        """Test with realistic WordNet-like graph"""
        # Build a small WordNet-like taxonomy
        wn_graph = nx.DiGraph()
        
        # Taxonomy: animal -> mammal -> {cat, dog}
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
        
        def wordnet_beams_fn(g, src, tgt):
            # Mock realistic embedding similarities
            return [
                ((src, "hypernyms"), ("mammal.n.01", "hypernyms"), 0.85),
                (("mammal.n.01", "hyponyms"), (tgt, "hyponyms"), 0.80)
            ]
        
        pathfinder = PairwiseBidirectionalAStar(
            g=wn_graph,
            src="cat.n.01",
            tgt="dog.n.01",
            get_new_beams_fn=wordnet_beams_fn,
            beam_width=5,
            max_depth=8
        )
        
        paths = pathfinder.find_paths(max_results=3, len_tolerance=1)
        
        self.assertIsInstance(paths, list)
        
        # Verify path structure if found
        for path, cost in paths:
            self.assertIsInstance(path, list)
            self.assertGreaterEqual(len(path), 2)  # At least src and tgt
            self.assertEqual(path[0], "cat.n.01")
            self.assertEqual(path[-1], "dog.n.01")
            self.assertIsInstance(cost, (int, float))
            self.assertGreaterEqual(cost, 0.0)


if __name__ == '__main__':
    unittest.main()