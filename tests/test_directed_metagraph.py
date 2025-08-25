import unittest
import networkx as nx
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.DirectedMetagraph import DirectedMetagraph
from tests.mocks.directed_metagraph_mocks import DirectedMetagraphMockFactory
from tests.config.directed_metagraph_config import DirectedMetagraphMockConfig


class TestDirectedMetagraphBase(unittest.TestCase):
    """Base class for DirectedMetagraph tests with common setup"""
    
    def setUp(self):
        """Set up test fixtures - common setup for all DirectedMetagraph test classes"""
        # Initialize mock factory and config
        self.mock_factory = DirectedMetagraphMockFactory()
        self.mock_config = DirectedMetagraphMockConfig()
        
        # Get basic test vertex lists from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        edge_structures = self.mock_config.get_edge_structures()
        
        self.simple_vert_list = [
            vertex_structures['simple_vertices'][0],  # ("word1", {"pos": "NOUN"})
            vertex_structures['simple_vertices'][1],  # ("word2", {"pos": "VERB"})
            edge_structures['simple_edges'][0]       # ((0, 1), {"relation": "subject"})
        ]
        
        self.complex_vert_list = [
            vertex_structures['person_action_vertices'][0],  # ("John", {"type": "person", "pos": "NOUN"})
            vertex_structures['person_action_vertices'][1],  # ("runs", {"type": "action", "pos": "VERB"})
            ("fast", {"type": "manner", "pos": "ADV"}),
            ((0, 1), {"relation": "agent"}),
            ((1, 2), {"relation": "manner"}),
            ([0, 1, 2], {"relation": "action_group"})
        ]


class TestDirectedMetagraph(TestDirectedMetagraphBase):
    """Test the DirectedMetagraph class basic functionality"""
    
    def test_initialization_empty(self):
        """Test initialization with no arguments creates empty graph"""
        dg = DirectedMetagraph()
        self.assertEqual(len(dg.metaverts), 0)
        self.assertIsInstance(dg.metaverts, dict)
    
    def test_initialization_none(self):
        """Test initialization with None creates empty graph"""
        dg = DirectedMetagraph(None)
        self.assertEqual(len(dg.metaverts), 0)
        self.assertIsInstance(dg.metaverts, dict)
    
    def test_initialization_with_vert_list(self):
        """Test initialization with valid vertex list"""
        dg = DirectedMetagraph(self.simple_vert_list)
        self.assertEqual(len(dg.metaverts), 3)
        self.assertEqual(dg.metaverts[0], ("word1", {"pos": "NOUN"}))
        self.assertEqual(dg.metaverts[1], ("word2", {"pos": "VERB"}))
        self.assertEqual(dg.metaverts[2], ((0, 1), {"relation": "subject"}))
    
    def test_initialization_with_complex_vert_list(self):
        """Test initialization with complex vertex list"""
        dg = DirectedMetagraph(self.complex_vert_list)
        self.assertEqual(len(dg.metaverts), 6)
        # Check that undirected metaedge is canonicalized (sorted)
        last_vert = dg.metaverts[5]  # Last vertex (index 5)
        self.assertEqual(last_vert[0], [0, 1, 2])  # Should be sorted


class TestDirectedMetagraphValidation(TestDirectedMetagraphBase):
    """Test validation methods of DirectedMetagraph"""
    
    def test_validate_vert_atomic_string_only(self):
        """Test validation of atomic string vertex with no metadata"""
        DirectedMetagraph.validate_vert(0, ("word",))
        # Should not raise any exception
    
    def test_validate_vert_atomic_string_with_metadata(self):
        """Test validation of atomic string vertex with metadata"""
        DirectedMetagraph.validate_vert(0, ("word", {"pos": "NOUN"}))
        # Should not raise any exception
    
    def test_validate_vert_directed_relation(self):
        """Test validation of directed relation vertex"""
        # First create some atomic vertices
        DirectedMetagraph.validate_vert(2, ((0, 1), {"relation": "subject"}))
        # Should not raise any exception
    
    def test_validate_vert_undirected_relation(self):
        """Test validation of undirected relation vertex"""
        DirectedMetagraph.validate_vert(3, ([0, 1, 2], {"relation": "group"}))
        # Should not raise any exception
    
    def test_validate_vert_invalid_length(self):
        """Test validation fails for invalid tuple length"""
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(0, ("word", {"pos": "NOUN"}, "extra"))
        
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(0, ())
    
    def test_validate_vert_invalid_metadata_type(self):
        """Test validation fails for non-dict metadata"""
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(0, ("word", "not_a_dict"))
        
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(0, ("word", ["not", "a", "dict"]))
    
    def test_validate_vert_directed_missing_relation(self):
        """Test validation fails for directed relation without 'relation' key"""
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(2, ((0, 1), {"type": "connection"}))
    
    def test_validate_vert_undirected_missing_relation(self):
        """Test validation fails for undirected relation without 'relation' key"""
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(3, ([0, 1, 2], {"type": "group"}))
    
    def test_validate_vert_directed_forward_reference(self):
        """Test validation fails for directed relation with forward references"""
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(1, ((0, 2), {"relation": "subject"}))
        
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(1, ((2, 0), {"relation": "subject"}))
    
    def test_validate_vert_undirected_forward_reference(self):
        """Test validation fails for undirected relation with forward references"""
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(2, ([0, 1, 3], {"relation": "group"}))
    
    def test_validate_vert_invalid_type(self):
        """Test validation fails for invalid vertex types"""
        with self.assertRaises(ValueError):
            DirectedMetagraph.validate_vert(0, (123, {"data": "value"}))
        
        with self.assertRaises(ValueError):
            DirectedMetagraph.validate_vert(0, ({"invalid": "type"}, {"data": "value"}))
    
    def test_validate_vert_directed_missing_metadata(self):
        """Test validation fails for directed relation without metadata"""
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(2, ((0, 1),))
    
    def test_validate_vert_undirected_missing_metadata(self):
        """Test validation fails for undirected relation without metadata"""
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_vert(3, ([0, 1, 2],))
    
    def test_validate_graph_valid(self):
        """Test validation of valid graph"""
        # Get validation test data from config
        validation_data = self.mock_config.get_validation_test_data()
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        edge_structures = self.mock_config.get_edge_structures()
        
        vert_list = [
            vertex_structures['simple_vertices'][0],  # ("word1", {"pos": "NOUN"})
            vertex_structures['simple_vertices'][1],  # ("word2", {"pos": "VERB"})
            edge_structures['simple_edges'][0]       # ((0, 1), {"relation": "subject"})
        ]
        DirectedMetagraph.validate_graph(vert_list)
        # Should not raise any exception
    
    def test_validate_graph_invalid(self):
        """Test validation fails for invalid graph"""
        # Get validation test data from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        
        vert_list = [
            vertex_structures['simple_vertices'][0],  # ("word1", {"pos": "NOUN"})
            vertex_structures['simple_vertices'][1],  # ("word2", {"pos": "VERB"})
            ((0, 2), {"relation": "subject"})  # Invalid forward reference
        ]
        with self.assertRaises(AssertionError):
            DirectedMetagraph.validate_graph(vert_list)


class TestDirectedMetagraphCanonicalization(TestDirectedMetagraphBase):
    """Test canonicalization methods of DirectedMetagraph"""
    
    def test_canonicalize_vert_atomic(self):
        """Test canonicalization of atomic vertex"""
        # Get vertex from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        original = vertex_structures['simple_vertices'][0]  # ("word1", {"pos": "NOUN"})
        canonical = DirectedMetagraph.canonicalize_vert(original)
        self.assertEqual(canonical, original)
    
    def test_canonicalize_vert_directed(self):
        """Test canonicalization of directed relation vertex"""
        # Get edge from config
        edge_structures = self.mock_config.get_edge_structures()
        original = edge_structures['simple_edges'][0]  # ((0, 1), {"relation": "subject"})
        canonical = DirectedMetagraph.canonicalize_vert(original)
        self.assertEqual(canonical, original)
    
    def test_canonicalize_vert_undirected_sorted(self):
        """Test canonicalization sorts undirected relation vertices"""
        # Get canonicalization test data from config
        canonicalization_scenarios = self.mock_config.get_canonicalization_test_scenarios()
        unsorted_case = canonicalization_scenarios['unsorted_groups']
        
        original = unsorted_case['input_group']
        canonical = DirectedMetagraph.canonicalize_vert(original)
        expected = unsorted_case['expected_canonical']
        self.assertEqual(canonical, expected)
    
    def test_canonicalize_vert_undirected_already_sorted(self):
        """Test canonicalization of already sorted undirected vertex"""
        # Get canonicalization test data from config
        canonicalization_scenarios = self.mock_config.get_canonicalization_test_scenarios()
        sorted_case = canonicalization_scenarios['already_sorted_groups']
        
        original = sorted_case['input_group']
        canonical = DirectedMetagraph.canonicalize_vert(original)
        expected = sorted_case['expected_canonical']
        self.assertEqual(canonical, expected)


class TestDirectedMetagraphNetworkXConversion(TestDirectedMetagraphBase):
    """Test NetworkX conversion functionality"""
    
    def setUp(self):
        """Set up test fixtures with NetworkX-specific data"""
        super().setUp()
        
        # Get vertex and edge structures from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        edge_structures = self.mock_config.get_edge_structures()
        
        # Set up graphs for NetworkX conversion testing
        self.simple_graph = DirectedMetagraph([
            vertex_structures['simple_vertices'][0],  # ("word1", {"pos": "NOUN"})
            vertex_structures['simple_vertices'][1],  # ("word2", {"pos": "VERB"})
            edge_structures['simple_edges'][0]       # ((0, 1), {"relation": "subject"})
        ])
        
        # Get complex structures from config for realistic testing
        complex_structures = self.mock_config.get_complex_graph_structures()
        self.complex_graph = DirectedMetagraph([
            ("John", {"type": "person"}),
            ("runs", {"type": "action"}),
            ("fast", {"type": "manner"}),
            ((0, 1), {"relation": "agent"}),
            ([1, 2], {"relation": "manner_group"})
        ])
    
    def test_add_vert_to_nx_atomic(self):
        """Test adding atomic vertex to NetworkX graph"""
        # Get vertex data from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        node_data = vertex_structures['simple_vertices'][0]  # ("word1", {"pos": "NOUN"})
        
        G = nx.DiGraph()
        G = DirectedMetagraph.add_vert_to_nx(G, 0, node_data)
        
        self.assertEqual(len(G.nodes()), 1)
        self.assertIn(0, G.nodes())
        self.assertEqual(G.nodes[0]["label"], node_data[0])
    
    def test_add_vert_to_nx_directed_relation(self):
        """Test adding directed relation vertex to NetworkX graph"""
        # Get vertex and edge data from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        edge_structures = self.mock_config.get_edge_structures()
        
        G = nx.DiGraph()
        # First add the atomic vertices that the relation references
        G.add_node(0, label=vertex_structures['simple_vertices'][0][0])  # "word1"
        G.add_node(1, label=vertex_structures['simple_vertices'][1][0])  # "word2"
        
        node_data = edge_structures['simple_edges'][0]  # ((0, 1), {"relation": "subject"})
        G = DirectedMetagraph.add_vert_to_nx(G, 2, node_data)
        
        self.assertEqual(len(G.nodes()), 3)
        self.assertIn(2, G.nodes())
        self.assertEqual(G.nodes[2]["label"], f"{node_data[1]['relation']}(0, 1)")
        
        # Check edges
        self.assertIn((0, 2), G.edges())
        self.assertIn((1, 2), G.edges())
        self.assertEqual(G.edges[0, 2]["label"], "arg0")
        self.assertEqual(G.edges[1, 2]["label"], "arg1")
    
    def test_add_vert_to_nx_undirected_relation(self):
        """Test adding undirected relation vertex to NetworkX graph"""
        G = nx.DiGraph()
        # First add the atomic vertices that the relation references
        G.add_node(0, label="word1")
        G.add_node(1, label="word2")
        
        node_data = ([0, 1], {"relation": "group"})
        G = DirectedMetagraph.add_vert_to_nx(G, 2, node_data)
        
        self.assertEqual(len(G.nodes()), 3)
        self.assertIn(2, G.nodes())
        self.assertEqual(G.nodes[2]["label"], "group([0, 1])")
        
        # Check edges
        self.assertIn((2, 0), G.edges())
        self.assertIn((2, 1), G.edges())
        self.assertEqual(G.edges[2, 0]["label"], "componentOf")
        self.assertEqual(G.edges[2, 1]["label"], "componentOf")
    
    def test_add_vert_to_nx_with_attributes(self):
        """Test adding vertex with additional attributes"""
        # Get validation data from config
        validation_data = self.mock_config.get_validation_test_data()
        valid_vertex_data = validation_data['valid_vertices'][0]  # ("valid_word", {"pos": "NOUN", "lemma": "valid"})
        
        G = nx.DiGraph()
        # Add relation key that should be filtered out
        test_vertex = (valid_vertex_data[0], {**valid_vertex_data[1], "relation": "should_be_filtered"})
        G = DirectedMetagraph.add_vert_to_nx(G, 0, test_vertex)
        
        self.assertEqual(G.nodes[0]["label"], valid_vertex_data[0])
        # 'relation' should be filtered out, others should be included
        self.assertNotIn("relation", G.nodes[0])
        # Note: pos and lemma are not added to atomic vertices in current implementation
    
    def test_to_nx_simple_graph(self):
        """Test conversion of simple graph to NetworkX"""
        # Get expected labels from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        edge_structures = self.mock_config.get_edge_structures()
        
        G = self.simple_graph.to_nx()
        
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(len(G.nodes()), 3)
        self.assertEqual(len(G.edges()), 2)  # Two edges from directed relation
        
        # Check node labels using config data
        self.assertEqual(G.nodes[0]["label"], vertex_structures['simple_vertices'][0][0])  # "word1"
        self.assertEqual(G.nodes[1]["label"], vertex_structures['simple_vertices'][1][0])  # "word2"
        relation_name = edge_structures['simple_edges'][0][1]['relation']  # "subject"
        self.assertEqual(G.nodes[2]["label"], f"{relation_name}(0, 1)")
    
    def test_to_nx_complex_graph(self):
        """Test conversion of complex graph to NetworkX"""
        G = self.complex_graph.to_nx()
        
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(len(G.nodes()), 5)
        
        # Check that undirected relation creates correct edges
        self.assertIn((4, 1), G.edges())  # componentOf edge
        self.assertIn((4, 2), G.edges())  # componentOf edge
    
    def test_to_nx_empty_graph(self):
        """Test conversion of empty graph to NetworkX"""
        empty_graph = DirectedMetagraph()
        G = empty_graph.to_nx()
        
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(len(G.nodes()), 0)
        self.assertEqual(len(G.edges()), 0)


class TestDirectedMetagraphManipulation(TestDirectedMetagraphBase):
    """Test graph manipulation methods"""
    
    def setUp(self):
        """Set up test fixtures with manipulation-specific data"""
        super().setUp()
        
        # Get vertex and edge structures from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        edge_structures = self.mock_config.get_edge_structures()
        
        # Set up graph for manipulation testing
        self.graph = DirectedMetagraph([
            vertex_structures['simple_vertices'][0],  # ("word1", {"pos": "NOUN"})
            vertex_structures['simple_vertices'][1],  # ("word2", {"pos": "VERB"})
            vertex_structures['simple_vertices'][2],  # ("word3", {"pos": "ADJ"})
            edge_structures['simple_edges'][0],      # ((0, 1), {"relation": "subject"})
            edge_structures['simple_edges'][2]       # ((0, 2), {"relation": "modifier"})
        ])
    
    def test_add_vert_atomic(self):
        """Test adding atomic vertex"""
        # Get test data from config
        manipulation_data = self.mock_config.get_graph_manipulation_scenarios()
        new_vertex = manipulation_data['addition_scenario']['vertices_to_add'][0]  # ("C", {"new": True})
        
        initial_length = len(self.graph.metaverts)
        self.graph.add_vert("word4", {"pos": "ADV"})
        
        self.assertEqual(len(self.graph.metaverts), initial_length + 1)
        self.assertEqual(self.graph.metaverts[initial_length], ("word4", {"pos": "ADV"}))
    
    def test_add_vert_atomic_no_metadata(self):
        """Test adding atomic vertex without metadata"""
        initial_length = len(self.graph.metaverts)
        self.graph.add_vert("word4", None)
        
        self.assertEqual(len(self.graph.metaverts), initial_length + 1)
        self.assertEqual(self.graph.metaverts[initial_length], ("word4", None))
    
    def test_add_vert_directed_relation(self):
        """Test adding directed relation vertex"""
        # Get edge data from config
        edge_structures = self.mock_config.get_edge_structures()
        
        initial_length = len(self.graph.metaverts)
        self.graph.add_vert((0, 2), {"relation": "direct_relation"})
        
        self.assertEqual(len(self.graph.metaverts), initial_length + 1)
        self.assertEqual(self.graph.metaverts[initial_length], ((0, 2), {"relation": "direct_relation"}))
    
    def test_add_vert_undirected_relation(self):
        """Test adding undirected relation vertex"""
        initial_length = len(self.graph.metaverts)
        self.graph.add_vert([0, 1, 2], {"relation": "group"})
        
        self.assertEqual(len(self.graph.metaverts), initial_length + 1)
        # Should be canonicalized (sorted)
        self.assertEqual(self.graph.metaverts[initial_length], ([0, 1, 2], {"relation": "group"}))
    
    def test_add_vert_undirected_relation_unsorted(self):
        """Test adding unsorted undirected relation vertex gets canonicalized"""
        initial_length = len(self.graph.metaverts)
        self.graph.add_vert([2, 0, 1], {"relation": "group"})
        
        self.assertEqual(len(self.graph.metaverts), initial_length + 1)
        # Should be canonicalized (sorted)
        self.assertEqual(self.graph.metaverts[initial_length], ([0, 1, 2], {"relation": "group"}))
    
    def test_add_vert_invalid_forward_reference(self):
        """Test that adding vertex with forward reference fails"""
        with self.assertRaises(AssertionError):
            self.graph.add_vert((0, 10), {"relation": "invalid"})
        
        with self.assertRaises(AssertionError):
            self.graph.add_vert([0, 1, 10], {"relation": "invalid"})
    
    def test_add_vert_invalid_missing_relation(self):
        """Test that adding relation vertex without relation key fails"""
        with self.assertRaises(AssertionError):
            self.graph.add_vert((0, 1), {"type": "connection"})
        
        with self.assertRaises(AssertionError):
            self.graph.add_vert([0, 1], {"type": "group"})
    
    def test_remove_vert_atomic(self):
        """Test removing atomic vertex"""
        initial_length = len(self.graph.metaverts)
        # Remove word3 (index 2)
        self.graph.remove_vert(2)
        
        self.assertLess(len(self.graph.metaverts), initial_length)
        # The relation ((1, 2), {"relation": "modifier"}) should also be removed
        # because it references the removed vertex
    
    def test_remove_vert_with_dependencies(self):
        """Test removing vertex that has dependent relations"""
        # Remove word1 (index 0)
        # This should also remove the relation ((0, 1), {"relation": "subject"})
        initial_length = len(self.graph.metaverts)
        self.graph.remove_vert(0)
        
        self.assertLess(len(self.graph.metaverts), initial_length)
        
        # Check that no remaining relations reference the removed vertex
        for mv in self.graph.metaverts.values():
            if isinstance(mv[0], tuple) and len(mv[0]) == 2:
                self.assertNotIn(0, mv[0])
            elif isinstance(mv[0], list):
                self.assertNotIn(0, mv[0])
    
    def test_remove_vert_last_vertex(self):
        """Test removing the last vertex"""
        initial_length = len(self.graph.metaverts)
        self.graph.remove_vert(initial_length - 1)
        
        self.assertEqual(len(self.graph.metaverts), initial_length - 1)
    
    def test_remove_vert_invalid_index(self):
        """Test removing vertex with invalid index"""
        # This should not crash, but behavior might be undefined
        # The implementation should handle edge cases gracefully
        initial_length = len(self.graph.metaverts)
        try:
            self.graph.remove_vert(100)  # Invalid high index
            # If it doesn't crash, verify graph state is still valid
            self.assertLessEqual(len(self.graph.metaverts), initial_length)
        except (IndexError, ValueError):
            # This is acceptable behavior for invalid indices
            pass


class TestDirectedMetagraphRemoveVertsHelper(TestDirectedMetagraphBase):
    """Test the _remove_verts helper method"""
    
    def test_remove_verts_empty_set(self):
        """Test _remove_verts with empty removal set"""
        metaverts = {0: ("word1",), 1: ("word2",)}
        result = DirectedMetagraph._remove_verts(set(), 0, metaverts)
        self.assertEqual(result, metaverts)
    
    def test_remove_verts_empty_metaverts(self):
        """Test _remove_verts with empty metaverts list"""
        result = DirectedMetagraph._remove_verts({0}, 0, {})
        self.assertEqual(result, {})
    
    def test_remove_verts_atomic_vertex(self):
        """Test _remove_verts with atomic vertices"""
        # Get vertex data from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        vertex_data = vertex_structures['simple_vertices'][1]  # ("word2", {"pos": "VERB"})
        
        metaverts = {1: vertex_data}  # Only vertex 1, vertex 0 already removed
        result = DirectedMetagraph._remove_verts({0}, 1, metaverts)
        self.assertEqual(result, {1: vertex_data})
    
    def test_remove_verts_directed_relation_partial_removal(self):
        """Test _remove_verts with directed relation that gets partially cleaned"""
        # Get test data from config
        vertex_structures = self.mock_config.get_basic_vertex_structures()
        edge_structures = self.mock_config.get_edge_structures()
        
        vertex_data = vertex_structures['simple_vertices'][1]  # ("word2", {"pos": "VERB"})
        edge_data = edge_structures['simple_edges'][0]        # ((0, 1), {"relation": "subject"})
        
        metaverts = {
            1: vertex_data,
            2: edge_data
        }
        # Remove vertex 0, which should also remove the relation (since it references vertex 0)
        result = DirectedMetagraph._remove_verts({0}, 1, metaverts)
        # The relation should be removed because it references vertex 0
        self.assertEqual(len(result), 1)
        self.assertEqual(result[1], vertex_data)
    
    def test_remove_verts_undirected_relation_partial_removal(self):
        """Test _remove_verts with undirected relation that gets partially cleaned"""
        metaverts = {
            1: ("word2", {"pos": "VERB"}),
            2: ("word3", {"pos": "ADJ"}),
            3: ([0, 1, 2], {"relation": "group"})
        }
        # Remove vertex 0, relation should be updated to [1, 2]
        result = DirectedMetagraph._remove_verts({0}, 1, metaverts)
        
        # Should have word2, word3, and updated group relation
        expected_group = ([1, 2], {"relation": "group"})
        self.assertEqual(result[3], expected_group)
    
    def test_remove_verts_undirected_relation_complete_removal(self):
        """Test _remove_verts with undirected relation that gets completely removed"""
        metaverts = {
            2: ([0, 1], {"relation": "group"})
        }
        # Remove both vertices, relation should be completely removed
        result = DirectedMetagraph._remove_verts({0, 1}, 2, metaverts)
        
        # Should be empty since the group relation depends on removed vertices
        self.assertEqual(len(result), 0)


class TestDirectedMetagraphEdgeCases(TestDirectedMetagraphBase):
    """Test edge cases and error conditions"""
    
    def test_initialization_with_invalid_vertex_list(self):
        """Test initialization fails with invalid vertex list"""
        with self.assertRaises(AssertionError):
            DirectedMetagraph([("word", {"pos": "NOUN"}), ((0, 2), {"relation": "invalid"})])
    
    def test_empty_graph_operations(self):
        """Test operations on empty graph"""
        dg = DirectedMetagraph()
        
        # to_nx should work on empty graph
        G = dg.to_nx()
        self.assertEqual(len(G.nodes()), 0)
        
        # Adding vertex to empty graph should work
        dg.add_vert("first_word", {"pos": "NOUN"})
        self.assertEqual(len(dg.metaverts), 1)
    
    def test_single_vertex_graph(self):
        """Test operations on single vertex graph"""
        dg = DirectedMetagraph([("single_word", {"pos": "NOUN"})])
        
        G = dg.to_nx()
        self.assertEqual(len(G.nodes()), 1)
        self.assertEqual(G.nodes[0]["label"], "single_word")
        
        # Remove the only vertex
        dg.remove_vert(0)
        self.assertEqual(len(dg.metaverts), 0)
    
    def test_complex_removal_scenario(self):
        """Test complex vertex removal scenario"""
        # Get complex removal scenario from config
        removal_scenarios = self.mock_config.get_complex_removal_scenarios()
        scenario = removal_scenarios['scenario_1']
        
        # Build vertex list from scenario
        vert_list = scenario['initial_vertices'] + scenario['initial_edges']
        dg = DirectedMetagraph(vert_list)
        
        initial_length = len(dg.metaverts)
        
        # Remove vertex using scenario data
        vertex_to_remove = scenario['vertex_to_remove']
        dg.remove_vert(vertex_to_remove)
        
        self.assertLess(len(dg.metaverts), initial_length)
        
        # Verify no relations reference the removed vertex
        for mv in dg.metaverts.values():
            if isinstance(mv[0], tuple) and len(mv[0]) == 2:
                self.assertNotIn(vertex_to_remove, mv[0])
            elif isinstance(mv[0], list):
                self.assertNotIn(vertex_to_remove, mv[0])


class TestDirectedMetagraphIntegration(TestDirectedMetagraphBase):
    """Integration tests for DirectedMetagraph"""
    
    def test_mock_factory_integration(self):
        """Test that mock factory creates proper mock instances"""
        # Test mock factory functionality
        mock_graph = self.mock_factory('MockDirectedMetagraph')
        self.assertIsNotNone(mock_graph)
        self.assertTrue(hasattr(mock_graph, 'add_vertex'))
        self.assertTrue(hasattr(mock_graph, 'to_networkx'))
        
        # Test validation mock
        validation_mock = self.mock_factory('MockDirectedMetagraphValidation')
        self.assertIsNotNone(validation_mock)
        self.assertTrue(hasattr(validation_mock, 'execute'))
        
        # Test available mocks listing
        available_mocks = self.mock_factory.get_available_mocks()
        self.assertIn('MockDirectedMetagraph', available_mocks)
        self.assertIn('MockDirectedMetagraphValidation', available_mocks)
        
        # Test invalid mock name raises error
        with self.assertRaises(ValueError):
            self.mock_factory('NonExistentMock')
    
    def test_build_and_manipulate_graph(self):
        """Test building and manipulating a graph step by step"""
        # Get traversal scenario from config for structured testing
        traversal_data = self.mock_config.get_graph_traversal_scenarios()
        linear_path = traversal_data['linear_path']
        
        # Start with empty graph
        dg = DirectedMetagraph()
        
        # Get linguistic test structure from config
        linguistic_structures = self.mock_config.get_linguistic_test_structures()
        simple_sentence = linguistic_structures['simple_sentence']
        
        # Add atomic vertices using config data
        for vertex in simple_sentence['vertices']:
            dg.add_vert(vertex[0], vertex[1])
        
        # Add relations
        for edge in simple_sentence['edges']:
            dg.add_vert(edge[0], edge[1])
        
        # Add groups  
        for group in simple_sentence['groups']:
            dg.add_vert(group[0], group[1])
        
        self.assertEqual(len(dg.metaverts), 8)
        
        # Convert to NetworkX
        G = dg.to_nx()
        self.assertGreater(len(G.nodes()), 0)
        self.assertGreater(len(G.edges()), 0)
        
        # Remove a vertex and check cascade
        dg.remove_vert(2)  # Remove "runs"
        self.assertLess(len(dg.metaverts), 8)
        
        # Graph should still be valid
        G2 = dg.to_nx()
        self.assertIsInstance(G2, nx.DiGraph)
    
    def test_canonicalization_in_practice(self):
        """Test that canonicalization works in practice"""
        # Get complex graph structure from config for canonical testing
        complex_structures = self.mock_config.get_complex_graph_structures()
        hierarchical = complex_structures['hierarchical_structure']
        
        # Get canonicalization test scenario from config
        canonicalization_scenarios = self.mock_config.get_canonicalization_test_scenarios()
        test_case = canonicalization_scenarios['unsorted_groups']
        
        # Create graph with unsorted undirected relations
        vert_list = [
            ("A", {}),
            ("B", {}), 
            ("C", {}),
            test_case['input_group']
        ]
        
        dg = DirectedMetagraph(vert_list)
        
        # Check that it was canonicalized
        last_vert = dg.metaverts[3]  # Last vertex has index 3
        expected_canonical = test_case['expected_canonical']
        self.assertEqual(last_vert[0], expected_canonical[0])  # Should be sorted
    
    def test_validation_prevents_invalid_graphs(self):
        """Test that validation prevents creation of invalid graphs"""
        # Get validation test data from config
        validation_data = self.mock_config.get_validation_test_data()
        
        # Forward reference should fail
        with self.assertRaises(AssertionError):
            DirectedMetagraph([
                ("A", {}),
                ((0, 2), {"relation": "invalid"})  # References non-existent vertex 2
            ])
        
        # Missing relation key should fail
        with self.assertRaises(AssertionError):
            DirectedMetagraph([
                ("A", {}),
                ("B", {}),
                ((0, 1), {"type": "connection"})  # Missing 'relation' key
            ])


if __name__ == '__main__':
    unittest.main()