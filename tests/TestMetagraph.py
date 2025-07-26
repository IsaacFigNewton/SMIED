import unittest
import spacy
import json

from noske.Metagraph import Metagraph


class TestMetagraph(unittest.TestCase):
    """Test cases for the base Metagraph class"""
    
    def setUp(self):
        self.graph = Metagraph()

    def test_init_empty(self):
        """Test initialization of empty Metagraph"""
        graph = Metagraph()
        self.assertIsNotNone(graph.G)
        # Check that the hypergraph is empty initially
        self.assertEqual(len(graph.get_nodes()), 0)
        self.assertEqual(len(graph.get_edges()), 0)

    def test_init_with_json(self):
        """Test initialization with JSON data"""
        # First create a graph with some data
        test_graph = Metagraph()
        test_graph.add_node(1, metadata={"text": "test"})
        test_graph.add_node(2, metadata={"text": "test2"}) 
        test_graph.add_edge((1, 2), metadata={"type": "test_edge"})
        
        # Serialize to JSON
        json_data = test_graph.to_json()
        
        # Create new graph from JSON
        new_graph = Metagraph(json_data=json_data)
        
        # Verify structure is preserved
        self.assertEqual(len(new_graph.get_nodes()), len(test_graph.get_nodes()))
        self.assertEqual(len(new_graph.get_edges()), len(test_graph.get_edges()))

    def test_add_single_node(self):
        """Test adding a single node"""
        self.graph.add_node(1, metadata={"text": "test"})
        nodes = self.graph.get_nodes()
        
        self.assertIn(1, nodes)
        self.assertEqual(nodes[1]["text"], "test")
        self.assertEqual(nodes[1]["id"], 1)
        self.assertEqual(nodes[1]["type"], "regular")

    def test_add_single_node_with_type(self):
        """Test adding a single node with specified type"""
        self.graph.add_node(1, metadata={"text": "test"}, node_type="meta")
        nodes = self.graph.get_nodes()
        
        self.assertIn(1, nodes)
        self.assertEqual(nodes[1]["type"], "meta")

    def test_add_single_node_no_metadata(self):
        """Test adding a single node with no metadata"""
        self.graph.add_node(1)
        nodes = self.graph.get_nodes()
        
        self.assertIn(1, nodes)
        self.assertEqual(nodes[1]["id"], 1)
        self.assertEqual(nodes[1]["type"], "regular")

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes"""
        self.graph.add_nodes(
            nodes=[1, 2, 3],
            metadata=[{"text": f"node{i}"} for i in [1, 2, 3]]
        )
        graph_nodes = self.graph.get_nodes()
        
        self.assertEqual(len(graph_nodes), 3)
        for i, node in enumerate([1, 2, 3]):
            self.assertEqual(graph_nodes[node]["text"], f"node{i+1}")
            self.assertEqual(graph_nodes[node]["id"], node)
            self.assertEqual(graph_nodes[node]["type"], "regular")

    def test_add_multiple_nodes_no_metadata(self):
        """Test adding multiple nodes with no metadata"""
        self.graph.add_nodes([1, 2, 3])
        graph_nodes = self.graph.get_nodes()
        
        self.assertEqual(len(graph_nodes), 3)
        for node in [1, 2, 3]:
            self.assertEqual(graph_nodes[node]["id"], node)
            self.assertEqual(graph_nodes[node]["type"], "regular")

    def test_add_regular_edge(self):
        """Test adding a regular (pairwise) edge"""
        # First add nodes
        self.graph.add_nodes([1, 2])
        
        # Add edge
        self.graph.add_edge((1, 2), metadata={"type": "test_edge"})
        
        edges = self.graph.get_edges()
        self.assertGreater(len(edges), 0)
        
        # Find the edge we added
        edge_found = False
        for edge_key, edge_data in edges.items():
            if edge_data.get("id") == (1, 2) and edge_data.get("type") == "regular":
                self.assertEqual(edge_data["type"], "regular")
                edge_found = True
                break
        self.assertTrue(edge_found, "Regular edge not found in graph")

    def test_add_regular_edge_no_metadata(self):
        """Test adding a regular edge with no metadata"""
        self.graph.add_nodes([1, 2])
        self.graph.add_edge((1, 2))
        
        edges = self.graph.get_edges()
        self.assertGreater(len(edges), 0)
        
        # Find the edge we added
        edge_found = False
        for edge_key, edge_data in edges.items():
            if edge_data.get("id") == (1, 2) and edge_data.get("type") == "regular":
                edge_found = True
                break
        self.assertTrue(edge_found, "Regular edge not found in graph")

    def test_add_hyperedge(self):
        """Test adding a hyperedge (more than 2 nodes)"""
        # First add nodes
        self.graph.add_nodes([1, 2, 3])
        self.graph.add_edge((1, 2, 3))
        
        # Should have created the hyperedge and a metavertex
        edges = self.graph.get_edges()
        self.assertGreater(len(edges), 0)
        print(json.dumps(self.graph.get_nodes(), indent=4))
        metaverts = self.graph.get_all_metaverts()
        self.assertGreater(len(metaverts), 0)

    def test_add_multiple_edges(self):
        """Test adding multiple edges"""
        # First add nodes
        self.graph.add_nodes([1, 2, 3])
        
        self.graph.add_edges(
            edges=[(1, 2), (2, 3)],
            metadata=[{"type": "edge1"}, {"type": "edge2"}]
        )
        
        graph_edges = self.graph.get_edges()
        self.assertGreaterEqual(len(graph_edges), 2)

    def test_add_multiple_edges_no_metadata(self):
        """Test adding multiple edges with no metadata"""
        self.graph.add_nodes([1, 2, 3])
        self.graph.add_edges([(1, 2), (2, 3)])
        
        graph_edges = self.graph.get_edges()
        self.assertGreaterEqual(len(graph_edges), 2)

    def test_add_edges_metadata_mismatch(self):
        """Test adding edges with mismatched metadata length"""
        with self.assertRaises(Exception):
            self.graph.add_edges(
                edges=[(1, 2), (2, 3)],
                metadata=[{"type": "edge1"}]  # Only one metadata for two edges
            )

    def test_get_nodes(self):
        """Test getting nodes from the hypergraph"""
        self.graph.add_node(1, metadata={"text": "node1"})
        self.graph.add_node(2, metadata={"text": "node2"})
        
        nodes = self.graph.get_nodes()
        
        self.assertEqual(len(nodes), 2)
        self.assertIn(1, nodes)
        self.assertIn(2, nodes)
        self.assertEqual(nodes[1]["text"], "node1")
        self.assertEqual(nodes[2]["text"], "node2")

    def test_get_edges(self):
        """Test getting edges from the hypergraph"""
        self.graph.add_nodes([1, 2, 3])
        self.graph.add_edge((1, 2), metadata={"type": "edge1"})
        self.graph.add_edge((2, 3), metadata={"type": "edge2"})
        
        edges = self.graph.get_edges()
        
        self.assertGreaterEqual(len(edges), 2)

    def test_get_all_metaverts(self):
        """Test getting all metavertices"""
        # Add regular nodes
        self.graph.add_nodes([1, 2, 3, 4])
        
        # Add a regular edge (should not create metavertex)
        self.graph.add_edge((1, 2), metadata={"type": "regular"})
        
        # Add a hyperedge (should create metavertex)
        self.graph.add_edge((1, 2, 3), metadata={"type": "hyper"})
        
        metaverts = self.graph.get_all_metaverts()
        
        # Should have at least one metavertex from the hyperedge
        self.assertGreater(len(metaverts), 0)
        
        # All returned nodes should be meta type
        for node_id, node_data in metaverts.items():
            self.assertEqual(node_data["type"], "meta")

    def test_get_node_with_id(self):
        """Test finding a node by its ID"""
        self.graph.add_node(42, metadata={"text": "answer"})
        
        result = self.graph.get_node_with_id(42)
        self.assertIsNotNone(result)
        node_key, node_data = result
        self.assertEqual(node_data["id"], 42)
        self.assertEqual(node_data["text"], "answer")
        
        # Test non-existent node
        result = self.graph.get_node_with_id(999)
        self.assertIsNone(result)

    def test_get_edge_with_id(self):
        """Test finding an edge by its ID"""
        self.graph.add_nodes([1, 2])
        self.graph.add_edge((1, 2), metadata={"type": "test"})
        
        result = self.graph.get_edge_with_id((1, 2))
        self.assertIsNotNone(result)
        edge_key, edge_data = result
        self.assertEqual(edge_data["id"], (1, 2))
        
        # Test non-existent edge
        result = self.graph.get_edge_with_id((999, 888))
        self.assertIsNone(result)

    def test_get_metavert_metadata(self):
        """Test getting metadata for a metavertex"""
        self.graph.add_nodes([1, 2, 3])
        self.graph.add_edge((1, 2, 3), metadata={"type": "test_hyper"})
        
        metadata = self.graph.get_metavert_metadata((1, 2, 3))
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["type"], "meta")

    def test_json_serialization_roundtrip(self):
        """Test complete JSON serialization and deserialization"""
        # Create a graph with various node and edge types
        self.graph.add_node(1, metadata={"text": "node1"})
        self.graph.add_node(2, metadata={"text": "node2"})
        self.graph.add_node(3, metadata={"text": "node3"})
        self.graph.add_edge((1, 2), metadata={"type": "regular_edge"})
        self.graph.add_edge((1, 2, 3), metadata={"type": "hyper_edge"})
        
        # Serialize
        json_data = self.graph.to_json()
        self.assertIn("nodes", json_data)
        self.assertIn("edges", json_data)
        
        # Deserialize using base class
        new_graph = Metagraph(json_data=json_data)
        
        # Compare structure
        original_nodes = self.graph.get_nodes()
        new_nodes = new_graph.get_nodes()
        original_edges = self.graph.get_edges()
        new_edges = new_graph.get_edges()
        
        self.assertEqual(len(original_nodes), len(new_nodes))
        self.assertEqual(len(original_edges), len(new_edges))

    def test_metavertex_creation_and_linking(self):
        """Test that metavertices are properly created and linked"""
        self.graph.add_nodes([1, 2, 3])
        
        # Add hyperedge which should create metavertex
        self.graph.add_edge((1, 2, 3), metadata={"type": "test_hyper"})
        
        # Check metavertex was created
        metaverts = self.graph.get_all_metaverts()
        self.assertGreater(len(metaverts), 0)
        
        # Check that metavertex has proper connections
        edges = self.graph.get_edges()
        
        # Should have metavert-to-hyperedge and hyperedge-to-metavert connections
        connection_types = set()
        for edge_data in edges.values():
            if edge_data.get("type") in ["metavert_to_hye", "hye_to_metavert"]:
                connection_types.add(edge_data["type"])
        
        # Should have both types of connections
        self.assertIn("metavert_to_hye", connection_types)
        self.assertIn("hye_to_metavert", connection_types)

    def test_invalid_metaedge_raises_error(self):
        """Test that invalid metaedge construction raises error"""
        with self.assertRaises(ValueError):
            # Try to add a metaedge with nested structure and length != 2
            self.graph.add_edge(((1, 2), 3, 4), metadata={"type": "invalid"})

    def test_directed_hyperedge(self):
        """Test adding a directed hyperedge (2 elements where one is nested)"""
        self.graph.add_nodes([1, 2, 3])
        
        # Add directed hyperedge
        self.graph.add_edge(((1, 2), 3), metadata={"type": "hyper"})
        
        edges = self.graph.get_edges()
        metaverts = self.graph.get_all_metaverts()
        
        # Should have created edges and metavertex
        self.assertGreater(len(edges), 0)
        self.assertGreater(len(metaverts), 0)