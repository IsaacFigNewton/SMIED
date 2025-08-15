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
        self.assertIsNotNone(graph.G, "Graph object G should not be None after initialization")
        # Check that the hypergraph is empty initially
        nodes = graph.get_nodes()
        edges = graph.get_edges()
        self.assertEqual(len(nodes), 0, 
                        f"Expected empty graph to have 0 nodes, but found {len(nodes)} nodes: {list(nodes.keys())}")
        self.assertEqual(len(edges), 0, 
                        f"Expected empty graph to have 0 edges, but found {len(edges)} edges: {list(edges.keys())}")

    def test_init_with_json(self):
        """Test initialization with JSON data"""
        # First create a graph with some data
        test_graph = Metagraph()
        test_graph.add_node(1, metadata={"text": "test"})
        test_graph.add_node(2, metadata={"text": "test2"}) 
        test_graph.add_edge((1, 2), metadata={"type": "test_edge"})
        
        original_nodes = test_graph.get_nodes()
        original_edges = test_graph.get_edges()
        
        # Serialize to JSON
        json_data_str = test_graph.to_json()
        
        # Create new graph from JSON
        new_graph = Metagraph(json_data=json_data_str)
        new_nodes = new_graph.get_nodes()
        new_edges = new_graph.get_edges()
        
        # Verify structure is preserved
        self.assertEqual(len(new_nodes), len(original_nodes),
                        f"JSON initialization failed: expected {len(original_nodes)} nodes but got {len(new_nodes)}. "
                        f"Original nodes: {list(original_nodes.keys())}, New nodes: {list(new_nodes.keys())}")
        self.assertEqual(len(new_edges), len(original_edges),
                        f"JSON initialization failed: expected {len(original_edges)} edges but got {len(new_edges)}. "
                        f"Original edges: {list(original_edges.keys())}, New edges: {list(new_edges.keys())}")

    def test_add_single_node(self):
        """Test adding a single node"""
        self.graph.add_node(1, metadata={"text": "test"})
        nodes = self.graph.get_nodes()
        
        self.assertIn(1, nodes, 
                     f"Node 1 not found in graph after adding. Available nodes: {list(nodes.keys())}")
        self.assertEqual(nodes[1]["text"], "test",
                        f"Node 1 text field mismatch: expected 'test' but got '{nodes[1].get('text', 'MISSING')}'. "
                        f"Full node data: {nodes[1]}")
        self.assertEqual(nodes[1]["id"], 1,
                        f"Node 1 id field mismatch: expected 1 but got {nodes[1].get('id', 'MISSING')}. "
                        f"Full node data: {nodes[1]}")
        self.assertEqual(nodes[1]["type"], "regular",
                        f"Node 1 type field mismatch: expected 'regular' but got '{nodes[1].get('type', 'MISSING')}'. "
                        f"Full node data: {nodes[1]}")

    def test_add_single_node_with_type(self):
        """Test adding a single node with specified type"""
        self.graph.add_node(1, metadata={"text": "test"}, node_type="meta")
        nodes = self.graph.get_nodes()
        
        self.assertIn(1, nodes, 
                     f"Node 1 not found in graph after adding with type 'meta'. Available nodes: {list(nodes.keys())}")
        self.assertEqual(nodes[1]["type"], "meta",
                        f"Node 1 type mismatch: expected 'meta' but got '{nodes[1].get('type', 'MISSING')}'. "
                        f"Full node data: {nodes[1]}")

    def test_add_single_node_no_metadata(self):
        """Test adding a single node with no metadata"""
        self.graph.add_node(1)
        nodes = self.graph.get_nodes()
        
        self.assertIn(1, nodes, 
                     f"Node 1 not found in graph after adding without metadata. Available nodes: {list(nodes.keys())}")
        self.assertEqual(nodes[1]["id"], 1,
                        f"Node 1 id field mismatch: expected 1 but got {nodes[1].get('id', 'MISSING')}. "
                        f"Full node data: {nodes[1]}")
        self.assertEqual(nodes[1]["type"], "regular",
                        f"Node 1 type field mismatch: expected 'regular' but got '{nodes[1].get('type', 'MISSING')}'. "
                        f"Full node data: {nodes[1]}")

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes"""
        node_ids = [1, 2, 3]
        metadata_list = [{"text": f"node{i}"} for i in node_ids]
        
        self.graph.add_nodes(
            nodes=node_ids,
            metadata=metadata_list
        )
        graph_nodes = self.graph.get_nodes()
        
        self.assertEqual(len(graph_nodes), 3,
                        f"Expected 3 nodes after adding nodes {node_ids}, but found {len(graph_nodes)} nodes. "
                        f"Available nodes: {list(graph_nodes.keys())}")
        
        for i, node in enumerate(node_ids):
            expected_text = f"node{node}"
            self.assertIn(node, graph_nodes,
                         f"Node {node} missing from graph after batch add. Available nodes: {list(graph_nodes.keys())}")
            self.assertEqual(graph_nodes[node]["text"], expected_text,
                           f"Node {node} text mismatch: expected '{expected_text}' but got '{graph_nodes[node].get('text', 'MISSING')}'. "
                           f"Full node data: {graph_nodes[node]}")
            self.assertEqual(graph_nodes[node]["id"], node,
                           f"Node {node} id mismatch: expected {node} but got {graph_nodes[node].get('id', 'MISSING')}. "
                           f"Full node data: {graph_nodes[node]}")
            self.assertEqual(graph_nodes[node]["type"], "regular",
                           f"Node {node} type mismatch: expected 'regular' but got '{graph_nodes[node].get('type', 'MISSING')}'. "
                           f"Full node data: {graph_nodes[node]}")

    def test_add_multiple_nodes_no_metadata(self):
        """Test adding multiple nodes with no metadata"""
        node_ids = [1, 2, 3]
        self.graph.add_nodes(node_ids)
        graph_nodes = self.graph.get_nodes()
        
        self.assertEqual(len(graph_nodes), 3,
                        f"Expected 3 nodes after adding nodes {node_ids} without metadata, but found {len(graph_nodes)} nodes. "
                        f"Available nodes: {list(graph_nodes.keys())}")
        
        for node in node_ids:
            self.assertIn(node, graph_nodes,
                         f"Node {node} missing from graph after batch add without metadata. Available nodes: {list(graph_nodes.keys())}")
            self.assertEqual(graph_nodes[node]["id"], node,
                           f"Node {node} id mismatch: expected {node} but got {graph_nodes[node].get('id', 'MISSING')}. "
                           f"Full node data: {graph_nodes[node]}")
            self.assertEqual(graph_nodes[node]["type"], "regular",
                           f"Node {node} type mismatch: expected 'regular' but got '{graph_nodes[node].get('type', 'MISSING')}'. "
                           f"Full node data: {graph_nodes[node]}")

    def test_add_regular_edge(self):
        """Test adding a regular (pairwise) edge"""
        # First add nodes
        self.graph.add_nodes([1, 2])
        
        # Add edge
        self.graph.add_edge((1, 2), metadata={"type": "test_edge"})
        
        edges = self.graph.get_edges()
        self.assertGreater(len(edges), 0,
                          f"Expected at least 1 edge after adding (1, 2), but found {len(edges)} edges. "
                          f"Available edges: {list(edges.keys())}")
        
        # Find the edge we added
        edge_found = False
        matching_edges = []
        for edge_key, edge_data in edges.items():
            if edge_data.get("id") == (1, 2):
                matching_edges.append((edge_key, edge_data))
                if edge_data.get("type") == "regular":
                    self.assertEqual(edge_data["type"], "regular",
                                   f"Edge (1, 2) type mismatch: expected 'regular' but got '{edge_data.get('type', 'MISSING')}'. "
                                   f"Full edge data: {edge_data}")
                    edge_found = True
                    break
        
        self.assertTrue(edge_found, 
                       f"Regular edge (1, 2) not found in graph. "
                       f"Matching edges by ID: {matching_edges}, "
                       f"All edges: {[(k, v.get('id'), v.get('type')) for k, v in edges.items()]}")

    def test_add_regular_edge_no_metadata(self):
        """Test adding a regular edge with no metadata"""
        self.graph.add_nodes([1, 2])
        self.graph.add_edge((1, 2))
        
        edges = self.graph.get_edges()
        self.assertGreater(len(edges), 0,
                          f"Expected at least 1 edge after adding (1, 2) without metadata, but found {len(edges)} edges. "
                          f"Available edges: {list(edges.keys())}")
        
        # Find the edge we added
        edge_found = False
        edge_details = []
        for edge_key, edge_data in edges.items():
            edge_details.append(f"{edge_key}: id={edge_data.get('id')}, type={edge_data.get('type')}")
            if edge_data.get("id") == (1, 2) and edge_data.get("type") == "regular":
                edge_found = True
                break
        
        self.assertTrue(edge_found, 
                       f"Regular edge (1, 2) not found in graph. Edge details: {edge_details}")

    def test_add_hyperedge(self):
        """Test adding a hyperedge (more than 2 nodes)"""
        # First add nodes
        node_ids = [1, 2, 3]
        self.graph.add_nodes(node_ids)
        self.graph.add_edge((1, 2, 3))
        
        # Should have created the hyperedge and a metavertex
        edges = self.graph.get_edges()
        metaverts = self.graph.get_all_metaverts()
        
        self.assertGreater(len(edges), 0,
                          f"Expected edges to be created after adding hyperedge (1, 2, 3), but found {len(edges)} edges. "
                          f"Available edges: {list(edges.keys())}")
        
        print(json.dumps(self.graph.get_nodes(), indent=4))
        
        self.assertGreater(len(metaverts), 0,
                          f"Expected metavertices to be created after adding hyperedge (1, 2, 3), but found {len(metaverts)} metaverts. "
                          f"Available metaverts: {list(metaverts.keys())}, "
                          f"All nodes: {[(k, v.get('type')) for k, v in self.graph.get_nodes().items()]}")

    def test_add_multiple_edges(self):
        """Test adding multiple edges"""
        # First add nodes
        self.graph.add_nodes([1, 2, 3])
        
        edge_list = [(1, 2), (2, 3)]
        metadata_list = [{"type": "edge1"}, {"type": "edge2"}]
        
        self.graph.add_edges(
            edges=edge_list,
            metadata=metadata_list
        )
        
        graph_edges = self.graph.get_edges()
        self.assertGreaterEqual(len(graph_edges), 2,
                               f"Expected at least 2 edges after adding {edge_list}, but found {len(graph_edges)} edges. "
                               f"Available edges: {list(graph_edges.keys())}")

    def test_add_multiple_edges_no_metadata(self):
        """Test adding multiple edges with no metadata"""
        self.graph.add_nodes([1, 2, 3])
        edge_list = [(1, 2), (2, 3)]
        self.graph.add_edges(edge_list)
        
        graph_edges = self.graph.get_edges()
        self.assertGreaterEqual(len(graph_edges), 2,
                               f"Expected at least 2 edges after adding {edge_list} without metadata, but found {len(graph_edges)} edges. "
                               f"Available edges: {list(graph_edges.keys())}")

    def test_add_edges_metadata_mismatch(self):
        """Test adding edges with mismatched metadata length"""
        edge_list = [(1, 2), (2, 3)]
        metadata_list = [{"type": "edge1"}]  # Only one metadata for two edges
        
        with self.assertRaises(Exception) as context:
            self.graph.add_edges(
                edges=edge_list,
                metadata=metadata_list
            )
        
        # Add more context to the assertion error if it doesn't raise as expected
        if not hasattr(context, 'exception'):
            self.fail(f"Expected Exception to be raised when adding {len(edge_list)} edges with {len(metadata_list)} metadata entries, "
                     f"but no exception was raised")

    def test_get_nodes(self):
        """Test getting nodes from the hypergraph"""
        self.graph.add_node(1, metadata={"text": "node1"})
        self.graph.add_node(2, metadata={"text": "node2"})
        
        nodes = self.graph.get_nodes()
        
        self.assertEqual(len(nodes), 2,
                        f"Expected 2 nodes after adding nodes 1 and 2, but found {len(nodes)} nodes. "
                        f"Available nodes: {list(nodes.keys())}")
        self.assertIn(1, nodes,
                     f"Node 1 not found in returned nodes. Available nodes: {list(nodes.keys())}")
        self.assertIn(2, nodes,
                     f"Node 2 not found in returned nodes. Available nodes: {list(nodes.keys())}")
        self.assertEqual(nodes[1]["text"], "node1",
                        f"Node 1 text mismatch: expected 'node1' but got '{nodes[1].get('text', 'MISSING')}'. "
                        f"Full node data: {nodes[1]}")
        self.assertEqual(nodes[2]["text"], "node2",
                        f"Node 2 text mismatch: expected 'node2' but got '{nodes[2].get('text', 'MISSING')}'. "
                        f"Full node data: {nodes[2]}")

    def test_get_edges(self):
        """Test getting edges from the hypergraph"""
        self.graph.add_nodes([1, 2, 3])
        self.graph.add_edge((1, 2), metadata={"type": "edge1"})
        self.graph.add_edge((2, 3), metadata={"type": "edge2"})
        
        edges = self.graph.get_edges()
        
        self.assertGreaterEqual(len(edges), 2,
                               f"Expected at least 2 edges after adding (1, 2) and (2, 3), but found {len(edges)} edges. "
                               f"Available edges: {list(edges.keys())}")

    def test_get_all_metaverts(self):
        """Test getting all metavertices"""
        # Add regular nodes
        self.graph.add_nodes([1, 2, 3, 4])
        
        # Add a regular edge (should not create metavertex)
        self.graph.add_edge((1, 2), metadata={"type": "regular"})
        
        # Add a hyperedge (should create metavertex)
        self.graph.add_edge((1, 2, 3), metadata={"type": "hyper"})
        
        metaverts = self.graph.get_all_metaverts()
        all_nodes = self.graph.get_nodes()
        
        # Should have at least one metavertex from the hyperedge
        self.assertGreater(len(metaverts), 0,
                          f"Expected at least 1 metavertex after adding hyperedge (1, 2, 3), but found {len(metaverts)} metaverts. "
                          f"All nodes by type: {[(k, v.get('type')) for k, v in all_nodes.items()]}")
        
        # All returned nodes should be meta type
        non_meta_nodes = []
        for node_id, node_data in metaverts.items():
            if node_data.get("type") != "meta":
                non_meta_nodes.append((node_id, node_data.get("type")))
        
        self.assertEqual(len(non_meta_nodes), 0,
                        f"get_all_metaverts() returned non-meta nodes: {non_meta_nodes}. "
                        f"All metaverts: {[(k, v.get('type')) for k, v in metaverts.items()]}")

    def test_get_node_with_id(self):
        """Test finding a node by its ID"""
        self.graph.add_node(42, metadata={"text": "answer"})
        
        result = self.graph.get_node_with_id(42)
        self.assertIsNotNone(result,
                           f"get_node_with_id(42) returned None. Available nodes: {list(self.graph.get_nodes().keys())}")
        
        node_key, node_data = result
        self.assertEqual(node_data["id"], 42,
                        f"Retrieved node id mismatch: expected 42 but got {node_data.get('id', 'MISSING')}. "
                        f"Full node data: {node_data}")
        self.assertEqual(node_data["text"], "answer",
                        f"Retrieved node text mismatch: expected 'answer' but got '{node_data.get('text', 'MISSING')}'. "
                        f"Full node data: {node_data}")
        
        # Test non-existent node
        result = self.graph.get_node_with_id(999)
        self.assertIsNone(result,
                         f"get_node_with_id(999) should return None for non-existent node, but got {result}. "
                         f"Available nodes: {list(self.graph.get_nodes().keys())}")

    def test_get_edge_with_id(self):
        """Test finding an edge by its ID"""
        self.graph.add_nodes([1, 2])
        self.graph.add_edge((1, 2), metadata={"type": "test"})
        
        all_edges = self.graph.get_edges()
        result = self.graph.get_edge_with_id((1, 2))
        
        self.assertIsNotNone(result, 
                           f"get_edge_with_id((1, 2)) returned None. "
                           f"Available edges: {[(k, v.get('id'), v.get('type')) for k, v in all_edges.items()]}")
        
        edge_key, edge_data = result
        self.assertEqual(edge_data["id"], (1, 2),
                        f"Retrieved edge id mismatch: expected (1, 2) but got {edge_data.get('id', 'MISSING')}. "
                        f"Full edge data: {edge_data}")
        
        # Test non-existent edge
        result = self.graph.get_edge_with_id((999, 888))
        self.assertIsNone(result,
                         f"get_edge_with_id((999, 888)) should return None for non-existent edge, but got {result}. "
                         f"Available edges: {[(k, v.get('id')) for k, v in all_edges.items()]}")

    def test_get_metavert_metadata(self):
        """Test getting metadata for a metavertex"""
        self.graph.add_nodes([1, 2, 3])
        self.graph.add_edge((1, 2, 3), metadata={"type": "test_hyper"})
        
        metaverts = self.graph.get_all_metaverts()
        metadata = self.graph.get_metavert_metadata((1, 2, 3))
        
        self.assertIsNotNone(metadata,
                           f"get_metavert_metadata((1, 2, 3)) returned None after adding hyperedge. "
                           f"Available metaverts: {list(metaverts.keys())}")
        self.assertEqual(metadata["type"], "meta",
                        f"Metavertex metadata type mismatch: expected 'meta' but got '{metadata.get('type', 'MISSING')}'. "
                        f"Full metadata: {metadata}")

    def test_json_serialization_roundtrip(self):
        """Test complete JSON serialization and deserialization"""
        # Create a graph with various node and edge types
        self.graph.add_node(1, metadata={"text": "node1"})
        self.graph.add_node(2, metadata={"text": "node2"})
        self.graph.add_node(3, metadata={"text": "node3"})
        self.graph.add_edge((1, 2), metadata={"type": "regular_edge"})
        self.graph.add_edge((1, 2, 3), metadata={"type": "hyper_edge"})
        
        # Get original data for comparison
        original_nodes = self.graph.get_nodes()
        original_edges = self.graph.get_edges()
        
        # Serialize
        json_data_str = self.graph.to_json()
        
        try:
            json_data_dict = json.loads(json_data_str)
        except json.JSONDecodeError as e:
            self.fail(f"JSON serialization produced invalid JSON: {e}. JSON string: {json_data_str[:200]}...")
        
        self.assertIn("nodes", json_data_dict,
                     f"JSON serialization missing 'nodes' key. Available keys: {list(json_data_dict.keys())}")
        self.assertIn("edges", json_data_dict,
                     f"JSON serialization missing 'edges' key. Available keys: {list(json_data_dict.keys())}")
        
        # Deserialize using base class
        try:
            new_graph = Metagraph(json_data=json_data_str)
        except Exception as e:
            self.fail(f"JSON deserialization failed: {e}. JSON data: {json_data_str[:200]}...")
        
        # Compare structure
        new_nodes = new_graph.get_nodes()
        new_edges = new_graph.get_edges()
        
        self.assertEqual(len(original_nodes), len(new_nodes),
                        f"JSON roundtrip node count mismatch: original had {len(original_nodes)} nodes, "
                        f"deserialized has {len(new_nodes)} nodes. "
                        f"Original: {list(original_nodes.keys())}, New: {list(new_nodes.keys())}")
        self.assertEqual(len(original_edges), len(new_edges),
                        f"JSON roundtrip edge count mismatch: original had {len(original_edges)} edges, "
                        f"deserialized has {len(new_edges)} edges. "
                        f"Original: {list(original_edges.keys())}, New: {list(new_edges.keys())}")

    def test_metavertex_creation_and_linking(self):
        """Test that metavertices are properly created and linked"""
        self.graph.add_nodes([1, 2, 3])
        
        # Add hyperedge which should create metavertex
        self.graph.add_edge((1, 2, 3), metadata={"type": "test_hyper"})
        
        # Check metavertex was created
        metaverts = self.graph.get_all_metaverts()
        all_nodes = self.graph.get_nodes()
        
        self.assertGreater(len(metaverts), 0,
                          f"Expected metavertex to be created for hyperedge (1, 2, 3), but found {len(metaverts)} metaverts. "
                          f"All nodes by type: {[(k, v.get('type')) for k, v in all_nodes.items()]}")
        
        # Check that metavertex has proper connections
        edges = self.graph.get_edges()
        
        # Should have metavert-to-hyperedge and hyperedge-to-metavert connections
        connection_types = set()
        connection_details = []
        for edge_key, edge_data in edges.items():
            edge_type = edge_data.get("type")
            connection_details.append(f"{edge_key}: {edge_type}")
            if edge_type in ["metavert_to_hye", "hye_to_metavert"]:
                connection_types.add(edge_type)
        
        # Should have both types of connections
        self.assertIn("metavert_to_hye", connection_types,
                     f"Missing 'metavert_to_hye' connection type. "
                     f"Found connection types: {connection_types}, "
                     f"All edges: {connection_details}")
        self.assertIn("hye_to_metavert", connection_types,
                     f"Missing 'hye_to_metavert' connection type. "
                     f"Found connection types: {connection_types}, "
                     f"All edges: {connection_details}")

    def test_invalid_metaedge_raises_error(self):
        """Test that invalid metaedge construction raises error"""
        invalid_edge = ((1, 2), 3, 4)
        
        with self.assertRaises(ValueError) as context:
            # Try to add a metaedge with nested structure and length != 2
            self.graph.add_edge(invalid_edge, metadata={"type": "invalid"})
        
        # Verify the error message provides context
        if not hasattr(context, 'exception'):
            self.fail(f"Expected ValueError when adding invalid metaedge {invalid_edge}, but no exception was raised")

    def test_directed_hyperedge(self):
        """Test adding a directed hyperedge (2 elements where one is nested)"""
        self.graph.add_nodes([1, 2, 3])
        
        # Add directed hyperedge
        directed_edge = ((1, 2), 3)
        self.graph.add_edge(directed_edge, metadata={"type": "hyper"})
        
        edges = self.graph.get_edges()
        metaverts = self.graph.get_all_metaverts()
        all_nodes = self.graph.get_nodes()
        
        # Should have created edges and metavertex
        self.assertGreater(len(edges), 0, 
                          f"Expected edges to be created after adding directed hyperedge {directed_edge}, "
                          f"but found {len(edges)} edges. Available edges: {list(edges.keys())}")
        self.assertGreater(len(metaverts), 0, 
                          f"Expected metavertex to be created after adding directed hyperedge {directed_edge}, "
                          f"but found {len(metaverts)} metaverts. "
                          f"All nodes by type: {[(k, v.get('type')) for k, v in all_nodes.items()]}")