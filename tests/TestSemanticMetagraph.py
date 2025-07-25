import unittest
from unittest.mock import Mock, patch
from noske.SemanticMetagraph import SemanticMetagraph
import spacy
from spacy.tokens import Doc, Token

class TestSemanticMetagraph(unittest.TestCase):
    def setUp(self):
        self.graph = SemanticMetagraph()
        # Create mock spacy doc
        self.nlp = spacy.load("en_core_web_sm")
        self.doc = self.nlp("The big cat sleeps.")
        
    def test_init_empty(self):
        graph = SemanticMetagraph()
        self.assertIsNotNone(graph.G)
        self.assertEqual(len(graph.get_nodes()), 0)
        self.assertEqual(len(graph.get_edges()), 0)

    def test_add_single_node(self):
        self.graph.add_node(1, metadata={"text": "test"})
        nodes = self.graph.get_nodes()
        self.assertIn(1, nodes)
        self.assertEqual(nodes[1]["text"], "test")
        self.assertEqual(nodes[1]["node_id"], "1")

    def test_add_multiple_nodes(self):
        nodes = [1, 2, 3]
        metadata = [{"text": f"node{i}"} for i in nodes]
        self.graph.add_nodes(nodes, metadata=metadata)
        graph_nodes = self.graph.get_nodes()
        self.assertEqual(len(graph_nodes), 3)
        for i, node in enumerate(nodes):
            self.assertEqual(graph_nodes[node]["text"], f"node{i+1}")

    def test_add_edge(self):
        self.graph.add_nodes([1, 2])
        self.graph.add_edge([1, 2], metadata={"type": "test_edge"})
        edges = self.graph.get_edges()
        self.assertEqual(len(edges), 1)
        edge_data = list(edges.values())[0]
        self.assertEqual(edge_data["type"], "test_edge")

    def test_json_serialization(self):
        # Add test data
        self.graph.add_node(1, metadata={"text": "test"})
        self.graph.add_node(2, metadata={"text": "test2"})
        self.graph.add_edge([1, 2], metadata={"type": "test_edge"})
        
        # Test serialization
        json_data = self.graph.to_json()
        self.assertIn("nodes", json_data)
        self.assertIn("edges", json_data)
        
        # Test deserialization
        new_graph = SemanticMetagraph.from_json(json_data)
        self.assertEqual(
            len(self.graph.get_nodes()), 
            len(new_graph.get_nodes())
        )
        self.assertEqual(
            len(self.graph.get_edges()),
            len(new_graph.get_edges())
        )

    def test_add_doc(self):
        graph = SemanticMetagraph(doc=self.doc)
        nodes = graph.get_nodes()
        
        # Check if all tokens are added as nodes
        self.assertEqual(len(nodes), len(self.doc))
        
        # Check if token metadata is preserved
        for token in self.doc:
            node_data = nodes[token.i]
            self.assertEqual(node_data["text"], token.text)
            self.assertEqual(node_data["pos"], token.pos_)

    def test_get_token_tags(self):
        token = self.doc[0]  # "The"
        tags = SemanticMetagraph.get_token_tags(token)
        self.assertIn("case", tags)
        self.assertIn("type", tags)
        
    def test_get_dep_edges(self):
        token = self.doc[2]  # "cat"
        edges, metadata = SemanticMetagraph.get_dep_edges(token)
        self.assertIsInstance(edges, list)
        self.assertIsInstance(metadata, list)
        self.assertEqual(len(edges), len(metadata))

    def test_empty_inputs(self):
        with self.assertRaises(Exception):
            SemanticMetagraph.from_json({"nodes": "", "edges": ""})

if __name__ == '__main__':
    unittest.main()
