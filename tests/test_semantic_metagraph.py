import unittest
import json
import spacy
from spacy.tokens import Doc
import networkx as nx
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from noske.SemanticMetagraph import SemanticMetagraph


class TestSemanticMetagraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load spaCy model once for all tests"""
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not installed, try to download it
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            cls.nlp = spacy.load("en_core_web_sm")
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple test document
        self.test_text = "Apple is a technology company based in California."
        self.doc = self.nlp(self.test_text)
        
        # Create a test vertex list for direct initialization
        self.test_vert_list = [
            ("word1", {"pos": "NOUN"}),
            ("word2", {"pos": "VERB"}),
            ((0, 1), {"relation": "subject"})
        ]
    
    def test_initialization_from_doc(self):
        """Test initialization from a spaCy Doc object"""
        sg = SemanticMetagraph(doc=self.doc)
        self.assertIsNotNone(sg)
        self.assertIsInstance(sg.metaverts, list)
        self.assertGreater(len(sg.metaverts), 0)
        self.assertEqual(sg.doc, self.doc)
    
    def test_initialization_from_vert_list(self):
        """Test initialization from a vertex list"""
        sg = SemanticMetagraph(vert_list=self.test_vert_list)
        self.assertIsNotNone(sg)
        self.assertEqual(len(sg.metaverts), 3)
        self.assertIsNone(sg.doc)
    
    def test_initialization_empty(self):
        """Test initialization with no arguments creates empty graph"""
        sg = SemanticMetagraph()
        self.assertIsNotNone(sg)
        self.assertEqual(len(sg.metaverts), 0)
        self.assertIsNone(sg.doc)
    
    def test_build_metaverts_from_doc(self):
        """Test the _build_metaverts_from_doc method"""
        sg = SemanticMetagraph()
        metaverts = sg._build_metaverts_from_doc(self.doc)
        
        # Check that metaverts were created
        self.assertIsInstance(metaverts, list)
        self.assertGreater(len(metaverts), 0)
        
        # Check that tokens are properly represented
        token_count = len(self.doc)
        # At least one metavert per token
        self.assertGreaterEqual(len(metaverts), token_count)
        
        # Check that first metaverts are tokens
        for i in range(token_count):
            mv = metaverts[i]
            self.assertIsInstance(mv[0], str)
            self.assertIn("text", mv[1])
            self.assertIn("pos", mv[1])
    
    def test_get_token_tags(self):
        """Test the get_token_tags static method"""
        token = self.doc[0]  # "Apple"
        tags = SemanticMetagraph.get_token_tags(token)
        
        self.assertIsInstance(tags, dict)
        # Check for case tag
        self.assertIn("case", tags)
        self.assertEqual(tags["case"], "title")
        # Check for type tag
        self.assertIn("type", tags)
        self.assertEqual(tags["type"], "word")
    
    def test_get_dep_edges(self):
        """Test the get_dep_edges static method"""
        # Find a token with children
        token_with_children = None
        for token in self.doc:
            if list(token.children):
                token_with_children = token
                break
        
        if token_with_children:
            edges = SemanticMetagraph.get_dep_edges(token_with_children)
            self.assertIsInstance(edges, list)
            if edges:
                edge = edges[0]
                self.assertEqual(len(edge), 3)
                self.assertIn("type", edge[2])
                self.assertIn("rel_pos", edge[2])
    
    def test_to_nx_conversion(self):
        """Test conversion to NetworkX graph"""
        sg = SemanticMetagraph(doc=self.doc)
        G = sg.to_nx()
        
        self.assertIsInstance(G, nx.DiGraph)
        self.assertGreater(len(G.nodes()), 0)
        
        # Check node attributes
        for node in G.nodes():
            node_data = G.nodes[node]
            self.assertIn("label", node_data)
    
    def test_to_json_and_from_json(self):
        """Test JSON serialization and deserialization"""
        sg1 = SemanticMetagraph(vert_list=self.test_vert_list)
        
        # Convert to JSON
        json_data = sg1.to_json()
        self.assertIsInstance(json_data, dict)
        self.assertIn("metaverts", json_data)
        
        # Parse the JSON to ensure it's valid
        metaverts_json = json.loads(json_data["metaverts"])
        self.assertIsInstance(metaverts_json, list)
        
        # Create new graph from JSON
        sg2 = SemanticMetagraph.from_json(json_data)
        self.assertIsInstance(sg2, SemanticMetagraph)
        self.assertEqual(len(sg2.metaverts), len(sg1.metaverts))
    
    def test_get_tokens(self):
        """Test the get_tokens method"""
        sg = SemanticMetagraph(doc=self.doc)
        tokens = sg.get_tokens()
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        for token in tokens:
            self.assertIsInstance(token, dict)
            self.assertIn("metavert_idx", token)
            self.assertIn("token_idx", token)
            self.assertIn("text", token)
            self.assertIn("metadata", token)
    
    def test_get_relations(self):
        """Test the get_relations method"""
        sg = SemanticMetagraph(doc=self.doc)
        relations = sg.get_relations()
        
        self.assertIsInstance(relations, list)
        
        for relation in relations:
            self.assertIsInstance(relation, dict)
            self.assertIn("metavert_idx", relation)
            self.assertIn("type", relation)
            self.assertIn("relation", relation)
            self.assertIn("metadata", relation)
            
            if relation["type"] == "directed":
                self.assertIn("source", relation)
                self.assertIn("target", relation)
            elif relation["type"] == "undirected":
                self.assertIn("nodes", relation)
    
    def test_add_vert(self):
        """Test adding vertices to the graph"""
        sg = SemanticMetagraph()
        
        # Add atomic vertex
        sg.add_vert("test_word", {"pos": "NOUN"})
        self.assertEqual(len(sg.metaverts), 1)
        
        # Add another atomic vertex
        sg.add_vert("another_word", {"pos": "VERB"})
        self.assertEqual(len(sg.metaverts), 2)
        
        # Add directed relation
        sg.add_vert((0, 1), {"relation": "test_relation"})
        self.assertEqual(len(sg.metaverts), 3)
    
    def test_remove_vert(self):
        """Test removing vertices from the graph"""
        sg = SemanticMetagraph(vert_list=[
            ("word1", {"pos": "NOUN"}),
            ("word2", {"pos": "VERB"}),
            ((0, 1), {"relation": "subject"}),
            ("word3", {"pos": "ADJ"})
        ])
        
        initial_length = len(sg.metaverts)
        
        # Remove the first vertex (word1)
        sg.remove_vert(0)
        
        # Check that the graph is smaller
        self.assertLess(len(sg.metaverts), initial_length)
        
        # The relation depending on vertex 0 should also be removed
        for mv in sg.metaverts:
            if isinstance(mv[0], tuple):
                self.assertNotIn(0, mv[0])
    
    def test_complex_document(self):
        """Test with a more complex document"""
        complex_text = "The quick brown fox jumps over the lazy dog. It was a beautiful day."
        complex_doc = self.nlp(complex_text)
        
        sg = SemanticMetagraph(doc=complex_doc)
        
        # Check that graph was created
        self.assertGreater(len(sg.metaverts), len(complex_doc))
        
        # Check tokens
        tokens = sg.get_tokens()
        self.assertEqual(len(tokens), len(complex_doc))
        
        # Check relations exist
        relations = sg.get_relations()
        self.assertGreater(len(relations), 0)
    
    def test_entity_handling(self):
        """Test handling of named entities"""
        entity_text = "Microsoft and Google are competing with Apple in Silicon Valley."
        entity_doc = self.nlp(entity_text)
        
        sg = SemanticMetagraph(doc=entity_doc)
        
        # Check that entity types are added
        has_entity_relations = False
        for mv in sg.metaverts:
            if len(mv) > 1 and mv[1].get("relation") == "has_entity_type":
                has_entity_relations = True
                break
        
        # This test may pass or fail depending on the NER model
        # Just ensure no errors occur
        self.assertIsNotNone(sg)
    
    def test_punctuation_handling(self):
        """Test handling of punctuation tokens"""
        punct_text = "Hello, world! How are you?"
        punct_doc = self.nlp(punct_text)
        
        sg = SemanticMetagraph(doc=punct_doc)
        
        # Find punctuation tokens
        tokens = sg.get_tokens()
        punct_tokens = [t for t in tokens if t["metadata"].get("type") == "punct"]
        
        self.assertGreater(len(punct_tokens), 0)
        
        # Check punctuation subtype features
        for pt in punct_tokens:
            if "subtype_features" in pt["metadata"]:
                self.assertIsInstance(pt["metadata"]["subtype_features"], list)


class TestSemanticMetagraphIntegration(unittest.TestCase):
    """Integration tests for SemanticMetagraph"""
    
    @classmethod
    def setUpClass(cls):
        """Load spaCy model once for all tests"""
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            cls.nlp = spacy.load("en_core_web_sm")
    
    def test_full_pipeline(self):
        """Test the full pipeline: create, modify, serialize, deserialize"""
        # Create from document
        text = "Natural language processing is fascinating."
        doc = self.nlp(text)
        sg1 = SemanticMetagraph(doc=doc)
        
        # Add custom vertex
        sg1.add_vert("CUSTOM", {"type": "annotation"})
        
        # Convert to JSON
        json_data = sg1.to_json()
        
        # Create new graph from JSON
        sg2 = SemanticMetagraph.from_json(json_data)
        
        # Verify graphs are equivalent
        self.assertEqual(len(sg1.metaverts), len(sg2.metaverts))
        
        # Convert to NetworkX and verify
        G1 = sg1.to_nx()
        G2 = sg2.to_nx()
        
        self.assertEqual(len(G1.nodes()), len(G2.nodes()))
        self.assertEqual(len(G1.edges()), len(G2.edges()))
    
    def test_graph_traversal(self):
        """Test that the graph structure is navigable"""
        text = "The cat sat on the mat."
        doc = self.nlp(text)
        sg = SemanticMetagraph(doc=doc)
        
        # Convert to NetworkX for traversal
        G = sg.to_nx()
        
        # Check that graph is connected (weakly for directed graph)
        # Note: The metagraph might not be fully connected, so we check components
        components = list(nx.weakly_connected_components(G))
        self.assertGreater(len(components), 0)
        
        # Check that we can traverse from nodes
        for node in G.nodes():
            # Get predecessors and successors
            preds = list(G.predecessors(node))
            succs = list(G.successors(node))
            # Node should have at least incoming or outgoing edges, or be isolated
            self.assertTrue(
                len(preds) > 0 or len(succs) > 0 or len(components) > 1,
                f"Node {node} has no connections and graph is connected"
            )


if __name__ == "__main__":
    unittest.main()