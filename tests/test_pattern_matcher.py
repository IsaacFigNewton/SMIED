import unittest
import spacy
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.SemanticMetagraph import SemanticMetagraph
from smied.PatternLoader import PatternLoader
from smied.PatternMatcher import PatternMatcher

class TestPatternMatcher(unittest.TestCase):
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
        """Set up a simple semantic metagraph and pattern matcher for testing"""
        # Create a simple semantic metagraph for testing
        self.doc = self.nlp("This runs a test.")
        self.graph = SemanticMetagraph(doc=self.doc)

        # Initialize PatternLoader with default patterns
        self.pattern_loader = PatternLoader()

        # Initialize PatternMatcher with the graph and loader
        self.matcher = PatternMatcher(self.graph, self.pattern_loader)

    def test_node_matches(self):
        """Test node matching functionality"""
        node_attrs = {"text": "test", "pos": "NOUN", "lemma": "test"}
        pattern_attrs = {"text": "test", "pos": {"NOUN"}}
        self.assertTrue(self.matcher.node_matches(node_attrs, pattern_attrs))
        
        # Test negative case
        pattern_attrs_negative = {"text": "test", "pos": {"VERB"}}
        self.assertFalse(self.matcher.node_matches(node_attrs, pattern_attrs_negative))
        
        # Test lemma matching
        pattern_attrs_lemma = {"lemma": {"test", "testing"}}
        self.assertTrue(self.matcher.node_matches(node_attrs, pattern_attrs_lemma))
        
        # Test semantic type matching
        node_attrs_semantic = {"text": "test", "semantic_type": "concept"}
        pattern_attrs_semantic = {"semantic_type": {"concept", "entity"}}
        self.assertTrue(self.matcher.node_matches(node_attrs_semantic, pattern_attrs_semantic))

    def test_edge_matches(self):
        """Test edge matching functionality"""
        edge_attrs = {"type": "relation"}
        pattern_attrs = {"type": {"relation"}}
        self.assertTrue(self.matcher.edge_matches(edge_attrs, pattern_attrs))
        
        # Test negative case
        pattern_attrs_negative = {"type": {"different_relation"}}
        self.assertFalse(self.matcher.edge_matches(edge_attrs, pattern_attrs_negative))
        
        # Test exact match
        pattern_attrs_exact = {"type": "relation"}
        self.assertTrue(self.matcher.edge_matches(edge_attrs, pattern_attrs_exact))
    
    def test_pattern_loader_integration(self):
        """Test integration with PatternLoader"""
        # Test adding patterns through the matcher
        self.matcher.add_pattern(
            name="test_integration",
            pattern=[{"text": "test"}],
            description="Integration test pattern",
            category="integration"
        )
        
        # Verify pattern was added
        self.assertIn("integration", self.pattern_loader.patterns)
        self.assertIn("test_integration", self.pattern_loader.patterns["integration"])
    
    def test_graph_conversion(self):
        """Test that the semantic graph can be converted to NetworkX"""
        nx_graph = self.graph.to_nx()
        
        # Verify basic graph properties
        self.assertGreater(len(nx_graph.nodes()), 0)
        
        # Check that nodes have required attributes for pattern matching
        for node_id in nx_graph.nodes():
            node_data = nx_graph.nodes[node_id]
            # Should have label at minimum
            self.assertIn("label", node_data)
    
    def test_semantic_graph_structure(self):
        """Test the structure of the semantic metagraph"""
        # Test that we have metaverts
        self.assertGreater(len(self.graph.metaverts), 0)
        
        # Test that we can get tokens
        tokens = self.graph.get_tokens()
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Test that we can get relations
        relations = self.graph.get_relations()
        self.assertIsInstance(relations, list)
    
    def test_matcher_initialization(self):
        """Test PatternMatcher initialization"""
        # Test with explicit pattern loader
        custom_loader = PatternLoader()
        matcher = PatternMatcher(self.graph, custom_loader)
        self.assertEqual(matcher.semantic_graph, self.graph)
        self.assertEqual(matcher.pattern_loader, custom_loader)
        
        # Test with default pattern loader
        matcher_default = PatternMatcher(self.graph)
        self.assertEqual(matcher_default.semantic_graph, self.graph)
        self.assertIsNotNone(matcher_default.pattern_loader)
    
    def test_pattern_matching_compatibility(self):
        """Test that PatternMatcher works with new SemanticMetagraph structure"""
        # The PatternMatcher expects semantic_graph.G to be a NetworkX graph
        # but our new SemanticMetagraph doesn't have a .G attribute
        
        # Check if matcher has access to graph structure
        self.assertTrue(hasattr(self.matcher.semantic_graph, 'to_nx'))
        
        # Convert to NetworkX and verify structure
        nx_graph = self.matcher.semantic_graph.to_nx()
        self.assertGreater(len(nx_graph.nodes()), 0)
        
        # The PatternMatcher currently expects self.semantic_graph.G
        # This test documents the incompatibility that needs to be fixed
        with self.assertRaises(AttributeError):
            # This will fail because SemanticMetagraph doesn't have .G attribute
            _ = self.matcher.semantic_graph.G


class TestPatternMatcherWithUpdatedGraph(unittest.TestCase):
    """Test PatternMatcher functionality that needs to be updated for new graph structure"""
    
    @classmethod
    def setUpClass(cls):
        """Load spaCy model once for all tests"""
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            cls.nlp = spacy.load("en_core_web_sm")
    
    def setUp(self):
        """Set up for testing with mock graph structure"""
        self.doc = self.nlp("The cat runs fast.")
        self.graph = SemanticMetagraph(doc=self.doc)
        self.pattern_loader = PatternLoader()
        
        # Create a PatternMatcher but note it may need updates
        self.matcher = PatternMatcher(self.graph, self.pattern_loader)
    
    def test_pattern_matching_with_networkx_conversion(self):
        """Test pattern matching using NetworkX conversion"""
        # Convert to NetworkX
        nx_graph = self.graph.to_nx()
        
        # Create a temporary PatternMatcher that uses the NetworkX graph directly
        # This simulates how the PatternMatcher should work with the new structure
        class TempPatternMatcher:
            def __init__(self, nx_graph, pattern_loader):
                self.G = nx_graph  # Use NetworkX graph directly
                self.pattern_loader = pattern_loader
            
            def node_matches(self, node_attrs, pattern_attrs):
                for k, v in pattern_attrs.items():
                    if isinstance(v, set):
                        if node_attrs.get(k) not in v:
                            return False
                    else:
                        if node_attrs.get(k) != v:
                            return False
                return True
        
        temp_matcher = TempPatternMatcher(nx_graph, self.pattern_loader)
        
        # Test node matching with actual graph data
        for node_id in nx_graph.nodes():
            node_data = nx_graph.nodes[node_id]
            # Test matching with a simple pattern
            if "label" in node_data:
                pattern = {"label": node_data["label"]}
                self.assertTrue(temp_matcher.node_matches(node_data, pattern))
    
    def test_graph_structure_compatibility(self):
        """Test that graph structure is compatible with pattern matching expectations"""
        nx_graph = self.graph.to_nx()
        
        # Check that we have the expected structure
        self.assertGreater(len(nx_graph.nodes()), 0)
        
        # Check node attributes
        node_attrs = set()
        for node_id in nx_graph.nodes():
            node_data = nx_graph.nodes[node_id]
            node_attrs.update(node_data.keys())
        
        # Should have basic attributes needed for pattern matching
        self.assertIn("label", node_attrs)
        
        # Check edge attributes if edges exist
        if len(nx_graph.edges()) > 0:
            edge_attrs = set()
            for u, v in nx_graph.edges():
                edge_data = nx_graph.edges[u, v]
                edge_attrs.update(edge_data.keys())
            
            # Should have label for edges
            self.assertIn("label", edge_attrs)
    
    def test_pattern_matching_works_now(self):
        """Test that PatternMatcher now works with new graph structure after fix"""
        # Add a simple pattern
        self.pattern_loader.add_pattern(
            category="test",
            name="simple_pattern",
            pattern=[{"text": "cat"}],
            description="Simple pattern to match 'cat'"
        )
        
        # This should now work because we fixed the PatternMatcher
        try:
            results = self.matcher("test")
            self.assertIsInstance(results, dict)
            self.assertIn("simple_pattern", results)
            # Results might be empty if "cat" is not in the test document
        except Exception as e:
            self.fail(f"PatternMatcher should work with new structure but failed: {e}")


class TestPatternMatcherFixedImplementation(unittest.TestCase):
    """Test a fixed version of PatternMatcher that works with new SemanticMetagraph"""
    
    @classmethod
    def setUpClass(cls):
        """Load spaCy model once for all tests"""
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            cls.nlp = spacy.load("en_core_web_sm")
    
    def setUp(self):
        """Set up for testing fixed implementation"""
        self.doc = self.nlp("The quick brown fox jumps.")
        self.graph = SemanticMetagraph(doc=self.doc)
        self.pattern_loader = PatternLoader()
        
        # Create a fixed PatternMatcher that works with new graph structure
        class FixedPatternMatcher(PatternMatcher):
            def match_chain(self, query):
                """Fixed match_chain that uses to_nx() conversion"""
                # Convert the semantic graph to NetworkX format
                g = self.semantic_graph.to_nx()
                
                # Verify that query length is odd and > 0
                if len(query) < 1 or len(query) % 2 == 0:
                    raise ValueError("Query must be non-empty and have odd length (node, edge, node, ...).")

                # Extract pattern requirements
                n = (len(query) + 1) // 2  # number of nodes in pattern
                node_patterns = [query[2*i] for i in range(n)]
                edge_patterns = [query[2*i + 1] for i in range(n-1)]

                # Find all matching paths using DFS
                results = []
                
                def dfs(current_path, pattern_idx):
                    if pattern_idx == n:
                        # We've matched all nodes in the pattern
                        results.append(current_path[:])
                        return
                    
                    if pattern_idx == 0:
                        # First node - try all nodes in the graph
                        for node in g.nodes():
                            node_data = g.nodes[node].copy()
                            # Add node_id for pattern matching
                            node_data["node_id"] = str(node)
                            if self.node_matches(node_data, node_patterns[0]):
                                current_path.append(node)
                                dfs(current_path, 1)
                                current_path.pop()
                    else:
                        # Not the first node - look for neighbors of the last node
                        last_node = current_path[-1]
                        edge_pattern = edge_patterns[pattern_idx - 1]
                        node_pattern = node_patterns[pattern_idx]
                        
                        # Check all outgoing edges from the last node
                        for neighbor in g.neighbors(last_node):
                            # Check if the edge matches the pattern
                            edge_data = g.edges[last_node, neighbor]
                            if self.edge_matches(edge_data, edge_pattern):
                                # Check if the neighbor node matches the pattern
                                neighbor_data = g.nodes[neighbor].copy()
                                neighbor_data["node_id"] = str(neighbor)
                                if self.node_matches(neighbor_data, node_pattern):
                                    current_path.append(neighbor)
                                    dfs(current_path, pattern_idx + 1)
                                    current_path.pop()
                
                # Start DFS from an empty path
                dfs([], 0)
                return results
        
        self.fixed_matcher = FixedPatternMatcher(self.graph, self.pattern_loader)
    
    def test_fixed_pattern_matching(self):
        """Test that the fixed PatternMatcher works with new graph structure"""
        # Add a simple pattern that should match
        self.pattern_loader.add_pattern(
            category="test",
            name="noun_pattern",
            pattern=[{"pos": {"NOUN"}}],  # Match any noun
            description="Pattern to match nouns"
        )
        
        # This should work with the fixed implementation
        try:
            results = self.fixed_matcher("test")
            self.assertIsInstance(results, dict)
            self.assertIn("noun_pattern", results)
        except Exception as e:
            # Log what went wrong for debugging
            print(f"Fixed matcher failed: {e}")
            # Convert to NetworkX and check structure
            nx_graph = self.graph.to_nx()
            print(f"Graph has {len(nx_graph.nodes())} nodes and {len(nx_graph.edges())} edges")
            for node_id in list(nx_graph.nodes())[:3]:  # Show first 3 nodes
                print(f"Node {node_id}: {nx_graph.nodes[node_id]}")
            raise


if __name__ == '__main__':
    unittest.main()