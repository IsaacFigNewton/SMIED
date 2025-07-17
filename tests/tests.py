import unittest

from noske.SemanticHypergraph import SemanticHypergraph
from noske.PatternLoader import PatternLoader
from noske.PatternMatcher import PatternMatcher

class TestPatternMatcher(unittest.TestCase):
    def setUp(self):
        # Create a simple semantic hypergraph for testing
        self.graph = SemanticHypergraph()
        self.graph.add_node(1, {"text": "test", "pos": "NOUN", "lemma": "test"})
        self.graph.add_node(2, {"text": "example", "pos": "VERB", "lemma": "example"})
        self.graph.add_edge(1, 2, {"type": "relation"})

        # Initialize PatternLoader with default patterns
        self.pattern_loader = PatternLoader()

        # Initialize PatternMatcher with the graph and loader
        self.matcher = PatternMatcher(self.graph, self.pattern_loader)

    def test_node_matches(self):
        node_attrs = {"text": "test", "pos": "NOUN", "lemma": "test"}
        pattern_attrs = {"text": "test", "pos": {"NOUN"}}
        self.assertTrue(self.matcher.node_matches(node_attrs, pattern_attrs))

    def test_edge_matches(self):
        edge_attrs = {"type": "relation"}
        pattern_attrs = {"type": {"relation"}}
        self.assertTrue(self.matcher.edge_matches(edge_attrs, pattern_attrs))
    
    def test_match_patterns(self):
        # Define a simple pattern to match
        pattern = {
            "name": "test_pattern",
            "pattern": [
                {"text": "test", "pos": {"NOUN"}},
                {"text": "example", "pos": {"VERB"}}
            ]
        }
        self.pattern_loader.add_pattern("test_category", pattern["name"], pattern["pattern"])

        # Match the pattern against the semantic graph
        results = self.matcher.match_patterns("test_category")
        self.assertIn("test_pattern", results)
        self.assertGreater(len(results["test_pattern"]), 0)
    
    def test_match_all_patterns(self):
        # Match all patterns across all categories
        results = self.matcher.match_all_patterns()
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
    
if __name__ == '__main__':
    unittest.main()