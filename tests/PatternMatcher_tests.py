import unittest
import spacy
import json

from noske.SemanticMetagraph import SemanticMetagraph
from noske.PatternLoader import PatternLoader
from noske.PatternMatcher import PatternMatcher

class TestPatternMatcher(unittest.TestCase):
    def setUp(self):
        """Set up a simple semantic hypergraph and pattern matcher for testing"""
        nlp = spacy.load("en_core_web_sm")

        # Create a simple semantic hypergraph for testing
        self.graph = SemanticMetagraph(doc=nlp("This runs a test."))

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
        test_category = "test_category"
        pattern = {
            "name": "test_pattern",
            "pattern": [
                {"text": "runs", "pos": {"VERB"}},
                {"rel_pos": "before"},
                {"text": "test", "pos": {"NOUN"}},
            ]
        }
        self.pattern_loader.add_pattern(
            category=test_category,
            name=pattern["name"],
            pattern=pattern["pattern"],
            description="A test pattern for matching 'test' and 'runs'"
        )
        self.assertIn(test_category, self.pattern_loader.patterns)
        self.assertIn(pattern["name"], self.pattern_loader.patterns[test_category])

        # Match the pattern against the semantic graph
        results = self.matcher(test_category)
        self.assertIn(pattern["name"], results)
        self.assertGreater(len(results[pattern["name"]]), 0)
    
    def test_match_all_patterns(self):
        # Match all patterns across all categories
        results = self.matcher()
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)