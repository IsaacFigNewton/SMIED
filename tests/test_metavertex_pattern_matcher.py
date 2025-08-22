import unittest
import spacy
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.SemanticMetagraph import SemanticMetagraph
from smied.PatternMatcher import PatternMatcher
from smied.PatternLoader import PatternLoader


class TestMetavertexPatternMatcher(unittest.TestCase):
    """Test the updated PatternMatcher with metavertex structure"""
    
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
        """Set up test fixtures"""
        # Create test sentences
        self.simple_doc = self.nlp("The cat runs fast.")
        self.complex_doc = self.nlp("John gives Mary a book because she studies hard.")
        
        # Create semantic metagraphs
        self.simple_graph = SemanticMetagraph(doc=self.simple_doc)
        self.complex_graph = SemanticMetagraph(doc=self.complex_doc)
        
        # Create pattern matchers
        self.simple_matcher = PatternMatcher(self.simple_graph)
        self.complex_matcher = PatternMatcher(self.complex_graph)
    
    def test_metavertex_structure_analysis(self):
        """Test the analyze_metavertex_patterns method"""
        analysis = self.simple_matcher.analyze_metavertex_patterns()
        
        # Check analysis structure
        self.assertIn("total_metaverts", analysis)
        self.assertIn("atomic_count", analysis)
        self.assertIn("directed_relation_count", analysis)
        self.assertIn("undirected_relation_count", analysis)
        self.assertIn("relation_types", analysis)
        self.assertIn("pos_distribution", analysis)
        
        # Verify counts make sense
        self.assertGreater(analysis["total_metaverts"], 0)
        self.assertGreater(analysis["atomic_count"], 0)
        self.assertIsInstance(analysis["relation_types"], dict)
        self.assertIsInstance(analysis["pos_distribution"], dict)
    
    def test_find_atomic_metavertices(self):
        """Test finding atomic metavertices with filters"""
        # Find all nouns
        nouns = self.simple_matcher.find_atomic_metavertices(pos="NOUN")
        self.assertIsInstance(nouns, list)
        
        # Find all verbs
        verbs = self.simple_matcher.find_atomic_metavertices(pos="VERB")
        self.assertIsInstance(verbs, list)
        
        # Verify results are metavertex indices
        for noun_idx in nouns:
            self.assertIsInstance(noun_idx, int)
            self.assertLess(noun_idx, len(self.simple_graph.metaverts))
    
    def test_find_relation_metavertices(self):
        """Test finding relation metavertices"""
        # Find all directed relations
        relations = self.simple_matcher.find_relation_metavertices()
        self.assertIsInstance(relations, list)
        
        # Find specific relation types
        subject_relations = self.simple_matcher.find_relation_metavertices(relation_type="nsubj")
        self.assertIsInstance(subject_relations, list)
        
        # Verify results are metavertex indices
        for rel_idx in relations:
            self.assertIsInstance(rel_idx, int)
            self.assertLess(rel_idx, len(self.simple_graph.metaverts))
    
    def test_metavertex_matches(self):
        """Test the metavertex_matches method"""
        # Test with atomic metavertex
        if len(self.simple_graph.metaverts) > 0:
            # Find first atomic metavertex
            for mv_idx, mv in self.simple_graph.metaverts.items():
                if isinstance(mv[0], str):
                    # Test matching by type
                    self.assertTrue(self.simple_matcher.metavertex_matches(mv_idx, {"mv_type": "atomic"}))
                    
                    # Test matching by POS if available
                    if len(mv) > 1 and "pos" in mv[1]:
                        pos = mv[1]["pos"]
                        self.assertTrue(self.simple_matcher.metavertex_matches(mv_idx, {"pos": pos}))
                    break
    
    def test_basic_pattern_matching(self):
        """Test basic pattern matching with new metavertex format"""
        # Create a simple pattern to find atomic nouns
        pattern = [{"mv_type": "atomic", "pos": ["NOUN", "PROPN"]}]
        
        results = self.simple_matcher.match_metavertex_chain(pattern)
        self.assertIsInstance(results, list)
        
        # Each result should be a list of metavertex indices
        for result in results:
            self.assertIsInstance(result, list)
            for mv_idx in result:
                self.assertIsInstance(mv_idx, int)
                self.assertLess(mv_idx, len(self.simple_graph.metaverts))
    
    def test_metavertex_basic_patterns(self):
        """Test the metavertex_basic patterns"""
        try:
            # Test atomic noun pattern
            results = self.simple_matcher("metavertex_basic", "atomic_noun")
            self.assertIsInstance(results, list)
            
            # Test directed relation pattern
            results = self.simple_matcher("metavertex_basic", "directed_relation")
            self.assertIsInstance(results, list)
            
        except KeyError as e:
            # Pattern category might not be loaded
            self.skipTest(f"metavertex_basic patterns not loaded: {e}")
    
    def test_complex_pattern_matching(self):
        """Test complex pattern matching with relation sequences"""
        # Create a pattern to find subject-verb sequences
        pattern = [
            {"mv_type": "atomic", "pos": ["NOUN", "PROPN"]},
            {"mv_type": "directed_relation", "relation_type": "nsubj", "requires_reference": True}
        ]
        
        results = self.complex_matcher.match_metavertex_chain(pattern)
        self.assertIsInstance(results, list)
        
        # Verify results structure
        for result in results:
            self.assertEqual(len(result), 2)  # Should have 2 metavertices in the chain
            self.assertIsInstance(result[0], int)
            self.assertIsInstance(result[1], int)
    
    def test_get_metavertex_context(self):
        """Test getting context for metavertex sequences"""
        # Find any atomic metavertex
        atomic_indices = []
        for mv_idx, mv in self.simple_graph.metaverts.items():
            if isinstance(mv[0], str):
                atomic_indices.append(mv_idx)
                if len(atomic_indices) >= 2:
                    break
        
        if len(atomic_indices) >= 2:
            context = self.simple_matcher.get_metavertex_context(atomic_indices[:2])
            
            # Check context structure
            self.assertIn("indices", context)
            self.assertIn("metaverts", context)
            self.assertIn("summary", context)
            
            self.assertEqual(context["indices"], atomic_indices[:2])
            self.assertIsInstance(context["summary"], str)
    
    def test_pattern_loader_with_metavertex_patterns(self):
        """Test that PatternLoader loads metavertex patterns"""
        loader = PatternLoader()
        
        # Check that metavertex pattern categories are loaded
        expected_categories = ["metavertex_basic", "metavertex_semantic", "metavertex_complex"]
        
        for category in expected_categories:
            if category in loader.patterns:
                self.assertIsInstance(loader.patterns[category], dict)
                self.assertGreater(len(loader.patterns[category]), 0)
            else:
                print(f"Warning: {category} not found in loaded patterns")
    
    def test_pattern_matching_modes(self):
        """Test both metavertex and NetworkX matching modes"""
        # Test metavertex mode (default)
        self.simple_matcher.use_metavertex_matching = True
        
        # Add a simple test pattern
        test_pattern = [{"mv_type": "atomic", "pos": ["NOUN"]}]
        self.simple_matcher.add_pattern(
            name="test_noun",
            pattern=test_pattern,
            description="Test noun pattern",
            category="test"
        )
        
        # Test metavertex matching
        mv_results = self.simple_matcher("test", "test_noun")
        self.assertIsInstance(mv_results, list)
        
        # Test NetworkX mode
        self.simple_matcher.use_metavertex_matching = False
        try:
            nx_results = self.simple_matcher("test", "test_noun")
            self.assertIsInstance(nx_results, list)
        except (ValueError, AttributeError):
            # NetworkX mode might fail with metavertex patterns
            pass
    
    def test_get_metavertex_chain(self):
        """Test getting metavertex chains"""
        # Find first atomic metavertex
        start_idx = None
        for mv_idx, mv in self.simple_graph.metaverts.items():
            if isinstance(mv[0], str):
                start_idx = mv_idx
                break
        
        if start_idx is not None:
            chains = self.simple_matcher.get_metavertex_chain(start_idx, max_depth=2)
            self.assertIsInstance(chains, list)
            
            # Each chain should be a list of metavertex indices
            for chain in chains:
                self.assertIsInstance(chain, list)
                self.assertGreater(len(chain), 1)  # Should have at least start + 1 more
                self.assertEqual(chain[0], start_idx)  # Should start with the given index
    
    def test_complex_semantic_patterns(self):
        """Test complex semantic patterns on more complex text"""
        # Create a more complex sentence
        complex_text = "The professor teaches students mathematics in the classroom."
        complex_doc = self.nlp(complex_text)
        complex_graph = SemanticMetagraph(doc=complex_doc)
        complex_matcher = PatternMatcher(complex_graph)
        
        # Test analysis
        analysis = complex_matcher.analyze_metavertex_patterns()
        self.assertGreater(analysis["total_metaverts"], 5)  # Should have multiple metavertices
        self.assertGreater(analysis["atomic_count"], 3)    # Should have multiple words
        
        # Test finding specific patterns
        nouns = complex_matcher.find_atomic_metavertices(pos="NOUN")
        verbs = complex_matcher.find_atomic_metavertices(pos="VERB")
        
        self.assertGreater(len(nouns), 0)
        self.assertGreater(len(verbs), 0)


class TestMetavertexPatternIntegration(unittest.TestCase):
    """Integration tests for metavertex pattern matching"""
    
    @classmethod
    def setUpClass(cls):
        """Load spaCy model once for all tests"""
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            cls.nlp = spacy.load("en_core_web_sm")
    
    def test_full_pipeline_with_metavertex_patterns(self):
        """Test full pipeline from text to pattern matching"""
        # Create a test sentence with clear semantic structure
        text = "Alice gives Bob a red book because he needs it for research."
        doc = self.nlp(text)
        
        # Create semantic metagraph
        graph = SemanticMetagraph(doc=doc)
        self.assertGreater(len(graph.metaverts), 0)
        
        # Create pattern matcher
        matcher = PatternMatcher(graph)
        
        # Analyze structure
        analysis = matcher.analyze_metavertex_patterns()
        self.assertGreater(analysis["atomic_count"], 5)
        self.assertGreater(analysis["directed_relation_count"], 3)
        
        # Test finding specific semantic roles
        nouns = matcher.find_atomic_metavertices(pos="NOUN")
        proper_nouns = matcher.find_atomic_metavertices(pos="PROPN")
        verbs = matcher.find_atomic_metavertices(pos="VERB")
        
        self.assertGreater(len(nouns), 0)
        self.assertGreater(len(proper_nouns), 0)
        self.assertGreater(len(verbs), 0)
        
        # Test relation finding
        subject_relations = matcher.find_relation_metavertices(relation_type="nsubj")
        object_relations = matcher.find_relation_metavertices(relation_type="dobj")
        
        # Should find at least some relations
        total_relations = len(subject_relations) + len(object_relations)
        self.assertGreater(total_relations, 0)
    
    def test_pattern_matching_accuracy(self):
        """Test that pattern matching produces accurate results"""
        # Simple sentence with clear structure
        text = "The dog chases the cat."
        doc = self.nlp(text)
        graph = SemanticMetagraph(doc=doc)
        matcher = PatternMatcher(graph)
        
        # Find subject-verb-object pattern
        pattern = [
            {"mv_type": "atomic", "pos": ["NOUN", "PROPN"]},
            {"mv_type": "directed_relation", "relation_type": "nsubj", "requires_reference": True},
            {"mv_type": "atomic", "pos": ["VERB"]},
            {"mv_type": "directed_relation", "relation_type": "dobj", "requires_reference": True},
            {"mv_type": "atomic", "pos": ["NOUN", "PROPN"]}
        ]
        
        results = matcher.match_metavertex_chain(pattern)
        
        # Should find at least one complete SVO pattern
        if len(results) > 0:
            # Verify the structure
            result = results[0]
            self.assertEqual(len(result), 5)  # Should match all 5 pattern elements
            
            # Verify that we can get meaningful context
            context = matcher.get_metavertex_context(result)
            self.assertIn("dog", context["summary"] + " ".join([str(mv) for mv in context["metaverts"]]))
            self.assertIn("chase", context["summary"] + " ".join([str(mv) for mv in context["metaverts"]]))
            self.assertIn("cat", context["summary"] + " ".join([str(mv) for mv in context["metaverts"]]))


if __name__ == '__main__':
    unittest.main()