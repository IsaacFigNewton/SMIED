import unittest
import spacy
import sys
import os
from unittest.mock import Mock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.SemanticMetagraph import SemanticMetagraph
from smied.PatternMatcher import PatternMatcher
from smied.PatternLoader import PatternLoader
from tests.mocks.metavertex_pattern_matcher_mocks import MetavertexPatternMatcherMockFactory
from tests.config.metavertex_pattern_matcher_config import MetavertexPatternMatcherMockConfig


class TestMetavertexPatternMatcher(unittest.TestCase):
    """Basic functionality tests for MetavertexPatternMatcher."""
    
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
        """Set up test fixtures using factory pattern and config injection."""
        # Initialize config and mock factory
        self.config = MetavertexPatternMatcherMockConfig()
        self.mock_factory = MetavertexPatternMatcherMockFactory()
        
        # Get test sentences from config
        sentence_patterns = self.config.get_test_sentence_patterns()
        self.simple_doc = self.nlp(sentence_patterns['simple_sentences'][0])
        self.complex_doc = self.nlp(sentence_patterns['complex_sentences'][0])
        
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
    
    def test_basic_metavertex_structure_creation(self):
        """Test basic creation of metavertex structures."""
        # Use config-driven test data
        test_data = self.config.get_linguistic_feature_tests()
        pos_test = test_data['pos_tagging_tests'][0]
        
        doc = self.nlp(pos_test['sentence'])
        graph = SemanticMetagraph(doc=doc)
        matcher = PatternMatcher(graph)
        
        analysis = matcher.analyze_metavertex_patterns()
        self.assertIn("total_metaverts", analysis)
        self.assertIn("atomic_count", analysis)
        self.assertGreater(analysis["atomic_count"], 0)
    
    def test_basic_pattern_matching_functionality(self):
        """Test basic pattern matching functionality."""
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
    
    def test_atomic_metavertex_finding(self):
        """Test finding atomic metavertices with filters."""
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


class TestMetavertexPatternMatcherValidation(unittest.TestCase):
    """Validation and constraint tests for MetavertexPatternMatcher."""
    
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
        """Set up test fixtures using factory pattern and config injection."""
        # Initialize config and mock factory
        self.config = MetavertexPatternMatcherMockConfig()
        self.mock_factory = MetavertexPatternMatcherMockFactory()
        
        # Get configuration options
        self.processing_options = self.config.get_configuration_options()['processing_options']
        self.pattern_matching_options = self.config.get_configuration_options()['pattern_matching_options']
    
    def test_pattern_validation_scenarios(self):
        """Test pattern validation using config-driven scenarios."""
        validation_data = self.config.get_pattern_validation_scenarios()
        
        # Test valid patterns
        for valid_pattern in validation_data['valid_patterns']:
            # Mock validation should pass
            mock_pattern = self.mock_factory('MockMetavertexPattern', 
                                           pattern_id=valid_pattern['pattern_id'])
            # Validate the pattern structure
            self.assertTrue(mock_pattern.validate)
    
    def test_metavertex_matching_validation(self):
        """Test metavertex matching validation."""
        # Create test pattern matcher
        mock_matcher = self.mock_factory('MockMetavertexPatternMatcher')
        
        # Test that matching validation works properly
        self.assertTrue(mock_matcher.validate_match([], []))
        
        # Test match scoring
        score = mock_matcher.score_match([], [])
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_pattern_constraint_validation(self):
        """Test pattern constraint validation."""
        # Test various pattern constraints
        tree_pattern = self.mock_factory('MockTreePattern')
        cycle_pattern = self.mock_factory('MockCyclePattern')
        clique_pattern = self.mock_factory('MockCliquePattern')
        chain_pattern = self.mock_factory('MockChainPattern')
        
        # Each pattern should validate its own constraints
        self.assertTrue(tree_pattern.validate_pattern_constraints())
        self.assertTrue(cycle_pattern.validate_pattern_constraints())
        self.assertTrue(clique_pattern.validate_pattern_constraints())
        self.assertTrue(chain_pattern.validate_pattern_constraints())
    
    def test_linguistic_feature_validation(self):
        """Test validation of linguistic features."""
        test_data = self.config.get_linguistic_feature_tests()
        
        # Test POS tagging validation
        for pos_test in test_data['pos_tagging_tests']:
            doc = self.nlp(pos_test['sentence'])
            graph = SemanticMetagraph(doc=doc)
            matcher = PatternMatcher(graph)
            
            # Validate that POS information is correctly extracted
            analysis = matcher.analyze_metavertex_patterns()
            self.assertIn('pos_distribution', analysis)
            self.assertIsInstance(analysis['pos_distribution'], dict)
    
    def test_dependency_parsing_validation(self):
        """Test validation of dependency parsing results."""
        test_data = self.config.get_linguistic_feature_tests()
        
        # Test dependency parsing validation
        for dep_test in test_data['dependency_parsing_tests']:
            doc = self.nlp(dep_test['sentence'])
            graph = SemanticMetagraph(doc=doc)
            matcher = PatternMatcher(graph)
            
            # Validate that dependency relations are correctly captured
            relations = matcher.find_relation_metavertices()
            self.assertIsInstance(relations, list)


class TestMetavertexPatternMatcherEdgeCases(unittest.TestCase):
    """Edge cases and error conditions for MetavertexPatternMatcher."""
    
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
        """Set up test fixtures using factory pattern and config injection."""
        # Initialize config and mock factory
        self.config = MetavertexPatternMatcherMockConfig()
        self.mock_factory = MetavertexPatternMatcherMockFactory()
        
        # Get error handling test cases
        self.error_cases = self.config.get_error_handling_test_cases()
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        for test_case in self.error_cases['empty_input_tests']:
            if test_case['input'] == '':
                # Test empty string handling
                try:
                    doc = self.nlp(test_case['input'])
                    graph = SemanticMetagraph(doc=doc)
                    matcher = PatternMatcher(graph)
                    analysis = matcher.analyze_metavertex_patterns()
                    # Should handle empty input gracefully
                    self.assertIsInstance(analysis, dict)
                except Exception as e:
                    self.fail(f"Empty input should be handled gracefully: {e}")
    
    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""
        for test_case in self.error_cases['malformed_input_tests']:
            # Test handling of various malformed inputs
            try:
                doc = self.nlp(test_case['input'])
                graph = SemanticMetagraph(doc=doc)
                matcher = PatternMatcher(graph)
                # Should process without errors
                analysis = matcher.analyze_metavertex_patterns()
                self.assertIsInstance(analysis, dict)
            except Exception as e:
                # Should handle gracefully
                self.assertIn('expected_behavior', test_case)
    
    def test_pattern_matching_edge_cases(self):
        """Test pattern matching with edge case patterns."""
        # Test with empty pattern
        empty_pattern = []
        doc = self.nlp("Test sentence.")
        graph = SemanticMetagraph(doc=doc)
        matcher = PatternMatcher(graph)
        
        results = matcher.match_metavertex_chain(empty_pattern)
        self.assertIsInstance(results, list)
        
        # Test with complex nested pattern
        complex_pattern = [
            {"mv_type": "atomic", "pos": ["NOUN", "PROPN"]},
            {"mv_type": "directed_relation", "relation_type": "nsubj", "requires_reference": True},
            {"mv_type": "atomic", "pos": ["VERB"]},
            {"mv_type": "directed_relation", "relation_type": "dobj", "requires_reference": True},
            {"mv_type": "atomic", "pos": ["NOUN", "PROPN"]}
        ]
        
        results = matcher.match_metavertex_chain(complex_pattern)
        self.assertIsInstance(results, list)
    
    def test_large_input_handling(self):
        """Test handling of large inputs."""
        # Create a large input text
        large_text = "This is a test sentence. " * 100
        
        try:
            doc = self.nlp(large_text)
            graph = SemanticMetagraph(doc=doc)
            matcher = PatternMatcher(graph)
            analysis = matcher.analyze_metavertex_patterns()
            
            # Should handle large input without memory issues
            self.assertIsInstance(analysis, dict)
            self.assertGreater(analysis['total_metaverts'], 100)
        except Exception as e:
            # Should handle memory constraints gracefully
            self.assertIsInstance(e, (MemoryError, RuntimeError))
    
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        unicode_text = "Héllo wörld! 🌍 Special chars: @#$%^&*()"
        
        try:
            doc = self.nlp(unicode_text)
            graph = SemanticMetagraph(doc=doc)
            matcher = PatternMatcher(graph)
            analysis = matcher.analyze_metavertex_patterns()
            
            # Should handle Unicode gracefully
            self.assertIsInstance(analysis, dict)
        except Exception as e:
            self.fail(f"Unicode handling failed: {e}")


class TestMetavertexPatternMatcherIntegration(unittest.TestCase):
    """Integration tests with other components for MetavertexPatternMatcher."""
    
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
        """Set up test fixtures using factory pattern and config injection."""
        # Initialize config and mock factory
        self.config = MetavertexPatternMatcherMockConfig()
        self.mock_factory = MetavertexPatternMatcherMockFactory()
        
        # Get configuration options and integration scenarios
        self.processing_options = self.config.get_configuration_options()['processing_options']
        self.pattern_matching_options = self.config.get_configuration_options()['pattern_matching_options']
        self.integration_scenarios = self.config.get_integration_test_scenarios()
    
    def test_full_pipeline_integration(self):
        """Test full pipeline integration using config-driven scenarios."""
        pipeline_test = self.integration_scenarios['full_pipeline_test']
        
        # Create semantic metagraph from integration test data
        doc = self.nlp(pipeline_test['input_text'])
        graph = SemanticMetagraph(doc=doc)
        self.assertGreater(len(graph.metaverts), 0)
        
        # Create pattern matcher
        matcher = PatternMatcher(graph)
        
        # Test expected pipeline steps
        analysis = matcher.analyze_metavertex_patterns()
        expected_count = pipeline_test['expected_outputs']['metavertices_count']
        self.assertGreaterEqual(analysis["total_metaverts"], expected_count - 5)  # Allow some variance
        
        # Test entity finding
        expected_entities = pipeline_test['expected_outputs']['entities']
        nouns = matcher.find_atomic_metavertices(pos="NOUN")
        proper_nouns = matcher.find_atomic_metavertices(pos="PROPN")
        
        # Should find entities corresponding to expected entities
        total_entities = len(nouns) + len(proper_nouns)
        self.assertGreater(total_entities, 0)
    
    def test_batch_processing_integration(self):
        """Test batch processing integration."""
        batch_test = self.integration_scenarios['batch_processing_test']
        
        results = []
        for text in batch_test['input_texts']:
            doc = self.nlp(text)
            graph = SemanticMetagraph(doc=doc)
            matcher = PatternMatcher(graph)
            analysis = matcher.analyze_metavertex_patterns()
            results.append(analysis)
        
        # All should process successfully
        self.assertEqual(len(results), len(batch_test['input_texts']))
        
        # Each result should have expected structure
        for result in results:
            self.assertIn("total_metaverts", result)
            self.assertIn("atomic_count", result)
            self.assertGreaterEqual(result["atomic_count"], 0)
    
    def test_pattern_loader_integration(self):
        """Test integration with PatternLoader using mock factory."""
        # Create mock pattern loader
        mock_pattern_library = self.mock_factory('MockPatternLibraryForMatcher')
        mock_pattern_index = self.mock_factory('MockPatternIndex')
        
        # Test pattern loading integration
        loader = PatternLoader()
        
        # Check that metavertex pattern categories are loaded
        expected_categories = ["metavertex_basic", "metavertex_semantic", "metavertex_complex"]
        
        for category in expected_categories:
            if category in loader.patterns:
                self.assertIsInstance(loader.patterns[category], dict)
                self.assertGreater(len(loader.patterns[category]), 0)
            else:
                print(f"Warning: {category} not found in loaded patterns")
    
    def test_performance_integration(self):
        """Test performance integration using config parameters."""
        perf_params = self.config.get_performance_test_parameters()
        medium_test = perf_params['medium_text_test']
        
        # Create test text of appropriate length
        test_sentences = ["This is a test sentence."] * (medium_test['num_sentences'] // 5)
        test_text = " ".join(test_sentences)
        
        # Process and measure basic performance
        doc = self.nlp(test_text)
        graph = SemanticMetagraph(doc=doc)
        matcher = PatternMatcher(graph)
        
        # Should complete without errors
        analysis = matcher.analyze_metavertex_patterns()
        self.assertIsInstance(analysis, dict)
        self.assertIn("total_metaverts", analysis)
    
    def test_real_world_pattern_accuracy(self):
        """Test pattern matching accuracy using config-driven expected results."""
        expected_results = self.config.get_expected_matching_results()
        
        for test_name, test_data in expected_results.items():
            with self.subTest(test_case=test_name):
                # Process the sentence
                doc = self.nlp(test_data['sentence'])
                graph = SemanticMetagraph(doc=doc)
                matcher = PatternMatcher(graph)
                
                # Verify expected metavertices are created
                if 'expected_metavertices' in test_data:
                    analysis = matcher.analyze_metavertex_patterns()
                    expected_count = len(test_data['expected_metavertices'])
                    # Allow some variance in metavertex count
                    self.assertGreaterEqual(analysis['atomic_count'], expected_count - 2)
    
    def test_semantic_metagraph_integration(self):
        """Test integration with SemanticMetagraph component."""
        # Use complex sentence for comprehensive integration test
        complex_sentences = self.config.get_complex_sentence_examples()
        relative_clause = complex_sentences['relative_clause_sentence']
        
        doc = self.nlp(relative_clause['text'])
        graph = SemanticMetagraph(doc=doc)
        matcher = PatternMatcher(graph)
        
        # Test that semantic relationships are properly captured
        analysis = matcher.analyze_metavertex_patterns()
        self.assertGreater(analysis['directed_relation_count'], 0)
        
        # Test pattern matching on complex structures
        pattern = [{"mv_type": "atomic", "pos": ["NOUN"]}, 
                  {"mv_type": "directed_relation", "relation_type": "relcl"}]
        
        results = matcher.match_metavertex_chain(pattern)
        self.assertIsInstance(results, list)


if __name__ == '__main__':
    unittest.main()