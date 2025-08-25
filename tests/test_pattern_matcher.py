import unittest
from unittest.mock import patch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.PatternMatcher import PatternMatcher
from tests.mocks.pattern_matcher_mocks import PatternMatcherMockFactory
from tests.config.pattern_matcher_config import PatternMatcherMockConfig


class TestPatternMatcher(unittest.TestCase):
    """Test the PatternMatcher class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize config
        self.config = PatternMatcherMockConfig()
        
        # Initialize mock factory from config
        self.mock_factory = PatternMatcherMockFactory()
        
        # Get mock semantic graph structure from config
        graph_structures = self.config.get_mock_semantic_graph_structures()
        simple_graph = graph_structures['simple_svo_graph']
        
        # Create semantic metagraph using factory
        self.mock_semantic_graph = self.mock_factory('MockSemanticGraphForPattern')
        self.mock_semantic_graph.metaverts = {
            0: ("cat", {"pos": "NOUN", "text": "cat"}),
            1: ("chases", {"pos": "VERB", "text": "chases"}),
            2: ((0, 1), {"relation": "subject"})
        }
        
        # Get pattern loader patterns from config
        pattern_data = self.config.get_pattern_loader_test_patterns()
        
        # Create pattern loader using factory
        self.mock_pattern_loader = self.mock_factory('MockPatternLoaderForPattern')
        self.mock_pattern_loader.patterns = {
            "test_category": {
                "noun_pattern": {
                    "description": "Simple noun pattern",
                    "pattern": [{"pos": {"NOUN"}}]
                },
                "verb_pattern": {
                    "description": "Simple verb pattern", 
                    "pattern": [{"pos": {"VERB"}}]
                }
            }
        }
        
        self.pattern_matcher = PatternMatcher(
            semantic_graph=self.mock_semantic_graph,
            pattern_loader=self.mock_pattern_loader
        )

    def test_initialization_with_pattern_loader(self):
        """Test PatternMatcher initialization with pattern loader"""
        matcher = PatternMatcher(
            semantic_graph=self.mock_semantic_graph,
            pattern_loader=self.mock_pattern_loader
        )
        
        self.assertEqual(matcher.semantic_graph, self.mock_semantic_graph)
        self.assertEqual(matcher.pattern_loader, self.mock_pattern_loader)
        self.assertTrue(matcher.use_metavertex_matching)

    def test_initialization_without_pattern_loader(self):
        """Test PatternMatcher initialization without pattern loader"""
        with patch('smied.PatternMatcher.PatternLoader') as mock_pl_class:
            mock_pl_instance = self.mock_factory('MockPatternLoaderForPattern')
            mock_pl_class.return_value = mock_pl_instance
            
            matcher = PatternMatcher(semantic_graph=self.mock_semantic_graph)
            
            self.assertEqual(matcher.semantic_graph, self.mock_semantic_graph)
            self.assertEqual(matcher.pattern_loader, mock_pl_instance)
            mock_pl_class.assert_called_once()

    def test_add_pattern_delegates_to_loader(self):
        """Test add_pattern delegates to pattern loader"""
        test_pattern = [{"text": "test"}]
        
        self.pattern_matcher.add_pattern(
            name="new_pattern",
            pattern=test_pattern,
            description="Test description",
            category="test_cat"
        )
        
        self.mock_pattern_loader.add_pattern.assert_called_once_with(
            name="new_pattern",
            pattern=test_pattern,
            description="Test description",
            category="test_cat"
        )

    def test_metavertex_matches_atomic_string(self):
        """Test metavertex_matches with atomic string metavertex"""
        # Test matching atomic metavertex (index 0: "cat")
        pattern_attrs = {"text": "cat", "pos": "NOUN"}
        
        result = self.pattern_matcher.metavertex_matches(0, pattern_attrs)
        
        self.assertTrue(result)

    def test_metavertex_matches_atomic_string_no_match(self):
        """Test metavertex_matches with non-matching atomic metavertex"""
        pattern_attrs = {"text": "dog", "pos": "NOUN"}
        
        result = self.pattern_matcher.metavertex_matches(0, pattern_attrs)
        
        self.assertFalse(result)

    def test_metavertex_matches_directed_relation(self):
        """Test metavertex_matches with directed relation metavertex"""
        # Test matching directed relation (index 2: ((0, 1), {"relation": "subject"}))
        pattern_attrs = {"is_directed_relation": True, "relation": "subject"}
        
        result = self.pattern_matcher.metavertex_matches(2, pattern_attrs)
        
        self.assertTrue(result)

    def test_metavertex_matches_directed_relation_source_target(self):
        """Test metavertex_matches checking source and target indices"""
        pattern_attrs = {"source_idx": 0, "target_idx": 1}
        
        result = self.pattern_matcher.metavertex_matches(2, pattern_attrs)
        
        self.assertTrue(result)

    def test_metavertex_matches_nonexistent_index(self):
        """Test metavertex_matches with non-existent metavertex index"""
        pattern_attrs = {"text": "anything"}
        
        result = self.pattern_matcher.metavertex_matches(999, pattern_attrs)
        
        self.assertFalse(result)

    def test_metavertex_matches_undirected_relation(self):
        """Test metavertex_matches with undirected relation"""
        # Add undirected relation to mock graph
        self.mock_semantic_graph.metaverts[3] = ([0, 1, 2], {"relation": "group"})
        
        pattern_attrs = {"is_undirected_relation": True, "relation": "group"}
        
        result = self.pattern_matcher.metavertex_matches(3, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_basic(self):
        """Test basic node_matches functionality"""
        node_attrs = {"text": "cat", "pos": "NOUN"}
        pattern_attrs = {"text": "cat", "pos": "NOUN"}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_with_sets(self):
        """Test node_matches with set values in patterns"""
        node_attrs = {"pos": "NOUN", "text": "cat"}
        pattern_attrs = {"pos": {"NOUN", "PROPN"}}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_with_sets_no_match(self):
        """Test node_matches with set values that don't match"""
        node_attrs = {"pos": "VERB", "text": "runs"}
        pattern_attrs = {"pos": {"NOUN", "PROPN"}}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertFalse(result)

    def test_node_matches_node_id_pattern(self):
        """Test node_matches with node_id_pattern"""
        node_attrs = {"node_id": "node_123", "text": "test"}
        pattern_attrs = {"node_id_pattern": "node_"}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_semantic_type(self):
        """Test node_matches with semantic_type"""
        node_attrs = {"semantic_type": "entity", "text": "test"}
        pattern_attrs = {"semantic_type": {"entity", "concept"}}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_lemma(self):
        """Test node_matches with lemma matching"""
        node_attrs = {"lemma": "run", "text": "running"}
        pattern_attrs = {"lemma": {"run", "walk"}}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_mv_type_atomic(self):
        """Test node_matches with mv_type for atomic"""
        node_attrs = {"is_atomic": True, "text": "cat"}
        pattern_attrs = {"mv_type": "atomic"}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_mv_type_directed_relation(self):
        """Test node_matches with mv_type for directed_relation"""
        node_attrs = {"is_directed_relation": True, "relation": "subject"}
        pattern_attrs = {"mv_type": "directed_relation"}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_references_mv(self):
        """Test node_matches with references_mv for directed relation"""
        node_attrs = {
            "is_directed_relation": True,
            "source_idx": 0,
            "target_idx": 1
        }
        pattern_attrs = {"references_mv": 0}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_references_mv_undirected(self):
        """Test node_matches with references_mv for undirected relation"""
        node_attrs = {
            "is_undirected_relation": True,
            "component_indices": [0, 1, 2]
        }
        pattern_attrs = {"references_mv": 1}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_references_mv_no_relation(self):
        """Test node_matches with references_mv for non-relation"""
        node_attrs = {"is_atomic": True, "text": "cat"}
        pattern_attrs = {"references_mv": 0}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertFalse(result)

    def test_edge_matches_basic(self):
        """Test basic edge_matches functionality"""
        edge_attrs = {"relation": "subject", "weight": 1.0}
        pattern_attrs = {"relation": "subject"}
        
        result = self.pattern_matcher.edge_matches(edge_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_edge_matches_with_set(self):
        """Test edge_matches with set pattern"""
        edge_attrs = {"relation": "subject"}
        pattern_attrs = {"relation": {"subject", "object"}}
        
        result = self.pattern_matcher.edge_matches(edge_attrs, pattern_attrs)
        
        self.assertTrue(result)

    def test_edge_matches_no_match(self):
        """Test edge_matches with no match"""
        edge_attrs = {"relation": "object"}
        pattern_attrs = {"relation": "subject"}
        
        result = self.pattern_matcher.edge_matches(edge_attrs, pattern_attrs)
        
        self.assertFalse(result)

    def test_match_metavertex_chain_empty_query(self):
        """Test match_metavertex_chain with empty query"""
        result = self.pattern_matcher.match_metavertex_chain([])
        
        self.assertEqual(result, [])

    def test_match_metavertex_chain_single_node(self):
        """Test match_metavertex_chain with single node query"""
        query = [{"pos": "NOUN"}]
        
        with patch.object(self.pattern_matcher, 'metavertex_matches') as mock_matches:
            mock_matches.side_effect = lambda idx, pattern: idx == 0  # Only index 0 matches
            
            result = self.pattern_matcher.match_metavertex_chain(query)
            
            self.assertEqual(result, [[0]])

    def test_match_metavertex_chain_multi_node(self):
        """Test match_metavertex_chain with multi-node query"""
        query = [{"pos": "NOUN"}, {"pos": "VERB"}]
        
        with patch.object(self.pattern_matcher, 'metavertex_matches') as mock_matches, \
             patch.object(self.pattern_matcher, 'is_metavertex_related') as mock_related:
            
            # First pattern matches index 0, second pattern matches index 1
            mock_matches.side_effect = lambda idx, pattern: (
                (idx == 0 and pattern.get("pos") == "NOUN") or
                (idx == 1 and pattern.get("pos") == "VERB")
            )
            mock_related.return_value = True
            
            result = self.pattern_matcher.match_metavertex_chain(query)
            
            self.assertIn([0, 1], result)

    def test_is_metavertex_related_directed_relation(self):
        """Test is_metavertex_related with directed relation"""
        # mv_idx2 (index 2) has directed relation ((0, 1), {...})
        result = self.pattern_matcher.is_metavertex_related(0, 2, {})
        
        self.assertTrue(result)

    def test_is_metavertex_related_undirected_relation(self):
        """Test is_metavertex_related with undirected relation"""
        # Add undirected relation
        self.mock_semantic_graph.metaverts[3] = ([0, 1], {"relation": "group"})
        
        result = self.pattern_matcher.is_metavertex_related(0, 3, {})
        
        self.assertTrue(result)

    def test_is_metavertex_related_requires_reference(self):
        """Test is_metavertex_related with requires_reference pattern"""
        pattern = {"requires_reference": True}
        
        result = self.pattern_matcher.is_metavertex_related(0, 1, pattern)
        
        self.assertTrue(result)  # mv_idx2 > mv_idx1

    def test_is_metavertex_related_no_relation(self):
        """Test is_metavertex_related with no specific relation"""
        result = self.pattern_matcher.is_metavertex_related(0, 1, {})
        
        self.assertTrue(result)  # Default: allow any relationship

    def test_match_chain_metavertex_enabled(self):
        """Test match_chain with metavertex matching enabled"""
        query = [{"pos": "NOUN"}]
        
        with patch.object(self.pattern_matcher, 'match_metavertex_chain') as mock_mv_match:
            mock_mv_match.return_value = [[0]]
            
            result = self.pattern_matcher.match_chain(query)
            
            mock_mv_match.assert_called_once_with(query)
            self.assertEqual(result, [[0]])

    def test_match_chain_networkx_fallback(self):
        """Test match_chain with NetworkX fallback"""
        # Disable metavertex matching
        self.pattern_matcher.use_metavertex_matching = False
        
        # Mock NetworkX graph
        mock_nx_graph = self.mock_factory('MockNetworkXForPattern')
        mock_nx_graph.nodes.return_value = ["node1", "node2"]
        mock_nx_graph.nodes.__getitem__ = lambda self, node: {"label": node, "pos": "NOUN"}
        mock_nx_graph.neighbors.return_value = []
        
        self.mock_semantic_graph.to_nx.return_value = mock_nx_graph
        
        query = [{"pos": "NOUN"}]
        
        with patch.object(self.pattern_matcher, 'node_matches') as mock_node_matches:
            mock_node_matches.return_value = True
            
            result = self.pattern_matcher.match_chain(query)
            
            self.assertIsInstance(result, list)

    def test_match_chain_invalid_query_length(self):
        """Test match_chain with invalid query length"""
        # Disable metavertex matching to test NetworkX path
        self.pattern_matcher.use_metavertex_matching = False
        
        # Even length query should raise error
        query = [{"pos": "NOUN"}, {"pos": "VERB"}]  # Even length
        
        self.mock_semantic_graph.to_nx.return_value = self.mock_factory('MockNetworkXForPattern')
        
        with self.assertRaises(ValueError) as context:
            self.pattern_matcher.match_chain(query)
        
        self.assertIn("odd length", str(context.exception))

    def test_get_pattern_summary(self):
        """Test get_pattern_summary method"""
        # Test without mocking __call__ to test actual behavior
        with patch('builtins.print') as mock_print:
            result = self.pattern_matcher.get_pattern_summary()
        
        # Expected summary should contain match counts, not full matches
        # Based on the mock setup: "cat" matches noun_pattern, "chases" matches verb_pattern
        expected_summary = {
            "test_category": {
                "noun_pattern": 1,  # 1 match: "cat" with pos="NOUN"
                "verb_pattern": 1   # 1 match: "chases" with pos="VERB"
            }
        }
        
        self.assertEqual(result, expected_summary)
        mock_print.assert_called()  # Should print pattern information

    def test_match_metavertex_pattern(self):
        """Test match_metavertex_pattern method"""
        pattern_dict = {
            "description": "Test pattern",
            "pattern": [{"pos": "NOUN"}]
        }
        
        with patch.object(self.pattern_matcher, 'match_metavertex_chain') as mock_match:
            mock_match.return_value = [[0]]
            
            result = self.pattern_matcher.match_metavertex_pattern(pattern_dict)
            
            mock_match.assert_called_once_with([{"pos": "NOUN"}])
            self.assertEqual(result, [[0]])

    def test_match_metavertex_pattern_no_pattern_key(self):
        """Test match_metavertex_pattern with missing pattern key"""
        pattern_dict = {"description": "No pattern"}
        
        result = self.pattern_matcher.match_metavertex_pattern(pattern_dict)
        
        self.assertEqual(result, [])

    def test_get_metavertex_context(self):
        """Test get_metavertex_context method"""
        mv_indices = [0, 1, 2]
        
        result = self.pattern_matcher.get_metavertex_context(mv_indices)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["indices"], mv_indices)
        self.assertIn("metaverts", result)
        self.assertIn("summary", result)
        
        # Check that summary includes text from atomic metaverts
        self.assertIn("cat", result["summary"])
        self.assertIn("chases", result["summary"])

    def test_call_method_all_patterns(self):
        """Test __call__ method matching all patterns"""
        with patch.object(self.pattern_matcher, '__call__') as mock_call:
            # Mock recursive calls
            mock_call.side_effect = [
                {"noun_pattern": [], "verb_pattern": []},  # First recursive call
                {"test_category": {"noun_pattern": [], "verb_pattern": []}}  # Final result
            ]
            
            # Call without arguments (original call)
            mock_call.side_effect = None
            mock_call.return_value = {"test_category": {"noun_pattern": [], "verb_pattern": []}}
            
            result = self.pattern_matcher()
            
            self.assertIsInstance(result, dict)
            self.assertIn("test_category", result)

    def test_call_method_category_only(self):
        """Test __call__ method with category only"""
        with patch.object(self.pattern_matcher, '__call__') as mock_call:
            # Mock recursive calls for each pattern in category
            mock_call.side_effect = [
                [],  # noun_pattern result
                [],  # verb_pattern result
                {"noun_pattern": [], "verb_pattern": []}  # Final aggregated result
            ]
            
            result = self.pattern_matcher("test_category")
            
            self.assertIsInstance(result, dict)

    def test_call_method_specific_pattern_metavertex(self):
        """Test __call__ method with specific pattern using metavertex matching"""
        with patch.object(self.pattern_matcher, 'match_metavertex_pattern') as mock_match, \
             patch.object(self.pattern_matcher, 'get_metavertex_context') as mock_context:
            
            mock_match.return_value = [[0], [1]]
            mock_context.side_effect = [
                {"indices": [0], "summary": "cat"},
                {"indices": [1], "summary": "runs"}
            ]
            
            result = self.pattern_matcher("test_category", "noun_pattern")
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)

    def test_call_method_specific_pattern_networkx(self):
        """Test __call__ method with specific pattern using NetworkX matching"""
        self.pattern_matcher.use_metavertex_matching = False
        
        with patch.object(self.pattern_matcher, 'match_chain') as mock_match:
            mock_match.return_value = [["node1"], ["node2"]]
            
            result = self.pattern_matcher("test_category", "noun_pattern")
            
            self.assertEqual(result, [["node1"], ["node2"]])

    def test_call_method_invalid_category(self):
        """Test __call__ method with invalid category"""
        with self.assertRaises(KeyError) as context:
            self.pattern_matcher("nonexistent_category")
        
        self.assertIn("does not exist", str(context.exception))

    def test_call_method_invalid_pattern(self):
        """Test __call__ method with invalid pattern name"""
        with self.assertRaises(KeyError) as context:
            self.pattern_matcher("test_category", "nonexistent_pattern")
        
        self.assertIn("does not exist", str(context.exception))

    def test_find_atomic_metavertices(self):
        """Test find_atomic_metavertices method"""
        result = self.pattern_matcher.find_atomic_metavertices(pos="NOUN")
        
        self.assertIn(0, result)  # "cat" with pos="NOUN"
        self.assertNotIn(1, result)  # "runs" with pos="VERB"
        self.assertNotIn(2, result)  # Relation metavertex

    def test_find_atomic_metavertices_text_filter(self):
        """Test find_atomic_metavertices with text filter"""
        result = self.pattern_matcher.find_atomic_metavertices(text="cat")
        
        self.assertEqual(result, [0])

    def test_find_atomic_metavertices_no_matches(self):
        """Test find_atomic_metavertices with no matches"""
        result = self.pattern_matcher.find_atomic_metavertices(pos="ADJ")
        
        self.assertEqual(result, [])

    def test_find_relation_metavertices(self):
        """Test find_relation_metavertices method"""
        result = self.pattern_matcher.find_relation_metavertices()
        
        self.assertIn(2, result)  # Relation metavertex
        self.assertNotIn(0, result)  # Atomic metavertex
        self.assertNotIn(1, result)  # Atomic metavertex

    def test_find_relation_metavertices_specific_type(self):
        """Test find_relation_metavertices with specific relation type"""
        result = self.pattern_matcher.find_relation_metavertices(relation_type="subject")
        
        self.assertEqual(result, [2])

    def test_find_relation_metavertices_no_matches(self):
        """Test find_relation_metavertices with no matching relation type"""
        result = self.pattern_matcher.find_relation_metavertices(relation_type="object")
        
        self.assertEqual(result, [])

    def test_get_metavertex_chain(self):
        """Test get_metavertex_chain method"""
        result = self.pattern_matcher.get_metavertex_chain(0, max_depth=2)
        
        self.assertIsInstance(result, list)
        # Should find chains starting from index 0

    def test_get_metavertex_chain_max_depth_limit(self):
        """Test get_metavertex_chain respects max_depth"""
        result = self.pattern_matcher.get_metavertex_chain(0, max_depth=0)
        
        self.assertEqual(result, [])

    def test_analyze_metavertex_patterns(self):
        """Test analyze_metavertex_patterns method"""
        result = self.pattern_matcher.analyze_metavertex_patterns()
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_metaverts", result)
        self.assertIn("atomic_count", result)
        self.assertIn("directed_relation_count", result)
        self.assertIn("undirected_relation_count", result)
        self.assertIn("relation_types", result)
        self.assertIn("pos_distribution", result)
        
        # Check specific counts based on our test data
        self.assertEqual(result["total_metaverts"], 3)
        self.assertEqual(result["atomic_count"], 2)  # "cat" and "runs"
        self.assertEqual(result["directed_relation_count"], 1)  # The subject relation
        
        # Check distributions
        self.assertIn("NOUN", result["pos_distribution"])
        self.assertIn("VERB", result["pos_distribution"])
        self.assertIn("subject", result["relation_types"])


class TestPatternMatcherValidation(unittest.TestCase):
    """Validation and constraint tests for PatternMatcher"""
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection"""
        # Initialize config
        self.config = PatternMatcherMockConfig()
        
        # Initialize mock factory from config
        self.mock_factory = PatternMatcherMockFactory()
        
        # Get validation criteria from config
        self.validation_criteria = self.config.get_validation_criteria()
        
        # Create semantic graph for validation testing
        self.mock_semantic_graph = self.mock_factory('MockSemanticGraphForPattern')
        graph_structures = self.config.get_mock_semantic_graph_structures()
        complex_graph = graph_structures['complex_sentence_graph']
        
        # Use config-driven graph structure
        self.mock_semantic_graph.metaverts = {
            0: ("John", {"pos": "PROPN", "text": "John", "ent_type": "PERSON"}),
            1: ("runs", {"pos": "VERB", "text": "runs"}),
            2: ("fast", {"pos": "ADV", "text": "fast"}),
            3: ((0, 1), {"relation": "subject"}),
            4: ((1, 2), {"relation": "modifier"})
        }
        
        # Get patterns for validation testing from config
        pattern_data = self.config.get_pattern_loader_test_patterns()
        semantic_patterns = pattern_data['semantic_patterns']
        
        self.mock_pattern_loader = self.mock_factory('MockPatternLoaderForPattern')
        self.mock_pattern_loader.patterns = {
            "validation_category": {
                "agent_action": {
                    "description": semantic_patterns[0]['description'],
                    "pattern": [{"pos": "NOUN"}, {"pos": "VERB"}]
                }
            }
        }
        
        self.pattern_matcher = PatternMatcher(
            semantic_graph=self.mock_semantic_graph,
            pattern_loader=self.mock_pattern_loader
        )
    
    def test_pattern_validation_basic(self):
        """Test basic pattern validation functionality"""
        # Test pattern structure validation
        valid_pattern = {"pattern": [{"pos": "NOUN"}]}
        result = self.pattern_matcher.match_metavertex_pattern(valid_pattern)
        self.assertIsInstance(result, list)
    
    def test_pattern_validation_constraints(self):
        """Test pattern validation with constraints"""
        # Use validation criteria from config
        structural_validation = self.validation_criteria['structural_validation']
        self.assertTrue(structural_validation['vertex_count_match'])
        self.assertTrue(structural_validation['edge_count_match'])
    
    def test_match_quality_validation(self):
        """Test match quality validation using config criteria"""
        quality_metrics = self.validation_criteria['match_quality_metrics']
        self.assertIn('precision', quality_metrics)
        self.assertIn('recall', quality_metrics)
        self.assertIn('f1_score', quality_metrics)
    
    def test_semantic_validation(self):
        """Test semantic validation using config criteria"""
        semantic_validation = self.validation_criteria['semantic_validation']
        self.assertTrue(semantic_validation['role_consistency'])
        self.assertTrue(semantic_validation['type_compatibility'])

    def test_pattern_matching_algorithm_validation(self):
        """Test pattern matching algorithm validation using config"""
        # Get pattern matching algorithms from config
        algorithms = self.config.get_pattern_matching_algorithms()
        
        # Test exact matching configuration
        exact_matching = algorithms['exact_matching']
        self.assertEqual(exact_matching['algorithm'], 'exact_match')
        self.assertIn('case_sensitive', exact_matching['parameters'])
        self.assertIn('match_threshold', exact_matching['parameters'])
        
        # Test fuzzy matching configuration
        fuzzy_matching = algorithms['fuzzy_matching']
        self.assertEqual(fuzzy_matching['algorithm'], 'fuzzy_match')
        self.assertIn('similarity_threshold', fuzzy_matching['parameters'])

    def test_pos_tag_pattern_validation(self):
        """Test POS tag pattern validation using config data"""
        # Get POS tag patterns from config
        pos_patterns = self.config.get_pos_tag_patterns()
        
        # Test noun phrase patterns
        noun_phrase_patterns = pos_patterns['noun_phrase_patterns']
        self.assertIn(["DET", "ADJ", "NOUN"], noun_phrase_patterns)
        self.assertIn(["DET", "NOUN", "NOUN"], noun_phrase_patterns)
        
        # Test dependency relations
        dep_relations = pos_patterns['dependency_relations']
        core_relations = dep_relations['core_relations']
        self.assertIn('nsubj', core_relations)
        self.assertIn('dobj', core_relations)


class TestPatternMatcherEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize config
        self.config = PatternMatcherMockConfig()
        
        # Initialize mock factory from config
        self.mock_factory = PatternMatcherMockFactory()
        
        # Get edge case patterns from config
        self.edge_case_scenarios = self.config.get_edge_case_scenarios()
        
        # Create empty semantic graph for edge case testing
        self.mock_semantic_graph = self.mock_factory('MockSemanticGraphForPattern')
        self.mock_semantic_graph.metaverts = {}
        
        self.mock_pattern_loader = self.mock_factory('MockPatternLoaderForPattern')
        self.mock_pattern_loader.patterns = {}
        
        self.pattern_matcher = PatternMatcher(
            semantic_graph=self.mock_semantic_graph,
            pattern_loader=self.mock_pattern_loader
        )

    def test_metavertex_matches_malformed_metavertex(self):
        """Test metavertex_matches with malformed metavertex structure using config edge cases"""
        # Use edge case scenarios from config
        malformed_patterns = self.edge_case_scenarios['malformed_patterns']
        
        # Test metavertex with only content, no metadata
        self.mock_semantic_graph.metaverts[0] = ("test",)
        
        pattern_attrs = {"text": "test"}
        
        result = self.pattern_matcher.metavertex_matches(0, pattern_attrs)
        
        self.assertTrue(result)

    def test_node_matches_empty_pattern(self):
        """Test node_matches with empty pattern using config edge cases"""
        # Use empty inputs from config edge cases
        empty_inputs = self.edge_case_scenarios['empty_inputs']
        expected_behavior = empty_inputs['expected_behavior']
        
        node_attrs = {"text": "test", "pos": "NOUN"}
        pattern_attrs = {}
        
        result = self.pattern_matcher.node_matches(node_attrs, pattern_attrs)
        
        self.assertTrue(result)  # Empty pattern should match anything
        self.assertEqual(expected_behavior, 'return_empty_results')

    def test_edge_matches_empty_pattern(self):
        """Test edge_matches with empty pattern"""
        edge_attrs = {"relation": "test"}
        pattern_attrs = {}
        
        result = self.pattern_matcher.edge_matches(edge_attrs, pattern_attrs)
        
        self.assertTrue(result)  # Empty pattern should match anything

    def test_is_metavertex_related_missing_metavertices(self):
        """Test is_metavertex_related with missing metavertices"""
        result = self.pattern_matcher.is_metavertex_related(999, 998, {})
        
        self.assertFalse(result)

    def test_match_metavertex_chain_empty_graph(self):
        """Test match_metavertex_chain with empty graph"""
        query = [{"pos": "NOUN"}]
        
        result = self.pattern_matcher.match_metavertex_chain(query)
        
        self.assertEqual(result, [])

    def test_get_metavertex_context_missing_indices(self):
        """Test get_metavertex_context with missing indices using config edge cases"""
        # Use edge case scenarios for complex nested structures
        complex_nested = self.edge_case_scenarios['complex_nested_structures']
        expected_behavior = complex_nested['expected_behavior']
        
        mv_indices = [999, 998]
        
        result = self.pattern_matcher.get_metavertex_context(mv_indices)
        
        self.assertEqual(result["indices"], mv_indices)
        self.assertEqual(result["metaverts"], [])
        self.assertEqual(result["summary"], "")
        self.assertEqual(expected_behavior, 'handle_gracefully')

    def test_find_atomic_metavertices_malformed_metaverts(self):
        """Test find_atomic_metavertices with malformed metaverts using config edge cases"""
        # Use malformed patterns from config
        malformed_patterns = self.edge_case_scenarios['malformed_patterns']
        
        # Test metavertex without metadata
        self.mock_semantic_graph.metaverts = {
            0: ("test",),  # No metadata tuple
            1: (123,),     # Non-string content
        }
        
        result = self.pattern_matcher.find_atomic_metavertices(text="test")
        
        self.assertEqual(result, [0])
        
        # Verify edge case handling expectations
        self.assertTrue(len(malformed_patterns) > 0)
        first_pattern = malformed_patterns[0]
        self.assertEqual(first_pattern['expected_error'], 'PatternValidationError')

    def test_find_relation_metavertices_no_metadata(self):
        """Test find_relation_metavertices with metaverts without metadata"""
        self.mock_semantic_graph.metaverts = {
            0: (((0, 1)),),  # Relation without metadata
            1: ("atomic",)   # Atomic (should be skipped)
        }
        
        result = self.pattern_matcher.find_relation_metavertices(relation_type="test")
        
        self.assertEqual(result, [])  # No metadata means no relation match

    def test_analyze_metavertex_patterns_complex_structures(self):
        """Test analyze_metavertex_patterns with complex metavertex structures"""
        self.mock_semantic_graph.metaverts = {
            0: ("word", {"pos": "NOUN"}),
            1: (((0, 5)), {"relation": "distant_ref"}),  # Forward reference
            2: ([0, 1], {"relation": "group"}),
            3: ({"unexpected": "structure"},),  # Unexpected structure
            4: ([],),  # Empty list
            5: ("target", {"pos": "VERB"})
        }
        
        result = self.pattern_matcher.analyze_metavertex_patterns()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["total_metaverts"], 6)
        self.assertGreaterEqual(result["atomic_count"], 2)  # At least "word" and "target"


class TestPatternMatcherIntegration(unittest.TestCase):
    """Integration tests for PatternMatcher with other components"""
    
    def setUp(self):
        """Set up realistic test scenario"""
        # Initialize config
        self.config = PatternMatcherMockConfig()
        
        # Initialize mock factory from config
        self.mock_factory = PatternMatcherMockFactory()
        
        # Get complex sentence graph from config
        graph_structures = self.config.get_mock_semantic_graph_structures()
        complex_graph = graph_structures['complex_sentence_graph']
        
        # Create realistic metavertex structure using config data
        self.mock_semantic_graph = self.mock_factory('MockSemanticGraphForPattern')
        self.mock_semantic_graph.metaverts = {
            0: ("The", {"pos": "DET", "text": "The"}),
            1: ("cat", {"pos": "NOUN", "text": "cat"}),
            2: ("runs", {"pos": "VERB", "text": "runs"}),
            3: ("fast", {"pos": "ADV", "text": "fast"}),
            4: ((0, 1), {"relation": "det"}),      # The -> cat
            5: ((1, 2), {"relation": "nsubj"}),   # cat -> runs (subject)
            6: ((2, 3), {"relation": "advmod"})   # runs -> fast (adverbial modifier)
        }
        
        # Get realistic patterns from config
        pattern_data = self.config.get_pattern_loader_test_patterns()
        self.integration_scenarios = self.config.get_integration_test_data()
        
        # Create realistic patterns
        self.mock_pattern_loader = self.mock_factory('MockPatternLoaderForPattern')
        self.mock_pattern_loader.patterns = {
            "syntactic": {
                "noun_phrase": {
                    "description": "Determiner + Noun",
                    "pattern": [
                        {"mv_type": "atomic", "pos": "DET"},
                        {"mv_type": "directed_relation", "relation": "det"}
                    ]
                },
                "subject_verb": {
                    "description": "Subject-Verb relation",
                    "pattern": [
                        {"mv_type": "atomic", "pos": "NOUN"},
                        {"mv_type": "directed_relation", "relation": "nsubj"}
                    ]
                }
            },
            "semantic": {
                "agent_action": {
                    "description": "Agent performing action",
                    "pattern": [
                        {"pos": "NOUN"},
                        {"pos": "VERB"}
                    ]
                }
            }
        }
        
        self.pattern_matcher = PatternMatcher(
            semantic_graph=self.mock_semantic_graph,
            pattern_loader=self.mock_pattern_loader
        )

    def test_realistic_pattern_matching_workflow(self):
        """Test complete pattern matching workflow with realistic data"""
        # Test finding noun phrases
        with patch.object(self.pattern_matcher, 'metavertex_matches') as mock_matches:
            # Mock matching logic for determiner-noun pattern
            def mock_match_logic(idx, pattern):
                mv = self.mock_semantic_graph.metaverts[idx]
                if pattern.get("mv_type") == "atomic" and pattern.get("pos") == "DET":
                    return idx == 0  # "The"
                elif pattern.get("mv_type") == "directed_relation" and pattern.get("relation") == "det":
                    return idx == 4  # det relation
                return False
            
            mock_matches.side_effect = mock_match_logic
            
            # Test matching determiner-noun pattern
            noun_phrase_pattern = self.mock_pattern_loader.patterns["syntactic"]["noun_phrase"]["pattern"]
            result = self.pattern_matcher.match_metavertex_chain(noun_phrase_pattern)
            
            # Should find the pattern starting with "The" (index 0)
            self.assertIsInstance(result, list)

    def test_pattern_analysis_comprehensive(self):
        """Test comprehensive pattern analysis"""
        analysis = self.pattern_matcher.analyze_metavertex_patterns()
        
        # Verify analysis results
        self.assertEqual(analysis["total_metaverts"], 7)
        self.assertEqual(analysis["atomic_count"], 4)  # The, cat, runs, fast
        self.assertEqual(analysis["directed_relation_count"], 3)  # det, nsubj, advmod
        self.assertEqual(analysis["undirected_relation_count"], 0)
        
        # Check POS distribution
        expected_pos = {"DET": 1, "NOUN": 1, "VERB": 1, "ADV": 1}
        self.assertEqual(analysis["pos_distribution"], expected_pos)
        
        # Check relation types
        expected_relations = {"det": 1, "nsubj": 1, "advmod": 1}
        self.assertEqual(analysis["relation_types"], expected_relations)

    def test_find_patterns_by_type(self):
        """Test finding specific types of patterns"""
        # Find all nouns
        nouns = self.pattern_matcher.find_atomic_metavertices(pos="NOUN")
        self.assertEqual(nouns, [1])  # "cat"
        
        # Find all verbs
        verbs = self.pattern_matcher.find_atomic_metavertices(pos="VERB")
        self.assertEqual(verbs, [2])  # "runs"
        
        # Find subject relations
        subject_rels = self.pattern_matcher.find_relation_metavertices(relation_type="nsubj")
        self.assertEqual(subject_rels, [5])  # subject relation

    def test_metavertex_chain_construction(self):
        """Test building metavertex chains from relationships"""
        # Start from "cat" (index 1) and find chains
        chains = self.pattern_matcher.get_metavertex_chain(1, max_depth=2)
        
        self.assertIsInstance(chains, list)
        # Should find chains that reference "cat"
        
        # Check that chains include relations that reference index 1
        for chain in chains:
            # Last element in chain should reference earlier elements
            if len(chain) > 1:
                last_idx = chain[-1]
                mv = self.mock_semantic_graph.metaverts[last_idx]
                # For directed relations, check if they reference index 1
                if isinstance(mv[0], tuple) and len(mv[0]) == 2:
                    self.assertTrue(1 in mv[0])

    def test_context_extraction(self):
        """Test extracting context from metavertex sequences"""
        # Test context for a meaningful sequence
        mv_sequence = [0, 1, 4]  # "The", "cat", det relation
        
        context = self.pattern_matcher.get_metavertex_context(mv_sequence)
        
        self.assertEqual(context["indices"], mv_sequence)
        self.assertEqual(len(context["metaverts"]), 3)
        
        # Summary should include text from atomic metaverts
        summary = context["summary"]
        self.assertIn("The", summary)
        self.assertIn("cat", summary)
        self.assertIn("det", summary)  # Relation should be included

    def test_pattern_matching_with_filters(self):
        """Test pattern matching with various filters and conditions"""
        # Test complex pattern matching scenario
        complex_pattern = [
            {"mv_type": "atomic", "pos": "NOUN"},
            {"mv_type": "directed_relation", "relation_type": {"nsubj", "dobj"}},
            {"mv_type": "atomic", "pos": "VERB"}
        ]
        
        # Mock the matching to simulate realistic results
        with patch.object(self.pattern_matcher, 'metavertex_matches') as mock_matches, \
             patch.object(self.pattern_matcher, 'is_metavertex_related') as mock_related:
            
            def match_logic(idx, pattern):
                mv = self.mock_semantic_graph.metaverts[idx]
                if pattern.get("mv_type") == "atomic":
                    if pattern.get("pos") == "NOUN" and idx == 1:  # cat
                        return True
                    elif pattern.get("pos") == "VERB" and idx == 2:  # runs
                        return True
                elif pattern.get("mv_type") == "directed_relation":
                    if idx == 5:  # nsubj relation
                        return True
                return False
            
            mock_matches.side_effect = match_logic
            mock_related.return_value = True
            
            result = self.pattern_matcher.match_metavertex_chain(complex_pattern)
            
            self.assertIsInstance(result, list)


if __name__ == '__main__':
    unittest.main()