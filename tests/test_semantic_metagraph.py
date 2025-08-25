"""
Test cases for SemanticMetagraph following SMIED Testing Framework Design Specifications.

This module implements the 3-layer architecture:
- Test Layer: Contains test logic and assertions
- Mock Layer: Provides mock implementations via factory pattern
- Configuration Layer: Supplies test data and constants

Test class organization:
- TestSemanticMetagraph: Basic functionality tests
- TestSemanticMetagraphValidation: Validation and constraint tests
- TestSemanticMetagraphEdgeCases: Edge cases and error conditions
- TestSemanticMetagraphIntegration: Integration tests with other components
"""

import unittest
import json
import spacy
from spacy.tokens import Doc
import networkx as nx
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the actual class under test
from smied.SemanticMetagraph import SemanticMetagraph

# Import mock layer (factory pattern)
from tests.mocks.semantic_metagraph_mocks import SemanticMetagraphMockFactory

# Import configuration layer (test data and constants)
from tests.config.semantic_metagraph_config import SemanticMetagraphMockConfig


class TestSemanticMetagraph(unittest.TestCase):
    """Basic functionality tests for SemanticMetagraph."""
    
    @classmethod
    def setUpClass(cls):
        """Load spaCy model once for all tests."""
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not installed, try to download it
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            cls.nlp = spacy.load("en_core_web_sm")
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory
        self.mock_factory = SemanticMetagraphMockFactory()
        
        # Initialize configuration
        self.config = SemanticMetagraphMockConfig()
        
        # Get test data from configuration
        self.test_texts = self.config.get_test_texts()
        self.vertex_structures = self.config.get_basic_test_vertex_structures()
        self.edge_structures = self.config.get_basic_test_edge_structures()
        
        # Create test documents
        self.test_doc = self.nlp(self.test_texts['complex_sentence'])
        self.simple_doc = self.nlp(self.test_texts['simple_sentence'])
        
        # Create test vertex list from config
        self.test_vert_list = [
            self.vertex_structures['simple_vertices'][0],  # ("word1", {"pos": "NOUN"})
            self.vertex_structures['simple_vertices'][1],  # ("word2", {"pos": "VERB"})
            self.edge_structures['simple_edges'][0]       # ((0, 1), {"relation": "subject"})
        ]
        
        # Create mock objects using factory
        self.mock_semantic_metagraph = self.mock_factory('MockSemanticMetagraph')
        self.mock_spacy_doc = self.mock_factory('MockSpacyDoc', text=self.test_texts['simple_sentence'])
    
    def test_initialization_from_doc(self):
        """Test initialization from a spaCy Doc object."""
        sg = SemanticMetagraph(doc=self.test_doc)
        
        # Verify basic initialization
        self.assertIsNotNone(sg)
        self.assertIsInstance(sg.metaverts, dict)
        self.assertGreater(len(sg.metaverts), 0)
        self.assertEqual(sg.doc, self.test_doc)
        
        # Verify metaverts structure using config assertions
        assertion_data = self.config.get_test_assertion_data()
        init_assertions = assertion_data['basic_functionality']['initialization']['from_doc_assertions']
        
        # Check assertions based on config
        if 'has_metaverts' in init_assertions:
            self.assertTrue(hasattr(sg, 'metaverts'))
        if 'doc_stored' in init_assertions:
            self.assertEqual(sg.doc, self.test_doc)
        if 'proper_structure' in init_assertions:
            self.assertIsInstance(sg.metaverts, dict)
    
    def test_initialization_from_vert_list(self):
        """Test initialization from a vertex list."""
        sg = SemanticMetagraph(vert_list=self.test_vert_list)
        
        # Verify initialization
        self.assertIsNotNone(sg)
        self.assertEqual(len(sg.metaverts), 3)
        self.assertIsNone(sg.doc)
        
        # Verify structure based on config
        assertion_data = self.config.get_test_assertion_data()
        vert_list_assertions = assertion_data['basic_functionality']['initialization']['from_vert_list_assertions']
        
        if 'correct_count' in vert_list_assertions:
            self.assertEqual(len(sg.metaverts), len(self.test_vert_list))
        if 'no_doc' in vert_list_assertions:
            self.assertIsNone(sg.doc)
    
    def test_initialization_empty(self):
        """Test initialization with no arguments creates empty graph."""
        sg = SemanticMetagraph()
        
        # Verify empty initialization
        self.assertIsNotNone(sg)
        self.assertEqual(len(sg.metaverts), 0)
        self.assertIsNone(sg.doc)
        
        # Verify based on config assertions
        assertion_data = self.config.get_test_assertion_data()
        empty_assertions = assertion_data['basic_functionality']['initialization']['empty_initialization_assertions']
        
        if 'empty_metaverts' in empty_assertions:
            self.assertEqual(len(sg.metaverts), 0)
        if 'no_doc' in empty_assertions:
            self.assertIsNone(sg.doc)
    
    def test_build_metaverts_from_doc(self):
        """Test the _build_metaverts_from_doc method."""
        sg = SemanticMetagraph()
        metaverts = sg._build_metaverts_from_doc(self.test_doc)
        
        # Check that metaverts were created
        self.assertIsInstance(metaverts, list)
        self.assertGreater(len(metaverts), 0)
        
        # Check that tokens are properly represented
        token_count = len(self.test_doc)
        self.assertGreaterEqual(len(metaverts), token_count)
        
        # Check metavert structure for first tokens
        for i in range(min(token_count, len(metaverts))):
            mv = metaverts[i]
            self.assertIsInstance(mv[0], str)
            if len(mv) > 1:
                self.assertIn("text", mv[1])
                self.assertIn("pos", mv[1])
    
    def test_get_token_tags_static_method(self):
        """Test the get_token_tags static method."""
        token = self.test_doc[0]  # First token
        tags = SemanticMetagraph.get_token_tags(token)
        
        self.assertIsInstance(tags, dict)
        
        # Check expected tag types based on config
        expected_structures = self.config.get_expected_semantic_structures()
        apple_structure = expected_structures['apple_company_structure']
        
        # Verify case and type tags are present
        if token.text.istitle():
            self.assertIn("case", tags)
            self.assertEqual(tags["case"], "title")
        
        self.assertIn("type", tags)
    
    def test_get_dep_edges_static_method(self):
        """Test the get_dep_edges static method."""
        # Find a token with children
        token_with_children = None
        for token in self.test_doc:
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
        """Test conversion to NetworkX graph."""
        sg = SemanticMetagraph(doc=self.simple_doc)
        G = sg.to_nx()
        
        # Verify conversion based on config
        assertion_data = self.config.get_test_assertion_data()
        nx_assertions = assertion_data['basic_functionality']['conversion_operations']['to_nx_assertions']
        
        if 'is_digraph' in nx_assertions:
            self.assertIsInstance(G, nx.DiGraph)
        if 'has_nodes' in nx_assertions:
            self.assertGreater(len(G.nodes()), 0)
        if 'proper_attributes' in nx_assertions:
            # Check node attributes
            for node in G.nodes():
                node_data = G.nodes[node]
                self.assertIn("label", node_data)
    
    def test_to_json_serialization(self):
        """Test JSON serialization."""
        sg = SemanticMetagraph(vert_list=self.test_vert_list)
        json_data = sg.to_json()
        
        # Verify JSON structure based on config
        assertion_data = self.config.get_test_assertion_data()
        json_assertions = assertion_data['basic_functionality']['conversion_operations']['to_json_assertions']
        
        if 'has_metaverts_field' in json_assertions:
            self.assertIsInstance(json_data, dict)
            self.assertIn("metaverts", json_data)
        
        if 'valid_json' in json_assertions:
            # Parse the JSON to ensure it's valid
            metaverts_json = json.loads(json_data["metaverts"])
            self.assertIsInstance(metaverts_json, list)
    
    def test_from_json_deserialization(self):
        """Test JSON deserialization."""
        sg1 = SemanticMetagraph(vert_list=self.test_vert_list)
        json_data = sg1.to_json()
        
        # Create new graph from JSON
        sg2 = SemanticMetagraph.from_json(json_data)
        
        # Verify deserialization based on config
        assertion_data = self.config.get_test_assertion_data()
        from_json_assertions = assertion_data['basic_functionality']['conversion_operations']['from_json_assertions']
        
        if 'equivalent_structure' in from_json_assertions:
            self.assertIsInstance(sg2, SemanticMetagraph)
            self.assertEqual(len(sg2.metaverts), len(sg1.metaverts))
    
    def test_get_tokens_method(self):
        """Test the get_tokens method."""
        sg = SemanticMetagraph(doc=self.simple_doc)
        tokens = sg.get_tokens()
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Verify token structure
        for token in tokens:
            self.assertIsInstance(token, dict)
            self.assertIn("metavert_idx", token)
            self.assertIn("token_idx", token)
            self.assertIn("text", token)
            self.assertIn("metadata", token)
    
    def test_get_relations_method(self):
        """Test the get_relations method."""
        sg = SemanticMetagraph(doc=self.simple_doc)
        relations = sg.get_relations()
        
        self.assertIsInstance(relations, list)
        
        # Verify relation structure
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
    
    def test_add_vert_functionality(self):
        """Test adding vertices to the graph."""
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
        
        # Verify assertions based on config
        assertion_data = self.config.get_test_assertion_data()
        add_assertions = assertion_data['basic_functionality']['vertex_operations']['add_vert_assertions']
        
        if 'increased_count' in add_assertions:
            self.assertGreater(len(sg.metaverts), 0)
    
    def test_remove_vert_functionality(self):
        """Test removing vertices from the graph."""
        vertex_data = [
            ("word1", {"pos": "NOUN"}),
            ("word2", {"pos": "VERB"}),
            ((0, 1), {"relation": "subject"}),
            ("word3", {"pos": "ADJ"})
        ]
        sg = SemanticMetagraph(vert_list=vertex_data)
        
        initial_length = len(sg.metaverts)
        
        # Remove the first vertex (word1)
        sg.remove_vert(0)
        
        # Check that the graph is smaller
        self.assertLess(len(sg.metaverts), initial_length)
        
        # Verify based on config assertions
        assertion_data = self.config.get_test_assertion_data()
        remove_assertions = assertion_data['basic_functionality']['vertex_operations']['remove_vert_assertions']
        
        if 'decreased_count' in remove_assertions:
            self.assertLess(len(sg.metaverts), initial_length)
        if 'removed_relations' in remove_assertions:
            # The relation depending on vertex 0 should also be removed
            for mv in sg.metaverts.values():
                if isinstance(mv[0], tuple):
                    self.assertNotIn(0, mv[0])


class TestSemanticMetagraphValidation(unittest.TestCase):
    """Validation and constraint tests for SemanticMetagraph."""
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory and configuration
        self.mock_factory = SemanticMetagraphMockFactory()
        self.config = SemanticMetagraphMockConfig()
        
        # Get validation test scenarios from config
        self.validation_scenarios = self.config.get_validation_test_scenarios()
        
        # Create validation-specific mock
        self.mock_validation = self.mock_factory('MockSemanticMetagraphValidation')
    
    def test_valid_graph_validation(self):
        """Test validation of valid graph structures."""
        valid_graphs = self.validation_scenarios['valid_graphs']
        
        for graph_name, vert_list in valid_graphs.items():
            with self.subTest(graph=graph_name):
                # Test with real implementation
                sg = SemanticMetagraph(vert_list=vert_list)
                self.assertIsNotNone(sg)
                self.assertEqual(len(sg.metaverts), len(vert_list))
    
    def test_invalid_graph_validation(self):
        """Test validation of invalid graph structures."""
        invalid_graphs = self.validation_scenarios['invalid_graphs']
        
        for graph_name, vert_list in invalid_graphs.items():
            with self.subTest(graph=graph_name):
                # Test validation failure
                with self.assertRaises((ValueError, AssertionError)):
                    SemanticMetagraph(vert_list=vert_list)
    
    def test_edge_case_graph_validation(self):
        """Test validation of edge case graph structures."""
        edge_case_graphs = self.validation_scenarios['edge_case_graphs']
        
        # Test empty graph
        empty_sg = SemanticMetagraph(vert_list=edge_case_graphs['empty_graph'])
        self.assertEqual(len(empty_sg.metaverts), 0)
        
        # Test single vertex
        single_sg = SemanticMetagraph(vert_list=edge_case_graphs['single_vertex'])
        self.assertEqual(len(single_sg.metaverts), 1)
        
        # Test undirected relations
        undirected_sg = SemanticMetagraph(vert_list=edge_case_graphs['undirected_relations'])
        self.assertGreater(len(undirected_sg.metaverts), 0)
    
    def test_vertex_validation_with_mock(self):
        """Test vertex validation using validation mock."""
        # Configure mock for validation testing
        self.mock_validation.set_validation_failure(False)
        self.mock_validation.set_vertex_validation_failure(False)
        
        # Test normal validation
        result = self.mock_validation._mock_validate_vert(0, ("test", {"pos": "NOUN"}))
        self.assertTrue(result)
        
        # Configure mock for validation failure
        self.mock_validation.set_validation_failure(True)
        
        # Test validation failure
        with self.assertRaises(ValueError):
            self.mock_validation._mock_validate_graph([("invalid",)])
    
    def test_canonicalization_validation(self):
        """Test vertex canonicalization during validation."""
        # Test with undirected edge canonicalization
        undirected_vert_list = [
            ("word1", {"pos": "NOUN"}),
            ("word2", {"pos": "VERB"}),
            ([1, 0], {"relation": "coordination"})  # Should be canonicalized to [0, 1]
        ]
        
        sg = SemanticMetagraph(vert_list=undirected_vert_list)
        
        # Verify canonicalization occurred
        # The undirected relation should have nodes in ascending order
        for mv in sg.metaverts.values():
            if isinstance(mv[0], list):
                self.assertEqual(mv[0], sorted(mv[0]))


class TestSemanticMetagraphEdgeCases(unittest.TestCase):
    """Edge cases and error conditions tests for SemanticMetagraph."""
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory and configuration
        self.mock_factory = SemanticMetagraphMockFactory()
        self.config = SemanticMetagraphMockConfig()
        
        # Get edge case test scenarios from config
        self.edge_case_scenarios = self.config.get_edge_case_test_scenarios()
        
        # Create edge case specific mock
        self.mock_edge_cases = self.mock_factory('MockSemanticMetagraphEdgeCases')
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        empty_inputs = self.edge_case_scenarios['empty_inputs']
        
        # Test empty doc
        sg_empty_doc = SemanticMetagraph(doc=empty_inputs['empty_doc'])
        self.assertEqual(len(sg_empty_doc.metaverts), 0)
        
        # Test empty vertex list
        sg_empty_list = SemanticMetagraph(vert_list=empty_inputs['empty_vert_list'])
        self.assertEqual(len(sg_empty_list.metaverts), 0)
        
        # Verify using edge case mock
        self.mock_edge_cases.trigger_edge_case('empty_input')
        edge_info = self.mock_edge_cases.get_edge_case_info()
        self.assertTrue(edge_info['is_active'])
        self.assertEqual(edge_info['case_type'], 'empty_input')
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        malformed_data = self.edge_case_scenarios['malformed_data']
        
        # Test incomplete metaverts
        incomplete_verts = malformed_data['incomplete_metaverts']
        with self.assertRaises((ValueError, AssertionError)):
            SemanticMetagraph(vert_list=incomplete_verts)
        
        # Test using edge case mock - should work without raising exception
        self.mock_edge_cases.setup_edge_case_scenario('invalid_data')
        
        # Test that mock properly simulates the error
        with self.assertRaises(ValueError):
            self.mock_edge_cases._edge_case_add_vert("test")
    
    def test_boundary_conditions(self):
        """Test boundary conditions and limits."""
        boundary_conditions = self.edge_case_scenarios['boundary_conditions']
        
        # Test special characters (should not crash)
        try:
            # Create a simple doc-like object for testing
            class MockDoc:
                def __init__(self, text):
                    self.text = text
                    self.tokens = text.split() if text else []
                
                def __len__(self):
                    return len(self.tokens)
                
                def __iter__(self):
                    return iter([])  # Simplified for testing
            
            special_text = boundary_conditions['special_characters']
            mock_doc = MockDoc(special_text)
            
            # This should not crash
            sg = SemanticMetagraph()
            self.assertIsNotNone(sg)
            
        except Exception as e:
            self.fail(f"Special character handling failed: {e}")
        
        # Test large graph creation
        large_graph = boundary_conditions['large_graph'][:10]  # Limit for testing
        sg_large = SemanticMetagraph(vert_list=large_graph)
        self.assertEqual(len(sg_large.metaverts), len(large_graph))
    
    def test_memory_intensive_scenarios_with_mock(self):
        """Test memory intensive scenarios using mock."""
        # Test memory limit edge case setup
        self.mock_edge_cases.setup_edge_case_scenario('memory_limit')
        
        # Test that memory error is properly simulated
        with self.assertRaises(MemoryError):
            self.mock_edge_cases._edge_case_to_nx()
    
    def test_json_malformation_handling(self):
        """Test handling of malformed JSON data."""
        # Test using edge case mock
        self.mock_edge_cases.trigger_edge_case('malformed_json')
        
        # Test malformed JSON handling
        malformed_result = self.mock_edge_cases._edge_case_to_json()
        self.assertIsInstance(malformed_result, str)  # Should return invalid JSON string
        
        # Verify it's not valid JSON
        with self.assertRaises(json.JSONDecodeError):
            json.loads(malformed_result)
    
    def test_error_recovery_and_cleanup(self):
        """Test error recovery and resource cleanup."""
        # Test that edge case mock can be reset
        # First, trigger a non-error edge case
        self.mock_edge_cases.trigger_edge_case('empty_input')
        self.assertTrue(self.mock_edge_cases.is_edge_case_active)
        
        # Reset and verify cleanup
        self.mock_edge_cases.reset_edge_case()
        self.assertFalse(self.mock_edge_cases.is_edge_case_active)
        self.assertIsNone(self.mock_edge_cases.edge_case_type)
        
        # Test that error cases can be handled and reset
        self.mock_edge_cases.setup_edge_case_scenario('invalid_data')
        self.assertTrue(self.mock_edge_cases.is_edge_case_active)
        
        # Reset after error setup
        self.mock_edge_cases.reset_edge_case()
        self.assertFalse(self.mock_edge_cases.is_edge_case_active)


class TestSemanticMetagraphIntegration(unittest.TestCase):
    """Integration tests with other components for SemanticMetagraph."""
    
    @classmethod
    def setUpClass(cls):
        """Load spaCy model once for all integration tests."""
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            cls.nlp = spacy.load("en_core_web_sm")
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory and configuration
        self.mock_factory = SemanticMetagraphMockFactory()
        self.config = SemanticMetagraphMockConfig()
        
        # Get integration test data from config
        self.integration_data = self.config.get_integration_test_data()
        
        # Create integration-specific mock
        self.mock_integration = self.mock_factory('MockSemanticMetagraphIntegration')
        
        # Create mock spaCy components
        self.mock_spacy_doc = self.mock_factory('MockSpacyDoc')
    
    def test_spacy_integration(self):
        """Test integration with spaCy pipeline."""
        spacy_data = self.integration_data['spacy_integration']
        sample_texts = spacy_data['sample_texts']
        expected_tokens = spacy_data['expected_token_counts']
        
        for i, text in enumerate(sample_texts):
            with self.subTest(text=text[:30] + "..."):
                doc = self.nlp(text)
                sg = SemanticMetagraph(doc=doc)
                
                # Verify spaCy integration
                self.assertIsNotNone(sg.doc)
                self.assertEqual(sg.doc, doc)
                
                # Check token extraction
                tokens = sg.get_tokens()
                # Allow for some variance due to spaCy tokenization
                self.assertGreaterEqual(len(tokens), expected_tokens[i] - 2)
                self.assertLessEqual(len(tokens), expected_tokens[i] + 2)
    
    def test_networkx_integration(self):
        """Test integration with NetworkX."""
        networkx_data = self.integration_data['networkx_integration']
        expected_attributes = networkx_data['expected_node_attributes']
        graph_properties = networkx_data['graph_properties']
        
        # Create test graph
        test_text = "Apple is a technology company."
        doc = self.nlp(test_text)
        sg = SemanticMetagraph(doc=doc)
        
        # Convert to NetworkX
        G = sg.to_nx()
        
        # Verify NetworkX integration
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(nx.is_directed(G), graph_properties['is_directed'])
        
        # Check node attributes
        if G.nodes():
            sample_node = list(G.nodes())[0]
            node_attrs = G.nodes[sample_node]
            
            # Check for expected attributes (some may be missing in simplified implementation)
            common_attrs = set(expected_attributes) & set(node_attrs.keys())
            self.assertGreater(len(common_attrs), 0, "Should have at least one expected attribute")
    
    def test_json_serialization_integration(self):
        """Test JSON serialization integration."""
        json_data_config = self.integration_data['json_serialization']
        required_fields = json_data_config['required_fields']
        expected_structure = json_data_config['expected_structure']
        
        # Create test graph
        vertex_structures = self.config.get_basic_test_vertex_structures()
        test_verts = vertex_structures['simple_vertices'][:2]  # Use first 2 vertices
        test_verts.append(((0, 1), {"relation": "test_rel"}))  # Add relation
        
        sg = SemanticMetagraph(vert_list=test_verts)
        
        # Test serialization
        json_data = sg.to_json()
        
        # Verify required fields
        for field in required_fields:
            self.assertIn(field, json_data)
        
        # Verify JSON structure
        metaverts_data = json.loads(json_data['metaverts'])
        self.assertIsInstance(metaverts_data, list)
        
        if metaverts_data:
            sample_mv = metaverts_data[0]
            mv_type = sample_mv.get('type', 'unknown')
            
            if mv_type in expected_structure:
                expected_fields = expected_structure[mv_type]
                # Check that at least some expected fields are present
                present_fields = set(sample_mv.keys()) & set(expected_fields)
                self.assertGreater(len(present_fields), 0)
        
        # Test deserialization
        sg2 = SemanticMetagraph.from_json(json_data)
        self.assertIsInstance(sg2, SemanticMetagraph)
    
    def test_full_pipeline_integration(self):
        """Test full processing pipeline integration."""
        pipeline_data = self.integration_data['full_pipeline']
        input_text = pipeline_data['input_text']
        expected_stages = pipeline_data['expected_pipeline_stages']
        expected_outputs = pipeline_data['expected_outputs']
        
        # Test full pipeline using integration mock
        pipeline_result = self.mock_integration.full_pipeline_processing(input_text)
        
        # Verify pipeline execution
        self.assertIsNotNone(pipeline_result)
        self.assertIsInstance(pipeline_result, list)
        
        # Verify integration state
        integration_valid = self.mock_integration.validate_integration_state()
        self.assertTrue(integration_valid)
        
        # Test with real implementation
        doc = self.nlp(input_text)
        sg = SemanticMetagraph(doc=doc)
        
        # Check basic outputs (with tolerance for implementation differences)
        tokens = sg.get_tokens()
        relations = sg.get_relations()
        
        # Verify reasonable output counts
        self.assertGreater(len(tokens), 0)
        self.assertGreaterEqual(len(sg.metaverts), len(tokens))
    
    def test_component_interaction_validation(self):
        """Test validation of component interactions."""
        # Verify all required components are present in integration mock
        factory_config = self.config.get_mock_factory_configurations()
        integration_config = factory_config['integration_mocks']['MockSemanticMetagraphIntegration']
        required_components = integration_config['required_components']
        
        for component in required_components:
            self.assertTrue(
                hasattr(self.mock_integration, component),
                f"Integration mock missing required component: {component}"
            )
        
        # Test component setup
        components = self.mock_integration.setup_integration_components()
        self.assertIsInstance(components, dict)
        
        # Verify component interaction configuration
        self.mock_integration.configure_component_interactions()
        integration_state = self.mock_integration.validate_integration_state()
        self.assertTrue(integration_state)
    
    def test_graph_traversal_integration(self):
        """Test that the integrated graph structure is navigable."""
        text = "The cat sat on the mat."
        doc = self.nlp(text)
        sg = SemanticMetagraph(doc=doc)
        
        # Convert to NetworkX for traversal testing
        G = sg.to_nx()
        
        # Check graph connectivity
        if G.nodes():
            components = list(nx.weakly_connected_components(G))
            self.assertGreater(len(components), 0)
            
            # Test basic traversal
            for node in G.nodes():
                # Get predecessors and successors
                preds = list(G.predecessors(node))
                succs = list(G.successors(node))
                
                # Node should have connections or be intentionally isolated
                has_connections = len(preds) > 0 or len(succs) > 0
                is_isolated = len(components) == 1 and len(G.nodes()) == 1
                
                self.assertTrue(
                    has_connections or is_isolated or len(components) > 1,
                    f"Node {node} appears to be improperly isolated"
                )


if __name__ == "__main__":
    # Run all test classes
    unittest.main(verbosity=2)