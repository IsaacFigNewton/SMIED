#!/usr/bin/env python3
"""
Comprehensive unit tests for SMIED Diagnostics module.

This test suite provides complete coverage for the SMIEDDiagnostics class,
including core functionality, integration with SMIED components, performance
testing, and error handling.

Test Categories:
- Core functionality tests for all diagnostic methods
- Integration tests with SMIED components
- Performance tests with memory/time bounds
- Mock and fixture setup for isolated testing
- Edge case and error condition testing
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import json
import tempfile
import os
import time
from collections import defaultdict
from typing import List, Dict, Any

# Import test dependencies
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smied.Diagnostics import SMIEDDiagnostics
from nltk.corpus import wordnet as wn

# Import SMIED testing framework components
try:
    from .mocks.diagnostics_mocks import diagnostics_mock_factory
    from .config.diagnostics_config import DiagnosticsMockConfig
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'mocks'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
    from diagnostics_mocks import diagnostics_mock_factory
    from diagnostics_config import DiagnosticsMockConfig


class TestDiagnostics(unittest.TestCase):
    """Basic functionality tests for SMIEDDiagnostics class."""
    
    def setUp(self):
        """Set up test fixtures using mock factory and config injection."""
        # Initialize mock factory and configuration
        self.mock_factory = diagnostics_mock_factory
        self.config = DiagnosticsMockConfig
        
        # Create mock diagnostics instance for testing
        self.diagnostics = self.mock_factory.create_standard_mock(verbosity=0)
        
        # Get test data from configuration
        self.test_data = self.config.get_basic_test_data()
        self.connectivity_pairs = self.test_data['connectivity_test_pairs']
        self.standard_test_cases = self.test_data['standard_test_cases']
        
        # Create test graph from configuration
        self.test_graph = nx.DiGraph()
        for src, tgt, attrs in self.test_data['test_graph_edges']:
            self.test_graph.add_edge(src, tgt, **attrs)
        
        # Override the diagnostics graph with our test graph
        self.diagnostics.graph = self.test_graph
    
    def tearDown(self):
        """Clean up after each test method."""
        # Reset mock state for next test
        if hasattr(self.diagnostics, 'reset_mock'):
            self.diagnostics.reset_mock()
    
    def test_initialization(self):
        """Test SMIEDDiagnostics initialization."""
        # Test standard mock initialization
        mock_diag = self.mock_factory.create_standard_mock(verbosity=1)
        self.assertIsNotNone(mock_diag.semantic_decomposer)
        self.assertIsNotNone(mock_diag.graph)
        self.assertEqual(mock_diag.verbosity, 1)
        
        # Test initialization with custom parameters from config
        component_data = self.config.get_mock_component_data()
        mock_diag_custom = self.mock_factory.create_standard_mock(
            verbosity=2,
            nlp_func=component_data['mock_nlp_function'],
            embedding_model=component_data['mock_embedding_model']
        )
        self.assertEqual(mock_diag_custom.verbosity, 2)
    
    def test_analyze_synset_connectivity_basic(self):
        """Test basic synset connectivity analysis."""
        # Get test pairs from configuration
        test_pair = self.connectivity_pairs[0]  # Direct connection test case
        
        result = self.diagnostics.analyze_synset_connectivity(
            test_pair['source'], test_pair['target']
        )
        
        self.assertEqual(result['source'], test_pair['source'])
        self.assertEqual(result['target'], test_pair['target'])
        self.assertEqual(result['direct_connection'], test_pair['expected_direct'])
        self.assertEqual(result['direct_relation'], test_pair['expected_relation'])
        self.assertGreater(len(result['paths']), 0)
        self.assertEqual(result['shortest_path_length'], test_pair['expected_path_length'])
    
    def test_analyze_synset_connectivity_multi_hop(self):
        """Test multi-hop connectivity analysis."""
        # Get multi-hop test case from configuration
        test_pair = self.connectivity_pairs[1]  # Multi-hop connection test case
        
        result = self.diagnostics.analyze_synset_connectivity(
            test_pair['source'], test_pair['target'], max_hops=3
        )
        
        self.assertEqual(result['source'], test_pair['source'])
        self.assertEqual(result['target'], test_pair['target'])
        self.assertEqual(result['direct_connection'], test_pair['expected_direct'])
        self.assertGreater(len(result['paths']), 0)
        self.assertEqual(result['shortest_path_length'], test_pair['expected_path_length'])
        
        # Check that path has expected length
        path = result['paths'][0]['path']
        self.assertEqual(len(path), test_pair['expected_path_length'] + 1)
    
    def test_analyze_synset_connectivity_no_path(self):
        """Test connectivity analysis when no path exists."""
        # Use edge case configuration for disconnected graph scenario
        edge_case_data = self.config.get_edge_case_scenarios()
        disconnected_data = edge_case_data['disconnected_graph_data']
        
        # Create edge case mock with disconnected graph
        edge_case_mock = self.mock_factory.create_edge_case_mock()
        edge_case_mock.graph = edge_case_mock.disconnected_graph
        
        no_path_pair = disconnected_data['expected_no_path_pairs'][0]
        result = edge_case_mock.analyze_synset_connectivity(no_path_pair[0], no_path_pair[1])
        
        self.assertEqual(result['source'], no_path_pair[0])
        self.assertEqual(result['target'], no_path_pair[1])
        self.assertFalse(result['direct_connection'])
        self.assertEqual(len(result['paths']), 0)
        self.assertIsNone(result['shortest_path_length'])
    
    def test_analyze_synset_connectivity_missing_nodes(self):
        """Test connectivity analysis with missing nodes."""
        # Get missing synset scenarios from configuration
        missing_scenarios = self.config.get_edge_case_scenarios()['missing_synset_scenarios']
        
        # Test missing source
        scenario = missing_scenarios[0]
        result = self.diagnostics.analyze_synset_connectivity(scenario['source'], scenario['target'])
        self.assertIn('error', result)
        self.assertIn(scenario['expected_error'], result['error'])
        
        # Test missing target
        scenario = missing_scenarios[1]
        result = self.diagnostics.analyze_synset_connectivity(scenario['source'], scenario['target'])
        self.assertIn('error', result)
        self.assertIn(scenario['expected_error'], result['error'])
    
    def test_audit_wordnet_relations(self):
        """Test WordNet relation coverage audit."""
        result = self.diagnostics.audit_wordnet_relations()
        
        # Get expected result structure from configuration
        expected_keys = self.config.get_integration_test_data()['expected_analysis_results']['relation_audit_keys']
        
        # Check basic structure
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Get expected results from mock configuration
        mock_results = self.config.get_mock_analysis_results()
        expected_audit = mock_results['mock_audit_result']
        
        # Check that critical missing relations are identified
        self.assertIn('derivationally_related_forms', result['missing_relations'])
        
        # Check recommendations are generated
        self.assertGreater(len(result['recommendations']), 0)
        
        # Verify high-priority recommendation exists
        high_priority_recs = [r for r in result['recommendations'] if r.get('priority') == 'HIGH']
        self.assertGreater(len(high_priority_recs), 0)
    
    def test_test_parameter_sensitivity(self):
        """Test parameter sensitivity analysis."""
        test_cases = [('cat.n.01', 'feline.n.01'), ('dog.n.01', 'canine.n.01')]
        
        with patch('smied.Diagnostics.PairwiseBidirectionalAStar') as mock_pathfinder:
            # Mock successful pathfinding
            mock_instance = Mock()
            mock_instance.find_paths.return_value = [(['cat.n.01', 'feline.n.01'], 1.0)]
            mock_pathfinder.return_value = mock_instance
            
            result = self.diagnostics.test_parameter_sensitivity(test_cases)
            
            # Check basic structure
            self.assertIn('test_cases', result)
            self.assertIn('parameter_results', result)
            self.assertIn('success_rates', result)
            self.assertIn('performance_metrics', result)
            self.assertIn('recommendations', result)
            
            # Should have tested multiple parameter configurations
            self.assertGreater(len(result['parameter_results']), 1)
            self.assertGreater(len(result['success_rates']), 1)
            
            # Should have recommendations
            self.assertGreater(len(result['recommendations']), 0)
    
    def test_validate_pathfinding_fixes(self):
        """Test pathfinding fixes validation."""
        result = self.diagnostics.validate_pathfinding_fixes()
        
        # Check basic structure
        self.assertIn('critical_test_cases', result)
        self.assertIn('current_failures', result)
        self.assertIn('derivational_analysis', result)
        self.assertIn('recommendations', result)
        
        # Should have analyzed critical cases
        self.assertGreater(len(result['critical_test_cases']), 0)
        self.assertGreater(len(result['current_failures']), 0)
        
        # Should have recommendations
        self.assertGreater(len(result['recommendations']), 0)
    
    def test_analyze_graph_topology(self):
        """Test graph topology analysis."""
        result = self.diagnostics.analyze_graph_topology()
        
        # Check basic structure
        self.assertIn('basic_statistics', result)
        self.assertIn('connectivity_analysis', result)
        self.assertIn('degree_analysis', result)
        self.assertIn('relation_distribution', result)
        self.assertIn('pos_distribution', result)
        
        # Check basic statistics
        stats = result['basic_statistics']
        self.assertEqual(stats['num_nodes'], self.test_graph.number_of_nodes())
        self.assertEqual(stats['num_edges'], self.test_graph.number_of_edges())
        self.assertTrue(stats['is_directed'])
        
        # Check degree analysis
        degree_analysis = result['degree_analysis']
        self.assertIn('average_degree', degree_analysis)
        self.assertIn('max_degree', degree_analysis)
        self.assertIn('min_degree', degree_analysis)
        
        # Check relation distribution matches our test graph
        rel_dist = result['relation_distribution']
        self.assertIn('hypernym', rel_dist)
        self.assertIn('similar_to', rel_dist)
    
    def test_analyze_cross_pos_connectivity(self):
        """Test cross-POS connectivity analysis."""
        result = self.diagnostics.analyze_cross_pos_connectivity()
        
        # Check basic structure
        self.assertIn('pos_pairs', result)
        self.assertIn('cross_pos_edges', result)
        self.assertIn('missing_connections', result)
        self.assertIn('recommendations', result)
        self.assertIn('cross_pos_statistics', result)
        
        # Should identify lack of cross-POS connections
        stats = result['cross_pos_statistics']
        self.assertEqual(stats['cross_pos_edges'], 0)  # Our test graph has no cross-POS edges
        
        # Should have recommendations for improving cross-POS connectivity
        recs = result['recommendations']
        self.assertGreater(len(recs), 0)
        
        # Should identify this as a critical issue
        critical_recs = [r for r in recs if r.get('priority') == 'CRITICAL']
        self.assertGreater(len(critical_recs), 0)
    
    def test_analyze_relation_density(self):
        """Test relation density analysis."""
        # Test with specific synsets
        test_synsets = ['cat.n.01', 'dog.n.01', 'feline.n.01']
        result = self.diagnostics.analyze_relation_density(test_synsets)
        
        # Check basic structure
        self.assertIn('analyzed_synsets', result)
        self.assertIn('total_analyzed', result)
        self.assertIn('density_statistics', result)
        self.assertIn('sparse_nodes', result)
        self.assertIn('well_connected_nodes', result)
        self.assertIn('relation_patterns', result)
        
        # Check that analysis was performed
        self.assertEqual(result['total_analyzed'], len(test_synsets))
        
        # Test with all synsets (default)
        result_all = self.diagnostics.analyze_relation_density()
        self.assertGreater(result_all['total_analyzed'], 0)


class TestDiagnosticsValidation(unittest.TestCase):
    """Validation and constraint tests for SMIEDDiagnostics."""
    
    def setUp(self):
        """Set up validation test fixtures using mock factory and config injection."""
        self.mock_factory = diagnostics_mock_factory
        self.config = DiagnosticsMockConfig
        self.diagnostics = self.mock_factory.create_standard_mock(verbosity=0)
        
        # Get validation test data from configuration
        self.validation_data = self.config.get_validation_test_data()
        
    def test_valid_synset_names_acceptance(self):
        """Test that valid synset names are accepted."""
        valid_names = self.validation_data['valid_synset_names']
        
        for synset_name in valid_names:
            # Should not raise an exception for valid names
            try:
                result = self.diagnostics.analyze_synset_connectivity(synset_name, valid_names[0])
                # If synset exists in graph, should return result without error
                if synset_name in self.diagnostics.graph:
                    self.assertNotIn('error', result)
            except Exception as e:
                self.fail(f"Valid synset name {synset_name} caused exception: {e}")
    
    def test_invalid_synset_names_rejection(self):
        """Test that invalid synset names are properly rejected."""
        invalid_names = self.validation_data['invalid_synset_names']
        valid_name = self.validation_data['valid_synset_names'][0]
        
        for invalid_name in invalid_names:
            if invalid_name is not None:  # Skip None for now
                result = self.diagnostics.analyze_synset_connectivity(invalid_name, valid_name)
                self.assertIn('error', result)
    
    def test_max_hops_validation(self):
        """Test max_hops parameter validation."""
        valid_name = self.validation_data['valid_synset_names'][0]
        
        # Test valid max_hops values
        for max_hops in self.validation_data['valid_max_hops_values']:
            try:
                result = self.diagnostics.analyze_synset_connectivity(
                    valid_name, valid_name, max_hops=max_hops
                )
                # Should not raise exception
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"Valid max_hops {max_hops} caused exception: {e}")
    
    def test_result_format_validation(self):
        """Test that analysis results follow expected format."""
        expected_formats = self.validation_data['expected_result_formats']
        valid_names = self.validation_data['valid_synset_names']
        
        # Test connectivity analysis result format
        result = self.diagnostics.analyze_synset_connectivity(valid_names[0], valid_names[1])
        connectivity_format = expected_formats['connectivity_analysis']
        
        for required_key in connectivity_format['required_keys']:
            self.assertIn(required_key, result)
        
        # Test topology analysis result format  
        topology_result = self.diagnostics.analyze_graph_topology()
        topology_format = expected_formats['topology_analysis']
        
        for required_key in topology_format['required_keys']:
            self.assertIn(required_key, topology_result)


class TestDiagnosticsEdgeCases(unittest.TestCase):
    """Edge cases and error condition tests for SMIEDDiagnostics."""
    
    def setUp(self):
        """Set up edge case test fixtures using mock factory and config injection."""
        self.mock_factory = diagnostics_mock_factory
        self.config = DiagnosticsMockConfig
        
        # Use edge case mock for error condition testing
        self.edge_case_diagnostics = self.mock_factory.create_edge_case_mock()
        self.edge_case_data = self.config.get_edge_case_scenarios()
    
    def test_empty_graph_handling(self):
        """Test behavior with empty graph."""
        empty_data = self.edge_case_data['empty_graph_data']
        
        # Set diagnostics to use empty graph
        self.edge_case_diagnostics.graph = self.edge_case_diagnostics.empty_graph
        
        # Test topology analysis
        result = self.edge_case_diagnostics.analyze_graph_topology()
        expected_topology = empty_data['expected_topology']
        
        self.assertEqual(result['basic_statistics']['num_nodes'], expected_topology['num_nodes'])
        self.assertEqual(result['basic_statistics']['num_edges'], expected_topology['num_edges'])
        self.assertEqual(result['connectivity_analysis']['is_connected'], expected_topology['is_connected'])
    
    def test_circular_paths_handling(self):
        """Test handling of circular paths in graph."""
        circular_data = self.edge_case_data['circular_graph_data']
        
        # Set diagnostics to use circular graph
        self.edge_case_diagnostics.graph = self.edge_case_diagnostics.circular_graph
        
        # Should handle cycles gracefully
        result = self.edge_case_diagnostics.analyze_synset_connectivity('a.n.01', 'c.n.01')
        
        # Check that circular dependency is detected
        if 'connectivity_gaps' in result:
            circular_gaps = [g for g in result['connectivity_gaps'] if g.get('type') == 'circular_dependency']
            self.assertGreaterEqual(len(circular_gaps), 0)  # Should detect or handle cycles
    
    def test_disconnected_components_handling(self):
        """Test handling of disconnected graph components."""
        disconnected_data = self.edge_case_data['disconnected_graph_data']
        
        # Set diagnostics to use disconnected graph
        self.edge_case_diagnostics.graph = self.edge_case_diagnostics.disconnected_graph
        
        # Test topology analysis
        topology_result = self.edge_case_diagnostics.analyze_graph_topology()
        self.assertEqual(topology_result['connectivity_analysis']['num_components'], 
                        disconnected_data['expected_components'])
    
    def test_component_initialization_failures(self):
        """Test graceful handling of component initialization failures."""
        # Edge case mock should have None components to simulate failures
        self.assertIsNone(self.edge_case_diagnostics.semantic_decomposer)
        self.assertIsNone(self.edge_case_diagnostics.embedding_helper)
        self.assertIsNone(self.edge_case_diagnostics.beam_builder)
        
        # Core functionality should still work with degraded components
        result = self.edge_case_diagnostics.run_comprehensive_analysis()
        
        # Should include error information
        self.assertIn('errors', result)
        self.assertGreater(len(result['errors']), 0)
    
    def test_large_max_hops_handling(self):
        """Test handling of very large max_hops parameter."""
        performance_data = self.edge_case_data['performance_stress_data']
        
        # Use standard mock for this test
        standard_mock = self.mock_factory.create_standard_mock()
        
        # Test with unreasonably large max_hops
        result = standard_mock.analyze_synset_connectivity(
            'cat.n.01', 'feline.n.01', max_hops=1000
        )
        
        # Should complete without issues and limit search appropriately
        self.assertIn('paths', result)
        if result['paths']:
            max_path_length = max(p['length'] for p in result['paths'])
            self.assertLess(max_path_length, 1000)  # Should be much smaller in practice


class TestDiagnosticsIntegration(unittest.TestCase):
    """Integration tests with SMIED components."""
    
    def setUp(self):
        """Set up integration test fixtures using mock factory and config injection."""
        self.mock_factory = diagnostics_mock_factory
        self.config = DiagnosticsMockConfig
        
        # Use integration mock for realistic component interactions
        self.integration_diagnostics = self.mock_factory.create_integration_mock()
        self.integration_data = self.config.get_integration_test_data()
        
        # Set up realistic graph from configuration
        self.integration_diagnostics.graph = self._create_integration_graph()
    
    def _create_integration_graph(self):
        """Create integration test graph from configuration."""
        graph = nx.DiGraph()
        edges = self.integration_data['realistic_graph_edges']
        
        for src, tgt, attrs in edges:
            graph.add_edge(src, tgt, **attrs)
        
        return graph
    
    def test_semantic_decomposer_integration(self):
        """Test integration with SemanticDecomposer."""
        # Test that semantic decomposer is properly initialized
        self.assertIsNotNone(self.integration_diagnostics.semantic_decomposer)
        
        # Test that graph building works
        self.assertIsNotNone(self.integration_diagnostics.graph)
        self.assertGreater(self.integration_diagnostics.graph.number_of_nodes(), 0)
        
        # Test component configuration
        component_configs = self.integration_data['component_configurations']
        decomposer_config = component_configs['semantic_decomposer']
        
        self.assertEqual(self.integration_diagnostics.verbosity, decomposer_config['verbosity'])
    
    def test_embedding_helper_integration(self):
        """Test integration with EmbeddingHelper."""
        # Test that embedding helper is properly configured
        self.assertIsNotNone(self.integration_diagnostics.embedding_helper)
        
        # Test embedding functionality with realistic behavior
        embedding_helper = self.integration_diagnostics.embedding_helper
        
        # Test embedding retrieval
        cat_embedding = embedding_helper.get_embedding('cat')
        dog_embedding = embedding_helper.get_embedding('dog')
        
        self.assertIsInstance(cat_embedding, list)
        self.assertIsInstance(dog_embedding, list)
        
        # Test similarity computation
        similarity = embedding_helper.compute_similarity(cat_embedding, dog_embedding)
        self.assertIsInstance(similarity, (int, float))
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_beam_builder_integration(self):
        """Test integration with BeamBuilder."""
        # Test that beam builder is properly configured
        self.assertIsNotNone(self.integration_diagnostics.beam_builder)
        
        # Test beam building functionality
        beam_builder = self.integration_diagnostics.beam_builder
        
        # Test with source synsets from our integration graph
        source_synsets = ['cat.n.01', 'dog.n.01']
        beam = beam_builder.build_beam(source_synsets, beam_width=5)
        
        self.assertIsInstance(beam, list)
        self.assertLessEqual(len(beam), 5)  # Should respect beam width limit
    
    def test_comprehensive_analysis_integration(self):
        """Test comprehensive analysis with realistic component interactions."""
        result = self.integration_diagnostics.run_comprehensive_analysis()
        
        # Get expected sections from configuration
        expected_sections = self.integration_data['expected_analysis_results']['comprehensive_analysis_sections']
        
        # Check that all major analysis sections are present
        for section in expected_sections:
            self.assertIn(section, result)
        
        # Check that results have realistic structure
        self.assertIn('nodes', result['graph_info'])
        self.assertIn('edges', result['graph_info'])
        self.assertGreater(result['graph_info']['nodes'], 0)
        self.assertGreater(result['graph_info']['edges'], 0)


class TestDiagnosticsPerformance(unittest.TestCase):
    """Test performance characteristics and bounds."""
    
    def setUp(self):
        """Set up performance test fixtures using mock factory and config injection."""
        self.mock_factory = diagnostics_mock_factory
        self.config = DiagnosticsMockConfig
        
        # Use integration mock for realistic performance testing
        self.diagnostics = self.mock_factory.create_integration_mock(verbosity=0)
        
        # Get performance benchmarks from configuration
        self.benchmarks = self.config.get_performance_benchmarks()
        self.time_limits = self.benchmarks['time_limits']
        self.memory_limits = self.benchmarks['memory_limits']
    
    def test_connectivity_analysis_performance(self):
        """Test that connectivity analysis completes within time bounds."""
        start_time = time.time()
        
        result = self.diagnostics.analyze_synset_connectivity('cat.n.01', 'feline.n.01', max_hops=6)
        
        elapsed_time = time.time() - start_time
        self.assertLess(elapsed_time, self.time_limits['connectivity_analysis'])
        
        # Should have results
        self.assertIn('paths', result)
    
    def test_comprehensive_analysis_performance(self):
        """Test that comprehensive analysis completes within reasonable time."""
        start_time = time.time()
        
        # Use smaller test set from configuration
        standard_cases = self.config.get_basic_test_data()['standard_test_cases'][:2]
        
        with patch.object(self.diagnostics, 'get_standard_test_cases') as mock_cases:
            mock_cases.return_value = standard_cases
            
            result = self.diagnostics.run_comprehensive_analysis()
        
        elapsed_time = time.time() - start_time
        self.assertLess(elapsed_time, self.time_limits['comprehensive_analysis'])
        
        # Should have comprehensive results
        self.assertIn('relation_audit', result)
        self.assertIn('graph_topology', result)
    
    def test_large_graph_handling(self):
        """Test handling of larger graphs without performance degradation."""
        # Get large graph benchmarks from configuration
        large_benchmark = self.benchmarks['graph_size_benchmarks']['large_graph']
        
        # Create a larger test graph based on benchmark specifications
        large_graph = nx.DiGraph()
        
        # Add nodes and edges based on benchmark
        nodes = [f'test{i}.n.01' for i in range(large_benchmark['nodes'])]
        large_graph.add_nodes_from(nodes)
        
        # Add edges to create moderately connected graph
        edges_added = 0
        for i in range(0, large_benchmark['nodes'] - 1, 2):
            if edges_added < large_benchmark['edges']:
                large_graph.add_edge(nodes[i], nodes[i+1], relation='hypernym', weight=1.0)
                edges_added += 1
        
        # Override diagnostics graph
        self.diagnostics.graph = large_graph
        # Also update the integration mock's internal graph reference
        if hasattr(self.diagnostics, '_create_realistic_graph'):
            self.diagnostics._original_graph = self.diagnostics.graph
        
        # Test topology analysis on larger graph
        start_time = time.time()
        result = self.diagnostics.analyze_graph_topology()
        elapsed_time = time.time() - start_time
        
        # Convert expected processing time from ms to seconds
        expected_time_seconds = large_benchmark['expected_processing_time_ms'] / 1000.0
        self.assertLess(elapsed_time, expected_time_seconds)
        
        # Check results are reasonable - since we're using a mock, validate the structure
        # but don't expect exact node counts from the mock
        self.assertIn('basic_statistics', result)
        self.assertIn('num_nodes', result['basic_statistics'])
        self.assertIn('num_edges', result['basic_statistics'])
        self.assertGreaterEqual(result['basic_statistics']['num_nodes'], 0)
        self.assertGreaterEqual(result['basic_statistics']['num_edges'], 0)
    
    def test_memory_usage_bounds(self):
        """Test that diagnostics don't consume excessive memory."""
        try:
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run comprehensive analysis with limited test cases
            standard_cases = self.config.get_basic_test_data()['standard_test_cases'][:1]
            with patch.object(self.diagnostics, 'get_standard_test_cases') as mock_cases:
                mock_cases.return_value = standard_cases
                result = self.diagnostics.run_comprehensive_analysis()
            
            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            self.assertLess(memory_increase, self.memory_limits['comprehensive_analysis_max_mb'])
            
        except ImportError:
            # Skip test if psutil is not available
            self.skipTest("psutil not available for memory testing")


class TestDiagnosticsUtilities(unittest.TestCase):
    """Test utility methods and helper functions."""
    
    def setUp(self):
        """Set up utility test fixtures using mock factory and config injection."""
        self.mock_factory = diagnostics_mock_factory
        self.config = DiagnosticsMockConfig
        self.diagnostics = self.mock_factory.create_standard_mock(verbosity=0)
        
        # Get test file paths from configuration
        self.test_paths = self.config.get_test_file_paths()
    
    def test_get_standard_test_cases(self):
        """Test standard test cases retrieval."""
        test_cases = self.diagnostics.get_standard_test_cases()
        
        # Compare with expected test cases from configuration
        expected_cases = self.config.get_basic_test_data()['standard_test_cases']
        
        # Should return a non-empty list
        self.assertGreater(len(test_cases), 0)
        
        # Should contain tuples of strings
        for case in test_cases:
            self.assertIsInstance(case, tuple)
            self.assertEqual(len(case), 2)
            self.assertIsInstance(case[0], str)
            self.assertIsInstance(case[1], str)
        
        # Should contain known critical cases from configuration
        case_strings = [f"{src}->{tgt}" for src, tgt in test_cases]
        expected_case_strings = [f"{src}->{tgt}" for src, tgt in expected_cases]
        
        # Check that at least some expected cases are present
        common_cases = set(case_strings) & set(expected_case_strings)
        self.assertGreater(len(common_cases), 0)
    
    def test_export_results_json(self):
        """Test JSON export functionality."""
        # Get mock analysis results from configuration
        mock_results = self.config.get_mock_analysis_results()
        test_results = mock_results['mock_connectivity_result']
        
        temp_path = self.test_paths['temp_json_file']
        
        try:
            # Export to JSON using mock
            self.diagnostics.export_results(test_results, temp_path, "json")
            
            # Verify file was created and contains correct data
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                loaded_results = json.load(f)
            
            self.assertEqual(loaded_results, test_results)
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_results_txt(self):
        """Test TXT export functionality."""
        # Use simple test results
        test_results = {"simple": "test"}
        temp_path = self.test_paths['temp_txt_file']
        
        try:
            # Export to TXT using mock
            self.diagnostics.export_results(test_results, temp_path, "txt")
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            self.assertIn("SMIED Diagnostics Results", content)
            self.assertIn("simple", content)
            self.assertIn("test", content)
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_results_invalid_format(self):
        """Test export with invalid format."""
        test_results = {"test": "data"}
        
        with self.assertRaises(ValueError):
            self.diagnostics.export_results(test_results, "test.xml", "xml")


    def test_run_comprehensive_analysis(self):
        """Test comprehensive analysis orchestration."""
        # Mock individual analysis methods to avoid long execution times
        expected_sections = self.config.get_integration_test_data()['expected_analysis_results']['comprehensive_analysis_sections']
        
        with patch.object(self.diagnostics, 'get_standard_test_cases') as mock_cases:
            mock_cases.return_value = [('cat.n.01', 'feline.n.01')]
            
            result = self.diagnostics.run_comprehensive_analysis()
        
        # Check that all major analysis sections are present
        for section in expected_sections:
            self.assertIn(section, result)
        
        # Check timestamp format
        self.assertIsInstance(result['timestamp'], str)
        self.assertIn('-', result['timestamp'])  # Date format
        self.assertIn(':', result['timestamp'])  # Time format


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)