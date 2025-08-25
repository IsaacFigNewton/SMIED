"""
Mock implementations for SMIED Diagnostics components.

This module provides mock classes and factory methods for testing the SMIEDDiagnostics
class and related diagnostic functionality. It follows the SMIED testing framework
design specifications with factory pattern, abstract base classes, and specialized
mock variants.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import networkx as nx
from collections import defaultdict
import time

from .base.entity_mock import AbstractEntityMock, EntityType, EntityStatus
from .base.operation_mock import AbstractOperationMock


class AbstractDiagnosticsMock(AbstractEntityMock):
    """
    Abstract base class for diagnostics-related mocks.
    
    Provides common functionality for diagnostic operations, graph analysis,
    and result formatting that all diagnostics mocks should implement.
    """
    
    def __init__(self, name: str = "diagnostics_mock", **kwargs):
        super().__init__(
            entity_type=EntityType.UNKNOWN,
            name=name,
            **kwargs
        )
        
        # Diagnostic-specific attributes
        self.verbosity = kwargs.get('verbosity', 1)
        self.graph = kwargs.get('graph', nx.DiGraph())
        self.semantic_decomposer = kwargs.get('semantic_decomposer', Mock())
        self.embedding_helper = kwargs.get('embedding_helper', Mock())
        self.beam_builder = kwargs.get('beam_builder', Mock())
        
        # Result caching
        self._result_cache = {}
        self._analysis_results = {}
    
    @abstractmethod
    def run_analysis(self, analysis_type: str, *args, **kwargs) -> Dict[str, Any]:
        """Run a specific type of analysis."""
        pass
    
    @abstractmethod
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a formatted report from analysis results."""
        pass
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute of this diagnostics mock."""
        return self.name
    
    def validate_entity(self) -> bool:
        """Validate the diagnostics mock entity."""
        return self.is_valid and bool(self.name) and self.graph is not None
    
    def get_entity_signature(self) -> str:
        """Get unique signature for this diagnostics mock."""
        return f"{self.entity_type.value}:{self.id}:{self.name}:{self.verbosity}"


class MockSMIEDDiagnostics(AbstractDiagnosticsMock):
    """
    Basic mock implementation of SMIEDDiagnostics for standard testing scenarios.
    
    Provides realistic behavior for most diagnostic operations with predictable
    results suitable for unit testing.
    """
    
    def __init__(self, verbosity: int = 1, nlp_func=None, embedding_model=None, **kwargs):
        super().__init__(name="smied_diagnostics", verbosity=verbosity, **kwargs)
        
        # Set up realistic component mocks
        self.semantic_decomposer = self._create_semantic_decomposer_mock()
        self.embedding_helper = self._create_embedding_helper_mock()
        self.beam_builder = self._create_beam_builder_mock()
        
        # Create test graph
        self.graph = self._create_test_graph()
        
        # Configure method behaviors
        self._setup_method_behaviors()
    
    def _create_test_graph(self) -> nx.DiGraph:
        """Create a test graph with realistic WordNet-like structure."""
        graph = nx.DiGraph()
        
        # Add test edges with WordNet-like relations
        test_edges = [
            ('cat.n.01', 'feline.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('feline.n.01', 'carnivore.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('carnivore.n.01', 'animal.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('dog.n.01', 'canine.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('canine.n.01', 'carnivore.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('hunt.v.01', 'search.v.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('chase.v.01', 'pursue.v.01', {'relation': 'similar_to', 'weight': 0.8}),
            ('run.v.01', 'move.v.01', {'relation': 'hypernym', 'weight': 1.0}),
        ]
        
        for src, tgt, attrs in test_edges:
            graph.add_edge(src, tgt, **attrs)
        
        return graph
    
    def _create_semantic_decomposer_mock(self) -> Mock:
        """Create a realistic semantic decomposer mock."""
        mock = Mock()
        mock.build_synset_graph.return_value = self.graph
        mock.verbosity = self.verbosity
        return mock
    
    def _create_embedding_helper_mock(self) -> Mock:
        """Create a realistic embedding helper mock."""
        mock = Mock()
        mock.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock.compute_similarity.return_value = 0.75
        return mock
    
    def _create_beam_builder_mock(self) -> Mock:
        """Create a realistic beam builder mock."""
        mock = Mock()
        mock.build_beam.return_value = ["test.n.01", "another.n.01"]
        return mock
    
    def _setup_method_behaviors(self):
        """Set up realistic behaviors for diagnostic methods."""
        # Configure connectivity analysis
        self.analyze_synset_connectivity = Mock(side_effect=self._mock_analyze_connectivity)
        self.audit_wordnet_relations = Mock(side_effect=self._mock_audit_relations)
        self.analyze_graph_topology = Mock(side_effect=self._mock_analyze_topology)
        self.test_parameter_sensitivity = Mock(side_effect=self._mock_parameter_sensitivity)
        self.validate_pathfinding_fixes = Mock(side_effect=self._mock_validate_fixes)
        self.analyze_cross_pos_connectivity = Mock(side_effect=self._mock_cross_pos_analysis)
        self.analyze_relation_density = Mock(side_effect=self._mock_relation_density)
        self.analyze_heuristic_effectiveness = Mock(side_effect=self._mock_heuristic_effectiveness)
        self.run_comprehensive_analysis = Mock(side_effect=self._mock_comprehensive_analysis)
        self.get_standard_test_cases = Mock(side_effect=self._mock_get_test_cases)
        self.export_results = Mock(side_effect=self._mock_export_results)
    
    def _mock_analyze_connectivity(self, source: str, target: str, max_hops: int = 6) -> Dict[str, Any]:
        """Mock connectivity analysis with realistic results."""
        if source not in self.graph or target not in self.graph:
            return {
                'source': source,
                'target': target,
                'error': f'Synset {source if source not in self.graph else target} not found in graph'
            }
        
        try:
            path = nx.shortest_path(self.graph, source, target)
            path_length = len(path) - 1
            
            return {
                'source': source,
                'target': target,
                'direct_connection': path_length == 1,
                'direct_relation': self.graph[source][target].get('relation') if path_length == 1 else None,
                'shortest_path_length': path_length,
                'paths': [{
                    'path': path,
                    'length': path_length,
                    'weight': 1.0 if path_length == 0 else 1.0 / path_length,
                    'relations': [self.graph[path[i]][path[i+1]].get('relation', 'unknown') 
                                for i in range(len(path)-1)]
                }],
                'connectivity_gaps': []
            }
        except nx.NetworkXNoPath:
            return {
                'source': source,
                'target': target,
                'direct_connection': False,
                'direct_relation': None,
                'shortest_path_length': None,
                'paths': [],
                'connectivity_gaps': [{'type': 'no_path', 'description': 'No path found between synsets'}]
            }
    
    def _mock_audit_relations(self) -> Dict[str, Any]:
        """Mock WordNet relation audit with realistic results."""
        return {
            'implemented_relations': ['hypernym', 'hyponym', 'similar_to', 'meronym', 'holonym'],
            'missing_relations': ['derivationally_related_forms', 'attributes', 'causes', 'entails'],
            'relation_coverage_by_pos': {
                'noun': 0.75,
                'verb': 0.65,
                'adjective': 0.50,
                'adverb': 0.40
            },
            'edge_statistics': {
                'total_edges': self.graph.number_of_edges(),
                'hypernym_edges': len([e for e in self.graph.edges(data=True) if e[2].get('relation') == 'hypernym']),
                'similar_to_edges': len([e for e in self.graph.edges(data=True) if e[2].get('relation') == 'similar_to'])
            },
            'recommendations': [
                {
                    'priority': 'HIGH',
                    'description': 'Add derivationally related forms for better cross-POS connectivity',
                    'estimated_impact': 'Significant improvement in pathfinding success rates'
                },
                {
                    'priority': 'MEDIUM',
                    'description': 'Implement attribute relations for adjective-noun connections',
                    'estimated_impact': 'Moderate improvement in semantic understanding'
                }
            ]
        }
    
    def _mock_analyze_topology(self) -> Dict[str, Any]:
        """Mock graph topology analysis with realistic metrics."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        if num_nodes > 0:
            degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            avg_degree = sum(degrees) / len(degrees)
            max_degree = max(degrees) if degrees else 0
            min_degree = min(degrees) if degrees else 0
        else:
            avg_degree = max_degree = min_degree = 0
        
        # Relation distribution
        relation_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            relation_counts[relation] += 1
        
        # POS distribution (simplified)
        pos_counts = defaultdict(int)
        for node in self.graph.nodes():
            if '.' in node:
                pos = node.split('.')[-1][0]  # Extract POS from synset name
                pos_counts[pos] += 1
        
        return {
            'basic_statistics': {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'is_directed': self.graph.is_directed(),
                'density': nx.density(self.graph) if num_nodes > 0 else 0.0
            },
            'connectivity_analysis': {
                'is_connected': nx.is_weakly_connected(self.graph) if num_nodes > 0 else False,
                'num_components': nx.number_weakly_connected_components(self.graph) if num_nodes > 0 else 0
            },
            'degree_analysis': {
                'average_degree': avg_degree,
                'max_degree': max_degree,
                'min_degree': min_degree
            },
            'relation_distribution': dict(relation_counts),
            'pos_distribution': dict(pos_counts)
        }
    
    def _mock_parameter_sensitivity(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Mock parameter sensitivity analysis."""
        return {
            'test_cases': test_cases,
            'parameter_results': {
                'beam_width_5': {'success_rate': 0.7, 'avg_time': 0.12},
                'beam_width_10': {'success_rate': 0.8, 'avg_time': 0.15},
                'beam_width_15': {'success_rate': 0.85, 'avg_time': 0.20}
            },
            'success_rates': {
                'overall': 0.78,
                'cross_pos': 0.45,
                'same_pos': 0.92
            },
            'performance_metrics': {
                'avg_execution_time': 0.156,
                'memory_usage_mb': 45.2
            },
            'recommendations': [
                'Use beam_width=10 for optimal balance of success rate and performance',
                'Consider different parameters for cross-POS vs same-POS pathfinding'
            ]
        }
    
    def _mock_validate_fixes(self) -> Dict[str, Any]:
        """Mock pathfinding fixes validation."""
        return {
            'critical_test_cases': [
                ('cat.n.01', 'chase.v.01'),
                ('run.v.01', 'exercise.n.01'),
                ('teacher.n.01', 'teach.v.01')
            ],
            'current_failures': [
                {'source': 'cat.n.01', 'target': 'chase.v.01', 'reason': 'No cross-POS relations'},
                {'source': 'run.v.01', 'target': 'exercise.n.01', 'reason': 'Missing derivational links'}
            ],
            'derivational_analysis': {
                'coverage': 0.23,
                'missing_pairs': 156,
                'potential_improvements': 89
            },
            'recommendations': [
                'Implement derivational relation extraction',
                'Add cross-POS similarity heuristics',
                'Expand WordNet with additional relation types'
            ]
        }
    
    def _mock_cross_pos_analysis(self) -> Dict[str, Any]:
        """Mock cross-POS connectivity analysis."""
        return {
            'pos_pairs': ['noun-verb', 'noun-adjective', 'verb-adjective', 'adjective-adverb'],
            'cross_pos_edges': 0,  # Realistic for basic WordNet graph
            'missing_connections': {
                'noun-verb': 245,
                'noun-adjective': 187,
                'verb-adjective': 98,
                'adjective-adverb': 67
            },
            'cross_pos_statistics': {
                'total_possible': 597,
                'cross_pos_edges': 0,
                'coverage_percentage': 0.0
            },
            'recommendations': [
                {
                    'priority': 'CRITICAL',
                    'description': 'Add derivational relations for noun-verb connectivity',
                    'estimated_improvement': '40% increase in cross-POS pathfinding success'
                }
            ]
        }
    
    def _mock_relation_density(self, synsets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Mock relation density analysis."""
        analyzed_synsets = synsets if synsets else list(self.graph.nodes())
        
        return {
            'analyzed_synsets': analyzed_synsets,
            'total_analyzed': len(analyzed_synsets),
            'density_statistics': {
                'average_relations_per_synset': 2.3,
                'max_relations': 8,
                'min_relations': 0,
                'sparse_threshold': 2
            },
            'sparse_nodes': [node for node in analyzed_synsets if self.graph.degree(node) <= 2],
            'well_connected_nodes': [node for node in analyzed_synsets if self.graph.degree(node) > 5],
            'relation_patterns': {
                'hypernym_heavy': 0.65,
                'similar_to_heavy': 0.15,
                'mixed_relations': 0.20
            }
        }
    
    def _mock_heuristic_effectiveness(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Mock heuristic effectiveness analysis."""
        if self.embedding_helper is None:
            return {'error': 'Embedding helper not available'}
        
        return {
            'heuristic_analysis': {
                'embedding_similarity': {
                    'average_accuracy': 0.72,
                    'correlation_with_success': 0.68
                },
                'wordnet_distance': {
                    'average_accuracy': 0.85,
                    'correlation_with_success': 0.79
                },
                'pos_compatibility': {
                    'average_accuracy': 0.91,
                    'correlation_with_success': 0.83
                }
            },
            'test_case_results': [
                {
                    'source': case[0],
                    'target': case[1],
                    'heuristic_scores': {
                        'embedding': 0.75,
                        'wordnet': 0.82,
                        'pos': 0.95
                    },
                    'actual_success': True
                }
                for case in test_cases[:3]  # Limit to first 3 for brevity
            ]
        }
    
    def _mock_comprehensive_analysis(self) -> Dict[str, Any]:
        """Mock comprehensive analysis orchestration."""
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'graph_info': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'components': 1
            },
            'relation_audit': self._mock_audit_relations(),
            'graph_topology': self._mock_analyze_topology(),
            'cross_pos_connectivity': self._mock_cross_pos_analysis(),
            'pathfinding_validation': self._mock_validate_fixes(),
            'parameter_sensitivity': self._mock_parameter_sensitivity([('cat.n.01', 'feline.n.01')]),
            'heuristic_effectiveness': self._mock_heuristic_effectiveness([('cat.n.01', 'feline.n.01')]),
            'performance_profile': {
                'total_analysis_time': 1.25,
                'memory_peak_mb': 123.4
            },
            'connectivity_analysis': {
                'sample_synsets_analyzed': 10,
                'average_connectivity': 2.3
            }
        }
    
    def _mock_get_test_cases(self) -> List[Tuple[str, str]]:
        """Mock standard test cases retrieval."""
        return [
            ('cat.n.01', 'chase.v.01'),
            ('run.v.01', 'exercise.n.01'),
            ('teacher.n.01', 'teach.v.01'),
            ('car.n.01', 'drive.v.01'),
            ('book.n.01', 'read.v.01'),
            ('dog.n.01', 'canine.n.01'),
            ('feline.n.01', 'carnivore.n.01')
        ]
    
    def _mock_export_results(self, results: Dict[str, Any], filepath: str, format_type: str):
        """Mock results export functionality."""
        if format_type not in ['json', 'txt']:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Mock file writing - in real tests, this would write to temp files
        if format_type == 'json':
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
        elif format_type == 'txt':
            with open(filepath, 'w') as f:
                f.write("SMIED Diagnostics Results\n")
                f.write("========================\n\n")
                f.write(str(results))
    
    def run_analysis(self, analysis_type: str, *args, **kwargs) -> Dict[str, Any]:
        """Run a specific type of analysis."""
        analysis_methods = {
            'connectivity': self.analyze_synset_connectivity,
            'topology': self.analyze_graph_topology,
            'relations': self.audit_wordnet_relations,
            'cross_pos': self.analyze_cross_pos_connectivity,
            'density': self.analyze_relation_density,
            'comprehensive': self.run_comprehensive_analysis
        }
        
        if analysis_type not in analysis_methods:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        return analysis_methods[analysis_type](*args, **kwargs)
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a formatted report from analysis results."""
        report = "SMIED Diagnostics Analysis Report\n"
        report += "=" * 35 + "\n\n"
        
        for section, data in analysis_results.items():
            report += f"{section.upper()}:\n"
            report += "-" * len(section) + "\n"
            report += str(data) + "\n\n"
        
        return report


class MockSMIEDDiagnosticsEdgeCases(AbstractDiagnosticsMock):
    """
    Edge case mock implementation for testing error conditions and boundary cases.
    
    This mock is designed to trigger various error conditions and edge cases
    that might occur during diagnostic operations.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="diagnostics_edge_cases", **kwargs)
        
        # Set up problematic scenarios
        self.semantic_decomposer = None  # Simulate initialization failure
        self.embedding_helper = None
        self.beam_builder = None
        
        # Create problematic graph scenarios
        self.empty_graph = nx.DiGraph()
        self.circular_graph = self._create_circular_graph()
        self.disconnected_graph = self._create_disconnected_graph()
        
        # Default to empty graph for edge case testing
        self.graph = self.empty_graph
        
        # Configure methods to trigger edge cases
        self._setup_edge_case_behaviors()
    
    def _create_circular_graph(self) -> nx.DiGraph:
        """Create a graph with circular dependencies."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('a.n.01', 'b.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('b.n.01', 'c.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('c.n.01', 'a.n.01', {'relation': 'hypernym', 'weight': 1.0})
        ])
        return graph
    
    def _create_disconnected_graph(self) -> nx.DiGraph:
        """Create a graph with disconnected components."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('group1_a.n.01', 'group1_b.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('group2_a.n.01', 'group2_b.n.01', {'relation': 'hypernym', 'weight': 1.0})
        ])
        # Note: no edges between group1 and group2
        return graph
    
    def _setup_edge_case_behaviors(self):
        """Set up behaviors that trigger edge cases."""
        self.analyze_synset_connectivity = Mock(side_effect=self._edge_case_connectivity)
        self.analyze_graph_topology = Mock(side_effect=self._edge_case_topology)
        self.run_comprehensive_analysis = Mock(side_effect=self._edge_case_comprehensive)
    
    def _edge_case_connectivity(self, source: str, target: str, max_hops: int = 6) -> Dict[str, Any]:
        """Connectivity analysis that triggers various edge cases."""
        # Simulate missing synsets
        if 'missing' in source or 'missing' in target:
            return {
                'source': source,
                'target': target,
                'error': f'Synset {source if "missing" in source else target} not found in graph'
            }
        
        # Simulate circular path detection
        if self.graph == self.circular_graph:
            return {
                'source': source,
                'target': target,
                'direct_connection': False,
                'shortest_path_length': None,
                'paths': [],
                'connectivity_gaps': [{'type': 'circular_dependency', 'description': 'Circular path detected'}]
            }
        
        # Simulate no path in disconnected graph
        return {
            'source': source,
            'target': target,
            'direct_connection': False,
            'shortest_path_length': None,
            'paths': [],
            'connectivity_gaps': [{'type': 'disconnected_components', 'description': 'Synsets in different components'}]
        }
    
    def _edge_case_topology(self) -> Dict[str, Any]:
        """Topology analysis for edge case graphs."""
        return {
            'basic_statistics': {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'is_directed': True,
                'density': 0.0
            },
            'connectivity_analysis': {
                'is_connected': False,
                'num_components': self.graph.number_of_nodes() if self.graph.number_of_edges() == 0 else 2
            },
            'degree_analysis': {
                'average_degree': 0.0,
                'max_degree': 0,
                'min_degree': 0
            },
            'relation_distribution': {},
            'pos_distribution': {}
        }
    
    def _edge_case_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive analysis that includes error conditions."""
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'graph_info': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'components': 0 if self.graph.number_of_nodes() == 0 else 1
            },
            'errors': [
                'Semantic decomposer initialization failed',
                'Embedding helper not available',
                'Beam builder initialization failed'
            ],
            'partial_results': {
                'relation_audit': {'error': 'Cannot audit empty graph'},
                'graph_topology': self._edge_case_topology()
            }
        }
    
    def run_analysis(self, analysis_type: str, *args, **kwargs) -> Dict[str, Any]:
        """Run analysis that may encounter edge cases."""
        if analysis_type == 'connectivity':
            return self._edge_case_connectivity(*args, **kwargs)
        elif analysis_type == 'topology':
            return self._edge_case_topology()
        elif analysis_type == 'comprehensive':
            return self._edge_case_comprehensive()
        else:
            raise ValueError(f"Analysis type '{analysis_type}' not supported in edge case testing")
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate report that includes error information."""
        report = "SMIED Diagnostics Edge Case Report\n"
        report += "=" * 35 + "\n\n"
        
        if 'errors' in analysis_results:
            report += "ERRORS ENCOUNTERED:\n"
            for error in analysis_results['errors']:
                report += f"- {error}\n"
            report += "\n"
        
        report += "PARTIAL RESULTS:\n"
        for section, data in analysis_results.get('partial_results', {}).items():
            report += f"{section}: {data}\n"
        
        return report


class MockSMIEDDiagnosticsIntegration(AbstractDiagnosticsMock):
    """
    Integration mock implementation for testing with realistic component interactions.
    
    This mock provides more realistic behavior suitable for integration testing
    with actual SMIED components.
    """
    
    def __init__(self, **kwargs):
        # Remove 'name' from kwargs if it exists to avoid duplicate parameter
        kwargs.pop('name', None)
        super().__init__(name="diagnostics_integration", **kwargs)
        
        # Create more realistic component mocks
        self.semantic_decomposer = self._create_realistic_decomposer()
        self.embedding_helper = self._create_realistic_embedding_helper()
        self.beam_builder = self._create_realistic_beam_builder()
        
        # Create larger, more realistic graph
        self.graph = self._create_realistic_graph()
        
        # Set up integration-focused behaviors
        self._setup_integration_behaviors()
    
    def _create_realistic_graph(self) -> nx.DiGraph:
        """Create a more realistic WordNet-like graph for integration testing."""
        graph = nx.DiGraph()
        
        # Add a more comprehensive set of synsets and relations
        realistic_edges = [
            # Animal hierarchy
            ('cat.n.01', 'feline.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('dog.n.01', 'canine.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('feline.n.01', 'carnivore.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('canine.n.01', 'carnivore.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('carnivore.n.01', 'animal.n.01', {'relation': 'hypernym', 'weight': 1.0}),
            
            # Action hierarchy  
            ('run.v.01', 'move.v.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('walk.v.01', 'move.v.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('chase.v.01', 'pursue.v.01', {'relation': 'hypernym', 'weight': 1.0}),
            ('hunt.v.01', 'search.v.01', {'relation': 'hypernym', 'weight': 1.0}),
            
            # Similarity relations
            ('chase.v.01', 'pursue.v.01', {'relation': 'similar_to', 'weight': 0.9}),
            ('run.v.01', 'jog.v.01', {'relation': 'similar_to', 'weight': 0.8}),
            
            # Cross-POS relations (limited, as in real WordNet)
            ('teacher.n.01', 'teach.v.01', {'relation': 'derivationally_related_form', 'weight': 1.0}),
            ('runner.n.01', 'run.v.01', {'relation': 'derivationally_related_form', 'weight': 1.0}),
        ]
        
        for src, tgt, attrs in realistic_edges:
            graph.add_edge(src, tgt, **attrs)
        
        return graph
    
    def _create_realistic_decomposer(self) -> Mock:
        """Create a more realistic semantic decomposer mock."""
        mock = Mock()
        mock.build_synset_graph.return_value = self.graph
        mock.verbosity = self.verbosity
        mock.get_synset_definitions = Mock(return_value={
            'cat.n.01': 'a small domesticated carnivorous mammal',
            'dog.n.01': 'a domesticated carnivorous mammal',
            'run.v.01': 'move fast by using one\'s feet'
        })
        return mock
    
    def _create_realistic_embedding_helper(self) -> Mock:
        """Create a more realistic embedding helper mock."""
        mock = Mock()
        
        def embedding_side_effect(word):
            # Return different embeddings for different words
            word_embeddings = {
                'cat': [0.2, 0.3, 0.1, 0.5, 0.4],
                'dog': [0.3, 0.2, 0.1, 0.4, 0.5],
                'run': [0.1, 0.5, 0.3, 0.2, 0.4],
                'chase': [0.2, 0.4, 0.3, 0.3, 0.3]
            }
            return word_embeddings.get(word, [0.1, 0.1, 0.1, 0.1, 0.1])
        
        mock.get_embedding.side_effect = embedding_side_effect
        
        def similarity_side_effect(emb1, emb2):
            # Simple cosine similarity mock
            return sum(a * b for a, b in zip(emb1, emb2)) / (
                (sum(a * a for a in emb1) * sum(b * b for b in emb2)) ** 0.5
            )
        
        mock.compute_similarity.side_effect = similarity_side_effect
        return mock
    
    def _create_realistic_beam_builder(self) -> Mock:
        """Create a more realistic beam builder mock."""
        mock = Mock()
        
        def beam_side_effect(source_synsets, beam_width=10):
            # Return related synsets based on graph structure
            related = set()
            for synset in source_synsets:
                if synset in self.graph:
                    related.update(self.graph.successors(synset))
                    related.update(self.graph.predecessors(synset))
            return list(related)[:beam_width]
        
        mock.build_beam.side_effect = beam_side_effect
        return mock
    
    def _setup_integration_behaviors(self):
        """Set up behaviors for integration testing."""
        # Use the standard mock behaviors but with more realistic data
        mock_standard = MockSMIEDDiagnostics()
        
        self.analyze_synset_connectivity = Mock(side_effect=mock_standard._mock_analyze_connectivity)
        self.audit_wordnet_relations = Mock(side_effect=mock_standard._mock_audit_relations)
        self.analyze_graph_topology = Mock(side_effect=mock_standard._mock_analyze_topology)
        self.run_comprehensive_analysis = Mock(side_effect=mock_standard._mock_comprehensive_analysis)
        self.get_standard_test_cases = Mock(side_effect=mock_standard._mock_get_test_cases)
    
    def run_analysis(self, analysis_type: str, *args, **kwargs) -> Dict[str, Any]:
        """Run analysis with integration-focused behavior."""
        # Delegate to appropriate mock method based on analysis type
        if analysis_type == 'connectivity':
            return self.analyze_synset_connectivity(*args, **kwargs)
        elif analysis_type == 'topology':
            return self.analyze_graph_topology()
        elif analysis_type == 'comprehensive':
            return self.run_comprehensive_analysis()
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate detailed integration test report."""
        report = "SMIED Diagnostics Integration Test Report\n"
        report += "=" * 42 + "\n\n"
        
        report += f"Graph Statistics:\n"
        report += f"- Nodes: {self.graph.number_of_nodes()}\n"
        report += f"- Edges: {self.graph.number_of_edges()}\n"
        report += f"- Relations: {len(set(data.get('relation', 'unknown') for _, _, data in self.graph.edges(data=True)))}\n\n"
        
        for section, data in analysis_results.items():
            report += f"{section.upper()}:\n"
            report += "-" * (len(section) + 1) + "\n"
            if isinstance(data, dict):
                for key, value in data.items():
                    report += f"  {key}: {value}\n"
            else:
                report += f"  {data}\n"
            report += "\n"
        
        return report


class DiagnosticsMockFactory:
    """
    Factory class for creating diagnostic mocks following the factory pattern.
    
    Provides standardized creation of different types of diagnostic mocks
    for various testing scenarios.
    """
    
    def __init__(self):
        """Initialize the mock factory with available mock types."""
        self._mock_classes = {
            'MockSMIEDDiagnostics': MockSMIEDDiagnostics,
            'MockSMIEDDiagnosticsEdgeCases': MockSMIEDDiagnosticsEdgeCases,
            'MockSMIEDDiagnosticsIntegration': MockSMIEDDiagnosticsIntegration,
        }
    
    def __call__(self, mock_name: str, *args, **kwargs) -> AbstractDiagnosticsMock:
        """
        Create a mock instance of the specified type.
        
        Args:
            mock_name (str): Name of the mock class to create
            *args: Positional arguments for mock constructor
            **kwargs: Keyword arguments for mock constructor
            
        Returns:
            AbstractDiagnosticsMock: Instance of the requested mock type
            
        Raises:
            ValueError: If mock_name is not a valid mock type
        """
        if mock_name not in self._mock_classes:
            available = ', '.join(self._mock_classes.keys())
            raise ValueError(f"Unknown mock type '{mock_name}'. Available: {available}")
        
        mock_class = self._mock_classes[mock_name]
        return mock_class(*args, **kwargs)
    
    def get_available_mocks(self) -> List[str]:
        """
        Get list of available mock types.
        
        Returns:
            List[str]: List of available mock class names
        """
        return list(self._mock_classes.keys())
    
    def create_standard_mock(self, **kwargs) -> MockSMIEDDiagnostics:
        """
        Create a standard diagnostics mock for basic testing.
        
        Args:
            **kwargs: Keyword arguments for mock constructor
            
        Returns:
            MockSMIEDDiagnostics: Standard diagnostics mock instance
        """
        return self('MockSMIEDDiagnostics', **kwargs)
    
    def create_edge_case_mock(self, **kwargs) -> MockSMIEDDiagnosticsEdgeCases:
        """
        Create an edge case diagnostics mock for error condition testing.
        
        Args:
            **kwargs: Keyword arguments for mock constructor
            
        Returns:
            MockSMIEDDiagnosticsEdgeCases: Edge case diagnostics mock instance
        """
        return self('MockSMIEDDiagnosticsEdgeCases', **kwargs)
    
    def create_integration_mock(self, **kwargs) -> MockSMIEDDiagnosticsIntegration:
        """
        Create an integration diagnostics mock for component interaction testing.
        
        Args:
            **kwargs: Keyword arguments for mock constructor
            
        Returns:
            MockSMIEDDiagnosticsIntegration: Integration diagnostics mock instance
        """
        return self('MockSMIEDDiagnosticsIntegration', **kwargs)


# Create factory instance for easy import
diagnostics_mock_factory = DiagnosticsMockFactory()