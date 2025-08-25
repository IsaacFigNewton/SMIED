"""
Configuration class containing mock constants and test data for SMIED Diagnostics tests.

This module provides structured test data and configuration for the SMIEDDiagnostics
testing framework, following the SMIED Testing Framework Design Specifications.
"""

from typing import Dict, List, Any, Tuple, Optional
import networkx as nx


class DiagnosticsMockConfig:
    """Configuration class containing test data and constants for diagnostics tests."""
    
    @staticmethod
    def get_basic_test_data() -> Dict[str, Any]:
        """Get basic test data for standard diagnostics functionality."""
        return {
            'test_graph_edges': [
                ('cat.n.01', 'feline.n.01', {'relation': 'hypernym', 'weight': 1.0}),
                ('feline.n.01', 'carnivore.n.01', {'relation': 'hypernym', 'weight': 1.0}),
                ('carnivore.n.01', 'animal.n.01', {'relation': 'hypernym', 'weight': 1.0}),
                ('dog.n.01', 'canine.n.01', {'relation': 'hypernym', 'weight': 1.0}),
                ('canine.n.01', 'carnivore.n.01', {'relation': 'hypernym', 'weight': 1.0}),
                ('hunt.v.01', 'search.v.01', {'relation': 'hypernym', 'weight': 1.0}),
                ('chase.v.01', 'pursue.v.01', {'relation': 'similar_to', 'weight': 0.8}),
                ('run.v.01', 'move.v.01', {'relation': 'hypernym', 'weight': 1.0}),
            ],
            'standard_test_cases': [
                ('cat.n.01', 'chase.v.01'),
                ('run.v.01', 'exercise.n.01'),
                ('teacher.n.01', 'teach.v.01'),
                ('car.n.01', 'drive.v.01'),
                ('book.n.01', 'read.v.01'),
                ('dog.n.01', 'canine.n.01'),
                ('feline.n.01', 'carnivore.n.01')
            ],
            'connectivity_test_pairs': [
                {
                    'source': 'cat.n.01',
                    'target': 'feline.n.01',
                    'expected_direct': True,
                    'expected_relation': 'hypernym',
                    'expected_path_length': 1
                },
                {
                    'source': 'cat.n.01',
                    'target': 'carnivore.n.01',
                    'expected_direct': False,
                    'expected_relation': None,
                    'expected_path_length': 2
                },
                {
                    'source': 'dog.n.01',
                    'target': 'animal.n.01',
                    'expected_direct': False,
                    'expected_relation': None,
                    'expected_path_length': 3
                }
            ]
        }
    
    @staticmethod
    def get_edge_case_scenarios() -> Dict[str, Any]:
        """Get edge case scenarios for error condition testing."""
        return {
            'empty_graph_data': {
                'nodes': [],
                'edges': [],
                'expected_topology': {
                    'num_nodes': 0,
                    'num_edges': 0,
                    'density': 0.0,
                    'is_connected': False
                }
            },
            'circular_graph_data': {
                'edges': [
                    ('a.n.01', 'b.n.01', {'relation': 'hypernym', 'weight': 1.0}),
                    ('b.n.01', 'c.n.01', {'relation': 'hypernym', 'weight': 1.0}),
                    ('c.n.01', 'a.n.01', {'relation': 'hypernym', 'weight': 1.0})
                ],
                'expected_behavior': 'handle_cycles_gracefully'
            },
            'disconnected_graph_data': {
                'edges': [
                    ('group1_a.n.01', 'group1_b.n.01', {'relation': 'hypernym', 'weight': 1.0}),
                    ('group2_a.n.01', 'group2_b.n.01', {'relation': 'hypernym', 'weight': 1.0})
                ],
                'expected_components': 2,
                'expected_no_path_pairs': [('group1_a.n.01', 'group2_a.n.01')]
            },
            'missing_synset_scenarios': [
                {
                    'source': 'missing.n.01',
                    'target': 'cat.n.01',
                    'expected_error': 'not found in graph'
                },
                {
                    'source': 'cat.n.01',
                    'target': 'missing.n.01',
                    'expected_error': 'not found in graph'
                }
            ],
            'invalid_synset_names': [
                'invalid_synset',
                'another_invalid',
                '',
                None
            ],
            'performance_stress_data': {
                'large_graph_nodes': 100,
                'max_processing_time_seconds': 30,
                'memory_threshold_mb': 500
            }
        }
    
    @staticmethod
    def get_integration_test_data() -> Dict[str, Any]:
        """Get integration test data for component interaction testing."""
        return {
            'realistic_graph_edges': [
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
                
                # Cross-POS relations
                ('teacher.n.01', 'teach.v.01', {'relation': 'derivationally_related_form', 'weight': 1.0}),
                ('runner.n.01', 'run.v.01', {'relation': 'derivationally_related_form', 'weight': 1.0}),
            ],
            'component_configurations': {
                'semantic_decomposer': {
                    'verbosity': 1,
                    'build_synset_graph': True
                },
                'embedding_helper': {
                    'embedding_dimension': 5,
                    'similarity_threshold': 0.5
                },
                'beam_builder': {
                    'default_beam_width': 10,
                    'max_beam_width': 20
                }
            },
            'expected_analysis_results': {
                'comprehensive_analysis_sections': [
                    'timestamp', 'graph_info', 'relation_audit', 'graph_topology',
                    'cross_pos_connectivity', 'pathfinding_validation',
                    'parameter_sensitivity', 'heuristic_effectiveness',
                    'performance_profile', 'connectivity_analysis'
                ],
                'relation_audit_keys': [
                    'implemented_relations', 'missing_relations', 
                    'relation_coverage_by_pos', 'edge_statistics', 'recommendations'
                ],
                'topology_analysis_keys': [
                    'basic_statistics', 'connectivity_analysis', 'degree_analysis',
                    'relation_distribution', 'pos_distribution'
                ]
            }
        }
    
    @staticmethod
    def get_mock_analysis_results() -> Dict[str, Any]:
        """Get mock analysis results for testing report generation and validation."""
        return {
            'mock_connectivity_result': {
                'source': 'cat.n.01',
                'target': 'feline.n.01',
                'direct_connection': True,
                'direct_relation': 'hypernym',
                'shortest_path_length': 1,
                'paths': [{
                    'path': ['cat.n.01', 'feline.n.01'],
                    'length': 1,
                    'weight': 1.0,
                    'relations': ['hypernym']
                }],
                'connectivity_gaps': []
            },
            'mock_audit_result': {
                'implemented_relations': ['hypernym', 'hyponym', 'similar_to', 'meronym', 'holonym'],
                'missing_relations': ['derivationally_related_forms', 'attributes', 'causes', 'entails'],
                'relation_coverage_by_pos': {
                    'noun': 0.75,
                    'verb': 0.65,
                    'adjective': 0.50,
                    'adverb': 0.40
                },
                'edge_statistics': {
                    'total_edges': 8,
                    'hypernym_edges': 6,
                    'similar_to_edges': 1
                },
                'recommendations': [
                    {
                        'priority': 'HIGH',
                        'description': 'Add derivationally related forms for better cross-POS connectivity',
                        'estimated_impact': 'Significant improvement in pathfinding success rates'
                    }
                ]
            },
            'mock_topology_result': {
                'basic_statistics': {
                    'num_nodes': 10,
                    'num_edges': 8,
                    'is_directed': True,
                    'density': 0.089
                },
                'connectivity_analysis': {
                    'is_connected': True,
                    'num_components': 1
                },
                'degree_analysis': {
                    'average_degree': 1.6,
                    'max_degree': 3,
                    'min_degree': 1
                },
                'relation_distribution': {
                    'hypernym': 6,
                    'similar_to': 1,
                    'derivationally_related_form': 1
                },
                'pos_distribution': {
                    'n': 6,
                    'v': 4
                }
            },
            'mock_parameter_sensitivity_result': {
                'test_cases': [('cat.n.01', 'feline.n.01')],
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
                    'Use beam_width=10 for optimal balance of success rate and performance'
                ]
            }
        }
    
    @staticmethod
    def get_validation_test_data() -> Dict[str, Any]:
        """Get validation test data for input/output verification."""
        return {
            'valid_synset_names': [
                'cat.n.01',
                'dog.n.01',
                'run.v.01',
                'happy.a.01',
                'quickly.r.01'
            ],
            'invalid_synset_names': [
                'invalid_synset',
                'missing.format',
                '',
                None
            ],
            'valid_max_hops_values': [1, 2, 3, 6, 10],
            'invalid_max_hops_values': [-1, 0, 1001, None, 'string'],
            'valid_verbosity_levels': [0, 1, 2],
            'invalid_verbosity_levels': [-1, 3, 'verbose', None],
            'expected_result_formats': {
                'connectivity_analysis': {
                    'required_keys': ['source', 'target', 'paths'],
                    'optional_keys': ['direct_connection', 'direct_relation', 'shortest_path_length', 'connectivity_gaps']
                },
                'topology_analysis': {
                    'required_keys': ['basic_statistics', 'connectivity_analysis'],
                    'optional_keys': ['degree_analysis', 'relation_distribution', 'pos_distribution']
                },
                'comprehensive_analysis': {
                    'required_keys': ['timestamp', 'graph_info'],
                    'optional_keys': ['relation_audit', 'graph_topology', 'cross_pos_connectivity']
                }
            }
        }
    
    @staticmethod
    def get_performance_benchmarks() -> Dict[str, Any]:
        """Get performance benchmarks and expectations."""
        return {
            'time_limits': {
                'connectivity_analysis': 5.0,  # seconds
                'topology_analysis': 10.0,
                'comprehensive_analysis': 90.0,
                'single_operation_max': 30.0
            },
            'memory_limits': {
                'baseline_mb': 100,
                'connectivity_analysis_max_mb': 150,
                'comprehensive_analysis_max_mb': 500,
                'stress_test_max_mb': 1000
            },
            'graph_size_benchmarks': {
                'small_graph': {
                    'nodes': 10,
                    'edges': 15,
                    'expected_processing_time_ms': 100
                },
                'medium_graph': {
                    'nodes': 100,
                    'edges': 200,
                    'expected_processing_time_ms': 1000
                },
                'large_graph': {
                    'nodes': 1000,
                    'edges': 2000,
                    'expected_processing_time_ms': 10000
                }
            },
            'success_rate_expectations': {
                'same_pos_connectivity': 0.85,
                'cross_pos_connectivity': 0.40,
                'overall_pathfinding': 0.70,
                'relation_audit_coverage': 0.60
            }
        }
    
    @staticmethod
    def get_error_scenarios() -> Dict[str, Any]:
        """Get error scenarios for testing exception handling."""
        return {
            'initialization_errors': [
                {
                    'scenario': 'missing_semantic_decomposer',
                    'error_type': 'ComponentInitializationError',
                    'expected_behavior': 'graceful_degradation'
                },
                {
                    'scenario': 'invalid_verbosity',
                    'error_type': 'ValueError',
                    'expected_behavior': 'use_default_verbosity'
                }
            ],
            'analysis_errors': [
                {
                    'scenario': 'graph_not_initialized',
                    'error_type': 'AttributeError',
                    'expected_behavior': 'return_error_result'
                },
                {
                    'scenario': 'circular_dependency',
                    'error_type': 'RecursionError',
                    'expected_behavior': 'detect_and_handle_cycles'
                }
            ],
            'export_errors': [
                {
                    'scenario': 'invalid_format',
                    'error_type': 'ValueError',
                    'expected_behavior': 'raise_descriptive_error'
                },
                {
                    'scenario': 'file_permission_denied',
                    'error_type': 'PermissionError',
                    'expected_behavior': 'suggest_alternative_path'
                }
            ]
        }
    
    @staticmethod
    def get_mock_component_data() -> Dict[str, Any]:
        """Get mock component data for dependency injection."""
        return {
            'mock_nlp_function': {
                'type': 'function',
                'behavior': 'return_mock_doc_with_tokens'
            },
            'mock_embedding_model': {
                'type': 'model',
                'embedding_size': 5,
                'vocabulary_size': 1000
            },
            'mock_wordnet_data': {
                'synsets': {
                    'cat.n.01': {
                        'pos': 'n',
                        'definition': 'feline mammal',
                        'lemmas': ['cat', 'true_cat']
                    },
                    'run.v.01': {
                        'pos': 'v', 
                        'definition': 'move fast using legs',
                        'lemmas': ['run']
                    }
                }
            }
        }
    
    @staticmethod
    def get_test_file_paths() -> Dict[str, str]:
        """Get test file paths for export functionality testing."""
        return {
            'temp_json_file': 'temp_diagnostics_test.json',
            'temp_txt_file': 'temp_diagnostics_test.txt',
            'invalid_path': '/invalid/path/file.json',
            'readonly_path': '/readonly/file.json'
        }