"""
Configuration class containing mock constants and test data for FrameNet integration tests.
"""


class FrameNetIntegrationMockConfig:
    """Configuration class containing mock constants and test data for FrameNet integration tests."""
    
    @staticmethod
    def get_framenet_srl_test_data():
        """Get test data for FrameNet SRL component testing."""
        return {
            'basic_srl_config': {
                'min_confidence': 0.4,
                'nlp_model': 'en_core_web_sm',
                'enable_wordnet_expansion': True
            },
            'validation_srl_config': {
                'min_confidence': 0.0,
                'nlp_model': None,
                'enable_wordnet_expansion': False
            },
            'edge_case_srl_config': {
                'min_confidence': 1.5,  # Invalid confidence
                'nlp_model': 'nonexistent_model',
                'enable_wordnet_expansion': True
            },
            'integration_srl_config': {
                'min_confidence': 0.6,
                'nlp_model': 'en_core_web_lg',
                'enable_wordnet_expansion': True
            }
        }
    
    @staticmethod
    def get_frame_data_structures():
        """Get test data for FrameNet data structures."""
        return {
            'frame_instances': [
                {
                    'name': 'Cotheme',
                    'confidence': 0.85,
                    'definition': 'One entity (the Theme) moves together with another (the Cotheme)',
                    'lexical_unit': 'chase.v'
                },
                {
                    'name': 'Self_motion',
                    'confidence': 0.90,
                    'definition': 'A Self_mover moves under their own direction along a Path',
                    'lexical_unit': 'run.v'
                },
                {
                    'name': 'Animals',
                    'confidence': 0.95,
                    'definition': 'This frame contains general words for animals',
                    'lexical_unit': 'cat.n'
                }
            ],
            'frame_elements': [
                {
                    'name': 'Theme',
                    'frame_name': 'Cotheme',
                    'confidence': 0.8,
                    'fe_type': 'Core',
                    'definition': 'The entity that moves together with the Cotheme'
                },
                {
                    'name': 'Cotheme', 
                    'frame_name': 'Cotheme',
                    'confidence': 0.75,
                    'fe_type': 'Core',
                    'definition': 'The entity that the Theme moves together with'
                },
                {
                    'name': 'Agent',
                    'frame_name': 'Self_motion',
                    'confidence': 0.85,
                    'fe_type': 'Core',
                    'definition': 'The entity that moves under its own power'
                },
                {
                    'name': 'Goal',
                    'frame_name': 'Self_motion',
                    'confidence': 0.70,
                    'fe_type': 'Non-Core',
                    'definition': 'The destination of the motion'
                },
                {
                    'name': 'Animal',
                    'frame_name': 'Animals',
                    'confidence': 0.90,
                    'fe_type': 'Core',
                    'definition': 'The animal being referenced'
                }
            ]
        }
    
    @staticmethod
    def get_semantic_decomposer_enhancement_data():
        """Get test data for enhanced SemanticDecomposer functionality."""
        return {
            'basic_enhancement': {
                'synsets': ['cat.n.01', 'chase.v.01', 'mouse.n.01'],
                'expected_frames': ['Animals', 'Cotheme'],
                'expected_connections': [('cat.n.01', 'chase.v.01'), ('chase.v.01', 'mouse.n.01')],
                'beam_width': 3,
                'max_depth': 6
            },
            'validation_enhancement': {
                'synsets': [],  # Empty synsets for validation
                'expected_frames': [],
                'expected_connections': [],
                'beam_width': 0,  # Invalid beam width
                'max_depth': -1   # Invalid max depth
            },
            'edge_case_enhancement': {
                'synsets': ['nonexistent.n.01', 'invalid.v.01'],
                'expected_frames': [],
                'expected_connections': [],
                'beam_width': 100,  # Very large beam width
                'max_depth': 50     # Very large max depth
            },
            'integration_enhancement': {
                'synsets': ['scientist.n.01', 'study.v.01', 'organism.n.01'],
                'expected_frames': ['Research', 'Education_teaching'],
                'expected_connections': [
                    ('scientist.n.01', 'study.v.01'),
                    ('study.v.01', 'organism.n.01')
                ],
                'beam_width': 5,
                'max_depth': 8
            }
        }
    
    @staticmethod
    def get_wordnet_synset_data():
        """Get WordNet synset data for FrameNet integration testing."""
        return {
            'animal_synsets': [
                {
                    'name': 'cat.n.01',
                    'definition': 'feline mammal usually having thick soft fur and no ability to roar',
                    'pos': 'n',
                    'lemmas': ['cat', 'true_cat'],
                    'examples': ['cats are often kept as pets']
                },
                {
                    'name': 'mouse.n.01',
                    'definition': 'any of numerous small rodents typically resembling diminutive rats',
                    'pos': 'n',
                    'lemmas': ['mouse'],
                    'examples': ['a mouse ran across the floor']
                },
                {
                    'name': 'animal.n.01',
                    'definition': 'a living organism characterized by voluntary movement',
                    'pos': 'n',
                    'lemmas': ['animal', 'animate_being', 'beast', 'brute', 'creature', 'fauna'],
                    'examples': ['animals in the zoo']
                }
            ],
            'action_synsets': [
                {
                    'name': 'chase.v.01',
                    'definition': 'go after with the intent to catch',
                    'pos': 'v',
                    'lemmas': ['chase', 'chase_after', 'trail', 'tail', 'tag', 'give_chase', 'dog', 'go_after', 'track'],
                    'examples': ['The policeman chased the mugger down the alley']
                },
                {
                    'name': 'run.v.01',
                    'definition': 'move fast by using one\'s feet, with one foot off the ground at any given time',
                    'pos': 'v',
                    'lemmas': ['run'],
                    'examples': ['Don\'t run--you\'ll be out of breath']
                },
                {
                    'name': 'hunt.v.01',
                    'definition': 'pursue for food or sport (as of wild animals)',
                    'pos': 'v',
                    'lemmas': ['hunt', 'run', 'hunt_down', 'track_down'],
                    'examples': ['Goering often hunted wild boars in Poland']
                }
            ],
            'location_synsets': [
                {
                    'name': 'park.n.01',
                    'definition': 'a large area of land preserved in its natural state as public property',
                    'pos': 'n',
                    'lemmas': ['park'],
                    'examples': ['there are laws that protect the wildlife in this park']
                },
                {
                    'name': 'forest.n.01',
                    'definition': 'the trees and other plants in a large densely wooded area',
                    'pos': 'n',
                    'lemmas': ['forest', 'wood', 'woods'],
                    'examples': ['they went for a walk in the forest']
                }
            ]
        }
    
    @staticmethod
    def get_derivational_morphology_data():
        """Get derivational morphology test data."""
        return {
            'basic_derivations': {
                'hunt': {
                    'derivatives': ['hunter', 'hunting'],
                    'relations': [
                        {'source': 'hunt.v.01', 'target': 'hunter.n.01', 'type': 'agentive'},
                        {'source': 'hunt.v.01', 'target': 'hunting.n.01', 'type': 'gerundive'}
                    ]
                },
                'chase': {
                    'derivatives': ['chaser', 'chasing'],
                    'relations': [
                        {'source': 'chase.v.01', 'target': 'chaser.n.01', 'type': 'agentive'},
                        {'source': 'chase.v.01', 'target': 'chasing.n.01', 'type': 'gerundive'}
                    ]
                },
                'teach': {
                    'derivatives': ['teacher', 'teaching'],
                    'relations': [
                        {'source': 'teach.v.01', 'target': 'teacher.n.01', 'type': 'agentive'},
                        {'source': 'teach.v.01', 'target': 'teaching.n.01', 'type': 'gerundive'}
                    ]
                }
            },
            'complex_derivations': {
                'analyze': {
                    'derivatives': ['analyzer', 'analysis', 'analytical', 'analytically'],
                    'relations': [
                        {'source': 'analyze.v.01', 'target': 'analyzer.n.01', 'type': 'agentive'},
                        {'source': 'analyze.v.01', 'target': 'analysis.n.01', 'type': 'nominalization'},
                        {'source': 'analyze.v.01', 'target': 'analytical.a.01', 'type': 'adjectival'},
                        {'source': 'analytical.a.01', 'target': 'analytically.r.01', 'type': 'adverbial'}
                    ]
                }
            },
            'edge_case_derivations': {
                'go': {
                    'derivatives': [],  # No clear derivational forms
                    'relations': []
                },
                'be': {
                    'derivatives': ['being'],  # Limited derivational forms
                    'relations': [
                        {'source': 'be.v.01', 'target': 'being.n.01', 'type': 'gerundive'}
                    ]
                }
            }
        }
    
    @staticmethod
    def get_pathfinding_scenarios():
        """Get pathfinding scenarios for FrameNet integration."""
        return {
            'framenet_pathfinding': {
                'scenario_1': {
                    'subject_synset': 'cat.n.01',
                    'predicate_synset': 'chase.v.01',
                    'object_synset': 'mouse.n.01',
                    'expected_frame': 'Cotheme',
                    'expected_subject_elements': ['Theme'],
                    'expected_object_elements': ['Cotheme'],
                    'expected_path_length': [1, 3]
                },
                'scenario_2': {
                    'subject_synset': 'scientist.n.01',
                    'predicate_synset': 'study.v.01',
                    'object_synset': 'organism.n.01',
                    'expected_frame': 'Research',
                    'expected_subject_elements': ['Researcher'],
                    'expected_object_elements': ['Topic'],
                    'expected_path_length': [2, 4]
                }
            },
            'derivational_pathfinding': {
                'scenario_1': {
                    'source_synset': 'hunt.v.01',
                    'target_synset': 'hunter.n.01',
                    'derivation_type': 'agentive',
                    'expected_path_length': 1,
                    'confidence_threshold': 0.9
                },
                'scenario_2': {
                    'source_synset': 'teach.v.01',
                    'target_synset': 'teaching.n.01',
                    'derivation_type': 'gerundive',
                    'expected_path_length': 1,
                    'confidence_threshold': 0.8
                }
            },
            'gloss_based_pathfinding': {
                'scenario_1': {
                    'source_synset': 'cat.n.01',
                    'target_synset': 'mammal.n.01',
                    'gloss_overlap': ['mammal', 'feline'],
                    'expected_path_length': [2, 4],
                    'similarity_threshold': 0.6
                }
            }
        }
    
    @staticmethod
    def get_cascading_strategy_data():
        """Get data for cascading strategy testing."""
        return {
            'strategy_order': [
                'framenet_subject_predicate',
                'framenet_predicate_object',
                'derivational_subject_predicate',
                'derivational_predicate_object',
                'gloss_based_subject_predicate',
                'gloss_based_predicate_object'
            ],
            'strategy_success_scenarios': {
                'framenet_success': {
                    'triple': ('cat', 'chase', 'mouse'),
                    'successful_strategy': 'framenet_subject_predicate',
                    'fallback_used': False
                },
                'derivational_fallback': {
                    'triple': ('hunt', 'hunter', 'prey'),
                    'successful_strategy': 'derivational_subject_predicate',
                    'fallback_used': True,
                    'failed_strategies': ['framenet_subject_predicate']
                },
                'gloss_fallback': {
                    'triple': ('animal', 'vertebrate', 'creature'),
                    'successful_strategy': 'gloss_based_subject_predicate',
                    'fallback_used': True,
                    'failed_strategies': ['framenet_subject_predicate', 'derivational_subject_predicate']
                }
            },
            'strategy_failure_scenarios': {
                'all_strategies_fail': {
                    'triple': ('rock', 'emotion', 'abstract'),
                    'successful_strategy': None,
                    'fallback_used': True,
                    'failed_strategies': [
                        'framenet_subject_predicate',
                        'derivational_subject_predicate', 
                        'gloss_based_subject_predicate'
                    ]
                }
            }
        }
    
    @staticmethod
    def get_graph_building_parameters():
        """Get parameters for enhanced graph building."""
        return {
            'basic_params': {
                'beam_width': 3,
                'max_depth': 6,
                'similarity_threshold': 0.5,
                'enable_framenet_edges': True,
                'enable_derivational_edges': True
            },
            'validation_params': {
                'beam_width': 0,   # Invalid
                'max_depth': -1,   # Invalid
                'similarity_threshold': 1.5,  # Invalid
                'enable_framenet_edges': None,  # Invalid
                'enable_derivational_edges': 'invalid'  # Invalid
            },
            'edge_case_params': {
                'beam_width': 1000,  # Very large
                'max_depth': 100,    # Very large
                'similarity_threshold': 0.0,  # Minimum
                'enable_framenet_edges': False,
                'enable_derivational_edges': False
            },
            'integration_params': {
                'beam_width': 5,
                'max_depth': 10,
                'similarity_threshold': 0.7,
                'enable_framenet_edges': True,
                'enable_derivational_edges': True,
                'use_wordnet_expansion': True,
                'confidence_threshold': 0.6
            }
        }
    
    @staticmethod
    def get_expected_graph_structures():
        """Get expected graph structures with FrameNet enhancements."""
        return {
            'basic_structure': {
                'nodes': ['cat.n.01', 'chase.v.01', 'mouse.n.01', 'animal.n.01'],
                'wordnet_edges': [
                    ('cat.n.01', 'animal.n.01', 'hypernym'),
                    ('mouse.n.01', 'animal.n.01', 'hypernym')
                ],
                'framenet_edges': [
                    ('cat.n.01', 'chase.v.01', 'frame_element'),
                    ('mouse.n.01', 'chase.v.01', 'frame_element')
                ],
                'derivational_edges': []
            },
            'enhanced_structure': {
                'nodes': [
                    'hunt.v.01', 'hunter.n.01', 'hunting.n.01',
                    'chase.v.01', 'chaser.n.01', 'prey.n.01'
                ],
                'wordnet_edges': [
                    ('hunt.v.01', 'chase.v.01', 'similar_to')
                ],
                'framenet_edges': [
                    ('hunt.v.01', 'prey.n.01', 'frame_element'),
                    ('chase.v.01', 'prey.n.01', 'frame_element')
                ],
                'derivational_edges': [
                    ('hunt.v.01', 'hunter.n.01', 'agentive'),
                    ('hunt.v.01', 'hunting.n.01', 'gerundive'),
                    ('chase.v.01', 'chaser.n.01', 'agentive')
                ]
            }
        }
    
    @staticmethod
    def get_validation_test_data():
        """Get test data for validation tests."""
        return {
            'valid_framenet_inputs': {
                'min_confidence': 0.5,
                'nlp_model': 'en_core_web_sm',
                'text': 'The cat chases the mouse',
                'frame_elements': ['Agent', 'Patient'],
                'frame_name': 'Cotheme'
            },
            'invalid_framenet_inputs': {
                'min_confidence': -0.1,  # Invalid
                'nlp_model': None,       # Invalid
                'text': '',              # Empty
                'frame_elements': [],    # Empty
                'frame_name': None       # Invalid
            },
            'boundary_framenet_inputs': {
                'min_confidence': 0.0,   # Boundary
                'nlp_model': 'minimal_model',
                'text': 'a',             # Minimal text
                'frame_elements': ['FE'], # Minimal elements
                'frame_name': 'Frame'    # Minimal name
            },
            'valid_decomposer_inputs': {
                'subject': 'cat',
                'predicate': 'chase',
                'object': 'mouse',
                'beam_width': 3,
                'max_depth': 6,
                'enable_framenet': True
            },
            'invalid_decomposer_inputs': {
                'subject': '',           # Empty
                'predicate': None,       # None
                'object': 123,           # Wrong type
                'beam_width': -1,        # Invalid
                'max_depth': 0,          # Invalid
                'enable_framenet': 'yes' # Wrong type
            }
        }
    
    @staticmethod
    def get_edge_case_test_data():
        """Get test data for edge case tests."""
        return {
            'empty_scenarios': {
                'no_frames_found': {
                    'text': 'xyz abc def',  # Nonsense text
                    'expected_frames': [],
                    'expected_elements': []
                },
                'empty_derivational_forms': {
                    'synset': 'be.v.01',
                    'expected_derivatives': [],
                    'expected_connections': []
                },
                'no_synsets_found': {
                    'words': ['nonexistent', 'invalid', 'missing'],
                    'expected_synsets': []
                }
            },
            'malformed_scenarios': {
                'invalid_frame_data': {
                    'name': None,
                    'confidence': 'invalid',
                    'elements': 'not_a_list'
                },
                'invalid_synset_names': [
                    'not.a.synset',
                    'invalid',
                    '',
                    None,
                    123
                ],
                'malformed_graph_data': {
                    'nodes': [None, '', 123, {'invalid': 'node'}],
                    'edges': [(None, 'valid'), ('', None), (123, 'invalid')]
                }
            },
            'performance_scenarios': {
                'large_frame_processing': {
                    'text_length': 10000,
                    'expected_timeout_ms': 30000,
                    'expected_frames': 'many'
                },
                'deep_derivational_analysis': {
                    'analysis_depth': 10,
                    'expected_derivatives': 'many',
                    'memory_limit_mb': 100
                },
                'complex_pathfinding': {
                    'path_length': 20,
                    'search_space_size': 10000,
                    'expected_timeout_ms': 60000
                }
            }
        }
    
    @staticmethod
    def get_integration_test_scenarios():
        """Get comprehensive integration test scenarios."""
        return {
            'realistic_semantic_decomposition': {
                'scientific_research': {
                    'subject': 'scientist',
                    'predicate': 'study',
                    'object': 'organism',
                    'expected_frames': ['Research', 'Scrutiny'],
                    'expected_subject_roles': ['Researcher', 'Cognizer'],
                    'expected_object_roles': ['Topic', 'Ground'],
                    'expected_strategies': ['framenet_srl', 'derivational_morphology'],
                    'expected_path_length': [2, 5],
                    'confidence_threshold': 0.6
                },
                'educational_activity': {
                    'subject': 'teacher',
                    'predicate': 'educate',
                    'object': 'student',
                    'expected_frames': ['Education_teaching', 'Training'],
                    'expected_subject_roles': ['Teacher', 'Agent'],
                    'expected_object_roles': ['Student', 'Skill'],
                    'expected_strategies': ['framenet_srl', 'derivational_morphology'],
                    'expected_path_length': [1, 4],
                    'confidence_threshold': 0.7
                }
            },
            'multi_strategy_scenarios': {
                'framenet_primary': {
                    'triple': ('cat', 'chase', 'mouse'),
                    'primary_strategy': 'framenet_srl',
                    'fallback_strategies': [],
                    'expected_success': True,
                    'expected_confidence': 0.8
                },
                'derivational_fallback': {
                    'triple': ('hunt', 'hunter', 'prey'),
                    'primary_strategy': 'framenet_srl',
                    'fallback_strategies': ['derivational_morphology'],
                    'expected_success': True,
                    'expected_confidence': 0.7
                },
                'gloss_final_fallback': {
                    'triple': ('mammal', 'vertebrate', 'animal'),
                    'primary_strategy': 'framenet_srl',
                    'fallback_strategies': ['derivational_morphology', 'gloss_parsing'],
                    'expected_success': True,
                    'expected_confidence': 0.6
                }
            },
            'performance_integration': {
                'large_scale_processing': {
                    'num_triples': 100,
                    'expected_processing_time_ms': 60000,
                    'memory_limit_mb': 500,
                    'success_rate_threshold': 0.8
                },
                'real_world_text': {
                    'text_samples': [
                        "The researcher analyzes the data to understand patterns.",
                        "Students learn mathematics through practice and instruction.",
                        "The cat silently stalks its prey through the tall grass."
                    ],
                    'expected_frames_per_sample': [3, 4, 2],
                    'processing_timeout_ms': 10000
                }
            }
        }
    
    @staticmethod
    def get_mock_setup_configurations():
        """Get configurations for mock setup in different test scenarios."""
        return {
            'basic_setup': {
                'framenet_srl_mock': 'MockFrameNetSpaCySRL',
                'semantic_decomposer_mock': 'MockSemanticDecomposerForFrameNet',
                'wordnet_mock': 'MockWordNetForFrameNet',
                'nlp_mock': 'MockNLPForFrameNet',
                'additional_components': ['FrameInstance', 'FrameElement']
            },
            'validation_setup': {
                'framenet_srl_mock': 'MockFrameNetSpaCySRLValidation',
                'semantic_decomposer_mock': 'MockSemanticDecomposerForFrameNet',
                'validation_mock': 'MockFrameNetIntegrationValidation',
                'enable_validation': True,
                'strict_validation': True
            },
            'edge_case_setup': {
                'framenet_srl_mock': 'MockFrameNetSpaCySRLEdgeCases',
                'semantic_decomposer_mock': 'MockSemanticDecomposerForFrameNet',
                'edge_case_mock': 'MockFrameNetIntegrationEdgeCases',
                'simulate_failures': True,
                'enable_timeouts': True
            },
            'integration_setup': {
                'framenet_srl_mock': 'MockFrameNetSpaCySRLIntegration',
                'semantic_decomposer_mock': 'MockSemanticDecomposerEnhanced',
                'integration_mock': 'MockFrameNetIntegrationIntegration',
                'use_realistic_behavior': True,
                'enable_all_strategies': True
            }
        }
    
    @staticmethod
    def get_expected_test_outcomes():
        """Get expected outcomes for different test scenarios."""
        return {
            'successful_framenet_processing': {
                'frames_extracted': True,
                'frame_elements_found': True,
                'semantic_roles_identified': True,
                'confidence_above_threshold': True,
                'processing_time_acceptable': True
            },
            'successful_pathfinding': {
                'subject_predicate_path_found': True,
                'predicate_object_path_found': True,
                'path_length_reasonable': True,
                'strategies_applied': ['framenet_srl'],
                'fallback_not_needed': True
            },
            'fallback_strategy_success': {
                'primary_strategy_failed': True,
                'fallback_strategy_succeeded': True,
                'strategies_applied': ['framenet_srl', 'derivational_morphology'],
                'final_result_found': True
            },
            'complete_failure': {
                'all_strategies_failed': True,
                'no_paths_found': True,
                'expected_exceptions': ['ValueError', 'TimeoutError'],
                'graceful_degradation': True
            },
            'validation_outcomes': {
                'invalid_inputs_rejected': True,
                'boundary_inputs_handled': True,
                'validation_errors_reported': True,
                'system_remains_stable': True
            },
            'edge_case_outcomes': {
                'empty_inputs_handled': True,
                'malformed_data_handled': True,
                'resource_limits_respected': True,
                'error_recovery_functional': True
            },
            'integration_outcomes': {
                'components_interact_correctly': True,
                'data_flow_maintained': True,
                'performance_within_limits': True,
                'end_to_end_functionality': True
            }
        }
    
    @staticmethod
    def get_test_text_samples():
        """Get text samples for processing tests."""
        return {
            'simple_texts': [
                'The cat chases the mouse.',
                'Dogs run in the park.',
                'Birds fly south in winter.'
            ],
            'complex_texts': [
                'The scientist carefully studies the behavior of various organisms in their natural habitat.',
                'Students learn advanced mathematics through rigorous practice and expert instruction.',
                'The experienced hunter silently tracks the elusive prey through the dense forest.'
            ],
            'edge_case_texts': [
                '',  # Empty
                'a',  # Single character
                'The the the the the.',  # Repetitive
                'Xyz abc def ghi jkl.',  # Nonsense words
                'A' * 1000  # Very long single word
            ],
            'multilingual_texts': [
                'El gato persigue al ratón.',  # Spanish
                'Le chat poursuit la souris.',  # French
                'Die Katze jagt die Maus.'     # German
            ]
        }
    
    @staticmethod
    def get_performance_benchmarks():
        """Get performance benchmarks for testing."""
        return {
            'framenet_processing': {
                'small_text': {
                    'max_chars': 100,
                    'expected_time_ms': 500,
                    'memory_limit_mb': 10
                },
                'medium_text': {
                    'max_chars': 1000,
                    'expected_time_ms': 2000,
                    'memory_limit_mb': 50
                },
                'large_text': {
                    'max_chars': 10000,
                    'expected_time_ms': 10000,
                    'memory_limit_mb': 200
                }
            },
            'pathfinding_performance': {
                'simple_paths': {
                    'max_path_length': 5,
                    'expected_time_ms': 100,
                    'memory_limit_mb': 20
                },
                'complex_paths': {
                    'max_path_length': 15,
                    'expected_time_ms': 1000,
                    'memory_limit_mb': 100
                },
                'exhaustive_search': {
                    'max_path_length': 50,
                    'expected_time_ms': 30000,
                    'memory_limit_mb': 500
                }
            }
        }