__version__ = '0.0.1'
__authors__ = 'Isaac Rudnick'

"""
A Python package for semantic knowledge extraction and pattern matching.
This package provides tools for working with semantic knowledge graphs and (soon) hypergraphs.
Tools include:
- (soon) semantic hypergraph representation
- pattern loading
- pattern matching
- visualization
- SpaCy integration
- and more to come!
"""

# Import base class stuff
from noske.Metagraph import Metagraph
from noske.MetagraphUtils import (
    _is_edge,
    _flatten_edge,
    _get_required_node_fields,
    _get_required_edge_fields,
)

# Import other classes and stuff
from noske.SemanticMetagraph import SemanticMetagraph
from noske.PatternLoader import PatternLoader
from noske.PatternMatcher import PatternMatcher
from noske.utils import to_nx

# Import modified Hypergraphx classes, functions
from noske.hypergraphx.Object import Object
from noske.hypergraphx.visualizations import draw

# Define what gets imported with "from package import *"
__all__ = [
    # Metagraph class and helpers
    'Metagraph',
    '_is_edge',
    '_flatten_edge',
    '_get_required_node_fields',
    '_get_required_edge_fields'

    # Main classes
    "SemanticMetagraph",
    "PatternLoader",
    "PatternMatcher",

    # Utility functions
    "to_nx",

    # Modified from old hypergraphx package file
    "Object",
    "draw",
]