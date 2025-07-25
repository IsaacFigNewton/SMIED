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

# Import stuff from the package
from noske.SemanticMetagraph import SemanticMetagraph
from noske.PatternLoader import PatternLoader
from noske.PatternMatcher import PatternMatcher
from noske.utils import to_nx

from noske.hypergraphx.Object import Object
from noske.hypergraphx.visualizations import draw

# Define what gets imported with "from package import *"
__all__ = [
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