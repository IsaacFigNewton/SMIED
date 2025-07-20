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
from .SemanticHypergraph import SemanticHypergraph
from .PatternLoader import PatternLoader
from .PatternMatcher import PatternMatcher

from .hypergraphx.Object import Object
from .hypergraphx.visualizations import draw

# Define what gets imported with "from package import *"
__all__ = [
    # Main classes
    "SemanticHypergraph",
    "PatternLoader",
    "PatternMatcher",

    # Modified from old hypergraphx package file
    "Object",
    "draw",
]