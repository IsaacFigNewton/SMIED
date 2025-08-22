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
from .DirectedMetagraph import DirectedMetagraph
from .SemanticMetagraph import SemanticMetagraph
from .PatternLoader import PatternLoader
from .PatternMatcher import PatternMatcher

# Define what gets imported with "from package import *"
__all__ = [
    # Main classes
    "DirectedMetagraph",
    "SemanticMetagraph",
    "PatternLoader",
    "PatternMatcher",
]