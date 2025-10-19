"""
Interfaces for the SMIED (Semantic Meaning in Embedding Dimensions) package.

This module contains abstract base classes defining the contracts for all
major components in the semantic decomposition system.
"""
from .IDirectedMetagraph import IDirectedMetagraph
from .IPatternLoader import IPatternLoader
from .IPatternMatcher import IPatternMatcher
from .ISemanticMetagraph import ISemanticMetagraph

__all__ = [
    'IDirectedMetagraph',
    'IPatternLoader',
    'IPatternMatcher',
    'ISemanticMetagraph'
]