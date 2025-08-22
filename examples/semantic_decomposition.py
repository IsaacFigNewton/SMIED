#!/usr/bin/env python3
"""
Example usage of SemanticDecomposer to find semantic paths through WordNet.

This example demonstrates how to use the SemanticDecomposer class to find 
semantic paths that connect subject-predicate-object triples like "fox", "jump", "dog".
"""

import sys
import os

# Add the src directory to path so we can import smied
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import nltk
import spacy
from nltk.corpus import wordnet as wn
from smied.SMIED import SMIED

if __name__ == "__main__":
    # Run the main example
    pipeline = SMIED()
    
    # Show alternative approaches
    pipeline.demonstrate_alternative_approaches("fox", "jump", "dog")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nTo experiment with different word combinations, modify the")
    print("subject_word, predicate_word, and object_word variables in main().")
    print("\nFor better results, consider:")
    print("- Installing spaCy: pip install spacy")
    print("- Installing word embeddings (Word2Vec, GloVe, etc.)")
    print("- Adjusting search parameters (max_depth, beam_width, etc.)")