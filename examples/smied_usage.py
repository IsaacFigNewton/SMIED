#!/usr/bin/env python3
"""
Example demonstrating the usage of the SMIED class interface.
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.SMIED import SMIED


def main():
    """Demonstrate SMIED class usage."""
    
    # Create SMIED instance
    print("Initializing SMIED pipeline...")
    pipeline = SMIED(
        nlp_model=None,  # Set to "en_core_web_sm" if spaCy is installed
        embedding_model=None,  # Can add embedding model for better similarity
        auto_download=True  # Automatically download required NLTK data
    )
    
    # Example 1: Analyze a triple
    print("\n" + "=" * 60)
    print("Example 1: Analyzing semantic triple")
    print("=" * 60)
    
    result = pipeline.analyze_triple(
        subject="cat",
        predicate="chase",
        object="mouse",
        verbose=True
    )
    
    # Example 2: Calculate word similarity
    print("\n" + "=" * 60)
    print("Example 2: Word similarity calculation")
    print("=" * 60)
    
    words_pairs = [
        ("cat", "dog"),
        ("cat", "car"),
        ("run", "walk"),
        ("happy", "sad")
    ]
    
    for word1, word2 in words_pairs:
        similarity = pipeline.calculate_similarity(word1, word2, method="path")
        if similarity:
            print(f"{word1} <-> {word2}: {similarity:.3f}")
        else:
            print(f"{word1} <-> {word2}: No similarity found")
    
    # Example 3: Get detailed word information
    print("\n" + "=" * 60)
    print("Example 3: Word information")
    print("=" * 60)
    
    word_info = pipeline.get_word_info("dog")
    print(f"Word: {word_info['word']}")
    print(f"Total senses: {word_info['total_senses']}")
    
    for i, synset in enumerate(word_info['synsets'][:3], 1):
        print(f"\n{i}. {synset['name']}")
        print(f"   Definition: {synset['definition']}")
        print(f"   POS: {synset['pos']}")
        if synset['examples']:
            print(f"   Example: {synset['examples'][0]}")
    
    # Example 4: Non-verbose analysis
    print("\n" + "=" * 60)
    print("Example 4: Silent analysis (non-verbose)")
    print("=" * 60)
    
    result = pipeline.analyze_triple("bird", "fly", "sky", verbose=False)
    subject_path, object_path, predicate = result
    
    if subject_path and object_path:
        print("[SUCCESS] Found semantic path connecting 'bird' -> 'fly' -> 'sky'")
    else:
        print("[NO PATH] No direct semantic path found for 'bird' -> 'fly' -> 'sky'")
    
    # Example 5: Alternative approaches
    print("\n" + "=" * 60)
    print("Example 5: Alternative semantic analysis")
    print("=" * 60)
    
    pipeline.demonstrate_alternative_approaches("eagle", "soar", "mountain")


if __name__ == "__main__":
    main()