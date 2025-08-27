#!/usr/bin/env python3
"""
Example demonstrating the usage of the SMIED class interface.
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import spacy
from smied.SMIED import SMIED
import gensim.downloader as api

def main():
    """Demonstrate SMIED class usage."""
    # nlp = spacy.load("en_core_web_sm")

    # Load Word2Vec model (optional)
    print("Loading Word2Vec model...")
    w2v_model = None
    try:
        w2v_model = api.load("word2vec-google-news-300")
        print("Word2Vec model loaded successfully.")
    except Exception as e:
        print(f"Failed to load Word2Vec model: {e}")

    # Create SMIED instance
    print("Initializing SMIED pipeline...")
    pipeline = SMIED(
        nlp_model="en_core_web_sm",  # Set to "en_core_web_sm" if spaCy is installed
        embedding_model=w2v_model,  # Can add embedding model for better similarity
        auto_download=True,  # Automatically download required NLTK data,
    )
    print("SMIED pipeline initialized.")

    triples = [
        ("cat", "chase", "mouse"),
        ("dog", "bark", "stranger"),
        ("bird", "fly", "sky"),
    ]
    for subj, pred, obj in triples:
        print(f"\nAnalyzing triple: {subj, pred, obj}")
        result = pipeline.analyze_triple(subj, pred, obj)
        print(f"Result: {result}\n")
    
    

if __name__ == "__main__":
    main()