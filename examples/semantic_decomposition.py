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
from smied.SemanticDecomposer import SemanticDecomposer

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def setup_nlp():
    """Set up spaCy NLP pipeline. Returns None if not available."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Warning: spaCy 'en_core_web_sm' model not found.")
        print("Install with: python -m spacy download en_core_web_sm")
        return None


def main():
    """
    Demonstrate semantic decomposition for the triple: fox, jump, dog.
    
    This example shows how to:
    1. Initialize the SemanticDecomposer
    2. Find semantic paths connecting the three words
    3. Display the results
    """
    
    print("=" * 80)
    print("SEMANTIC DECOMPOSITION EXAMPLE")
    print("Finding semantic paths linking: fox -> jump -> dog")
    print("=" * 80)
    print()
    
    # Set up spaCy NLP function
    nlp = setup_nlp()
    if nlp is None:
        print("Using basic fallback NLP processing...")
        nlp = lambda text: None  # Fallback that will limit functionality
    
    # Initialize the SemanticDecomposer
    print("Initializing SemanticDecomposer...")
    decomposer = SemanticDecomposer(
        wn_module=wn,
        nlp_func=nlp,
        embedding_model=None  # No embedding model for this basic example
    )
    
    # Words to connect
    subject_word = "fox"
    predicate_word = "jump"
    object_word = "dog"
    
    print(f"Subject: {subject_word}")
    print(f"Predicate: {predicate_word}")
    print(f"Object: {object_word}")
    print()
    
    # Show available synsets for each word
    print("Available synsets:")
    print("-" * 40)
    
    fox_synsets = wn.synsets(subject_word, pos=wn.NOUN)
    jump_synsets = wn.synsets(predicate_word, pos=wn.VERB)
    dog_synsets = wn.synsets(object_word, pos=wn.NOUN)
    
    print(f"Fox synsets ({len(fox_synsets)}):")
    for i, synset in enumerate(fox_synsets[:3]):  # Show first 3
        print(f"  {i+1}. {synset.name()}: {synset.definition()}")
    print()
    
    print(f"Jump synsets ({len(jump_synsets)}):")
    for i, synset in enumerate(jump_synsets[:3]):  # Show first 3
        print(f"  {i+1}. {synset.name()}: {synset.definition()}")
    print()
    
    print(f"Dog synsets ({len(dog_synsets)}):")
    for i, synset in enumerate(dog_synsets[:3]):  # Show first 3
        print(f"  {i+1}. {synset.name()}: {synset.definition()}")
    print()
    
    # Build the synset graph (this may take a moment)
    print("Building WordNet synset graph... (this may take a moment)")
    synset_graph = decomposer.build_synset_graph()
    print(f"Graph built with {synset_graph.number_of_nodes()} nodes and {synset_graph.number_of_edges()} edges")
    print()
    
    # Find connected shortest paths
    print("Searching for semantic paths...")
    print("-" * 40)
    
    try:
        subject_path, object_path, connecting_predicate = decomposer.find_connected_shortest_paths(
            subject_word=subject_word,
            predicate_word=predicate_word,
            object_word=object_word,
            g=synset_graph,
            max_depth=6,          # Limit search depth for performance
            beam_width=3,         # Limit beam width
            max_results_per_pair=2,  # Limit results per pair
            len_tolerance=2       # Allow some length variation
        )
        
        # Display results
        if subject_path and object_path and connecting_predicate:
            print("SUCCESS: Found connected semantic paths!")
            print()
            
            # Display the paths
            SemanticDecomposer.show_connected_paths(
                subject_path, object_path, connecting_predicate
            )
            
            # Additional analysis
            print("ANALYSIS:")
            print("-" * 20)
            print(f"Connecting predicate: {connecting_predicate.name()}")
            print(f"Predicate definition: {connecting_predicate.definition()}")
            print(f"Subject path length: {len(subject_path)}")
            print(f"Object path length: {len(object_path)}")
            print(f"Total semantic distance: {len(subject_path) + len(object_path) - 1}")
            print()
            
            # Show word sense disambiguation
            print("WORD SENSE DISAMBIGUATION:")
            print("-" * 30)
            if subject_path:
                print(f"Selected '{subject_word}' sense: {subject_path[0].name()} - {subject_path[0].definition()}")
            if connecting_predicate:
                print(f"Selected '{predicate_word}' sense: {connecting_predicate.name()} - {connecting_predicate.definition()}")
            if object_path:
                print(f"Selected '{object_word}' sense: {object_path[-1].name()} - {object_path[-1].definition()}")
            
        else:
            print("No connected semantic path found.")
            print("This could mean:")
            print("- The words are semantically too distant")
            print("- The search parameters are too restrictive")
            print("- The gloss parsing couldn't find suitable connections")
            print()
            
            # Show some individual synset relationships as fallback
            print("Individual synset relationships:")
            print("-" * 35)
            
            if fox_synsets and dog_synsets:
                print("Fox-Dog relationships:")
                for fox_syn in fox_synsets[:2]:
                    for dog_syn in dog_synsets[:2]:
                        try:
                            similarity = fox_syn.path_similarity(dog_syn)
                            if similarity:
                                print(f"  {fox_syn.name()} <-> {dog_syn.name()}: {similarity:.3f}")
                        except:
                            pass
            
            if fox_synsets and jump_synsets:
                print("Fox-Jump relationships:")
                for fox_syn in fox_synsets[:2]:
                    for jump_syn in jump_synsets[:2]:
                        try:
                            # Check if there are any shared hypernyms or relations
                            fox_hypernyms = set(fox_syn.hypernyms())
                            jump_related = set(jump_syn.entailments() + jump_syn.causes())
                            if fox_hypernyms.intersection(jump_related):
                                print(f"  {fox_syn.name()} ~ {jump_syn.name()}: related")
                        except:
                            pass
    
    except Exception as e:
        print(f"Error during semantic decomposition: {e}")
        print("This might be due to missing dependencies or WordNet issues.")


def demonstrate_alternative_approaches():
    """Show alternative approaches when direct semantic paths aren't found."""
    
    print("\n" + "=" * 80)
    print("ALTERNATIVE SEMANTIC ANALYSIS")
    print("=" * 80)
    
    # Show hypernym paths
    print("Exploring hypernym hierarchies:")
    print("-" * 35)
    
    fox_synset = wn.synsets("fox", pos=wn.NOUN)[0] if wn.synsets("fox", pos=wn.NOUN) else None
    dog_synset = wn.synsets("dog", pos=wn.NOUN)[0] if wn.synsets("dog", pos=wn.NOUN) else None
    
    if fox_synset and dog_synset:
        print(f"\nFox ({fox_synset.name()}) hypernym path:")
        current = fox_synset
        depth = 0
        while current and depth < 5:
            print(f"  {'  ' * depth}{current.name()}: {current.definition()}")
            hypernyms = current.hypernyms()
            current = hypernyms[0] if hypernyms else None
            depth += 1
        
        print(f"\nDog ({dog_synset.name()}) hypernym path:")
        current = dog_synset
        depth = 0
        while current and depth < 5:
            print(f"  {'  ' * depth}{current.name()}: {current.definition()}")
            hypernyms = current.hypernyms()
            current = hypernyms[0] if hypernyms else None
            depth += 1
    
    # Show verb entailments and causes for 'jump'
    jump_synset = wn.synsets("jump", pos=wn.VERB)[0] if wn.synsets("jump", pos=wn.VERB) else None
    if jump_synset:
        print(f"\nJump ({jump_synset.name()}) verb relations:")
        
        entailments = jump_synset.entailments()
        if entailments:
            print("  Entailments (what jumping necessarily involves):")
            for ent in entailments[:3]:
                print(f"    {ent.name()}: {ent.definition()}")
        
        causes = jump_synset.causes()
        if causes:
            print("  Causes (what jumping can cause):")
            for cause in causes[:3]:
                print(f"    {cause.name()}: {cause.definition()}")
        
        verb_groups = jump_synset.verb_groups()
        if verb_groups:
            print("  Verb groups (related verbs):")
            for vg in verb_groups[:3]:
                print(f"    {vg.name()}: {vg.definition()}")


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Show alternative approaches
    demonstrate_alternative_approaches()
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nTo experiment with different word combinations, modify the")
    print("subject_word, predicate_word, and object_word variables in main().")
    print("\nFor better results, consider:")
    print("- Installing spaCy: pip install spacy")
    print("- Installing word embeddings (Word2Vec, GloVe, etc.)")
    print("- Adjusting search parameters (max_depth, beam_width, etc.)")