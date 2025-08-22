"""
SMIED: Semantic Metagraph-based Information Extraction and Decomposition
Main class for semantic decomposition and path finding.
"""

import sys
import os
from typing import Optional, Tuple, List, Any, Dict
from smied.interfaces.ISMIEDPipeline import ISMIEDPipeline

import nltk
from nltk.corpus import wordnet as wn
from smied.SemanticDecomposer import SemanticDecomposer


class SMIED(ISMIEDPipeline):
    """
    Main SMIED class for semantic decomposition and analysis.
    
    This class provides a unified interface for analyzing semantic relationships
    between words using WordNet and optional NLP models.
    """
    
    def __init__(
        self,
        nlp_model: Optional[str] = "en_core_web_sm",
        embedding_model: Optional[Any] = None,
        auto_download: bool = True,
        build_graph_on_init: bool = False
    ):
        """
        Initialize the SMIED pipeline and all its components.
        
        Args:
            nlp_model: Name of spaCy model to load
            embedding_model: Optional embedding model for similarity
            auto_download: Whether to automatically download required data
            build_graph_on_init: Whether to build the synset graph during initialization
        """
        self.nlp_model_name = nlp_model
        self.embedding_model = embedding_model
        self.auto_download = auto_download
        
        # Download required NLTK data if needed
        if self.auto_download:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        
        # Set up NLP pipeline
        self.nlp = self._setup_nlp()
        
        # Initialize semantic decomposer
        self.decomposer = SemanticDecomposer(
            wn_module=wn,
            nlp_func=self.nlp,
            embedding_model=self.embedding_model
        )
        
        # Build graph if requested
        if build_graph_on_init:
            self.synset_graph = self.build_synset_graph()
        else:
            self.synset_graph = None
        
    def reinitialize(
        self,
        nlp_model: Optional[str] = None,
        embedding_model: Optional[Any] = None
    ) -> None:
        """
        Reinitialize the pipeline with new settings.
        
        Args:
            nlp_model: New spaCy model name (optional)
            embedding_model: New embedding model (optional)
        """
        # Update settings if provided
        if nlp_model is not None:
            self.nlp_model_name = nlp_model
            self.nlp = self._setup_nlp()
        
        if embedding_model is not None:
            self.embedding_model = embedding_model
        
        # Reinitialize decomposer with new settings
        self.decomposer = SemanticDecomposer(
            wn_module=wn,
            nlp_func=self.nlp,
            embedding_model=self.embedding_model
        )
        
        # Clear cached graph to force rebuild with new settings
        self.synset_graph = None
    
    def _setup_nlp(self) -> Optional[Any]:
        """Set up spaCy NLP pipeline."""
        if not self.nlp_model_name:
            return None
            
        try:
            import spacy
            nlp = spacy.load(self.nlp_model_name)
            return nlp
        except OSError:
            print(f"Warning: spaCy '{self.nlp_model_name}' model not found.")
            print(f"Install with: python -m spacy download {self.nlp_model_name}")
            return None
    
    def build_synset_graph(self, verbose: bool = True) -> Any:
        """
        Build or retrieve the synset graph.
        
        Args:
            verbose: Whether to print progress messages
            
        Returns:
            NetworkX graph of WordNet synsets
        """
        if self.synset_graph is None:
            if verbose:
                print("Building WordNet synset graph... (this may take a moment)")
            self.synset_graph = self.decomposer.build_synset_graph()
            if verbose:
                print(f"Graph built with {self.synset_graph.number_of_nodes()} nodes "
                      f"and {self.synset_graph.number_of_edges()} edges")
        
        return self.synset_graph
    
    def analyze_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        max_depth: int = 6,
        beam_width: int = 3,
        max_results_per_pair: int = 2,
        len_tolerance: int = 2,
        verbose: bool = True
    ) -> Tuple[Optional[List], Optional[List], Optional[Any]]:
        """
        Analyze a subject-predicate-object triple.
        
        Args:
            subject: Subject word
            predicate: Predicate word
            object: Object word
            max_depth: Maximum search depth
            beam_width: Beam width for search
            max_results_per_pair: Maximum results per word pair
            len_tolerance: Path length tolerance
            verbose: Whether to print progress
            
        Returns:
            Tuple of (subject_path, object_path, connecting_predicate)
        """
        if verbose:
            print("=" * 80)
            print("SEMANTIC DECOMPOSITION ANALYSIS")
            print(f"Finding semantic paths linking: {subject} -> {predicate} -> {object}")
            print("=" * 80)
            print()
        
        # Build graph if needed
        graph = self.build_synset_graph(verbose=verbose)
        
        # Show available synsets if verbose
        if verbose:
            self._display_synsets(subject, predicate, object)
        
        # Find connected paths
        if verbose:
            print("Searching for semantic paths...")
            print("-" * 40)
        
        try:
            result = self.decomposer.find_connected_shortest_paths(
                subject_word=subject,
                predicate_word=predicate,
                object_word=object,
                g=graph,
                max_depth=max_depth,
                beam_width=beam_width,
                max_results_per_pair=max_results_per_pair,
                len_tolerance=len_tolerance
            )
            
            subject_path, object_path, connecting_predicate = result
            
            if verbose:
                self.display_results(
                    subject_path, object_path, connecting_predicate,
                    subject, predicate, object
                )
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"Error during semantic decomposition: {e}")
            raise
    
    def get_synsets(self, word: str, pos: Optional[str] = None) -> List:
        """
        Get WordNet synsets for a word.
        
        Args:
            word: Word to look up
            pos: Part of speech (optional)
            
        Returns:
            List of synsets
        """
        if pos:
            return wn.synsets(word, pos=pos)
        return wn.synsets(word)
    
    def _display_synsets(self, subject: str, predicate: str, object: str) -> None:
        """Display available synsets for the triple."""
        print("Available synsets:")
        print("-" * 40)
        
        # Get synsets
        subject_synsets = self.get_synsets(subject, pos=wn.NOUN)
        predicate_synsets = self.get_synsets(predicate, pos=wn.VERB)
        object_synsets = self.get_synsets(object, pos=wn.NOUN)
        
        # Display subject synsets
        print(f"{subject.capitalize()} synsets ({len(subject_synsets)}):")
        for i, synset in enumerate(subject_synsets[:3]):
            print(f"  {i+1}. {synset.name()}: {synset.definition()}")
        print()
        
        # Display predicate synsets
        print(f"{predicate.capitalize()} synsets ({len(predicate_synsets)}):")
        for i, synset in enumerate(predicate_synsets[:3]):
            print(f"  {i+1}. {synset.name()}: {synset.definition()}")
        print()
        
        # Display object synsets
        print(f"{object.capitalize()} synsets ({len(object_synsets)}):")
        for i, synset in enumerate(object_synsets[:3]):
            print(f"  {i+1}. {synset.name()}: {synset.definition()}")
        print()
    
    def display_results(
        self,
        subject_path: Optional[List],
        object_path: Optional[List],
        connecting_predicate: Optional[Any],
        subject_word: str,
        predicate_word: str,
        object_word: str
    ) -> None:
        """Display analysis results."""
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
                print(f"Selected '{subject_word}' sense: {subject_path[0].name()} - "
                      f"{subject_path[0].definition()}")
            if connecting_predicate:
                print(f"Selected '{predicate_word}' sense: {connecting_predicate.name()} - "
                      f"{connecting_predicate.definition()}")
            if object_path:
                print(f"Selected '{object_word}' sense: {object_path[-1].name()} - "
                      f"{object_path[-1].definition()}")
        else:
            print("No connected semantic path found.")
            print("This could mean:")
            print("- The words are semantically too distant")
            print("- The search parameters are too restrictive")
            print("- The gloss parsing couldn't find suitable connections")
            print()
            
            # Show individual relationships
            self._show_fallback_relationships(subject_word, predicate_word, object_word)
    
    def _show_fallback_relationships(
        self,
        subject: str,
        predicate: str,
        object: str
    ) -> None:
        """Show fallback relationships when no path is found."""
        print("Individual synset relationships:")
        print("-" * 35)
        
        subject_synsets = self.get_synsets(subject, pos=wn.NOUN)
        predicate_synsets = self.get_synsets(predicate, pos=wn.VERB)
        object_synsets = self.get_synsets(object, pos=wn.NOUN)
        
        # Subject-Object relationships
        if subject_synsets and object_synsets:
            print(f"{subject.capitalize()}-{object.capitalize()} relationships:")
            for subj_syn in subject_synsets[:2]:
                for obj_syn in object_synsets[:2]:
                    try:
                        similarity = subj_syn.path_similarity(obj_syn)
                        if similarity:
                            print(f"  {subj_syn.name()} <-> {obj_syn.name()}: {similarity:.3f}")
                    except:
                        pass
        
        # Subject-Predicate relationships
        if subject_synsets and predicate_synsets:
            print(f"{subject.capitalize()}-{predicate.capitalize()} relationships:")
            for subj_syn in subject_synsets[:2]:
                for pred_syn in predicate_synsets[:2]:
                    try:
                        subj_hypernyms = set(subj_syn.hypernyms())
                        pred_related = set(pred_syn.entailments() + pred_syn.causes())
                        if subj_hypernyms.intersection(pred_related):
                            print(f"  {subj_syn.name()} ~ {pred_syn.name()}: related")
                    except:
                        pass
    
    def demonstrate_alternative_approaches(
        self,
        subject: str = "fox",
        predicate: str = "jump",
        object: str = "dog"
    ) -> None:
        """
        Show alternative semantic analysis approaches.
        
        Args:
            subject: Subject word
            predicate: Predicate word
            object: Object word
        """
        print("\n" + "=" * 80)
        print("ALTERNATIVE SEMANTIC ANALYSIS")
        print("=" * 80)
        
        # Show hypernym paths
        print("Exploring hypernym hierarchies:")
        print("-" * 35)
        
        subject_synsets = self.get_synsets(subject, pos=wn.NOUN)
        object_synsets = self.get_synsets(object, pos=wn.NOUN)
        
        if subject_synsets:
            subject_synset = subject_synsets[0]
            print(f"\n{subject.capitalize()} ({subject_synset.name()}) hypernym path:")
            self._show_hypernym_path(subject_synset, max_depth=5)
        
        if object_synsets:
            object_synset = object_synsets[0]
            print(f"\n{object.capitalize()} ({object_synset.name()}) hypernym path:")
            self._show_hypernym_path(object_synset, max_depth=5)
        
        # Show verb relations
        predicate_synsets = self.get_synsets(predicate, pos=wn.VERB)
        if predicate_synsets:
            predicate_synset = predicate_synsets[0]
            self._show_verb_relations(predicate_synset, predicate)
    
    def _show_hypernym_path(self, synset: Any, max_depth: int = 5) -> None:
        """Display hypernym path for a synset."""
        current = synset
        depth = 0
        while current and depth < max_depth:
            print(f"  {'  ' * depth}{current.name()}: {current.definition()}")
            hypernyms = current.hypernyms()
            current = hypernyms[0] if hypernyms else None
            depth += 1
    
    def _show_verb_relations(self, synset: Any, verb: str) -> None:
        """Display verb relations for a synset."""
        print(f"\n{verb.capitalize()} ({synset.name()}) verb relations:")
        
        entailments = synset.entailments()
        if entailments:
            print("  Entailments (what it necessarily involves):")
            for ent in entailments[:3]:
                print(f"    {ent.name()}: {ent.definition()}")
        
        causes = synset.causes()
        if causes:
            print("  Causes (what it can cause):")
            for cause in causes[:3]:
                print(f"    {cause.name()}: {cause.definition()}")
        
        verb_groups = synset.verb_groups()
        if verb_groups:
            print("  Verb groups (related verbs):")
            for vg in verb_groups[:3]:
                print(f"    {vg.name()}: {vg.definition()}")
    
    def calculate_similarity(
        self,
        word1: str,
        word2: str,
        method: str = "path"
    ) -> Optional[float]:
        """
        Calculate similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            method: Similarity method ('path', 'wu_palmer', 'lch')
            
        Returns:
            Similarity score or None if not computable
        """
        synsets1 = self.get_synsets(word1)
        synsets2 = self.get_synsets(word2)
        
        if not synsets1 or not synsets2:
            return None
        
        # Get first synset for each word
        syn1 = synsets1[0]
        syn2 = synsets2[0]
        
        try:
            if method == "path":
                return syn1.path_similarity(syn2)
            elif method == "wu_palmer":
                return syn1.wup_similarity(syn2)
            elif method == "lch":
                return syn1.lch_similarity(syn2)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
        except:
            return None
    
    def get_word_info(self, word: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a word.
        
        Args:
            word: Word to analyze
            
        Returns:
            Dictionary with word information
        """
        info = {
            "word": word,
            "synsets": [],
            "total_senses": 0
        }
        
        synsets = self.get_synsets(word)
        info["total_senses"] = len(synsets)
        
        for synset in synsets:
            synset_info = {
                "name": synset.name(),
                "definition": synset.definition(),
                "pos": synset.pos(),
                "examples": synset.examples(),
                "lemmas": [lemma.name() for lemma in synset.lemmas()],
                "hypernyms": [h.name() for h in synset.hypernyms()],
                "hyponyms": [h.name() for h in synset.hyponyms()]
            }
            info["synsets"].append(synset_info)
        
        return info