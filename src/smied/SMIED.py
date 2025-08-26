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
        self.synset_graph = None
        if build_graph_on_init:
            self.synset_graph = self.build_synset_graph()
    
    def _setup_nlp(self) -> Optional[Any]:
        """Set up spaCy NLP pipeline."""
        if not self.nlp_model_name:
            return None
            
        try:
            import spacy
            nlp = spacy.load(self.nlp_model_name)
            return nlp
        except OSError:
            print(f"[WARNING] spaCy '{self.nlp_model_name}' model not found.")
            print(f"[WARNING] Install with: python -m spacy download {self.nlp_model_name}")
            return None
    
    def build_synset_graph(self) -> Any:
        """
        Build or retrieve the synset graph.
        
        Returns:
            NetworkX graph of WordNet synsets
        """
        if self.synset_graph is None:
            self.synset_graph = self.decomposer.build_synset_graph()
        
        return self.synset_graph
    
    def analyze_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        max_depth: int = 10,
        beam_width: int = 3,
        max_results_per_pair: int = 4,
        len_tolerance: int = 5
    ) -> List[str]:
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
            
        Returns:
            Tuple of (subject_path, object_path, connecting_predicate)
        """
        # Build graph if needed
        graph = self.build_synset_graph()
        
        # Find connected paths
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
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Error during semantic decomposition: {e}")
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
    
    def reinitialize(self, **kwargs) -> None:
        """
        Reinitialize the pipeline components with new settings.
        
        Args:
            **kwargs: Settings to update (nlp_model, embedding_model, etc.)
        """
        # Update settings if provided
        if 'nlp_model' in kwargs:
            self.nlp_model_name = kwargs['nlp_model']
            self.nlp = self._setup_nlp()
        
        if 'embedding_model' in kwargs:
            self.embedding_model = kwargs['embedding_model']
        
        # Reinitialize decomposer with new settings
        self.decomposer = SemanticDecomposer(
            wn_module=wn,
            nlp_func=self.nlp,
            embedding_model=self.embedding_model
        )
        
        # Clear cached graph to force rebuild with new settings
        self.synset_graph = None
    
    def display_results(
        self,
        subject_path: Optional[List],
        object_path: Optional[List],
        connecting_predicate: Optional[Any],
        subject_word: str,
        predicate_word: str,
        object_word: str
    ) -> None:
        """
        Display analysis results (minimal implementation).
        
        Args:
            subject_path: Path from subject to predicate
            object_path: Path from predicate to object
            connecting_predicate: The connecting predicate synset
            subject_word: Original subject word
            predicate_word: Original predicate word
            object_word: Original object word
        """
        if subject_path and object_path and connecting_predicate:
            print(f"Found semantic connection: {subject_word} -> {predicate_word} -> {object_word}")
            print(f"Subject path length: {len(subject_path)}")
            print(f"Object path length: {len(object_path)}")
            print(f"Connecting predicate: {connecting_predicate.name()}")
        else:
            print(f"No semantic connection found for: {subject_word} -> {predicate_word} -> {object_word}")
    
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