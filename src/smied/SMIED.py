"""
SMIED: Semantic Metagraph-based Information Extraction and Decomposition
Main class for semantic decomposition and path finding.
"""

import sys
import os
from typing import Optional, Tuple, List, Any, Dict
import spacy
from spacy.tokens import Token, Doc
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet as wn

from smied.interfaces.ISMIEDPipeline import ISMIEDPipeline
from smied.SemanticDecomposer import SemanticDecomposer


class SMIED(ISMIEDPipeline):
    """
    Main SMIED class for semantic decomposition and analysis.
    
    This class provides a unified interface for analyzing semantic relationships
    between words using WordNet and optional NLP models.
    """
    
    def __init__(
        self,
        nlp_model: str = "en_core_web_sm",
        embedding_model: Optional[Any] = None,
        auto_download: bool = True,
    ):
        """
        Initialize the SMIED pipeline and all its components.
        
        Args:
            nlp_model: Name of spaCy model to load
            embedding_model: Optional embedding model for similarity
            auto_download: Whether to automatically download required data
            build_graph_on_init: Whether to build the synset graph during initialization
        """
        self.embedding_model = embedding_model
        self.auto_download = auto_download
        
        # Download required NLTK data if needed
        if self.auto_download:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        
        # Set up NLP pipeline
        try:
            self.nlp = spacy.load(nlp_model)
        except:
            print(f"WARNING: INVALID SPACY MODEL PROVIDED.")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize semantic decomposer
        self.decomposer = SemanticDecomposer(
            wn_module=wn,
            nlp_func=self.nlp,
            embedding_model=self.embedding_model
        )

    
    def analyze_triple(
        self,
        subj_tok: Token|str,
        pred_tok: Token|str,
        obj_tok: Token|str,
        beam_width: int = 3,
        max_results_per_pair: int = 4,
        len_tolerance: int = 5
    ) -> List[List[str]]:
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

        if isinstance(subj_tok, str):
            subj_tok = self.nlp(subj_tok)[0]
        if isinstance(pred_tok, str):
            pred_tok = self.nlp(pred_tok)[0]
        if isinstance(obj_tok, str):
            obj_tok = self.nlp(obj_tok)[0]

        # Find connected paths
        try:
            result = self.decomposer.find_connected_shortest_paths(
                subj_tok=subj_tok,
                pred_tok=pred_tok,
                obj_tok=obj_tok,
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
            # Set up NLP pipeline
            try:
                self.nlp = spacy.load(kwargs['nlp_model'])
            except:
                print(f"WARNING: INVALID SPACY MODEL PROVIDED.")
        
        if 'embedding_model' in kwargs:
            self.embedding_model = kwargs['embedding_model']
        
        # Reinitialize decomposer with new settings
        self.decomposer = SemanticDecomposer(
            wn_module=wn,
            nlp_func=self.nlp,
            embedding_model=self.embedding_model
        )

    
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