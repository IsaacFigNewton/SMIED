from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable, Set
import numpy as np

# Type aliases
SynsetName = str
BeamElement = Tuple[Tuple[SynsetName, str], Tuple[SynsetName, str], float]
TopKBranchFn = Callable[[List[List], object, int], List[BeamElement]]


class IEmbeddingHelper(ABC):
    """
    Interface for embedding-based computations for synset analysis.
    
    Defines the contract for computing synset centroids, embedding lexical relations,
    and finding similarity-based alignments between synsets.
    """
    
    @abstractmethod
    def get_synset_embedding_centroid(self, synset, model) -> np.ndarray:
        """
        Given a synset, compute centroid (mean) of embeddings for lemmas.
        
        Args:
            synset: WordNet synset object
            model: Embedding model
            
        Returns:
            Centroid embedding as numpy array, empty array if nothing found
        """
        pass
    
    @abstractmethod
    def embed_lexical_relations(self, synset, model) -> Dict[str, List[Tuple[SynsetName, np.ndarray]]]:
        """
        Return map: lexical_rel_name -> list of (synset_name, centroid ndarray)
        Filters out relations whose centroid is empty.
        
        Args:
            synset: WordNet synset object
            model: Embedding model
            
        Returns:
            Dictionary mapping relation names to lists of (synset_name, embedding) tuples
        """
        pass
    
    @abstractmethod
    def get_embedding_similarities(self, rel_embs_1: List[Tuple[str, np.ndarray]], rel_embs_2: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """
        Return cosine similarity matrix (m x n) for lists of (name, centroid).
        
        Args:
            rel_embs_1: List of (name, embedding) tuples for first set
            rel_embs_2: List of (name, embedding) tuples for second set
            
        Returns:
            Cosine similarity matrix, empty if either input is empty
        """
        pass
    
    @abstractmethod
    def get_top_k_aligned_lex_rel_pairs(
        self,
        src_tgt_rel_map: Dict[str, str],
        src_emb_dict: Dict[str, List[Tuple[SynsetName, np.ndarray]]],
        tgt_emb_dict: Dict[str, List[Tuple[SynsetName, np.ndarray]]],
        beam_width: int = 3,
    ) -> List[BeamElement]:
        """
        Get top-k aligned lexical relation pairs based on embedding similarity.
        
        Args:
            src_tgt_rel_map: mapping from relation name in src to relation name in tgt
            src_emb_dict: source synset's lexical relation embeddings
            tgt_emb_dict: target synset's lexical relation embeddings
            beam_width: maximum number of pairs to return
            
        Returns:
            List of ((src_syn_name, src_rel), (tgt_syn_name, tgt_rel), similarity) tuples
        """
        pass
    
    @abstractmethod
    def get_new_beams_from_embeddings(
        self,
        g,  # nx.DiGraph
        src_name: SynsetName,
        tgt_name: SynsetName,
        wn_module,
        model,
        beam_width: int = 3,
        asymm_map: Optional[Dict[str, str]] = None,
        symm_map: Optional[Dict[str, str]] = None,
    ) -> List[BeamElement]:
        """
        Adapter to produce the beam format expected by PairwiseBidirectionalAStar.
        
        Args:
            g: NetworkX graph
            src_name: Source synset name string
            tgt_name: Target synset name string
            wn_module: WordNet interface
            model: Embedding model
            beam_width: Maximum beam width
            asymm_map: Asymmetric relation mappings
            symm_map: Symmetric relation mappings
            
        Returns:
            List of beam elements for pathfinding
        """
        pass
    
    @abstractmethod
    def build_gloss_seed_nodes_from_predicate(
        self,
        pred_syn,
        wn_module,
        nlp_func,
        mode: str = "subjects",
        extract_subjects_fn: Optional[Callable] = None,
        extract_objects_fn: Optional[Callable] = None,
        extract_verbs_fn: Optional[Callable] = None,
        top_k_branch_fn: Optional[TopKBranchFn] = None,
        target_synsets: Optional[List] = None,
        max_sample_size: int = 5,
        beam_width: int = 3,
    ) -> Set[SynsetName]:
        """
        Extract tokens from predicate synset gloss and return a set of synset-name seeds.
        
        Args:
            pred_syn: Predicate synset
            wn_module: WordNet interface
            nlp_func: spaCy NLP function
            mode: 'subjects'|'objects'|'verbs' decides which extractor to use
            extract_subjects_fn: Function to extract subjects from gloss
            extract_objects_fn: Function to extract objects from gloss
            extract_verbs_fn: Function to extract verbs from gloss
            top_k_branch_fn: Function to select top-k matching synsets
            target_synsets: Target synsets for alignment
            max_sample_size: Maximum number of tokens to process
            beam_width: Beam width for selection
            
        Returns:
            Set of synset names to use as seeds
        """
        pass