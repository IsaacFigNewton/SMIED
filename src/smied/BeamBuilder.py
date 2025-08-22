import networkx as nx

import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

from typing import List, Tuple, Dict
from .EmbeddingHelper import EmbeddingHelper


class BeamBuilder:
    """
    Constructs embedding-based beams for synset pair searches.
    
    Builds beams by analyzing lexical relations between synsets using embedding similarities.
    """
    
    def __init__(self, embedding_helper: EmbeddingHelper):
        self.embedding_helper = embedding_helper
        
        self.asymmetric_pairs_map = {
            # holonyms
            "part_holonyms": "part_meronyms",
            "substance_holonyms": "substance_meronyms",
            "member_holonyms": "member_meronyms",

            # meronyms
            "part_meronyms": "part_holonyms",
            "substance_meronyms": "substance_holonyms",
            "member_meronyms": "member_holonyms",

            # other
            "hypernyms": "hyponyms",
            "hyponyms": "hypernyms"
        }

        self.symmetric_pairs_map = {
            # holonyms
            "part_holonyms": "part_holonyms",
            "substance_holonyms": "substance_holonyms",
            "member_holonyms": "member_holonyms",

            # meronyms
            "part_meronyms": "part_meronyms",
            "substance_meronyms": "substance_meronyms",
            "member_meronyms": "member_meronyms",

            # other
            "hypernyms": "hypernyms",
            "hyponyms": "hyponyms",
            "entailments": "entailments",
            "causes": "causes",
            "also_sees": "also_sees",
            "verb_groups": "verb_groups"
        }

    def get_new_beams(
        self,
        g: nx.DiGraph,
        src: str,
        tgt: str,
        model,
        beam_width: int = 3
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str], float]]:
        """
        Get the k closest pairs of lexical relations between 2 synsets.

        Args:
            g: NetworkX graph of synsets
            src: Source synset name (e.g., 'dog.n.01')
            tgt: Target synset name (e.g., 'cat.n.01')
            model: Embedding model
            beam_width: max number of pairs to return

        Returns:
            List of tuples of the form:
              (
                (synset1, lexical_rel),
                (synset2, lexical_rel),
                relatedness
              )
        """
        # Build a map of each synset's associated lexical relations
        src_lex_rel_embs = self.embedding_helper.embed_lexical_relations(wn.synset(src), model)
        tgt_lex_rel_embs = self.embedding_helper.embed_lexical_relations(wn.synset(tgt), model)

        # ensure the edges in the nx graph align with those in the embedding maps
        src_neighbors = {n for n in g.neighbors(src)}
        for rel, synset_list in src_lex_rel_embs.items():
            if not all(s[0] in src_neighbors for s in synset_list):
                raise ValueError(f"Not all lexical properties of {src} ({[s[0] for s in synset_list]}) in graph for relation {rel}")
        
        tgt_neighbors = {n for n in g.neighbors(tgt)}
        for rel, synset_list in tgt_lex_rel_embs.items():
            if not all(s[0] in tgt_neighbors for s in synset_list):
                raise ValueError(f"Not all lexical properties of {tgt} ({[s[0] for s in synset_list]}) in graph for relation {rel}")

        # Get the asymmetric lexical relation pairings
        asymm_lex_rel_sims = self.embedding_helper.get_top_k_aligned_lex_rel_pairs(
            self.asymmetric_pairs_map,
            src_lex_rel_embs,
            tgt_lex_rel_embs,
            beam_width
        )
        
        # Get the symmetric lexical relation pairings
        symm_lex_rel_sims = self.embedding_helper.get_top_k_aligned_lex_rel_pairs(
            self.symmetric_pairs_map,
            src_lex_rel_embs,
            tgt_lex_rel_embs,
            beam_width
        )
        
        combined = asymm_lex_rel_sims + symm_lex_rel_sims
        beam = sorted(combined, key=lambda x: x[2], reverse=True)[:beam_width]
        return beam