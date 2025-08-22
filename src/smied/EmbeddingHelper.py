
import numpy as np

import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

from typing import Dict, List, Tuple, Optional, Callable, Set
from collections import defaultdict

# Type aliases
SynsetName = str
BeamElement = Tuple[Tuple[SynsetName, str], Tuple[SynsetName, str], float]
TopKBranchFn = Callable[[List[List], object, int], List[BeamElement]]


class EmbeddingHelper:
    """
    Handles embedding-based computations for synset analysis.
    
    Provides methods for computing synset centroids, embedding lexical relations,
    and finding similarity-based alignments between synsets.
    """
    
    def __init__(self):
        pass

    def get_synset_embedding_centroid(self, synset, model) -> np.ndarray:
        """
        Given a wn.synset, compute centroid (mean) of embeddings for lemmas.
        Returns empty np.array if nothing found.
        """
        try:
            lemmas = [lemma.name().lower().replace("_", " ") for lemma in synset.lemmas()]
            embeddings = []
            for lemma in lemmas:
                if lemma in model:
                    embeddings.append(np.asarray(model[lemma], dtype=float))
                elif lemma.replace(" ", "_") in model:
                    embeddings.append(np.asarray(model[lemma.replace(" ", "_")], dtype=float))
                elif " " in lemma:
                    # try individual words
                    words = lemma.split()
                    word_embs = [np.asarray(model[w], dtype=float) for w in words if w in model]
                    if word_embs:
                        embeddings.append(np.mean(word_embs, axis=0))
            if not embeddings:
                return np.array([])  # empty
            return np.mean(embeddings, axis=0)
        except Exception as e:
            # defensive: return empty arr on any error
            print(f"[get_synset_embedding_centroid] Error for {getattr(synset, 'name', lambda: synset)()}: {e}")
            return np.array([])

    def embed_lexical_relations(self, synset, model) -> Dict[str, List[Tuple[SynsetName, np.ndarray]]]:
        """
        Return map: lexical_rel_name -> list of (synset_name, centroid ndarray)
        Filters out relations whose centroid is empty.
        """
        def _rel_centroids(get_attr):
            try:
                items = []
                for s in get_attr(synset):
                    cent = self.get_synset_embedding_centroid(s, model)
                    if cent.size > 0:
                        items.append((s.name(), cent))
                return items
            except Exception as e:
                print(f"[embed_lexical_relations] error for {synset.name()}: {e}")
                return []

        return {
            "part_holonyms": _rel_centroids(lambda x: x.part_holonyms()),
            "substance_holonyms": _rel_centroids(lambda x: x.substance_holonyms()),
            "member_holonyms": _rel_centroids(lambda x: x.member_holonyms()),
            "part_meronyms": _rel_centroids(lambda x: x.part_meronyms()),
            "substance_meronyms": _rel_centroids(lambda x: x.substance_meronyms()),
            "member_meronyms": _rel_centroids(lambda x: x.member_meronyms()),
            "hypernyms": _rel_centroids(lambda x: x.hypernyms()),
            "hyponyms": _rel_centroids(lambda x: x.hyponyms()),
            "entailments": _rel_centroids(lambda x: x.entailments()),
            "causes": _rel_centroids(lambda x: x.causes()),
            "also_sees": _rel_centroids(lambda x: x.also_sees()),
            "verb_groups": _rel_centroids(lambda x: x.verb_groups()),
        }

    def get_embedding_similarities(self, rel_embs_1: List[Tuple[str, np.ndarray]], rel_embs_2: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """
        Return cosine similarity matrix (m x n) for lists of (name, centroid).
        If either is empty, returns empty (0xN or Mx0) array.
        """
        if not rel_embs_1 or not rel_embs_2:
            return np.zeros((0, 0))

        e1 = np.array([x[1] for x in rel_embs_1], dtype=float)  # (m,d)
        e2 = np.array([x[1] for x in rel_embs_2], dtype=float)  # (n,d)

        # avoid divide-by-zero: replace zero norms with eps
        e1_norms = np.linalg.norm(e1, axis=1, keepdims=True)
        e2_norms = np.linalg.norm(e2, axis=1, keepdims=True)
        e1_norms[e1_norms == 0] = 1e-8
        e2_norms[e2_norms == 0] = 1e-8

        e1u = e1 / e1_norms
        e2u = e2 / e2_norms

        sims = np.dot(e1u, e2u.T)
        return sims

    def get_top_k_aligned_lex_rel_pairs(
        self,
        src_tgt_rel_map: Dict[str, str],
        src_emb_dict: Dict[str, List[Tuple[SynsetName, np.ndarray]]],
        tgt_emb_dict: Dict[str, List[Tuple[SynsetName, np.ndarray]]],
        beam_width: int = 3,
    ) -> List[BeamElement]:
        """
        src_tgt_rel_map: mapping from relation name in src to relation name in tgt,
          e.g., {'hypernyms': 'hyponyms', ...}

        Returns list of ((src_syn_name, src_rel), (tgt_syn_name, tgt_rel), similarity)
        """
        rel_sims = []
        for e1_rel, e2_rel in src_tgt_rel_map.items():
            e1_list = src_emb_dict.get(e1_rel, [])
            e2_list = tgt_emb_dict.get(e2_rel, [])
            if not e1_list or not e2_list:
                continue
            sims = self.get_embedding_similarities(e1_list, e2_list)  # shape (m,n)
            if sims.size == 0:
                continue
            for i in range(sims.shape[0]):
                for j in range(sims.shape[1]):
                    try:
                        rel_sims.append(((e1_list[i][0], e1_rel), (e2_list[j][0], e2_rel), float(sims[i, j])))
                    except IndexError as ex:
                        raise IndexError(f"Index error in get_top_k_aligned_lex_rel_pairs: i={i}, j={j}, shapes e1={len(e1_list)}, e2={len(e2_list)}") from ex

        # sort and return top-k
        rel_sims.sort(key=lambda x: x[2], reverse=True)
        if beam_width <= 0:
            return rel_sims
        return rel_sims[:beam_width]

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
        - src_name / tgt_name are synset name strings (e.g., 'dog.n.01')
        - wn_module is your WordNet interface (e.g., nltk.corpus.wordnet as wn)
        - model is embedding model (contains token -> vector)
        """
        # default relation maps (tweak as needed)
        if asymm_map is None:
            asymm_map = {
                "hypernyms": "hyponyms",
                "hyponyms": "hypernyms",
                "part_meronyms": "part_holonyms",
                "member_meronyms": "member_holonyms",
                "substance_meronyms": "substance_holonyms",
                "entailments": "causes",
                "causes": "entailments",
            }
        if symm_map is None:
            symm_map = {
                "hypernyms": "hypernyms",
                "hyponyms": "hyponyms",
                "part_meronyms": "part_meronyms",
                "member_meronyms": "member_meronyms",
                "also_sees": "also_sees",
                "verb_groups": "verb_groups",
            }

        try:
            src_syn = wn_module.synset(src_name)
            tgt_syn = wn_module.synset(tgt_name)
        except Exception:
            # If synset names invalid, return empty beams
            return []

        # Embed lexical relations for both synsets
        src_emb_dict = self.embed_lexical_relations(src_syn, model)
        tgt_emb_dict = self.embed_lexical_relations(tgt_syn, model)

        # get top k pairs from asymmetric and symmetric maps
        asymm_pairs = self.get_top_k_aligned_lex_rel_pairs(asymm_map, src_emb_dict, tgt_emb_dict, beam_width=beam_width)
        symm_pairs = self.get_top_k_aligned_lex_rel_pairs(symm_map, src_emb_dict, tgt_emb_dict, beam_width=beam_width)

        combined = asymm_pairs + symm_pairs
        # sort by similarity and trim to beam_width
        combined.sort(key=lambda x: x[2], reverse=True)
        return combined[:beam_width]

    def build_gloss_seed_nodes_from_predicate(
        self,
        pred_syn,
        wn_module,
        nlp_func,
        mode: str = "subjects",  # "subjects" or "objects" or "verbs"
        extract_subjects_fn: Optional[Callable] = None,
        extract_objects_fn: Optional[Callable] = None,
        extract_verbs_fn: Optional[Callable] = None,
        top_k_branch_fn: Optional[TopKBranchFn] = None,
        target_synsets: Optional[List] = None,
        max_sample_size: int = 5,
        beam_width: int = 3,
    ) -> Set[SynsetName]:
        """
        Extract tokens from pred_syn gloss and return a set of synset-name seeds.
        If top_k_branch_fn provided, use it to select top-k matching synsets.
        - pred_syn: wn.synset
        - nlp_func: spaCy call (text -> doc)
        - mode: 'subjects'|'objects'|'verbs' decides which extractor to use
        """
        doc = nlp_func(pred_syn.definition())
        tokens = []
        if mode == "subjects" and extract_subjects_fn is not None:
            tokens, _ = extract_subjects_fn(doc)
        elif mode == "objects" and extract_objects_fn is not None:
            tokens = extract_objects_fn(doc)
        elif mode == "verbs" and extract_verbs_fn is not None:
            tokens = extract_verbs_fn(doc)
        else:
            # fallback: use any nouns in doc
            tokens = [tok for tok in doc if tok.pos_ == "NOUN"]

        # candidate synset lists for each token
        candidate_synsets = []
        for tok in tokens[:max_sample_size]:
            try:
                cand = wn_module.synsets(tok.text, pos=wn_module.NOUN if mode != "verbs" else wn_module.VERB)
                candidate_synsets.append(cand)
            except Exception:
                candidate_synsets.append([])

        seeds = set()
        if top_k_branch_fn and target_synsets is not None:
            # top_k_branch_fn is expected to accept (candidates, target_synset_or_list, beam_width)
            top_k = top_k_branch_fn(candidate_synsets[:max_sample_size], target_synsets, beam_width)
            for (s_pair, _, _) in top_k:
                # s_pair is (synset_obj_or_name, lexical_rel); convert to name if synset object
                s = s_pair[0]
                if hasattr(s, "name"):
                    seeds.add(s.name())
                elif isinstance(s, str):
                    seeds.add(s)
        else:
            # conservative: add the first few candidate synsets' names
            for cand_list in candidate_synsets:
                for s in cand_list[:min(3, len(cand_list))]:
                    seeds.add(s.name())
        return seeds
