import heapq
import itertools
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Any
import networkx as nx
import numpy as np

# Import WordNet for heuristic calculations
import nltk
from nltk.corpus import wordnet as wn

# Type aliases
SynsetName = str  # e.g., "dog.n.01"
Path = List[SynsetName]
BeamElement = Tuple[Tuple[SynsetName, str], Tuple[SynsetName, str], float]
GetNewBeamsFn = Callable[[nx.DiGraph, SynsetName, SynsetName], List[BeamElement]]
TopKBranchFn = Callable[[List[List], object, int], List[BeamElement]]
GlossSeedFn = Callable[[object], List]  # e.g., (parsed_gloss_doc) -> list of tokens



class PairwiseBidirectionalAStar:
    """
    Beam+depth-constrained, gloss-seeded bidirectional A* for pairwise synset search.

    Dependencies / injectable functions:
      - get_new_beams_fn(g, src, tgt) -> List[((src_node, rel),(tgt_node, rel), sim)]
      - top_k_branch_fn(candidates_lists, target_synset, beam_width) -> List[((synset, rel),(name,rel),sim)]
      - gloss_seed_fn(gloss_doc) -> list of tokens (subject/object/verb tokens)
      - wn and nlp (spaCy) are used by the outer pipeline; this class accepts seeds instead.

    Heuristics:
      - Embedding similarity -> h = 1 - sim (lower is better)
      - Gloss seeds get a small bonus (h -= GLOSS_BONUS)
    """
    GLOSS_BONUS = 0.15  # subtract from h for gloss-seeded nodes (tune)

    def __init__(
        self,
        g: nx.DiGraph,
        src: SynsetName,
        tgt: SynsetName,
        get_new_beams_fn: Optional[GetNewBeamsFn] = None,
        beam_width: int = 10,  # Optimized: increased from 3 to 10
        max_depth: int = 10,   # Optimized: increased from 6 to 10
        relax_beam: bool = True,  # Optimized: changed from False to True
        embedding_helper: Optional[object] = None,  # For hybrid heuristic
    ):
        """
        Args:
          g: nx.DiGraph — graph of synsets (nodes as synset names).
          src, tgt: synset node ids (strings).
          get_new_beams_fn: function to produce embedding-based beam pairs (optional).
          beam_width: beam width for initial seeding (optimized default: 10).
          max_depth: maximum hops allowed per side (optimized default: 10).
          relax_beam: if True, allow exploring nodes outside the allowed beams (optimized default: True).
          heuristic_type: type of heuristic function ("hybrid", "embedding", "wordnet", "uniform").
          embedding_helper: embedding helper instance for hybrid heuristic calculations.
        """
        self.g = g
        self.src = src
        self.tgt = tgt

        if not get_new_beams_fn:
            def default_get_new_beams_fn(g, s, t) -> List[BeamElement]:
                return []
            get_new_beams_fn = default_get_new_beams_fn
        self.get_new_beams_fn = get_new_beams_fn

        self.beam_width = beam_width
        self.max_depth = max_depth
        self.relax_beam = relax_beam
        self.embedding_helper = embedding_helper

        # will be set by _build_allowed_and_heuristics
        self.src_allowed: Set[SynsetName] = set()
        self.tgt_allowed: Set[SynsetName] = set()
        self.h_forward: Dict[SynsetName, float] = {}
        self.h_backward: Dict[SynsetName, float] = {}

        # search state
        self._counter = itertools.count()
        self.open_f: List[Tuple[float, int, SynsetName]] = []
        self.open_b: List[Tuple[float, int, SynsetName]] = []
        self.g_f: Dict[SynsetName, float] = {}
        self.g_b: Dict[SynsetName, float] = {}
        self.depth_f: Dict[SynsetName, int] = {}
        self.depth_b: Dict[SynsetName, int] = {}
        self.parent_f: Dict[SynsetName, Optional[SynsetName]] = {}
        self.parent_b: Dict[SynsetName, Optional[SynsetName]] = {}
        self.closed_f: Set[SynsetName] = set()
        self.closed_b: Set[SynsetName] = set()

    # -------------------------
    # WordNet Distance Estimation for Hybrid Heuristic
    # -------------------------
    def _overlapping_lch_paths(self, syn1, syn2) -> List[Any]:
        lchs = syn1.lowest_common_hypernyms(syn2)
        print("LCHs:", [lch.name() for lch in lchs])
        common_paths = []

        for p1 in syn1.hypernym_paths():
            for p2 in syn2.hypernym_paths():
                if any(lch in p1 for lch in lchs) and any(lch in p2 for lch in lchs):
                    # truncate the paths until they've got one of the lchs
                    while p1 and p2 and p1[0] not in lchs:
                        last_lch = p1[0]
                        p1 = p1[1:]
                        p2 = p2[1:]
                    # get the shared lch path
                    common_paths.append(p1[::-1] + p2[1:])

        return common_paths

    def _get_embedding_similarity(self, current: SynsetName, target: SynsetName) -> float:
        """Get embedding-based similarity between two synsets."""
        if not self.embedding_helper:
            return 0.5  # Default similarity
        
        try:
            # This would need to be implemented based on the embedding helper interface
            # For now, return a placeholder that could be replaced with actual implementation
            return 0.5
        except Exception:
            return 0.5

    def _calculate_heuristic(self, current: SynsetName, target: SynsetName) -> float:
        # Use embedding similarity if available
        embedding_sim = self._get_embedding_similarity(current, target)

        current_synset = wn.synset(current)
        target_synset = wn.synset(target)
        if current_synset and target_synset:
            # Wu-Palmer Similarity: Return a score denoting how similar two word senses are,
            #   based on the depth of the two senses in the taxonomy and that of their Least Common Subsumer (most specific ancestor node).
            wordnet_sim = current_synset.wup_similarity(wn.synset(target_synset)) or 0.5
            return (embedding_sim + wordnet_sim) / 2.0
        
        return embedding_sim

    # -------------------------
    # Setup: allowed sets & heuristics
    # -------------------------
    def _build_allowed_and_heuristics(self):
        """
        Build allowed node sets and heuristic maps using get_new_beams_fn and gloss seeds.
        - src_allowed/tgt_allowed: union of beam nodes and explicit gloss seeds + src/tgt.
        - h_forward/h_backward: h = 1 - sim (embedding), gloss seeds get bonus.
        """
        beams = self.get_new_beams_fn(self.g, self.src, self.tgt)

        # base allowed sets from embedding beams
        src_beam_pairs = [b[0] for b in beams]
        tgt_beam_pairs = [b[1] for b in beams]
        self.src_allowed = {p[0] for p in src_beam_pairs}
        self.tgt_allowed = {p[0] for p in tgt_beam_pairs}

        # always include src and tgt
        self.src_allowed.add(self.src)
        self.tgt_allowed.add(self.tgt)

        # Build heuristics using the new hybrid system
        self.h_forward = {}
        self.h_backward = {}
        
        # Calculate heuristics for all allowed nodes using the new system
        all_nodes = self.src_allowed.union(self.tgt_allowed)
        all_nodes.update({self.src, self.tgt})
        
        for node in all_nodes:
            # Forward heuristic: estimate distance to target
            h_forward_val = self._calculate_heuristic(node, self.tgt)
            self.h_forward[node] = h_forward_val
            
            # Backward heuristic: estimate distance from source
            h_backward_val = self._calculate_heuristic(self.src, node)
            self.h_backward[node] = h_backward_val
        
        # For nodes from embedding beams, use the beam similarity if it's better
        for (s_pair, t_pair, sim) in beams:
            s_node = s_pair[0]
            t_node = t_pair[0]
            h_val = max(0.0, 1.0 - float(sim))
            # Use beam similarity if it's better (lower)
            if s_node in self.h_forward:
                self.h_forward[s_node] = min(self.h_forward[s_node], h_val)
            if t_node in self.h_backward:
                self.h_backward[t_node] = min(self.h_backward[t_node], h_val)

    # -------------------------
    # Initialization of queues
    # -------------------------
    def _init_search_state(self):
        self._counter = itertools.count()
        self.open_f = []
        self.open_b = []
        self.g_f = {self.src: 0.0}
        self.g_b = {self.tgt: 0.0}
        self.depth_f = {self.src: 0}
        self.depth_b = {self.tgt: 0}
        self.parent_f = {self.src: None}
        self.parent_b = {self.tgt: None}
        self.closed_f = set()
        self.closed_b = set()

        heapq.heappush(self.open_f, (self.h_forward.get(self.src, 0.0), next(self._counter), self.src))
        heapq.heappush(self.open_b, (self.h_backward.get(self.tgt, 0.0), next(self._counter), self.tgt))

    # -------------------------
    # Utilities
    # -------------------------
    def _edge_weight(self, u: SynsetName, v: SynsetName) -> float:
        try:
            return float(self.g[u][v].get("weight", 1.0))
        except Exception:
            return 1.0

    def _allowed_forward(self, node: SynsetName) -> bool:
        return self.relax_beam\
            or node in self.src_allowed\
                or node in self.tgt_allowed\
                    or node == self.tgt\
                        or node == self.src

    def _allowed_backward(self, node: SynsetName) -> bool:
        return self.relax_beam\
            or node in self.tgt_allowed\
                or node in self.src_allowed\
                    or node == self.src\
                        or node == self.tgt

    # -------------------------
    # Expand one node from forward/back
    # -------------------------
    def _expand_forward_once(self) -> Optional[SynsetName]:
        """Pop one element from forward open and expand. Return meeting node if found."""
        while self.open_f:
            _, _, current = heapq.heappop(self.open_f)
            if current in self.closed_f:
                continue
            self.closed_f.add(current)

            # if already settled by backward:
            if current in self.closed_b:
                return current

            curr_depth = self.depth_f.get(current, 0)
            if curr_depth >= self.max_depth:
                continue

            for nbr in self.g.neighbors(current):
                if not self._allowed_forward(nbr):
                    continue
                tentative_g = self.g_f[current] + self._edge_weight(current, nbr)
                tentative_depth = curr_depth + 1
                if tentative_depth > self.max_depth:
                    continue
                if tentative_g < self.g_f.get(nbr, float("inf")):
                    self.g_f[nbr] = tentative_g
                    self.depth_f[nbr] = tentative_depth
                    self.parent_f[nbr] = current
                    # Calculate heuristic dynamically if not in map
                    if nbr not in self.h_forward:
                        self.h_forward[nbr] = self._calculate_heuristic(nbr, self.tgt)
                    f_score = tentative_g + self.h_forward[nbr]
                    heapq.heappush(self.open_f, (f_score, next(self._counter), nbr))
                    if nbr in self.closed_b:
                        return nbr
            return None
        return None

    def _expand_backward_once(self) -> Optional[SynsetName]:
        """Pop one element from backward open and expand predecessors. Return meeting node if found."""
        while self.open_b:
            _, _, current = heapq.heappop(self.open_b)
            if current in self.closed_b:
                continue
            self.closed_b.add(current)

            if current in self.closed_f:
                return current

            curr_depth = self.depth_b.get(current, 0)
            if curr_depth >= self.max_depth:
                continue

            for nbr in self.g.predecessors(current):
                if not self._allowed_backward(nbr):
                    continue
                tentative_g = self.g_b[current] + self._edge_weight(nbr, current)
                tentative_depth = curr_depth + 1
                if tentative_depth > self.max_depth:
                    continue
                if tentative_g < self.g_b.get(nbr, float("inf")):
                    self.g_b[nbr] = tentative_g
                    self.depth_b[nbr] = tentative_depth
                    self.parent_b[nbr] = current
                    # Calculate heuristic dynamically if not in map
                    if nbr not in self.h_backward:
                        self.h_backward[nbr] = self._calculate_heuristic(self.src, nbr)
                    f_score = tentative_g + self.h_backward[nbr]
                    heapq.heappush(self.open_b, (f_score, next(self._counter), nbr))
                    if nbr in self.closed_f:
                        return nbr
            return None
        return None

    # -------------------------
    # Path reconstruction
    # -------------------------
    def _reconstruct_path(self, meet: SynsetName) -> Path:
        # forward part
        path_f: Path = []
        n = meet
        while n is not None:
            path_f.append(n)
            n = self.parent_f.get(n)
        path_f.reverse()

        # backward part (exclude meet to avoid dup)
        path_b: Path = []
        n = self.parent_b.get(meet)
        while n is not None:
            path_b.append(n)
            n = self.parent_b.get(n)

        return path_f + path_b

    # -------------------------
    # Core: find multiple paths
    # -------------------------
    def find_paths(self, max_results: int = 3, len_tolerance: int = 3) -> List[Tuple[Path, float]]:
        """
        Run bidirectional beam+depth constrained search and return up to max_results unique paths.
        Paths returned have total cost (sum of g_f + g_b at meet) and are kept while <= best_cost + len_tolerance.

        len_tolerance: integer extra hops allowed beyond the best (shortest) path length.
        """
        # Setup
        self._build_allowed_and_heuristics()
        self._init_search_state()

        results: List[Tuple[Path, float]] = []
        seen_paths: Set[Tuple[SynsetName, ...]] = set()
        best_cost: Optional[float] = None

        # helper to compute current lower bound on any next path cost
        def current_lower_bound() -> float:
            min_f_f = self.open_f[0][0] if self.open_f else float("inf")
            min_f_b = self.open_b[0][0] if self.open_b else float("inf")
            return min_f_f + min_f_b

        # main loop
        while (self.open_f or self.open_b) and len(results) < max_results:
            # stopping condition: if we have a best_cost and the conservative lower bound
            # exceeds best_cost + len_tolerance, we can stop.
            if best_cost is not None and current_lower_bound() > best_cost + float(len_tolerance):
                break

            # expand side with smaller top f
            top_f = self.open_f[0][0] if self.open_f else float("inf")
            top_b = self.open_b[0][0] if self.open_b else float("inf")

            meet = None
            if top_f <= top_b:
                meet = self._expand_forward_once()
            else:
                meet = self._expand_backward_once()

            if meet is None:
                continue

            # When meet occurs, reconstruct path and compute cost.
            path = self._reconstruct_path(meet)
            path_key = tuple(path)
            # compute cost: if both g maps contain meet, sum them; otherwise try to compute from edges
            cost_f = self.g_f.get(meet, float("inf"))
            cost_b = self.g_b.get(meet, float("inf"))
            total_cost = cost_f + cost_b if (cost_f < float("inf") and cost_b < float("inf")) else float("inf")

            # Hop-based cost fallback if edge weights are all 1 or inexact: length-1 equals hops
            if total_cost == float("inf"):
                # fallback to hop count
                total_cost = len(path) - 1

            if path_key not in seen_paths:
                seen_paths.add(path_key)
                # Accept path if within tolerance of current best
                if best_cost is None or total_cost <= best_cost + float(len_tolerance):
                    results.append((path, total_cost))
                    if best_cost is None or total_cost < best_cost:
                        best_cost = total_cost

            # continue searching for more meets until stopping condition triggers

        return results