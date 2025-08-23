import networkx as nx
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet as wn

from typing import Optional, Callable, Dict, List, Tuple, Any
from .PairwiseBidirectionalAStar import PairwiseBidirectionalAStar
from .BeamBuilder import BeamBuilder
from .EmbeddingHelper import EmbeddingHelper
from .GlossParser import GlossParser


class SemanticDecomposer:
    """
    Main orchestrator class for semantic decomposition using WordNet paths.
    
    This class integrates the PairwiseBidirectionalAStar, BeamBuilder, EmbeddingHelper,
    and GlossParser components to find semantic paths between subject-predicate-object triples.
    
    Verbosity levels:
    - 0: Critical errors and final results only
    - 1: Method entry/exit, high-level progress, warnings
    - 2: Detailed debugging, intermediate results, parameter values
    """
    
    def __init__(self, wn_module, nlp_func, embedding_model=None, verbosity=0):
        """
        Initialize the SemanticDecomposer with required dependencies.
        
        Args:
            wn_module: WordNet module (e.g., nltk.corpus.wordnet)
            nlp_func: spaCy NLP function for text processing
            embedding_model: Optional embedding model for similarity computations
            verbosity: Debug verbosity level (0-2)
        """
        # Set verbosity level for debugging output
        self.verbosity = verbosity
        
        if self.verbosity >= 1:
            print(f"[SemanticDecomposer] Initializing with verbosity level {verbosity}")
        
        # Store core dependencies
        self.wn_module = wn_module
        self.nlp_func = nlp_func
        self.embedding_model = embedding_model
        
        if self.verbosity >= 2:
            print(f"[SemanticDecomposer] WordNet module: {type(wn_module).__name__}")
            print(f"[SemanticDecomposer] NLP function: {type(nlp_func).__name__}")
            print(f"[SemanticDecomposer] Embedding model: {type(embedding_model).__name__ if embedding_model else 'None'}")
        
        # Initialize component classes
        if self.verbosity >= 1:
            print("[SemanticDecomposer] Initializing component classes...")
        
        self.embedding_helper = EmbeddingHelper()
        self.beam_builder = BeamBuilder(self.embedding_helper)
        self.gloss_parser = GlossParser(nlp_func=nlp_func)
        
        if self.verbosity >= 2:
            print("[SemanticDecomposer] Components initialized:")
            print(f"  - EmbeddingHelper: {type(self.embedding_helper).__name__}")
            print(f"  - BeamBuilder: {type(self.beam_builder).__name__}")
            print(f"  - GlossParser: {type(self.gloss_parser).__name__}")
        
        # Cached graph for performance optimization
        self._synset_graph = None
        
        if self.verbosity >= 1:
            print("[SemanticDecomposer] Initialization complete")

    def find_connected_shortest_paths(
        self,
        subject_word: str,
        predicate_word: str,
        object_word: str,
        model=None,  # embedding model
        g: nx.DiGraph = None,  # synset graph
        max_depth: int = 10,
        max_self_intersection: int = 5,
        beam_width: int = 3,
        max_results_per_pair: int = 3,
        len_tolerance: int = 1,
        relax_beam: bool = False,
        max_sample_size: int = 5
    ):
        """
        Main entry point for finding semantic paths between subject-predicate-object triples.
        Uses gloss parsing and hypernym exploration based on the old BFS seeding approach.
        Returns the best connected path in the old format for backward compatibility.
        """
        if self.verbosity >= 1:
            print(f"\n[find_connected_shortest_paths] Starting path search for: '{subject_word}' -> '{predicate_word}' -> '{object_word}'")
            
        if self.verbosity >= 2:
            print(f"[find_connected_shortest_paths] Parameters:")
            print(f"  - max_depth: {max_depth}")
            print(f"  - max_self_intersection: {max_self_intersection}")
            print(f"  - beam_width: {beam_width}")
            print(f"  - max_results_per_pair: {max_results_per_pair}")
            print(f"  - len_tolerance: {len_tolerance}")
            print(f"  - relax_beam: {relax_beam}")
            print(f"  - max_sample_size: {max_sample_size}")
        
        # Use provided model or fallback to instance model
        if model is None:
            model = self.embedding_model
            
        if self.verbosity >= 2:
            print(f"[find_connected_shortest_paths] Using embedding model: {type(model).__name__ if model else 'None'}")

        # Create the beam function if we have a model
        get_new_beams_fn = None
        if model is not None and g is not None:
            if self.verbosity >= 2:
                print("[find_connected_shortest_paths] Creating beam function with embedding model")
            get_new_beams_fn = lambda graph, src, tgt: self.embedding_helper.get_new_beams_from_embeddings(
                graph, src, tgt, self.wn_module, model, beam_width=beam_width
            )
        elif self.verbosity >= 2:
            print("[find_connected_shortest_paths] No beam function created (missing model or graph)")

        # Build the graph if not provided
        if g is None:
            if self.verbosity >= 1:
                print("[find_connected_shortest_paths] Building synset graph...")
            g = self.build_synset_graph()
            if self.verbosity >= 2:
                print(f"[find_connected_shortest_paths] Graph built with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
        elif self.verbosity >= 2:
            print(f"[find_connected_shortest_paths] Using provided graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")

        # Get synsets for each word
        if self.verbosity >= 1:
            print("[find_connected_shortest_paths] Looking up synsets for each word...")
            
        subject_synsets = self.wn_module.synsets(subject_word, pos=self.wn_module.NOUN)
        predicate_synsets = self.wn_module.synsets(predicate_word, pos=self.wn_module.VERB)
        object_synsets = self.wn_module.synsets(object_word, pos=self.wn_module.NOUN)
        
        if self.verbosity >= 2:
            print(f"[find_connected_shortest_paths] Found synsets:")
            print(f"  - Subject '{subject_word}': {len(subject_synsets)} synsets")
            print(f"  - Predicate '{predicate_word}': {len(predicate_synsets)} synsets")
            print(f"  - Object '{object_word}': {len(object_synsets)} synsets")
            
        # Check if we found any synsets
        if not subject_synsets:
            if self.verbosity >= 0:
                print(f"[ERROR] No noun synsets found for subject word: '{subject_word}'")
            return None, None, None
            
        if not predicate_synsets:
            if self.verbosity >= 0:
                print(f"[ERROR] No verb synsets found for predicate word: '{predicate_word}'")
            return None, None, None
            
        if not object_synsets:
            if self.verbosity >= 0:
                print(f"[ERROR] No noun synsets found for object word: '{object_word}'")
            return None, None, None

        # Initialize tracking variables for best path found
        best_combined_path_length = float('inf')
        best_subject_path = None
        best_object_path = None
        best_predicate = None
        
        if self.verbosity >= 1:
            print(f"[find_connected_shortest_paths] Searching for connected paths through {len(predicate_synsets)} predicate synsets...")

        # Try each predicate synset as the connector
        for i, pred in enumerate(predicate_synsets):
            if self.verbosity >= 2:
                print(f"\n[find_connected_shortest_paths] Trying predicate {i+1}/{len(predicate_synsets)}: {pred.name()} - '{pred.definition()[:100]}...'")
                
            # Find paths from all subjects to this specific predicate
            if self.verbosity >= 2:
                print(f"[find_connected_shortest_paths] Finding subject->predicate paths...")
            subject_paths = self._find_subject_to_predicate_paths(
                subject_synsets, pred, g, get_new_beams_fn, beam_width, max_depth, 
                relax_beam, max_results_per_pair, len_tolerance, max_sample_size
            )
            
            if self.verbosity >= 2:
                print(f"[find_connected_shortest_paths] Found {len(subject_paths)} subject->predicate paths")

            # Find paths from this specific predicate to all objects  
            if self.verbosity >= 2:
                print(f"[find_connected_shortest_paths] Finding predicate->object paths...")
            object_paths = self._find_predicate_to_object_paths(
                pred, object_synsets, g, get_new_beams_fn, beam_width, max_depth,
                relax_beam, max_results_per_pair, len_tolerance, max_sample_size
            )
            
            if self.verbosity >= 2:
                print(f"[find_connected_shortest_paths] Found {len(object_paths)} predicate->object paths")

            # If we have both paths through this predicate, check if it's the best
            if subject_paths and object_paths:
                if self.verbosity >= 2:
                    print(f"[find_connected_shortest_paths] Evaluating path combinations (checking self-intersection <= {max_self_intersection})...")
                    
                valid_pairs = []
                for subj_path in subject_paths:
                    for obj_path in object_paths:
                        # Check self-intersection constraint
                        intersection_count = len(set(subj_path).intersection(set(obj_path)))
                        if intersection_count <= max_self_intersection:
                            combined_length = len(subj_path) + len(obj_path) - 1  # subtract 1 to avoid counting predicate twice
                            valid_pairs.append((
                                subj_path,
                                obj_path,
                                combined_length
                            ))
                            if self.verbosity >= 2:
                                print(f"[find_connected_shortest_paths] Valid pair found: length={combined_length}, intersection={intersection_count}")
                        elif self.verbosity >= 2:
                            print(f"[find_connected_shortest_paths] Path pair rejected: intersection={intersection_count} > {max_self_intersection}")

                if valid_pairs:
                    shortest_comb_path = min(valid_pairs, key=lambda x: x[2])
                    if shortest_comb_path[2] < best_combined_path_length:
                        if self.verbosity >= 2:
                            print(f"[find_connected_shortest_paths] New best path found! Length: {shortest_comb_path[2]} (previous best: {best_combined_path_length})")
                        best_combined_path_length = shortest_comb_path[2]
                        best_subject_path = shortest_comb_path[0]
                        best_object_path = shortest_comb_path[1]
                        best_predicate = pred
                    elif self.verbosity >= 2:
                        print(f"[find_connected_shortest_paths] Path length {shortest_comb_path[2]} not better than current best {best_combined_path_length}")
                elif self.verbosity >= 2:
                    print(f"[find_connected_shortest_paths] No valid path pairs found for predicate {pred.name()}")
            elif self.verbosity >= 2:
                print(f"[find_connected_shortest_paths] Missing paths for predicate {pred.name()}: subject_paths={len(subject_paths)}, object_paths={len(object_paths)}")

        # Final result logging
        if best_subject_path and best_object_path and best_predicate:
            if self.verbosity >= 1:
                print(f"\n[find_connected_shortest_paths] SUCCESS: Found connected path with total length {best_combined_path_length}")
                print(f"[find_connected_shortest_paths] Best predicate: {best_predicate.name()}")
        else:
            if self.verbosity >= 0:
                print(f"\n[find_connected_shortest_paths] WARNING: No connected path found for '{subject_word}' -> '{predicate_word}' -> '{object_word}'")
        
        return best_subject_path, best_object_path, best_predicate

    def _find_subject_to_predicate_paths(self, subject_synsets, predicate_synset, g, get_new_beams_fn, 
                                       beam_width, max_depth, relax_beam, max_results_per_pair, 
                                       len_tolerance, max_sample_size):
        """Find paths from subject synsets to predicate using gloss parsing strategies."""
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Starting with {len(subject_synsets)} subject synsets -> predicate {predicate_synset.name()}")
            
        subject_paths = []
        
        # Strategy 1: Look for active subjects in predicate's gloss
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Strategy 1: Parsing predicate gloss for subjects...")
            
        pred_gloss_parsed = self.gloss_parser.parse_gloss(predicate_synset.definition())
        
        if self.verbosity >= 2:
            if pred_gloss_parsed:
                print(f"[_find_subject_to_predicate_paths] Parsed gloss contains: {list(pred_gloss_parsed.keys())}")
            else:
                print(f"[_find_subject_to_predicate_paths] Gloss parsing failed or returned empty")
                
        if pred_gloss_parsed and pred_gloss_parsed.get('subjects'):
            if self.verbosity >= 2:
                print(f"[_find_subject_to_predicate_paths] Found {len(pred_gloss_parsed['subjects'])} subject candidates in predicate gloss")
                
            # Get subject synsets from parsed gloss
            matched_synsets = self._get_best_synset_matches(
                [pred_gloss_parsed['subjects']], subject_synsets
            )
            
            if self.verbosity >= 2:
                print(f"[_find_subject_to_predicate_paths] Found {len(matched_synsets)} matching synsets from gloss")
            
            for subj_synset in subject_synsets:
                for matched_synset in matched_synsets:
                    path = self._find_path_between_synsets(
                        subj_synset, matched_synset, g, get_new_beams_fn, 
                        beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
                    )
                    if path:
                        # Extend path to include predicate
                        complete_path = path + [predicate_synset]
                        subject_paths.append(complete_path)
                        if self.verbosity >= 2:
                            print(f"[_find_subject_to_predicate_paths] Strategy 1 found path of length {len(complete_path)}")
        elif self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Strategy 1: No subjects found in predicate gloss")
        
        # Strategy 2: Look for verbs in subject glosses
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Strategy 2: Looking for verbs in subject glosses...")
            
        for subj_synset in subject_synsets:
            if self.verbosity >= 2:
                print(f"[_find_subject_to_predicate_paths] Processing subject synset: {subj_synset.name()}")
                
            subj_gloss_parsed = self.gloss_parser.parse_gloss(subj_synset.definition())
            if subj_gloss_parsed and subj_gloss_parsed.get('predicates'):
                if self.verbosity >= 2:
                    print(f"[_find_subject_to_predicate_paths] Found {len(subj_gloss_parsed['predicates'])} predicate candidates in subject gloss")
                    
                # Get predicate synsets from parsed subject gloss
                matched_synsets = self._get_best_synset_matches(
                    [subj_gloss_parsed['predicates']], [predicate_synset]
                )
                
                for matched_synset in matched_synsets:
                    path = self._find_path_between_synsets(
                        matched_synset, predicate_synset, g, get_new_beams_fn,
                        beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
                    )
                    if path:
                        # Prepend subject to path
                        complete_path = [subj_synset] + path
                        subject_paths.append(complete_path)
                        if self.verbosity >= 2:
                            print(f"[_find_subject_to_predicate_paths] Strategy 2 found path of length {len(complete_path)}")
        
        # Strategy 3: Explore hypernyms if no direct paths found
        if not subject_paths:
            if self.verbosity >= 2:
                print(f"[_find_subject_to_predicate_paths] Strategy 3: No direct paths found, exploring hypernym connections...")
            hypernym_paths = self._explore_hypernym_paths(
                subject_synsets, [predicate_synset], g, get_new_beams_fn,
                beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
            )
            subject_paths.extend(hypernym_paths)
            if self.verbosity >= 2:
                print(f"[_find_subject_to_predicate_paths] Strategy 3 found {len(hypernym_paths)} hypernym paths")
        elif self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Skipping Strategy 3: {len(subject_paths)} direct paths already found")
            
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Returning {len(subject_paths)} total subject->predicate paths")
            
        return subject_paths

    def _find_predicate_to_object_paths(self, predicate_synset, object_synsets, g, get_new_beams_fn,
                                      beam_width, max_depth, relax_beam, max_results_per_pair,
                                      len_tolerance, max_sample_size):
        """Find paths from predicate to object synsets using gloss parsing strategies."""
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Starting with predicate {predicate_synset.name()} -> {len(object_synsets)} object synsets")
            
        object_paths = []
        
        # Strategy 1: Look for objects (including passive subjects) in predicate's gloss
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Strategy 1: Parsing predicate gloss for objects...")
            
        pred_gloss_parsed = self.gloss_parser.parse_gloss(predicate_synset.definition())
        if pred_gloss_parsed and pred_gloss_parsed.get('objects'):
            if self.verbosity >= 2:
                print(f"[_find_predicate_to_object_paths] Found {len(pred_gloss_parsed['objects'])} object candidates in predicate gloss")
                
            # Get object synsets from parsed gloss
            matched_synsets = self._get_best_synset_matches(
                [pred_gloss_parsed['objects']], object_synsets
            )
            
            if self.verbosity >= 2:
                print(f"[_find_predicate_to_object_paths] Found {len(matched_synsets)} matching synsets from gloss")
            
            for matched_synset in matched_synsets:
                for obj_synset in object_synsets:
                    path = self._find_path_between_synsets(
                        matched_synset, obj_synset, g, get_new_beams_fn,
                        beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
                    )
                    if path:
                        # Prepend predicate to path
                        complete_path = [predicate_synset] + path
                        object_paths.append(complete_path)
                        if self.verbosity >= 2:
                            print(f"[_find_predicate_to_object_paths] Strategy 1 found path of length {len(complete_path)}")
        elif self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Strategy 1: No objects found in predicate gloss")
        
        # Strategy 2: Look for verbs in object glosses (including instrumental verbs)
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Strategy 2: Looking for verbs in object glosses...")
            
        for obj_synset in object_synsets:
            if self.verbosity >= 2:
                print(f"[_find_predicate_to_object_paths] Processing object synset: {obj_synset.name()}")
                
            obj_gloss_parsed = self.gloss_parser.parse_gloss(obj_synset.definition())
            if obj_gloss_parsed and obj_gloss_parsed.get('predicates'):
                if self.verbosity >= 2:
                    print(f"[_find_predicate_to_object_paths] Found {len(obj_gloss_parsed['predicates'])} predicate candidates in object gloss")
                    
                # Get predicate synsets from parsed object gloss
                matched_synsets = self._get_best_synset_matches(
                    [obj_gloss_parsed['predicates']], [predicate_synset]
                )
                
                for matched_synset in matched_synsets:
                    path = self._find_path_between_synsets(
                        predicate_synset, matched_synset, g, get_new_beams_fn,
                        beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
                    )
                    if path:
                        # Extend path to include object
                        complete_path = path + [obj_synset]
                        object_paths.append(complete_path)
                        if self.verbosity >= 2:
                            print(f"[_find_predicate_to_object_paths] Strategy 2 found path of length {len(complete_path)}")
        
        # Strategy 3: Explore hypernyms if no direct paths found
        if not object_paths:
            if self.verbosity >= 2:
                print(f"[_find_predicate_to_object_paths] Strategy 3: No direct paths found, exploring hypernym connections...")
            hypernym_paths = self._explore_hypernym_paths(
                [predicate_synset], object_synsets, g, get_new_beams_fn,
                beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
            )
            object_paths.extend(hypernym_paths)
            if self.verbosity >= 2:
                print(f"[_find_predicate_to_object_paths] Strategy 3 found {len(hypernym_paths)} hypernym paths")
        elif self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Skipping Strategy 3: {len(object_paths)} direct paths already found")
            
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Returning {len(object_paths)} total predicate->object paths")
            
        return object_paths

    def _get_best_synset_matches(self, candidate_lists, target_synsets, top_k=3):
        """Get the best matching synsets from candidates based on similarity to targets."""
        if self.verbosity >= 2:
            print(f"[_get_best_synset_matches] Matching {len(candidate_lists)} candidate lists against {len(target_synsets)} targets (top_k={top_k})")
            
        matches = []
        
        for i, candidates in enumerate(candidate_lists):
            if not candidates:
                if self.verbosity >= 2:
                    print(f"[_get_best_synset_matches] Candidate list {i+1} is empty")
                continue
                
            if self.verbosity >= 2:
                print(f"[_get_best_synset_matches] Processing {len(candidates)} candidates from list {i+1}")
                
            for candidate in candidates:
                for target in target_synsets:
                    try:
                        # Use path similarity if available, otherwise fall back to simple heuristics
                        similarity = candidate.path_similarity(target) or 0.0
                        matches.append((candidate, similarity))
                        if self.verbosity >= 2 and similarity > 0.5:
                            print(f"[_get_best_synset_matches] High similarity match: {candidate.name()} <-> {target.name()} = {similarity:.3f}")
                    except Exception as e:
                        if self.verbosity >= 2:
                            print(f"[_get_best_synset_matches] Path similarity failed for {candidate.name()} <-> {target.name()}: {e}")
                        # Fallback for synsets that don't support path_similarity
                        matches.append((candidate, 0.1))
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        result_synsets = [match[0] for match in matches[:top_k]]
        
        if self.verbosity >= 2:
            print(f"[_get_best_synset_matches] Returning {len(result_synsets)} best matches (from {len(matches)} total)")
            for i, synset in enumerate(result_synsets):
                print(f"[_get_best_synset_matches] Match {i+1}: {synset.name()} (similarity: {matches[i][1]:.3f})")
                
        return result_synsets

    def _find_path_between_synsets(self, src_synset, tgt_synset, g, get_new_beams_fn,
                                 beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find a path between two synsets using the pathfinder."""
        if self.verbosity >= 2:
            print(f"[_find_path_between_synsets] Searching path: {src_synset.name()} -> {tgt_synset.name()}")
            
        # Check if both synsets exist in the graph
        if src_synset.name() not in g:
            if self.verbosity >= 2:
                print(f"[_find_path_between_synsets] Source synset {src_synset.name()} not found in graph")
            return None
            
        if tgt_synset.name() not in g:
            if self.verbosity >= 2:
                print(f"[_find_path_between_synsets] Target synset {tgt_synset.name()} not found in graph")
            return None
            
        # Create pathfinder instance
        if self.verbosity >= 2:
            print(f"[_find_path_between_synsets] Creating PairwiseBidirectionalAStar with max_depth={max_depth}, beam_width={beam_width}")
            
        path_finder = PairwiseBidirectionalAStar(
            g=g,
            src=src_synset.name(),
            tgt=tgt_synset.name(),
            get_new_beams_fn=get_new_beams_fn,
            beam_width=beam_width,
            max_depth=max_depth,
            relax_beam=relax_beam
        )
        
        # Search for paths
        paths = path_finder.find_paths(max_results=max_results_per_pair, len_tolerance=len_tolerance)
        
        if paths:
            # Convert first path back to synset objects
            path_names, cost = paths[0]
            synset_path = [self.wn_module.synset(name) for name in path_names]
            
            if self.verbosity >= 2:
                print(f"[_find_path_between_synsets] Found path of length {len(synset_path)} with cost {cost:.3f}")
                print(f"[_find_path_between_synsets] Path: {' -> '.join(path_names)}")
                
            return synset_path
        else:
            if self.verbosity >= 2:
                print(f"[_find_path_between_synsets] No path found between {src_synset.name()} and {tgt_synset.name()}")
                
        return None

    def _explore_hypernym_paths(self, src_synsets, tgt_synsets, g, get_new_beams_fn,
                              beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Explore paths through hypernyms when direct connections aren't found."""
        if self.verbosity >= 2:
            print(f"[_explore_hypernym_paths] Exploring hypernym connections between {len(src_synsets)} source and {len(tgt_synsets)} target synsets")
            
        paths = []
        
        for src_synset in src_synsets:
            src_hypernyms = src_synset.hypernyms()
            
            if self.verbosity >= 2:
                print(f"[_explore_hypernym_paths] Source synset {src_synset.name()} has {len(src_hypernyms)} hypernyms")
            
            for tgt_synset in tgt_synsets:
                tgt_hypernyms = tgt_synset.hypernyms()
                
                if self.verbosity >= 2:
                    print(f"[_explore_hypernym_paths] Target synset {tgt_synset.name()} has {len(tgt_hypernyms)} hypernyms")
                
                # Try connecting through hypernyms (limit to top 3 for performance)
                for i, src_hyp in enumerate(src_hypernyms[:3]):
                    for j, tgt_hyp in enumerate(tgt_hypernyms[:3]):
                        if self.verbosity >= 2:
                            print(f"[_explore_hypernym_paths] Trying hypernym connection: {src_hyp.name()} -> {tgt_hyp.name()}")
                            
                        path = self._find_path_between_synsets(
                            src_hyp, tgt_hyp, g, get_new_beams_fn,
                            beam_width, max_depth-2, relax_beam, max_results_per_pair, len_tolerance
                        )
                        if path:
                            # Create complete path: src -> src_hyp -> ... -> tgt_hyp -> tgt
                            complete_path = [src_synset] + path + [tgt_synset]
                            paths.append(complete_path)
                            
                            if self.verbosity >= 2:
                                print(f"[_explore_hypernym_paths] Found hypernym path of length {len(complete_path)}")
        
        if self.verbosity >= 2:
            print(f"[_explore_hypernym_paths] Returning {len(paths)} hypernym-based paths")
            
        return paths

    def build_synset_graph(self) -> nx.DiGraph:
        """
        Build a directed graph of synsets with their lexical relations.
        Uses caching to avoid rebuilding the graph multiple times.
        """
        if self._synset_graph is not None:
            if self.verbosity >= 1:
                print(f"[build_synset_graph] Using cached graph with {self._synset_graph.number_of_nodes()} nodes")
            return self._synset_graph
            
        if self.verbosity >= 1:
            print("[build_synset_graph] Building new synset graph from WordNet...")
            
        g = nx.DiGraph()

        # Get all synsets (you may want to limit this for performance)
        if self.verbosity >= 1:
            print("[build_synset_graph] Retrieving all synsets from WordNet...")
            
        all_synsets = list(self.wn_module.all_synsets())
        
        if self.verbosity >= 1:
            print(f"[build_synset_graph] Found {len(all_synsets)} total synsets")
            print("[build_synset_graph] Adding nodes to graph...")

        # Add nodes to the graph
        for i, synset in enumerate(all_synsets):
            g.add_node(synset.name())
            
            # Progress reporting for large graphs
            if self.verbosity >= 2 and (i + 1) % 10000 == 0:
                print(f"[build_synset_graph] Added {i + 1}/{len(all_synsets)} nodes")
                
        if self.verbosity >= 1:
            print(f"[build_synset_graph] All {len(all_synsets)} nodes added. Adding edges...")

        # Add edges based on lexical relations
        edge_count = 0
        relation_counts = {}
        
        for i, synset in enumerate(all_synsets):
            synset_name = synset.name()
            
            # Progress reporting for large graphs
            if self.verbosity >= 2 and (i + 1) % 10000 == 0:
                print(f"[build_synset_graph] Processing relations for synset {i + 1}/{len(all_synsets)}")

            # Add various relation types as edges
            relations = {
                'hypernyms': synset.hypernyms(),
                'hyponyms': synset.hyponyms(),
                'member_holonyms': synset.member_holonyms() if hasattr(synset, 'member_holonyms') else [],
                'part_holonyms': synset.part_holonyms() if hasattr(synset, 'part_holonyms') else [],
                'substance_holonyms': synset.substance_holonyms() if hasattr(synset, 'substance_holonyms') else [],
                'member_meronyms': synset.member_meronyms() if hasattr(synset, 'member_meronyms') else [],
                'part_meronyms': synset.part_meronyms() if hasattr(synset, 'part_meronyms') else [],
                'substance_meronyms': synset.substance_meronyms() if hasattr(synset, 'substance_meronyms') else [],
                'similar_tos': synset.similar_tos() if hasattr(synset, 'similar_tos') else [],
                'also_sees': synset.also_sees() if hasattr(synset, 'also_sees') else [],
                'verb_groups': synset.verb_groups() if hasattr(synset, 'verb_groups') else [],
                'entailments': synset.entailments() if hasattr(synset, 'entailments') else [],
                'causes': synset.causes() if hasattr(synset, 'causes') else [],
            }

            # Process each relation type
            for rel_type, related_synsets in relations.items():
                if rel_type not in relation_counts:
                    relation_counts[rel_type] = 0
                    
                for related in related_synsets:
                    if related.name() in g:
                        g.add_edge(synset_name, related.name(),
                                  relation=rel_type, weight=1.0)
                        edge_count += 1
                        relation_counts[rel_type] += 1
                    elif self.verbosity >= 2:
                        print(f"[build_synset_graph] WARNING: Related synset {related.name()} not found in graph")

        # Cache the graph and report statistics
        self._synset_graph = g
        
        if self.verbosity >= 1:
            print(f"[build_synset_graph] Graph construction complete:")
            print(f"  - Nodes: {g.number_of_nodes()}")
            print(f"  - Edges: {g.number_of_edges()}")
            
        if self.verbosity >= 2:
            print(f"[build_synset_graph] Relation type breakdown:")
            for rel_type, count in sorted(relation_counts.items()):
                print(f"  - {rel_type}: {count} edges")
                
        return g

    @staticmethod
    def show_path(label, path):
        """Pretty print a path of synsets."""
        if path:
            print(f"{label}:")
            # Handle both synset objects and name strings
            path_str = []
            for s in path:
                if hasattr(s, 'name'):
                    path_str.append(f"{s.name()} ({s.definition()})")
                else:
                    path_str.append(str(s))
            print(" -> ".join(path_str))
            print(f"Path length: {len(path)}")
            print()
        else:
            print(f"{label}: No path found")
            print()

    @staticmethod
    def show_connected_paths(subject_path, object_path, predicate):
        """Display the connected paths with their shared predicate."""
        if subject_path and object_path and predicate:
            print("=" * 70)
            pred_name = predicate.name() if hasattr(predicate, 'name') else str(predicate)
            print(f"CONNECTED PATH through predicate: {pred_name}")
            print("=" * 70)

            SemanticDecomposer.show_path("Subject -> Predicate path", subject_path)
            SemanticDecomposer.show_path("Predicate -> Object path", object_path)

            # Show the complete connected path
            complete_path = subject_path + object_path[1:]  # Avoid duplicating the predicate
            print("Complete connected path:")
            path_names = []
            for s in complete_path:
                if hasattr(s, 'name'):
                    path_names.append(s.name())
                else:
                    path_names.append(str(s))
            print(" -> ".join(path_names))
            print(f"Total path length: {len(complete_path)}")
            print()
        else:
            print("No connected path found through any predicate synset.")