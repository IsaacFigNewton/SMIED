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
    """
    
    def __init__(self, wn_module, nlp_func, embedding_model=None):
        """
        Initialize the SemanticDecomposer with required dependencies.
        
        Args:
            wn_module: WordNet module (e.g., nltk.corpus.wordnet)
            nlp_func: spaCy NLP function for text processing
            embedding_model: Optional embedding model for similarity computations
        """
        self.wn_module = wn_module
        self.nlp_func = nlp_func
        self.embedding_model = embedding_model
        
        # Initialize component classes
        self.embedding_helper = EmbeddingHelper()
        self.beam_builder = BeamBuilder(self.embedding_helper)
        self.gloss_parser = GlossParser(nlp_func=nlp_func)
        
        # Cached graph
        self._synset_graph = None

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
        # Use provided model or fallback to instance model
        if model is None:
            model = self.embedding_model

        # Create the beam function if we have a model
        get_new_beams_fn = None
        if model is not None and g is not None:
            get_new_beams_fn = lambda graph, src, tgt: self.embedding_helper.get_new_beams_from_embeddings(
                graph, src, tgt, self.wn_module, model, beam_width=beam_width
            )

        # Build the graph if not provided
        if g is None:
            g = self.build_synset_graph()

        # Get synsets for each word
        subject_synsets = self.wn_module.synsets(subject_word, pos=self.wn_module.NOUN)
        predicate_synsets = self.wn_module.synsets(predicate_word, pos=self.wn_module.VERB)
        object_synsets = self.wn_module.synsets(object_word, pos=self.wn_module.NOUN)

        best_combined_path_length = float('inf')
        best_subject_path = None
        best_object_path = None
        best_predicate = None

        # Try each predicate synset as the connector
        for pred in predicate_synsets:
            # Find paths from all subjects to this specific predicate
            subject_paths = self._find_subject_to_predicate_paths(
                subject_synsets, pred, g, get_new_beams_fn, beam_width, max_depth, 
                relax_beam, max_results_per_pair, len_tolerance, max_sample_size
            )

            # Find paths from this specific predicate to all objects  
            object_paths = self._find_predicate_to_object_paths(
                pred, object_synsets, g, get_new_beams_fn, beam_width, max_depth,
                relax_beam, max_results_per_pair, len_tolerance, max_sample_size
            )

            # If we have both paths through this predicate, check if it's the best
            if subject_paths and object_paths:
                valid_pairs = []
                for subj_path in subject_paths:
                    for obj_path in object_paths:
                        if len(set(subj_path).intersection(set(obj_path))) <= max_self_intersection:
                            valid_pairs.append((
                                subj_path,
                                obj_path,
                                len(subj_path) + len(obj_path) - 1  # subtract 1 to avoid counting predicate twice
                            ))

                if valid_pairs:
                    shortest_comb_path = min(valid_pairs, key=lambda x: x[2])
                    if shortest_comb_path[2] < best_combined_path_length:
                        best_combined_path_length = shortest_comb_path[2]
                        best_subject_path = shortest_comb_path[0]
                        best_object_path = shortest_comb_path[1]
                        best_predicate = pred

        return best_subject_path, best_object_path, best_predicate

    def _find_subject_to_predicate_paths(self, subject_synsets, predicate_synset, g, get_new_beams_fn, 
                                       beam_width, max_depth, relax_beam, max_results_per_pair, 
                                       len_tolerance, max_sample_size):
        """Find paths from subject synsets to predicate using gloss parsing strategies."""
        subject_paths = []
        
        # Strategy 1: Look for active subjects in predicate's gloss
        pred_gloss_parsed = self.gloss_parser.parse_gloss(predicate_synset.definition())
        if pred_gloss_parsed and pred_gloss_parsed.get('subjects'):
            # Get subject synsets from parsed gloss
            matched_synsets = self._get_best_synset_matches(
                [pred_gloss_parsed['subjects']], subject_synsets
            )
            
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
        
        # Strategy 2: Look for verbs in subject glosses
        for subj_synset in subject_synsets:
            subj_gloss_parsed = self.gloss_parser.parse_gloss(subj_synset.definition())
            if subj_gloss_parsed and subj_gloss_parsed.get('predicates'):
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
        
        # Strategy 3: Explore hypernyms if no direct paths found
        if not subject_paths:
            subject_paths.extend(self._explore_hypernym_paths(
                subject_synsets, [predicate_synset], g, get_new_beams_fn,
                beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
            ))
            
        return subject_paths

    def _find_predicate_to_object_paths(self, predicate_synset, object_synsets, g, get_new_beams_fn,
                                      beam_width, max_depth, relax_beam, max_results_per_pair,
                                      len_tolerance, max_sample_size):
        """Find paths from predicate to object synsets using gloss parsing strategies."""
        object_paths = []
        
        # Strategy 1: Look for objects (including passive subjects) in predicate's gloss
        pred_gloss_parsed = self.gloss_parser.parse_gloss(predicate_synset.definition())
        if pred_gloss_parsed and pred_gloss_parsed.get('objects'):
            # Get object synsets from parsed gloss
            matched_synsets = self._get_best_synset_matches(
                [pred_gloss_parsed['objects']], object_synsets
            )
            
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
        
        # Strategy 2: Look for verbs in object glosses (including instrumental verbs)
        for obj_synset in object_synsets:
            obj_gloss_parsed = self.gloss_parser.parse_gloss(obj_synset.definition())
            if obj_gloss_parsed and obj_gloss_parsed.get('predicates'):
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
        
        # Strategy 3: Explore hypernyms if no direct paths found
        if not object_paths:
            object_paths.extend(self._explore_hypernym_paths(
                [predicate_synset], object_synsets, g, get_new_beams_fn,
                beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
            ))
            
        return object_paths

    def _get_best_synset_matches(self, candidate_lists, target_synsets, top_k=3):
        """Get the best matching synsets from candidates based on similarity to targets."""
        matches = []
        
        for candidates in candidate_lists:
            if not candidates:
                continue
                
            for candidate in candidates:
                for target in target_synsets:
                    try:
                        # Use path similarity if available, otherwise fall back to simple heuristics
                        similarity = candidate.path_similarity(target) or 0.0
                        matches.append((candidate, similarity))
                    except:
                        # Fallback for synsets that don't support path_similarity
                        matches.append((candidate, 0.1))
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches[:top_k]]

    def _find_path_between_synsets(self, src_synset, tgt_synset, g, get_new_beams_fn,
                                 beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find a path between two synsets using the pathfinder."""
        if src_synset.name() not in g or tgt_synset.name() not in g:
            return None
            
        path_finder = PairwiseBidirectionalAStar(
            g=g,
            src=src_synset.name(),
            tgt=tgt_synset.name(),
            get_new_beams_fn=get_new_beams_fn,
            beam_width=beam_width,
            max_depth=max_depth,
            relax_beam=relax_beam
        )
        
        paths = path_finder.find_paths(max_results=max_results_per_pair, len_tolerance=len_tolerance)
        if paths:
            # Convert first path back to synset objects
            path_names, _ = paths[0]
            return [self.wn_module.synset(name) for name in path_names]
        
        return None

    def _explore_hypernym_paths(self, src_synsets, tgt_synsets, g, get_new_beams_fn,
                              beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Explore paths through hypernyms when direct connections aren't found."""
        paths = []
        
        for src_synset in src_synsets:
            src_hypernyms = src_synset.hypernyms()
            
            for tgt_synset in tgt_synsets:
                tgt_hypernyms = tgt_synset.hypernyms()
                
                # Try connecting through hypernyms
                for src_hyp in src_hypernyms[:3]:  # Limit to top 3 hypernyms
                    for tgt_hyp in tgt_hypernyms[:3]:
                        path = self._find_path_between_synsets(
                            src_hyp, tgt_hyp, g, get_new_beams_fn,
                            beam_width, max_depth-2, relax_beam, max_results_per_pair, len_tolerance
                        )
                        if path:
                            # Create complete path: src -> src_hyp -> ... -> tgt_hyp -> tgt
                            complete_path = [src_synset] + path + [tgt_synset]
                            paths.append(complete_path)
        
        return paths


    def build_synset_graph(self) -> nx.DiGraph:
        """
        Build a directed graph of synsets with their lexical relations.
        Uses caching to avoid rebuilding the graph multiple times.
        """
        if self._synset_graph is not None:
            return self._synset_graph
            
        g = nx.DiGraph()

        # Get all synsets (you may want to limit this for performance)
        all_synsets = list(self.wn_module.all_synsets())

        # Add nodes
        for synset in all_synsets:
            g.add_node(synset.name())

        # Add edges based on relations
        for synset in all_synsets:
            synset_name = synset.name()

            # Add various relation types as edges
            relations = {
                'hypernyms': synset.hypernyms(),
                'hyponyms': synset.hyponyms(),
                'holonyms': synset.holonyms(),
                'meronyms': synset.meronyms(),
                'similar_tos': synset.similar_tos() if hasattr(synset, 'similar_tos') else [],
                'also_sees': synset.also_sees() if hasattr(synset, 'also_sees') else [],
                'verb_groups': synset.verb_groups() if hasattr(synset, 'verb_groups') else [],
                'entailments': synset.entailments() if hasattr(synset, 'entailments') else [],
                'causes': synset.causes() if hasattr(synset, 'causes') else [],
            }

            for rel_type, related_synsets in relations.items():
                for related in related_synsets:
                    if related.name() in g:
                        g.add_edge(synset_name, related.name(),
                                  relation=rel_type, weight=1.0)

        self._synset_graph = g
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