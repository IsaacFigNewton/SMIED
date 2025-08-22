import networkx as nx
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import nltk.corpus.wordnet as wn
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
        self.gloss_parser = GlossParser()
        
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
        relax_beam: bool = False
    ):
        """
        Main entry point for finding semantic paths between subject-predicate-object triples.
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
            # Parse the predicate gloss to get subject and object synsets
            pred_gloss = self.gloss_parser.parse_gloss(pred.definition())
            if not pred_gloss:
                continue
            # Get subject and object synsets from the predicate gloss
            matched_subj_pred_synsets = pred_gloss.get('subject', [])
            matched_pred_obj_synsets = pred_gloss.get('object', [])

            # Find paths from all subjects to this specific predicate
            subject_paths = []
            for subj in subject_synsets:
                # if there are matched_pred_subj_synsets,
                #   try pathing from subject to one of them
                if matched_subj_pred_synsets:
                    # todo: implement pathing from subj
                    #   to one of the matched subj synsets (matched_subj_pred_synsets) to pred
                    path_finder = PairwiseBidirectionalAStar(
                        g=g,
                        src=subj.name(),
                        tgt=matched_subj_pred_synsets.name(),
                        get_new_beams_fn=get_new_beams_fn,
                        beam_width=beam_width,
                        max_depth=max_depth,
                        relax_beam=relax_beam
                    )
                    paths = path_finder.find_paths(max_results=max_results_per_pair, len_tolerance=len_tolerance)
                    if paths:
                        # Convert to synset objects for backward compatibility
                        for path, cost in paths:
                            synset_path = [self.wn_module.synset(name) for name in path]
                            subject_paths.append(synset_path)
                # otherwise, try parsing the subject gloss
                #   to find any verbs with suitable synsets to connect to the predicate
                else:
                    subj_gloss = self.gloss_parser.parse_gloss(subj.definition())
                    if not subj_gloss:
                        continue
                    matched_subj_pred_synsets = subj_gloss.get('predicate', [])
                    # If we have found a suitable predicate synset in the subject gloss,
                    #   we can use it to find paths
                    #   so try pathing from subject to one of them
                    if matched_subj_pred_synsets:
                        # todo: implement pathing from subject gloss's
                        #   matched pred synsets (matched_subj_pred_synsets) to pred
                        path_finder = PairwiseBidirectionalAStar(
                            g=g,
                            src=matched_subj_pred_synsets.name(),
                            tgt=pred.name(),
                            get_new_beams_fn=get_new_beams_fn,
                            beam_width=beam_width,
                            max_depth=max_depth,
                            relax_beam=relax_beam
                        )
                        paths = path_finder.find_paths(max_results=max_results_per_pair, len_tolerance=len_tolerance)
                        if paths:
                            # Convert to synset objects for backward compatibility
                            for path, cost in paths:
                                synset_path = [self.wn_module.synset(name) for name in path]
                                subject_paths.append(synset_path)
                    # If we still don't have any matched synsets,
                    else:
                        # retry process on predicate and subject hypernyms
                        pred_hypernyms = pred.hypernyms()
                        subj_hypernyms = subj.hypernyms()
                        if pred_hypernyms and subj_hypernyms:
                            pass


            # Find paths from this specific predicate to all objects
            #   similar to how we did for subjects, but src and tgt are reversed
            object_paths = []
            for obj in object_synsets:
                # if there are matched_pred_subj_synsets,
                #   try pathing from subject to one of them
                if matched_pred_obj_synsets:
                    # todo: implement pathing from pred
                    #   to one of the matched obj synsets (matched_pred_obj_synsets) to pred
                    path_finder = PairwiseBidirectionalAStar(
                        g=g,
                        src=matched_pred_obj_synsets.name(),
                        tgt=obj.name(),
                        get_new_beams_fn=get_new_beams_fn,
                        beam_width=beam_width,
                        max_depth=max_depth,
                        relax_beam=relax_beam
                    )
                    paths = path_finder.find_paths(max_results=max_results_per_pair, len_tolerance=len_tolerance)
                    if paths:
                        # Convert to synset objects for backward compatibility
                        for path, cost in paths:
                            synset_path = [self.wn_module.synset(name) for name in path]
                            object_paths.append(synset_path)
                # otherwise, try parsing the object gloss
                #   to find any verbs with suitable synsets to connect to the predicate
                else:
                    subj_gloss = self.gloss_parser.parse_gloss(obj.definition())
                    if not subj_gloss:
                        continue
                    matched_pred_obj_synsets = subj_gloss.get('predicate', [])
                    # If we have found a suitable predicate synset in the subject gloss,
                    #   we can use it to find paths
                    #   so try pathing from subject to one of them
                    if matched_pred_obj_synsets:
                        # todo: implement pathing from subject gloss's
                        #   matched pred synsets (matched_pred_obj_synsets) to pred
                        path_finder = PairwiseBidirectionalAStar(
                            g=g,
                            src=matched_pred_obj_synsets.name(),
                            tgt=pred.name(),
                            get_new_beams_fn=get_new_beams_fn,
                            beam_width=beam_width,
                            max_depth=max_depth,
                            relax_beam=relax_beam
                        )
                        paths = path_finder.find_paths(max_results=max_results_per_pair, len_tolerance=len_tolerance)
                        if paths:
                            # Convert to synset objects for backward compatibility
                            for path, cost in paths:
                                synset_path = [self.wn_module.synset(name) for name in path]
                                object_paths.append(synset_path)
                    # If we still don't have any matched synsets,
                    else:
                        # retry process on predicate and subject hypernyms
                        pred_hypernyms = pred.hypernyms()
                        obj_hypernyms = obj.hypernyms()
                        if pred_hypernyms and obj_hypernyms:
                            pass

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