import networkx as nx
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet as wn
from typing import Optional, Callable, Dict, List, Tuple, Any, Set
from .PairwiseBidirectionalAStar import PairwiseBidirectionalAStar
from .BeamBuilder import BeamBuilder
from .EmbeddingHelper import EmbeddingHelper
from .FramenetSpacySRL import FrameNetSpaCySRL


class SemanticDecomposer:
    """
    Main orchestrator class for semantic decomposition using WordNet paths.
    
    This class integrates FrameNet SRL and WordNet to find semantic paths 
    between subject-predicate-object triples.
    """
    
    def __init__(self, wn_module, nlp_func, embedding_model=None):
        """
        Initialize the SemanticDecomposer with required dependencies.
        
        Args:
            wn_module: WordNet module (e.g., nltk.corpus.wordnet)
            nlp_func: spaCy NLP function for text processing
            embedding_model: Optional embedding model for similarity computations
        """
        # Store core dependencies
        self.wn_module = wn_module
        self.nlp_func = nlp_func
        self.embedding_model = embedding_model
        
        # Initialize component classes
        self.embedding_helper = EmbeddingHelper()
        self.beam_builder = BeamBuilder(self.embedding_helper)
        
        # Initialize FrameNetSpaCySRL
        if nlp_func is not None:
            self.framenet_srl = FrameNetSpaCySRL(nlp=nlp_func, min_confidence=0.2)
        else:
            self.framenet_srl = FrameNetSpaCySRL(nlp=None, min_confidence=0.2)
        
        # Cached graph for performance optimization
        self._synset_graph = None

    def find_connected_shortest_paths(
        self,
        subject_word: str,
        predicate_word: str,
        object_word: str,
        model=None,  # embedding model
        g: nx.DiGraph|None = None,  # synset graph
        max_depth: int = 10,
        max_self_intersection: int = 5,
        beam_width: int = 3,
        max_results_per_pair: int = 3,
        len_tolerance: int = 1,
        relax_beam: bool = False,
        max_sample_size: int = 5
    ) -> List[str]:
        """
        Main entry point for finding semantic paths between subject-predicate-object triples.
        Returns the best connected path between the subject, predicate, and object.
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
        
        # Check if we found any synsets
        if not subject_synsets or not predicate_synsets or not object_synsets:
            return []

        # Collect all possible interpretations, then select most coherent
        all_interpretation_paths = []
        
        # Try each predicate synset
        for pred in predicate_synsets:
            # Get paths for frame interpretations
            subject_paths = self._find_subject_to_predicate_paths(
                subject_synsets, pred, g, get_new_beams_fn, beam_width, max_depth, 
                relax_beam, max_results_per_pair, len_tolerance, max_sample_size
            )
            
            object_paths = self._find_predicate_to_object_paths(
                pred, object_synsets, g, get_new_beams_fn, beam_width, max_depth,
                relax_beam, max_results_per_pair, len_tolerance, max_sample_size
            )

            # Score valid combinations
            for subj_path in subject_paths:
                for obj_path in object_paths:
                    # Check self-intersection constraint
                    intersection_count = len(set(subj_path).intersection(set(obj_path)))
                    if intersection_count <= max_self_intersection:
                        combined_length = len(subj_path) + len(obj_path) - 1
                        all_interpretation_paths.append((
                            subj_path, obj_path, pred, combined_length
                        ))

        # Select best interpretation by path length
        if all_interpretation_paths:
            best = min(all_interpretation_paths, key=lambda x: x[3])  # Select shortest path
            best_subject_path, best_object_path, best_predicate, _ = best
            return best_subject_path + [best_predicate] + best_object_path
        else:
            # Try fallback strategy
            fallback_result = self._try_fallback_strategy(
                subject_synsets, predicate_synsets, object_synsets, 
                subject_word, predicate_word, object_word
            )
            return fallback_result if fallback_result else []

    def _try_fallback_strategy(self, subject_synsets, predicate_synsets, object_synsets,
                             subject_word, predicate_word, object_word) -> List[str]:
        """
        Fallback strategy: Find semantic paths through WordNet relationships.
        """
        # Try the most common synsets first
        for predicate_synset in predicate_synsets[:2]:  # Try top 2 predicate senses
            for subject_synset in subject_synsets[:3]:  # Try top 3 subject senses
                for object_synset in object_synsets[:3]:  # Try top 3 object senses
                    try:
                        # Find paths through semantic relationships
                        subject_path = self._find_semantic_path(
                            subject_synset, predicate_synset, max_depth=4
                        )
                        object_path = self._find_semantic_path(
                            predicate_synset, object_synset, max_depth=4
                        )
                        
                        if subject_path and object_path:
                            return subject_path + [predicate_synset] + object_path 
                    except Exception:
                        continue
        
        return []

    def _find_semantic_path(self, start_synset, end_synset, max_depth=4) -> List[str]:
        """
        Find a semantic path between two synsets using WordNet relationships.
        """
        if start_synset == end_synset:
            return [start_synset.name()]
            
        # BFS to find shortest semantic path
        from collections import deque
        
        queue = deque([(start_synset, [start_synset])])
        visited = {start_synset}
        
        while queue and len(queue[0][1]) < max_depth:
            current_synset, path = queue.popleft()
            
            # Get all semantically related synsets
            related_synsets = self._get_related_synsets(current_synset)
            
            for related_synset in related_synsets:
                if related_synset == end_synset:
                    # Found path to target
                    return path + [related_synset]
                    
                if related_synset not in visited:
                    visited.add(related_synset)
                    new_path = path + [related_synset]
                    if len(new_path) <= max_depth:
                        queue.append((related_synset, new_path))
        
        # If no path found through direct semantic relationships, try hypernym path
        try:
            similarity = start_synset.path_similarity(end_synset)
            if similarity and similarity > 0.05:  # Threshold for meaningful connection
                # Find a path through their lowest common hypernym
                path = self._find_hypernym_path(start_synset, end_synset)
                if path and len(path) <= max_depth:
                    return path
        except:
            pass
            
        return []
    
    def _get_related_synsets(self, synset):
        """
        Get all semantically related synsets for path finding.
        """
        related = set()
        
        # Hypernyms (more general concepts)
        related.update(synset.hypernyms())
        
        # Hyponyms (more specific concepts) - limit to avoid explosion
        hyponyms = synset.hyponyms()
        related.update(hyponyms[:5])  # Limit to top 5 to control search space
        
        # Meronyms (part-of relationships)
        related.update(synset.part_meronyms())
        related.update(synset.member_meronyms())
        related.update(synset.substance_meronyms())
        
        # Holonyms (whole-of relationships)  
        related.update(synset.part_holonyms())
        related.update(synset.member_holonyms())
        related.update(synset.substance_holonyms())
        
        # For verbs, include entailments and causes
        if synset.pos() == 'v':
            related.update(synset.entailments())
            related.update(synset.causes())
            related.update(synset.verb_groups())
        
        # For adjectives, include similar_tos
        elif synset.pos() in ['a', 's']:
            related.update(synset.similar_tos())
        
        return list(related)
    
    def _find_hypernym_path(self, start_synset, end_synset) -> List[str]:
        """
        Find path between synsets through their hypernym hierarchy.
        """
        # Get hypernym paths for both synsets
        start_hypernyms = self._get_hypernym_chain(start_synset, max_depth=6)
        end_hypernyms = self._get_hypernym_chain(end_synset, max_depth=6)
        
        # Find common ancestors
        start_set = set(start_hypernyms)
        end_set = set(end_hypernyms)
        common_ancestors = start_set.intersection(end_set)
        
        if common_ancestors:
            # Find the most specific (lowest) common ancestor
            best_ancestor = None
            min_depth = float('inf')
            
            for ancestor in common_ancestors:
                start_depth = start_hypernyms.index(ancestor) if ancestor in start_hypernyms else float('inf')
                end_depth = end_hypernyms.index(ancestor) if ancestor in end_hypernyms else float('inf')
                total_depth = start_depth + end_depth
                
                if total_depth < min_depth:
                    min_depth = total_depth
                    best_ancestor = ancestor
            
            if best_ancestor:
                # Build path: start -> ... -> ancestor -> ... -> end
                start_to_ancestor = start_hypernyms[:start_hypernyms.index(best_ancestor) + 1]
                end_to_ancestor = end_hypernyms[:end_hypernyms.index(best_ancestor)]
                end_to_ancestor.reverse()  # Reverse to go from ancestor to end
                
                # Combine paths, avoiding duplicate ancestor
                full_path = start_to_ancestor + end_to_ancestor
                return full_path if len(full_path) <= 6 else []
        
        return []
    
    def _get_hypernym_chain(self, synset, max_depth=6):
        """Get the hypernym chain for a synset up to max_depth."""
        chain = [synset]
        current = synset
        
        for _ in range(max_depth):
            hypernyms = current.hypernyms()
            if not hypernyms:
                break
            current = hypernyms[0]  # Take the first (most common) hypernym
            chain.append(current)
            
        return chain
    
    def _find_subject_to_predicate_paths(self, subject_synsets, predicate_synset, g, get_new_beams_fn, 
                                       beam_width, max_depth, relax_beam, max_results_per_pair, 
                                       len_tolerance, max_sample_size):
        """Find paths from subject synsets to predicate using cascading strategies."""
        subject_paths = []
        
        # Strategy 1: FrameNet-based connections
        framenet_paths = self._find_framenet_subject_predicate_paths(
            subject_synsets, predicate_synset, g, get_new_beams_fn, 
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        subject_paths.extend(framenet_paths)
        
        # Strategy 2: Derivational relations
        derivational_paths = self._find_derivational_subject_predicate_paths(
            subject_synsets, predicate_synset, g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        subject_paths.extend(derivational_paths)
        
        # Strategy 3: Hypernym/hyponym relationships
        hypernym_paths = self._explore_hypernym_paths(
            subject_synsets, [predicate_synset], g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        subject_paths.extend(hypernym_paths)
        
        return subject_paths

    def _find_framenet_subject_predicate_paths(self, subject_synsets, predicate_synset, g, get_new_beams_fn,
                                             beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find paths from subject synsets to predicate using FrameNet semantic frame connections."""
        framenet_paths = []
        
        try:
            # Get frames for predicate using lexical unit cache
            pred_lemma = predicate_synset.lemmas()[0].name().lower()
            key = (pred_lemma, 'v')
            frame_names = []
            if key in self.framenet_srl.lexical_unit_cache:
                frame_names = self.framenet_srl.lexical_unit_cache[key]
            frame_scores = [(name, 1.0) for name in frame_names]
            
            # Try each frame interpretation
            for frame_name, _ in frame_scores:
                try:
                    if frame_name in self.framenet_srl.frame_cache:
                        frame_paths = self._find_paths_via_frame(
                            subject_synsets, predicate_synset, frame_name, 
                            g, get_new_beams_fn, beam_width, max_depth, relax_beam, 
                            max_results_per_pair, len_tolerance
                        )
                        framenet_paths.extend(frame_paths)
                except:
                    continue
        except Exception:
            pass
        
        return framenet_paths

    def _find_paths_via_frame(self, subject_synsets, predicate_synset, frame_name, 
                             g, get_new_beams_fn, beam_width, max_depth, relax_beam, 
                             max_results_per_pair, len_tolerance):
        """Find paths from subject synsets to predicate via a specific semantic frame."""
        frame_paths = []
        
        if frame_name not in self.framenet_srl.frame_cache:
            return frame_paths
            
        frame = self.framenet_srl.frame_cache[frame_name]
        
        # Get frame elements that typically correspond to subjects
        subject_fe_to_synset = {
            "Agent": "agent.n.01",
            "Experiencer": "experiencer.n.01",
            "Cognizer": "thinker.n.01",
            "Speaker": "speaker.n.01",
            "Actor": "actor.n.01",
            "Performer": "performer.n.01",
            "Causer": "cause.n.01",
            "Entity": "entity.n.01",
            "Theme": "theme.n.03",
        }

        # For each subject synset, try to connect through frame elements
        for subj_synset in subject_synsets:
            for fe_name, fe_synset in subject_fe_to_synset.items():
                if fe_name in frame.FE:
                    if g is not None and subj_synset.name() in g and fe_synset in g:
                        path = self._find_path_between_synsets(
                            subj_synset, wn.synset(fe_synset), g, get_new_beams_fn,
                            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
                        )
                        if path:                            
                            frame_paths.append(path)
        
        return frame_paths

    def _find_derivational_subject_predicate_paths(self, subject_synsets, predicate_synset, g, get_new_beams_fn,
                                                 beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find paths from subject synsets to predicate using WordNet derivational relations."""
        derivational_paths = []
        
        # Get derivationally related forms for the predicate
        pred_derivational = self._get_derivational_connections(predicate_synset)
        
        # For each subject synset
        for subj_synset in subject_synsets:
            # Get derivationally related forms for the subject
            subj_derivational = self._get_derivational_connections(subj_synset)
            
            # Try direct connections through derivational relations
            for subj_deriv in subj_derivational:
                if g is not None and subj_deriv.name() in g and predicate_synset.name() in g:
                    path = self._find_path_between_synsets(
                        subj_deriv, predicate_synset, g, get_new_beams_fn,
                        beam_width, max_depth-1, relax_beam, max_results_per_pair, len_tolerance
                    )
                    if path:
                        # Prepend original subject
                        complete_path = [subj_synset] + path
                        derivational_paths.append(complete_path)
            
            # Try cross connections (subject derivational to predicate derivational)
            for subj_deriv in subj_derivational:
                for pred_deriv in pred_derivational:
                    if g is not None and subj_deriv.name() in g and pred_deriv.name() in g:
                        path = self._find_path_between_synsets(
                            subj_deriv, pred_deriv, g, get_new_beams_fn,
                            beam_width, max_depth-2, relax_beam, max_results_per_pair, len_tolerance
                        )
                        if path:
                            # Build complete path: subject -> subj_deriv -> ... -> pred_deriv -> predicate
                            complete_path = [subj_synset] + path + [predicate_synset]
                            derivational_paths.append(complete_path)
        
        return derivational_paths

    def _get_derivational_connections(self, synset) -> List:
        """Get derivationally related synsets using WordNet's derivationally_related_forms()."""
        derivational_synsets = []
        
        # Get all lemmas for this synset
        for lemma in synset.lemmas():
            # Get derivationally related forms
            for related_lemma in lemma.derivationally_related_forms():
                related_synset = related_lemma.synset()
                if related_synset != synset:  # Avoid self-loops
                    derivational_synsets.append(related_synset)
        
        return derivational_synsets

    def _find_predicate_to_object_paths(self, predicate_synset, object_synsets, g, get_new_beams_fn,
                                      beam_width, max_depth, relax_beam, max_results_per_pair,
                                      len_tolerance, max_sample_size):
        """Find paths from predicate to object synsets using cascading strategies."""
        object_paths = []
        
        # Strategy 1: FrameNet-based connections
        framenet_paths = self._find_framenet_predicate_object_paths(
            predicate_synset, object_synsets, g, get_new_beams_fn, 
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        object_paths.extend(framenet_paths)
        
        # Strategy 2: Derivational relations
        derivational_paths = self._find_derivational_predicate_object_paths(
            predicate_synset, object_synsets, g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        object_paths.extend(derivational_paths)
        
        # Strategy 3: Hypernym/hyponym relationships
        hypernym_paths = self._explore_hypernym_paths(
            [predicate_synset], object_synsets, g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        object_paths.extend(hypernym_paths)
        
        return object_paths

    def _find_framenet_predicate_object_paths(self, predicate_synset, object_synsets, g, get_new_beams_fn,
                                            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find paths from predicate to object synsets using FrameNet semantic frame connections."""
        framenet_paths = []
        
        try:
            # Get frames for predicate using lexical unit cache
            pred_lemma = predicate_synset.lemmas()[0].name().lower()
            key = (pred_lemma, 'v')
            frame_names = []
            if key in self.framenet_srl.lexical_unit_cache:
                frame_names = self.framenet_srl.lexical_unit_cache[key]
            frame_scores = [(name, 1.0) for name in frame_names]
            
            # Try each frame interpretation
            for frame_name, _ in frame_scores:
                try:
                    if frame_name in self.framenet_srl.frame_cache:
                        frame_paths = self._find_paths_via_frame_to_objects(
                            predicate_synset, object_synsets, frame_name, 
                            g, get_new_beams_fn, beam_width, max_depth, relax_beam, 
                            max_results_per_pair, len_tolerance
                        )
                        framenet_paths.extend(frame_paths)
                except:
                    continue
        except Exception:
            pass
        
        return framenet_paths

    def _find_paths_via_frame_to_objects(self, predicate_synset, object_synsets, frame_name, 
                                        g, get_new_beams_fn, beam_width, max_depth, relax_beam, 
                                        max_results_per_pair, len_tolerance):
        """Find paths from predicate to object synsets via a specific semantic frame."""
        frame_paths = []
        
        if frame_name not in self.framenet_srl.frame_cache:
            return frame_paths
            
        frame = self.framenet_srl.frame_cache[frame_name]
        
        # Get frame elements that typically correspond to objects
        object_fe_names = {
            'Theme', 'Patient', 'Goal', 'Recipient', 'Beneficiary', 'Content', 'Message',
            'Stimulus', 'Topic', 'Undergoer', 'Item', 'Entity', 'Target'
        }
        
        # For each object synset, try to connect through frame elements
        for obj_synset in object_synsets:
            for fe_name in object_fe_names:
                if fe_name in frame.FE:
                    if g is not None and predicate_synset.name() in g and obj_synset.name() in g:
                        path = self._find_path_between_synsets(
                            predicate_synset, obj_synset, g, get_new_beams_fn,
                            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
                        )
                        if path:
                            frame_paths.append(path)
        
        return frame_paths

    def _find_derivational_predicate_object_paths(self, predicate_synset, object_synsets, g, get_new_beams_fn,
                                                beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find paths from predicate to object synsets using WordNet derivational relations."""
        derivational_paths = []
        
        # Get derivationally related forms for the predicate
        pred_derivational = self._get_derivational_connections(predicate_synset)
        
        # For each object synset
        for obj_synset in object_synsets:
            # Get derivationally related forms for the object
            obj_derivational = self._get_derivational_connections(obj_synset)
            
            # Try direct connections through derivational relations
            for obj_deriv in obj_derivational:
                if g is not None and predicate_synset.name() in g and obj_deriv.name() in g:
                    path = self._find_path_between_synsets(
                        predicate_synset, obj_deriv, g, get_new_beams_fn,
                        beam_width, max_depth-1, relax_beam, max_results_per_pair, len_tolerance
                    )
                    if path:
                        # Extend path to include original object
                        complete_path = path + [obj_synset]
                        derivational_paths.append(complete_path)
            
            # Try cross connections (predicate derivational to object derivational)
            for pred_deriv in pred_derivational:
                for obj_deriv in obj_derivational:
                    if g is not None and pred_deriv.name() in g and obj_deriv.name() in g:
                        path = self._find_path_between_synsets(
                            pred_deriv, obj_deriv, g, get_new_beams_fn,
                            beam_width, max_depth-2, relax_beam, max_results_per_pair, len_tolerance
                        )
                        if path:
                            # Build complete path: predicate -> pred_deriv -> ... -> obj_deriv -> object
                            complete_path = [predicate_synset] + path + [obj_synset]
                            derivational_paths.append(complete_path)
        
        return derivational_paths

    def _find_path_between_synsets(self, src_synset, tgt_synset, g, get_new_beams_fn,
                                 beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find a path between two synsets using the pathfinder."""        
        # If no graph is provided, we can't find paths
        if g is None or src_synset.name() not in g or tgt_synset.name() not in g:
            return None
            
        # Create pathfinder instance
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
            return synset_path
        
        return None

    def _explore_hypernym_paths(self, src_synsets, tgt_synsets, g, get_new_beams_fn,
                              beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Explore paths through hypernyms when direct connections aren't found."""
        paths = []
        
        for src_synset in src_synsets:
            src_hypernyms = src_synset.hypernyms()
            
            for tgt_synset in tgt_synsets:
                tgt_hypernyms = tgt_synset.hypernyms()
                
                # Try connecting through hypernyms (limit to top 3 for performance)
                for src_hyp in src_hypernyms[:3]:
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

        # Get all synsets
        all_synsets = list(self.wn_module.all_synsets())

        # Add nodes to the graph
        for synset in all_synsets:
            g.add_node(synset.name())

        # Add edges based on lexical relations
        edge_count = 0
        relation_counts = {}
        
        for synset in all_synsets:
            synset_name = synset.name()

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
                'attributes': synset.attributes() if hasattr(synset, 'attributes') else [],
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
            
            # Add lemma-level antonym relations
            for lemma in synset.lemmas():
                antonyms = lemma.antonyms() if hasattr(lemma, 'antonyms') else []
                if antonyms:
                    if 'antonyms' not in relation_counts:
                        relation_counts['antonyms'] = 0
                    
                    for antonym_lemma in antonyms:
                        antonym_synset_name = antonym_lemma.synset().name()
                        if antonym_synset_name in g:
                            g.add_edge(synset_name, antonym_synset_name,
                                      relation='antonyms', weight=1.5)
                            edge_count += 1
                            relation_counts['antonyms'] += 1
            
            # Add derivational relations
            derivational_synsets = self._get_derivational_connections(synset)
            if derivational_synsets:
                if 'derivational' not in relation_counts:
                    relation_counts['derivational'] = 0
                
                for deriv_synset in derivational_synsets:
                    if deriv_synset.name() in g:
                        g.add_edge(synset_name, deriv_synset.name(),
                                  relation='derivational', weight=0.8)
                        edge_count += 1
                        relation_counts['derivational'] += 1

        # Add FrameNet-based semantic frame connections
        frame_edge_count = self._add_frame_based_edges(g, all_synsets)
        edge_count += frame_edge_count
        relation_counts['framenet'] = frame_edge_count
        
        # Cache the graph
        self._synset_graph = g
        
        return g

    def _add_frame_based_edges(self, graph: nx.DiGraph, all_synsets: List, sample_size: int = 1000) -> int:
        """
        Add edges based on FrameNet semantic frame connections.
        """
        frame_edge_count = 0
        
        # Sample synsets for performance
        import random
        sample_synsets = random.sample(all_synsets, min(sample_size, len(all_synsets)))
        
        # Group synsets by frame participation
        frame_to_synsets = {}
        
        try:
            for synset in sample_synsets:
                # Create text for frame analysis using lemma and definition
                synset_text = f"{synset.lemmas()[0].name()} {synset.definition()}"
                
                try:
                    # Process with FrameNet SRL
                    doc = self.framenet_srl.process_text(synset_text)
                    
                    # For each frame identified for this synset
                    for frame_instance in doc._.frames:
                        frame_name = frame_instance.name
                        
                        if frame_name not in frame_to_synsets:
                            frame_to_synsets[frame_name] = []
                        
                        frame_to_synsets[frame_name].append(synset)
                        
                except Exception:
                    continue
            
            # Create edges between synsets that share frames
            for frame_name, synsets_in_frame in frame_to_synsets.items():
                if len(synsets_in_frame) < 2:
                    continue
                    
                # Create edges between all pairs of synsets in the same frame
                for i, synset1 in enumerate(synsets_in_frame):
                    for synset2 in synsets_in_frame[i+1:]:
                        synset1_name = synset1.name()
                        synset2_name = synset2.name()
                        
                        if synset1_name in graph and synset2_name in graph:
                            # Add bidirectional frame-based edges
                            if not graph.has_edge(synset1_name, synset2_name):
                                graph.add_edge(synset1_name, synset2_name,
                                             relation='framenet', weight=0.9, frame=frame_name)
                                frame_edge_count += 1
                                
                            if not graph.has_edge(synset2_name, synset1_name):
                                graph.add_edge(synset2_name, synset1_name,
                                             relation='framenet', weight=0.9, frame=frame_name)
                                frame_edge_count += 1
        
        except Exception:
            pass
        
        return frame_edge_count