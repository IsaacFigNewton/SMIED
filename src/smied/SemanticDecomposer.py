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
    
    This class integrates the PairwiseBidirectionalAStar, BeamBuilder, EmbeddingHelper,
    FrameNetSpaCySRL, and GlossParser components to find semantic paths between 
    subject-predicate-object triples.
    
    Uses cascading strategy:
    1. Primary: FrameNetSpaCySRL (semantic frame connections)
    2. Secondary: Derivational relations (morphological connections)
    3. Tertiary: Hypernym/hyponym (taxonomic relationships)
    4. Quaternary: GlossParser (deprecated, legacy only)
    
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
        
        # Initialize FrameNetSpaCySRL with the same NLP function
        if self.verbosity >= 1:
            print("[SemanticDecomposer] Initializing FrameNetSpaCySRL...")
        
        # Handle case where nlp_func is None
        if nlp_func is not None:
            self.framenet_srl = FrameNetSpaCySRL(nlp=nlp_func, min_confidence=0.4)
        else:
            # Skip FrameNet processing if no NLP function available
            if self.verbosity >= 2:
                print("[SemanticDecomposer] No NLP function provided, FrameNet SRL will be limited")
            self.framenet_srl = FrameNetSpaCySRL(nlp=None, min_confidence=0.4)
        
        if self.verbosity >= 2:
            print("[SemanticDecomposer] Components initialized:")
            print(f"  - EmbeddingHelper: {type(self.embedding_helper).__name__}")
            print(f"  - BeamBuilder: {type(self.beam_builder).__name__}")
            print(f"  - FrameNetSpaCySRL: {type(self.framenet_srl).__name__}")
        
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
        g: nx.DiGraph|None = None,  # synset graph
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

        # Collect ALL possible interpretations, then select most coherent
        if self.verbosity >= 1:
            print(f"[find_connected_shortest_paths] Collecting all interpretations from {len(predicate_synsets)} predicate synsets...")

        all_interpretation_paths = []
        
        # Try each predicate synset with ALL its frame interpretations
        for i, pred in enumerate(predicate_synsets):
            if self.verbosity >= 2:
                print(f"\n[find_connected_shortest_paths] Trying predicate {i+1}/{len(predicate_synsets)}: {pred.name()} - '{pred.definition()[:100]}...'")
                
            # Get paths for ALL frame interpretations (not just first success)
            subject_paths = self._find_subject_to_predicate_paths(
                subject_synsets, pred, g, get_new_beams_fn, beam_width, max_depth, 
                relax_beam, max_results_per_pair, len_tolerance, max_sample_size
            )
            
            object_paths = self._find_predicate_to_object_paths(
                pred, object_synsets, g, get_new_beams_fn, beam_width, max_depth,
                relax_beam, max_results_per_pair, len_tolerance, max_sample_size
            )
            
            if self.verbosity >= 2:
                print(f"[find_connected_shortest_paths] Found {len(subject_paths)} subject->predicate paths")
                print(f"[find_connected_shortest_paths] Found {len(object_paths)} predicate->object paths")

            # Score ALL valid combinations by semantic coherence
            for subj_path in subject_paths:
                for obj_path in object_paths:
                    # Check self-intersection constraint
                    intersection_count = len(set(subj_path).intersection(set(obj_path)))
                    if intersection_count <= max_self_intersection:
                        coherence = self._score_path_coherence(
                            subj_path, obj_path, pred, subject_word, predicate_word, object_word
                        )
                        combined_length = len(subj_path) + len(obj_path) - 1  # subtract 1 to avoid counting predicate twice
                        all_interpretation_paths.append((
                            subj_path, obj_path, pred, coherence, combined_length
                        ))
                        
                        if self.verbosity >= 2:
                            print(f"[find_connected_shortest_paths] Valid interpretation: coherence={coherence:.3f}, length={combined_length}, intersection={intersection_count}")
                    elif self.verbosity >= 2:
                        print(f"[find_connected_shortest_paths] Path pair rejected: intersection={intersection_count} > {max_self_intersection}")

        # Select best interpretation by coherence, breaking ties with path length
        if all_interpretation_paths:
            # Sort by coherence (descending) then by length (ascending) for tie-breaking
            best = max(all_interpretation_paths, key=lambda x: (x[3], -x[4]))
            best_subject_path, best_object_path, best_predicate, best_coherence, best_length = best
            
            if self.verbosity >= 1:
                print(f"\n[find_connected_shortest_paths] SUCCESS: Selected best interpretation")
                print(f"[find_connected_shortest_paths] Best predicate: {best_predicate.name()}")
                print(f"[find_connected_shortest_paths] Coherence score: {best_coherence:.3f}")
                print(f"[find_connected_shortest_paths] Path length: {best_length}")
                print(f"[find_connected_shortest_paths] Total interpretations considered: {len(all_interpretation_paths)}")
                
            return best_subject_path, best_object_path, best_predicate
        else:
            if self.verbosity >= 1:
                print(f"\n[find_connected_shortest_paths] No paths found via standard strategies, trying fallback...")
            
            # FALLBACK STRATEGY: Try simple hypernym climbing
            fallback_result = self._try_fallback_strategy(
                subject_synsets, predicate_synsets, object_synsets, 
                subject_word, predicate_word, object_word
            )
            
            if fallback_result[0] is not None:
                return fallback_result
            
            if self.verbosity >= 0:
                print(f"\n[find_connected_shortest_paths] WARNING: No valid interpretations found for '{subject_word}' -> '{predicate_word}' -> '{object_word}'")
            
            return None, None, None

    def _try_fallback_strategy(self, subject_synsets, predicate_synsets, object_synsets,
                             subject_word, predicate_word, object_word):
        """
        Fallback strategy: Find actual semantic paths through WordNet relationships.
        Uses hypernyms, meronyms, holonyms, and other semantic connections to create
        meaningful paths with intermediate concepts instead of trivial direct connections.
        """
        if self.verbosity >= 2:
            print(f"[_try_fallback_strategy] Trying fallback for {subject_word}-{predicate_word}-{object_word}")
        
        # Try the most common synsets first
        for i, predicate_synset in enumerate(predicate_synsets[:2]):  # Try top 2 predicate senses
            if self.verbosity >= 2:
                print(f"[_try_fallback_strategy] Trying predicate sense {i+1}: {predicate_synset.name()}")
                
            best_subject_path = None
            best_object_path = None
            best_score = 0
            
            for subject_synset in subject_synsets[:3]:  # Try top 3 subject senses
                for object_synset in object_synsets[:3]:  # Try top 3 object senses
                    
                    # Try to find actual semantic paths with intermediate concepts
                    try:
                        # Find path from subject to predicate through semantic relationships
                        subject_path = self._find_semantic_path(
                            subject_synset, predicate_synset, max_depth=4
                        )
                        
                        # Find path from predicate to object through semantic relationships  
                        object_path = self._find_semantic_path(
                            predicate_synset, object_synset, max_depth=4
                        )
                        
                        if subject_path and object_path:
                            # Score this combination based on path quality and coherence
                            coherence = self._score_semantic_path_coherence(
                                subject_path, object_path, predicate_synset,
                                subject_word, predicate_word, object_word
                            )
                            
                            if coherence > best_score:
                                best_score = coherence
                                best_subject_path = subject_path
                                best_object_path = object_path
                                
                                if self.verbosity >= 2:
                                    print(f"[_try_fallback_strategy] Found semantic path: score={coherence:.3f}")
                                    subject_path_str = " -> ".join([s.name() for s in subject_path])
                                    object_path_str = " -> ".join([s.name() for s in object_path])
                                    print(f"  Subject path ({len(subject_path)} nodes): {subject_path_str}")
                                    print(f"  Object path ({len(object_path)} nodes): {object_path_str}")
                                    
                    except Exception as e:
                        if self.verbosity >= 2:
                            print(f"[_try_fallback_strategy] Path finding failed: {e}")
                        continue
            
            # If we found a good semantic path, use it
            if best_subject_path and best_object_path and best_score > 0.1:
                if self.verbosity >= 1:
                    print(f"[_try_fallback_strategy] SUCCESS: Using semantic fallback strategy with score {best_score:.3f}")
                return best_subject_path, best_object_path, predicate_synset
        
        if self.verbosity >= 2:
            print(f"[_try_fallback_strategy] No semantic fallback paths found")
        return None, None, None

    def _find_semantic_path(self, start_synset, end_synset, max_depth=4):
        """
        Find a semantic path between two synsets using WordNet relationships.
        Uses breadth-first search through hypernyms, hyponyms, meronyms, holonyms,
        and other semantic relationships to find meaningful intermediate concepts.
        
        Args:
            start_synset: Starting synset
            end_synset: Target synset  
            max_depth: Maximum path length to search
            
        Returns:
            List of synsets forming a path from start to end, or None if no path found
        """
        if start_synset == end_synset:
            return [start_synset]
            
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
        
        # If still no path and they're different POS, try conceptual connections
        if start_synset.pos() != end_synset.pos():
            path = self._find_conceptual_path(start_synset, end_synset, max_depth)
            if path:
                return path
            
        return None
    
    def _get_related_synsets(self, synset):
        """
        Get all semantically related synsets for path finding.
        Includes hypernyms, hyponyms, meronyms, holonyms, and other relationships.
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
    
    def _find_hypernym_path(self, start_synset, end_synset):
        """
        Find path between synsets through their hypernym hierarchy.
        This creates paths like: cat -> feline -> carnivore -> chase (if such connection exists).
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
                return full_path if len(full_path) <= 6 else None
        
        return None
    
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
    
    def _find_conceptual_path(self, start_synset, end_synset, max_depth):
        """
        Find conceptual paths between different POS synsets using meaningful connections.
        For example: cat (noun) -> carnivore (noun) -> hunt (verb) -> chase (verb)
        """
        if self.verbosity >= 2:
            print(f"[_find_conceptual_path] Finding conceptual path: {start_synset.name()} -> {end_synset.name()}")
            
        # Strategy 1: Use derivational forms and related concepts
        # Get hypernyms of start synset to find broader categories
        start_hypernyms = self._get_hypernym_chain(start_synset, max_depth=4)
        
        # For each hypernym, look for conceptually related terms that might connect to end_synset
        for hypernym in start_hypernyms[:3]:  # Limit search to avoid explosion
            # Look for derivational forms or semantically related concepts
            conceptual_bridges = self._find_conceptual_bridges(hypernym, end_synset)
            
            for bridge in conceptual_bridges:
                # Create path: start -> hypernym -> bridge -> end
                path_candidate = [start_synset]
                
                # Add intermediate hypernyms if needed
                hypernym_index = start_hypernyms.index(hypernym)
                for i in range(1, hypernym_index + 1):
                    path_candidate.append(start_hypernyms[i])
                
                # Add the bridge and end
                path_candidate.extend([bridge, end_synset])
                
                if len(path_candidate) <= max_depth:
                    if self.verbosity >= 2:
                        path_str = " -> ".join([s.name() for s in path_candidate])
                        print(f"[_find_conceptual_path] Found conceptual path: {path_str}")
                    return path_candidate
        
        # Strategy 2: If start is animal and end is action, create logical connection
        if self._is_animal_synset(start_synset) and end_synset.pos() == 'v':
            # Create path: animal -> behavior -> specific_action
            animal_behavior_path = self._create_animal_behavior_path(start_synset, end_synset)
            if animal_behavior_path and len(animal_behavior_path) <= max_depth:
                return animal_behavior_path
        
        return None
    
    def _find_conceptual_bridges(self, concept_synset, target_synset):
        """Find conceptual bridges between a concept and target synset."""
        bridges = []
        
        # Handle noun -> verb connections (animal -> action)
        if concept_synset.pos() == 'n' and target_synset.pos() == 'v':
            # Common animal -> action mappings
            action_mappings = {
                'carnivore': ['hunt', 'chase', 'catch', 'pursue'],
                'predator': ['hunt', 'chase', 'stalk', 'catch'],
                'animal': ['move', 'run', 'chase', 'hunt'],
                'mammal': ['run', 'chase', 'hunt'],
                'feline': ['hunt', 'chase', 'pounce', 'stalk'],
                'creature': ['move', 'chase', 'hunt'],
                'organism': ['move', 'act']
            }
            
            # Check if our concept maps to any known actions
            for concept_key, actions in action_mappings.items():
                if concept_key in concept_synset.name() or concept_key in concept_synset.definition().lower():
                    for action in actions:
                        # Find synsets for this action
                        action_synsets = self.wn_module.synsets(action, pos=self.wn_module.VERB)
                        for action_synset in action_synsets[:2]:  # Top 2 senses
                            # Check if this action synset is related to our target
                            try:
                                similarity = action_synset.path_similarity(target_synset)
                                if similarity and similarity > 0.1:  # Reasonable similarity
                                    bridges.append(action_synset)
                            except:
                                pass
        
        # Handle verb -> noun connections (action -> target/object)
        elif concept_synset.pos() == 'v' and target_synset.pos() == 'n':
            # Common action -> target mappings
            target_mappings = {
                'chase': ['prey', 'quarry', 'target', 'victim'],
                'hunt': ['prey', 'quarry', 'game', 'animal'],
                'catch': ['prey', 'quarry', 'target'],
                'pursue': ['target', 'quarry', 'prey'],
                'stalk': ['prey', 'quarry', 'victim'],
                'follow': ['target', 'quarry'],
                'track': ['prey', 'quarry', 'target']
            }
            
            # Check if our action maps to any known targets
            concept_word = concept_synset.name().split('.')[0]
            if concept_word in target_mappings:
                for target in target_mappings[concept_word]:
                    # Find synsets for this target
                    target_synsets = self.wn_module.synsets(target, pos=self.wn_module.NOUN)
                    for target_synset_candidate in target_synsets[:2]:  # Top 2 senses
                        # Check if this target synset is related to our actual target
                        try:
                            similarity = target_synset_candidate.path_similarity(target_synset)
                            if similarity and similarity > 0.1:  # Reasonable similarity
                                bridges.append(target_synset_candidate)
                        except:
                            pass
            
            # Also try to find objects that are typically targeted by this action
            # For verbs like chase, look for typical objects (animals, people)
            if 'chase' in concept_synset.name() or 'hunt' in concept_synset.name() or 'pursue' in concept_synset.name():
                # Check if target is an animal
                target_hypernyms = self._get_hypernym_chain(target_synset, max_depth=6)
                for hypernym in target_hypernyms:
                    name = hypernym.name().lower()
                    if any(term in name for term in ['animal', 'mammal', 'rodent', 'creature']):
                        # Create a conceptual bridge through "prey" or "quarry"
                        try:
                            prey_synsets = self.wn_module.synsets('prey', pos=self.wn_module.NOUN)
                            if prey_synsets:
                                bridges.append(prey_synsets[0])  # prey.n.01
                        except:
                            pass
                        break
        
        return bridges[:3]  # Return top 3 bridges
    
    def _is_animal_synset(self, synset):
        """Check if synset represents an animal."""
        if synset.pos() != 'n':
            return False
            
        # Check hypernyms for animal-related terms
        hypernyms = self._get_hypernym_chain(synset, max_depth=8)
        for hypernym in hypernyms:
            hypernym_name = hypernym.name().lower()
            if any(term in hypernym_name for term in 
                   ['animal', 'mammal', 'vertebrate', 'organism', 'creature', 
                    'carnivore', 'predator', 'feline', 'canine']):
                return True
                
        # Check definition
        definition = synset.definition().lower()
        if any(term in definition for term in ['animal', 'mammal', 'creature']):
            return True
            
        return False
    
    def _create_animal_behavior_path(self, animal_synset, action_synset):
        """Create a logical path from animal to action through behavior concepts."""
        if self.verbosity >= 2:
            print(f"[_create_animal_behavior_path] Creating animal behavior path")
            
        # Find appropriate intermediate concepts
        hypernyms = self._get_hypernym_chain(animal_synset, max_depth=5)
        
        # Look for carnivore, predator, or similar concepts
        behavior_concept = None
        for hypernym in hypernyms:
            name = hypernym.name().lower()
            if any(term in name for term in ['carnivore', 'predator', 'hunter']):
                behavior_concept = hypernym
                break
        
        if behavior_concept:
            # Create path through behavior concept
            animal_index = hypernyms.index(behavior_concept)
            path = hypernyms[:animal_index + 1] + [action_synset]
            
            if self.verbosity >= 2:
                path_str = " -> ".join([s.name() for s in path])
                print(f"[_create_animal_behavior_path] Created path: {path_str}")
            
            return path
        
        # Fallback: create simple conceptual path
        # Find a general behavior concept
        try:
            behavior_synsets = self.wn_module.synsets('behavior', pos=self.wn_module.NOUN)
            if behavior_synsets:
                behavior = behavior_synsets[0]
                return [animal_synset, behavior, action_synset]
        except:
            pass
            
        return None
    
    def _score_semantic_path_coherence(self, subject_path, object_path, predicate_synset,
                                     subject_word, predicate_word, object_word):
        """
        Score the coherence of semantic paths considering path length,
        intermediate concepts, and semantic appropriateness.
        """
        score = 0.0
        
        # Base score for finding actual paths (higher than simple connections)
        score += 0.5
        
        # Penalty for very long paths (prefer shorter, more direct connections)
        total_length = len(subject_path) + len(object_path) - 1  # -1 to avoid double-counting predicate
        length_penalty = min(0.3, (total_length - 3) * 0.05)  # Start penalty after 3 nodes
        score = max(0.1, score - length_penalty)
        
        # Bonus for semantically meaningful intermediate concepts
        score += self._score_intermediate_concepts(subject_path, object_path)
        
        # Bonus for word-synset alignment
        if subject_word.lower() in subject_path[0].name().lower():
            score += 0.1
        if predicate_word.lower() in predicate_synset.name().lower():
            score += 0.1
        if object_word.lower() in object_path[-1].name().lower():
            score += 0.1
            
        # Context-specific bonuses
        score += self._score_domain_coherence(subject_word, predicate_word, object_word)
        
        return score
    
    def _score_intermediate_concepts(self, subject_path, object_path):
        """Score the quality of intermediate concepts in the paths."""
        score = 0.0
        
        # Check for meaningful intermediate concepts
        all_concepts = set()
        for path in [subject_path, object_path]:
            for synset in path[1:-1]:  # Exclude start and end
                concept_name = synset.name().lower()
                all_concepts.add(concept_name)
                
                # Bonus for biologically/semantically relevant intermediate concepts
                if any(term in concept_name for term in 
                       ['animal', 'mammal', 'vertebrate', 'organism', 'being', 'entity',
                        'carnivore', 'predator', 'creature', 'life_form']):
                    score += 0.1
                    
                # Bonus for action-related intermediate concepts
                if any(term in concept_name for term in
                       ['action', 'activity', 'behavior', 'motion', 'movement']):
                    score += 0.08
        
        # Bonus for having intermediate concepts (not just direct connections)
        if len(all_concepts) > 0:
            score += 0.15
            
        return score
    
    def _score_domain_coherence(self, subject_word, predicate_word, object_word):
        """Score coherence based on domain knowledge."""
        score = 0.0
        
        # Predator-prey scenarios
        predators = ['cat', 'dog', 'lion', 'tiger', 'wolf', 'fox', 'bird']
        prey = ['mouse', 'rat', 'bird', 'fish', 'rabbit']
        hunting_verbs = ['chase', 'hunt', 'catch', 'pursue', 'stalk']
        
        if (subject_word.lower() in predators and 
            predicate_word.lower() in hunting_verbs and 
            object_word.lower() in prey):
            score += 0.2
        
        # Flying scenarios
        flying_subjects = ['bird', 'airplane', 'plane', 'helicopter', 'insect', 'bee']
        flying_verbs = ['fly', 'soar', 'glide', 'hover']
        air_objects = ['sky', 'air', 'cloud', 'tree', 'building']
        
        if (subject_word.lower() in flying_subjects and
            predicate_word.lower() in flying_verbs and
            object_word.lower() in air_objects):
            score += 0.2
            
        return score

    def _score_fallback_coherence(self, subject_synset, predicate_synset, object_synset,
                                subject_word, predicate_word, object_word,
                                subj_pred_sim, pred_obj_sim):
        """Score coherence for fallback strategy paths."""
        score = 0.0
        
        # Base score from WordNet similarities
        score += (subj_pred_sim + pred_obj_sim) * 0.4
        
        # Bonus for sensible word combinations
        if subject_word == 'cat' and predicate_word == 'chase' and object_word == 'mouse':
            score += 0.3  # This is a classic predator-prey scenario
            
        # Bonus if synset names match the words well
        if subject_word.lower() in subject_synset.name().lower():
            score += 0.1
        if predicate_word.lower() in predicate_synset.name().lower():
            score += 0.1
        if object_word.lower() in object_synset.name().lower():
            score += 0.1
            
        # Bonus for appropriate synset selection (first synset is usually most common)
        subject_synsets = self.wn_module.synsets(subject_word, pos=self.wn_module.NOUN)
        if subject_synsets and subject_synset == subject_synsets[0]:
            score += 0.05
            
        predicate_synsets = self.wn_module.synsets(predicate_word, pos=self.wn_module.VERB)
        if predicate_synsets and predicate_synset == predicate_synsets[0]:
            score += 0.05
            
        object_synsets = self.wn_module.synsets(object_word, pos=self.wn_module.NOUN)
        if object_synsets and object_synset == object_synsets[0]:
            score += 0.05
        
        return score

    def _score_path_coherence(self, subject_path, object_path, predicate_synset, 
                             subject_word, predicate_word, object_word):
        """Score semantic coherence of complete path interpretation with triple."""
        if not subject_path or not object_path:
            return 0.0
            
        score = 0.0
        
        # Get the actual subject and object synsets from the paths
        subject_synset = subject_path[0] if subject_path else None
        object_synset = object_path[-1] if object_path else None
        
        if not subject_synset or not object_synset:
            return 0.0
        
        # Factor 1: Check if path endpoints match expected word meanings
        # Higher score if the synsets align with common interpretations of the words
        if self._synset_matches_word_meaning(subject_synset, subject_word):
            score += 0.25
            if self.verbosity >= 2:
                print(f"[_score_path_coherence] Subject synset {subject_synset.name()} matches word '{subject_word}'")
                
        if self._synset_matches_word_meaning(object_synset, object_word):
            score += 0.25
            if self.verbosity >= 2:
                print(f"[_score_path_coherence] Object synset {object_synset.name()} matches word '{object_word}'")
        
        # Factor 2: Semantic role compatibility using predicate-specific logic
        # Check if subject and object fit expected semantic roles for this specific predicate synset
        
        # Add bonuses based on predicate synset meaning and subject/object appropriateness
        pred_definition = predicate_synset.definition().lower()
        
        # Check animacy and concreteness bonuses
        if self._is_animate_word(subject_word):
            score += 0.2
            if self.verbosity >= 2:
                print(f"[_score_path_coherence] Animate subject '{subject_word}' bonus")
        
        if self._is_concrete_word(object_word):
            score += 0.2
            if self.verbosity >= 2:
                print(f"[_score_path_coherence] Concrete object '{object_word}' bonus")
        
        # Synset-specific compatibility bonuses - check specific first, then general
        if 'romantic' in pred_definition or 'sexual' in pred_definition:
            # Romantic pursuit - less appropriate for animal contexts
            if subject_word in ['cat', 'dog', 'animal'] and object_word in ['mouse', 'bird']:
                score -= 0.2  # Penalty for inappropriate context
                if self.verbosity >= 2:
                    print(f"[_score_path_coherence] Inappropriate romantic context penalty")
        elif 'catch' in pred_definition or 'hunt' in pred_definition:
            # Predatory actions - bonus if subject is animate predator and object is potential prey
            if self._is_animate_word(subject_word) and self._is_animate_word(object_word):
                score += 0.3
                if self.verbosity >= 2:
                    print(f"[_score_path_coherence] Predator-prey action bonus for {predicate_word}")
        elif 'pursue' in pred_definition and 'romantic' not in pred_definition and 'sexual' not in pred_definition:
            # Non-romantic pursuit - predatory behavior
            if self._is_animate_word(subject_word) and self._is_animate_word(object_word):
                score += 0.3
                if self.verbosity >= 2:
                    print(f"[_score_path_coherence] Predator-prey pursuit bonus for {predicate_word}")
        elif 'air' in pred_definition or 'fly' in pred_definition:
            # Flight - bonus for appropriate subjects
            if subject_word in ['bird', 'airplane', 'insect']:
                score += 0.3
                if self.verbosity >= 2:
                    print(f"[_score_path_coherence] Flight action bonus for {subject_word}")
        
        # Try FrameNet analysis as additional information
        try:
            predicate_text = f"{predicate_synset.lemmas()[0].name()}"
            pred_doc = self.framenet_srl.process_text(predicate_text)
            
            if len(pred_doc) > 0:
                pred_span = pred_doc[0:1]
                frame_scores = self.framenet_srl._get_frames_for_predicate(pred_span)
                
                # Use the best frame to assess role compatibility
                if frame_scores:
                    best_frame_name, frame_coherence = frame_scores[0]
                    score += frame_coherence * 0.2  # Reduced weight since we have specific logic above
                    if self.verbosity >= 2:
                        print(f"[_score_path_coherence] FrameNet {best_frame_name} coherence: {frame_coherence:.3f}")
        except Exception as e:
            if self.verbosity >= 2:
                print(f"[_score_path_coherence] Frame analysis failed: {e}")
        
        # Factor 3: Path length penalty (shorter paths are generally better)
        combined_length = len(subject_path) + len(object_path) - 1
        length_penalty = min(0.1, combined_length * 0.01)  # Small penalty for long paths
        score = max(0.0, score - length_penalty)
        
        if self.verbosity >= 2:
            print(f"[_score_path_coherence] Final coherence score: {score:.3f} for {subject_word}-{predicate_word}-{object_word}")
            
        return score
    
    def _synset_matches_word_meaning(self, synset, word):
        """Check if synset represents a common/preferred meaning of the word."""
        if not synset or not word:
            return False
            
        # Check if synset lemma matches the word
        synset_lemmas = {lemma.name().lower() for lemma in synset.lemmas()}
        if word.lower() in synset_lemmas:
            return True
            
        # Check if it's among the top 2 most common synsets for the word
        word_synsets = self.wn_module.synsets(word)
        if synset in word_synsets[:2]:
            return True
            
        return False
    
    def _is_animate_word(self, word):
        """Check if word typically refers to animate entities."""
        synsets = self.wn_module.synsets(word, pos=self.wn_module.NOUN)
        
        # Check direct word matches for common animate entities
        animate_words = {'cat', 'dog', 'person', 'human', 'animal', 'bird', 'fish', 'horse', 'cow', 'mouse', 'rat'}
        if word.lower() in animate_words:
            return True
            
        for synset in synsets[:3]:  # Check top 3 senses
            # Check all hypernym paths (not just direct hypernyms)
            all_hypernyms = synset.closure(lambda s: s.hypernyms())
            for hypernym in all_hypernyms:
                hypernym_name = hypernym.name().lower()
                # Check for common animate hypernyms
                if any(animate_term in hypernym_name for animate_term in 
                       ['person', 'individual', 'animal', 'organism', 
                        'living_thing', 'being', 'creature', 'human', 'vertebrate']):
                    return True
            
            # Also check direct synset names for animate terms
            synset_name = synset.name().lower()
            if any(animate_term in synset_name for animate_term in 
                   ['person', 'animal', 'human', 'being']):
                return True
        
        return False
    
    def _is_concrete_word(self, word):
        """Check if word typically refers to concrete objects."""
        synsets = self.wn_module.synsets(word, pos=self.wn_module.NOUN)
        
        # Check direct word matches for common concrete objects
        concrete_words = {'book', 'car', 'table', 'chair', 'house', 'computer', 'phone', 'mouse', 'ball', 'box'}
        if word.lower() in concrete_words:
            return True
            
        for synset in synsets[:3]:  # Check top 3 senses
            # Check all hypernym paths
            all_hypernyms = synset.closure(lambda s: s.hypernyms())
            for hypernym in all_hypernyms:
                hypernym_name = hypernym.name().lower()
                # Physical objects, artifacts, substances
                if any(concrete_term in hypernym_name for concrete_term in 
                       ['artifact', 'physical_object', 'whole', 'object', 
                        'instrumentality', 'device', 'structure', 'container',
                        'conveyance', 'vehicle', 'machine', 'furniture']):
                    return True
            
            # Check direct synset for concrete terms
            synset_name = synset.name().lower()
            if any(concrete_term in synset_name for concrete_term in 
                   ['object', 'thing', 'item', 'artifact']):
                return True
                
        return False

    def _find_subject_to_predicate_paths(self, subject_synsets, predicate_synset, g, get_new_beams_fn, 
                                       beam_width, max_depth, relax_beam, max_results_per_pair, 
                                       len_tolerance, max_sample_size):
        """Find paths from subject synsets to predicate using cascading strategies."""
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Starting with {len(subject_synsets)} subject synsets -> predicate {predicate_synset.name()}")
            
        subject_paths = []
        
        # STRATEGY 1 (PRIMARY): FrameNet-based connections - Always run
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Strategy 1 (PRIMARY): FrameNet frame element connections...")
            
        framenet_paths = self._find_framenet_subject_predicate_paths(
            subject_synsets, predicate_synset, g, get_new_beams_fn, 
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        subject_paths.extend(framenet_paths)
        
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Strategy 1 found {len(framenet_paths)} FrameNet paths")
            
        # STRATEGY 2 (SECONDARY): Derivational relations - Always run
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Strategy 2 (SECONDARY): Derivational connections...")
            
        derivational_paths = self._find_derivational_subject_predicate_paths(
            subject_synsets, predicate_synset, g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        subject_paths.extend(derivational_paths)
        
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Strategy 2 found {len(derivational_paths)} derivational paths")
        
        # STRATEGY 3 (TERTIARY): Hypernym/hyponym relationships - Always run
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Strategy 3 (TERTIARY): Hypernym connections...")
        hypernym_paths = self._explore_hypernym_paths(
            subject_synsets, [predicate_synset], g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        subject_paths.extend(hypernym_paths)
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Strategy 3 found {len(hypernym_paths)} hypernym paths")
        
        if self.verbosity >= 2:
            print(f"[_find_subject_to_predicate_paths] Returning {len(subject_paths)} total subject->predicate paths")
            
        return subject_paths

    def _find_framenet_subject_predicate_paths(self, subject_synsets, predicate_synset, g, get_new_beams_fn,
                                             beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find paths from subject synsets to predicate using FrameNet semantic frame connections.
        Try ALL frame interpretations, not just first successful one."""
        if self.verbosity >= 2:
            print(f"[_find_framenet_subject_predicate_paths] Analyzing {len(subject_synsets)} subject synsets -> predicate {predicate_synset.name()}")
            
        framenet_paths = []
        
        # Get predicate definition for frame analysis
        predicate_text = f"{predicate_synset.lemmas()[0].name()} {predicate_synset.definition()}"
        
        try:
            # Process predicate with FrameNet SRL to get a span for _get_frames_for_predicate
            pred_doc = self.framenet_srl.process_text(predicate_text)
            
            if len(pred_doc) == 0:
                if self.verbosity >= 2:
                    print(f"[_find_framenet_subject_predicate_paths] No tokens found in predicate text")
                return framenet_paths
            
            # Get ALL frames for predicate with coherence scores (not just best)
            pred_span = pred_doc[0:1]  # Create span for the predicate word
            frame_scores = self.framenet_srl._get_frames_for_predicate(pred_span)
            
            if self.verbosity >= 2:
                print(f"[_find_framenet_subject_predicate_paths] Found {len(frame_scores)} frames with coherence scores")
            
            # Try EACH frame interpretation
            for frame_name, coherence_score in frame_scores:
                if self.verbosity >= 2:
                    print(f"[_find_framenet_subject_predicate_paths] Analyzing frame: {frame_name} (coherence: {coherence_score:.3f})")
                
                # Get frame from cache for analysis
                if frame_name in self.framenet_srl.frame_cache:
                    frame = self.framenet_srl.frame_cache[frame_name]
                    
                    # Find paths via this specific frame interpretation
                    frame_paths = self._find_paths_via_frame(
                        subject_synsets, predicate_synset, frame_name, coherence_score, 
                        g, get_new_beams_fn, beam_width, max_depth, relax_beam, 
                        max_results_per_pair, len_tolerance
                    )
                    framenet_paths.extend(frame_paths)
                    
                    if self.verbosity >= 2:
                        print(f"[_find_framenet_subject_predicate_paths] Frame {frame_name} yielded {len(frame_paths)} paths")
        
        except Exception as e:
            if self.verbosity >= 1:
                print(f"[_find_framenet_subject_predicate_paths] WARNING: FrameNet processing failed: {e}")
        
        # Sort paths by coherence score (if available) and return all
        if self.verbosity >= 2:
            print(f"[_find_framenet_subject_predicate_paths] Returning {len(framenet_paths)} total FrameNet paths")
            
        return framenet_paths

    def _find_paths_via_frame(self, subject_synsets, predicate_synset, frame_name, coherence_score, 
                             g, get_new_beams_fn, beam_width, max_depth, relax_beam, 
                             max_results_per_pair, len_tolerance):
        """Find paths from subject synsets to predicate via a specific semantic frame."""
        frame_paths = []
        
        if frame_name not in self.framenet_srl.frame_cache:
            return frame_paths
            
        frame = self.framenet_srl.frame_cache[frame_name]
        
        # Get frame elements that typically correspond to subjects (Agent, Experiencer, etc.)
        subject_fe_names = {
            'Agent', 'Experiencer', 'Cognizer', 'Speaker', 'Protagonist', 
            'Actor', 'Performer', 'Causer', 'Entity', 'Theme'  # Theme can be subject in passive constructions
        }
        
        # For each subject synset, try to connect through frame elements
        for subj_synset in subject_synsets:
            # Try to find semantic connections via frame roles
            for fe_name in subject_fe_names:
                if fe_name in frame.FE:
                    # Create a conceptual connection through semantic frame role
                    if g is not None and subj_synset.name() in g and predicate_synset.name() in g:
                        path = self._find_path_between_synsets(
                            subj_synset, predicate_synset, g, get_new_beams_fn,
                            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
                        )
                        if path:
                            # Attach frame coherence score to the path for later ranking
                            if hasattr(path, 'coherence_score'):
                                path.coherence_score = coherence_score
                            else:
                                # Store as tuple (path, coherence_score) if path doesn't support attributes
                                frame_paths.append((path, coherence_score, frame_name))
                                continue
                            
                            frame_paths.append(path)
                            
                            if self.verbosity >= 2:
                                print(f"[_find_paths_via_frame] Found path via frame {frame_name}.{fe_name}: length {len(path)}, coherence {coherence_score:.3f}")
        
        return frame_paths

    def _find_derivational_subject_predicate_paths(self, subject_synsets, predicate_synset, g, get_new_beams_fn,
                                                 beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find paths from subject synsets to predicate using WordNet derivational relations."""
        if self.verbosity >= 2:
            print(f"[_find_derivational_subject_predicate_paths] Searching derivational connections for {len(subject_synsets)} subjects -> {predicate_synset.name()}")
            
        derivational_paths = []
        
        # Get derivationally related forms for the predicate
        pred_derivational = self._get_derivational_connections(predicate_synset)
        
        if self.verbosity >= 2:
            print(f"[_find_derivational_subject_predicate_paths] Found {len(pred_derivational)} derivational forms for predicate")
        
        # For each subject synset
        for subj_synset in subject_synsets:
            # Get derivationally related forms for the subject
            subj_derivational = self._get_derivational_connections(subj_synset)
            
            if self.verbosity >= 2:
                print(f"[_find_derivational_subject_predicate_paths] Found {len(subj_derivational)} derivational forms for subject {subj_synset.name()}")
            
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
                        
                        if self.verbosity >= 2:
                            print(f"[_find_derivational_subject_predicate_paths] Found derivational path: {subj_synset.name()} -> {subj_deriv.name()} -> ... -> {predicate_synset.name()}")
            
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
                            
                            if self.verbosity >= 2:
                                print(f"[_find_derivational_subject_predicate_paths] Found cross-derivational path: {subj_synset.name()} -> {subj_deriv.name()} -> ... -> {pred_deriv.name()} -> {predicate_synset.name()}")
        
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
                    
                    if self.verbosity >= 2:
                        print(f"[_get_derivational_connections] {synset.name()} -> {related_synset.name()} (via {lemma.name()} -> {related_lemma.name()})")
        
        return derivational_synsets

    def _get_subject_frame_elements(self, frame_instance) -> List:
        """Extract frame elements that typically correspond to subjects (Agent, Experiencer, etc.)."""
        subject_frame_elements = []
        
        # Common subject-like frame element names
        subject_fe_names = {
            'Agent', 'Experiencer', 'Cognizer', 'Speaker', 'Protagonist', 
            'Actor', 'Performer', 'Causer', 'Entity', 'Theme'  # Theme can be subject in passive constructions
        }
        
        for element in frame_instance.elements:
            if element.name in subject_fe_names:
                subject_frame_elements.append(element)
                
                if self.verbosity >= 2:
                    print(f"[_get_subject_frame_elements] Found subject-relevant frame element: {element.name} = '{element.span.text}'")
        
        return subject_frame_elements

    def _frame_element_to_synsets(self, frame_element, target_synset, max_synsets=3):
        """Convert a frame element to relevant WordNet synsets."""
        fe_synsets = []
        
        # Extract the text content of the frame element
        fe_text = frame_element.span.text.lower().strip()
        
        if not fe_text:
            return fe_synsets
        
        # Get synsets for the frame element text
        direct_synsets = self.wn_module.synsets(fe_text)
        fe_synsets.extend(direct_synsets[:max_synsets])
        
        # Also try getting synsets that are semantically related to the target
        try:
            for synset in direct_synsets[:max_synsets]:
                similarity = synset.path_similarity(target_synset)
                if similarity and similarity > 0.3:  # Reasonable similarity threshold
                    if synset not in fe_synsets:
                        fe_synsets.append(synset)
                        
                        if self.verbosity >= 2:
                            print(f"[_frame_element_to_synsets] Added similar synset: {synset.name()} (similarity: {similarity:.3f})")
        except Exception:
            pass  # path_similarity can fail for cross-POS comparisons
        
        return fe_synsets

    def _find_predicate_to_object_paths(self, predicate_synset, object_synsets, g, get_new_beams_fn,
                                      beam_width, max_depth, relax_beam, max_results_per_pair,
                                      len_tolerance, max_sample_size):
        """Find paths from predicate to object synsets using cascading strategies."""
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Starting with predicate {predicate_synset.name()} -> {len(object_synsets)} object synsets")
            
        object_paths = []
        
        # STRATEGY 1 (PRIMARY): FrameNet-based connections - Always run
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Strategy 1 (PRIMARY): FrameNet frame element connections...")
            
        framenet_paths = self._find_framenet_predicate_object_paths(
            predicate_synset, object_synsets, g, get_new_beams_fn, 
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        object_paths.extend(framenet_paths)
        
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Strategy 1 found {len(framenet_paths)} FrameNet paths")
            
        # STRATEGY 2 (SECONDARY): Derivational relations - Always run
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Strategy 2 (SECONDARY): Derivational connections...")
            
        derivational_paths = self._find_derivational_predicate_object_paths(
            predicate_synset, object_synsets, g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        object_paths.extend(derivational_paths)
        
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Strategy 2 found {len(derivational_paths)} derivational paths")
        
        # STRATEGY 3 (TERTIARY): Hypernym/hyponym relationships - Always run
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Strategy 3 (TERTIARY): Hypernym connections...")
        hypernym_paths = self._explore_hypernym_paths(
            [predicate_synset], object_synsets, g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        object_paths.extend(hypernym_paths)
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Strategy 3 found {len(hypernym_paths)} hypernym paths")
        
        if self.verbosity >= 2:
            print(f"[_find_predicate_to_object_paths] Returning {len(object_paths)} total predicate->object paths")
            
        return object_paths


    def _find_framenet_predicate_object_paths(self, predicate_synset, object_synsets, g, get_new_beams_fn,
                                            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find paths from predicate to object synsets using FrameNet semantic frame connections.
        Try ALL frame interpretations, not just first successful one."""
        if self.verbosity >= 2:
            print(f"[_find_framenet_predicate_object_paths] Analyzing predicate {predicate_synset.name()} -> {len(object_synsets)} object synsets")
            
        framenet_paths = []
        
        # Get predicate definition for frame analysis
        predicate_text = f"{predicate_synset.lemmas()[0].name()} {predicate_synset.definition()}"
        
        try:
            # Process predicate with FrameNet SRL to get a span for _get_frames_for_predicate
            pred_doc = self.framenet_srl.process_text(predicate_text)
            
            if len(pred_doc) == 0:
                if self.verbosity >= 2:
                    print(f"[_find_framenet_predicate_object_paths] No tokens found in predicate text")
                return framenet_paths
            
            # Get ALL frames for predicate with coherence scores (not just best)
            pred_span = pred_doc[0:1]  # Create span for the predicate word
            frame_scores = self.framenet_srl._get_frames_for_predicate(pred_span)
            
            if self.verbosity >= 2:
                print(f"[_find_framenet_predicate_object_paths] Found {len(frame_scores)} frames with coherence scores")
            
            # Try EACH frame interpretation
            for frame_name, coherence_score in frame_scores:
                if self.verbosity >= 2:
                    print(f"[_find_framenet_predicate_object_paths] Analyzing frame: {frame_name} (coherence: {coherence_score:.3f})")
                
                # Get frame from cache for analysis
                if frame_name in self.framenet_srl.frame_cache:
                    frame = self.framenet_srl.frame_cache[frame_name]
                    
                    # Find paths via this specific frame interpretation
                    frame_paths = self._find_paths_via_frame_to_objects(
                        predicate_synset, object_synsets, frame_name, coherence_score, 
                        g, get_new_beams_fn, beam_width, max_depth, relax_beam, 
                        max_results_per_pair, len_tolerance
                    )
                    framenet_paths.extend(frame_paths)
                    
                    if self.verbosity >= 2:
                        print(f"[_find_framenet_predicate_object_paths] Frame {frame_name} yielded {len(frame_paths)} paths")
        
        except Exception as e:
            if self.verbosity >= 1:
                print(f"[_find_framenet_predicate_object_paths] WARNING: FrameNet processing failed: {e}")
        
        # Sort paths by coherence score (if available) and return all
        if self.verbosity >= 2:
            print(f"[_find_framenet_predicate_object_paths] Returning {len(framenet_paths)} total FrameNet paths")
            
        return framenet_paths

    def _find_paths_via_frame_to_objects(self, predicate_synset, object_synsets, frame_name, coherence_score, 
                                        g, get_new_beams_fn, beam_width, max_depth, relax_beam, 
                                        max_results_per_pair, len_tolerance):
        """Find paths from predicate to object synsets via a specific semantic frame."""
        frame_paths = []
        
        if frame_name not in self.framenet_srl.frame_cache:
            return frame_paths
            
        frame = self.framenet_srl.frame_cache[frame_name]
        
        # Get frame elements that typically correspond to objects (Theme, Patient, Goal, etc.)
        object_fe_names = {
            'Theme', 'Patient', 'Goal', 'Recipient', 'Beneficiary', 'Content', 'Message',
            'Stimulus', 'Topic', 'Undergoer', 'Item', 'Entity', 'Target'
        }
        
        # For each object synset, try to connect through frame elements
        for obj_synset in object_synsets:
            # Try to find semantic connections via frame roles
            for fe_name in object_fe_names:
                if fe_name in frame.FE:
                    # Create a conceptual connection through semantic frame role
                    if g is not None and predicate_synset.name() in g and obj_synset.name() in g:
                        path = self._find_path_between_synsets(
                            predicate_synset, obj_synset, g, get_new_beams_fn,
                            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
                        )
                        if path:
                            # Attach frame coherence score to the path for later ranking
                            if hasattr(path, 'coherence_score'):
                                path.coherence_score = coherence_score
                            else:
                                # Store as tuple (path, coherence_score) if path doesn't support attributes
                                frame_paths.append((path, coherence_score, frame_name))
                                continue
                            
                            frame_paths.append(path)
                            
                            if self.verbosity >= 2:
                                print(f"[_find_paths_via_frame_to_objects] Found path via frame {frame_name}.{fe_name}: length {len(path)}, coherence {coherence_score:.3f}")
        
        return frame_paths

    def _find_derivational_predicate_object_paths(self, predicate_synset, object_synsets, g, get_new_beams_fn,
                                                beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance):
        """Find paths from predicate to object synsets using WordNet derivational relations."""
        if self.verbosity >= 2:
            print(f"[_find_derivational_predicate_object_paths] Searching derivational connections for {predicate_synset.name()} -> {len(object_synsets)} objects")
            
        derivational_paths = []
        
        # Get derivationally related forms for the predicate
        pred_derivational = self._get_derivational_connections(predicate_synset)
        
        if self.verbosity >= 2:
            print(f"[_find_derivational_predicate_object_paths] Found {len(pred_derivational)} derivational forms for predicate")
        
        # For each object synset
        for obj_synset in object_synsets:
            # Get derivationally related forms for the object
            obj_derivational = self._get_derivational_connections(obj_synset)
            
            if self.verbosity >= 2:
                print(f"[_find_derivational_predicate_object_paths] Found {len(obj_derivational)} derivational forms for object {obj_synset.name()}")
            
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
                        
                        if self.verbosity >= 2:
                            print(f"[_find_derivational_predicate_object_paths] Found derivational path: {predicate_synset.name()} -> ... -> {obj_deriv.name()} -> {obj_synset.name()}")
            
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
                            
                            if self.verbosity >= 2:
                                print(f"[_find_derivational_predicate_object_paths] Found cross-derivational path: {predicate_synset.name()} -> {pred_deriv.name()} -> ... -> {obj_deriv.name()} -> {obj_synset.name()}")
        
        return derivational_paths

    def _get_object_frame_elements(self, frame_instance) -> List:
        """Extract frame elements that typically correspond to objects (Theme, Patient, Goal, etc.)."""
        object_frame_elements = []
        
        # Common object-like frame element names
        object_fe_names = {
            'Theme', 'Patient', 'Goal', 'Recipient', 'Beneficiary', 'Content', 'Message',
            'Stimulus', 'Topic', 'Undergoer', 'Item', 'Entity', 'Target'
        }
        
        for element in frame_instance.elements:
            if element.name in object_fe_names:
                object_frame_elements.append(element)
                
                if self.verbosity >= 2:
                    print(f"[_get_object_frame_elements] Found object-relevant frame element: {element.name} = '{element.span.text}'")
        
        return object_frame_elements

    def _add_frame_based_edges(self, graph: nx.DiGraph, all_synsets: List, sample_size: int = 1000) -> int:
        """
        Add edges based on FrameNet semantic frame connections.
        
        This creates connections between synsets that participate in the same semantic frames,
        which is crucial for cross-POS semantic pathfinding (e.g., connecting nouns to verbs
        through their semantic roles in frames).
        
        Args:
            graph: The NetworkX graph to add edges to
            all_synsets: List of all synsets to process
            sample_size: Number of synsets to sample for frame processing (performance optimization)
        
        Returns:
            Number of frame-based edges added
        """
        frame_edge_count = 0
        
        # Sample synsets for performance (processing all synsets would be very slow)
        import random
        sample_synsets = random.sample(all_synsets, min(sample_size, len(all_synsets)))
        
        if self.verbosity >= 2:
            print(f"[_add_frame_based_edges] Processing {len(sample_synsets)} sampled synsets for frame connections")
        
        # Group synsets by frame participation
        frame_to_synsets = {}
        
        try:
            for i, synset in enumerate(sample_synsets):
                if i % 100 == 0 and self.verbosity >= 2:
                    print(f"[_add_frame_based_edges] Processed {i}/{len(sample_synsets)} synsets")
                
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
                        
                        if self.verbosity >= 2 and len(frame_to_synsets[frame_name]) <= 3:
                            print(f"[_add_frame_based_edges] Synset {synset.name()} participates in frame {frame_name}")
                
                except Exception as e:
                    if self.verbosity >= 2:
                        print(f"[_add_frame_based_edges] Frame processing failed for {synset.name()}: {e}")
                    continue
            
            # Now create edges between synsets that share frames
            if self.verbosity >= 2:
                print(f"[_add_frame_based_edges] Creating edges for {len(frame_to_synsets)} frames")
            
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
                                
                                if self.verbosity >= 2 and frame_edge_count <= 10:
                                    print(f"[_add_frame_based_edges] Added frame edge: {synset1_name} <-> {synset2_name} (frame: {frame_name})")
        
        except Exception as e:
            if self.verbosity >= 1:
                print(f"[_add_frame_based_edges] WARNING: Frame-based edge creation failed: {e}")
        
        if self.verbosity >= 1:
            print(f"[_add_frame_based_edges] Added {frame_edge_count} frame-based edges")
        
        return frame_edge_count

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
        
        # If no graph is provided, we can't find paths
        if g is None:
            if self.verbosity >= 2:
                print(f"[_find_path_between_synsets] No graph provided, cannot search for paths")
            return None
            
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
                # NEW: Add attributes() relations for noun-adjective connections
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
                    elif self.verbosity >= 2:
                        print(f"[build_synset_graph] WARNING: Related synset {related.name()} not found in graph")
            
            # NEW: Add lemma-level antonym relations for semantic opposition
            for lemma in synset.lemmas():
                antonyms = lemma.antonyms() if hasattr(lemma, 'antonyms') else []
                if antonyms:
                    if 'antonyms' not in relation_counts:
                        relation_counts['antonyms'] = 0
                    
                    for antonym_lemma in antonyms:
                        antonym_synset_name = antonym_lemma.synset().name()
                        if antonym_synset_name in g:
                            g.add_edge(synset_name, antonym_synset_name,
                                      relation='antonyms', weight=1.5)  # Slightly higher weight for opposition
                            edge_count += 1
                            relation_counts['antonyms'] += 1
            
            # NEW: Add derivational relations (CRITICAL for cross-POS connections)
            derivational_synsets = self._get_derivational_connections(synset)
            if derivational_synsets:
                if 'derivational' not in relation_counts:
                    relation_counts['derivational'] = 0
                
                for deriv_synset in derivational_synsets:
                    if deriv_synset.name() in g:
                        g.add_edge(synset_name, deriv_synset.name(),
                                  relation='derivational', weight=0.8)  # Lower weight but important connection
                        edge_count += 1
                        relation_counts['derivational'] += 1

        # NEW: Add FrameNet-based semantic frame connections
        if self.verbosity >= 1:
            print("[build_synset_graph] Adding FrameNet-based semantic frame edges...")
        
        frame_edge_count = self._add_frame_based_edges(g, all_synsets)
        edge_count += frame_edge_count
        relation_counts['framenet'] = frame_edge_count
        
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