from collections import deque
from typing import List, Optional, Tuple, Set
import nltk.corpus.wordnet as wn
from .GlossParser import GlossParser
from .interfaces.ICrossPOSBrancher import ICrossPOSBrancher


class CrossPOSBrancher(ICrossPOSBrancher):
    """
    Handles cross-part-of-speech path finding and legacy/experimental functionality.
    
    This class contains older pathfinding methods and cross-POS search functionality
    that may be useful for specialized use cases or as fallback methods.
    """
    
    def __init__(self, wn_module, nlp_func, gloss_parser: Optional[GlossParser] = None):
        """
        Initialize with WordNet module and NLP function.
        
        Args:
            wn_module: WordNet module (e.g., nltk.corpus.wordnet)
            nlp_func: spaCy NLP function for text processing
            gloss_parser: Optional GlossParser instance (will create one if not provided)
        """
        self.wn_module = wn_module
        self.nlp_func = nlp_func
        self.gloss_parser = gloss_parser if gloss_parser is not None else GlossParser()

    def path_syn_to_syn(self, start_synset, end_synset, max_depth=6):
        """
        Find shortest path between synsets of the same POS using bidirectional BFS.
        Returns a list of synsets forming the path, or None if no path found.
        """
        if not (start_synset.pos() == end_synset.pos() and start_synset.pos() in {'n', 'v'}):
            raise ValueError(f"{start_synset.name()} POS tag != {end_synset.name()}. Synsets must be of the same POS (noun or verb).")

        # Handle the trivial case where start and end are the same
        if start_synset.name() == end_synset.name():
            return [start_synset]

        # Initialize two search frontiers
        forward_queue = deque([(start_synset, 0)])
        forward_visited = {start_synset.name(): [start_synset]}

        backward_queue = deque([(end_synset, 0)])
        backward_visited = {end_synset.name(): [end_synset]}

        def expand_frontier(queue, visited_from_this_side, visited_from_other_side, is_forward):
            """Expand one step of the search frontier."""
            if not queue:
                return None

            curr_synset, depth = queue.popleft()

            if depth >= (max_depth + 1) // 2:
                return None

            path_to_current = visited_from_this_side[curr_synset.name()]

            for neighbor in self.gloss_parser.get_all_neighbors(curr_synset):
                neighbor_name = neighbor.name()

                if neighbor_name in visited_from_this_side:
                    continue

                if is_forward:
                    new_path = path_to_current + [neighbor]
                else:
                    new_path = [neighbor] + path_to_current

                if neighbor_name in visited_from_other_side:
                    other_path = visited_from_other_side[neighbor_name]

                    if is_forward:
                        full_path = path_to_current + other_path
                    else:
                        full_path = other_path + path_to_current

                    return full_path

                visited_from_this_side[neighbor_name] = new_path
                queue.append((neighbor, depth + 1))

            return None

        # Alternate between forward and backward search
        while forward_queue or backward_queue:
            if forward_queue:
                result = expand_frontier(forward_queue, forward_visited, backward_visited, True)
                if result:
                    return result

            if backward_queue:
                result = expand_frontier(backward_queue, backward_visited, forward_visited, False)
                if result:
                    return result

        return None

    def get_all_neighbors(self, synset):
        """Get all lexically related neighbors of a synset."""
        return self.gloss_parser.get_all_neighbors(synset)

    def extract_subjects_from_gloss(self, gloss_doc):
        """Extract subject tokens from a parsed gloss."""
        return self.gloss_parser.extract_subjects_from_gloss(gloss_doc)

    def extract_objects_from_gloss(self, gloss_doc):
        """Extract various types of object tokens from a parsed gloss."""
        return self.gloss_parser.extract_objects_from_gloss(gloss_doc)

    def extract_verbs_from_gloss(self, gloss_doc, include_passive=False):
        """Extract verb tokens from a parsed gloss."""
        return self.gloss_parser.extract_verbs_from_gloss(gloss_doc, include_passive)

    def find_instrumental_verbs(self, gloss_doc):
        """Find verbs associated with instrumental use (e.g., 'used for')."""
        return self.gloss_parser.find_instrumental_verbs(gloss_doc)

    def get_top_k_synset_branch_pairs(
        self,
        candidates: List[List],
        target_synset,
        beam_width=3
    ) -> List[Tuple[Tuple, Tuple, float]]:
        """
        Given a list of candidate tokens and a target synset,
        Return the k synset subrelation pairs most similar to the target of the form:
          ((synset, lexical_rel), (name, lexical_rel), relatedness)
        
        NOTE: This is a legacy/experimental implementation with incomplete functionality.
        """
        beam = []
        # for each list of possible synsets for a candidate token
        for synsets in candidates:
            if synsets:
                for synset in synsets:
                    # Placeholder implementation - in a real scenario you would implement
                    # proper synset relatedness computation here
                    try:
                        # Simple placeholder similarity based on path similarity if available
                        sim = synset.path_similarity(target_synset) or 0.0
                        beam.append(((synset, "placeholder_rel"), (target_synset, "placeholder_rel"), sim))
                    except Exception:
                        # Fallback for synsets that don't support path_similarity
                        beam.append(((synset, "placeholder_rel"), (target_synset, "placeholder_rel"), 0.1))
        
        beam = sorted(beam, key=lambda x: x[2], reverse=True)[:beam_width]
        return beam
# ============================================================================
# Core Path Finding Functions
# ============================================================================

def path_syn_to_syn(start_synset, end_synset, max_depth=6):
    """
    Find shortest path between synsets of the same POS using bidirectional BFS.
    Returns a list of synsets forming the path, or None if no path found.
    """

    if not (start_synset.pos() == end_synset.pos() and start_synset.pos() in {'n', 'v'}):
      raise ValueError(f"{start_synset.name()} POS tag != {end_synset.name()}. Synsets must be of the same POS (noun or verb).")

    # Handle the trivial case where start and end are the same
    if start_synset.name() == end_synset.name():
        return [start_synset]

    # Initialize two search frontiers
    forward_queue = deque([(start_synset, 0)])
    forward_visited = {start_synset.name(): [start_synset]}

    backward_queue = deque([(end_synset, 0)])
    backward_visited = {end_synset.name(): [end_synset]}

    def expand_frontier(queue, visited_from_this_side, visited_from_other_side, is_forward):
        """Expand one step of the search frontier."""
        if not queue:
            return None

        curr_synset, depth = queue.popleft()

        if depth >= (max_depth + 1) // 2:
            return None

        path_to_current = visited_from_this_side[curr_synset.name()]

        for neighbor in get_all_neighbors(curr_synset):
            neighbor_name = neighbor.name()

            if neighbor_name in visited_from_this_side:
                continue

            if is_forward:
                new_path = path_to_current + [neighbor]
            else:
                new_path = [neighbor] + path_to_current

            if neighbor_name in visited_from_other_side:
                other_path = visited_from_other_side[neighbor_name]

                if is_forward:
                    full_path = path_to_current + other_path
                else:
                    full_path = other_path + path_to_current

                return full_path

            visited_from_this_side[neighbor_name] = new_path
            queue.append((neighbor, depth + 1))

        return None

    # Alternate between forward and backward search
    while forward_queue or backward_queue:
        if forward_queue:
            result = expand_frontier(forward_queue, forward_visited, backward_visited, True)
            if result:
                return result

        if backward_queue:
            result = expand_frontier(backward_queue, backward_visited, forward_visited, False)
            if result:
                return result

    return None


# ============================================================================
# Cross-POS Path Finding Functions
# ============================================================================
def get_top_k_synset_branch_pairs(
      candidates: List[List[wn.synset]],
      target_synset: wn.synset,
      beam_width=3
    ) -> List[Tuple[
        Tuple[wn.synset, str],
        Tuple[wn.synset, str],
        float
    ]]:
    """
    Given a list of candidate tokens and a target synset,


    Return the k synset subrelation pairs most similar to the target of the form:
      ((synset, lexical_rel), (name, lexical_rel), relatedness)
    """
    top_k_asymm_branches = list()
    top_k_symm_branches = list()
    beam = list()
    # for each list of possible synsets for a candidate token
    for synsets in candidates:
        # filter to subjects based on whether they reside in the same sub-category
        #   where the subcategory != 'entity.n.01' or a similar top-level
        synsets = [
            s for s in synsets
            if s.root_hypernyms() != s.lowest_common_hypernyms(target_synset)
        ]
        # # if there are synsets left for the candidate token after pruning
        if synsets:
            # if the target is a verb,
            #   filter out any synsets with no lemma frames matching the target
            #   frame patterns: (Somebody [v] something), (Somebody [v]), ...
            if target_synset.pos() == 'v':
                synsets = [
                    s for s in synsets
                    if any(
                        frame in s.frame_ids()
                        for frame in target_synset.frame_ids()
                    )
                ]
            for synset in synsets:
              beam += get_synset_relatedness(synset, target_synset)
              beam += get_synset_relatedness(synset, target_synset)
    beam = sorted(
        beam,
        key=lambda x: x[2],
        reverse=True
    )[:beam_width]
    return beam



    
def find_subject_to_predicate_path(
      subject_synset: wn.synset,
      predicate_synset: wn.synset,
      max_depth:int,
      visited=set(),
      max_sample_size=5,
    ):
    """Find path from subject (noun) to predicate (verb)."""
    if subject_synset.name() in visited or predicate_synset.name() in visited:
      return None

    paths = []
    print()
    print(f"Finding path from {subject_synset.name()} to {predicate_synset.name()}")

    # Strategy 1: Look for active subjects in verb's gloss
    pred_gloss_doc = nlp(predicate_synset.definition())
    # passive subjects are semantically equivalent to objects
    active_subjects, _ = extract_subjects_from_gloss(pred_gloss_doc)
    # convert spacy tokens to lists of synsets
    subjects = [wn.synsets(s.text, pos=subject_synset.pos()) for s in active_subjects]
    # of the remaining subjects, get the most similar
    top_k = get_top_k_synset_branches(active_subjects[:max_sample_size], subject_synset)
    if top_k:
      print(f"Found best matches for {subject_synset.name()}: {top_k} using strategy 1")
      for matched_synset, _ in top_k:
        path = path_syn_to_syn(subject_synset, matched_synset, max_depth-1)
        if path:
            paths.append(path + [predicate_synset])

    # Strategy 2: Look for verbs in the noun's gloss
    subj_gloss_doc = nlp(subject_synset.definition())
    verbs = extract_verbs_from_gloss(subj_gloss_doc, include_passive=False)
    # convert spacy tokens to lists of synsets
    verbs = [wn.synsets(v.text, pos=predicate_synset.pos()) for v in verbs]
    # of the remaining subjects, get the most similar
    top_k = get_top_k_synset_branches(verbs[:max_sample_size], predicate_synset)
    if top_k:
      print(f"Found best matches for {predicate_synset.name()}: {top_k} using strategy 2")
      for matched_synset, _ in top_k:
        path = path_syn_to_syn(matched_synset, predicate_synset, max_depth-1)
        if path:
            paths.append([subject_synset] + path)

    # Strategy 3: Explore the 3 most promising pairs of neighbors
    subject_neighbors = get_all_neighbors(subject_synset)
    predicate_neighbors = get_all_neighbors(predicate_synset)
    top_k = get_k_closest_synset_pairs(subject_neighbors, predicate_neighbors)
    if top_k:
      print(f"Most promising pairs for bidirectional exploration: {top_k} using strategy 3")
      for s, p, _ in top_k:
        visited.add(subject_synset.name())
        visited.add(predicate_synset.name())
        path = find_subject_to_predicate_path(s, p, max_depth-1, visited)
        if path:
            paths.append([subject_synset] + path + [predicate_synset])


    # Return shortest path if any found
    return min(paths, key=len) if paths else None


def find_predicate_to_object_path(
      predicate_synset: wn.synset,
      object_synset: wn.synset,
      max_depth:int,
      visited=set(),
      max_sample_size=5,
    ):
    """Find path from predicate (verb) to object (noun)."""

    if predicate_synset.name() in visited or object_synset.name() in visited:
      return None

    paths = []
    print()
    print(f"Finding path from {predicate_synset.name()} to {object_synset.name()}")

    # === Strategy 1: Objects in predicate gloss (incl. passive subjects) ===
    pred_gloss_doc = nlp(predicate_synset.definition())
    objects = extract_objects_from_gloss(pred_gloss_doc)
    _, passive_subjects = extract_subjects_from_gloss(pred_gloss_doc)
    objects.extend(passive_subjects)
    # convert spacy tokens to lists of synsets
    objects = [wn.synsets(o.text, pos=object_synset.pos()) for o in objects]
    top_k = get_top_k_synset_branches(objects[:max_sample_size], object_synset)
    if top_k:
      print(f"Found best matches for {object_synset.name()}: {top_k} using strategy 1")
      for matched_synset, _ in top_k:
        path = path_syn_to_syn(matched_synset, object_synset, max_depth-1)
        if path:
            paths.append([predicate_synset] + path)

    # === Strategy 2: Verbs in object's gloss ===
    obj_gloss_doc = nlp(object_synset.definition())
    verbs = extract_verbs_from_gloss(obj_gloss_doc, include_passive=True)
    # Use instrumental verbs in object's gloss as backup
    verbs.extend(find_instrumental_verbs(obj_gloss_doc))
    # convert spacy tokens to lists of synsets
    verbs = [wn.synsets(v.text, pos=predicate_synset.pos()) for v in verbs]
    top_k = get_top_k_synset_branches(verbs[:max_sample_size], predicate_synset)
    if top_k:
      print(f"Found best matches for {predicate_synset.name()}: {top_k} using strategy 2")
      for matched_synset, _ in top_k:
        path = path_syn_to_syn(predicate_synset, matched_synset, max_depth-1)
        if path:
            paths.append(path + [object_synset])

    # Strategy 3: Explore the 3 most promising neighbors
    predicate_neighbors = get_all_neighbors(predicate_synset)
    object_neighbors = get_all_neighbors(object_synset)
    top_k = get_k_closest_synset_pairs(predicate_neighbors, object_neighbors)
    if top_k:
      print(f"Most promising pairs for bidirectional exploration: {top_k} using strategy 3")
      for p, o, _ in top_k:
        visited.add(predicate_synset.name())
        visited.add(object_synset.name())
        path = find_predicate_to_object_path(p, o, max_depth-1, visited)
        if path:
            paths.append([predicate_synset] + path + [object_synset])


    # Return shortest path if any found
    return min(paths, key=len) if paths else None


# ============================================================================
# Main Connected Path Finding Function
# ============================================================================

def find_connected_shortest_paths(
      subject_word,
      predicate_word,
      object_word,
      max_depth=10,
      max_self_intersection=5
    ):
    """
    Find shortest connected paths from subject through predicate to object.
    Ensures that the same predicate synset connects both paths.
    """

    # Get synsets for each word
    subject_synsets = wn.synsets(subject_word, pos=wn.NOUN)
    predicate_synsets = wn.synsets(predicate_word, pos=wn.VERB)
    object_synsets = wn.synsets(object_word, pos=wn.NOUN)

    best_combined_path_length = float('inf')
    best_subject_path = None
    best_object_path = None
    best_predicate = None

    # Try each predicate synset as the connector
    for pred in predicate_synsets:
        # Find paths from all subjects to this specific predicate
        subject_paths = []
        for subj in subject_synsets:
            path = find_subject_to_predicate_path(subj, pred, max_depth)
            if path:
                subject_paths.append(path)

        # Find paths from this specific predicate to all objects
        object_paths = []
        for obj in object_synsets:
            path = find_predicate_to_object_path(pred, obj, max_depth)
            if path:
                object_paths.append(path)

        # If we have both paths through this predicate, check if it's the best
        if subject_paths and object_paths:
            # find pairs of paths that don't intersect with eachother
            #   i.e. burglar > break_in > attack > strike > shoot > strike > attack > woman
            #   would not be allowed, since tautological statements are uninformative
            valid_pairs = list()
            for subj_path in subject_paths:
              for obj_path in object_paths:
                if len(set(subj_path).intersection(set(obj_path))) <= max_self_intersection:
                  valid_pairs.append((
                      subj_path,
                      obj_path,
                      # Calculate combined length (subtract 1 to avoid counting predicate twice)
                      len(subj_path) + len(obj_path) - 1
                  ))

            if not valid_pairs:
              print(f"No valid pairs of subj, obj paths found for {pred.name()}")
              break

            shortest_comb_path = min(valid_pairs, key=lambda x: x[2])

            if shortest_comb_path[2] < best_combined_path_length:
                best_combined_path_length = shortest_comb_path[2]
                best_subject_path = shortest_comb_path[0]
                best_object_path = shortest_comb_path[1]
                best_predicate = pred

    return best_subject_path, best_object_path, best_predicate


# ============================================================================
# Display Functions
# ============================================================================

def show_path(label, path):
    """Pretty print a path of synsets."""
    if path:
        print(f"{label}:")
        print(" -> ".join(f"{s.name()} ({s.definition()})" for s in path))
        print(f"Path length: {len(path)}")
        print()
    else:
        print(f"{label}: No path found")
        print()


def show_connected_paths(subject_path, object_path, predicate):
    """Display the connected paths with their shared predicate."""
    if subject_path and object_path and predicate:
        print("=" * 70)
        print(f"CONNECTED PATH through predicate: {predicate.name()}")
        print("=" * 70)

        show_path("Subject -> Predicate path", subject_path)
        show_path("Predicate -> Object path", object_path)

        # Show the complete connected path
        complete_path = subject_path + object_path[1:]  # Avoid duplicating the predicate
        print("Complete connected path:")
        print(" -> ".join(f"{s.name()}" for s in complete_path))
        print(f"Total path length: {len(complete_path)}")
        print()
    else:
        print("No connected path found through any predicate synset.")
