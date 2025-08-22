# ============================================================================
# Gloss Analysis Helper Functions (Keep as-is, they work well)
# ============================================================================

def extract_subjects_from_gloss(gloss_doc):
    """Extract subject tokens from a parsed gloss."""
    subjects = []

    # Direct subjects
    subjects.extend([tok for tok in gloss_doc if tok.dep_ == "nsubj"])

    # Passive subjects (which are actually objects semantically)
    # Skip these for actor identification
    passive_subjects = [tok for tok in gloss_doc if tok.dep_ == "nsubjpass"]

    # Filter out passive subjects from the main list
    subjects = [s for s in subjects if s not in passive_subjects]

    return subjects, passive_subjects


def extract_objects_from_gloss(gloss_doc):
    """Extract various types of object tokens from a parsed gloss."""
    objs = []

    # Indirect objects
    iobjs = [tok for tok in gloss_doc if tok.dep_ == "iobj"]
    objs.extend(iobjs)

    # Direct objects
    # Only include if there were no indirect objects,
    #   crude, but good for MVP
    if not iobjs:
        objs.extend([tok for tok in gloss_doc if tok.dep_ == "dobj"])

    # Prepositional objects
    objs.extend([tok for tok in gloss_doc if tok.dep_ == "pobj"])

    # General objects
    objs.extend([tok for tok in gloss_doc if tok.dep_ == "obj"])

    # Check for noun chunks related to root verb
    root_verbs = [tok for tok in gloss_doc if tok.dep_ == "ROOT" and tok.pos_ == "VERB"]
    if root_verbs and not objs:
        for noun_chunk in gloss_doc.noun_chunks:
            if any(token.head == root_verbs[0] for token in noun_chunk):
                objs.append(noun_chunk.root)

    return objs


def extract_verbs_from_gloss(gloss_doc, include_passive=False):
    """Extract verb tokens from a parsed gloss."""
    verbs = [tok for tok in gloss_doc if tok.pos_ == "VERB"]

    if include_passive:
        # Past participles used as adjectives or in relative clauses
        passive_verbs = [tok for tok in gloss_doc if
                        tok.tag_ in ["VBN", "VBD"] and
                        tok.dep_ in ["acl", "relcl", "amod"]]
        verbs.extend(passive_verbs)

    return verbs


def find_instrumental_verbs(gloss_doc):
    """Find verbs associated with instrumental use (e.g., 'used for')."""
    instrumental_verbs = []

    if "used" in gloss_doc.text.lower():
        for i, token in enumerate(gloss_doc):
            if token.text.lower() == "used":
                # Check tokens after "used"
                for j in range(i+1, min(i+4, len(gloss_doc))):
                    if gloss_doc[j].pos_ == "VERB":
                        instrumental_verbs.append(gloss_doc[j])

    return instrumental_verbs


# ============================================================================
# Simple same-POS path finding (for backward compatibility)
# ============================================================================

# def get_all_neighbors(synset, wn_module=None):
#     """Get all lexically related neighbors of a synset."""
#     neighbors = set()

#     # Add all types of relations
#     relation_methods = [
#         'hypernyms', 'hyponyms', 'holonyms', 'meronyms',
#         'similar_tos', 'also_sees', 'verb_groups',
#         'entailments', 'causes', 'attributes'
#     ]

#     for method_name in relation_methods:
#         if hasattr(synset, method_name):
#             try:
#                 related = getattr(synset, method_name)()
#                 neighbors.update(related)
#             except:
#                 pass

#     return list(neighbors)


def path_syn_to_syn(start_synset, end_synset, max_depth=6, wn_module=None):
    """
    Find shortest path between synsets of the same POS using bidirectional BFS.
    Returns a list of synset names (strings) forming the path, or None if no path found.
    """
    # Convert to names for consistency
    start_name = start_synset.name() if hasattr(start_synset, 'name') else str(start_synset)
    end_name = end_synset.name() if hasattr(end_synset, 'name') else str(end_synset)

    # Check if same POS (if we have synset objects)
    if hasattr(start_synset, 'pos') and hasattr(end_synset, 'pos'):
        if start_synset.pos() != end_synset.pos():
            return None

    # Handle the trivial case where start and end are the same
    if start_name == end_name:
        return [start_name]

    # Initialize two search frontiers
    forward_queue = deque([(start_synset, 0)])
    forward_visited = {start_name: [start_name]}

    backward_queue = deque([(end_synset, 0)])
    backward_visited = {end_name: [end_name]}

    def expand_frontier(queue, visited_from_this_side, visited_from_other_side, is_forward):
        """Expand one step of the search frontier."""
        if not queue:
            return None

        curr_synset, depth = queue.popleft()

        if depth >= (max_depth + 1) // 2:
            return None

        curr_name = curr_synset.name() if hasattr(curr_synset, 'name') else str(curr_synset)
        path_to_current = visited_from_this_side[curr_name]

        for neighbor in get_all_neighbors(curr_synset, wn_module):
            neighbor_name = neighbor.name() if hasattr(neighbor, 'name') else str(neighbor)

            if neighbor_name in visited_from_this_side:
                continue

            if is_forward:
                new_path = path_to_current + [neighbor_name]
            else:
                new_path = [neighbor_name] + path_to_current

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