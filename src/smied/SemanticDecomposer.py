# ============================================================================
# Main wrapper function to replace find_connected_shortest_paths
# ============================================================================

def find_connected_shortest_paths(
    subject_word: str,
    predicate_word: str,
    object_word: str,
    wn_module,
    nlp_func,
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
    Wrapper that uses the new find_connected_paths architecture.
    Returns the best connected path in the old format for backward compatibility.
    """

    # Import the necessary components
    from pathfinding_core import (
        PairwiseBidirectionalAStar,
        find_connected_paths,
        get_new_beams_from_embeddings
    )

    # Create the beam function if we have a model
    get_new_beams_fn = None
    if model is not None and g is not None:
        get_new_beams_fn = lambda g, src, tgt: get_new_beams_from_embeddings(
            g, src, tgt, wn_module, model, beam_width=beam_width
        )

    # Create the top_k_branch function
    top_k_branch_fn = None
    if model is not None:
        top_k_branch_fn = lambda candidates, target, bw: get_top_k_synset_branch_pairs(
            candidates, target, bw, model, wn_module
        )

    # Build the graph if not provided
    if g is None:
        g = build_synset_graph(wn_module)  # You'll need to implement this

    # Call the new find_connected_paths function
    results = find_connected_paths(
        g=g,
        subject_word=subject_word,
        predicate_word=predicate_word,
        object_word=object_word,
        wn_module=wn_module,
        nlp_func=nlp_func,
        get_new_beams_fn=get_new_beams_fn,
        top_k_branch_fn=top_k_branch_fn,
        extract_subjects_from_gloss=extract_subjects_from_gloss,
        extract_objects_from_gloss=extract_objects_from_gloss,
        beam_width=beam_width,
        max_depth=max_depth,
        max_self_intersection=max_self_intersection,
        max_results_per_pair=max_results_per_pair,
        len_tolerance=len_tolerance,
        relax_beam=relax_beam
    )

    # Convert results to old format (best_subject_path, best_object_path, best_predicate)
    if results:
        best_result = results[0]  # Take the best result
        # Convert synset names back to synset objects if needed
        subject_path = [wn_module.synset(name) if isinstance(name, str) else name
                       for name in best_result["subject_path"]]
        object_path = [wn_module.synset(name) if isinstance(name, str) else name
                      for name in best_result["object_path"]]
        pred_synset = wn_module.synset(best_result["predicate_synset"]) \
                     if isinstance(best_result["predicate_synset"], str) \
                     else best_result["predicate_synset"]

        return subject_path, object_path, pred_synset

    return None, None, None


# ============================================================================
# Helper function to build synset graph (if needed)
# ============================================================================

def build_synset_graph(wn_module) -> nx.DiGraph:
    """
    Build a directed graph of synsets with their lexical relations.
    """
    g = nx.DiGraph()

    # Get all synsets (you may want to limit this for performance)
    all_synsets = list(wn_module.all_synsets())

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

    return g


# ============================================================================
# Display Functions (keep as-is for backward compatibility)
# ============================================================================

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


def show_connected_paths(subject_path, object_path, predicate):
    """Display the connected paths with their shared predicate."""
    if subject_path and object_path and predicate:
        print("=" * 70)
        pred_name = predicate.name() if hasattr(predicate, 'name') else str(predicate)
        print(f"CONNECTED PATH through predicate: {pred_name}")
        print("=" * 70)

        show_path("Subject -> Predicate path", subject_path)
        show_path("Predicate -> Object path", object_path)

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