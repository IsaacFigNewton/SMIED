asymmetric_pairs_map = {
    # holonyms
    "part_holonyms": "part_meronyms",
    "substance_holonyms": "substance_meronyms",
    "member_holonyms": "member_meronyms",

    # meronyms
    "part_meronyms": "part_holonyms",
    "substance_meronyms": "substance_holonyms",
    "member_meronyms": "member_holonyms",

    # other
    "hypernyms": "hyponyms",
    "hyponyms": "hyponyms"
}


symmetric_pairs_map = {
    # holonyms
    "part_holonyms": "part_holonyms",
    "substance_holonyms": "substance_holonyms",
    "member_holonyms": "member_holonyms",

    # meronyms
    "part_meronyms": "part_meronyms",
    "substance_meronyms": "substance_meronyms",
    "member_meronyms": "member_meronyms",

    # other
    "hypernyms": "hypernyms",
    "hyponyms": "hyponyms",
    "entailments": "entailments",
    "causes": "causes",
    "also_sees": "also_sees",
    "verb_groups": "verb_groups"
}

def get_new_beams(
      g: nx.DiGraph,
      src: str,
      tgt: str,
      model=embedding_model,
      beam_width=3
    ) -> List[Tuple[
        Tuple[str, str],
        Tuple[str, str],
        float
    ]]:
    """
    Get the k closest pairs of lexical relations between 2 synsets.

    Args:
        src: WordNet Synset object (e.g., 'dog.n.01')
        tgt: WordNet Synset object (e.g., 'cat.n.01')
        model: token model (if None, will load default)
        beam_width: max number of pairs to return

    Returns:
        List of tuples of the form:
          (
            (synset1, lexical_rel),
            (synset2, lexical_rel),
            relatedness
          )
    """

    # Build a map of each synset's associated lexical relations
    #   and the centroids of their associated synsets
    src_lex_rel_embs = embed_lexical_relations(wn.synset(src), model)
    tgt_lex_rel_embs = embed_lexical_relations(wn.synset(tgt), model)

    # ensure the edges in the nx graph align with those in the embedding maps
    src_neighbors = {n for n in g.neighbors(src)}
    for rel, synset_list in src_lex_rel_embs.items():
      if not all(s[0] in src_neighbors for s in synset_list):
        raise ValueError(f"Not all lexical properties of {src} ({[s[0] for s in synset_list]}) in graph for relation {rel}")
    tgt_neighbors = {n for n in g.neighbors(tgt)}
    for rel, synset_list in tgt_lex_rel_embs.items():
      if not all(s[0] in tgt_neighbors for s in synset_list):
        raise ValueError(f"Not all lexical properties of {tgt} ({[s[0] for s in synset_list]}) in graph for relation {rel}")
    # in the future, get neighbor relation in node metadata with g.adj[n]

    # Get the asymmetric lexical relation pairings,
    #   sorted in descending order of embedding similarity
    #   e.x. similarity of synset1's hypernyms to synset2's hypernyms
    asymm_lex_rel_sims = get_top_k_aligned_lex_rel_pairs(
        asymmetric_pairs_map,
        src_lex_rel_embs,
        tgt_lex_rel_embs,
        model,
        beam_width
    )
    # Get the symmetric lexical relation pairings,
    #   sorted in descending order of embedding similarity
    #   e.x. similarity of synset1's hypernyms to synset2's hypernyms
    symm_lex_rel_sims = get_top_k_aligned_lex_rel_pairs(
        symmetric_pairs_map,
        src_lex_rel_embs,
        tgt_lex_rel_embs,
        model,
        beam_width
    )
    combined = asymm_lex_rel_sims + symm_lex_rel_sims
    beam = sorted(combined, key=lambda x: x[2], reverse=True)[:beam_width]
    return beam