# Query-Based Semantic Role Labeling with UVI Package

# Helpers
```python
import networkx as nx

def json_to_digraph(data, graph=None, parent=None):
    """
    Convert a nested JSON-like dict into a networkx.DiGraph.
    
    - Each dict key becomes a node if its value is another dict.
    - Each leaf value becomes a node labeled by its stringified value.
    - Leaf nodes store the raw (non-string) value in 'raw_value' metadata.
    """
    if graph is None:
        graph = nx.DiGraph()
    
    if isinstance(data, dict):
        for key, value in data.items():
            key_node = str(key)
            graph.add_node(key_node, raw_value=key)  # store key as metadata
            if parent is not None:
                graph.add_edge(parent, key_node)
            
            if isinstance(value, dict):
                json_to_digraph(value, graph, key_node)
            else:
                leaf_node = str(value)
                graph.add_node(leaf_node, raw_value=value)
                graph.add_edge(key_node, leaf_node)
    else:
        # root-level non-dict value
        leaf_node = str(data)
        graph.add_node(leaf_node, raw_value=data)
        if parent is not None:
            graph.add_edge(parent, leaf_node)
    
    return graph

def is_child(g, child:str, parent:str):
    # check if the child node is a successor of the parent node
    return child in g.successors(parent)
```

## 1 Input
A spacy.tokens.Doc object containing one verb with at least 1 subject, ex: "Jenny studied hard" 

## 2 Processing
## 2.1 Get all possible frames
- although the proper noun "Jenny" lacks a lemma, it would still have possible roles assigned to it, since the lemma of "studied", "study", does have a lemma.

Combine the following functions in a coherent, cohesive UVI_SRL class
```python
def get_frame_syn_pairs(uvi: UVI, predicate_token: spacy.tokens.Token) -> Set[Tuple[str, str]]:
    all_corpus_matches = uvi.search_lemmas(predicate_token.lemma_)["matches"]
    # uvi semantic subgraph
    g = json_to_digraph(all_corpus_matches)
    seed_nodes = all_corpus_matches.keys()
    # each synset node
    synset_nodes = uvi.get_wn_synset_nodes(g)
    frame_nodes = uvi.get_frame_nodes(g)

    # filter frame_nodes to just those with at least one neighbor of type "role" with a subtype of "Agent"
    def _has_subj_role(frame_node: str):
        subj_roles = ["Agent", "Actor"]
        return any(is_child(g, r, frame_node) for r in subj_roles)

    frame_nodes = [
        n for n in frame_nodes
        if _has_subj_role(n)
    ]

    # get all pairs of neighboring frames, synsets from the lemma's filtered semantic subgraph
    frame_syn_pairs = set()
    for f_node in frame_nodes:
        for s_node in synset_nodes:
            if s_node in g.successors(f_node)\
                or f_node in g.successors(s_node):
                frame_syn_pairs.append(f_node, s_node)

    return frame_syn_pairs
```

```python
def _align_roles_deps(
        uvi,
        f_node: str,
        pred_tok: spacy.tokens.Token
    ) -> List[Tuple[str, str]]:
    # check if the frame node has the required neighbors
    fe_neighbor_map = {
        "subj": "Agent",
        "obj": "Patient"
    }

    f_node_neighbors = uvi.semantic_graph.neighbors(f_node)
    # TODO: finish
    #   if a "perfect" alignment can't be found, align the nodes that can be aligned
    #   and leave the rest as separate
    pass
```

```python
def get_srl_digraph(uvi, pred_tok) -> Optional[nx.DiGraph]:
    frame_syn_pairs = get_frame_syn_pairs(pred_tok)

    # if WSD frames were found, filter to just those that match the dependency structure of doc
    aligned_frame_syn_roles_edge_list: List[Tuple[str, str]] = list()
    aligned_sem_graph = nx.DiGraph()
    # align the frame entities of each frame with the tokens of the original doc
    #   based on their dependency relations to the predicate (verb)
    for p in frame_syn_pairs:
        aligned_srls: List[Tuple[str, str]] = _align_roles_deps(uvi, p[0], pred_tok)
        if aligned_srls:
            aligned_frame_syn_roles_edge_list.extend(aligned_srls)

    # build the SRL graph
    aligned_sem_graph.add_edges_from(aligned_frame_syn_roles_edge_list)
    # if there are complete, aligned structures of frames, synsets, and frame entities
    if aligned_frame_syn_roles_edge_list:
        return aligned_sem_graph

    # if no subgraph homomorphisms existed
    return None
```

```python
uvi = UVI()
doc = nlp("Jenny studied hard.")
pred_tok = doc[0]
G = get_srl_digraph(uvi, pred_tok)
```


## 2.2
