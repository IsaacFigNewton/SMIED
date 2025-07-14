# Semantic Graph Matcher Patterns - JSON-based Configuration
from typing import Dict, List, Any, Union
import json
import networkx as nx
import matplotlib.pyplot as plt

def get_token_tags(t):
  tags = dict()

  # get token case
  if t.is_lower:
    tags["case"] = "lower"
  elif t.is_upper:
    tags["case"] = "upper"
  elif t.is_title:
    tags["case"] = "title"

  # get token type
  if t.is_currency:
    tags["type"] = "currency"
  elif t.like_url:
    tags["type"] = "url"
  elif t.like_email:
    tags["type"] = "email"
  elif t.is_alpha:
    tags["type"] = "word"
  elif t.like_num:
    tags["type"] = "num"
  elif t.is_space:
    tags["type"] = "whitespace"
  elif t.is_punct:
    tags["type"] = "punct"
    if t.is_left_punct:
      tags["subtype_features"] = ["left"]
    elif t.is_right_punct:
      tags["subtype_features"] = ["right"]
    if t.is_bracket:
      tags["subtype_features"].append("bracket")
    elif t.is_quote:
      tags["subtype_features"].append("quote")
  
  # get morphologic analysis as lists of tags
  morph_dict = t.morph.to_dict()
  morph_dict = {k: v.split(",") for k, v in morph_dict.items()}
  tags.update(morph_dict)
  return tags


def get_dep_edges(t):
  edge_list = list()
  for child in t.lefts:
    edge_list.append((
        t.i,
        child.i,
        {"type": child.dep_, "rel_pos": "after"}
    ))
  for child in t.rights:
    edge_list.append((
        t.i,
        child.i,
        {"type": child.dep_, "rel_pos": "before"}
    ))
  return edge_list


def get_dep_graph(doc):
  g = nx.DiGraph()

  # add tokens and their relations
  for t in doc:
    # define and add the token's node to G
    node_dict = {
        "text": t.text,
        "pos": t.pos_,
        "head": t.head.i,
        "lemma": t.lemma_
    }
    node_dict.update(get_token_tags(t))
    g.add_nodes_from([(t.i, node_dict)])

    # add token NER relations
    if t.ent_type_:
      g.add_nodes_from([
        (t.ent_type_, {"text": t.ent_type_})
      ])
      g.add_edges_from([
          (t.i, t.ent_type_, {"type":"type"}),
      ])
    # add dependency relations
    g.add_edges_from(get_dep_edges(t))
  
  return g

# Enhanced helper functions with additional matching capabilities
def node_matches(node_attrs, pattern_attrs):
    """Enhanced node matching with additional pattern types"""
    for k, v in pattern_attrs.items():
        if k == "node_id_pattern":
            # Special pattern for matching node IDs with prefixes
            node_id = node_attrs.get("node_id", str(node_attrs.get("id", "")))
            if not node_id.startswith(v):
                return False
        elif k == "semantic_type":
            # Match semantic types (for domain-specific patterns)
            if isinstance(v, (list, set)):
                if node_attrs.get(k) not in v:
                    return False
            else:
                if node_attrs.get(k) != v:
                    return False
        elif k == "lemma":
            # Match lemmatized forms
            if isinstance(v, (list, set)):
                if node_attrs.get(k) not in v:
                    return False
            else:
                if node_attrs.get(k) != v:
                    return False
        elif isinstance(v, (list, set)):
            if node_attrs.get(k) not in v:
                return False
        else:
            if node_attrs.get(k) != v:
                return False
    return True

def edge_matches(edge_attrs, pattern_attrs):
    """Enhanced edge matching"""
    for k, v in pattern_attrs.items():
        if isinstance(v, (list, set)):
            if edge_attrs.get(k) not in v:
                return False
        else:
            if edge_attrs.get(k) != v:
                return False
    return True

def match_chain(G: nx.DiGraph, query: list[dict]) -> list[list]:
    """
    Find all paths in G matching the alternating node/edge attribute requirements
    Enhanced to work with JSON-loaded patterns
    """
    # Verify that query length is odd and > 0
    if len(query) < 1 or len(query) % 2 == 0:
        raise ValueError("Query must be non-empty and have odd length (node, edge, node, ...).")

    # Extract pattern requirements
    n = (len(query) + 1) // 2  # number of nodes in pattern
    node_patterns = [query[2*i] for i in range(n)]
    edge_patterns = [query[2*i + 1] for i in range(n-1)]

    # Find all matching paths using DFS
    results = []
    
    def dfs(current_path, pattern_idx):
        if pattern_idx == n:
            # We've matched all nodes in the pattern
            results.append(current_path[:])
            return
        
        if pattern_idx == 0:
            # First node - try all nodes in the graph
            for node in G.nodes():
                node_data = G.nodes[node].copy()
                # Add node_id for pattern matching
                node_data["node_id"] = str(node)
                if node_matches(node_data, node_patterns[0]):
                    current_path.append(node)
                    dfs(current_path, 1)
                    current_path.pop()
        else:
            # Not the first node - look for neighbors of the last node
            last_node = current_path[-1]
            edge_pattern = edge_patterns[pattern_idx - 1]
            node_pattern = node_patterns[pattern_idx]
            
            # Check all outgoing edges from the last node
            for neighbor in G.neighbors(last_node):
                # Check if the edge matches the pattern
                edge_data = G.edges[last_node, neighbor]
                if edge_matches(edge_data, edge_pattern):
                    # Check if the neighbor node matches the pattern
                    neighbor_data = G.nodes[neighbor].copy()
                    neighbor_data["node_id"] = str(neighbor)
                    if node_matches(neighbor_data, node_pattern):
                        current_path.append(neighbor)
                        dfs(current_path, pattern_idx + 1)
                        current_path.pop()
    
    dfs([], 0)
    return results

def kg_to_json(g):
  nodes = json.dumps(list(G.nodes(data=True)), indent=2)
  edges = json.dumps(list(G.edges(data=True)), indent=2)
  return nodes, edges

def plot_kg(g):
  # Extract node labels and edge labels
  node_labels = {node: g.nodes[node]['text'] for node in g.nodes() if 'text' in g.nodes[node]}
  edge_labels = {(u, v): d['type'] for u, v, d in g.edges(data=True)}

  # Position nodes using spring layout
  pos = nx.spring_layout(G, k=50)

  # Draw the graph
  plt.figure(figsize=(12, 8))
  nx.draw(
      g,
      pos,
      with_labels=True,
      labels=node_labels,
      node_size=1500,
      node_color="skyblue",
      alpha=0.8,
      linewidths=2,
      edge_color="gray"
  )
  nx.draw_networkx_edge_labels(
      g,
      pos,
      edge_labels=edge_labels,
      font_color='red'
  )
  plt.title("Semantic Knowledge Graph")
  plt.axis("off")
  plt.show()