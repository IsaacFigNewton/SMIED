from typing import Dict, List, Any

from hypergraphx.linalg import *
from hypergraphx.generation.random import *

def _is_edge(edge: Any) -> bool:
    return isinstance(edge, tuple) and len(edge) > 1

def _flatten_edge(edge: Any) -> List[Any]:
    flattened_edge = list()
    for e in edge:
        if not _is_edge(e):
            flattened_edge.append(e)
        else:
            flattened_edge = flattened_edge + _flatten_edge(e)
    return flattened_edge

def _get_required_node_fields(id: int | str | tuple,
                                node_type: str) -> Dict[Any, Any]:
    # regular vertex ids:   v1
    # metavertex ids:       (v1, v2)
    #                       (v1, v2, v3, ...)
    #                       ((v11, ...), (v21, ...))
    #                       (((v11, ...), (v21, ...)), v31)
    #                       (((v11, ...), (v21, ...)), (v31, ...))
    #                       (((v11, ...), (v21, ...)), (v31, ...), (v41, ...)))
    #                       ...
    valid_types = {"regular", "meta"}
    if node_type not in valid_types:
        raise ValueError(f"Invalid node type: {node_type}. Valid types: {valid_types}")
    return {
        "id": id,
        "type": node_type
    }

def _get_required_edge_fields(id: tuple,
                                edge_type: str) -> Dict[Any, Any]:
    # regular edge ids:     (v1, v2)
    # hyperedge ids:        (v1, v2, v3, ...)
    #                       ((v11, ...), (v21, ...))
    # metaedge ids:         (((v11, ...), (v21, ...)), v31)
    #                       (((v11, ...), (v21, ...)), (v31, ...))
    #                       (((v11, ...), (v21, ...)), (v31, ...), (v41, ...)))
    #                       ...
    valid_types = {"regular", "hyper", "meta", "metavert_to_hye", "hye_to_metavert"}
    if edge_type not in valid_types:
        raise ValueError(f"Invalid edge type: {edge_type}. Valid types: {valid_types}")
    return {
        "id": id,
        "type": edge_type
    }