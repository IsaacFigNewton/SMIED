import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Set

class DirectedMetagraph:
    def __init__(self, vert_list: List[Tuple] | None = None):
        self.current_mv_idx = 0
        self.metaverts = dict()
        # If metaverts are provided, initialize the metagraph
        if vert_list:
            self.validate_graph(vert_list)
            for mv in vert_list:
                self.metaverts[self.current_mv_idx] = self.canonicalize_vert(mv)
                self.current_mv_idx += 1
    

    @classmethod
    def validate_vert(cls, mv_idx: int, mv: Tuple):
        # Check length
        assert len(mv) in {1, 2}
        # Get type of core structure
        mv_type_0 = type(mv[0])
        # Ensure metavert metadata is in the correct format
        if len(mv) == 2:
            assert mv[1] is None or isinstance(mv[1], dict)

        # Check that tuple is in a valid format
        # If it's an atomic type 
        if mv_type_0 == str:
            pass

        # If it's a directed complex type/relation
        elif isinstance(mv[0], tuple) and len(mv[0]) == 2:
            # Make sure it's only referencing previously declared metaverts
            assert mv[0][0] < mv_idx
            assert mv[0][1] < mv_idx
            # Anything more complex than an atomic type must describe the inter-nodal relation
            assert len(mv) == 2
            assert "relation" in mv[1].keys()

        # If it's an undirected complex type/relation
        elif isinstance(mv[0], list):
            # Make sure it's only referencing previously declared metaverts
            assert all(id < mv_idx for id in mv[0])
            # Anything more complex than an atomic type must describe the inter-nodal relation
            assert len(mv) == 2
            assert "relation" in mv[1].keys()
        
        # If the first arg of the metavert definition is of an invalid type
        else:
            raise ValueError(f"First value of metavert {mv} is of invalid type {str(mv_type_0)}")


    @classmethod
    def validate_graph(cls, vert_list: List[Tuple]):
        for mv_idx, mv in enumerate(vert_list):
            cls.validate_vert(mv_idx, mv)


    @classmethod
    def canonicalize_vert(cls, mv: Tuple):
        # Get type of core structure
        mv_type_0 = type(mv[0])

        # If it's an atomic type 
        if mv_type_0 == str:
            return mv
        # If it's a directed complex type/relation
        elif isinstance(mv[0], tuple) and len(mv[0]) == 2:
            return mv
        # If it's an undirected complex type/relation
        elif isinstance(mv[0], list):
            # Canonical form has component verts in ascending order
            return (sorted(mv[0]), mv[1])


    @classmethod
    def add_vert_to_nx(cls,
            G: nx.DiGraph,
            node_id: int,
            node_data: Tuple[
                str|Tuple|List,\
                Dict[str, Any]|None
            ]
        ) -> nx.DiGraph:
        # Extract node attributes
        node_attributes = dict()
        if len(node_data) > 1 and node_data[1] is not None:
            node_attributes = {
                k: v
                for k, v in node_data[1].items()
                if k != 'relation'
            }

        # If it's an atomic type
        if isinstance(node_data[0], str):
            G.add_node(
                node_id,
                label=node_data[0]
            )

        # If it's a complex type/relation
        else:

            # If it's a directed metaedge
            if isinstance(node_data[0], tuple):
                subj_id, obj_id = node_data[0]
                label = f"{node_data[1]['relation']}({subj_id}, {obj_id})"
                G.add_node(node_id, label=label, **node_attributes)
                G.add_edge(subj_id, node_id, label="arg0")
                G.add_edge(obj_id, node_id, label="arg1")

            # If it's an undirected metaedge
            elif isinstance(node_data[0], list):
                label = f"{node_data[1]['relation']}({node_data[0]})"
                G.add_node(node_id, label=label, **node_attributes)
                for nested_node_id in node_data[0]:
                    G.add_edge(node_id, nested_node_id, label="componentOf")

        return G
    

    def to_nx(self):
        """
        Builds a networkx graph from a metagraph.

        Args:
            metaverts (dict): A dictionary representing the metagraph.

        Returns:
            networkx.Graph: The constructed graph.
        """
        G = nx.DiGraph()

        # Add nodes from the metagraph dictionary
        for node_id, node_data in self.metaverts.items():
            G = self.add_vert_to_nx(G, node_id, node_data)

        return G
    

    def add_vert(self,
            vert: str|Tuple|List,
            data: Dict[str, Any]|None
        ):
        proposed_vert = (vert, data)
        self.validate_vert(self.current_mv_idx, proposed_vert)
        proposed_vert = self.canonicalize_vert(proposed_vert)
        self.metaverts[self.current_mv_idx] = proposed_vert
        self.current_mv_idx += 1


    @classmethod
    def _remove_verts(cls,
            mv_ids: Set[int],
            current_mv_idx: int,
            metaverts: Dict[int, Tuple]
        ) -> Dict[int, Tuple]:
        # Base case
        if len(mv_ids) == 0 or len(metaverts) == 0:
            return metaverts

        # Process all remaining metaverts
        result = {}
        for mv_key, current_mv in metaverts.items():
            # Skip if this vertex is marked for removal
            if mv_key in mv_ids:
                continue
                
            # if the current mv is an atomic type, keep it
            if isinstance(current_mv[0], str):
                result[mv_key] = current_mv
                continue
            
            # if it's a complex type, check for bad references
            referenced_ids = set()
            if isinstance(current_mv[0], tuple) and len(current_mv[0]) == 2:
                referenced_ids = {current_mv[0][0], current_mv[0][1]}
            elif isinstance(current_mv[0], list):
                referenced_ids = set(current_mv[0])
                
            # if it's a directed metaedge with >=1 bad ids, skip it
            if isinstance(current_mv[0], tuple) and len(current_mv[0]) == 2:
                if len(referenced_ids.intersection(mv_ids)) > 0:
                    mv_ids.add(mv_key)  # Mark for removal in case other verts reference it
                    continue
                else:
                    result[mv_key] = current_mv
            
            # if it's an undirected metaedge with >=1 bad id
            elif isinstance(current_mv[0], list):
                # get new contents without the problem ids
                clean_contents = [i for i in current_mv[0] if i not in mv_ids]
                # if it was dependent on exclusively invalid metaverts, skip it
                if len(clean_contents) == 0:
                    mv_ids.add(mv_key)  # Mark for removal
                    continue
                # otherwise, keep it with cleaned contents
                else:
                    result[mv_key] = (clean_contents, current_mv[1])
            else:
                # Keep it if no issues
                result[mv_key] = current_mv
        
        return result
    

    def remove_vert(self, mv_idx: int):
        # Remove the specified vertex if it exists
        if mv_idx not in self.metaverts:
            return
        
        # Clean up all references to the removed vertex
        self.metaverts = self._remove_verts(
            {mv_idx},
            0,
            self.metaverts
        )