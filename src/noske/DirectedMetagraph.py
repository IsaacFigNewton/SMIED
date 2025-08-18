import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Set

class DirectedMetagraph:
    def __init__(self, vert_list: List[Tuple] | None = None):
        if vert_list is None:
            self.metaverts = list()
        else:
            self.validate_graph(vert_list)
            self.metaverts = [self.canonicalize_vert(mv) for mv in vert_list]
    

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
        for node_id, node_data in enumerate(self.metaverts):
            G = self.add_vert_to_nx(G, node_id, node_data)

        return G
    

    def add_vert(self,
            vert: str|Tuple|List,
            data: Dict[str, Any]|None
        ):
        proposed_vert = (vert, data)
        self.validate_vert(len(self.metaverts), proposed_vert)
        proposed_vert = self.canonicalize_vert(proposed_vert)
        self.metaverts.append(proposed_vert)


    @classmethod
    def _remove_verts(cls,
            mv_ids: Set[int],
            current_mv_idx: int,
            metaverts: List[Tuple]
        ) -> List[Tuple]:
        # Base case
        if len(mv_ids) == 0 or len(metaverts) == 0:
            return metaverts

        # Remove all references to any mv ids in mv_idx
        current_mv = metaverts[0]
        # if the current mv is an atomic type, move to the next one
        if isinstance(current_mv, str):
            return [current_mv] + cls._remove_verts(
                mv_ids,
                current_mv_idx+1,
                metaverts[1:]
            )
        
        # if it's a complex type with none of the bad ids, move on
        if len(set(current_mv[0]).intersection(mv_ids)) == 0:
            return [current_mv] + cls._remove_verts(
                mv_ids,
                current_mv_idx+1,
                metaverts[1:]
            )
        
        # if it's a directed metaedge with >=1 bad ids, 
        if isinstance(current_mv[0], tuple) and len(current_mv[0]) == 2:
            # add it to the list of verts to remove
            mv_ids.add(current_mv_idx)
        
        # if it's an undirected metaedge with >=1 bad id, 
        if isinstance(current_mv[0], list):
            # get new contents without the problem ids
            clean_contents = list()
            for i in current_mv[0]:
                if i not in mv_ids:
                    clean_contents.append(i)
            # if it was dependent on exclusively invalid metaverts
            if len(clean_contents) == 0:
                # add it to the list of verts to remove
                mv_ids.add(current_mv_idx)
            # otherwise, just replace it and move on
            else:
                return [(clean_contents, current_mv[1])] + cls._remove_verts(
                    mv_ids,
                    current_mv_idx+1,
                    metaverts[1:]
                )
        
        # if it didn't return via some other path, the current_mv must have been invalid
        return cls._remove_verts(
            mv_ids,
            current_mv_idx+1,
            metaverts[1:]
        )
    

    def remove_vert(self, mv_idx: int):
        later_verts = self._remove_verts(
            {mv_idx},
            mv_idx+1,
            self.metaverts[mv_idx+1:]
        )
        self.metaverts = self.metaverts[:mv_idx] + later_verts