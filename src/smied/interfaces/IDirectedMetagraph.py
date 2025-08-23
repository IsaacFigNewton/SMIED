from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Set
import networkx as nx


class IDirectedMetagraph(ABC):
    """
    Interface for directed metagraph data structures.
    
    Defines the contract for creating, validating, and manipulating directed 
    metagraphs that can contain atomic vertices and complex relations between them.
    """
    
    @abstractmethod
    def __init__(self, vert_list: List[Tuple] | None = None):
        """
        Initialize the directed metagraph.
        
        Args:
            vert_list: Optional list of vertex tuples to initialize the graph with
        """
        pass
    
    @classmethod
    @abstractmethod
    def validate_vert(cls, mv_idx: int, mv: Tuple):
        """
        Validate a metavertex definition.
        
        Args:
            mv_idx: Index of the metavertex being validated
            mv: Metavertex tuple to validate
            
        Raises:
            AssertionError: If the metavertex is invalid
            ValueError: If the metavertex type is invalid
        """
        pass
    
    @classmethod
    @abstractmethod
    def validate_graph(cls, vert_list: List[Tuple]):
        """
        Validate an entire graph definition.
        
        Args:
            vert_list: List of metavertex tuples to validate
            
        Raises:
            AssertionError: If any metavertex in the graph is invalid
            ValueError: If any metavertex type is invalid
        """
        pass
    
    @classmethod
    @abstractmethod
    def canonicalize_vert(cls, mv: Tuple):
        """
        Convert a metavertex to its canonical form.
        
        Args:
            mv: Metavertex tuple to canonicalize
            
        Returns:
            Canonicalized metavertex tuple
        """
        pass
    
    @classmethod
    @abstractmethod
    def add_vert_to_nx(cls,
            G: nx.DiGraph,
            node_id: int,
            node_data: Tuple[
                str|Tuple|List,
                Dict[str, Any]|None
            ]
        ) -> nx.DiGraph:
        """
        Add a metavertex to a NetworkX directed graph.
        
        Args:
            G: NetworkX directed graph to modify
            node_id: Unique identifier for the node
            node_data: Metavertex data containing structure and metadata
            
        Returns:
            Modified NetworkX directed graph with the vertex added
        """
        pass
    
    @abstractmethod
    def to_nx(self):
        """
        Convert the metagraph to a NetworkX directed graph.
        
        Returns:
            NetworkX.DiGraph representation of the metagraph
        """
        pass
    
    @abstractmethod
    def add_vert(self,
            vert: str|Tuple|List,
            data: Dict[str, Any]|None
        ):
        """
        Add a new vertex to the metagraph.
        
        Args:
            vert: Vertex structure (string for atomic, tuple/list for complex)
            data: Optional metadata dictionary for the vertex
            
        Raises:
            AssertionError: If the vertex is invalid
            ValueError: If the vertex type is invalid
        """
        pass
    
    @classmethod
    @abstractmethod
    def _remove_verts(cls,
            mv_ids: Set[int],
            current_mv_idx: int,
            metaverts: Dict[int, Tuple]
        ) -> Dict[int, Tuple]:
        """
        Remove specified vertices and clean up references.
        
        Args:
            mv_ids: Set of vertex IDs to remove
            current_mv_idx: Current maximum vertex index
            metaverts: Dictionary of existing metavertices
            
        Returns:
            Cleaned dictionary with removed vertices and updated references
        """
        pass
    
    @abstractmethod
    def remove_vert(self, mv_idx: int):
        """
        Remove a vertex from the metagraph by its index.
        
        Args:
            mv_idx: Index of the vertex to remove
        """
        pass