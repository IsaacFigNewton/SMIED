"""
Abstract base class for collection-like mocks.

This module provides the AbstractCollectionMock class that serves as a base
for collection-like mocks including vocabularies, ontologies, knowledge bases,
and other container-like objects.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple, Set, Callable
from collections.abc import Collection, Sized, Iterable, Container


class AbstractCollectionMock(ABC, Mock):
    """
    Abstract base class for collection-like mocks.
    
    This class provides a common interface for mocks that represent collections
    of items with methods like __contains__, __len__, __iter__, keys(), etc.
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, initial_data: Optional[Any] = None, *args, **kwargs):
        """
        Initialize the AbstractCollectionMock.
        
        Args:
            initial_data: Initial data for the collection
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes()
        self._setup_collection_interface()
        self._initialize_data(initial_data)
    
    def _setup_common_attributes(self):
        """Set up common attributes for collections."""
        # Collection properties
        self.collection_type = "unknown"
        self.is_mutable = True
        self.is_ordered = False
        self.allows_duplicates = True
        self.is_indexed = False
        
        # Internal storage
        self._data = {}
        self._items = []
        self._keys = set()
        self._values = []
        
        # Metadata
        self.metadata = {}
        self.creation_time = Mock()  # Would be datetime in real implementation
        self.last_modified = Mock()
        self.version = "1.0"
        
        # Statistics
        self.access_count = 0
        self.modification_count = 0
        self.query_count = 0
    
    def _setup_collection_interface(self):
        """Set up collection interface methods."""
        # Core collection methods
        self.__contains__ = Mock(side_effect=self._contains_impl)
        self.__len__ = Mock(side_effect=self._len_impl)
        self.__iter__ = Mock(side_effect=self._iter_impl)
        self.__getitem__ = Mock(side_effect=self._getitem_impl)
        
        # Mutable collection methods (if applicable)
        if self.is_mutable:
            self.__setitem__ = Mock(side_effect=self._setitem_impl)
            self.__delitem__ = Mock(side_effect=self._delitem_impl)
            self.add = Mock(side_effect=self._add_impl)
            self.remove = Mock(side_effect=self._remove_impl)
            self.clear = Mock(side_effect=self._clear_impl)
            self.update = Mock(side_effect=self._update_impl)
        
        # Dictionary-like methods
        self.keys = Mock(side_effect=self._keys_impl)
        self.values = Mock(side_effect=self._values_impl)
        self.items = Mock(side_effect=self._items_impl)
        self.get = Mock(side_effect=self._get_impl)
        
        # Set-like methods
        self.union = Mock(side_effect=self._union_impl)
        self.intersection = Mock(side_effect=self._intersection_impl)
        self.difference = Mock(side_effect=self._difference_impl)
        
        # List-like methods (if ordered)
        if self.is_ordered:
            self.append = Mock(side_effect=self._append_impl)
            self.insert = Mock(side_effect=self._insert_impl)
            self.pop = Mock(side_effect=self._pop_impl)
            self.index = Mock(side_effect=self._index_impl)
    
    def _initialize_data(self, initial_data: Optional[Any]) -> None:
        """
        Initialize the collection with initial data.
        
        Args:
            initial_data: Initial data to populate the collection
        """
        if initial_data is None:
            return
        
        if isinstance(initial_data, dict):
            self._data = initial_data.copy()
            self._keys = set(initial_data.keys())
            self._values = list(initial_data.values())
        elif isinstance(initial_data, (list, tuple)):
            self._items = list(initial_data)
            if self.is_indexed:
                self._data = {i: item for i, item in enumerate(initial_data)}
                self._keys = set(range(len(initial_data)))
        elif isinstance(initial_data, set):
            self._keys = initial_data.copy()
            self._items = list(initial_data)
        else:
            # Try to iterate over initial_data
            try:
                self._items = list(initial_data)
                if self.is_indexed:
                    self._data = {i: item for i, item in enumerate(self._items)}
                    self._keys = set(range(len(self._items)))
            except TypeError:
                # Single item
                self._items = [initial_data]
                if self.is_indexed:
                    self._data = {0: initial_data}
                    self._keys = {0}
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about this collection.
        
        Returns:
            Dictionary containing collection-specific information
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def validate_item(self, item: Any) -> bool:
        """
        Validate that an item can be stored in this collection.
        
        Args:
            item: Item to validate
            
        Returns:
            True if item is valid, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def transform_item(self, item: Any) -> Any:
        """
        Transform an item before storing it in the collection.
        
        Args:
            item: Item to transform
            
        Returns:
            Transformed item
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def _contains_impl(self, item: Any) -> bool:
        """Implementation of __contains__ method."""
        self.access_count += 1
        
        if isinstance(self._data, dict):
            return item in self._data or item in self._data.values()
        else:
            return item in self._items
    
    def _len_impl(self) -> int:
        """Implementation of __len__ method."""
        if isinstance(self._data, dict):
            return len(self._data)
        else:
            return len(self._items)
    
    def _iter_impl(self) -> Iterator[Any]:
        """Implementation of __iter__ method."""
        self.access_count += 1
        
        if isinstance(self._data, dict):
            return iter(self._data.keys() if self.is_indexed else self._data.values())
        else:
            return iter(self._items)
    
    def _getitem_impl(self, key: Any) -> Any:
        """Implementation of __getitem__ method."""
        self.access_count += 1
        
        if isinstance(self._data, dict):
            return self._data[key]
        else:
            return self._items[key]
    
    def _setitem_impl(self, key: Any, value: Any) -> None:
        """Implementation of __setitem__ method."""
        if not self.is_mutable:
            raise TypeError("Collection is immutable")
        
        if not self.validate_item(value):
            raise ValueError("Invalid item for collection")
        
        transformed_value = self.transform_item(value)
        
        if isinstance(self._data, dict):
            self._data[key] = transformed_value
            self._keys.add(key)
        else:
            if isinstance(key, int):
                self._items[key] = transformed_value
        
        self.modification_count += 1
        self.last_modified = Mock()
    
    def _delitem_impl(self, key: Any) -> None:
        """Implementation of __delitem__ method."""
        if not self.is_mutable:
            raise TypeError("Collection is immutable")
        
        if isinstance(self._data, dict):
            del self._data[key]
            self._keys.discard(key)
        else:
            if isinstance(key, int):
                del self._items[key]
        
        self.modification_count += 1
        self.last_modified = Mock()
    
    def _add_impl(self, item: Any) -> None:
        """Implementation of add method."""
        if not self.is_mutable:
            raise TypeError("Collection is immutable")
        
        if not self.validate_item(item):
            raise ValueError("Invalid item for collection")
        
        transformed_item = self.transform_item(item)
        
        if not self.allows_duplicates and transformed_item in self:
            return  # Item already exists
        
        if isinstance(self._data, dict):
            # Generate key for the item
            key = len(self._data)
            self._data[key] = transformed_item
            self._keys.add(key)
        else:
            self._items.append(transformed_item)
        
        self.modification_count += 1
        self.last_modified = Mock()
    
    def _remove_impl(self, item: Any) -> None:
        """Implementation of remove method."""
        if not self.is_mutable:
            raise TypeError("Collection is immutable")
        
        if isinstance(self._data, dict):
            # Find key for the item
            key_to_remove = None
            for key, value in self._data.items():
                if value == item:
                    key_to_remove = key
                    break
            
            if key_to_remove is not None:
                del self._data[key_to_remove]
                self._keys.discard(key_to_remove)
            else:
                raise ValueError("Item not found in collection")
        else:
            self._items.remove(item)
        
        self.modification_count += 1
        self.last_modified = Mock()
    
    def _clear_impl(self) -> None:
        """Implementation of clear method."""
        if not self.is_mutable:
            raise TypeError("Collection is immutable")
        
        self._data.clear()
        self._items.clear()
        self._keys.clear()
        self._values.clear()
        
        self.modification_count += 1
        self.last_modified = Mock()
    
    def _update_impl(self, other: Any) -> None:
        """Implementation of update method."""
        if not self.is_mutable:
            raise TypeError("Collection is immutable")
        
        if hasattr(other, 'items'):
            # Dictionary-like
            for key, value in other.items():
                self._setitem_impl(key, value)
        else:
            # Iterable
            for item in other:
                self._add_impl(item)
    
    def _keys_impl(self) -> Iterator[Any]:
        """Implementation of keys method."""
        self.access_count += 1
        return iter(self._keys)
    
    def _values_impl(self) -> Iterator[Any]:
        """Implementation of values method."""
        self.access_count += 1
        
        if isinstance(self._data, dict):
            return iter(self._data.values())
        else:
            return iter(self._items)
    
    def _items_impl(self) -> Iterator[Tuple[Any, Any]]:
        """Implementation of items method."""
        self.access_count += 1
        
        if isinstance(self._data, dict):
            return iter(self._data.items())
        else:
            return iter(enumerate(self._items))
    
    def _get_impl(self, key: Any, default: Any = None) -> Any:
        """Implementation of get method."""
        self.access_count += 1
        
        try:
            return self._getitem_impl(key)
        except (KeyError, IndexError):
            return default
    
    def _union_impl(self, other: 'AbstractCollectionMock') -> 'AbstractCollectionMock':
        """Implementation of union method."""
        result = self.__class__()
        
        # Add all items from self
        for item in self:
            result._add_impl(item)
        
        # Add all items from other
        for item in other:
            result._add_impl(item)
        
        return result
    
    def _intersection_impl(self, other: 'AbstractCollectionMock') -> 'AbstractCollectionMock':
        """Implementation of intersection method."""
        result = self.__class__()
        
        for item in self:
            if item in other:
                result._add_impl(item)
        
        return result
    
    def _difference_impl(self, other: 'AbstractCollectionMock') -> 'AbstractCollectionMock':
        """Implementation of difference method."""
        result = self.__class__()
        
        for item in self:
            if item not in other:
                result._add_impl(item)
        
        return result
    
    def _append_impl(self, item: Any) -> None:
        """Implementation of append method for ordered collections."""
        if not self.is_ordered:
            raise TypeError("Collection is not ordered")
        
        self._add_impl(item)
    
    def _insert_impl(self, index: int, item: Any) -> None:
        """Implementation of insert method for ordered collections."""
        if not self.is_ordered or not self.is_mutable:
            raise TypeError("Collection does not support insertion")
        
        if not self.validate_item(item):
            raise ValueError("Invalid item for collection")
        
        transformed_item = self.transform_item(item)
        self._items.insert(index, transformed_item)
        
        self.modification_count += 1
        self.last_modified = Mock()
    
    def _pop_impl(self, index: int = -1) -> Any:
        """Implementation of pop method for ordered collections."""
        if not self.is_ordered or not self.is_mutable:
            raise TypeError("Collection does not support popping")
        
        if not self._items:
            raise IndexError("Pop from empty collection")
        
        item = self._items.pop(index)
        
        self.modification_count += 1
        self.last_modified = Mock()
        
        return item
    
    def _index_impl(self, item: Any) -> int:
        """Implementation of index method for ordered collections."""
        if not self.is_ordered:
            raise TypeError("Collection is not ordered")
        
        return self._items.index(item)
    
    def filter(self, predicate: Callable[[Any], bool]) -> 'AbstractCollectionMock':
        """
        Filter collection items based on a predicate.
        
        Args:
            predicate: Function that returns True for items to keep
            
        Returns:
            New collection with filtered items
        """
        result = self.__class__()
        
        for item in self:
            if predicate(item):
                result._add_impl(item)
        
        return result
    
    def map(self, func: Callable[[Any], Any]) -> 'AbstractCollectionMock':
        """
        Apply a function to all items in the collection.
        
        Args:
            func: Function to apply to each item
            
        Returns:
            New collection with transformed items
        """
        result = self.__class__()
        
        for item in self:
            result._add_impl(func(item))
        
        return result
    
    def find(self, predicate: Callable[[Any], bool]) -> Optional[Any]:
        """
        Find the first item matching a predicate.
        
        Args:
            predicate: Function that returns True for the target item
            
        Returns:
            First matching item, or None if not found
        """
        for item in self:
            if predicate(item):
                return item
        return None
    
    def count(self, item: Any) -> int:
        """
        Count occurrences of an item in the collection.
        
        Args:
            item: Item to count
            
        Returns:
            Number of occurrences
        """
        if not self.allows_duplicates:
            return 1 if item in self else 0
        
        return sum(1 for x in self if x == item)
    
    def sample(self, n: int = 1) -> List[Any]:
        """
        Get a random sample of items from the collection.
        
        Args:
            n: Number of items to sample
            
        Returns:
            List of sampled items
        """
        import random
        items = list(self)
        
        if n >= len(items):
            return items.copy()
        
        return random.sample(items, n)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary containing collection statistics
        """
        return {
            'collection_type': self.collection_type,
            'size': len(self),
            'is_empty': len(self) == 0,
            'is_mutable': self.is_mutable,
            'is_ordered': self.is_ordered,
            'allows_duplicates': self.allows_duplicates,
            'access_count': self.access_count,
            'modification_count': self.modification_count,
            'query_count': self.query_count,
            'creation_time': self.creation_time,
            'last_modified': self.last_modified,
            'version': self.version
        }
    
    def to_list(self) -> List[Any]:
        """
        Convert collection to a list.
        
        Returns:
            List representation of the collection
        """
        return list(self)
    
    def to_set(self) -> Set[Any]:
        """
        Convert collection to a set.
        
        Returns:
            Set representation of the collection
        """
        return set(self)
    
    def to_dict(self) -> Dict[Any, Any]:
        """
        Convert collection to a dictionary.
        
        Returns:
            Dictionary representation of the collection
        """
        if isinstance(self._data, dict):
            return self._data.copy()
        else:
            return {i: item for i, item in enumerate(self)}
    
    def is_subset_of(self, other: 'AbstractCollectionMock') -> bool:
        """
        Check if this collection is a subset of another.
        
        Args:
            other: Other collection to compare with
            
        Returns:
            True if this is a subset of other, False otherwise
        """
        return all(item in other for item in self)
    
    def is_superset_of(self, other: 'AbstractCollectionMock') -> bool:
        """
        Check if this collection is a superset of another.
        
        Args:
            other: Other collection to compare with
            
        Returns:
            True if this is a superset of other, False otherwise
        """
        return all(item in self for item in other)