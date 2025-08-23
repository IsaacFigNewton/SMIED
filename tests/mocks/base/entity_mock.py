"""
Abstract base class for domain entity mocks.

This module provides the AbstractEntityMock class that serves as a base
for domain entity mocks including lemmas, concepts, relations, synsets,
and other domain-specific objects with simple attributes.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class EntityType(Enum):
    """Enumeration of entity types."""
    CONCEPT = "concept"
    RELATION = "relation"
    LEMMA = "lemma" 
    SYNSET = "synset"
    CLUSTER = "cluster"
    PATH = "path"
    NODE = "node"
    EDGE = "edge"
    UNKNOWN = "unknown"


class EntityStatus(Enum):
    """Enumeration of entity statuses."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    UNDER_REVIEW = "under_review"


class AbstractEntityMock(ABC, Mock):
    """
    Abstract base class for domain entity mocks.
    
    This class provides a common interface for mocks that represent domain
    entities with simple attributes and properties. These are typically
    used for concepts, relations, lemmas, synsets, and other domain objects.
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, 
                 entity_id: Optional[str] = None,
                 entity_type: EntityType = EntityType.UNKNOWN,
                 *args, **kwargs):
        """
        Initialize the AbstractEntityMock.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes(entity_id, entity_type)
        self._setup_entity_properties()
        self._setup_relationship_management()
    
    def _setup_common_attributes(self, entity_id: Optional[str], entity_type: EntityType):
        """Set up common entity attributes."""
        # Core identification
        self.id = entity_id or self._generate_entity_id()
        self.entity_type = entity_type
        self.name = ""
        self.label = ""
        
        # Metadata
        self.description = ""
        self.definition = ""
        self.version = "1.0"
        self.status = EntityStatus.ACTIVE
        
        # Timestamps and tracking
        self.created_at = Mock()  # Would be datetime in real implementation
        self.updated_at = Mock()
        self.accessed_at = Mock()
        self.access_count = 0
        
        # Source and provenance
        self.source = ""
        self.source_id = ""
        self.confidence_score = 1.0
        self.provenance = {}
        
        # Classification
        self.category = ""
        self.domain = ""
        self.language = "en"
        self.tags = set()
    
    def _setup_entity_properties(self):
        """Set up entity-specific properties."""
        # Core properties
        self.properties = {}
        self.attributes = {}
        self.features = {}
        self.metadata = {}
        
        # Computed properties
        self.computed_properties = {}
        self.derived_features = {}
        
        # Validation and constraints
        self.constraints = {}
        self.validation_rules = []
        self.is_valid = True
        self.validation_errors = []
        
        # Statistics
        self.usage_count = 0
        self.reference_count = 0
        self.similarity_scores = {}
        
        # Caching
        self._property_cache = {}
        self._cache_valid = False
    
    def _setup_relationship_management(self):
        """Set up relationship management."""
        # Direct relationships
        self.parents = set()
        self.children = set()
        self.siblings = set()
        self.related_entities = set()
        
        # Relationship types
        self.incoming_relations = {}  # relation_type -> set of entities
        self.outgoing_relations = {}  # relation_type -> set of entities
        
        # Relationship properties
        self.relationship_weights = {}
        self.relationship_metadata = {}
        
        # Graph properties
        self.depth = 0
        self.path_to_root = []
        self.descendants_count = 0
    
    def _generate_entity_id(self) -> str:
        """Generate a unique entity ID."""
        import uuid
        return f"entity_{uuid.uuid4().hex[:8]}"
    
    @abstractmethod
    def get_primary_attribute(self) -> Any:
        """
        Get the primary attribute that identifies this entity.
        
        Returns:
            Primary identifying attribute
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def validate_entity(self) -> bool:
        """
        Validate that the entity is consistent and valid.
        
        Returns:
            True if entity is valid, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def get_entity_signature(self) -> str:
        """
        Get a unique signature for this entity.
        
        Returns:
            String signature that uniquely identifies the entity
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def set_property(self, name: str, value: Any, computed: bool = False) -> None:
        """
        Set a property on the entity.
        
        Args:
            name: Property name
            value: Property value
            computed: Whether this is a computed property
        """
        if computed:
            self.computed_properties[name] = value
        else:
            self.properties[name] = value
        
        # Invalidate cache
        self._cache_valid = False
        self.updated_at = Mock()
    
    def get_property(self, name: str, default: Any = None, use_cache: bool = True) -> Any:
        """
        Get a property value.
        
        Args:
            name: Property name
            default: Default value if property not found
            use_cache: Whether to use cached values
            
        Returns:
            Property value or default
        """
        self.access_count += 1
        self.accessed_at = Mock()
        
        # Check cache first
        if use_cache and self._cache_valid and name in self._property_cache:
            return self._property_cache[name]
        
        # Look in computed properties first
        if name in self.computed_properties:
            value = self.computed_properties[name]
        elif name in self.properties:
            value = self.properties[name]
        elif name in self.attributes:
            value = self.attributes[name]
        else:
            value = default
        
        # Cache the result
        if use_cache:
            self._property_cache[name] = value
        
        return value
    
    def has_property(self, name: str) -> bool:
        """
        Check if entity has a specific property.
        
        Args:
            name: Property name
            
        Returns:
            True if property exists, False otherwise
        """
        return (name in self.properties or 
                name in self.computed_properties or 
                name in self.attributes)
    
    def remove_property(self, name: str) -> bool:
        """
        Remove a property from the entity.
        
        Args:
            name: Property name
            
        Returns:
            True if property was removed, False if not found
        """
        removed = False
        
        if name in self.properties:
            del self.properties[name]
            removed = True
        
        if name in self.computed_properties:
            del self.computed_properties[name]
            removed = True
        
        if name in self.attributes:
            del self.attributes[name]
            removed = True
        
        if removed:
            self._cache_valid = False
            self.updated_at = Mock()
        
        return removed
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the entity.
        
        Args:
            tag: Tag to add
        """
        self.tags.add(tag.lower().strip())
        self.updated_at = Mock()
    
    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag from the entity.
        
        Args:
            tag: Tag to remove
            
        Returns:
            True if tag was removed, False if not found
        """
        tag_normalized = tag.lower().strip()
        if tag_normalized in self.tags:
            self.tags.remove(tag_normalized)
            self.updated_at = Mock()
            return True
        return False
    
    def has_tag(self, tag: str) -> bool:
        """
        Check if entity has a specific tag.
        
        Args:
            tag: Tag to check
            
        Returns:
            True if entity has the tag, False otherwise
        """
        return tag.lower().strip() in self.tags
    
    def add_relationship(self, 
                        relation_type: str, 
                        target_entity: 'AbstractEntityMock',
                        weight: float = 1.0,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a relationship to another entity.
        
        Args:
            relation_type: Type of relationship
            target_entity: Target entity
            weight: Relationship weight
            metadata: Additional relationship metadata
        """
        # Add to outgoing relations
        if relation_type not in self.outgoing_relations:
            self.outgoing_relations[relation_type] = set()
        self.outgoing_relations[relation_type].add(target_entity)
        
        # Add to target's incoming relations
        if relation_type not in target_entity.incoming_relations:
            target_entity.incoming_relations[relation_type] = set()
        target_entity.incoming_relations[relation_type].add(self)
        
        # Store weight and metadata
        rel_key = (relation_type, target_entity.id)
        self.relationship_weights[rel_key] = weight
        if metadata:
            self.relationship_metadata[rel_key] = metadata
        
        # Update general relationship sets
        self.related_entities.add(target_entity)
        target_entity.related_entities.add(self)
        
        self.updated_at = Mock()
    
    def remove_relationship(self, 
                          relation_type: str, 
                          target_entity: 'AbstractEntityMock') -> bool:
        """
        Remove a relationship to another entity.
        
        Args:
            relation_type: Type of relationship
            target_entity: Target entity
            
        Returns:
            True if relationship was removed, False if not found
        """
        removed = False
        
        # Remove from outgoing relations
        if (relation_type in self.outgoing_relations and 
            target_entity in self.outgoing_relations[relation_type]):
            self.outgoing_relations[relation_type].remove(target_entity)
            removed = True
        
        # Remove from target's incoming relations
        if (relation_type in target_entity.incoming_relations and 
            self in target_entity.incoming_relations[relation_type]):
            target_entity.incoming_relations[relation_type].remove(self)
            removed = True
        
        # Clean up weight and metadata
        rel_key = (relation_type, target_entity.id)
        if rel_key in self.relationship_weights:
            del self.relationship_weights[rel_key]
        if rel_key in self.relationship_metadata:
            del self.relationship_metadata[rel_key]
        
        if removed:
            self.updated_at = Mock()
        
        return removed
    
    def get_related_entities(self, 
                           relation_type: Optional[str] = None,
                           direction: str = "outgoing") -> Set['AbstractEntityMock']:
        """
        Get related entities by relationship type and direction.
        
        Args:
            relation_type: Type of relationship (None for all types)
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            Set of related entities
        """
        entities = set()
        
        if direction in ("outgoing", "both"):
            if relation_type:
                entities.update(self.outgoing_relations.get(relation_type, set()))
            else:
                for entity_set in self.outgoing_relations.values():
                    entities.update(entity_set)
        
        if direction in ("incoming", "both"):
            if relation_type:
                entities.update(self.incoming_relations.get(relation_type, set()))
            else:
                for entity_set in self.incoming_relations.values():
                    entities.update(entity_set)
        
        return entities
    
    def get_relationship_weight(self, 
                              relation_type: str, 
                              target_entity: 'AbstractEntityMock') -> float:
        """
        Get the weight of a relationship.
        
        Args:
            relation_type: Type of relationship
            target_entity: Target entity
            
        Returns:
            Relationship weight (1.0 if not found)
        """
        rel_key = (relation_type, target_entity.id)
        return self.relationship_weights.get(rel_key, 1.0)
    
    def calculate_similarity(self, other: 'AbstractEntityMock') -> float:
        """
        Calculate similarity with another entity.
        
        Args:
            other: Other entity to compare with
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple similarity based on shared properties and relationships
        similarity = 0.0
        factors = 0
        
        # Entity type similarity
        if self.entity_type == other.entity_type:
            similarity += 0.2
        factors += 1
        
        # Property similarity
        self_props = set(self.properties.keys())
        other_props = set(other.properties.keys())
        if self_props or other_props:
            prop_similarity = len(self_props & other_props) / len(self_props | other_props)
            similarity += prop_similarity * 0.3
        factors += 1
        
        # Tag similarity
        if self.tags or other.tags:
            tag_similarity = len(self.tags & other.tags) / len(self.tags | other.tags)
            similarity += tag_similarity * 0.2
        factors += 1
        
        # Relationship similarity
        self_related = self.related_entities
        other_related = other.related_entities
        if self_related or other_related:
            rel_similarity = len(self_related & other_related) / len(self_related | other_related)
            similarity += rel_similarity * 0.3
        factors += 1
        
        return similarity / factors if factors > 0 else 0.0
    
    def clone(self, new_id: Optional[str] = None) -> 'AbstractEntityMock':
        """
        Create a clone of this entity.
        
        Args:
            new_id: ID for the new entity (generated if None)
            
        Returns:
            Cloned entity
        """
        # Create new instance of same class
        cloned = self.__class__(entity_id=new_id, entity_type=self.entity_type)
        
        # Copy basic attributes
        cloned.name = self.name
        cloned.label = self.label
        cloned.description = self.description
        cloned.definition = self.definition
        cloned.status = self.status
        cloned.category = self.category
        cloned.domain = self.domain
        cloned.language = self.language
        
        # Copy collections (deep copy)
        cloned.properties = self.properties.copy()
        cloned.attributes = self.attributes.copy()
        cloned.features = self.features.copy()
        cloned.metadata = self.metadata.copy()
        cloned.tags = self.tags.copy()
        
        # Note: Relationships are not copied to avoid circular references
        
        return cloned
    
    def to_dict(self, include_relationships: bool = False) -> Dict[str, Any]:
        """
        Convert entity to dictionary representation.
        
        Args:
            include_relationships: Whether to include relationship information
            
        Returns:
            Dictionary representation of the entity
        """
        entity_dict = {
            'id': self.id,
            'entity_type': self.entity_type.value,
            'name': self.name,
            'label': self.label,
            'description': self.description,
            'definition': self.definition,
            'version': self.version,
            'status': self.status.value,
            'category': self.category,
            'domain': self.domain,
            'language': self.language,
            'tags': list(self.tags),
            'properties': self.properties.copy(),
            'attributes': self.attributes.copy(),
            'features': self.features.copy(),
            'metadata': self.metadata.copy(),
            'confidence_score': self.confidence_score,
            'access_count': self.access_count,
            'usage_count': self.usage_count
        }
        
        if include_relationships:
            entity_dict['relationships'] = {
                'outgoing': {
                    rel_type: [entity.id for entity in entities]
                    for rel_type, entities in self.outgoing_relations.items()
                },
                'incoming': {
                    rel_type: [entity.id for entity in entities]
                    for rel_type, entities in self.incoming_relations.items()
                }
            }
        
        return entity_dict
    
    def matches_criteria(self, criteria: Dict[str, Any]) -> bool:
        """
        Check if entity matches the given criteria.
        
        Args:
            criteria: Dictionary of criteria to match against
            
        Returns:
            True if entity matches all criteria, False otherwise
        """
        for key, value in criteria.items():
            if key == 'entity_type':
                if self.entity_type.value != value:
                    return False
            elif key == 'status':
                if self.status.value != value:
                    return False
            elif key == 'tags':
                required_tags = set(value) if isinstance(value, list) else {value}
                if not required_tags.issubset(self.tags):
                    return False
            elif key == 'has_property':
                if not self.has_property(value):
                    return False
            elif key == 'property_value':
                prop_name, expected_value = value
                if self.get_property(prop_name) != expected_value:
                    return False
            elif hasattr(self, key):
                if getattr(self, key) != value:
                    return False
            else:
                # Check in properties
                if self.get_property(key) != value:
                    return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the entity.
        
        Returns:
            Dictionary containing entity summary
        """
        return {
            'id': self.id,
            'type': self.entity_type.value,
            'name': self.name,
            'status': self.status.value,
            'property_count': len(self.properties),
            'relationship_count': len(self.related_entities),
            'tag_count': len(self.tags),
            'access_count': self.access_count,
            'confidence': self.confidence_score
        }