"""
Abstract base class for NLP token mocks.

This module provides the AbstractNLPTokenMock class that serves as a base
for NLP token mocks that simulate spaCy Token-like objects with linguistic
attributes like text, lemma, POS tags, dependencies, etc.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock
from typing import Any, Dict, List, Optional, Union, Iterator


class AbstractNLPTokenMock(ABC, Mock):
    """
    Abstract base class for NLP token mocks.
    
    This class provides a common interface for mocks that represent NLP tokens
    with linguistic attributes such as text, lemma, part-of-speech, dependencies,
    morphological features, and other token-level annotations.
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, text: str = "test", *args, **kwargs):
        """
        Initialize the AbstractNLPTokenMock.
        
        Args:
            text: The text content of the token
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes(text)
        self._setup_linguistic_attributes()
        self._setup_token_properties()
    
    def _setup_common_attributes(self, text: str):
        """Set up common token attributes."""
        # Core token content
        self.text = text
        self.text_ = text  # Alternative accessor
        self.orth = hash(text)  # Orthographic form ID
        self.orth_ = text
        
        # Token position and indexing
        self.i = 0  # Token index in document
        self.idx = 0  # Character offset in document
        
        # Basic token properties
        self.whitespace_ = " "  # Trailing whitespace
        self.shape_ = self._compute_shape(text)
        self.prefix_ = text[:3] if len(text) >= 3 else text
        self.suffix_ = text[-3:] if len(text) >= 3 else text
    
    def _setup_linguistic_attributes(self):
        """Set up linguistic annotation attributes."""
        # Morphological analysis
        self.lemma = hash(self.text.lower())  # Lemma ID
        self.lemma_ = self.text.lower()  # Lemma text
        
        # Part-of-speech tagging
        self.pos = 1  # POS tag ID
        self.pos_ = "NOUN"  # POS tag
        self.tag = 1  # Fine-grained POS tag ID
        self.tag_ = "NN"  # Fine-grained POS tag
        
        # Dependency parsing
        self.dep = 1  # Dependency relation ID
        self.dep_ = "ROOT"  # Dependency relation
        self.head = self  # Head token (self for root)
        
        # Morphological features
        self.morph = Mock()  # Morphological features
        
        # Named entities
        self.ent_type = 0  # Entity type ID
        self.ent_type_ = ""  # Entity type
        self.ent_iob = 0  # IOB tag ID
        self.ent_iob_ = "O"  # IOB tag
        self.ent_id = 0  # Entity ID
        self.ent_id_ = ""  # Entity ID text
    
    def _setup_token_properties(self):
        """Set up token property flags and methods."""
        # Character type flags
        self.is_alpha = self.text.isalpha()
        self.is_ascii = self.text.isascii()
        self.is_digit = self.text.isdigit()
        self.is_lower = self.text.islower()
        self.is_upper = self.text.isupper()
        self.is_title = self.text.istitle()
        self.is_space = self.text.isspace()
        self.is_punct = not self.text.isalnum() and not self.text.isspace()
        
        # Linguistic flags
        self.is_stop = self.text.lower() in {"the", "a", "an", "and", "or", "but", "in", "on", "at"}
        self.is_sent_start = False
        self.is_sent_end = False
        self.is_bracket = self.text in "()[]{}<>"
        self.is_quote = self.text in "\"'`"
        self.is_left_punct = self.text in "([{<"
        self.is_right_punct = self.text in ")]}>"
        self.is_currency = self.text in "$€£¥"
        
        # Token classification
        self.like_url = self._is_like_url(self.text)
        self.like_email = self._is_like_email(self.text)
        self.like_num = self._is_like_num(self.text)
        
        # Vector properties
        self.has_vector = False
        self.vector = Mock()
        self.vector_norm = 0.0
        
        # String operations
        self.__str__ = Mock(return_value=self.text)
        self.__repr__ = Mock(return_value=f"AbstractNLPTokenMock('{self.text}')")
        self.__len__ = Mock(return_value=len(self.text))
    
    def _compute_shape(self, text: str) -> str:
        """
        Compute the orthographic shape of the token.
        
        Args:
            text: Token text
            
        Returns:
            Shape string representing character patterns
        """
        shape = ""
        for char in text:
            if char.isupper():
                shape += "X"
            elif char.islower():
                shape += "x"
            elif char.isdigit():
                shape += "d"
            else:
                shape += char
        return shape
    
    def _is_like_url(self, text: str) -> bool:
        """Check if token looks like a URL."""
        return any(text.startswith(prefix) for prefix in ["http://", "https://", "www.", "ftp://"])
    
    def _is_like_email(self, text: str) -> bool:
        """Check if token looks like an email address."""
        return "@" in text and "." in text.split("@")[-1]
    
    def _is_like_num(self, text: str) -> bool:
        """Check if token looks like a number."""
        # Remove common number formatting characters
        cleaned = text.replace(",", "").replace(".", "").replace("-", "")
        return cleaned.isdigit() or any(char.isdigit() for char in text)
    
    @abstractmethod
    def get_linguistic_features(self) -> Dict[str, Any]:
        """
        Get linguistic features specific to this token type.
        
        Returns:
            Dictionary containing linguistic features
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def set_linguistic_attributes(self, **kwargs) -> None:
        """
        Set linguistic attributes for the token.
        
        Args:
            **kwargs: Linguistic attributes to set
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def validate_token_consistency(self) -> bool:
        """
        Validate that token attributes are consistent.
        
        Returns:
            True if token is consistent, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def set_position(self, token_index: int, char_offset: int) -> None:
        """
        Set token position in document.
        
        Args:
            token_index: Index of token in document token sequence
            char_offset: Character offset of token in document text
        """
        self.i = token_index
        self.idx = char_offset
    
    def set_head(self, head_token: 'AbstractNLPTokenMock') -> None:
        """
        Set the syntactic head of this token.
        
        Args:
            head_token: The head token in dependency relation
        """
        self.head = head_token
    
    def set_dependency(self, dep_label: str, head_token: Optional['AbstractNLPTokenMock'] = None) -> None:
        """
        Set dependency relation and head.
        
        Args:
            dep_label: Dependency relation label
            head_token: Head token (optional, defaults to self for root)
        """
        self.dep_ = dep_label
        self.dep = hash(dep_label)
        
        if head_token is not None:
            self.head = head_token
        elif dep_label.upper() == "ROOT":
            self.head = self
    
    def set_pos_tag(self, pos_tag: str, fine_pos_tag: Optional[str] = None) -> None:
        """
        Set part-of-speech tag.
        
        Args:
            pos_tag: Coarse-grained POS tag
            fine_pos_tag: Fine-grained POS tag (optional)
        """
        self.pos_ = pos_tag
        self.pos = hash(pos_tag)
        
        if fine_pos_tag:
            self.tag_ = fine_pos_tag
            self.tag = hash(fine_pos_tag)
    
    def set_lemma(self, lemma: str) -> None:
        """
        Set lemma form.
        
        Args:
            lemma: Lemmatized form of the token
        """
        self.lemma_ = lemma
        self.lemma = hash(lemma)
    
    def set_entity_annotation(self, 
                            ent_type: str = "", 
                            ent_iob: str = "O", 
                            ent_id: str = "") -> None:
        """
        Set named entity annotation.
        
        Args:
            ent_type: Entity type (e.g., PERSON, ORG, LOC)
            ent_iob: IOB tag (I, O, B)
            ent_id: Entity identifier
        """
        self.ent_type_ = ent_type
        self.ent_type = hash(ent_type) if ent_type else 0
        self.ent_iob_ = ent_iob
        self.ent_iob = hash(ent_iob)
        self.ent_id_ = ent_id
        self.ent_id = hash(ent_id) if ent_id else 0
    
    def set_sentence_boundaries(self, is_sent_start: bool = False, is_sent_end: bool = False) -> None:
        """
        Set sentence boundary flags.
        
        Args:
            is_sent_start: Whether token starts a sentence
            is_sent_end: Whether token ends a sentence
        """
        self.is_sent_start = is_sent_start
        self.is_sent_end = is_sent_end
    
    def get_children(self) -> List['AbstractNLPTokenMock']:
        """
        Get syntactic children of this token.
        
        Returns:
            List of child tokens in dependency tree
        """
        # Mock implementation - would need access to full document
        return []
    
    def get_ancestors(self) -> Iterator['AbstractNLPTokenMock']:
        """
        Get syntactic ancestors of this token.
        
        Yields:
            Ancestor tokens up to the root
        """
        current = self.head
        visited = {self}
        
        while current != self and current not in visited:
            yield current
            visited.add(current)
            current = getattr(current, 'head', self)
    
    def get_subtree(self) -> Iterator['AbstractNLPTokenMock']:
        """
        Get subtree rooted at this token.
        
        Yields:
            All tokens in the subtree
        """
        # Mock implementation - would need access to full document
        yield self
    
    def similarity(self, other: 'AbstractNLPTokenMock') -> float:
        """
        Calculate similarity with another token.
        
        Args:
            other: Another token to compare with
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not self.has_vector or not other.has_vector:
            # Simple string similarity
            if self.text == other.text:
                return 1.0
            if self.lemma_ == other.lemma_:
                return 0.8
            if self.pos_ == other.pos_:
                return 0.3
            return 0.0
        
        # Mock vector similarity
        return 0.8
    
    def is_ancestor(self, other: 'AbstractNLPTokenMock') -> bool:
        """
        Check if this token is an ancestor of another token.
        
        Args:
            other: Token to check
            
        Returns:
            True if this token is an ancestor of other
        """
        for ancestor in other.get_ancestors():
            if ancestor == self:
                return True
        return False
    
    def nbor(self, offset: int = 1) -> Optional['AbstractNLPTokenMock']:
        """
        Get neighboring token by offset.
        
        Args:
            offset: Offset from current token (default: 1 for next token)
            
        Returns:
            Neighboring token or None if out of bounds
        """
        # Mock implementation - would need access to document
        return Mock() if abs(offset) == 1 else None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert token to dictionary representation.
        
        Returns:
            Dictionary containing token attributes
        """
        return {
            'text': self.text,
            'lemma': self.lemma_,
            'pos': self.pos_,
            'tag': self.tag_,
            'dep': self.dep_,
            'shape': self.shape_,
            'is_alpha': self.is_alpha,
            'is_stop': self.is_stop,
            'is_punct': self.is_punct,
            'like_num': self.like_num,
            'like_url': self.like_url,
            'like_email': self.like_email,
            'ent_type': self.ent_type_,
            'ent_iob': self.ent_iob_,
            'index': self.i,
            'char_offset': self.idx
        }
    
    def matches_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Check if token matches a pattern.
        
        Args:
            pattern: Dictionary specifying pattern criteria
            
        Returns:
            True if token matches pattern, False otherwise
        """
        for attr, expected_value in pattern.items():
            if hasattr(self, attr):
                actual_value = getattr(self, attr)
                if actual_value != expected_value:
                    return False
            else:
                return False
        
        return True
    
    def create_variant(self, **modifications) -> 'AbstractNLPTokenMock':
        """
        Create a variant of this token with modifications.
        
        Args:
            **modifications: Attributes to modify
            
        Returns:
            New token instance with modifications
        """
        # Create new instance with same text
        variant = self.__class__(self.text)
        
        # Copy all attributes
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                try:
                    setattr(variant, attr, getattr(self, attr))
                except (AttributeError, TypeError):
                    pass  # Skip read-only attributes
        
        # Apply modifications
        for attr, value in modifications.items():
            if hasattr(variant, attr):
                setattr(variant, attr, value)
        
        return variant