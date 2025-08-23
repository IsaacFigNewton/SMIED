"""
Abstract base class for NLP document mocks.

This module provides the AbstractNLPDocMock class that serves as a base
for NLP document mocks that simulate spaCy Doc-like objects with text, tokens,
entities, and other NLP analysis results.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple


class AbstractNLPDocMock(ABC, Mock):
    """
    Abstract base class for NLP document mocks.
    
    This class provides a common interface for mocks that represent NLP documents
    with text, tokens, entities, noun chunks, and other linguistic annotations.
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, text: str = "test document", *args, **kwargs):
        """
        Initialize the AbstractNLPDocMock.
        
        Args:
            text: The text content of the document
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes(text)
        self._setup_linguistic_annotations()
        self._setup_document_properties()
    
    def _setup_common_attributes(self, text: str):
        """Set up common attributes for NLP documents."""
        # Core document content
        self.text = text
        self.text_ = text  # Alternative accessor
        
        # Document structure
        self.tokens = []
        self.ents = []
        self.noun_chunks = []
        self.sents = []
        
        # Document metadata
        self.lang = "en"
        self.lang_ = "en"
        self.vocab = Mock()
        self.is_parsed = True
        self.is_tagged = True
        self.is_sentenced = True
        
        # Processing flags
        self.has_vector = False
        self.is_nered = True  # Named entity recognition performed
        self.tensor = None
    
    def _setup_linguistic_annotations(self):
        """Set up linguistic annotation containers."""
        # Token-level annotations
        self._token_annotations = {}
        self._pos_tags = []
        self._lemmas = []
        self._dependencies = []
        
        # Entity annotations
        self._entity_annotations = {}
        self._entity_labels = []
        self._entity_spans = []
        
        # Sentence boundaries
        self._sentence_boundaries = []
        self._sentence_spans = []
        
        # Chunk annotations
        self._noun_chunk_spans = []
        self._chunk_labels = []
    
    def _setup_document_properties(self):
        """Set up document-level properties and methods."""
        # Vector operations (if available)
        self.vector = Mock()
        self.similarity = Mock(return_value=0.8)
        
        # Document statistics
        self.__len__ = Mock(return_value=len(self.text.split()))
        self.count_by = Mock(return_value=Mock())
        
        # Iteration support
        self.__iter__ = Mock(return_value=iter(self.tokens))
        self.__getitem__ = Mock()
        
        # String representation
        self.__str__ = Mock(return_value=self.text)
        self.__repr__ = Mock(return_value=f"AbstractNLPDocMock('{self.text[:50]}...')")
    
    @abstractmethod
    def create_tokens(self) -> List[Any]:
        """
        Create token objects for the document.
        
        Returns:
            List of token mock objects
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def create_entities(self) -> List[Any]:
        """
        Create named entity objects for the document.
        
        Returns:
            List of entity mock objects
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def create_sentences(self) -> List[Any]:
        """
        Create sentence span objects for the document.
        
        Returns:
            List of sentence mock objects
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def setup_basic_tokenization(self) -> None:
        """Set up basic tokenization based on whitespace."""
        words = self.text.split()
        self.tokens = []
        
        for i, word in enumerate(words):
            token_mock = Mock()
            token_mock.text = word
            token_mock.text_ = word
            token_mock.i = i
            token_mock.idx = self.text.index(word, sum(len(w) + 1 for w in words[:i]))
            token_mock.lemma_ = word.lower()
            token_mock.pos_ = "NOUN"  # Default POS tag
            token_mock.tag_ = "NN"
            token_mock.dep_ = "ROOT" if i == 0 else "nmod"
            token_mock.is_alpha = word.isalpha()
            token_mock.is_stop = word.lower() in {"the", "a", "an", "and", "or", "but"}
            token_mock.is_punct = not word.isalnum()
            token_mock.like_num = word.isdigit()
            
            self.tokens.append(token_mock)
        
        # Update __len__ to reflect actual token count
        self.__len__ = Mock(return_value=len(self.tokens))
    
    def setup_basic_entities(self, entity_patterns: Optional[List[Tuple[str, str]]] = None) -> None:
        """
        Set up basic named entity recognition.
        
        Args:
            entity_patterns: List of (text, label) tuples for entities
        """
        if entity_patterns is None:
            entity_patterns = []
            # Auto-detect some basic entity patterns
            words = self.text.split()
            for word in words:
                if word.istitle() and len(word) > 1:
                    entity_patterns.append((word, "PERSON"))
        
        self.ents = []
        for text, label in entity_patterns:
            if text in self.text:
                entity_mock = Mock()
                entity_mock.text = text
                entity_mock.text_ = text
                entity_mock.label_ = label
                entity_mock.label = Mock(return_value=label)
                entity_mock.start = self.text.index(text) // len(self.text.split()[0]) if self.text.split() else 0
                entity_mock.end = entity_mock.start + len(text.split())
                entity_mock.start_char = self.text.index(text)
                entity_mock.end_char = entity_mock.start_char + len(text)
                
                self.ents.append(entity_mock)
    
    def setup_basic_sentences(self) -> None:
        """Set up basic sentence segmentation."""
        # Simple sentence splitting on periods, exclamation marks, and question marks
        import re
        sentence_texts = re.split(r'[.!?]+', self.text)
        sentence_texts = [s.strip() for s in sentence_texts if s.strip()]
        
        self.sents = []
        char_offset = 0
        
        for i, sent_text in enumerate(sentence_texts):
            sent_mock = Mock()
            sent_mock.text = sent_text
            sent_mock.text_ = sent_text
            sent_mock.start_char = self.text.find(sent_text, char_offset)
            sent_mock.end_char = sent_mock.start_char + len(sent_text)
            sent_mock.root = Mock()  # Root token of the sentence
            
            # Mock sentence-level tokens
            sent_tokens = [t for t in self.tokens if t.text in sent_text]
            sent_mock.__iter__ = Mock(return_value=iter(sent_tokens))
            sent_mock.__len__ = Mock(return_value=len(sent_tokens))
            
            self.sents.append(sent_mock)
            char_offset = sent_mock.end_char
    
    def setup_basic_noun_chunks(self) -> None:
        """Set up basic noun chunk detection."""
        # Simple noun chunk detection based on POS patterns
        self.noun_chunks = []
        
        # Group consecutive noun-like tokens
        current_chunk = []
        for token in self.tokens:
            if hasattr(token, 'pos_') and token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                current_chunk.append(token)
            else:
                if current_chunk:
                    chunk_mock = self._create_noun_chunk_mock(current_chunk)
                    self.noun_chunks.append(chunk_mock)
                    current_chunk = []
        
        # Add final chunk if exists
        if current_chunk:
            chunk_mock = self._create_noun_chunk_mock(current_chunk)
            self.noun_chunks.append(chunk_mock)
    
    def _create_noun_chunk_mock(self, tokens: List[Any]) -> Mock:
        """
        Create a mock noun chunk from a list of tokens.
        
        Args:
            tokens: List of token mocks
            
        Returns:
            Mock object representing a noun chunk
        """
        chunk_mock = Mock()
        chunk_mock.text = ' '.join(token.text for token in tokens)
        chunk_mock.text_ = chunk_mock.text
        chunk_mock.start = tokens[0].i if tokens else 0
        chunk_mock.end = tokens[-1].i + 1 if tokens else 0
        chunk_mock.root = tokens[-1] if tokens else Mock()  # Last token as root
        chunk_mock.label_ = "NP"  # Noun phrase
        chunk_mock.__iter__ = Mock(return_value=iter(tokens))
        chunk_mock.__len__ = Mock(return_value=len(tokens))
        
        return chunk_mock
    
    def get_token_by_index(self, index: int) -> Optional[Any]:
        """
        Get token by index.
        
        Args:
            index: Token index
            
        Returns:
            Token mock at the specified index, or None if out of bounds
        """
        if 0 <= index < len(self.tokens):
            return self.tokens[index]
        return None
    
    def get_tokens_in_range(self, start: int, end: int) -> List[Any]:
        """
        Get tokens in a specified range.
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            
        Returns:
            List of token mocks in the specified range
        """
        return self.tokens[start:end]
    
    def get_text_span(self, start_char: int, end_char: int) -> str:
        """
        Get text span by character indices.
        
        Args:
            start_char: Start character index
            end_char: End character index
            
        Returns:
            Text substring
        """
        return self.text[start_char:end_char]
    
    def find_entities_by_label(self, label: str) -> List[Any]:
        """
        Find entities with a specific label.
        
        Args:
            label: Entity label to search for
            
        Returns:
            List of entity mocks with the specified label
        """
        return [ent for ent in self.ents if hasattr(ent, 'label_') and ent.label_ == label]
    
    def get_pos_tags(self) -> List[Tuple[str, str]]:
        """
        Get part-of-speech tags for all tokens.
        
        Returns:
            List of (token_text, pos_tag) tuples
        """
        return [(token.text, getattr(token, 'pos_', 'UNKNOWN')) for token in self.tokens]
    
    def get_lemmas(self) -> List[str]:
        """
        Get lemmas for all tokens.
        
        Returns:
            List of lemmatized token texts
        """
        return [getattr(token, 'lemma_', token.text.lower()) for token in self.tokens]
    
    def get_dependencies(self) -> List[Tuple[str, str, str]]:
        """
        Get dependency relations for all tokens.
        
        Returns:
            List of (token_text, dependency_relation, head_text) tuples
        """
        dependencies = []
        for token in self.tokens:
            dep = getattr(token, 'dep_', 'UNKNOWN')
            head_text = getattr(token, 'head', Mock()).text if hasattr(token, 'head') else 'ROOT'
            dependencies.append((token.text, dep, head_text))
        return dependencies
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert document to JSON representation.
        
        Returns:
            Dictionary representation of the document
        """
        return {
            'text': self.text,
            'tokens': [
                {
                    'text': token.text,
                    'lemma': getattr(token, 'lemma_', token.text),
                    'pos': getattr(token, 'pos_', 'UNKNOWN'),
                    'tag': getattr(token, 'tag_', 'UNKNOWN'),
                    'dep': getattr(token, 'dep_', 'UNKNOWN'),
                    'is_alpha': getattr(token, 'is_alpha', True),
                    'is_stop': getattr(token, 'is_stop', False)
                }
                for token in self.tokens
            ],
            'entities': [
                {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': getattr(ent, 'start', 0),
                    'end': getattr(ent, 'end', 0)
                }
                for ent in self.ents
            ],
            'lang': self.lang
        }
    
    def similarity_with(self, other: 'AbstractNLPDocMock') -> float:
        """
        Calculate similarity with another document.
        
        Args:
            other: Another document mock
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple mock similarity based on shared words
        self_words = set(self.text.lower().split())
        other_words = set(other.text.lower().split())
        
        if not self_words and not other_words:
            return 1.0
        if not self_words or not other_words:
            return 0.0
        
        intersection = len(self_words & other_words)
        union = len(self_words | other_words)
        
        return intersection / union if union > 0 else 0.0