"""
Abstract base class for NLP processing function mocks.

This module provides the AbstractNLPFunctionMock class that serves as a base
for NLP processing function mocks that simulate spaCy nlp() functions and
other NLP pipeline components.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock
from typing import Any, Dict, List, Optional, Union, Callable, Iterator


class AbstractNLPFunctionMock(ABC, Mock):
    """
    Abstract base class for NLP processing function mocks.
    
    This class provides a common interface for mocks that represent NLP
    processing functions, typically returning Doc objects from text input.
    These are commonly used to mock spaCy's nlp() pipeline or similar functions.
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the AbstractNLPFunctionMock.
        
        Args:
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes()
        self._setup_pipeline_components()
        self._setup_processing_options()
    
    def _setup_common_attributes(self):
        """Set up common attributes for NLP functions."""
        # Pipeline configuration
        self.lang = "en"
        self.model_name = "test_model"
        self.model_version = "1.0.0"
        
        # Processing components
        self.pipe_names = ["tagger", "parser", "ner"]
        self.disabled_components = set()
        self.enabled_components = set(self.pipe_names)
        
        # Pipeline state
        self.is_initialized = True
        self.vocab = Mock()
        self.meta = {
            "name": self.model_name,
            "lang": self.lang,
            "version": self.model_version,
            "pipeline": self.pipe_names
        }
        
        # Performance tracking
        self.processed_texts = []
        self.processing_times = []
        self.error_count = 0
    
    def _setup_pipeline_components(self):
        """Set up mock pipeline components."""
        # Core NLP components
        self.tokenizer = Mock()
        self.tagger = Mock()  # POS tagger
        self.parser = Mock()  # Dependency parser
        self.ner = Mock()  # Named entity recognizer
        self.lemmatizer = Mock()
        self.attribute_ruler = Mock()
        
        # Additional components
        self.sentencizer = Mock()
        self.entity_ruler = Mock()
        self.matcher = Mock()
        self.phrase_matcher = Mock()
        
        # Component registry
        self.components = {
            "tokenizer": self.tokenizer,
            "tagger": self.tagger,
            "parser": self.parser,
            "ner": self.ner,
            "lemmatizer": self.lemmatizer,
            "attribute_ruler": self.attribute_ruler,
            "sentencizer": self.sentencizer,
            "entity_ruler": self.entity_ruler
        }
    
    def _setup_processing_options(self):
        """Set up processing options and configurations."""
        # Processing settings
        self.batch_size = 32
        self.n_process = 1
        self.disable_pipes = []
        
        # Output configuration
        self.return_doc = True
        self.include_vectors = False
        self.include_sentiment = False
        
        # Error handling
        self.ignore_errors = False
        self.error_callback = None
        
        # Caching
        self.use_cache = False
        self.cache_size = 0
        self._doc_cache = {}
    
    @abstractmethod
    def process_text(self, text: str, **kwargs) -> Any:
        """
        Process a single text and return a Doc object.
        
        Args:
            text: Input text to process
            **kwargs: Additional processing options
            
        Returns:
            Mock Doc object representing processed text
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def create_doc_mock(self, text: str, **kwargs) -> Any:
        """
        Create a mock Doc object for the given text.
        
        Args:
            text: Input text
            **kwargs: Additional parameters for Doc creation
            
        Returns:
            Mock Doc object
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def configure_pipeline(self, components: List[str]) -> None:
        """
        Configure the processing pipeline.
        
        Args:
            components: List of component names to include
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def __call__(self, text: str, **kwargs) -> Any:
        """
        Make the mock callable like a real NLP function.
        
        Args:
            text: Input text to process
            **kwargs: Additional processing options
            
        Returns:
            Mock Doc object
        """
        # Track processing
        self.processed_texts.append(text)
        
        # Check cache if enabled
        if self.use_cache and text in self._doc_cache:
            return self._doc_cache[text]
        
        # Process text
        try:
            doc = self.process_text(text, **kwargs)
            
            # Cache result if enabled
            if self.use_cache:
                self._doc_cache[text] = doc
                
            return doc
        except Exception as e:
            self.error_count += 1
            if self.error_callback:
                self.error_callback(e, text)
            if not self.ignore_errors:
                raise
            return self.create_error_doc(text, e)
    
    def pipe(self, texts: Iterator[str], **kwargs) -> Iterator[Any]:
        """
        Process multiple texts in a pipeline.
        
        Args:
            texts: Iterator of input texts
            **kwargs: Additional processing options
            
        Yields:
            Mock Doc objects for each input text
        """
        batch = []
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        for text in texts:
            batch.append(text)
            
            if len(batch) >= batch_size:
                yield from self._process_batch(batch, **kwargs)
                batch = []
        
        # Process remaining batch
        if batch:
            yield from self._process_batch(batch, **kwargs)
    
    def _process_batch(self, batch: List[str], **kwargs) -> List[Any]:
        """
        Process a batch of texts.
        
        Args:
            batch: List of texts to process
            **kwargs: Processing options
            
        Returns:
            List of mock Doc objects
        """
        return [self(text, **kwargs) for text in batch]
    
    def disable_pipe(self, component_name: str) -> None:
        """
        Disable a pipeline component.
        
        Args:
            component_name: Name of component to disable
        """
        if component_name in self.enabled_components:
            self.enabled_components.remove(component_name)
            self.disabled_components.add(component_name)
    
    def enable_pipe(self, component_name: str) -> None:
        """
        Enable a pipeline component.
        
        Args:
            component_name: Name of component to enable
        """
        if component_name in self.disabled_components:
            self.disabled_components.remove(component_name)
            self.enabled_components.add(component_name)
    
    def select_pipes(self, enable: Optional[List[str]] = None, 
                    disable: Optional[List[str]] = None):
        """
        Context manager for temporarily enabling/disabling components.
        
        Args:
            enable: Components to enable
            disable: Components to disable
            
        Returns:
            Context manager
        """
        class PipeSelector:
            def __init__(self, nlp_mock, enable_list, disable_list):
                self.nlp_mock = nlp_mock
                self.enable_list = enable_list or []
                self.disable_list = disable_list or []
                self.original_enabled = None
                self.original_disabled = None
            
            def __enter__(self):
                self.original_enabled = self.nlp_mock.enabled_components.copy()
                self.original_disabled = self.nlp_mock.disabled_components.copy()
                
                for component in self.enable_list:
                    self.nlp_mock.enable_pipe(component)
                
                for component in self.disable_list:
                    self.nlp_mock.disable_pipe(component)
                
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.nlp_mock.enabled_components = self.original_enabled
                self.nlp_mock.disabled_components = self.original_disabled
        
        return PipeSelector(self, enable, disable)
    
    def get_pipe(self, component_name: str) -> Any:
        """
        Get a pipeline component by name.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Mock component object
        """
        return self.components.get(component_name, Mock())
    
    def add_pipe(self, component: Any, name: str, last: bool = True) -> None:
        """
        Add a component to the pipeline.
        
        Args:
            component: Component object to add
            name: Name for the component
            last: Whether to add at the end of pipeline
        """
        self.components[name] = component
        
        if last:
            self.pipe_names.append(name)
        else:
            self.pipe_names.insert(0, name)
        
        self.enabled_components.add(name)
    
    def remove_pipe(self, component_name: str) -> Any:
        """
        Remove a component from the pipeline.
        
        Args:
            component_name: Name of component to remove
            
        Returns:
            Removed component
        """
        component = self.components.pop(component_name, None)
        
        if component_name in self.pipe_names:
            self.pipe_names.remove(component_name)
        
        self.enabled_components.discard(component_name)
        self.disabled_components.discard(component_name)
        
        return component
    
    def create_error_doc(self, text: str, error: Exception) -> Any:
        """
        Create a minimal doc object for error cases.
        
        Args:
            text: Original input text
            error: The error that occurred
            
        Returns:
            Mock Doc object with error information
        """
        error_doc = Mock()
        error_doc.text = text
        error_doc.tokens = []
        error_doc.ents = []
        error_doc.sents = []
        error_doc.error = error
        error_doc.is_parsed = False
        error_doc.is_tagged = False
        error_doc.is_sentenced = False
        
        return error_doc
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about text processing.
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            'texts_processed': len(self.processed_texts),
            'total_characters': sum(len(text) for text in self.processed_texts),
            'average_text_length': (
                sum(len(text) for text in self.processed_texts) / len(self.processed_texts)
                if self.processed_texts else 0
            ),
            'errors_encountered': self.error_count,
            'cache_size': len(self._doc_cache) if self.use_cache else 0,
            'enabled_components': list(self.enabled_components),
            'disabled_components': list(self.disabled_components),
            'model_info': self.meta.copy()
        }
    
    def reset_processing_stats(self) -> None:
        """Reset processing statistics and clear cache."""
        self.processed_texts.clear()
        self.processing_times.clear()
        self.error_count = 0
        self._doc_cache.clear()
    
    def simulate_processing_delay(self, text: str) -> float:
        """
        Simulate processing delay based on text length.
        
        Args:
            text: Input text
            
        Returns:
            Simulated processing time in seconds
        """
        # Simple simulation: 0.1ms per character + base overhead
        base_time = 0.01  # 10ms base
        char_time = len(text) * 0.0001  # 0.1ms per character
        return base_time + char_time
    
    def create_realistic_doc(self, text: str) -> Any:
        """
        Create a more realistic Doc mock with typical NLP annotations.
        
        Args:
            text: Input text
            
        Returns:
            Mock Doc object with realistic annotations
        """
        doc_mock = Mock()
        doc_mock.text = text
        doc_mock.lang = self.lang
        doc_mock.vocab = self.vocab
        
        # Basic tokenization
        words = text.split()
        doc_mock.tokens = []
        
        for i, word in enumerate(words):
            token_mock = Mock()
            token_mock.text = word
            token_mock.i = i
            token_mock.lemma_ = word.lower()
            token_mock.pos_ = "NOUN" if word.isalpha() else "PUNCT"
            token_mock.tag_ = "NN" if word.isalpha() else "."
            token_mock.dep_ = "ROOT" if i == 0 else "nmod"
            token_mock.is_alpha = word.isalpha()
            token_mock.is_stop = word.lower() in {"the", "a", "an"}
            doc_mock.tokens.append(token_mock)
        
        # Mock entities
        doc_mock.ents = []
        for word in words:
            if word.istitle() and len(word) > 1:
                ent_mock = Mock()
                ent_mock.text = word
                ent_mock.label_ = "PERSON"
                doc_mock.ents.append(ent_mock)
        
        # Mock sentences
        doc_mock.sents = [Mock(text=text)]
        
        # Processing flags
        doc_mock.is_parsed = "parser" in self.enabled_components
        doc_mock.is_tagged = "tagger" in self.enabled_components
        doc_mock.is_sentenced = True
        
        return doc_mock