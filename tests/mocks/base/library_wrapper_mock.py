"""
Abstract base class for external library interface mocks.

This module provides the AbstractLibraryWrapperMock class that serves as a base
for mocks that wrap external libraries like NLTK, spaCy, WordNet, NetworkX, etc.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock
from typing import Any, Dict, List, Optional, Union, Callable, Iterator


class AbstractLibraryWrapperMock(ABC, Mock):
    """
    Abstract base class for external library interface mocks.
    
    This class provides a common interface for mocks that wrap external libraries
    and provide a consistent interface for testing. These wrappers typically
    handle library initialization, configuration, and method delegation.
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the AbstractLibraryWrapperMock.
        
        Args:
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes()
        self._setup_library_interface()
        self._setup_version_management()
    
    def _setup_common_attributes(self):
        """Set up common attributes for library wrappers."""
        # Library identification
        self.library_name = "unknown_library"
        self.library_version = "1.0.0"
        self.wrapper_version = "1.0.0"
        
        # Library state
        self.is_initialized = False
        self.is_available = True
        self.initialization_error = None
        
        # Configuration
        self.config = {}
        self.default_config = {}
        self.required_config_keys = []
        
        # Library resources
        self.models = {}
        self.data_files = {}
        self.cache_dir = None
        self.temp_dir = None
        
        # Compatibility
        self.supported_versions = []
        self.deprecated_methods = set()
        self.compatibility_mode = False
    
    def _setup_library_interface(self):
        """Set up library interface and method delegation."""
        # Core library interface
        self._wrapped_library = Mock()
        self._method_mapping = {}
        self._attribute_mapping = {}
        
        # Interface adaptation
        self.method_adapters = {}
        self.result_transformers = {}
        self.parameter_transformers = {}
        
        # Error handling
        self.error_handlers = {}
        self.fallback_methods = {}
        self.retry_policies = {}
        
        # Caching
        self.method_cache = {}
        self.cache_enabled = False
        self.cache_ttl = 300  # 5 minutes default
    
    def _setup_version_management(self):
        """Set up version compatibility and management."""
        # Version tracking
        self.installed_version = None
        self.minimum_required_version = None
        self.maximum_supported_version = None
        
        # Feature availability
        self.available_features = set()
        self.disabled_features = set()
        self.experimental_features = set()
        
        # Migration support
        self.migration_handlers = {}
        self.deprecated_api_usage = {}
    
    @abstractmethod
    def initialize_library(self, **config) -> bool:
        """
        Initialize the wrapped library.
        
        Args:
            **config: Configuration parameters for initialization
            
        Returns:
            True if initialization successful, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def get_library_info(self) -> Dict[str, Any]:
        """
        Get information about the wrapped library.
        
        Returns:
            Dictionary containing library information
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def check_compatibility(self) -> bool:
        """
        Check if the library version is compatible.
        
        Returns:
            True if compatible, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def setup_wrapper(self, **config) -> None:
        """
        Set up the wrapper with configuration.
        
        Args:
            **config: Configuration parameters
        """
        # Update configuration
        self.config.update(config)
        
        # Validate required configuration
        missing_keys = [key for key in self.required_config_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Initialize library
        try:
            if self.initialize_library(**self.config):
                self.is_initialized = True
            else:
                raise RuntimeError("Library initialization failed")
        except Exception as e:
            self.initialization_error = e
            self.is_initialized = False
            raise
    
    def is_method_available(self, method_name: str) -> bool:
        """
        Check if a method is available in the wrapped library.
        
        Args:
            method_name: Name of the method to check
            
        Returns:
            True if method is available, False otherwise
        """
        if not self.is_initialized:
            return False
        
        if method_name in self.deprecated_methods:
            return not self.compatibility_mode
        
        return hasattr(self._wrapped_library, method_name)
    
    def call_library_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call a method on the wrapped library with error handling.
        
        Args:
            method_name: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Method result
            
        Raises:
            AttributeError: If method is not available
            RuntimeError: If library is not initialized
        """
        if not self.is_initialized:
            raise RuntimeError(f"Library {self.library_name} is not initialized")
        
        if not self.is_method_available(method_name):
            if method_name in self.fallback_methods:
                return self.fallback_methods[method_name](*args, **kwargs)
            raise AttributeError(f"Method '{method_name}' not available in {self.library_name}")
        
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = self._create_cache_key(method_name, args, kwargs)
            if cache_key in self.method_cache:
                return self.method_cache[cache_key]
        
        try:
            # Transform parameters if adapter exists
            if method_name in self.parameter_transformers:
                args, kwargs = self.parameter_transformers[method_name](args, kwargs)
            
            # Call the method
            method = getattr(self._wrapped_library, method_name)
            result = method(*args, **kwargs)
            
            # Transform result if transformer exists
            if method_name in self.result_transformers:
                result = self.result_transformers[method_name](result)
            
            # Cache result if caching is enabled
            if self.cache_enabled:
                self.method_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            # Handle error with custom handler if available
            if method_name in self.error_handlers:
                return self.error_handlers[method_name](e, *args, **kwargs)
            raise
    
    def _create_cache_key(self, method_name: str, args: tuple, kwargs: dict) -> str:
        """
        Create a cache key for method call caching.
        
        Args:
            method_name: Name of the method
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Simple cache key generation (would be more sophisticated in real implementation)
        key_parts = [method_name, str(args), str(sorted(kwargs.items()))]
        return "|".join(key_parts)
    
    def add_method_adapter(self, method_name: str, adapter: Callable) -> None:
        """
        Add a method adapter for parameter/result transformation.
        
        Args:
            method_name: Name of the method
            adapter: Adapter function
        """
        self.method_adapters[method_name] = adapter
    
    def add_parameter_transformer(self, method_name: str, transformer: Callable) -> None:
        """
        Add a parameter transformer for a specific method.
        
        Args:
            method_name: Name of the method
            transformer: Function to transform parameters
        """
        self.parameter_transformers[method_name] = transformer
    
    def add_result_transformer(self, method_name: str, transformer: Callable) -> None:
        """
        Add a result transformer for a specific method.
        
        Args:
            method_name: Name of the method
            transformer: Function to transform results
        """
        self.result_transformers[method_name] = transformer
    
    def add_error_handler(self, method_name: str, handler: Callable) -> None:
        """
        Add an error handler for a specific method.
        
        Args:
            method_name: Name of the method
            handler: Function to handle errors
        """
        self.error_handlers[method_name] = handler
    
    def add_fallback_method(self, method_name: str, fallback: Callable) -> None:
        """
        Add a fallback method for when the original is not available.
        
        Args:
            method_name: Name of the method
            fallback: Fallback function
        """
        self.fallback_methods[method_name] = fallback
    
    def enable_caching(self, ttl: Optional[int] = None) -> None:
        """
        Enable method result caching.
        
        Args:
            ttl: Time-to-live for cache entries in seconds
        """
        self.cache_enabled = True
        if ttl is not None:
            self.cache_ttl = ttl
    
    def disable_caching(self) -> None:
        """Disable method result caching."""
        self.cache_enabled = False
        self.clear_cache()
    
    def clear_cache(self) -> None:
        """Clear the method cache."""
        self.method_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return {
            'enabled': self.cache_enabled,
            'size': len(self.method_cache),
            'ttl': self.cache_ttl,
            'hit_rate': 0.8,  # Mock hit rate
            'miss_rate': 0.2   # Mock miss rate
        }
    
    def create_mock_library_object(self, object_type: str, **attributes) -> Mock:
        """
        Create a mock object that resembles a library-specific object.
        
        Args:
            object_type: Type of object to create
            **attributes: Attributes to set on the mock object
            
        Returns:
            Mock object with library-specific interface
        """
        mock_obj = Mock()
        
        # Set specified attributes
        for attr_name, attr_value in attributes.items():
            setattr(mock_obj, attr_name, attr_value)
        
        # Add common library object methods
        mock_obj.__str__ = Mock(return_value=f"Mock{object_type}")
        mock_obj.__repr__ = Mock(return_value=f"Mock{object_type}({attributes})")
        
        return mock_obj
    
    def simulate_library_download(self, resource_name: str) -> bool:
        """
        Simulate downloading a library resource (model, data file, etc.).
        
        Args:
            resource_name: Name of the resource to download
            
        Returns:
            True if download successful, False otherwise
        """
        # Mock download simulation
        if resource_name not in self.data_files:
            self.data_files[resource_name] = {
                'downloaded': True,
                'size': 1024000,  # 1MB mock size
                'version': '1.0.0',
                'checksum': 'mock_checksum_12345'
            }
            return True
        
        return False  # Already exists
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this library.
        
        Returns:
            List of model names
        """
        # Mock model list based on library type
        if "nlp" in self.library_name.lower():
            return ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
        elif "word" in self.library_name.lower():
            return ["wordnet", "brown", "reuters"]
        else:
            return ["default_model", "advanced_model"]
    
    def load_model(self, model_name: str) -> Mock:
        """
        Load a specific model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Mock model object
        """
        if model_name not in self.get_available_models():
            raise ValueError(f"Model '{model_name}' not available")
        
        model_mock = Mock()
        model_mock.name = model_name
        model_mock.is_loaded = True
        model_mock.version = "1.0.0"
        
        self.models[model_name] = model_mock
        return model_mock
    
    def get_version_info(self) -> Dict[str, str]:
        """
        Get version information for the library and wrapper.
        
        Returns:
            Dictionary containing version information
        """
        return {
            'library_name': self.library_name,
            'library_version': self.library_version,
            'wrapper_version': self.wrapper_version,
            'installed_version': self.installed_version or "unknown",
            'compatibility_status': "compatible" if self.check_compatibility() else "incompatible"
        }
    
    def diagnose_installation(self) -> Dict[str, Any]:
        """
        Diagnose library installation and configuration.
        
        Returns:
            Dictionary containing diagnostic information
        """
        return {
            'library_available': self.is_available,
            'initialization_status': self.is_initialized,
            'initialization_error': str(self.initialization_error) if self.initialization_error else None,
            'version_compatible': self.check_compatibility(),
            'required_features': list(self.available_features),
            'missing_features': list(self.disabled_features),
            'config_valid': len([k for k in self.required_config_keys if k not in self.config]) == 0,
            'models_loaded': list(self.models.keys()),
            'data_files': list(self.data_files.keys())
        }
    
    def create_compatibility_layer(self, target_version: str) -> 'AbstractLibraryWrapperMock':
        """
        Create a compatibility layer for a different library version.
        
        Args:
            target_version: Target version for compatibility
            
        Returns:
            New wrapper instance with compatibility layer
        """
        compat_wrapper = self.__class__()
        compat_wrapper.library_name = self.library_name
        compat_wrapper.library_version = target_version
        compat_wrapper.compatibility_mode = True
        
        # Copy configuration
        compat_wrapper.config = self.config.copy()
        compat_wrapper.is_initialized = self.is_initialized
        
        return compat_wrapper