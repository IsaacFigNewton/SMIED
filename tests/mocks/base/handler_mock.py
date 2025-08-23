"""
Abstract base class for format handler mocks.

This module provides the AbstractHandlerMock class that serves as a base
for all format handler mocks (JSON, YAML, XML, etc.).
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock
from typing import Any, Dict, List, Optional, Union


class AbstractHandlerMock(ABC, Mock):
    """
    Abstract base class for format handler mocks.
    
    This class provides a common interface for format handlers that typically
    handle loading, saving, and validating data in various formats (JSON, YAML, XML, etc.).
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the AbstractHandlerMock.
        
        Args:
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes()
        self._setup_common_methods()
    
    def _setup_common_attributes(self):
        """Set up common attributes shared by all handler mocks."""
        # Format-specific configuration
        self.supported_extensions = []
        self.encoding = "utf-8"
        self.format_name = "unknown"
        
        # Validation settings
        self.strict_mode = True
        self.validation_rules = {}
        
        # Processing options
        self.parse_options = {}
        self.output_options = {}
    
    def _setup_common_methods(self):
        """Set up common mock methods shared by all handler mocks."""
        # Core handler methods are abstract and must be implemented
        
        # Common validation helpers
        self.is_valid_format = Mock(return_value=True)
        self.get_format_info = Mock(return_value={})
        self.set_encoding = Mock()
        
        # Error handling helpers
        self.handle_parse_error = Mock()
        self.handle_validation_error = Mock()
        
        # Utility methods
        self.normalize_data = Mock(side_effect=lambda data: data)
        self.preprocess_data = Mock(side_effect=lambda data: data)
        self.postprocess_data = Mock(side_effect=lambda data: data)
    
    @abstractmethod
    def load(self, source: Union[str, Any]) -> Any:
        """
        Load data from the specified source.
        
        Args:
            source: Source to load data from (file path, string, stream, etc.)
            
        Returns:
            Loaded and parsed data
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def save(self, data: Any, target: Union[str, Any]) -> None:
        """
        Save data to the specified target.
        
        Args:
            data: Data to save
            target: Target to save data to (file path, string, stream, etc.)
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate data according to format-specific rules.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this handler.
        
        Returns:
            List of supported file extensions (e.g., ['.json', '.jsonl'])
        """
        return self.supported_extensions.copy()
    
    def supports_format(self, file_path_or_extension: str) -> bool:
        """
        Check if this handler supports the given format.
        
        Args:
            file_path_or_extension: File path or extension to check
            
        Returns:
            True if format is supported, False otherwise
        """
        if file_path_or_extension.startswith('.'):
            return file_path_or_extension in self.supported_extensions
        
        # Extract extension from file path
        extension = '.' + file_path_or_extension.split('.')[-1].lower()
        return extension in self.supported_extensions
    
    def configure_validation(self, rules: Dict[str, Any]) -> None:
        """
        Configure validation rules for the handler.
        
        Args:
            rules: Dictionary of validation rules
        """
        self.validation_rules.update(rules)
    
    def set_parse_options(self, options: Dict[str, Any]) -> None:
        """
        Set parsing options for the handler.
        
        Args:
            options: Dictionary of parsing options
        """
        self.parse_options.update(options)
    
    def set_output_options(self, options: Dict[str, Any]) -> None:
        """
        Set output formatting options for the handler.
        
        Args:
            options: Dictionary of output options
        """
        self.output_options.update(options)
    
    def create_error_context(self, error: Exception, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Create contextual information for error handling.
        
        Args:
            error: The exception that occurred
            source: Optional source information (file path, etc.)
            
        Returns:
            Dictionary containing error context information
        """
        context = {
            'handler_type': self.format_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'encoding': self.encoding,
            'strict_mode': self.strict_mode
        }
        
        if source:
            context['source'] = source
            
        return context