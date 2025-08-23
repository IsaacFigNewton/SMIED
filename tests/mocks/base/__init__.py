"""
Abstract base classes for mock refactoring.

This module provides abstract base classes that serve as foundations for
specific mock implementations. These classes provide common interfaces,
shared functionality, and enforce consistent patterns across the test suite.

The abstract base classes are organized by functional category:

Handler Mocks:
    - AbstractHandlerMock: Base for format handler mocks (JSON, YAML, XML, etc.)

Edge Case and Integration Testing:
    - AbstractEdgeCaseMock: Base for edge case testing scenarios
    - AbstractIntegrationMock: Base for integration testing mocks

Graph and Pattern Mocks:
    - AbstractGraphPatternMock: Base for graph/network pattern mocks

NLP Processing Mocks:
    - AbstractNLPDocMock: Base for NLP document mocks
    - AbstractNLPTokenMock: Base for NLP token mocks  
    - AbstractNLPFunctionMock: Base for NLP processing function mocks

Algorithmic and Operation Mocks:
    - AbstractAlgorithmicFunctionMock: Base for algorithmic/mathematical function mocks
    - AbstractOperationMock: Base for operation-specific mocks

Library and Collection Mocks:
    - AbstractLibraryWrapperMock: Base for external library interface mocks
    - AbstractCollectionMock: Base for collection-like mocks

Domain and Reasoning Mocks:
    - AbstractEntityMock: Base for domain entity mocks
    - AbstractReasoningMock: Base for reasoning/inference engine mocks

Usage:
    Each abstract base class provides:
    - Common __init__ that calls super().__init__(*args, **kwargs)
    - Abstract properties/methods that subclasses must implement
    - Common helper methods to reduce duplication
    - Proper documentation and type hints

Example:
    ```python
    from tests.mocks.base import AbstractHandlerMock
    
    class MockJSONHandler(AbstractHandlerMock):
        def load(self, source):
            # Implementation specific to JSON handling
            pass
            
        def save(self, data, target):
            # Implementation specific to JSON handling
            pass
            
        def validate(self, data):
            # Implementation specific to JSON validation
            return True
    ```
"""

# Handler Mocks
from .handler_mock import AbstractHandlerMock

# Edge Case and Integration Testing
from .edge_case_mock import AbstractEdgeCaseMock
from .integration_mock import AbstractIntegrationMock

# Graph and Pattern Mocks
from .graph_pattern_mock import AbstractGraphPatternMock

# NLP Processing Mocks
from .nlp_doc_mock import AbstractNLPDocMock
from .nlp_token_mock import AbstractNLPTokenMock
from .nlp_function_mock import AbstractNLPFunctionMock

# Algorithmic and Operation Mocks
from .algorithmic_function_mock import AbstractAlgorithmicFunctionMock
from .operation_mock import AbstractOperationMock, OperationStatus, OperationType

# Library and Collection Mocks
from .library_wrapper_mock import AbstractLibraryWrapperMock
from .collection_mock import AbstractCollectionMock

# Domain and Reasoning Mocks
from .entity_mock import AbstractEntityMock, EntityType, EntityStatus
from .reasoning_mock import (
    AbstractReasoningMock, 
    ReasoningType, 
    InferenceStrategy, 
    ConfidenceLevel,
    InferenceResult,
    Rule
)

# Version information
__version__ = "1.0.0"
__author__ = "SMIED Test Framework"

# Export all abstract base classes
__all__ = [
    # Handler Mocks
    "AbstractHandlerMock",
    
    # Edge Case and Integration Testing
    "AbstractEdgeCaseMock",
    "AbstractIntegrationMock",
    
    # Graph and Pattern Mocks
    "AbstractGraphPatternMock",
    
    # NLP Processing Mocks
    "AbstractNLPDocMock",
    "AbstractNLPTokenMock", 
    "AbstractNLPFunctionMock",
    
    # Algorithmic and Operation Mocks
    "AbstractAlgorithmicFunctionMock",
    "AbstractOperationMock",
    "OperationStatus",
    "OperationType",
    
    # Library and Collection Mocks
    "AbstractLibraryWrapperMock",
    "AbstractCollectionMock",
    
    # Domain and Reasoning Mocks
    "AbstractEntityMock",
    "EntityType",
    "EntityStatus",
    "AbstractReasoningMock",
    "ReasoningType",
    "InferenceStrategy",
    "ConfidenceLevel",
    "InferenceResult",
    "Rule",
]

# Utility functions for working with abstract base classes
def get_all_abstract_classes():
    """
    Get list of all available abstract base classes.
    
    Returns:
        List of abstract base class names
    """
    return [
        "AbstractHandlerMock",
        "AbstractEdgeCaseMock", 
        "AbstractIntegrationMock",
        "AbstractGraphPatternMock",
        "AbstractNLPDocMock",
        "AbstractNLPTokenMock",
        "AbstractNLPFunctionMock", 
        "AbstractAlgorithmicFunctionMock",
        "AbstractOperationMock",
        "AbstractLibraryWrapperMock",
        "AbstractCollectionMock",
        "AbstractEntityMock",
        "AbstractReasoningMock",
    ]

def get_abstract_classes_by_category():
    """
    Get abstract base classes organized by category.
    
    Returns:
        Dictionary mapping categories to lists of class names
    """
    return {
        "handler_mocks": ["AbstractHandlerMock"],
        "testing_mocks": ["AbstractEdgeCaseMock", "AbstractIntegrationMock"],
        "graph_mocks": ["AbstractGraphPatternMock"],
        "nlp_mocks": [
            "AbstractNLPDocMock", 
            "AbstractNLPTokenMock", 
            "AbstractNLPFunctionMock"
        ],
        "algorithmic_mocks": [
            "AbstractAlgorithmicFunctionMock", 
            "AbstractOperationMock"
        ],
        "infrastructure_mocks": [
            "AbstractLibraryWrapperMock", 
            "AbstractCollectionMock"
        ],
        "domain_mocks": ["AbstractEntityMock", "AbstractReasoningMock"],
    }

def validate_inheritance(mock_class, expected_base_classes):
    """
    Validate that a mock class inherits from expected base classes.
    
    Args:
        mock_class: Class to validate
        expected_base_classes: List of expected base class names
        
    Returns:
        Tuple of (is_valid, missing_bases, extra_bases)
    """
    actual_bases = {base.__name__ for base in mock_class.__mro__[1:]}  # Skip the class itself
    expected_bases = set(expected_base_classes)
    
    # Check for ABC and Mock in inheritance
    required_bases = {"ABC", "Mock"}
    
    missing_bases = (expected_bases | required_bases) - actual_bases
    extra_bases = actual_bases - (expected_bases | required_bases | {"object"})
    
    is_valid = len(missing_bases) == 0
    
    return is_valid, list(missing_bases), list(extra_bases)

def get_mock_implementation_template(base_class_name):
    """
    Get a template for implementing a specific abstract base class.
    
    Args:
        base_class_name: Name of the abstract base class
        
    Returns:
        String containing implementation template
    """
    templates = {
        "AbstractHandlerMock": '''
class My{format}Handler(AbstractHandlerMock):
    """Mock {format} format handler."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.format_name = "{format_lower}"
        self.supported_extensions = [".{ext}"]
    
    def load(self, source):
        # Implement {format}-specific loading logic
        return {{}}
    
    def save(self, data, target):
        # Implement {format}-specific saving logic
        pass
    
    def validate(self, data):
        # Implement {format}-specific validation
        return True
''',
        "AbstractEdgeCaseMock": '''
class My{component}EdgeCases(AbstractEdgeCaseMock):
    """Mock edge cases for {component}."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setup_edge_case_scenario(self, scenario_name):
        if scenario_name == "empty_result":
            # Setup empty result scenario
            pass
        elif scenario_name == "invalid_input":
            # Setup invalid input scenario  
            pass
    
    def get_edge_case_scenarios(self):
        return ["empty_result", "invalid_input", "network_error"]
''',
        # Add more templates as needed
    }
    
    return templates.get(base_class_name, f"# Template for {base_class_name} not available")

# Documentation helpers
def print_inheritance_hierarchy():
    """Print the inheritance hierarchy of abstract base classes."""
    hierarchy = """
    Abstract Base Class Hierarchy:
    
    ABC + Mock
    ├── AbstractHandlerMock
    ├── AbstractEdgeCaseMock  
    ├── AbstractIntegrationMock
    ├── AbstractGraphPatternMock
    ├── AbstractNLPDocMock
    ├── AbstractNLPTokenMock
    ├── AbstractNLPFunctionMock
    ├── AbstractAlgorithmicFunctionMock
    ├── AbstractOperationMock
    ├── AbstractLibraryWrapperMock
    ├── AbstractCollectionMock
    ├── AbstractEntityMock
    └── AbstractReasoningMock
    
    All classes inherit from both ABC (for abstract method enforcement) 
    and Mock (for testing functionality).
    """
    print(hierarchy)

def get_refactoring_guide():
    """
    Get guidance for refactoring existing mocks to use abstract base classes.
    
    Returns:
        String containing refactoring guide
    """
    return """
    Mock Refactoring Guide:
    
    1. Identify the appropriate abstract base class for your mock
    2. Update the class declaration to inherit from the abstract base class
    3. Implement all abstract methods required by the base class
    4. Remove duplicate code that's now handled by the base class
    5. Update imports to include the abstract base class
    6. Run tests to ensure functionality is preserved
    
    Example refactoring:
    
    Before:
    ```python
    class MockJSONHandler(Mock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.load = Mock(return_value={})
            self.save = Mock()
            self.validate = Mock(return_value=True)
    ```
    
    After:
    ```python
    from tests.mocks.base import AbstractHandlerMock
    
    class MockJSONHandler(AbstractHandlerMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.format_name = "json"
            self.supported_extensions = [".json", ".jsonl"]
        
        def load(self, source):
            return {}
        
        def save(self, data, target):
            pass
        
        def validate(self, data):
            return True
    ```
    """