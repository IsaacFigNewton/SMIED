# SMIED Testing Framework Design Specifications

## Overview

The SMIED testing framework employs a sophisticated, multi-layered architecture designed for maintainability, reusability, and comprehensive test coverage. The framework separates concerns across multiple dimensions and uses advanced design patterns to ensure robust testing of complex semantic processing components.

## Architectural Design Principles

### 1. **Separation of Concerns Architecture**
The testing framework is organized into three distinct layers:

- **Test Layer (`tests/test_*.py`)**: Contains test logic and assertions
- **Mock Layer (`tests/mocks/*.py`)**: Provides mock implementations and factories
- **Configuration Layer (`tests/config/*.py`)**: Supplies test data and constants

This separation ensures:
- **Maintainability**: Changes to test data don't require modifying test logic
- **Reusability**: Mock objects and test data can be shared across multiple test files
- **Clarity**: Each layer has a single, well-defined responsibility

### 2. **Factory Pattern for Mock Creation**
Each component implements a `MockFactory` class that provides standardized mock creation:

```python
class ComponentMockFactory:
    def __init__(self):
        self._mock_classes = {
            'MockComponent': MockComponent,
            'MockComponentEdgeCases': MockComponentEdgeCases,
            'MockComponentIntegration': MockComponentIntegration,
            # ... additional mock types
        }
    
    def __call__(self, mock_name: str, *args, **kwargs) -> Mock:
        # Factory method implementation
```

**Benefits:**
- **Consistency**: Standardized mock creation across all components
- **Discoverability**: `get_available_mocks()` method lists all available mocks
- **Type Safety**: Proper error handling for unknown mock types
- **Extensibility**: Easy to add new mock variants without modifying existing code

### 3. **Abstract Base Class Hierarchy**
The framework uses a sophisticated inheritance structure:

```
AbstractEntityMock (ABC + Mock)
├── AbstractNLPDocMock
├── AbstractNLPTokenMock  
├── AbstractNLPFunctionMock
├── AbstractCollectionMock
├── AbstractOperationMock
└── ... (component-specific base classes)
```

**Key Features:**
- **Interface Enforcement**: ABC ensures required methods are implemented
- **Mock Functionality**: Inherits from `unittest.Mock` for testing features
- **Common Functionality**: Shared methods for property management, relationships, validation
- **Type Polymorphism**: Consistent interfaces across different mock types

### 4. **Configuration-Driven Test Data Management**
Each component has a corresponding configuration class with static methods:

```python
class ComponentMockConfig:
    @staticmethod
    def get_basic_test_data():
        # Returns structured test data
    
    @staticmethod  
    def get_edge_case_scenarios():
        # Returns edge case test data
    
    @staticmethod
    def get_integration_test_data():
        # Returns integration test data
```

**Advantages:**
- **Centralization**: All test data in one location per component
- **Versioning**: Easy to update test data without touching test logic
- **Sharing**: Test data can be reused across multiple test methods
- **Documentation**: Test data serves as examples of expected inputs/outputs

## Testing Hierarchy and Organization

### Test Class Structure
Each test file follows a consistent organization pattern:

```python
class TestComponent(unittest.TestCase):
    """Basic functionality tests"""
    
class TestComponentValidation(unittest.TestCase):
    """Validation and constraint tests"""
    
class TestComponentEdgeCases(unittest.TestCase): 
    """Edge cases and error conditions"""
    
class TestComponentIntegration(unittest.TestCase):
    """Integration tests with other components"""
```

### Test Method Naming Convention
- **`test_method_name_scenario()`**: Clear, descriptive names
- **Functionality focus**: What is being tested
- **Scenario description**: Under what conditions
- **Expected outcome**: Implicit in the test assertions

### Fixture Management
- **`setUp()`**: Initializes mock factory, config, and common test objects
- **Factory-based creation**: All mocks created through factories
- **Configuration injection**: Test data injected from config classes
- **Clean initialization**: Each test starts with fresh mock instances

## Mock Implementation Patterns

### 1. **Specialized Mock Variants**
Each component provides multiple mock variants for different testing scenarios:

- **`MockComponent`**: Basic functionality mock
- **`MockComponentEdgeCases`**: Handles error conditions and edge cases
- **`MockComponentIntegration`**: Provides realistic behavior for integration testing

### 2. **Behavioral Configuration**
Mocks can be configured with realistic behavior:

```python
class MockSynsetForBeam(Mock):
    def __init__(self, name="test.n.01", definition="test definition"):
        super().__init__()
        self._name = name
        self._definition = definition
        # Configure realistic method returns
        self.name = Mock(return_value=name)
        self.definition = Mock(return_value=definition)
        # Set up relationship methods
        self.hypernyms = Mock(return_value=[])
        self.hyponyms = Mock(return_value=[])
```

### 3. **Abstract Method Implementation**
All concrete mocks implement required abstract methods:

```python
def get_primary_attribute(self) -> Any:
    return self.name

def validate_entity(self) -> bool:
    return self.is_valid and bool(self.name)
    
def get_entity_signature(self) -> str:
    return f"{self.entity_type.value}:{self.id}:{self.name}"
```

## Test Data Management

### Structured Test Data
Configuration classes provide hierarchically organized test data:

```python
def get_basic_vertex_structures():
    return {
        'simple_vertices': [
            ("word1", {"pos": "NOUN"}),
            ("word2", {"pos": "VERB"}),
        ],
        'person_action_vertices': [
            ("John", {"type": "person", "pos": "NOUN"}),
            ("runs", {"type": "action", "pos": "VERB"}),
        ]
    }
```

### Scenario-Based Organization
Test data is organized by testing scenarios:

- **Basic scenarios**: Simple, happy-path test cases
- **Edge case scenarios**: Boundary conditions and error cases
- **Integration scenarios**: Multi-component interaction data
- **Performance scenarios**: Large-scale data for performance testing

### Data Consistency
- **Realistic values**: Test data mirrors real-world inputs
- **Relationship consistency**: Related data elements are properly connected
- **Type safety**: All data types match expected component interfaces

## Advanced Testing Features

### 1. **Relationship Management**
Entity mocks support complex relationship modeling:

```python
def add_relationship(self, relation_type: str, target_entity, weight: float = 1.0):
    # Bidirectional relationship setup
    # Weight and metadata storage
    # Automatic inverse relationship creation
```

### 2. **Property System**
Dynamic property management with caching:

```python
def set_property(self, name: str, value: Any, computed: bool = False):
    # Property storage
    # Cache invalidation
    # Update tracking

def get_property(self, name: str, default: Any = None, use_cache: bool = True):
    # Cache-aware retrieval
    # Access tracking
    # Fallback handling
```

### 3. **Validation Framework**
Built-in validation for entity consistency:

```python
def validate_entity(self) -> bool:
    # Abstract method for entity-specific validation
    
def matches_criteria(self, criteria: Dict[str, Any]) -> bool:
    # Flexible criteria matching
    # Property-based filtering
    # Type-safe comparisons
```

### 4. **Similarity Calculations**
Mock entities can calculate similarity scores:

```python
def calculate_similarity(self, other: 'AbstractEntityMock') -> float:
    # Multi-factor similarity calculation
    # Property, tag, and relationship similarity
    # Normalized scoring (0.0 to 1.0)
```

## Integration with Testing Tools

### unittest Framework Integration
- **Full compatibility**: All classes inherit from `unittest.TestCase`
- **Standard assertions**: Uses unittest assertion methods
- **Test discovery**: Compatible with unittest test discovery
- **Fixture support**: setUp/tearDown methods work as expected

### Mock Library Integration  
- **unittest.mock compatibility**: All mocks inherit from `unittest.Mock`
- **Patch support**: Works with `@patch` decorators
- **Side effect configuration**: Supports complex mock behaviors
- **Call verification**: Standard mock verification methods available

### External Library Mocking
- **NLP libraries**: spaCy, NLTK integration mocking
- **Graph libraries**: NetworkX graph mocking
- **ML libraries**: Embedding model mocking
- **Database libraries**: Connection and query mocking

## Best Practices and Guidelines

### Mock Creation
1. **Always use factories**: Create mocks through factory methods
2. **Configure realistically**: Set up mocks to behave like real components
3. **Use appropriate variants**: Choose the right mock type for the test scenario
4. **Validate mock state**: Ensure mocks are properly initialized

### Test Data Usage
1. **Load from config**: Always get test data from configuration classes
2. **Don't hardcode values**: Avoid magic numbers and strings in tests
3. **Reuse common data**: Share test data across related tests
4. **Document data meaning**: Provide clear names for test data sets

### Test Organization
1. **Group by functionality**: Organize tests by what they test
2. **Separate concerns**: Keep different test types in different classes
3. **Use descriptive names**: Test methods should clearly indicate their purpose
4. **Follow the pattern**: Maintain consistency with existing test structure

### Error Handling
1. **Test error conditions**: Include tests for error scenarios
2. **Use appropriate assertions**: `assertRaises`, `assertWarns`, etc.
3. **Mock exceptions properly**: Configure mocks to raise realistic exceptions
4. **Verify error messages**: Check that errors contain expected information

## Framework Extension Guidelines

### Adding New Components
1. **Create mock factory**: Implement `ComponentMockFactory` class
2. **Create configuration**: Implement `ComponentMockConfig` class  
3. **Extend base classes**: Inherit from appropriate abstract base classes
4. **Follow naming conventions**: Use consistent naming patterns
5. **Update documentation**: Document new mock types and test data

### Adding New Mock Types
1. **Identify purpose**: Determine what testing scenario the mock serves
2. **Choose base class**: Select appropriate abstract base class
3. **Implement required methods**: Fulfill all abstract method requirements
4. **Add to factory**: Register new mock in factory class
5. **Create test data**: Add corresponding test data to configuration

### Performance Considerations
1. **Mock efficiency**: Keep mock creation lightweight
2. **Data size management**: Use appropriately sized test data sets
3. **Caching strategy**: Implement caching for expensive mock operations
4. **Memory management**: Clean up resources in tearDown methods

## Quality Assurance

### Code Coverage
- **Target coverage**: Aim for >90% test coverage
- **Branch coverage**: Test all conditional paths
- **Mock coverage**: Ensure all mock methods are tested
- **Integration coverage**: Test component interactions

### Test Reliability
- **Deterministic tests**: Avoid randomness in test outcomes
- **Isolation**: Tests should not depend on each other
- **Cleanup**: Proper resource cleanup in tearDown
- **Repeatability**: Tests should produce consistent results

### Maintainability
- **Clear documentation**: Document complex test scenarios
- **Consistent patterns**: Follow established patterns throughout
- **Regular refactoring**: Keep test code clean and organized
- **Version control**: Track changes to test data and mock implementations

This testing framework provides a robust, extensible foundation for comprehensive testing of the SMIED semantic processing system, ensuring reliability and maintainability as the system evolves.