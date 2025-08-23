"""
Mock classes for PatternLoader tests.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional, Union
import json
from tests.mocks.base.handler_mock import AbstractHandlerMock
from tests.mocks.base.edge_case_mock import AbstractEdgeCaseMock


class PatternLoaderMockFactory:
    """Factory class for creating PatternLoader mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockPatternLoader': MockPatternLoader,
            'MockPatternLoaderEdgeCases': MockPatternLoaderEdgeCases,
            'MockPatternLoaderIntegration': MockPatternLoaderIntegration,
            'MockPatternForLoader': MockPatternForLoader,
            'MockValidationRules': MockValidationRules,
            'MockFileSystemForLoader': MockFileSystemForLoader,
            'MockPatternRegistry': MockPatternRegistry,
            'MockFormatHandlers': MockFormatHandlers,
            'MockJSONHandler': MockJSONHandler,
            'MockYAMLHandler': MockYAMLHandler,
            'MockXMLHandler': MockXMLHandler,
        }
    
    def __call__(self, mock_name: str, *args, **kwargs) -> Mock:
        """
        Create and return a mock instance by name.
        
        Args:
            mock_name: Name of the mock class to instantiate
            *args: Arguments to pass to the mock constructor
            **kwargs: Keyword arguments to pass to the mock constructor
            
        Returns:
            Mock instance of the specified type
            
        Raises:
            ValueError: If mock_name is not found
        """
        if mock_name not in self._mock_classes:
            available = ', '.join(self._mock_classes.keys())
            raise ValueError(f"Mock '{mock_name}' not found. Available mocks: {available}")
        
        mock_class = self._mock_classes[mock_name]
        return mock_class(*args, **kwargs)
    
    def get_available_mocks(self) -> List[str]:
        """Return list of available mock class names."""
        return list(self._mock_classes.keys())


class MockPatternLoader(Mock):
    """Mock for PatternLoader class testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize loader components
        self.patterns = []
        self.file_cache = {}
        self.validation_rules = MockValidationRules()
        
        # Set up methods
        self.load_patterns = Mock(return_value=[])
        self.load_from_file = Mock(return_value=[])
        self.load_from_directory = Mock(return_value=[])
        self.load_from_string = Mock(return_value=[])
        self.save_patterns = Mock()
        self.save_to_file = Mock()
        
        # Pattern management
        self.add_pattern = Mock()
        self.remove_pattern = Mock()
        self.get_pattern = Mock(return_value=None)
        self.get_all_patterns = Mock(return_value=[])
        self.clear_patterns = Mock()
        
        # Validation and parsing
        self.validate_pattern = Mock(return_value=True)
        self.parse_pattern = Mock(return_value=MockPatternForLoader())
        self.serialize_pattern = Mock(return_value="")


class MockPatternLoaderEdgeCases(AbstractEdgeCaseMock):
    """Mock for PatternLoader edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Edge case scenarios - using parent class methods
        self.invalid_file = self.file_not_found_error
        self.malformed_json = self.json_decode_error
        self.invalid_pattern = self.value_error
        self.permission_error = self.permission_error
    
    def setup_edge_case_scenario(self, scenario_name: str) -> None:
        """Set up a specific edge case scenario for PatternLoader."""
        scenarios = {
            "file_not_found": lambda: setattr(self, 'load_patterns', self.file_not_found_error),
            "malformed_json": lambda: setattr(self, 'load_from_file', self.json_decode_error),
            "invalid_pattern": lambda: setattr(self, 'validate_pattern', self.value_error),
            "permission_denied": lambda: setattr(self, 'save_patterns', self.permission_error),
            "empty_patterns": lambda: setattr(self, 'get_all_patterns', self.return_empty_list),
            "memory_error": lambda: setattr(self, 'load_from_directory', self.memory_error),
            "io_error": lambda: setattr(self, 'save_to_file', self.io_error)
        }
        
        if scenario_name in scenarios:
            scenarios[scenario_name]()
        else:
            raise ValueError(f"Unknown edge case scenario: {scenario_name}")
    
    def get_edge_case_scenarios(self) -> List[str]:
        """Get list of available edge case scenarios for PatternLoader."""
        return [
            "file_not_found",
            "malformed_json", 
            "invalid_pattern",
            "permission_denied",
            "empty_patterns",
            "memory_error",
            "io_error"
        ]


class MockPatternLoaderIntegration(Mock):
    """Mock for PatternLoader integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integration components
        self.file_system = MockFileSystemForLoader()
        self.pattern_registry = MockPatternRegistry()
        self.format_handlers = MockFormatHandlers()


class MockPatternForLoader(Mock):
    """Mock pattern object for PatternLoader."""
    
    def __init__(self, name="test_pattern", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.version = "1.0"
        self.description = "Test pattern"
        self.author = "test_author"
        self.created_date = "2023-01-01"
        self.modified_date = "2023-01-01"
        
        # Pattern structure
        self.vertices = []
        self.edges = []
        self.constraints = []
        self.parameters = {}
        self.metadata = {}
        
        # Pattern methods
        self.to_dict = Mock(return_value={})
        self.from_dict = Mock()
        self.validate = Mock(return_value=True)
        self.clone = Mock(return_value=Mock())


class MockValidationRules(Mock):
    """Mock validation rules for PatternLoader."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Validation rules
        self.required_fields = ["name", "vertices", "edges"]
        self.optional_fields = ["description", "constraints", "metadata"]
        
        # Validation methods
        self.validate_structure = Mock(return_value=True)
        self.validate_vertices = Mock(return_value=True)
        self.validate_edges = Mock(return_value=True)
        self.validate_constraints = Mock(return_value=True)
        self.validate_metadata = Mock(return_value=True)


class MockFileSystemForLoader(Mock):
    """Mock file system for PatternLoader."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # File system operations
        self.exists = Mock(return_value=True)
        self.is_file = Mock(return_value=True)
        self.is_dir = Mock(return_value=True)
        self.listdir = Mock(return_value=["pattern1.json", "pattern2.json"])
        self.read_file = Mock(return_value='{"name": "test"}')
        self.write_file = Mock()
        self.create_dir = Mock()


class MockPatternRegistry(Mock):
    """Mock pattern registry for PatternLoader."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Registry operations
        self.register_pattern = Mock()
        self.unregister_pattern = Mock()
        self.get_registered_patterns = Mock(return_value=[])
        self.find_pattern = Mock(return_value=None)
        self.check_conflicts = Mock(return_value=[])


class MockFormatHandlers(Mock):
    """Mock format handlers for PatternLoader."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Format handlers
        self.json_handler = MockJSONHandler()
        self.yaml_handler = MockYAMLHandler()
        self.xml_handler = MockXMLHandler()
        
        # Handler management
        self.get_handler = Mock(return_value=self.json_handler)
        self.register_handler = Mock()
        self.supported_formats = Mock(return_value=[".json", ".yaml", ".xml"])


class MockJSONHandler(AbstractHandlerMock):
    """Mock JSON format handler."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set format-specific attributes
        self.format_name = "json"
        self.supported_extensions = [".json", ".jsonl"]
        self.parse_options = {"ensure_ascii": False, "indent": 2}
        
        # Set up JSON-specific validation rules
        self.validation_rules = {
            "require_dict": True,
            "allow_null": True,
            "max_depth": 100
        }
    
    def load(self, source: Union[str, Any]) -> Any:
        """Load JSON data from source."""
        return {}
    
    def save(self, data: Any, target: Union[str, Any]) -> None:
        """Save data as JSON to target."""
        pass
    
    def validate(self, data: Any) -> bool:
        """Validate JSON data structure."""
        return True


class MockYAMLHandler(AbstractHandlerMock):
    """Mock YAML format handler."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set format-specific attributes
        self.format_name = "yaml"
        self.supported_extensions = [".yaml", ".yml"]
        self.parse_options = {"default_flow_style": False, "allow_unicode": True}
        
        # Set up YAML-specific validation rules
        self.validation_rules = {
            "require_dict": True,
            "allow_anchors": True,
            "safe_load": True
        }
    
    def load(self, source: Union[str, Any]) -> Any:
        """Load YAML data from source."""
        return {}
    
    def save(self, data: Any, target: Union[str, Any]) -> None:
        """Save data as YAML to target."""
        pass
    
    def validate(self, data: Any) -> bool:
        """Validate YAML data structure."""
        return True


class MockXMLHandler(AbstractHandlerMock):
    """Mock XML format handler."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set format-specific attributes
        self.format_name = "xml"
        self.supported_extensions = [".xml", ".xsd"]
        self.parse_options = {"remove_blank_text": True, "strip_cdata": False}
        
        # Set up XML-specific validation rules
        self.validation_rules = {
            "require_root": True,
            "validate_schema": False,
            "namespace_aware": True
        }
    
    def load(self, source: Union[str, Any]) -> Any:
        """Load XML data from source."""
        return {}
    
    def save(self, data: Any, target: Union[str, Any]) -> None:
        """Save data as XML to target."""
        pass
    
    def validate(self, data: Any) -> bool:
        """Validate XML data structure."""
        return True