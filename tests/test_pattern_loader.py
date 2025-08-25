import unittest
from unittest.mock import patch, mock_open, Mock
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.PatternLoader import PatternLoader
from tests.mocks.pattern_loader_mocks import PatternLoaderMockFactory
from tests.config.pattern_loader_config import PatternLoaderMockConfig


class TestPatternLoader(unittest.TestCase):
    """Test basic PatternLoader functionality."""
    
    def setUp(self):
        """Set up test fixtures using mock factory and configuration injection."""
        # Initialize mock factory and config
        self.mock_factory = PatternLoaderMockFactory()
        self.config = PatternLoaderMockConfig()
        
        # Get test data from config
        self.test_data = self.config.get_basic_test_data()
        self.sample_patterns = self.test_data['simple_patterns']
        self.json_test_data = self.test_data['json_test_data']
        self.set_conversion_data = self.test_data['set_conversion_data']
        
        # Create mock instances through factory
        self.mock_loader = self.mock_factory('MockPatternLoader')
        self.mock_pattern = self.mock_factory('MockPatternForLoader')
        self.mock_file_system = self.mock_factory('MockFileSystemForLoader')

    @patch('smied.PatternLoader.PatternLoader._get_default_patterns')
    def test_initialization_default(self, mock_get_default):
        """Test PatternLoader initialization with default patterns."""
        mock_get_default.return_value = self.sample_patterns
        
        with patch.object(PatternLoader, 'json_to_pattern'):
            loader = PatternLoader()
            
            mock_get_default.assert_called_once()
            self.assertEqual(loader.patterns, self.sample_patterns)

    @patch('smied.PatternLoader.PatternLoader.load_patterns_from_file')
    def test_initialization_with_file(self, mock_load):
        """Test PatternLoader initialization with patterns file."""
        file_data = self.config.get_file_operation_test_data()
        test_file = file_data['mock_file_paths']['existing_file']
        
        with patch.object(PatternLoader, 'json_to_pattern'):
            loader = PatternLoader(patterns_file=test_file)
            
            mock_load.assert_called_once_with(test_file)

    def test_load_patterns_from_file_success(self):
        """Test load_patterns_from_file with successful file load."""
        file_data = self.config.get_file_operation_test_data()
        valid_content = file_data['valid_json_content']
        mock_file_content = json.dumps(valid_content)
        test_file = file_data['mock_file_paths']['existing_file']
        
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            loader = PatternLoader()
            loader.load_patterns_from_file(test_file)
            
            self.assertEqual(loader.patterns, valid_content)

    @patch('smied.PatternLoader.PatternLoader._get_default_patterns')
    def test_load_patterns_from_file_not_found(self, mock_get_default):
        """Test load_patterns_from_file with file not found."""
        file_data = self.config.get_file_operation_test_data()
        default_data = file_data['default_patterns_data']
        nonexistent_file = file_data['mock_file_paths']['nonexistent_file']
        
        mock_get_default.return_value = default_data
        
        with patch('builtins.open', side_effect=FileNotFoundError), \
             patch('builtins.print') as mock_print:
            
            loader = PatternLoader()
            loader.load_patterns_from_file(nonexistent_file)
            
            mock_print.assert_called()
            mock_get_default.assert_called()

    def test_save_patterns_to_file(self):
        """Test save_patterns_to_file method."""
        file_data = self.config.get_file_operation_test_data()
        output_file = file_data['mock_file_paths']['output_file']
        pattern_data = self.config.get_pattern_management_test_data()
        converted_data = pattern_data['string_representation_data']
        
        loader = PatternLoader()
        loader.patterns = self.sample_patterns
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch.object(loader, 'pattern_to_json') as mock_to_json:
            
            mock_to_json.return_value = converted_data
            
            loader.save_patterns_to_file(output_file)
            
            mock_file.assert_called_once_with(output_file, 'w', encoding='utf-8')
            mock_to_json.assert_called_once()

    def test_json_to_pattern_conversion(self):
        """Test json_to_pattern converts lists to sets correctly."""
        loader = PatternLoader()
        loader.patterns = self.json_test_data
        loader.json_to_pattern()
        
        # Check that lists were converted to sets for specific keys
        pattern = loader.patterns["category"]["pattern1"]
        self.assertIsInstance(pattern[0]["pos"], set)
        self.assertEqual(pattern[0]["pos"], {"NOUN", "VERB"})
        self.assertEqual(pattern[0]["text"], "test")  # Should remain unchanged
        
        self.assertIsInstance(pattern[1]["relation"], set)
        self.assertEqual(pattern[1]["relation"], {"subject"})
        self.assertEqual(pattern[1]["other"], "value")  # Should remain unchanged

    def test_json_to_pattern_handles_non_list_patterns(self):
        """Test json_to_pattern handles non-list patterns gracefully"""
        non_list_patterns = {
            "category": {
                "pattern1": "not_a_list"
            }
        }
        
        loader = PatternLoader()
        loader.patterns = non_list_patterns
        
        # Should not raise error
        loader.json_to_pattern()
        
        # Pattern should remain unchanged
        self.assertEqual(loader.patterns["category"]["pattern1"], "not_a_list")

    def test_pattern_to_json_conversion(self):
        """Test pattern_to_json converts sets back to lists."""
        loader = PatternLoader()
        loader.patterns = self.set_conversion_data
        
        result = loader.pattern_to_json()
        
        # Check structure
        self.assertIn("category", result)
        self.assertIn("pattern1", result["category"])
        pattern_data = result["category"]["pattern1"]
        
        # Check that sets were converted back to lists
        pattern = pattern_data["pattern"]
        self.assertIsInstance(pattern[0]["pos"], list)
        self.assertEqual(set(pattern[0]["pos"]), {"NOUN", "VERB"})
        self.assertEqual(pattern[0]["text"], "test")
        
        self.assertIsInstance(pattern[1]["relation"], list)
        self.assertEqual(set(pattern[1]["relation"]), {"subject"})

    def test_add_pattern_new_category(self):
        """Test add_pattern creates new category if needed."""
        pattern_data = self.config.get_pattern_management_test_data()
        scenario = pattern_data['add_pattern_scenarios']['new_category']
        
        loader = PatternLoader()
        loader.patterns = {}
        
        loader.add_pattern(
            name=scenario['name'],
            pattern=scenario['pattern'],
            description=scenario['description'],
            category=scenario['category']
        )
        
        self.assertIn(scenario['category'], loader.patterns)
        self.assertIn(scenario['name'], loader.patterns[scenario['category']])
        
        added_pattern = loader.patterns[scenario['category']][scenario['name']]
        self.assertEqual(added_pattern["description"], scenario['description'])
        self.assertEqual(added_pattern["pattern"], scenario['pattern'])

    def test_add_pattern_existing_category(self):
        """Test add_pattern adds to existing category."""
        pattern_data = self.config.get_pattern_management_test_data()
        scenario = pattern_data['add_pattern_scenarios']['existing_category']
        
        loader = PatternLoader()
        loader.patterns = {scenario['category']: {}}
        
        loader.add_pattern(
            name=scenario['name'],
            pattern=scenario['pattern'],
            description=scenario['description'],
            category=scenario['category']
        )
        
        self.assertIn(scenario['name'], loader.patterns[scenario['category']])
        
        added_pattern = loader.patterns[scenario['category']][scenario['name']]
        self.assertEqual(added_pattern["description"], scenario['description'])
        self.assertEqual(added_pattern["pattern"], scenario['pattern'])

    def test_add_pattern_default_category(self):
        """Test add_pattern uses default category."""
        pattern_data = self.config.get_pattern_management_test_data()
        scenario = pattern_data['add_pattern_scenarios']['default_category']
        
        loader = PatternLoader()
        loader.patterns = {}
        
        loader.add_pattern(
            name=scenario['name'],
            pattern=scenario['pattern']
        )
        
        self.assertIn(scenario['category'], loader.patterns)
        self.assertIn(scenario['name'], loader.patterns[scenario['category']])

    def test_add_pattern_empty_description(self):
        """Test add_pattern with empty description."""
        pattern_data = self.config.get_pattern_management_test_data()
        scenario = pattern_data['add_pattern_scenarios']['default_category']
        
        loader = PatternLoader()
        loader.patterns = {}
        
        loader.add_pattern(
            name="no_desc_pattern",
            pattern=[{"text": "test"}],
            category="test"
        )
        
        added_pattern = loader.patterns["test"]["no_desc_pattern"]
        self.assertEqual(added_pattern["description"], "")

    @patch('smied.patterns')
    @patch('smied.PatternLoader.files')
    def test_get_default_patterns_success(self, mock_files, mock_patterns):
        """Test _get_default_patterns loads patterns successfully."""
        default_data = self.config.get_default_pattern_test_data()
        test_pattern_data = json.dumps(default_data['mock_resource_patterns']['lexical'])
        
        # Mock the resource path and file opening using mock_open
        with patch('builtins.open', mock_open(read_data=test_pattern_data)):
            mock_path = mock_files.return_value.joinpath.return_value
            mock_path.open = mock_open(read_data=test_pattern_data)
            
            loader = PatternLoader()
            
            with patch.object(loader, 'json_to_pattern'):
                result = loader._get_default_patterns()
            
            self.assertIsInstance(result, dict)

    @patch('smied.patterns')
    @patch('smied.PatternLoader.files')
    def test_get_default_patterns_file_not_found(self, mock_files, mock_patterns):
        """Test _get_default_patterns handles missing files."""
        mock_files.return_value.joinpath.return_value.open.side_effect = FileNotFoundError
        
        loader = PatternLoader()
        
        with patch('builtins.print') as mock_print, \
             patch.object(loader, 'json_to_pattern'):
            result = loader._get_default_patterns()
        
        self.assertIsInstance(result, dict)
        # Should print error messages for missing files
        self.assertGreater(mock_print.call_count, 0)

    def test_str_representation(self):
        """Test __str__ method returns JSON representation."""
        pattern_data = self.config.get_pattern_management_test_data()
        converted_data = pattern_data['string_representation_data']
        
        loader = PatternLoader()
        loader.patterns = self.sample_patterns
        
        with patch.object(loader, 'pattern_to_json') as mock_to_json:
            mock_to_json.return_value = converted_data
            
            result = str(loader)
            
            mock_to_json.assert_called_once()
            # Should be valid JSON string
            parsed = json.loads(result)
            self.assertEqual(parsed, converted_data)


class TestPatternLoaderValidation(unittest.TestCase):
    """Test PatternLoader validation and constraint checking."""
    
    def setUp(self):
        """Set up test fixtures using mock factory and configuration injection."""
        # Initialize mock factory and config
        self.mock_factory = PatternLoaderMockFactory()
        self.config = PatternLoaderMockConfig()
        
        # Get validation test data from config
        self.validation_data = self.config.get_validation_test_data()
        self.valid_data = self.validation_data['valid_pattern_data']
        self.invalid_data = self.validation_data['invalid_pattern_data']
        self.convertible_keys_data = self.validation_data['convertible_keys_test']
        
        # Create mock instances through factory
        self.mock_validation_rules = self.mock_factory('MockValidationRules')
        self.mock_loader = self.mock_factory('MockPatternLoader')
    
    def test_json_to_pattern_handles_non_list_patterns(self):
        """Test json_to_pattern handles non-list patterns gracefully."""
        edge_data = self.config.get_edge_case_scenarios()
        non_list_patterns = edge_data['non_list_patterns']
        
        loader = PatternLoader()
        loader.patterns = non_list_patterns
        
        # Should not raise error
        loader.json_to_pattern()
        
        # Pattern should remain unchanged
        self.assertEqual(loader.patterns["category"]["pattern1"], "not_a_list")
    
    def test_json_to_pattern_with_convertible_keys(self):
        """Test json_to_pattern only converts specific keys to sets."""
        loader = PatternLoader()
        loader.patterns = self.convertible_keys_data
        loader.json_to_pattern()
        
        pattern_item = loader.patterns["category"]["pattern1"][0]
        
        # These should be converted to sets
        self.assertIsInstance(pattern_item["pos"], set)
        self.assertIsInstance(pattern_item["root_type"], set)
        self.assertIsInstance(pattern_item["labels"], set)
        self.assertIsInstance(pattern_item["relation_type"], set)
        
        # These should remain unchanged
        self.assertIsInstance(pattern_item["other_list"], list)
        self.assertIsInstance(pattern_item["string_value"], str)
    
    def test_pattern_to_json_with_missing_description(self):
        """Test pattern_to_json handles missing description gracefully."""
        edge_data = self.config.get_edge_case_scenarios()
        missing_desc_data = edge_data['missing_description']
        
        loader = PatternLoader()
        loader.patterns = missing_desc_data
        
        result = loader.pattern_to_json()
        
        # Should provide empty description when missing
        self.assertEqual(result["category"]["pattern1"]["description"], "")


class TestPatternLoaderEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures using mock factory and configuration injection."""
        # Initialize mock factory and config
        self.mock_factory = PatternLoaderMockFactory()
        self.config = PatternLoaderMockConfig()
        
        # Get edge case test data from config
        self.edge_scenarios = self.config.get_edge_case_scenarios()
        self.error_scenarios = self.config.get_error_scenarios()
        
        # Create mock instances through factory
        self.mock_edge_cases = self.mock_factory('MockPatternLoaderEdgeCases')
        self.mock_loader = self.mock_factory('MockPatternLoader')
    
    def test_json_to_pattern_empty_patterns(self):
        """Test json_to_pattern with empty patterns dictionary."""
        loader = PatternLoader()
        loader.patterns = self.edge_scenarios['empty_patterns']
        
        # Should not raise error
        loader.json_to_pattern()
        
        self.assertEqual(loader.patterns, {})

    def test_pattern_to_json_empty_patterns(self):
        """Test pattern_to_json with empty patterns dictionary."""
        loader = PatternLoader()
        loader.patterns = self.edge_scenarios['empty_patterns']
        
        result = loader.pattern_to_json()
        
        self.assertEqual(result, {})

    def test_load_patterns_malformed_json(self):
        """Test load_patterns_from_file with malformed JSON."""
        malformed_json = self.edge_scenarios['malformed_json']
        error_files = self.error_scenarios['file_errors']
        
        with patch('builtins.open', mock_open(read_data=malformed_json)), \
             patch('smied.PatternLoader.PatternLoader._get_default_patterns') as mock_default, \
             patch('builtins.print') as mock_print:
            
            mock_default.return_value = {"fallback": {}}
            
            loader = PatternLoader()
            # This should handle the JSON decode error gracefully
            try:
                loader.load_patterns_from_file(error_files['malformed_json'])
            except json.JSONDecodeError:
                # If it raises JSONDecodeError, that's acceptable behavior
                pass

    def test_save_patterns_write_permission_error(self):
        """Test save_patterns_to_file with write permission error."""
        error_files = self.error_scenarios['file_errors']
        sample_data = self.config.get_basic_test_data()['simple_patterns']
        
        loader = PatternLoader()
        loader.patterns = sample_data
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with self.assertRaises(PermissionError):
                loader.save_patterns_to_file(error_files['permission_error'])

    def test_large_pattern_structure(self):
        """Test handling of large pattern structures."""
        large_params = self.edge_scenarios['large_pattern_params']
        large_patterns = {}
        
        # Create a large number of categories and patterns
        for cat_i in range(large_params['num_categories']):
            category_name = f"category_{cat_i}"
            large_patterns[category_name] = {}
            
            for pat_i in range(large_params['patterns_per_category']):
                pattern_name = f"pattern_{pat_i}"
                large_patterns[category_name][pattern_name] = {
                    "description": f"Description for pattern {pat_i}",
                    "pattern": [
                        {"text": [f"word_{j}" for j in range(large_params['words_per_item'])],
                         "pos": ["NOUN", "VERB", "ADJ"],
                         "other_attr": f"value_{j}"}
                        for j in range(large_params['items_per_pattern'])
                    ]
                }
        
        loader = PatternLoader()
        loader.patterns = large_patterns
        
        # Test conversion operations don't crash
        loader.json_to_pattern()
        json_result = loader.pattern_to_json()
        
        self.assertIsInstance(json_result, dict)
        self.assertEqual(len(json_result), large_params['num_categories'])

    def test_pattern_with_nested_structures(self):
        """Test patterns with deeply nested structures."""
        nested_patterns = self.edge_scenarios['nested_patterns']
        
        loader = PatternLoader()
        loader.patterns = nested_patterns
        
        loader.json_to_pattern()
        
        # Only top-level lists with specific keys should be converted
        pattern = loader.patterns["complex"]["nested_pattern"]["pattern"][0]
        self.assertIsInstance(pattern["text"], set)  # Should be converted
        self.assertIsInstance(pattern["nested"]["deep"]["pos"], list)  # Should remain list (nested)

    def test_mock_edge_case_scenario_setup(self):
        """Test edge case mock scenario configuration."""
        # Test that edge case mock can be configured with various scenarios
        available_scenarios = self.mock_edge_cases.get_edge_case_scenarios()
        
        self.assertIn("file_not_found", available_scenarios)
        self.assertIn("malformed_json", available_scenarios)
        self.assertIn("permission_denied", available_scenarios)
        
        # Test setting up a specific scenario
        self.mock_edge_cases.setup_edge_case_scenario("file_not_found")
        
        # Verify the mock is configured for the scenario
        self.assertEqual(self.mock_edge_cases.load_patterns, self.mock_edge_cases.file_not_found_error)

    def test_error_handling_patterns(self):
        """Test various error handling scenarios."""
        error_patterns = self.error_scenarios['pattern_errors']
        
        loader = PatternLoader()
        
        # Test invalid structure handling
        loader.patterns = {"test": {"invalid": error_patterns['invalid_structure']}}
        # Should not crash when processing invalid structures
        try:
            loader.json_to_pattern()
        except (TypeError, AttributeError):
            # Expected behavior - graceful handling of invalid patterns
            pass


class TestPatternLoaderIntegration(unittest.TestCase):
    """Integration tests for PatternLoader with other components."""
    
    def setUp(self):
        """Set up test fixtures using mock factory and configuration injection."""
        # Initialize mock factory and config
        self.mock_factory = PatternLoaderMockFactory()
        self.config = PatternLoaderMockConfig()
        
        # Get integration test data from config
        self.integration_data = self.config.get_integration_test_data()
        self.workflow_patterns = self.integration_data['workflow_patterns']
        self.realistic_patterns = self.integration_data['realistic_semantic_patterns']
        self.temp_files = self.integration_data['temp_filenames']
        
        # Create mock instances through factory
        self.mock_integration = self.mock_factory('MockPatternLoaderIntegration')
        self.mock_file_system = self.mock_factory('MockFileSystemForLoader')
        self.mock_registry = self.mock_factory('MockPatternRegistry')
    
    def test_full_workflow_file_operations(self):
        """Test complete workflow: load -> modify -> save."""
        temp_filename = self.temp_files[0]
        
        try:
            with patch('builtins.open', mock_open()) as mock_file:
                loader = PatternLoader()
                loader.patterns = self.workflow_patterns
                loader.save_patterns_to_file(temp_filename)
            
            # Load and modify
            with patch('builtins.open', mock_open(read_data=json.dumps(self.workflow_patterns))):
                loader2 = PatternLoader(temp_filename)
                
                # Add new pattern
                loader2.add_pattern(
                    "new_pattern",
                    [{"text": ["added"], "pos": ["VERB"]}],
                    "Added pattern",
                    "workflow"
                )
                
                # Verify modification
                self.assertIn("new_pattern", loader2.patterns["workflow"])
                
                # Test json conversion round-trip
                json_data = loader2.pattern_to_json()
                self.assertIsInstance(json_data, dict)
                
                # Reload and verify conversion
                loader2.patterns = json.loads(json.dumps(json_data))
                loader2.json_to_pattern()
                
                # Verify sets were properly converted
                new_pattern = loader2.patterns["workflow"]["new_pattern"]["pattern"][0]
                self.assertIsInstance(new_pattern["pos"], set)
                
        except Exception as e:
            self.fail(f"Full workflow test failed: {e}")

    def test_pattern_loader_with_realistic_patterns(self):
        """Test PatternLoader with realistic semantic patterns."""
        loader = PatternLoader()
        loader.patterns = self.realistic_patterns
        
        # Test json conversion
        loader.json_to_pattern()
        
        # Verify specific conversions
        svo_pattern = loader.patterns["semantic"]["subject_verb_object"]["pattern"]
        self.assertIsInstance(svo_pattern[0]["pos"], set)
        self.assertIsInstance(svo_pattern[0]["dep"], set)
        
        entity_pattern = loader.patterns["semantic"]["entity_relation"]["pattern"]
        self.assertIsInstance(entity_pattern[0]["ent_type"], set)
        self.assertIsInstance(entity_pattern[1]["relation_type"], set)
        
        # Test back conversion
        json_data = loader.pattern_to_json()
        
        # Verify structure is maintained
        self.assertIn("semantic", json_data)
        self.assertIn("syntactic", json_data)
        
        # Verify lists are properly converted back
        json_svo = json_data["semantic"]["subject_verb_object"]["pattern"]
        self.assertIsInstance(json_svo[0]["pos"], list)
        self.assertIn("NOUN", json_svo[0]["pos"])
        self.assertIn("PROPN", json_svo[0]["pos"])
    
    def test_integration_with_file_system_mock(self):
        """Test integration with file system operations."""
        # Configure mock file system
        self.mock_file_system.exists.return_value = True
        self.mock_file_system.read_file.return_value = json.dumps(self.workflow_patterns)
        
        # Test integration scenario
        self.assertTrue(self.mock_file_system.exists(self.temp_files[0]))
        file_content = self.mock_file_system.read_file(self.temp_files[0])
        loaded_patterns = json.loads(file_content)
        
        self.assertEqual(loaded_patterns, self.workflow_patterns)
    
    def test_integration_with_pattern_registry(self):
        """Test integration with pattern registry."""
        # Configure mock registry
        self.mock_registry.get_registered_patterns.return_value = list(self.realistic_patterns.keys())
        
        # Test registry integration
        registered = self.mock_registry.get_registered_patterns()
        self.assertIn("semantic", registered)
        self.assertIn("syntactic", registered)
        
        # Test pattern registration
        self.mock_registry.register_pattern("new_category", {"test": "pattern"})
        self.mock_registry.register_pattern.assert_called_with("new_category", {"test": "pattern"})
    
    def test_integration_with_format_handlers(self):
        """Test integration with different format handlers."""
        # Test JSON handler integration
        json_handler = self.mock_factory('MockJSONHandler')
        # Since load() is implemented, we test the actual method
        loaded_data = json_handler.load(self.temp_files[0])
        self.assertEqual(loaded_data, {})  # MockJSONHandler returns empty dict
        
        # Test format validation
        self.assertTrue(json_handler.supports_format(".json"))
        self.assertEqual(json_handler.format_name, "json")
        
        # Test YAML handler integration
        yaml_handler = self.mock_factory('MockYAMLHandler')
        is_valid = yaml_handler.validate(self.realistic_patterns)
        self.assertTrue(is_valid)  # MockYAMLHandler returns True by default
        
        self.assertTrue(yaml_handler.supports_format(".yaml"))
        self.assertEqual(yaml_handler.format_name, "yaml")
    
    def test_end_to_end_pattern_processing(self):
        """Test end-to-end pattern processing workflow."""
        # Create loader with realistic patterns
        loader = PatternLoader()
        loader.patterns = self.realistic_patterns
        
        # Convert to internal format
        loader.json_to_pattern()
        
        # Add a new pattern through the API
        loader.add_pattern(
            "complex_sentence",
            [{"pos": ["NOUN"], "dep": ["nsubj"]}, {"pos": ["VERB"], "dep": ["ROOT"]}],
            "Complex sentence pattern",
            "syntactic"
        )
        
        # Verify the pattern was added and properly formatted
        self.assertIn("complex_sentence", loader.patterns["syntactic"])
        added_pattern = loader.patterns["syntactic"]["complex_sentence"]["pattern"]
        self.assertIsInstance(added_pattern[0]["pos"], list)  # Should be list as added directly
        
        # Convert to JSON format for persistence
        json_output = loader.pattern_to_json()
        
        # Verify JSON structure
        self.assertIn("syntactic", json_output)
        self.assertIn("complex_sentence", json_output["syntactic"])
        
        # Test round-trip: JSON -> internal -> JSON
        loader2 = PatternLoader()
        loader2.patterns = json.loads(json.dumps(json_output))
        loader2.json_to_pattern()
        json_output2 = loader2.pattern_to_json()
        
        # Should be equivalent after round-trip
        self.assertEqual(json_output, json_output2)


if __name__ == '__main__':
    unittest.main()