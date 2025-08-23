import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.PatternLoader import PatternLoader
from tests.mocks.pattern_loader_mocks import PatternLoaderMockFactory


class TestPatternLoader(unittest.TestCase):
    """Test the PatternLoader class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize mock factory
        self.mock_factory = PatternLoaderMockFactory()
        
        self.sample_patterns = {
            "test_category": {
                "test_pattern": {
                    "description": "Test pattern",
                    "pattern": [
                        {"text": ["cat"], "pos": ["NOUN"]},
                        {"relation": ["subject"]}
                    ]
                }
            }
        }

    @patch('smied.PatternLoader.PatternLoader._get_default_patterns')
    def test_initialization_default(self, mock_get_default):
        """Test PatternLoader initialization with default patterns"""
        mock_get_default.return_value = self.sample_patterns
        
        with patch.object(PatternLoader, 'json_to_pattern'):
            loader = PatternLoader()
            
            mock_get_default.assert_called_once()
            self.assertEqual(loader.patterns, self.sample_patterns)

    @patch('smied.PatternLoader.PatternLoader.load_patterns_from_file')
    def test_initialization_with_file(self, mock_load):
        """Test PatternLoader initialization with patterns file"""
        with patch.object(PatternLoader, 'json_to_pattern'):
            loader = PatternLoader(patterns_file="test.json")
            
            mock_load.assert_called_once_with("test.json")

    def test_load_patterns_from_file_success(self):
        """Test load_patterns_from_file with successful file load"""
        mock_file_content = json.dumps(self.sample_patterns)
        
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            loader = PatternLoader()
            loader.load_patterns_from_file("test.json")
            
            self.assertEqual(loader.patterns, self.sample_patterns)

    @patch('smied.PatternLoader.PatternLoader._get_default_patterns')
    def test_load_patterns_from_file_not_found(self, mock_get_default):
        """Test load_patterns_from_file with file not found"""
        mock_get_default.return_value = {"default": {"pattern": {}}}
        
        with patch('builtins.open', side_effect=FileNotFoundError), \
             patch('builtins.print') as mock_print:
            
            loader = PatternLoader()
            loader.load_patterns_from_file("nonexistent.json")
            
            mock_print.assert_called()
            mock_get_default.assert_called()

    def test_save_patterns_to_file(self):
        """Test save_patterns_to_file method"""
        loader = PatternLoader()
        loader.patterns = self.sample_patterns
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch.object(loader, 'pattern_to_json') as mock_to_json:
            
            mock_to_json.return_value = {"converted": "patterns"}
            
            loader.save_patterns_to_file("output.json")
            
            mock_file.assert_called_once_with("output.json", 'w', encoding='utf-8')
            mock_to_json.assert_called_once()

    def test_json_to_pattern_conversion(self):
        """Test json_to_pattern converts lists to sets correctly"""
        json_patterns = {
            "category": {
                "pattern1": [
                    {"pos": ["NOUN", "VERB"], "text": "test"},
                    {"relation": ["subject"], "other": "value"}
                ]
            }
        }
        
        loader = PatternLoader()
        loader.patterns = json_patterns
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
        """Test pattern_to_json converts sets back to lists"""
        set_patterns = {
            "category": {
                "pattern1": {
                    "description": "Test description",
                    "pattern": [
                        {"pos": {"NOUN", "VERB"}, "text": "test"},
                        {"relation": {"subject"}, "other": "value"}
                    ]
                }
            }
        }
        
        loader = PatternLoader()
        loader.patterns = set_patterns
        
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
        """Test add_pattern creates new category if needed"""
        loader = PatternLoader()
        loader.patterns = {}
        
        test_pattern = [{"text": "test"}]
        loader.add_pattern(
            name="new_pattern",
            pattern=test_pattern,
            description="New pattern",
            category="new_category"
        )
        
        self.assertIn("new_category", loader.patterns)
        self.assertIn("new_pattern", loader.patterns["new_category"])
        
        added_pattern = loader.patterns["new_category"]["new_pattern"]
        self.assertEqual(added_pattern["description"], "New pattern")
        self.assertEqual(added_pattern["pattern"], test_pattern)

    def test_add_pattern_existing_category(self):
        """Test add_pattern adds to existing category"""
        loader = PatternLoader()
        loader.patterns = {"existing": {}}
        
        test_pattern = [{"pos": ["NOUN"]}]
        loader.add_pattern(
            name="another_pattern",
            pattern=test_pattern,
            description="Another pattern",
            category="existing"
        )
        
        self.assertIn("another_pattern", loader.patterns["existing"])
        
        added_pattern = loader.patterns["existing"]["another_pattern"]
        self.assertEqual(added_pattern["description"], "Another pattern")
        self.assertEqual(added_pattern["pattern"], test_pattern)

    def test_add_pattern_default_category(self):
        """Test add_pattern uses default category"""
        loader = PatternLoader()
        loader.patterns = {}
        
        test_pattern = [{"text": "default"}]
        loader.add_pattern(
            name="default_pattern",
            pattern=test_pattern
        )
        
        self.assertIn("custom", loader.patterns)
        self.assertIn("default_pattern", loader.patterns["custom"])

    def test_add_pattern_empty_description(self):
        """Test add_pattern with empty description"""
        loader = PatternLoader()
        loader.patterns = {}
        
        test_pattern = [{"text": "test"}]
        loader.add_pattern(
            name="no_desc_pattern",
            pattern=test_pattern,
            category="test"
        )
        
        added_pattern = loader.patterns["test"]["no_desc_pattern"]
        self.assertEqual(added_pattern["description"], "")

    @patch('smied.patterns')
    @patch('smied.PatternLoader.files')
    def test_get_default_patterns_success(self, mock_files, mock_patterns):
        """Test _get_default_patterns loads patterns successfully"""
        # Mock the resource file system
        mock_resource = Mock()
        mock_resource.open.return_value.__enter__.return_value.read.return_value = json.dumps({
            "test_pattern": {"description": "test", "pattern": []}
        })
        mock_files.return_value.joinpath.return_value = mock_resource
        
        loader = PatternLoader()
        
        with patch.object(loader, 'json_to_pattern'):
            result = loader._get_default_patterns()
        
        self.assertIsInstance(result, dict)

    @patch('smied.patterns')
    @patch('smied.PatternLoader.files')
    def test_get_default_patterns_file_not_found(self, mock_files, mock_patterns):
        """Test _get_default_patterns handles missing files"""
        mock_files.return_value.joinpath.return_value.open.side_effect = FileNotFoundError
        
        loader = PatternLoader()
        
        with patch('builtins.print') as mock_print, \
             patch.object(loader, 'json_to_pattern'):
            result = loader._get_default_patterns()
        
        self.assertIsInstance(result, dict)
        # Should print error messages for missing files
        self.assertGreater(mock_print.call_count, 0)

    def test_str_representation(self):
        """Test __str__ method returns JSON representation"""
        loader = PatternLoader()
        loader.patterns = self.sample_patterns
        
        with patch.object(loader, 'pattern_to_json') as mock_to_json:
            mock_to_json.return_value = {"converted": "data"}
            
            result = str(loader)
            
            mock_to_json.assert_called_once()
            # Should be valid JSON string
            parsed = json.loads(result)
            self.assertEqual(parsed, {"converted": "data"})


class TestPatternLoaderEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_json_to_pattern_with_convertible_keys(self):
        """Test json_to_pattern only converts specific keys to sets"""
        patterns_with_mixed_keys = {
            "category": {
                "pattern1": [
                    {
                        "pos": ["NOUN", "VERB"],         # Should convert
                        "root_type": ["entity"],         # Should convert  
                        "labels": ["person", "animal"],  # Should convert
                        "relation_type": ["subject"],    # Should convert
                        "other_list": ["keep", "as", "list"],  # Should NOT convert
                        "string_value": "unchanged"      # Should remain unchanged
                    }
                ]
            }
        }
        
        loader = PatternLoader()
        loader.patterns = patterns_with_mixed_keys
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
        """Test pattern_to_json handles missing description gracefully"""
        patterns_no_desc = {
            "category": {
                "pattern1": {
                    "pattern": [{"text": "test"}]
                    # Missing "description" key
                }
            }
        }
        
        loader = PatternLoader()
        loader.patterns = patterns_no_desc
        
        result = loader.pattern_to_json()
        
        # Should provide empty description when missing
        self.assertEqual(result["category"]["pattern1"]["description"], "")

    def test_json_to_pattern_empty_patterns(self):
        """Test json_to_pattern with empty patterns dictionary"""
        loader = PatternLoader()
        loader.patterns = {}
        
        # Should not raise error
        loader.json_to_pattern()
        
        self.assertEqual(loader.patterns, {})

    def test_pattern_to_json_empty_patterns(self):
        """Test pattern_to_json with empty patterns dictionary"""
        loader = PatternLoader()
        loader.patterns = {}
        
        result = loader.pattern_to_json()
        
        self.assertEqual(result, {})

    def test_load_patterns_malformed_json(self):
        """Test load_patterns_from_file with malformed JSON"""
        malformed_json = '{"incomplete": json'
        
        with patch('builtins.open', mock_open(read_data=malformed_json)), \
             patch('smied.PatternLoader.PatternLoader._get_default_patterns') as mock_default, \
             patch('builtins.print') as mock_print:
            
            mock_default.return_value = {"fallback": {}}
            
            loader = PatternLoader()
            # This should handle the JSON decode error gracefully
            try:
                loader.load_patterns_from_file("malformed.json")
            except json.JSONDecodeError:
                # If it raises JSONDecodeError, that's acceptable behavior
                pass

    def test_save_patterns_write_permission_error(self):
        """Test save_patterns_to_file with write permission error"""
        loader = PatternLoader()
        loader.patterns = self.sample_patterns
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with self.assertRaises(PermissionError):
                loader.save_patterns_to_file("readonly.json")

    def test_large_pattern_structure(self):
        """Test handling of large pattern structures"""
        large_patterns = {}
        
        # Create a large number of categories and patterns
        for cat_i in range(10):
            category_name = f"category_{cat_i}"
            large_patterns[category_name] = {}
            
            for pat_i in range(20):
                pattern_name = f"pattern_{pat_i}"
                large_patterns[category_name][pattern_name] = {
                    "description": f"Description for pattern {pat_i}",
                    "pattern": [
                        {"text": [f"word_{j}" for j in range(10)],
                         "pos": ["NOUN", "VERB", "ADJ"],
                         "other_attr": f"value_{j}"}
                        for j in range(5)
                    ]
                }
        
        loader = PatternLoader()
        loader.patterns = large_patterns
        
        # Test conversion operations don't crash
        loader.json_to_pattern()
        json_result = loader.pattern_to_json()
        
        self.assertIsInstance(json_result, dict)
        self.assertEqual(len(json_result), 10)  # 10 categories

    def test_pattern_with_nested_structures(self):
        """Test patterns with deeply nested structures"""
        nested_patterns = {
            "complex": {
                "nested_pattern": {
                    "description": "Complex nested pattern",
                    "pattern": [
                        {
                            "text": ["surface_text"],
                            "nested": {
                                "deep": {
                                    "pos": ["NOUN"],  # This should NOT be converted (not at top level)
                                    "very_deep": ["value"]
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        loader = PatternLoader()
        loader.patterns = nested_patterns
        
        loader.json_to_pattern()
        
        # Only top-level lists with specific keys should be converted
        pattern = loader.patterns["complex"]["nested_pattern"]["pattern"][0]
        self.assertIsInstance(pattern["text"], set)  # Should be converted
        self.assertIsInstance(pattern["nested"]["deep"]["pos"], list)  # Should remain list (nested)


class TestPatternLoaderIntegration(unittest.TestCase):
    """Integration tests for PatternLoader"""
    
    def test_full_workflow_file_operations(self):
        """Test complete workflow: load -> modify -> save"""
        original_patterns = {
            "workflow": {
                "test_pattern": {
                    "description": "Workflow test",
                    "pattern": [{"pos": ["NOUN"], "text": "test"}]
                }
            }
        }
        
        # Save original patterns
        temp_filename = "temp_test_patterns.json"
        
        try:
            with patch('builtins.open', mock_open()) as mock_file:
                loader = PatternLoader()
                loader.patterns = original_patterns
                loader.save_patterns_to_file(temp_filename)
            
            # Load and modify
            with patch('builtins.open', mock_open(read_data=json.dumps(original_patterns))):
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
        """Test PatternLoader with realistic semantic patterns"""
        realistic_patterns = {
            "semantic": {
                "subject_verb_object": {
                    "description": "Basic SVO pattern",
                    "pattern": [
                        {"pos": ["NOUN", "PROPN"], "dep": ["nsubj"]},
                        {"pos": ["VERB"], "dep": ["ROOT"]},
                        {"pos": ["NOUN", "PROPN"], "dep": ["dobj", "pobj"]}
                    ]
                },
                "entity_relation": {
                    "description": "Entity relation pattern",
                    "pattern": [
                        {"ent_type": ["PERSON", "ORG"], "pos": ["NOUN"]},
                        {"relation_type": ["works_for", "part_of"]},
                        {"ent_type": ["ORG", "PLACE"], "pos": ["NOUN"]}
                    ]
                }
            },
            "syntactic": {
                "noun_phrase": {
                    "description": "Noun phrase pattern",
                    "pattern": [
                        {"pos": ["DET"], "optional": True},
                        {"pos": ["ADJ"], "optional": True, "repeat": True},
                        {"pos": ["NOUN", "PROPN"]}
                    ]
                }
            }
        }
        
        loader = PatternLoader()
        loader.patterns = realistic_patterns
        
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


if __name__ == '__main__':
    unittest.main()