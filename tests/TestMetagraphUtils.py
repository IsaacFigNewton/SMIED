import unittest

from noske.MetagraphUtils import (
    _is_edge,
    _flatten_edge,
    _get_required_node_fields,
    _get_required_edge_fields
)

class TestMetagraphUtils(unittest.TestCase):
    """Test utility functions used by SemanticMetagraph"""
    
    def test_is_edge(self):
        """Test _is_edge function"""
        self.assertTrue(_is_edge((1, 2)))
        self.assertTrue(_is_edge((1, 2, 3)))
        self.assertTrue(_is_edge(((1, 2), 3)))
        
        self.assertFalse(_is_edge(1))
        self.assertFalse(_is_edge("string"))
        self.assertFalse(_is_edge([1, 2]))
        self.assertFalse(_is_edge((1,)))  # Single element tuple
        self.assertFalse(_is_edge(()))    # Empty tuple


    def test_flatten_edge_simple(self):
        """Test _flatten_edge function with simple cases"""
        # Simple edge
        self.assertEqual(_flatten_edge((1, 2)), [1, 2])
        self.assertEqual(_flatten_edge((1, 2, 3)), [1, 2, 3])


    def test_flatten_edge_nested(self):
        """Test _flatten_edge function with nested cases"""
        # Nested edge
        self.assertEqual(_flatten_edge((1, (2, 3))), [1, 2, 3])
        self.assertEqual(_flatten_edge(((1, 2), 3)), [1, 2, 3])
        
        # Deeply nested edge
        self.assertEqual(_flatten_edge(((1, 2), (3, 4))), [1, 2, 3, 4])
        self.assertEqual(_flatten_edge((((1, 2), 3), 4)), [1, 2, 3, 4])


    def test_flatten_edge_complex(self):
        """Test _flatten_edge function with complex nested structures"""
        # Complex nesting
        complex_edge = (((1, 2), (3, 4)), (5, (6, 7)))
        expected = [1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(_flatten_edge(complex_edge), expected)


    def test_get_required_node_fields_regular(self):
        """Test _get_required_node_fields function for regular nodes"""
        fields = _get_required_node_fields(1, "regular")
        self.assertEqual(fields, {"id": 1, "type": "regular"})
        
        # Test with string ID
        fields = _get_required_node_fields("node1", "regular")
        self.assertEqual(fields, {"id": "node1", "type": "regular"})


    def test_get_required_node_fields_meta(self):
        """Test _get_required_node_fields function for meta nodes"""
        fields = _get_required_node_fields((1, 2), "meta")
        self.assertEqual(fields, {"id": (1, 2), "type": "meta"})
        
        # Test with complex tuple ID
        fields = _get_required_node_fields(((1, 2), 3), "meta")
        self.assertEqual(fields, {"id": ((1, 2), 3), "type": "meta"})


    def test_get_required_node_fields_invalid_type(self):
        """Test _get_required_node_fields function with invalid type"""
        with self.assertRaises(ValueError) as context:
            _get_required_node_fields(1, "invalid")
        
        self.assertIn("Invalid node type", str(context.exception))
        self.assertIn("invalid", str(context.exception))


    def test_get_required_edge_fields_all_types(self):
        """Test _get_required_edge_fields function for all valid types"""
        for edge_type in ["regular", "hyper", "meta", "metavert_to_hye", "hye_to_metavert"]:
            fields = _get_required_edge_fields((1, 2), edge_type)
            self.assertEqual(fields, {"id": (1, 2), "type": edge_type})


    def test_get_required_edge_fields_complex_ids(self):
        """Test _get_required_edge_fields function with complex IDs"""
        # Hyperedge ID
        fields = _get_required_edge_fields((1, 2, 3), "hyper")
        self.assertEqual(fields, {"id": (1, 2, 3), "type": "hyper"})
        
        # Complex nested ID
        fields = _get_required_edge_fields(((1, 2), (3, 4)), "meta")
        self.assertEqual(fields, {"id": ((1, 2), (3, 4)), "type": "meta"})


    def test_get_required_edge_fields_invalid_type(self):
        """Test _get_required_edge_fields function with invalid type"""
        with self.assertRaises(ValueError) as context:
            _get_required_edge_fields((1, 2), "invalid_type")
        
        self.assertIn("Invalid edge type", str(context.exception))
        self.assertIn("invalid_type", str(context.exception))