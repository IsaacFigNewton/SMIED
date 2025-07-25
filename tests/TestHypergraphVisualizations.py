import unittest
import numpy as np
from typing import Dict, List, Union
from noske.hypergraphx.visualizations import get_hyperedge_styling_data

class TestHypergraphVisualizations(unittest.TestCase):
    def test_three_node_hyperedge(self):
        # Test data
        hye = [0, 1, 2]
        pos = {
            0: (0, 0),
            1: (1, 0), 
            2: (0.5, 1)
        }
        colors = {2: "#FFBC79"}
        face_colors = {2: "#79BCFF"}
        
        # Execute
        x1, y1, color, facecolor = get_hyperedge_styling_data(
            hye, pos, colors, face_colors
        )
        
        # Verify 
        self.assertGreater(len(x1), 0)  # Should have coordinates
        self.assertGreater(len(y1), 0)
        self.assertEqual(len(x1), len(y1))  # Equal number of x,y points
        self.assertEqual(color, "#FFBC79")  # Predefined color used
        self.assertEqual(facecolor, "#79BCFF")

    def test_four_node_hyperedge(self):
        hye = [0, 1, 2, 3]
        pos = {
            0: (0, 0),
            1: (1, 0),
            2: (1, 1),
            3: (0, 1)
        }
        colors = {3: "#4C9F4C"}
        face_colors = {3: "#4C9F4C"}
        
        x1, y1, color, facecolor = get_hyperedge_styling_data(
            hye, pos, colors, face_colors
        )
        
        self.assertGreater(len(x1), 0)
        self.assertGreater(len(y1), 0)
        self.assertEqual(color, "#4C9F4C")
        self.assertEqual(facecolor, "#4C9F4C")

    def test_undefined_order_color_generation(self):
        hye = [0, 1, 2, 3, 4]  # 4th order (5 nodes)
        pos = {i: (i, i) for i in range(5)}
        colors = {2: "#FFBC79"}  # No color defined for order 4
        face_colors = {2: "#79BCFF"}
        
        x1, y1, color, facecolor = get_hyperedge_styling_data(
            hye, pos, colors, face_colors
        )
        
        # Verify random color generation
        self.assertTrue(color.startswith('#'))
        self.assertEqual(len(color), 7)  # Format #RRGGBB
        self.assertTrue(facecolor.startswith('#'))
        self.assertEqual(len(facecolor), 7)

    def test_clockwise_ordering(self):
        # Test points are ordered clockwise
        hye = [0, 1, 2]
        pos = {
            0: (0, 0),
            1: (1, 0),
            2: (0.5, 1)
        }
        colors = {2: "#FFBC79"}
        face_colors = {2: "#79BCFF"}
        
        x1, y1, *_ = get_hyperedge_styling_data(
            hye, pos, colors, face_colors
        )
        
        # Check points are in clockwise order
        angles = []
        center_x = np.mean(x1)
        center_y = np.mean(y1)
        for i in range(len(x1)):
            angle = np.arctan2(y1[i] - center_y, x1[i] - center_x)
            angles.append(angle)
        
        # Check angles are decreasing (clockwise)
        angles_diff = np.diff(angles)
        self.assertTrue(all(diff <= 0 for diff in angles_diff[-2:]))  # Check last points

    def test_empty_inputs(self):
        with self.assertRaises(Exception):
            get_hyperedge_styling_data([], {}, {}, {})

    def test_semantic_hypergraph_visualization(self):
        # Create a simple semantic hypergraph
        from noske.SemanticMetagraph import SemanticMetagraph
        
        h = SemanticMetagraph()
        # Add some test nodes and edges
        h.add_nodes([0, 1, 2], metadata=[
            {"text": "The", "pos": "DET"},
            {"text": "cat", "pos": "NOUN"},
            {"text": "sleeps", "pos": "VERB"}
        ])
        h.add_edge([0, 1, 2], metadata={"type": "sentence"})
        
        # Test visualization data
        pos = {0: (0,0), 1: (1,0), 2: (0.5,1)}
        colors = {2: "#FFBC79"} 
        face_colors = {2: "#79BCFF"}
        
        x1, y1, color, facecolor = get_hyperedge_styling_data(
            [0, 1, 2], pos, colors, face_colors
        )
        
        self.assertGreater(len(x1), 0)
        self.assertGreater(len(y1), 0)
        self.assertEqual(color, "#FFBC79")
        self.assertEqual(facecolor, "#79BCFF")

    def test_semantic_hypergraph_complex(self):
        from noske.SemanticMetagraph import SemanticMetagraph
        
        h = SemanticMetagraph()
        # Create a more complex structure with nested hyperedges
        h.add_nodes([0,1,2,3,4], metadata=[
            {"text": "The", "pos": "DET"},
            {"text": "big", "pos": "ADJ"},
            {"text": "black", "pos": "ADJ"}, 
            {"text": "cat", "pos": "NOUN"},
            {"text": "sleeps", "pos": "VERB"}
        ])
        h.add_edge([0,1,2,3], metadata={"type": "noun_phrase"})
        h.add_edge([0,1,2,3,4], metadata={"type": "sentence"})

        pos = {i: (i/2, i%2) for i in range(5)}
        colors = {3: "#FFBC79", 4: "#4C9F4C"}
        face_colors = {3: "#79BCFF", 4: "#9F4C4C"}

        x1, y1, color, facecolor = get_hyperedge_styling_data(
            [0,1,2,3,4], pos, colors, face_colors
        )

        self.assertGreater(len(x1), 0) 
        self.assertEqual(len(x1), len(y1))
        self.assertEqual(color, "#4C9F4C")
        self.assertEqual(facecolor, "#9F4C4C")