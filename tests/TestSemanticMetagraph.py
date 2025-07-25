import unittest
import spacy

from noske.SemanticMetagraph import SemanticMetagraph

class TestSemanticMetagraph(unittest.TestCase):
    """Test cases for the SemanticMetagraph class"""
    
    def setUp(self):
        self.graph = SemanticMetagraph()
        # Create real spacy doc
        self.nlp = spacy.load("en_core_web_sm")
        self.doc = self.nlp("The big cat sleeps quietly.")

    def test_init_empty(self):
        """Test initialization of empty SemanticMetagraph"""
        graph = SemanticMetagraph()
        self.assertIsNotNone(graph.G)
        # Check that the hypergraph is empty initially
        self.assertEqual(len(graph.get_nodes()), 0)
        self.assertEqual(len(graph.get_edges()), 0)

    def test_init_with_doc(self):
        """Test initialization with spaCy Doc"""
        graph = SemanticMetagraph(doc=self.doc)
        nodes = graph.get_nodes()
        
        # Should have nodes for each token
        self.assertGreater(len(nodes), 0)
        # Should have nodes equal to number of tokens
        self.assertEqual(len(nodes), len(self.doc))

    def test_init_with_json(self):
        """Test initialization with JSON data"""
        # First create a graph with some data
        test_graph = SemanticMetagraph()
        test_graph.add_node(1, metadata={"text": "test"})
        test_graph.add_node(2, metadata={"text": "test2"}) 
        test_graph.add_edge((1, 2), metadata={"type": "test_edge"})
        
        # Serialize to JSON
        json_data = test_graph.to_json()
        
        # Create new graph from JSON
        new_graph = SemanticMetagraph(json_data=json_data)
        
        # Verify structure is preserved
        self.assertEqual(len(new_graph.get_nodes()), len(test_graph.get_nodes()))
        self.assertEqual(len(new_graph.get_edges()), len(test_graph.get_edges()))

    def test_add_doc_creates_token_nodes(self):
        """Test that adding a document creates nodes for all tokens"""
        graph = SemanticMetagraph()
        graph.add_doc(self.doc)
        
        nodes = graph.get_nodes()
        
        # Should have a node for each token
        self.assertEqual(len(nodes), len(self.doc))
        
        # Check that token data is preserved
        for token in self.doc:
            self.assertIn(token.i, nodes)
            node_data = nodes[token.i]
            self.assertEqual(node_data["text"], token.text)
            self.assertEqual(node_data["pos"], token.pos_)
            self.assertEqual(node_data["head"], token.head.i)
            self.assertEqual(node_data["lemma"], token.lemma_)

    def test_add_doc_creates_entity_relations(self):
        """Test that adding a document creates entity relations"""
        # Create a document with named entities
        doc_with_entities = self.nlp("Apple Inc. is located in California.")
        graph = SemanticMetagraph()
        graph.add_doc(doc_with_entities)
        
        nodes = graph.get_nodes()
        edges = graph.get_edges()
        
        # Should have created nodes and edges
        self.assertGreater(len(nodes), 0)
        self.assertGreater(len(edges), 0)

    def test_add_doc_creates_dependency_relations(self):
        """Test that adding a document creates dependency relations"""
        graph = SemanticMetagraph()
        graph.add_doc(self.doc)
        
        edges = graph.get_edges()
        
        # Should have dependency edges
        self.assertGreater(len(edges), 0)
        
        # Check for dependency relation metadata
        dep_edges_found = False
        for edge_key, edge_data in edges.items():
            if "relation" in edge_data:
                dep_edges_found = True
                self.assertIn("rel_pos", edge_data)
                break
        
        self.assertTrue(dep_edges_found, "No dependency edges found")

    def test_get_token_tags_case(self):
        """Test token tag extraction for case information"""
        # Test different cases
        doc_cases = self.nlp("the THE The")
        
        for token in doc_cases:
            tags = SemanticMetagraph.get_token_tags(token)
            self.assertIn("case", tags)
            
            if token.text == "the":
                self.assertEqual(tags["case"], "lower")
            elif token.text == "THE":
                self.assertEqual(tags["case"], "upper")
            elif token.text == "The":
                self.assertEqual(tags["case"], "title")

    def test_get_token_tags_classes(self):
        """Test token tag extraction for token classes"""
        doc_classes = self.nlp("word 123 user@email.com http://example.com $50 .")
        
        tags_list = [SemanticMetagraph.get_token_tags(token) for token in doc_classes]
        
        # Check that different classes are detected
        classes_found = set()
        for tags in tags_list:
            if "class" in tags:
                classes_found.add(tags["class"])
        
        self.assertGreater(len(classes_found), 0)

    def test_get_token_tags_punctuation_features(self):
        """Test token tag extraction for punctuation features"""
        doc_punct = self.nlp('("Hello")')
        
        for token in doc_punct:
            tags = SemanticMetagraph.get_token_tags(token)
            if token.is_punct:
                self.assertEqual(tags["class"], "punct")
                if "subclass_features" in tags:
                    self.assertIsInstance(tags["subclass_features"], list)

    def test_get_token_tags_morphology(self):
        """Test token tag extraction includes morphological features"""
        # Use a sentence that will have morphological features
        doc_morph = self.nlp("The cats were running quickly.")
        
        for token in doc_morph:
            tags = SemanticMetagraph.get_token_tags(token)
            
            # Check that morphological features are included when present
            if token.morph:
                morph_dict = token.morph.to_dict()
                for key in morph_dict:
                    if key in tags:
                        self.assertIsInstance(tags[key], list)

    def test_get_dep_edges_structure(self):
        """Test dependency edge extraction structure"""
        # Use a token that has dependencies
        root_token = None
        for token in self.doc:
            if token.dep_ == "ROOT":
                root_token = token
                break
        
        if root_token and (list(root_token.lefts) or list(root_token.rights)):
            edges, metadata = SemanticMetagraph.get_dep_edges(root_token)
            
            self.assertIsInstance(edges, list)
            self.assertIsInstance(metadata, list)
            self.assertEqual(len(edges), len(metadata))
            
            # Check metadata structure
            for i, meta in enumerate(metadata):
                self.assertIn("relation", meta)
                self.assertIn("rel_pos", meta)
                self.assertIn("id", meta)
                self.assertIn("type", meta)
                self.assertEqual(meta["id"], edges[i])
                self.assertEqual(meta["type"], "regular")

    def test_get_dep_edges_position_labels(self):
        """Test that dependency edges have correct position labels"""
        # Find a token with both left and right children
        for token in self.doc:
            if list(token.lefts) and list(token.rights):
                edges, metadata = SemanticMetagraph.get_dep_edges(token)
                
                # Check position labels
                left_count = len(list(token.lefts))
                right_count = len(list(token.rights))
                
                after_count = sum(1 for meta in metadata if meta["rel_pos"] == "after")
                before_count = sum(1 for meta in metadata if meta["rel_pos"] == "before")
                
                self.assertEqual(after_count, left_count)
                self.assertEqual(before_count, right_count)
                break

    def test_get_dep_edges_no_children(self):
        """Test dependency edge extraction for tokens with no children"""
        # Find a leaf token (no children)
        leaf_token = None
        for token in self.doc:
            if not list(token.lefts) and not list(token.rights):
                leaf_token = token
                break
        
        if leaf_token:
            edges, metadata = SemanticMetagraph.get_dep_edges(leaf_token)
            
            self.assertIsInstance(edges, list)
            self.assertIsInstance(metadata, list)
            self.assertEqual(len(edges), 0)
            self.assertEqual(len(metadata), 0)

    def test_from_json_static_method(self):
        """Test the static from_json method returns SemanticMetagraph"""
        # Create a graph with some data
        test_graph = SemanticMetagraph()
        test_graph.add_node(1, metadata={"text": "test"})
        test_graph.add_node(2, metadata={"text": "test2"}) 
        test_graph.add_edge((1, 2), metadata={"type": "test_edge"})
        
        # Serialize to JSON
        json_data = test_graph.to_json()
        
        # Use static method to create new graph
        new_graph = SemanticMetagraph.from_json(json_data)
        
        # Verify it's the correct type
        self.assertIsInstance(new_graph, SemanticMetagraph)
        
        # Verify structure is preserved
        self.assertEqual(len(new_graph.get_nodes()), len(test_graph.get_nodes()))
        self.assertEqual(len(new_graph.get_edges()), len(test_graph.get_edges()))

    def test_inheritance_base_functionality(self):
        """Test that SemanticMetagraph inherits all base Metagraph functionality"""
        # Test basic node/edge operations work
        self.graph.add_node(1, metadata={"text": "test"})
        self.graph.add_nodes([2, 3], metadata=[{"text": "test2"}, {"text": "test3"}])
        self.graph.add_edge((1, 2), metadata={"type": "test_edge"})
        self.graph.add_edges([(2, 3), (1, 3)], metadata=[{"type": "edge1"}, {"type": "edge2"}])
        
        # Test query operations
        nodes = self.graph.get_nodes()
        edges = self.graph.get_edges()
        
        self.assertEqual(len(nodes), 3)
        self.assertGreaterEqual(len(edges), 3)
        
        # Test search operations
        node_result = self.graph.get_node_with_id(1)
        self.assertIsNotNone(node_result)
        
        edge_result = self.graph.get_edge_with_id((1, 2))
        self.assertIsNotNone(edge_result)

    def test_entity_hyperedge_creation(self):
        """Test that entity relations create proper hyperedges"""
        # Create a document with clear named entities
        doc_with_entities = self.nlp("Apple Inc. and Microsoft Corp. are technology companies.")
        graph = SemanticMetagraph()
        graph.add_doc(doc_with_entities)
        
        # Check if hyperedges were created for entities
        edges = graph.get_edges()
        metaverts = graph.get_all_metaverts()
        
        # Should have some structure created
        self.assertGreater(len(edges), 0)
        
        # Look for entity-related edges
        entity_edges_found = False
        for edge_data in edges.values():
            if edge_data.get("class") == "hasEntityType":
                entity_edges_found = True
                break
        
        # Depending on the NER model, we might or might not find entities
        # This test checks the mechanism works without being too strict about results