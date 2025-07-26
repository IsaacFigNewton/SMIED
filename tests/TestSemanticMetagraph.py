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
        self.assertIsNotNone(graph.G, "Graph object G should not be None after initialization")
        # Check that the hypergraph is empty initially
        nodes = graph.get_nodes()
        edges = graph.get_edges()
        self.assertEqual(len(nodes), 0, 
                        f"Expected empty graph to have 0 nodes, but found {len(nodes)} nodes: {list(nodes.keys())}")
        self.assertEqual(len(edges), 0, 
                        f"Expected empty graph to have 0 edges, but found {len(edges)} edges: {list(edges.keys())}")

    def test_init_with_doc(self):
        """Test initialization with spaCy Doc"""
        graph = SemanticMetagraph(doc=self.doc)
        nodes = graph.get_nodes()
        
        # Should have nodes for each token
        self.assertGreater(len(nodes), 0, 
                          f"Expected graph initialized with doc to have nodes, but found 0 nodes. Doc tokens: {[token.text for token in self.doc]}")
        # Should have nodes equal to number of tokens
        self.assertGreaterEqual(len(nodes), len(self.doc), 
                               f"Expected at least {len(self.doc)} nodes (one per token), but found {len(nodes)} nodes. "
                               f"Doc tokens: {[token.text for token in self.doc]}, Node IDs: {list(nodes.keys())}")

    def test_init_with_json(self):
        """Test initialization with JSON data"""
        # First create a graph with some data
        test_graph = SemanticMetagraph()
        test_graph.add_node(1, metadata={"text": "test"})
        test_graph.add_node(2, metadata={"text": "test2"}) 
        test_graph.add_edge((1, 2), metadata={"type": "test_edge"})
        
        original_nodes = test_graph.get_nodes()
        original_edges = test_graph.get_edges()
        
        # Create new graph from JSON
        json_data = test_graph.to_json()
        new_graph = SemanticMetagraph(json_data=json_data)
        new_nodes = new_graph.get_nodes()
        new_edges = new_graph.get_edges()
        
        # Verify structure is preserved
        self.assertEqual(len(new_nodes), len(original_nodes),
                        f"JSON deserialization failed: expected {len(original_nodes)} nodes but got {len(new_nodes)}. "
                        f"Original: {list(original_nodes.keys())}, New: {list(new_nodes.keys())}")
        self.assertEqual(len(new_edges), len(original_edges),
                        f"JSON deserialization failed: expected {len(original_edges)} edges but got {len(new_edges)}. "
                        f"Original: {list(original_edges.keys())}, New: {list(new_edges.keys())}")

    def test_add_doc_creates_token_nodes(self):
        """Test that adding a document creates nodes for all tokens"""
        graph = SemanticMetagraph()
        graph.add_doc(self.doc)
        
        # Should have a node for each token
        nodes = graph.get_nodes()
        self.assertEqual(len(nodes), len(self.doc),
                        f"Expected {len(self.doc)} nodes (one per token), but found {len(nodes)} nodes. "
                        f"Tokens: {[(i, token.text) for i, token in enumerate(self.doc)]}, "
                        f"Node IDs: {list(nodes.keys())}")
        
        # Check that token data is preserved
        for token in self.doc:
            self.assertIn(token.i, nodes, 
                         f"Missing node for token {token.i} ('{token.text}'). Available nodes: {list(nodes.keys())}")
            
            node_data = nodes[token.i]
            self.assertEqual(node_data["text"], token.text,
                           f"Node {token.i} text mismatch: expected '{token.text}' but got '{node_data.get('text', 'MISSING')}'")
            self.assertEqual(node_data["pos"], token.pos_,
                           f"Node {token.i} POS mismatch: expected '{token.pos_}' but got '{node_data.get('pos', 'MISSING')}'")
            self.assertEqual(node_data["head"], token.head.i,
                           f"Node {token.i} head mismatch: expected {token.head.i} but got {node_data.get('head', 'MISSING')}")
            self.assertEqual(node_data["lemma"], token.lemma_,
                           f"Node {token.i} lemma mismatch: expected '{token.lemma_}' but got '{node_data.get('lemma', 'MISSING')}'")

    def test_add_doc_creates_entity_relations(self):
        """Test that adding a document creates entity relations"""
        # Create a document with named entities
        doc_with_entities = self.nlp("Apple Inc. is located in California.")
        entities_info = [(ent.text, ent.label_) for ent in doc_with_entities.ents]
        
        graph = SemanticMetagraph()
        graph.add_doc(doc_with_entities)
        
        nodes = graph.get_nodes()
        edges = graph.get_edges()
        
        # Should have created nodes and edges
        self.assertGreater(len(nodes), 0,
                          f"Expected nodes to be created for document with entities. "
                          f"Document: '{doc_with_entities.text}', Entities: {entities_info}")
        self.assertGreater(len(edges), 0,
                          f"Expected edges to be created for document with entities. "
                          f"Document: '{doc_with_entities.text}', Entities: {entities_info}, "
                          f"Created {len(nodes)} nodes but 0 edges")

    def test_add_doc_creates_dependency_relations(self):
        """Test that adding a document creates dependency relations"""
        graph = SemanticMetagraph()
        graph.add_doc(self.doc)
        
        # Should have dependency edges
        edges = graph.get_edges()
        dep_info = [(token.i, token.text, token.dep_, token.head.i) for token in self.doc]
        
        self.assertGreater(len(edges), 0,
                          f"Expected dependency edges to be created. Document: '{self.doc.text}', "
                          f"Dependencies: {dep_info}")
        
        # Check for dependency relation metadata
        dep_edges_found = False
        dep_edge_types = []
        for edge_key, edge_data in edges.items():
            if "relation" in edge_data:
                dep_edges_found = True
                dep_edge_types.append(edge_data["relation"])
                self.assertIn("rel_pos", edge_data,
                            f"Dependency edge {edge_key} missing 'rel_pos' field. Edge data: {edge_data}")
                break
        
        self.assertTrue(dep_edges_found, 
                       f"No dependency edges found among {len(edges)} edges. "
                       f"Edge keys: {list(edges.keys())}, "
                       f"Edge types found: {[edge_data.get('type', 'NO_TYPE') for edge_data in edges.values()]}")

    def test_get_token_tags_case(self):
        """Test token tag extraction for case information"""
        # Test different cases
        doc_cases = self.nlp("the THE The")
        
        for token in doc_cases:
            tags = SemanticMetagraph.get_token_tags(token)
            self.assertIn("case", tags,
                         f"Token '{token.text}' missing 'case' tag. Available tags: {list(tags.keys())}")
            
            expected_case = None
            if token.text == "the":
                expected_case = "lower"
            elif token.text == "THE":
                expected_case = "upper"
            elif token.text == "The":
                expected_case = "title"
            
            if expected_case:
                self.assertEqual(tags["case"], expected_case,
                               f"Token '{token.text}' case mismatch: expected '{expected_case}' but got '{tags['case']}'")

    def test_get_token_tags_classes(self):
        """Test token tag extraction for token classes"""
        doc_classes = self.nlp("word 123 user@email.com http://example.com $50 .")
        tags_list = [SemanticMetagraph.get_token_tags(token) for token in doc_classes]
        token_texts = [token.text for token in doc_classes]
        
        # Check that different classes are detected
        classes_found = set()
        class_details = []
        for i, tags in enumerate(tags_list):
            if "class" in tags:
                classes_found.add(tags["class"])
                class_details.append(f"'{token_texts[i]}': {tags['class']}")
        
        self.assertGreater(len(classes_found), 0,
                          f"No token classes detected in varied input. "
                          f"Tokens: {token_texts}, "
                          f"Tag details: {[(token_texts[i], tags) for i, tags in enumerate(tags_list)]}")

    def test_get_token_tags_punctuation_features(self):
        """Test token tag extraction for punctuation features"""
        doc_punct = self.nlp('("Hello")')
        
        for token in doc_punct:
            tags = SemanticMetagraph.get_token_tags(token)
            if token.is_punct:
                self.assertEqual(tags["class"], "punct",
                               f"Punctuation token '{token.text}' should have class 'punct' but got '{tags.get('class', 'MISSING')}'. "
                               f"All tags: {tags}")
                if "subclass_features" in tags:
                    self.assertIsInstance(tags["subclass_features"], list,
                                        f"Punctuation token '{token.text}' subclass_features should be a list but got {type(tags['subclass_features'])}: {tags['subclass_features']}")

    def test_get_token_tags_morphology(self):
        """Test token tag extraction includes morphological features"""
        # Use a sentence that will have morphological features
        doc_morph = self.nlp("The cats were running quickly.")
        
        for token in doc_morph:
            tags = SemanticMetagraph.get_token_tags(token)
            
            # Check that morphological features are included when present
            if token.morph:
                morph_dict = token.morph.to_dict()
                morph_features_found = []
                for key in morph_dict:
                    if key in tags:
                        self.assertIsInstance(tags[key], list,
                                            f"Token '{token.text}' morphological feature '{key}' should be a list but got {type(tags[key])}: {tags[key]}. "
                                            f"Original morph: {morph_dict}")
                        morph_features_found.append(key)

    def test_get_dep_edges_structure(self):
        """Test dependency edge extraction structure"""
        # Use a token that has dependencies
        root_token = None
        tokens_with_children = []
        
        for token in self.doc:
            children = list(token.lefts) + list(token.rights)
            if children:
                tokens_with_children.append((token.text, len(children)))
            if token.dep_ == "ROOT":
                root_token = token
        
        if root_token and (list(root_token.lefts) or list(root_token.rights)):
            edges, metadata = SemanticMetagraph.get_dep_edges(root_token)
            
            self.assertIsInstance(edges, list,
                                f"get_dep_edges should return edges as list, got {type(edges)}")
            self.assertIsInstance(metadata, list,
                                f"get_dep_edges should return metadata as list, got {type(metadata)}")
            self.assertEqual(len(edges), len(metadata),
                           f"Edges and metadata lists should have same length: edges={len(edges)}, metadata={len(metadata)}. "
                           f"Root token: '{root_token.text}', Children: {[child.text for child in root_token.lefts] + [child.text for child in root_token.rights]}")
            
            # Check metadata structure
            for i, meta in enumerate(metadata):
                edge_id = edges[i]
                required_fields = ["relation", "rel_pos", "id", "type"]
                for field in required_fields:
                    self.assertIn(field, meta,
                                f"Edge {i} (id: {edge_id}) missing required field '{field}'. "
                                f"Available fields: {list(meta.keys())}, Full metadata: {meta}")
                
                self.assertEqual(meta["id"], edge_id,
                               f"Edge {i} metadata 'id' mismatch: expected {edge_id} but got {meta['id']}")
                self.assertEqual(meta["type"], "regular",
                               f"Edge {i} metadata 'type' should be 'regular' but got '{meta['type']}'")
        else:
            self.skipTest(f"No suitable root token with children found. "
                         f"Tokens with children: {tokens_with_children}")

    def test_get_dep_edges_position_labels(self):
        """Test that dependency edges have correct position labels"""
        # Find a token with both left and right children
        suitable_token = None
        token_info = []
        
        for token in self.doc:
            lefts = list(token.lefts)
            rights = list(token.rights)
            token_info.append(f"'{token.text}': {len(lefts)} left, {len(rights)} right")
            
            if lefts and rights:
                suitable_token = token
                edges, metadata = SemanticMetagraph.get_dep_edges(token)
                
                # Check position labels
                left_count = len(lefts)
                right_count = len(rights)
                after_count = sum(1 for meta in metadata if meta["rel_pos"] == "after")
                before_count = sum(1 for meta in metadata if meta["rel_pos"] == "before")
                
                self.assertEqual(after_count, left_count,
                               f"Token '{token.text}' should have {left_count} 'after' edges but found {after_count}. "
                               f"Left children: {[child.text for child in lefts]}, "
                               f"Metadata positions: {[meta['rel_pos'] for meta in metadata]}")
                self.assertEqual(before_count, right_count,
                               f"Token '{token.text}' should have {right_count} 'before' edges but found {before_count}. "
                               f"Right children: {[child.text for child in rights]}, "
                               f"Metadata positions: {[meta['rel_pos'] for meta in metadata]}")
                break
        
        if not suitable_token:
            self.skipTest(f"No token found with both left and right children. Token analysis: {token_info}")

    def test_get_dep_edges_no_children(self):
        """Test dependency edge extraction for tokens with no children"""
        # Find a leaf token (no children)
        leaf_token = None
        leaf_candidates = []
        
        for token in self.doc:
            lefts = list(token.lefts)
            rights = list(token.rights)
            if not lefts and not rights:
                leaf_token = token
                leaf_candidates.append(token.text)
        
        if leaf_token:
            edges, metadata = SemanticMetagraph.get_dep_edges(leaf_token)
            self.assertIsInstance(edges, list,
                                f"get_dep_edges should return edges as list for leaf token '{leaf_token.text}', got {type(edges)}")
            self.assertIsInstance(metadata, list,
                                f"get_dep_edges should return metadata as list for leaf token '{leaf_token.text}', got {type(metadata)}")
            self.assertEqual(len(edges), 0,
                           f"Leaf token '{leaf_token.text}' should have 0 edges but got {len(edges)}: {edges}")
            self.assertEqual(len(metadata), 0,
                           f"Leaf token '{leaf_token.text}' should have 0 metadata entries but got {len(metadata)}: {metadata}")
        else:
            self.skipTest(f"No leaf tokens found in document '{self.doc.text}'. "
                         f"All tokens have children.")

    def test_from_json_static_method(self):
        """Test the static from_json method returns SemanticMetagraph"""
        # Create a graph with some data
        test_graph = SemanticMetagraph()
        test_graph.add_node(1, metadata={"text": "test"})
        test_graph.add_node(2, metadata={"text": "test2"}) 
        test_graph.add_edge((1, 2), metadata={"type": "test_edge"})
        
        original_nodes = test_graph.get_nodes()
        original_edges = test_graph.get_edges()
        
        # Serialize to JSON
        json_data = test_graph.to_json()
        
        # Use static method to create new graph
        new_graph = SemanticMetagraph.from_json(json_data)

        # Verify it's the correct type
        self.assertIsInstance(new_graph, SemanticMetagraph,
                            f"from_json should return SemanticMetagraph instance but got {type(new_graph)}")
        
        # Verify structure is preserved
        new_nodes = new_graph.get_nodes()
        new_edges = new_graph.get_edges()
        
        self.assertEqual(len(new_nodes), len(original_nodes),
                        f"Static from_json failed: expected {len(original_nodes)} nodes but got {len(new_nodes)}. "
                        f"Original: {list(original_nodes.keys())}, New: {list(new_nodes.keys())}")
        self.assertEqual(len(new_edges), len(original_edges),
                        f"Static from_json failed: expected {len(original_edges)} edges but got {len(new_edges)}. "
                        f"Original: {list(original_edges.keys())}, New: {list(new_edges.keys())}")

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
        
        self.assertEqual(len(nodes), 3,
                        f"Expected 3 nodes after adding nodes 1, 2, 3 but got {len(nodes)}. "
                        f"Node IDs: {list(nodes.keys())}")
        self.assertEqual(len(edges), 3,
                        f"Expected 3 edges after adding (1,2), (2,3), (1,3) but got {len(edges)}. "
                        f"Edge IDs: {list(edges.keys())}")
        
        # Test search operations
        node_result = self.graph.get_node_with_id(1)
        self.assertIsNotNone(node_result,
                           f"get_node_with_id(1) returned None. Available nodes: {list(nodes.keys())}")
        
        edge_result = self.graph.get_edge_with_id((1, 2))
        self.assertIsNotNone(edge_result,
                           f"get_edge_with_id((1, 2)) returned None. Available edges: {list(edges.keys())}")

    def test_entity_hyperedge_creation(self):
        """Test that entity relations create proper hyperedges"""
        # Create a document with clear named entities
        doc_with_entities = self.nlp("Apple Inc. and Microsoft Corp. are technology companies.")
        entities_info = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc_with_entities.ents]
        
        graph = SemanticMetagraph()
        graph.add_doc(doc_with_entities)
        
        # Check if hyperedges were created for entities
        edges = graph.get_edges()
        metaverts = graph.get_all_metaverts()
        
        # Should have some structure created
        self.assertGreater(len(edges), 0,
                          f"Expected edges to be created for document with entities. "
                          f"Document: '{doc_with_entities.text}', "
                          f"Entities found: {entities_info}")

        # Look for at least one 'hasEntityType' edge, which indicates an entity is present
        entity_edge_found = False
        edge_classes = []
        edge_details = []
        
        for edge_id, edge_data in edges.items():
            edge_class = edge_data.get("class", "NO_CLASS")
            edge_classes.append(edge_class)
            edge_details.append(f"{edge_id}: {edge_class}")
            
            if edge_class == "hasEntityType":
                entity_edge_found = True
                break
        
        self.assertTrue(entity_edge_found,
                       f"Expected at least one 'hasEntityType' edge indicating an entity, but none found. "
                       f"Document entities: {entities_info}, "
                       f"Edge classes found: {set(edge_classes)}, "
                       f"All edges: {edge_details}")