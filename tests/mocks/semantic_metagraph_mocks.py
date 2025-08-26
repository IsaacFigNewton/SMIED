"""
Mock classes for SemanticMetagraph tests following SMIED Testing Framework Design Specifications.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional, Set, Tuple
import json

# Import abstract base classes
from tests.mocks.base.library_wrapper_mock import AbstractLibraryWrapperMock
from tests.mocks.base.integration_mock import AbstractIntegrationMock
from tests.mocks.base.collection_mock import AbstractCollectionMock
from tests.mocks.base.entity_mock import AbstractEntityMock, EntityType
from tests.mocks.base.reasoning_mock import AbstractReasoningMock, InferenceResult
from tests.mocks.base.edge_case_mock import AbstractEdgeCaseMock
from tests.mocks.base.nlp_doc_mock import AbstractNLPDocMock
from tests.mocks.base.nlp_token_mock import AbstractNLPTokenMock


class SemanticMetagraphMockFactory:
    """Factory class for creating SemanticMetagraph mock instances following factory pattern."""
    
    def __init__(self):
        self._mock_classes = {
            'MockSemanticMetagraph': MockSemanticMetagraph,
            'MockSemanticMetagraphValidation': MockSemanticMetagraphValidation,
            'MockSemanticMetagraphEdgeCases': MockSemanticMetagraphEdgeCases,
            'MockSemanticMetagraphIntegration': MockSemanticMetagraphIntegration,
            'MockSpacyDoc': MockSpacyDoc,
            'MockSpacyToken': MockSpacyToken,
            'MockNetworkXGraph': MockNetworkXGraph,
            'MockOntology': MockOntology,
            'MockConcept': MockConcept,
            'MockRelation': MockRelation,
            'MockWordNetIntegration': MockWordNetIntegration,
            'MockKnowledgeBase': MockKnowledgeBase,
            'MockReasoningEngine': MockReasoningEngine,
            'MockSemanticCluster': MockSemanticCluster,
            'MockSemanticPath': MockSemanticPath,
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


class MockSemanticMetagraph(Mock):
    """Mock for basic SemanticMetagraph functionality testing."""
    
    def __init__(self, doc=None, vert_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Core attributes from real SemanticMetagraph
        self.doc = doc
        self.metaverts = {}
        self.current_mv_idx = 0
        
        # Initialize with test data if provided
        if vert_list:
            for mv in vert_list:
                self.metaverts[self.current_mv_idx] = mv
                self.current_mv_idx += 1
        
        # Core methods from real SemanticMetagraph
        self.add_vert = Mock(side_effect=self._mock_add_vert)
        self.remove_vert = Mock(side_effect=self._mock_remove_vert)
        self.to_json = Mock(side_effect=self._mock_to_json)
        self.from_json = Mock(side_effect=self._mock_from_json)
        self.to_nx = Mock(return_value=MockNetworkXGraph())
        self.get_tokens = Mock(side_effect=self._mock_get_tokens)
        self.get_relations = Mock(side_effect=self._mock_get_relations)
        self.plot = Mock()
        
        # Static methods
        self.get_token_tags = Mock(return_value={"case": "title", "type": "word"})
        self.get_dep_edges = Mock(return_value=[])
        
        # Internal methods
        self._build_metaverts_from_doc = Mock(side_effect=self._mock_build_metaverts_from_doc)
    
    def _mock_add_vert(self, vertex_data, metadata=None):
        """Mock implementation of add_vert."""
        if metadata:
            self.metaverts[self.current_mv_idx] = (vertex_data, metadata)
        else:
            self.metaverts[self.current_mv_idx] = (vertex_data,)
        self.current_mv_idx += 1
    
    def _mock_remove_vert(self, vert_idx):
        """Mock implementation of remove_vert."""
        if vert_idx in self.metaverts:
            # Remove vertex and any relations that reference it
            del self.metaverts[vert_idx]
            to_remove = []
            for mv_id, mv in self.metaverts.items():
                if isinstance(mv[0], tuple) and (vert_idx in mv[0]):
                    to_remove.append(mv_id)
            for mv_id in to_remove:
                del self.metaverts[mv_id]
    
    def _mock_to_json(self):
        """Mock implementation of to_json."""
        json_metaverts = []
        for mv_id, mv in self.metaverts.items():
            mv_data = {"id": mv_id}
            if len(mv) == 1:
                mv_data.update({"type": "atomic", "value": mv[0]})
            elif len(mv) == 2:
                if isinstance(mv[0], str):
                    mv_data.update({"type": "atomic", "value": mv[0], "metadata": mv[1]})
                elif isinstance(mv[0], tuple):
                    mv_data.update({"type": "directed", "source": mv[0][0], "target": mv[0][1], "metadata": mv[1]})
            json_metaverts.append(mv_data)
        return {"metaverts": json.dumps(json_metaverts, indent=4)}
    
    def _mock_from_json(self, json_data):
        """Mock implementation of from_json."""
        # Returns a new mock instance
        return MockSemanticMetagraph()
    
    def _mock_build_metaverts_from_doc(self, doc):
        """Mock implementation of _build_metaverts_from_doc."""
        metaverts = []
        if hasattr(doc, '__len__'):  # Simple mock doc with length
            for i in range(len(doc)):
                metaverts.append((f"token_{i}", {"pos": "NOUN", "text": f"token_{i}", "idx": i}))
        return metaverts
    
    def _mock_get_tokens(self):
        """Mock implementation of get_tokens."""
        tokens = []
        for i, mv in self.metaverts.items():
            if isinstance(mv[0], str) and len(mv) > 1 and 'idx' in mv[1]:
                tokens.append({
                    "metavert_idx": i,
                    "token_idx": mv[1]['idx'],
                    "text": mv[0],
                    "metadata": mv[1]
                })
        return tokens
    
    def _mock_get_relations(self):
        """Mock implementation of get_relations."""
        relations = []
        for i, mv in self.metaverts.items():
            if len(mv) > 1 and 'relation' in mv[1]:
                if isinstance(mv[0], tuple):
                    relations.append({
                        "metavert_idx": i,
                        "type": "directed",
                        "source": mv[0][0],
                        "target": mv[0][1],
                        "relation": mv[1]['relation'],
                        "metadata": mv[1]
                    })
        return relations


class MockSemanticMetagraphIntegration(AbstractIntegrationMock):
    """Mock for SemanticMetagraph integration testing with realistic component interaction."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize as both an integration mock and semantic metagraph mock
        self.doc = kwargs.get('doc')
        self.metaverts = {}
        self.current_mv_idx = 0
        
        # Integration components
        self.wordnet_integration = MockWordNetIntegration()
        self.knowledge_base = MockKnowledgeBase()
        self.reasoning_engine = MockReasoningEngine()
        self.spacy_nlp = Mock()  # Mock spaCy pipeline
        
        # Register components
        self.register_component("wordnet", self.wordnet_integration)
        self.register_component("knowledge_base", self.knowledge_base)
        self.register_component("reasoning_engine", self.reasoning_engine)
        self.register_component("spacy_nlp", self.spacy_nlp)
        
        # Mock SemanticMetagraph methods for integration testing
        self.to_nx = Mock(return_value=MockNetworkXGraph())
        self.to_json = Mock(return_value={"metaverts": "[]"})
        self.from_json = Mock(return_value=self)
        self.get_tokens = Mock(return_value=[])
        self.get_relations = Mock(return_value=[])
        
        # Integration-specific methods
        self.sync_with_wordnet = Mock()
        self.query_knowledge_base = Mock(return_value=[])
        self.perform_reasoning = Mock(return_value=[])
        self.full_pipeline_processing = Mock(side_effect=self._mock_full_pipeline)
    
    # Implement required abstract methods from AbstractIntegrationMock
    def setup_integration_components(self) -> Dict[str, Any]:
        """Setup SemanticMetagraph integration components."""
        return {
            'wordnet_integration': self.wordnet_integration,
            'knowledge_base': self.knowledge_base,
            'reasoning_engine': self.reasoning_engine,
            'semantic_metagraph': Mock()
        }
    
    def configure_component_interactions(self) -> None:
        """Configure SemanticMetagraph component interactions."""
        # Configure interactions between components
        pass  # Mock implementation
    
    def validate_integration_state(self) -> bool:
        """Validate SemanticMetagraph integration state."""
        return (hasattr(self, 'wordnet_integration') and 
                hasattr(self, 'knowledge_base') and 
                hasattr(self, 'reasoning_engine') and
                hasattr(self, 'spacy_nlp'))
    
    def _mock_full_pipeline(self, text):
        """Mock implementation of full processing pipeline."""
        # Simulate full pipeline: text -> spaCy -> SemanticMetagraph -> enrichment
        doc = MockSpacyDoc(text)
        # Create basic metaverts
        metaverts = [(token.text, {"pos": token.pos_, "idx": token.i}) for token in doc.tokens]
        # Add some relations
        if len(metaverts) > 1:
            metaverts.append(((0, 1), {"relation": "subject"}))
        return metaverts


class MockOntology(AbstractCollectionMock):
    """Mock ontology for SemanticMetagraph."""
    
    def __init__(self, *args, **kwargs):
        # Initialize with initial data for the collection aspects
        initial_data = kwargs.pop('initial_data', {})
        super().__init__(initial_data=initial_data, *args, **kwargs)
        
        # Set collection-specific attributes
        self.collection_type = "ontology"
        self.is_mutable = True
        self.is_ordered = False
        self.allows_duplicates = False
        
        # Ontology structure
        self.classes = set()
        self.properties = set()
        self.individuals = set()
        
        # Ontology methods
        self.get_class_hierarchy = Mock(return_value={})
        self.get_property_hierarchy = Mock(return_value={})
        self.is_subclass_of = Mock(return_value=False)
        self.get_instances = Mock(return_value=[])
        self.infer_class = Mock(return_value=None)
    
    # Implement required abstract methods from AbstractCollectionMock
    def get_collection_info(self) -> Dict[str, Any]:
        """Get ontology collection information."""
        return {
            'type': 'ontology',
            'classes_count': len(self.classes),
            'properties_count': len(self.properties),
            'individuals_count': len(self.individuals),
            'supports_inference': True,
            'supports_hierarchy': True
        }
    
    def validate_item(self, item: Any) -> bool:
        """Validate that an item can be stored in the ontology."""
        # Accept classes, properties, individuals, or general ontology items
        return True  # Flexible validation for mock
    
    def transform_item(self, item: Any) -> Any:
        """Transform an item before storing it in the ontology."""
        # For ontology, items are stored as-is
        return item


class MockConcept(AbstractEntityMock):
    """Mock concept for SemanticMetagraph."""
    
    def __init__(self, concept_id="c1", label="test_concept", *args, **kwargs):
        super().__init__(entity_id=concept_id, entity_type=EntityType.CONCEPT, *args, **kwargs)
        # Set concept-specific attributes
        self.label = label
        self.name = label
        self.type = "concept"  # Keep for backwards compatibility
        self.semantic_features = {}
        
        # Concept methods
        self.get_semantic_type = Mock(return_value="entity")
        self.get_features = Mock(return_value={})
        self.add_feature = Mock()
        self.remove_feature = Mock()
        self.similarity = Mock(return_value=0.8)
    
    # Implement required abstract methods from AbstractEntityMock
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this concept."""
        return self.label
    
    def validate_entity(self) -> bool:
        """Validate that the concept is consistent and valid."""
        return bool(self.id and self.label)
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this concept."""
        return f"concept:{self.id}:{self.label}"


class MockRelation(AbstractEntityMock):
    """Mock relation for SemanticMetagraph."""
    
    def __init__(self, relation_id="r1", relation_type="relates_to", *args, **kwargs):
        super().__init__(entity_id=relation_id, entity_type=EntityType.RELATION, *args, **kwargs)
        # Set relation-specific attributes
        self.type = relation_type  # Keep for backwards compatibility
        self.name = relation_type
        self.relation_type = relation_type
        self.domain = None
        self.range = None
        
        # Relation methods
        self.is_symmetric = Mock(return_value=False)
        self.is_transitive = Mock(return_value=False)
        self.is_reflexive = Mock(return_value=False)
        self.get_inverse = Mock(return_value=None)
        self.validate = Mock(return_value=True)
    
    # Implement required abstract methods from AbstractEntityMock
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this relation."""
        return self.relation_type
    
    def validate_entity(self) -> bool:
        """Validate that the relation is consistent and valid."""
        return bool(self.id and self.relation_type)
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this relation."""
        return f"relation:{self.id}:{self.relation_type}"


class MockWordNetIntegration(AbstractIntegrationMock, AbstractLibraryWrapperMock):
    """Mock WordNet integration for SemanticMetagraph."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set library-specific attributes
        self.library_name = "wordnet"
        self.library_version = "3.1"
        
        # WordNet integration methods
        self.import_synsets = Mock()
        self.map_concepts = Mock(return_value={})
        self.sync_relations = Mock()
        self.get_wordnet_concept = Mock(return_value=None)
    
    # Implement required abstract methods from AbstractLibraryWrapperMock
    def initialize_library(self, **config) -> bool:
        """Initialize WordNet library."""
        self.is_initialized = True
        return True
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get WordNet library information."""
        return {
            'name': self.library_name,
            'version': self.library_version,
            'type': 'lexical_database',
            'synsets_available': True
        }
    
    def check_compatibility(self) -> bool:
        """Check WordNet compatibility."""
        return True
    
    # Implement required abstract methods from AbstractIntegrationMock
    def setup_integration_components(self) -> Dict[str, Any]:
        """Setup WordNet integration components."""
        return {
            'wordnet': Mock(),
            'synset_mapper': Mock(),
            'relation_syncer': Mock()
        }
    
    def configure_component_interactions(self) -> None:
        """Configure WordNet component interactions."""
        pass  # Mock implementation
    
    def validate_integration_state(self) -> bool:
        """Validate WordNet integration state."""
        return self.is_initialized


class MockKnowledgeBase(AbstractCollectionMock):
    """Mock knowledge base for SemanticMetagraph."""
    
    def __init__(self, *args, **kwargs):
        # Initialize with initial data for the collection aspects
        initial_data = kwargs.pop('initial_data', set())
        super().__init__(initial_data=initial_data, *args, **kwargs)
        
        # Set collection-specific attributes
        self.collection_type = "knowledge_base"
        self.is_mutable = True
        self.is_ordered = False
        self.allows_duplicates = False
        
        # Knowledge base structure
        self.facts = set()
        self.rules = set()
        self.axioms = set()
        
        # KB methods
        self.add_fact = Mock()
        self.add_rule = Mock()
        self.query = Mock(return_value=[])
        self.infer = Mock(return_value=[])
        self.is_consistent = Mock(return_value=True)
    
    # Implement required abstract methods from AbstractCollectionMock
    def get_collection_info(self) -> Dict[str, Any]:
        """Get knowledge base collection information."""
        return {
            'type': 'knowledge_base',
            'facts_count': len(self.facts),
            'rules_count': len(self.rules),
            'axioms_count': len(self.axioms),
            'supports_inference': True,
            'supports_queries': True,
            'is_consistent': True
        }
    
    def validate_item(self, item: Any) -> bool:
        """Validate that an item can be stored in the knowledge base."""
        # Accept facts, rules, axioms, or general knowledge items
        return True  # Flexible validation for mock
    
    def transform_item(self, item: Any) -> Any:
        """Transform an item before storing it in the knowledge base."""
        # For knowledge base, items are stored as-is
        return item


class MockReasoningEngine(AbstractReasoningMock):
    """Mock reasoning engine for SemanticMetagraph."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set reasoning engine specific attributes
        self.engine_name = "semantic_metagraph_reasoning_engine"
        self.engine_version = "1.0"
        
        # Reasoning capabilities
        self.forward_chaining = Mock(return_value=[])
        self.backward_chaining = Mock(return_value=[])
        self.abductive_reasoning = Mock(return_value=[])
        self.analogical_reasoning = Mock(return_value=[])
        
        # Reasoning validation
        self.validate_inference = Mock(return_value=True)
        self.explain_inference = Mock(return_value="")
    
    # Implement required abstract methods from AbstractReasoningMock
    def infer(self, query: Any, context: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """Perform inference on a query."""
        self.inference_count += 1
        
        # Mock inference result
        return InferenceResult(
            conclusion=f"Inferred result for {query}",
            confidence=0.8,
            reasoning_path=["Mock reasoning step 1", "Mock reasoning step 2"],
            evidence=[query],
            metadata={
                'engine': self.engine_name,
                'query_type': str(type(query)),
                'context': context or {}
            }
        )
    
    def compute_similarity(self, entity1: Any, entity2: Any) -> float:
        """Compute similarity between two entities."""
        # Mock similarity computation
        if entity1 == entity2:
            return 1.0
        
        # Use simple hash-based mock similarity
        hash1 = hash(str(entity1)) % 1000
        hash2 = hash(str(entity2)) % 1000
        similarity = 1.0 - abs(hash1 - hash2) / 1000.0
        
        return max(0.0, similarity)
    
    def explain_reasoning(self, inference_result: InferenceResult) -> List[str]:
        """Generate explanation for reasoning process."""
        explanations = [
            f"Query processed: {inference_result.metadata.get('query_type', 'unknown')}",
            f"Confidence level: {inference_result.confidence}",
            f"Evidence considered: {len(inference_result.evidence)} items",
            f"Reasoning steps: {len(inference_result.reasoning_path)}"
        ]
        
        # Add reasoning path details
        for i, step in enumerate(inference_result.reasoning_path, 1):
            explanations.append(f"Step {i}: {step}")
        
        return explanations


class MockSemanticCluster(AbstractEntityMock):
    """Mock semantic cluster for SemanticMetagraph."""
    
    def __init__(self, cluster_id="cluster1", *args, **kwargs):
        super().__init__(entity_id=cluster_id, entity_type=EntityType.CLUSTER, *args, **kwargs)
        # Set cluster-specific attributes
        self.name = f"cluster_{cluster_id}"
        self.concepts = set()
        self.centroid = None
        
        # Cluster methods
        self.add_concept = Mock()
        self.remove_concept = Mock()
        self.compute_centroid = Mock()
        self.split_cluster = Mock(return_value=[])
        self.merge_with = Mock()
    
    # Implement required abstract methods from AbstractEntityMock
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this cluster."""
        return len(self.concepts)
    
    def validate_entity(self) -> bool:
        """Validate that the cluster is consistent and valid."""
        return bool(self.id)
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this cluster."""
        return f"cluster:{self.id}:{len(self.concepts)}:0.8"


class MockSemanticPath(AbstractEntityMock):
    """Mock semantic path for SemanticMetagraph."""
    
    def __init__(self, path_id="path1", *args, **kwargs):
        super().__init__(entity_id=path_id, entity_type=EntityType.PATH, *args, **kwargs)
        # Set path-specific attributes
        self.name = f"path_{path_id}"
        self.nodes = []
        self.edges = []
        self.length = 0
        self.semantic_weight = 1.0
        
        # Path methods
        self.add_node = Mock()
        self.add_edge = Mock()
        self.get_length = Mock(return_value=self.length)
        self.get_weight = Mock(return_value=self.semantic_weight)
        self.reverse = Mock()
        self.validate = Mock(return_value=True)
    
    # Implement required abstract methods from AbstractEntityMock
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this path."""
        return self.length
    
    def validate_entity(self) -> bool:
        """Validate that the path is consistent and valid."""
        return bool(self.id and len(self.nodes) >= 0 and len(self.edges) >= 0)
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this path."""
        return f"path:{self.id}:{len(self.nodes)}:{len(self.edges)}:{self.semantic_weight}"


class MockSemanticMetagraphValidation(MockSemanticMetagraph):
    """Mock for SemanticMetagraph validation testing scenarios."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override methods for validation testing
        self.validate_graph = Mock(side_effect=self._mock_validate_graph)
        self.validate_vert = Mock(side_effect=self._mock_validate_vert)
        self.canonicalize_vert = Mock(side_effect=self._mock_canonicalize_vert)
        
    def _mock_validate_graph(self, vert_list):
        """Mock graph validation with configurable behavior."""
        if hasattr(self, '_validation_should_fail') and self._validation_should_fail:
            raise ValueError("Mock validation failure")
        return True
    
    def _mock_validate_vert(self, mv_idx, mv):
        """Mock vertex validation."""
        if hasattr(self, '_vertex_validation_should_fail') and self._vertex_validation_should_fail:
            raise ValueError(f"Mock vertex validation failure for {mv_idx}")
        return True
    
    def _mock_canonicalize_vert(self, mv):
        """Mock vertex canonicalization."""
        return mv
    
    def set_validation_failure(self, should_fail=True):
        """Configure mock to simulate validation failures."""
        self._validation_should_fail = should_fail
    
    def set_vertex_validation_failure(self, should_fail=True):
        """Configure mock to simulate vertex validation failures."""
        self._vertex_validation_should_fail = should_fail


class MockSemanticMetagraphEdgeCases(AbstractEdgeCaseMock):
    """Mock for SemanticMetagraph edge case testing scenarios."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize as SemanticMetagraph mock but with edge case behaviors
        self.doc = None
        self.metaverts = {}
        self.current_mv_idx = 0
        
        # Configure edge case behaviors
        self.is_edge_case_active = True
        self.edge_case_type = kwargs.get('edge_case_type', 'empty_input')
        
        # Methods that can trigger edge cases
        self.add_vert = Mock(side_effect=self._edge_case_add_vert)
        self.to_json = Mock(side_effect=self._edge_case_to_json)
        self.to_nx = Mock(side_effect=self._edge_case_to_nx)
        self._build_metaverts_from_doc = Mock(side_effect=self._edge_case_build_metaverts)
    
    # Implement required abstract methods from AbstractEdgeCaseMock
    def trigger_edge_case(self, case_type: str) -> Any:
        """Trigger specific edge case scenario."""
        self.edge_case_type = case_type
        self.is_edge_case_active = True
        
        if case_type == 'empty_input':
            return self._handle_empty_input()
        elif case_type == 'invalid_data':
            return self._handle_invalid_data()
        elif case_type == 'memory_limit':
            return self._handle_memory_limit()
        elif case_type == 'malformed_json':
            return self._handle_malformed_json()
        else:
            return self._handle_unknown_edge_case()
    
    def reset_edge_case(self) -> None:
        """Reset edge case state to normal."""
        self.is_edge_case_active = False
        self.edge_case_type = None
    
    def get_edge_case_info(self) -> Dict[str, Any]:
        """Get information about current edge case state."""
        return {
            'is_active': self.is_edge_case_active,
            'case_type': self.edge_case_type,
            'available_cases': ['empty_input', 'invalid_data', 'memory_limit', 'malformed_json']
        }
    
    def _handle_empty_input(self):
        """Handle empty input edge case."""
        self.metaverts = {}
        return self
    
    def _handle_invalid_data(self):
        """Handle invalid data edge case."""
        raise ValueError("Mock invalid data error")
    
    def _handle_memory_limit(self):
        """Handle memory limit edge case."""
        raise MemoryError("Mock memory limit exceeded")
    
    def _handle_malformed_json(self):
        """Handle malformed JSON edge case."""
        return {"malformed": "json data"}
    
    def _handle_unknown_edge_case(self):
        """Handle unknown edge case."""
        raise NotImplementedError("Mock unknown edge case")
    
    def _edge_case_add_vert(self, *args, **kwargs):
        """Edge case version of add_vert."""
        if self.edge_case_type == 'invalid_data':
            raise ValueError("Cannot add vertex: mock invalid data")
        return Mock()
    
    def _edge_case_to_json(self):
        """Edge case version of to_json."""
        if self.edge_case_type == 'malformed_json':
            return "invalid json string"
        return {"metaverts": "[]"}
    
    def _edge_case_to_nx(self):
        """Edge case version of to_nx."""
        if self.edge_case_type == 'memory_limit':
            raise MemoryError("Mock memory error in NetworkX conversion")
        return Mock()
    
    def _edge_case_build_metaverts(self, doc):
        """Edge case version of _build_metaverts_from_doc."""
        if self.edge_case_type == 'empty_input':
            return []
        if self.edge_case_type == 'invalid_data':
            raise ValueError("Mock invalid document format")
        return [("test", {"pos": "NOUN"})]
    
    # Implement abstract methods from AbstractEdgeCaseMock
    def setup_edge_case_scenario(self, scenario_name: str) -> None:
        """Set up a specific edge case scenario."""
        self.edge_case_type = scenario_name
        self.current_scenario = scenario_name
        self.is_edge_case_active = True
        
        # Track scenario setup
        self._track_scenario_execution(scenario_name)
        
        # Configure specific behaviors based on scenario
        if scenario_name == 'empty_input':
            self.metaverts = {}
        elif scenario_name == 'invalid_data':
            # Set up for data validation errors
            pass
        elif scenario_name == 'memory_limit':
            # Set up for memory-related errors
            pass
        elif scenario_name == 'malformed_json':
            # Set up for JSON parsing errors
            pass
    
    def get_edge_case_scenarios(self) -> List[str]:
        """Get list of available edge case scenarios."""
        return [
            'empty_input',
            'invalid_data', 
            'memory_limit',
            'malformed_json',
            'boundary_conditions',
            'unicode_handling',
            'large_data'
        ]


class MockSpacyDoc(AbstractNLPDocMock):
    """Mock spaCy Doc object for testing."""
    
    def __init__(self, text="Apple is a technology company.", *args, **kwargs):
        # Pass text to the AbstractNLPDocMock constructor
        super().__init__(text=text, *args, **kwargs)
        
        # Set NLP doc specific attributes
        self.doc_type = "spacy_doc"
        self.language = "en"
        
        # Mock spaCy Doc attributes and methods
        self.text = text
        self.tokens = self._create_mock_tokens(text)
        self.ents = self._create_mock_entities()
        
        # Mock spaCy methods
        self.__len__ = Mock(return_value=len(self.tokens))
        self.__iter__ = Mock(return_value=iter(self.tokens))
        self.__getitem__ = Mock(side_effect=lambda i: self.tokens[i])
    
    def _create_mock_tokens(self, text):
        """Create mock tokens from text."""
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
            token = MockSpacyToken(text=word, i=i)
            tokens.append(token)
        return tokens
    
    def _create_mock_entities(self):
        """Create mock named entities."""
        # Simple mock: if "Apple" in text, create ORG entity
        if "Apple" in self.text:
            return [Mock(text="Apple", label_="ORG", start=0, end=5)]
        return []
    
    # Implement required abstract methods from AbstractNLPDocMock
    def get_tokens(self) -> List[Any]:
        """Get document tokens."""
        return self.tokens
    
    def get_entities(self) -> List[Any]:
        """Get named entities in the document."""
        return self.ents
    
    def get_sentences(self) -> List[Any]:
        """Get sentences in the document."""
        return [Mock(text=self.text)]  # Simple mock: whole text as one sentence
    
    def get_doc_metadata(self) -> Dict[str, Any]:
        """Get document metadata."""
        return {
            'language': self.language,
            'token_count': len(self.tokens),
            'entity_count': len(self.ents),
            'text_length': len(self.text)
        }
    
    def create_tokens(self, text: str) -> List[Any]:
        """Create tokens from text - required abstract method."""
        return self._create_mock_tokens(text)
    
    def create_entities(self, text: str) -> List[Any]:
        """Create entities from text - required abstract method."""
        return self._create_mock_entities()
    
    def create_sentences(self, text: str) -> List[Any]:
        """Create sentences from text - required abstract method."""
        return [Mock(text=text)]


class MockSpacyToken(AbstractNLPTokenMock):
    """Mock spaCy Token object for testing."""
    
    def __init__(self, text="Apple", i=0, *args, **kwargs):
        # Call the AbstractNLPTokenMock constructor with just the text argument
        super().__init__(text=text, *args, **kwargs)
        
        # Set NLP token specific attributes
        self.token_type = "spacy_token"
        
        # Override index if provided
        self.i = i
        
        # Mock spaCy Token attributes
        self.pos_ = "PROPN" if text.istitle() else "NOUN"
        self.lemma_ = text.lower()
        self.dep_ = "ROOT" if i == 0 else "compound"
        self.ent_type_ = "ORG" if text in ["Apple", "Microsoft", "Google"] else ""
        self.head = Mock(i=max(0, i-1))  # Simple mock head
        self.children = []  # Simple mock: no children
        self.lefts = []
        self.rights = []
        
        # Mock token properties
        self.is_alpha = text.isalpha()
        self.is_lower = text.islower()
        self.is_upper = text.isupper()
        self.is_title = text.istitle()
        self.is_punct = not text.isalnum()
        self.is_space = text.isspace()
        self.like_num = text.isdigit()
        self.like_url = text.startswith('http')
        self.like_email = '@' in text
        self.is_currency = text in ['$', '€', '£']
        self.is_left_punct = text in ['(', '[', '{']
        self.is_right_punct = text in [')', ']', '}']
        self.is_bracket = text in ['(', ')', '[', ']', '{', '}']
        self.is_quote = text in ['"', "'", '`']
        
        # Mock morph analysis
        self.morph = Mock()
        self.morph.to_dict = Mock(return_value={})
    
    # Implement required abstract methods from AbstractNLPTokenMock
    def get_token_text(self) -> str:
        """Get token text."""
        return self.text
    
    def get_token_attributes(self) -> Dict[str, Any]:
        """Get token linguistic attributes."""
        return {
            'pos': self.pos_,
            'lemma': self.lemma_,
            'dep': self.dep_,
            'ent_type': self.ent_type_,
            'is_alpha': self.is_alpha,
            'is_punct': self.is_punct
        }
    
    def get_token_relations(self) -> List[Tuple[str, Any]]:
        """Get token dependency relations."""
        relations = []
        if self.head:
            relations.append(('head', self.head))
        for child in self.children:
            relations.append(('child', child))
        return relations
    
    def get_token_metadata(self) -> Dict[str, Any]:
        """Get token metadata."""
        return {
            'index': self.i,
            'token_type': self.token_type,
            'has_children': len(self.children) > 0,
            'dependency_label': self.dep_
        }
    
    def create_linguistic_features(self) -> Dict[str, Any]:
        """Create linguistic features - required abstract method."""
        return self.get_token_attributes()
    
    def create_dependency_relations(self) -> List[Tuple[str, Any]]:
        """Create dependency relations - required abstract method."""
        return self.get_token_relations()
    
    # Implement abstract methods from AbstractNLPTokenMock
    def get_linguistic_features(self) -> Dict[str, Any]:
        """Get linguistic features specific to this token type."""
        return {
            'pos': self.pos_,
            'lemma': self.lemma_,
            'dep': self.dep_,
            'ent_type': self.ent_type_,
            'is_alpha': self.is_alpha,
            'is_punct': self.is_punct,
            'is_space': self.is_space,
            'is_title': self.is_title,
            'like_num': self.like_num,
            'like_url': self.like_url,
            'like_email': self.like_email
        }
    
    def set_linguistic_attributes(self, **kwargs) -> None:
        """Set linguistic attributes for the token."""
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
    
    def validate_token_consistency(self) -> bool:
        """Validate that token attributes are consistent."""
        # Basic consistency checks
        if not self.text:
            return False
        if self.i < 0:
            return False
        if not isinstance(self.pos_, str):
            return False
        if not isinstance(self.lemma_, str):
            return False
        return True


class MockNetworkXGraph(Mock):
    """Mock NetworkX DiGraph for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Mock graph structure
        self._nodes = {}
        self._edges = {}
        
        # Mock NetworkX methods
        self.nodes = Mock(return_value=self._nodes.keys())
        self.edges = Mock(return_value=self._edges.keys())
        self.add_node = Mock(side_effect=self._mock_add_node)
        self.add_edge = Mock(side_effect=self._mock_add_edge)
    
    def _mock_add_node(self, node_id, **attrs):
        """Mock implementation of add_node."""
        self._nodes[node_id] = attrs
    
    def _mock_add_edge(self, source, target, **attrs):
        """Mock implementation of add_edge."""
        self._edges[(source, target)] = attrs