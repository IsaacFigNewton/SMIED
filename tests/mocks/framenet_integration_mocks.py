"""
Mock classes for FrameNet integration tests.
"""

from unittest.mock import Mock
import networkx as nx
from typing import List, Optional, Any, Dict, Tuple, Set
from dataclasses import dataclass

# Import abstract base classes
from tests.mocks.base.entity_mock import AbstractEntityMock, EntityType, EntityStatus
from tests.mocks.base.nlp_doc_mock import AbstractNLPDocMock
from tests.mocks.base.nlp_token_mock import AbstractNLPTokenMock
from tests.mocks.base.nlp_function_mock import AbstractNLPFunctionMock
from tests.mocks.base.algorithmic_function_mock import AbstractAlgorithmicFunctionMock
from tests.mocks.base.reasoning_mock import AbstractReasoningMock, ReasoningType, InferenceStrategy
from tests.mocks.base.library_wrapper_mock import AbstractLibraryWrapperMock


class FrameNetIntegrationMockFactory:
    """Factory class for creating FrameNet integration mock instances.
    
    This factory follows the SMIED Testing Framework Design Specifications
    for mock creation using factory pattern with abstract base class hierarchy.
    """
    
    def __init__(self):
        self._mock_classes = {
            # Core FrameNet integration mocks
            'MockFrameNetIntegration': MockFrameNetIntegration,
            'MockFrameNetIntegrationValidation': MockFrameNetIntegrationValidation,
            'MockFrameNetIntegrationEdgeCases': MockFrameNetIntegrationEdgeCases,
            'MockFrameNetIntegrationIntegration': MockFrameNetIntegrationIntegration,
            
            # FrameNet SRL component mocks
            'MockFrameNetSpaCySRL': MockFrameNetSpaCySRL,
            'MockFrameNetSpaCySRLValidation': MockFrameNetSpaCySRLValidation,
            'MockFrameNetSpaCySRLEdgeCases': MockFrameNetSpaCySRLEdgeCases,
            'MockFrameNetSpaCySRLIntegration': MockFrameNetSpaCySRLIntegration,
            
            # Frame data structure mocks
            'MockFrameInstance': MockFrameInstance,
            'MockFrameElement': MockFrameElement,
            'MockFrameInstanceForIntegration': MockFrameInstanceForIntegration,
            'MockFrameElementForIntegration': MockFrameElementForIntegration,
            
            # SemanticDecomposer enhancement mocks
            'MockSemanticDecomposerForFrameNet': MockSemanticDecomposerForFrameNet,
            'MockSemanticDecomposerEnhanced': MockSemanticDecomposerEnhanced,
            
            # WordNet and NLP component mocks
            'MockWordNetForFrameNet': MockWordNetForFrameNet,
            'MockSynsetForFrameNet': MockSynsetForFrameNet,
            'MockLemmaForFrameNet': MockLemmaForFrameNet,
            'MockNLPForFrameNet': MockNLPForFrameNet,
            'MockDocForFrameNet': MockDocForFrameNet,
            'MockTokenForFrameNet': MockTokenForFrameNet,
            'MockSpanForFrameNet': MockSpanForFrameNet,
            
            # Graph and pathfinding mocks
            'MockNetworkXGraphForFrameNet': MockNetworkXGraphForFrameNet,
            'MockPathfinderForFrameNet': MockPathfinderForFrameNet,
            
            # Derivational and morphology mocks
            'MockDerivationalConnection': MockDerivationalConnection,
            'MockMorphologicalAnalyzer': MockMorphologicalAnalyzer,
            
            # Integration scenario mocks
            'MockFrameNetScenarioBuilder': MockFrameNetScenarioBuilder,
            'MockFrameNetTestDataGenerator': MockFrameNetTestDataGenerator,
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


# Data structure mocks for FrameNet
class MockFrameElement(AbstractEntityMock):
    """Mock for FrameElement data class."""
    
    def __init__(self, name="Agent", frame_name="Action", confidence=0.8, fe_type="Core", *args, **kwargs):
        super().__init__(entity_type=EntityType.NODE, *args, **kwargs)
        self.name = name
        self.frame_name = frame_name
        self.confidence = confidence
        self.fe_type = fe_type
        self.definition = f"Definition for {name} frame element"
        
        # Create mock span
        self.span = MockSpanForFrameNet(text="test span")
        
        # Set entity properties
        self.label = name
        self.category = "frame_element"
        self.domain = "framenet"
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this frame element."""
        return self.name
    
    def validate_entity(self) -> bool:
        """Validate that the frame element is consistent and valid."""
        return (self.is_valid and 
                bool(self.name) and 
                bool(self.frame_name) and
                0.0 <= self.confidence <= 1.0 and
                self.fe_type in ["Core", "Non-Core", "Extra-Thematic"])
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this frame element."""
        return f"{self.entity_type.value}:{self.frame_name}:{self.name}:{self.confidence}"


class MockFrameInstance(AbstractEntityMock):
    """Mock for FrameInstance data class."""
    
    def __init__(self, name="Action", confidence=0.9, *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.name = name
        self.confidence = confidence
        self.definition = f"Definition for {name} frame"
        self.lexical_unit = "test.v"
        
        # Create mock target span
        self.target = MockSpanForFrameNet(text="test target")
        
        # Create mock frame elements
        self.elements = [
            MockFrameElement("Agent", name, 0.8, "Core"),
            MockFrameElement("Patient", name, 0.7, "Core"),
            MockFrameElement("Instrument", name, 0.6, "Non-Core")
        ]
        
        # Set entity properties
        self.label = name
        self.category = "frame_instance"
        self.domain = "framenet"
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this frame instance."""
        return self.name
    
    def validate_entity(self) -> bool:
        """Validate that the frame instance is consistent and valid."""
        return (self.is_valid and 
                bool(self.name) and 
                0.0 <= self.confidence <= 1.0 and
                isinstance(self.elements, list))
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this frame instance."""
        return f"{self.entity_type.value}:{self.name}:{len(self.elements)}:{self.confidence}"


class MockSpanForFrameNet(AbstractEntityMock):
    """Mock for spaCy Span objects used in FrameNet integration."""
    
    def __init__(self, text="test span", start=0, end=1, *args, **kwargs):
        super().__init__(entity_type=EntityType.NODE, *args, **kwargs)
        self.text = text
        self.start = start
        self.end = end
        self.start_char = start * 5  # Mock character positions
        self.end_char = end * 5
        self.label_ = "TEST"
        
        # Set entity properties
        self.name = text
        self.label = text
        self.category = "span"
        self.domain = "spacy"
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this span."""
        return self.text
    
    def validate_entity(self) -> bool:
        """Validate that the span is consistent and valid."""
        return (self.is_valid and 
                bool(self.text) and
                isinstance(self.start, int) and
                isinstance(self.end, int) and
                self.start <= self.end)
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this span."""
        return f"{self.entity_type.value}:{self.text}:{self.start}:{self.end}"


# Main FrameNet SRL mock classes
class MockFrameNetSpaCySRL(AbstractLibraryWrapperMock):
    """Mock for FrameNetSpaCySRL class."""
    
    def __init__(self, nlp=None, min_confidence=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nlp = nlp or MockNLPForFrameNet()
        self.min_confidence = min_confidence
        
        # Configure library wrapper
        self.library_name = "framenet_spacy_srl"
        self.library_version = "1.0.0"
        self.wrapper_type = "srl_processor"
        
        # Set up main methods
        self.process_text = Mock(return_value=MockDocForFrameNet())
        self.extract_frames = Mock(return_value=[])
        self.find_frame_connections = Mock(return_value=[])
        self.get_semantic_roles = Mock(return_value={})
        self.analyze_frame_relations = Mock(return_value=[])
    
    def create_mock_processing_result(self, text="test text", frame_count=1):
        """Create a mock processing result with frames."""
        doc = MockDocForFrameNet(text)
        frames = []
        
        for i in range(frame_count):
            frame = MockFrameInstance(f"TestFrame{i}", 0.8)
            frames.append(frame)
        
        doc._.frames = frames
        return doc
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get information about the wrapped library."""
        return {
            "name": self.library_name,
            "version": self.library_version,
            "type": self.wrapper_type,
            "min_confidence": self.min_confidence,
            "nlp_model": getattr(self.nlp, "model_name", "unknown")
        }
    
    def configure_wrapper(self, **kwargs) -> None:
        """Configure the library wrapper."""
        if "min_confidence" in kwargs:
            self.min_confidence = kwargs["min_confidence"]
        if "nlp" in kwargs:
            self.nlp = kwargs["nlp"]
    
    def validate_wrapper_state(self) -> bool:
        """Validate that the wrapper is in a valid state."""
        return (self.nlp is not None and 
                0.0 <= self.min_confidence <= 1.0)
    
    def check_compatibility(self) -> bool:
        """Check if the library is compatible with current environment."""
        return True
    
    def initialize_library(self) -> None:
        """Initialize the wrapped library."""
        pass


class MockSemanticDecomposerForFrameNet(AbstractReasoningMock):
    """Mock for SemanticDecomposer with FrameNet enhancements."""
    
    def __init__(self, wn_module=None, nlp_func=None, embedding_model=None, verbosity=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configure reasoning mock
        self.engine_name = "semantic_decomposer_framenet"
        self.reasoning_type = ReasoningType.SIMILARITY
        self.inference_strategy = InferenceStrategy.BREADTH_FIRST
        
        # Set up components
        self.wn_module = wn_module or MockWordNetForFrameNet()
        self.nlp_func = nlp_func or MockNLPForFrameNet()
        self.embedding_model = embedding_model
        self.verbosity = verbosity
        
        # Initialize FrameNet SRL
        self.framenet_srl = MockFrameNetSpaCySRL()
        
        # Set up core methods
        self.build_synset_graph = Mock(return_value=MockNetworkXGraphForFrameNet())
        self.find_connected_shortest_paths = Mock(return_value=(None, None, None))
        
        # Set up FrameNet-enhanced methods
        self._find_framenet_subject_predicate_paths = Mock(return_value=[])
        self._find_framenet_predicate_object_paths = Mock(return_value=[])
        self._find_derivational_subject_predicate_paths = Mock(return_value=[])
        self._find_derivational_predicate_object_paths = Mock(return_value=[])
        self._find_gloss_based_subject_predicate_paths = Mock(return_value=[])
        self._find_gloss_based_predicate_object_paths = Mock(return_value=[])
        
        # Set up helper methods
        self._get_subject_frame_elements = Mock(return_value=[])
        self._get_object_frame_elements = Mock(return_value=[])
        self._frame_element_to_synsets = Mock(return_value=[])
        self._get_derivational_connections = Mock(return_value=[])
    
    def infer(self, query: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Perform FrameNet-enhanced semantic decomposition inference."""
        from tests.mocks.base.reasoning_mock import InferenceResult
        
        # Mock inference process using FrameNet integration
        reasoning_path = [
            "Analyzed query for semantic components",
            "Applied FrameNet SRL processing",
            "Searched for frame-based connections",
            "Applied derivational morphology fallback",
            "Constructed semantic paths"
        ]
        
        return InferenceResult(
            conclusion="Semantic paths found",
            confidence=0.8,
            reasoning_path=reasoning_path,
            evidence=[query],
            metadata={
                'inference_type': 'framenet_semantic_decomposition',
                'strategies_used': ['framenet_srl', 'derivational_morphology'],
                'verbosity': self.verbosity
            }
        )
    
    def compute_similarity(self, entity1: Any, entity2: Any) -> float:
        """Compute similarity between two entities for semantic decomposition."""
        if entity1 == entity2:
            return 1.0
        # Mock similarity based on type and attributes
        return 0.7
    
    def explain_reasoning(self, inference_result) -> List[str]:
        """Generate explanation for semantic decomposition reasoning."""
        explanations = [
            "Applied FrameNet-enhanced semantic decomposition",
            "Used cascading strategy for path finding",
            f"Found paths with confidence: {inference_result.confidence}"
        ]
        return explanations


class MockWordNetForFrameNet(Mock):
    """Mock WordNet module for FrameNet integration tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synsets = Mock(side_effect=self._mock_synsets)
        self.synset = Mock(side_effect=self._mock_synset)
        self.all_synsets = Mock(return_value=self._create_multiple_synsets())
        
        # WordNet constants
        self.NOUN = 'n'
        self.VERB = 'v'
        self.ADJ = 'a'
        self.ADV = 'r'
    
    def _mock_synsets(self, word, pos=None):
        """Mock synsets method with FrameNet-relevant examples."""
        synset_map = {
            "cat": [self._create_mock_synset("cat.n.01", "feline mammal")],
            "chase": [self._create_mock_synset("chase.v.01", "go after with intent to catch")],
            "mouse": [self._create_mock_synset("mouse.n.01", "small rodent")],
            "hunter": [self._create_mock_synset("hunter.n.01", "person who hunts")],
            "hunt": [self._create_mock_synset("hunt.v.01", "pursue for food or sport")],
            "prey": [self._create_mock_synset("prey.n.01", "animal hunted for food")]
        }
        return synset_map.get(word, [])
    
    def _mock_synset(self, name):
        """Mock synset method by name."""
        return self._create_mock_synset(name)
    
    def _create_mock_synset(self, name, definition=None):
        """Create a mock synset with FrameNet-relevant attributes."""
        synset = MockSynsetForFrameNet(name, definition or f"Definition for {name}")
        return synset
    
    def _create_multiple_synsets(self):
        """Create multiple mock synsets for comprehensive testing."""
        synsets = []
        test_data = [
            ("cat.n.01", "feline mammal"),
            ("chase.v.01", "pursue"),
            ("mouse.n.01", "small rodent"),
            ("animal.n.01", "living organism"),
            ("mammal.n.01", "warm-blooded vertebrate")
        ]
        
        for name, definition in test_data:
            synsets.append(self._create_mock_synset(name, definition))
        
        return synsets


class MockSynsetForFrameNet(AbstractEntityMock):
    """Mock synset with FrameNet integration features."""
    
    def __init__(self, name="test.n.01", definition="test definition", pos="n", *args, **kwargs):
        super().__init__(entity_type=EntityType.SYNSET, *args, **kwargs)
        self._name = name
        self._definition = definition
        self._pos = pos
        
        # Set up basic synset methods
        self.name = Mock(return_value=name)
        self.definition = Mock(return_value=definition)
        self.pos = Mock(return_value=pos)
        self.examples = Mock(return_value=[f"Example for {name}"])
        
        # Create lemma mocks with derivational relations
        self.lemmas = Mock(return_value=self._create_lemma_mocks())
        
        # Set up semantic relations
        self._setup_semantic_relations()
        
        # Set entity attributes
        self.label = name
        self.category = "synset"
        self.domain = "wordnet"
    
    def _create_lemma_mocks(self):
        """Create mock lemmas with derivational relations."""
        lemmas = []
        if self._name and '.' in self._name:
            base_word = self._name.split('.')[0]
        else:
            base_word = self._name if self._name else "unknown"
        
        lemma = MockLemmaForFrameNet(base_word)
        lemmas.append(lemma)
        
        return lemmas
    
    def _setup_semantic_relations(self):
        """Set up semantic relations for the synset."""
        # Basic relations
        self.hypernyms = Mock(return_value=[])
        self.hyponyms = Mock(return_value=[])
        self.member_holonyms = Mock(return_value=[])
        self.part_holonyms = Mock(return_value=[])
        self.substance_holonyms = Mock(return_value=[])
        self.member_meronyms = Mock(return_value=[])
        self.part_meronyms = Mock(return_value=[])
        self.substance_meronyms = Mock(return_value=[])
        self.similar_tos = Mock(return_value=[])
        self.also_sees = Mock(return_value=[])
        self.verb_groups = Mock(return_value=[])
        self.entailments = Mock(return_value=[])
        self.causes = Mock(return_value=[])
        self.attributes = Mock(return_value=[])
        
        # Similarity methods
        self.path_similarity = Mock(return_value=0.5)
        self.wup_similarity = Mock(return_value=0.8)
        self.lch_similarity = Mock(return_value=2.5)
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this synset."""
        return self._name
    
    def validate_entity(self) -> bool:
        """Validate that the synset is consistent and valid."""
        return (self.is_valid and 
                bool(self._name) and
                bool(self._definition) and
                self._pos in ['n', 'v', 'a', 'r'])
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this synset."""
        return f"{self.entity_type.value}:{self._name}:{self._pos}"


class MockLemmaForFrameNet(AbstractEntityMock):
    """Mock lemma with derivational relations for FrameNet tests."""
    
    def __init__(self, lemma_name="test", *args, **kwargs):
        super().__init__(entity_type=EntityType.LEMMA, *args, **kwargs)
        self._lemma_name = lemma_name
        
        # Set up lemma methods
        self.name = Mock(return_value=lemma_name)
        self.derivationally_related_forms = Mock(return_value=self._create_derivational_forms())
        self.antonyms = Mock(return_value=[])
        self.pertainyms = Mock(return_value=[])
        
        # Mock synset method - avoid circular reference by returning simple Mock
        self.synset = Mock()
        self.synset.return_value = Mock()
        self.synset.return_value.name = Mock(return_value=f"{lemma_name}.n.01")
        
        # Set entity attributes
        self.label = lemma_name
        self.category = "lemma"
        self.domain = "wordnet"
    
    def _create_derivational_forms(self):
        """Create mock derivational forms."""
        base_forms = {
            "hunt": ["hunter", "hunting"],
            "chase": ["chaser", "chasing"],
            "teach": ["teacher", "teaching"],
            "learn": ["learner", "learning"]
        }
        
        related_forms = []
        if self._lemma_name in base_forms:
            for form in base_forms[self._lemma_name]:
                # Create simple Mock instead of full MockLemmaForFrameNet to avoid circular references
                related_lemma = Mock()
                related_lemma.name = Mock(return_value=form)
                related_lemma.synset = Mock()
                related_lemma.synset.return_value = Mock()
                related_lemma.synset.return_value.name = Mock(return_value=f"{form}.n.01")
                related_forms.append(related_lemma)
        
        return related_forms
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this lemma."""
        return self._lemma_name
    
    def validate_entity(self) -> bool:
        """Validate that the lemma is consistent and valid."""
        return (self.is_valid and 
                bool(self._lemma_name))
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this lemma."""
        return f"{self.entity_type.value}:{self._lemma_name}"


class MockNLPForFrameNet(AbstractNLPFunctionMock):
    """Mock NLP function for FrameNet integration tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_value = MockDocForFrameNet()
        self.model_name = "framenet_nlp_model"
        self.lang = "en"
    
    def process_text(self, text: str, **kwargs) -> Any:
        """Process text and return Doc with FrameNet frames."""
        doc = MockDocForFrameNet(text)
        return doc
    
    def create_doc_mock(self, text: str, **kwargs) -> Any:
        """Create a mock Doc object with FrameNet processing."""
        return MockDocForFrameNet(text)
    
    def configure_pipeline(self, components: List[str]) -> None:
        """Configure the NLP processing pipeline."""
        self.pipe_names = components
        self.enabled_components = set(components)
        self.disabled_components = set()


class MockDocForFrameNet(AbstractNLPDocMock):
    """Mock document with FrameNet frames."""
    
    def __init__(self, text="test document", *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        
        # Set up FrameNet frames attribute
        self._ = Mock()
        self._.frames = self._create_mock_frames()
        
        # Set up basic NLP attributes
        self.tokens = self.create_tokens()
        self.ents = self.create_entities()
        self.sents = self.create_sentences()
    
    def _create_mock_frames(self):
        """Create mock FrameNet frames for the document."""
        frames = []
        
        # Create a basic frame for testing
        frame = MockFrameInstance("Cotheme", 0.85)
        frames.append(frame)
        
        return frames
    
    def create_tokens(self) -> List[Any]:
        """Create token objects for the document."""
        words = self.text.split()
        tokens = []
        for i, word in enumerate(words):
            token = MockTokenForFrameNet(word, i)
            tokens.append(token)
        return tokens
    
    def create_entities(self) -> List[Any]:
        """Create named entity objects for the document."""
        return []  # No entities by default
    
    def create_sentences(self) -> List[Any]:
        """Create sentence span objects for the document."""
        sent_mock = MockSpanForFrameNet(self.text, 0, len(self.tokens))
        return [sent_mock]


class MockTokenForFrameNet(AbstractNLPTokenMock):
    """Mock token for FrameNet integration tests."""
    
    def __init__(self, text="test", idx=0, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.idx = idx
        self.lemma_ = text.lower()
        self.pos_ = "NOUN" if idx % 2 == 0 else "VERB"
        self.dep_ = "ROOT" if idx == 0 else "dobj"
        self.head = self if idx == 0 else None
    
    def get_linguistic_features(self) -> Dict[str, Any]:
        """Get linguistic features for the token."""
        return {
            'text': self.text,
            'lemma': self.lemma_,
            'pos': self.pos_,
            'dep': self.dep_,
            'idx': self.idx,
            'is_alpha': self.is_alpha,
            'is_stop': self.is_stop
        }
    
    def set_linguistic_attributes(self, **kwargs) -> None:
        """Set linguistic attributes for the token."""
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
    
    def validate_token_consistency(self) -> bool:
        """Validate token consistency."""
        return (isinstance(self.text, str) and
                isinstance(self.lemma_, str) and
                isinstance(self.pos_, str) and
                isinstance(self.idx, int))


class MockNetworkXGraphForFrameNet(Mock):
    """Mock NetworkX graph for FrameNet integration tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nodes = set()
        self._edges = set()
        
        # Initialize with some test nodes
        test_nodes = ["cat.n.01", "chase.v.01", "mouse.n.01", "animal.n.01"]
        for node in test_nodes:
            self._nodes.add(node)
        
        # Add some test edges
        test_edges = [("cat.n.01", "animal.n.01"), ("mouse.n.01", "animal.n.01")]
        for edge in test_edges:
            self._edges.add(edge)
        
        # Set up graph methods
        self.nodes = Mock(return_value=list(self._nodes))
        self.edges = Mock(return_value=list(self._edges))
        self.number_of_nodes = Mock(return_value=len(self._nodes))
        self.number_of_edges = Mock(return_value=len(self._edges))
        self.add_node = Mock(side_effect=lambda n: self._nodes.add(n))
        self.add_edge = Mock(side_effect=lambda u, v, **attr: self._edges.add((u, v)))
        self.has_node = Mock(side_effect=lambda n: n in self._nodes)
        self.__contains__ = Mock(side_effect=lambda n: n in self._nodes)


# Validation and Edge Case Mocks
class MockFrameNetIntegrationValidation(Mock):
    """Mock for FrameNet integration validation testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up validation-specific behavior
        self.validate_framenet_srl = Mock(return_value=True)
        self.validate_frame_elements = Mock(return_value=True)
        self.validate_derivational_connections = Mock(return_value=True)
        self.validate_integration_parameters = Mock(return_value=True)
        
        # Error reporting methods
        self.get_validation_errors = Mock(return_value=[])
        self.is_valid_frame_instance = Mock(return_value=True)
        self.is_valid_frame_element = Mock(return_value=True)


class MockFrameNetIntegrationEdgeCases(Mock):
    """Mock for FrameNet integration edge case testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up edge case behaviors
        self.handle_empty_frames = Mock(return_value=(None, None, None))
        self.handle_invalid_confidence = Mock(return_value=(None, None, None))
        self.handle_missing_frame_elements = Mock(return_value=(None, None, None))
        self.handle_framenet_timeout = Mock(side_effect=TimeoutError("FrameNet processing timeout"))
        self.handle_derivational_failure = Mock(side_effect=ValueError("Derivational analysis failed"))


class MockFrameNetIntegrationIntegration(Mock):
    """Mock for comprehensive FrameNet integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integration components - use simple mocks to avoid circular dependencies
        self.real_framenet_srl = Mock()  # Simplified to avoid complex initialization
        self.real_semantic_decomposer = Mock()  # Simplified to avoid abstract method issues
        self.pathfinder = Mock()  # Simplified to avoid complex initialization
        
        # Add basic methods to the simplified mocks
        self.real_semantic_decomposer._find_framenet_subject_predicate_paths = Mock(return_value=[])
        self.real_semantic_decomposer._find_derivational_subject_predicate_paths = Mock(return_value=[])
        self.real_semantic_decomposer._apply_cascading_strategy = Mock(return_value=(None, None, None))
        
        # Integration scenario methods
        self.create_integration_scenario = Mock()
        self.setup_framenet_processing = Mock()
        self.validate_integration_results = Mock(return_value=True)


class MockFrameNetSpaCySRLValidation(MockFrameNetSpaCySRL):
    """Mock FrameNetSpaCySRL for validation testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override for validation-specific behavior
        self.validate_confidence_threshold = Mock(return_value=True)
        self.validate_frame_quality = Mock(return_value=True)
        self.get_processing_errors = Mock(return_value=[])


class MockFrameNetSpaCySRLEdgeCases(MockFrameNetSpaCySRL):
    """Mock FrameNetSpaCySRL for edge case testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override for edge case behavior
        self.process_text = Mock(side_effect=self._edge_case_processing)
        self.handle_processing_failure = Mock(return_value=MockDocForFrameNet(""))
    
    def _edge_case_processing(self, text):
        """Handle edge case processing scenarios."""
        if not text or len(text.strip()) == 0:
            doc = MockDocForFrameNet("")
            doc._.frames = []
            return doc
        elif len(text) > 1000:
            raise MemoryError("Text too long for processing")
        else:
            return MockDocForFrameNet(text)


class MockFrameNetSpaCySRLIntegration(MockFrameNetSpaCySRL):
    """Mock FrameNetSpaCySRL for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # More realistic behavior for integration testing
        self.process_text = Mock(side_effect=self._realistic_processing)
        self.confidence_threshold = 0.4
    
    def _realistic_processing(self, text):
        """Provide more realistic processing for integration tests."""
        doc = MockDocForFrameNet(text)
        
        # Create frames based on text content
        frames = []
        if "cat" in text.lower() and "chase" in text.lower():
            frame = MockFrameInstance("Cotheme", 0.85)
            frame.elements = [
                MockFrameElement("Theme", "Cotheme", 0.8, "Core"),
                MockFrameElement("Cotheme", "Cotheme", 0.75, "Core")
            ]
            frames.append(frame)
        
        doc._.frames = frames
        return doc


# Additional specialized mocks
class MockSemanticDecomposerEnhanced(MockSemanticDecomposerForFrameNet):
    """Enhanced mock for SemanticDecomposer with full FrameNet integration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enhanced methods for comprehensive testing
        self._find_path_between_synsets = Mock(return_value=None)
        self._explore_hypernym_paths = Mock(return_value=[])
        self._apply_cascading_strategy = Mock(return_value=(None, None, None))
    
    def compute_similarity(self, entity1: Any, entity2: Any) -> float:
        """Enhanced similarity computation for comprehensive testing."""
        if entity1 == entity2:
            return 1.0
        # Enhanced similarity logic for integration testing
        return 0.8
    
    def explain_reasoning(self, inference_result) -> List[str]:
        """Enhanced reasoning explanation for comprehensive testing."""
        explanations = [
            "Applied enhanced FrameNet semantic decomposition",
            "Used full cascading strategy with multiple fallbacks",
            "Integrated derivational morphology and frame-based reasoning",
            f"Achieved confidence: {inference_result.confidence}"
        ]
        return explanations


class MockPathfinderForFrameNet(AbstractAlgorithmicFunctionMock):
    """Mock pathfinder for FrameNet integration tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_name = "framenet_pathfinder"
        self.algorithm_type = "semantic_pathfinding"
        self.complexity_class = "O(V + E)"
        
        # Set up pathfinding methods
        self.find_shortest_path = Mock(return_value=[])
        self.find_all_paths = Mock(return_value=[])
    
    def compute(self, *args, **kwargs) -> Any:
        """Compute semantic pathfinding with FrameNet integration."""
        if "source" in kwargs and "target" in kwargs:
            return self.find_shortest_path(kwargs["source"], kwargs["target"])
        return []
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate pathfinding inputs."""
        return ("source" in kwargs and "target" in kwargs and
                kwargs["source"] is not None and kwargs["target"] is not None)
    
    def get_algorithm_properties(self) -> Dict[str, Any]:
        """Get pathfinding algorithm properties."""
        return {
            "supports_framenet": True,
            "supports_derivational": True,
            "cascading_strategy": True,
            "semantic_aware": True
        }


# Specialized data and scenario builders
class MockFrameNetScenarioBuilder(Mock):
    """Mock for building FrameNet integration test scenarios."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_frame_scenario = Mock()
        self.build_derivational_scenario = Mock()
        self.build_pathfinding_scenario = Mock()
        self.build_integration_scenario = Mock()


class MockFrameNetTestDataGenerator(Mock):
    """Mock for generating FrameNet integration test data."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_frame_instances = Mock()
        self.generate_frame_elements = Mock()
        self.generate_synset_relationships = Mock()
        self.generate_derivational_connections = Mock()


# Main mock classes for the 4-class test structure
class MockFrameNetIntegration(Mock):
    """Main mock for FrameNet integration basic functionality tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_decomposer = MockSemanticDecomposerForFrameNet()
        self.framenet_srl = MockFrameNetSpaCySRL()
        self.test_synsets = self._create_test_synsets()
        self.test_graph = MockNetworkXGraphForFrameNet()
    
    def _create_test_synsets(self):
        """Create basic test synsets for functionality testing."""
        synsets = {
            "cat": MockSynsetForFrameNet("cat.n.01", "feline mammal", "n"),
            "chase": MockSynsetForFrameNet("chase.v.01", "pursue", "v"),
            "mouse": MockSynsetForFrameNet("mouse.n.01", "small rodent", "n")
        }
        return synsets


# Specialized mocks for derivational and morphological analysis
class MockDerivationalConnection(AbstractEntityMock):
    """Mock for derivational morphology connections."""
    
    def __init__(self, source_lemma="hunt", target_lemma="hunter", relation_type="agentive", *args, **kwargs):
        super().__init__(entity_type=EntityType.RELATION, *args, **kwargs)
        self.source_lemma = source_lemma
        self.target_lemma = target_lemma
        self.relation_type = relation_type
        self.confidence = 0.9
        
        # Set entity attributes
        self.name = f"{source_lemma}-{target_lemma}"
        self.label = f"{source_lemma} → {target_lemma}"
        self.category = "derivational_connection"
        self.domain = "morphology"
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this connection."""
        return f"{self.source_lemma}-{self.target_lemma}"
    
    def validate_entity(self) -> bool:
        """Validate that the derivational connection is valid."""
        return (self.is_valid and 
                bool(self.source_lemma) and
                bool(self.target_lemma) and
                bool(self.relation_type))
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this derivational connection."""
        return f"{self.entity_type.value}:{self.source_lemma}:{self.target_lemma}:{self.relation_type}"


class MockMorphologicalAnalyzer(AbstractAlgorithmicFunctionMock):
    """Mock for morphological analysis component."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_name = "morphological_analyzer"
        self.algorithm_type = "linguistic_analysis"
        self.complexity_class = "O(n)"
        
        # Set up analysis methods
        self.analyze_derivation = Mock(return_value=[])
        self.find_morphological_relations = Mock(return_value=[])
        self.get_word_formation_patterns = Mock(return_value={})
    
    def compute(self, *args, **kwargs) -> Any:
        """Compute morphological analysis."""
        if "word" in kwargs:
            return self.analyze_derivation(kwargs["word"])
        return []
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate morphological analysis inputs."""
        return "word" in kwargs and isinstance(kwargs["word"], str)
    
    def get_algorithm_properties(self) -> Dict[str, Any]:
        """Get morphological analysis properties."""
        return {
            "supports_derivation": True,
            "supports_inflection": False,
            "language": "english",
            "pattern_based": True
        }


# Frame element and instance mocks for integration testing
class MockFrameElementForIntegration(MockFrameElement):
    """Enhanced frame element mock for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enhanced attributes for integration scenarios
        self.semantic_type = "Physical_object"
        self.core_type = "Core"
        self.frame_relations = []
        self.wordnet_synsets = []


class MockFrameInstanceForIntegration(MockFrameInstance):
    """Enhanced frame instance mock for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enhanced attributes for integration scenarios
        self.frame_relations = []
        self.inheritance_relations = []
        self.using_relations = []
        self.perspective_on_relations = []
        self.causative_of_relations = []