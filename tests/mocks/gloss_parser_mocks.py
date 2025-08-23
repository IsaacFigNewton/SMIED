"""
Mock classes for GlossParser tests.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional
from tests.mocks.base.nlp_doc_mock import AbstractNLPDocMock
from tests.mocks.base.nlp_token_mock import AbstractNLPTokenMock
from tests.mocks.base.entity_mock import AbstractEntityMock, EntityType
from tests.mocks.base.nlp_function_mock import AbstractNLPFunctionMock


class GlossParserMockFactory:
    """Factory class for creating GlossParser mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockGlossParser': MockGlossParser,
            'MockGlossParserIntegration': MockGlossParserIntegration,
            'MockNLPForGloss': MockNLPForGloss,
            'MockDocForGloss': MockDocForGloss,
            'MockTokenForGloss': MockTokenForGloss,
            'MockEntityForGloss': MockEntityForGloss,
            'MockChunkForGloss': MockChunkForGloss,
            'MockSentenceForGloss': MockSentenceForGloss,
            'MockRealNLPForGloss': MockRealNLPForGloss,
            'MockComplexDoc': MockComplexDoc,
            'MockComplexGlosses': MockComplexGlosses,
            # Common cross-factory mock types
            'MockSynset': MockSynsetForGloss,
            'MockLemma': MockLemmaForGloss,
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


class MockGlossParser(Mock):
    """Mock for GlossParser class testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize components
        self.nlp_func = MockNLPForGloss()
        self.synset_cache = {}
        
        # Set up methods
        self.parse_gloss = Mock(return_value={
            'subjects': [],
            'predicates': [],
            'objects': [],
            'modifiers': [],
            'relations': []
        })
        self.extract_semantic_roles = Mock(return_value=[])
        self.identify_key_concepts = Mock(return_value=[])
        self.parse_definition = Mock(return_value={})
        self.extract_examples = Mock(return_value=[])
        
        # Internal methods
        self._tokenize = Mock(return_value=[])
        self._pos_tag = Mock(return_value=[])
        self._dependency_parse = Mock(return_value=[])
        self._extract_entities = Mock(return_value=[])


class MockGlossParserIntegration(Mock):
    """Mock for GlossParser integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integration components
        self.real_nlp = MockRealNLPForGloss()
        self.complex_glosses = MockComplexGlosses()
    
    def create_integration_tokens(self, token_configs):
        """Create structured tokens for integration testing.
        
        Args:
            token_configs: List of dicts with token config like:
                [{'text': 'cat', 'pos_': 'NOUN', 'dep_': 'nsubj', 'lemma_': 'cat'}]
        """
        tokens = []
        for config in token_configs:
            token = MockTokenForGloss(config.get('text', 'test'))
            token.pos_ = config.get('pos_', 'NOUN')
            token.dep_ = config.get('dep_', 'ROOT')
            token.lemma_ = config.get('lemma_', config.get('text', 'test').lower())
            token.is_punct = config.get('is_punct', False)
            token.is_stop = config.get('is_stop', False)
            tokens.append(token)
        return tokens
    
    def create_integration_document(self, text, tokens):
        """Create a structured document for integration testing.
        
        Args:
            text: The document text
            tokens: List of token mocks
        """
        mock_doc = MockDocForGloss()
        mock_doc.text = text
        mock_doc.__iter__ = Mock(side_effect=lambda: iter(tokens))
        mock_doc.noun_chunks = []
        return mock_doc
    
    def create_integration_nlp_mock(self):
        """Create NLP mock for integration testing."""
        return MockNLPForGloss()
    
    def setup_integration_parsing_scenario(self, scenario_name='simple_sentence'):
        """Setup complete integration parsing scenario.
        
        Returns dict with configured nlp, doc, tokens, and synsets.
        """
        scenarios = {
            'simple_sentence': {
                'text': 'The cat runs',
                'token_configs': [
                    {'text': 'The', 'pos_': 'DT', 'dep_': 'det', 'is_stop': True},
                    {'text': 'cat', 'pos_': 'NOUN', 'dep_': 'nsubj', 'lemma_': 'cat'},
                    {'text': 'runs', 'pos_': 'VERB', 'dep_': 'ROOT', 'lemma_': 'run'}
                ],
                'synsets': [
                    {'name': 'cat.n.01', 'for_lemma': 'cat'},
                    {'name': 'run.v.01', 'for_lemma': 'run'}
                ]
            },
            'complex_sentence': {
                'text': 'The quick brown fox jumps over the lazy dog',
                'token_configs': [
                    {'text': 'The', 'pos_': 'DT', 'dep_': 'det', 'is_stop': True},
                    {'text': 'quick', 'pos_': 'ADJ', 'dep_': 'amod'},
                    {'text': 'brown', 'pos_': 'ADJ', 'dep_': 'amod'},
                    {'text': 'fox', 'pos_': 'NOUN', 'dep_': 'nsubj'},
                    {'text': 'jumps', 'pos_': 'VERB', 'dep_': 'ROOT', 'lemma_': 'jump'},
                    {'text': 'over', 'pos_': 'ADP', 'dep_': 'prep'},
                    {'text': 'the', 'pos_': 'DT', 'dep_': 'det', 'is_stop': True},
                    {'text': 'lazy', 'pos_': 'ADJ', 'dep_': 'amod'},
                    {'text': 'dog', 'pos_': 'NOUN', 'dep_': 'pobj'}
                ],
                'synsets': [
                    {'name': 'fox.n.01', 'for_lemma': 'fox'},
                    {'name': 'jump.v.01', 'for_lemma': 'jump'},
                    {'name': 'dog.n.01', 'for_lemma': 'dog'}
                ]
            }
        }
        
        if scenario_name not in scenarios:
            scenario_name = 'simple_sentence'
        
        scenario = scenarios[scenario_name]
        
        # Create components
        tokens = self.create_integration_tokens(scenario['token_configs'])
        doc = self.create_integration_document(scenario['text'], tokens)
        nlp = self.create_integration_nlp_mock()
        nlp.return_value = doc
        
        # Create synset mocks
        synsets = []
        for synset_config in scenario['synsets']:
            synset = MockSynsetForGloss(synset_config['name'])
            synsets.append(synset)
        
        return {
            'nlp': nlp,
            'doc': doc,
            'tokens': tokens,
            'synsets': synsets,
            'text': scenario['text']
        }


class MockNLPForGloss(AbstractNLPFunctionMock):
    """Mock NLP function for GlossParser."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_value = MockDocForGloss()
    
    def process_text(self, text: str, **kwargs) -> Any:
        """Process a single text and return a Doc object."""
        return MockDocForGloss(text=text)
    
    def create_doc_mock(self, text: str, **kwargs) -> Any:
        """Create a mock Doc object for the given text."""
        return MockDocForGloss(text=text)
    
    def configure_pipeline(self, components: List[str]) -> None:
        """Configure the processing pipeline."""
        self.pipe_names = components
        self.enabled_components = set(components)


class MockDocForGloss(AbstractNLPDocMock):
    """Mock document for GlossParser NLP processing."""
    
    def __init__(self, text="test gloss definition", *args, **kwargs):
        super().__init__(text=text, *args, **kwargs)
        # Initialize specific tokens and entities for gloss parsing
        self.tokens = self.create_tokens()
        self.ents = self.create_entities()
        self.noun_chunks = [MockChunkForGloss("test gloss")]
        self.sents = self.create_sentences()
    
    def create_tokens(self) -> List[Any]:
        """Create token objects for the document."""
        words = self.text.split()
        tokens = []
        for word in words:
            token = MockTokenForGloss(word)
            tokens.append(token)
        return tokens
    
    def create_entities(self) -> List[Any]:
        """Create named entity objects for the document."""
        # Create a basic entity for testing
        entities = []
        words = self.text.split()
        if words:
            entity = MockEntityForGloss(words[0], "TEST")
            entities.append(entity)
        return entities
    
    def create_sentences(self) -> List[Any]:
        """Create sentence span objects for the document."""
        return [MockSentenceForGloss(self.text)]


class MockTokenForGloss(AbstractNLPTokenMock):
    """Mock token for GlossParser NLP processing."""
    
    def __init__(self, text="test", *args, **kwargs):
        super().__init__(text=text, *args, **kwargs)
        # Override defaults for gloss-specific behavior
        self.head = self
        self.children = []
    
    def get_linguistic_features(self) -> Dict[str, Any]:
        """Get linguistic features specific to this token type."""
        return {
            'text': self.text,
            'lemma': self.lemma_,
            'pos': self.pos_,
            'tag': self.tag_,
            'dep': self.dep_,
            'is_stop': self.is_stop,
            'is_alpha': self.is_alpha,
            'is_punct': self.is_punct,
            'gloss_specific': True
        }
    
    def set_linguistic_attributes(self, **kwargs) -> None:
        """Set linguistic attributes for the token."""
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
    
    def validate_token_consistency(self) -> bool:
        """Validate that token attributes are consistent."""
        # Basic validation for gloss parser tokens
        if not self.text:
            return False
        if not self.lemma_:
            return False
        return True


class MockEntityForGloss(AbstractNLPTokenMock, AbstractEntityMock):
    """Mock entity for GlossParser NLP processing with both NLP token capabilities and entity-like behavior."""
    
    def __init__(self, text="test", label="NOUN", *args, **kwargs):
        # Extract entity-specific parameters for AbstractEntityMock
        entity_id = kwargs.pop('entity_id', f"entity_{text}")
        entity_type = kwargs.pop('entity_type', EntityType.CONCEPT)
        
        # Initialize Mock base class first to avoid MRO issues
        Mock.__init__(self, *args, **kwargs)
        
        # Manually set up attributes from both parent classes
        # Set up token attributes
        AbstractNLPTokenMock._setup_common_attributes(self, text)
        AbstractNLPTokenMock._setup_linguistic_attributes(self)  
        AbstractNLPTokenMock._setup_token_properties(self)
        
        # Set up entity attributes  
        AbstractEntityMock._setup_common_attributes(self, entity_id, entity_type)
        AbstractEntityMock._setup_entity_properties(self)
        AbstractEntityMock._setup_relationship_management(self)
        
        # Add NLP entity-specific attributes
        self.label_ = label
        self.start = 0
        self.end = len(text)
        
        # Sync between token and entity interfaces
        self.name = text
        self.label = label
        
        # Set entity-specific named entity annotations
        self.set_entity_annotation(ent_type=label, ent_iob="B", ent_id=entity_id)
    
    def get_linguistic_features(self) -> Dict[str, Any]:
        """Get linguistic features specific to this token type."""
        return {
            'text': self.text,
            'lemma': self.lemma_,
            'pos': self.pos_,
            'entity_label': self.label_,
            'start': self.start,
            'end': self.end,
            'is_entity': True,
            'entity_id': self.id,
            'entity_type': self.entity_type.value
        }
    
    def set_linguistic_attributes(self, **kwargs) -> None:
        """Set linguistic attributes for the token."""
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
    
    def validate_token_consistency(self) -> bool:
        """Validate that token attributes are consistent."""
        return bool(self.text and self.label_)
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this entity."""
        return self.text
    
    def validate_entity(self) -> bool:
        """Validate that the entity is consistent and valid."""
        return bool(self.text and self.label_ and self.id)
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this entity."""
        return f"{self.entity_type.value}:{self.text}:{self.label_}"


class MockChunkForGloss(Mock):
    """Mock noun chunk for GlossParser."""
    
    def __init__(self, text="test chunk", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        self.root = MockTokenForGloss(text.split()[0])
        self.start = 0
        self.end = len(text.split())


class MockSentenceForGloss(Mock):
    """Mock sentence for GlossParser."""
    
    def __init__(self, text="test sentence", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        self.root = MockTokenForGloss(text.split()[0])
        self.start = 0
        self.end = len(text.split())


class MockRealNLPForGloss(AbstractNLPFunctionMock):
    """Mock representing real NLP behavior for GlossParser."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # More realistic NLP processing
        self.complex_processing = Mock(return_value=MockComplexDoc())
    
    def process_text(self, text: str, **kwargs) -> Any:
        """Process a single text and return a realistic Doc object."""
        return self.create_realistic_doc(text)
    
    def create_doc_mock(self, text: str, **kwargs) -> Any:
        """Create a mock Doc object for the given text."""
        doc = MockComplexDoc()
        doc.text = text
        return doc
    
    def configure_pipeline(self, components: List[str]) -> None:
        """Configure the processing pipeline."""
        self.pipe_names = components
        self.enabled_components = set(components)
        # Enable more complex processing for specified components
        if 'parser' in components:
            self.parser = Mock(return_value="parsed")
        if 'ner' in components:
            self.ner = Mock(return_value="ner_processed")


class MockComplexDoc(AbstractNLPDocMock):
    """Mock complex document for realistic NLP processing."""
    
    def __init__(self, text="The cat sat on the mat", *args, **kwargs):
        super().__init__(text=text, *args, **kwargs)
        # Complex document structure
        self.complex_tokens = self._create_complex_tokens()
        self.complex_entities = self._create_complex_entities()
        self.complex_dependencies = self._create_complex_dependencies()
        
        # Override tokens and entities with complex versions
        self.tokens = self.complex_tokens
        self.ents = self.complex_entities
    
    def _create_complex_tokens(self):
        """Create complex token structure."""
        return [MockTokenForGloss(word) for word in ["The", "cat", "sat", "on", "the", "mat"]]
    
    def _create_complex_entities(self):
        """Create complex entity structure."""
        return [MockEntityForGloss("cat", "ANIMAL"), MockEntityForGloss("mat", "OBJECT")]
    
    def _create_complex_dependencies(self):
        """Create complex dependency structure."""
        dependencies = []
        # Mock dependency relations
        return dependencies
    
    def create_tokens(self) -> List[Any]:
        """Create token objects for the document."""
        return self._create_complex_tokens()
    
    def create_entities(self) -> List[Any]:
        """Create named entity objects for the document."""
        return self._create_complex_entities()
    
    def create_sentences(self) -> List[Any]:
        """Create sentence span objects for the document."""
        return [MockSentenceForGloss(self.text)]


class MockComplexGlosses(Mock):
    """Mock complex glosses for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Complex gloss scenarios
        self.simple_gloss = "a small furry animal"
        self.complex_gloss = "a carnivorous mammal with retractile claws that hunts small prey"
        self.technical_gloss = "a member of the family Felidae characterized by specific morphological features"


class MockSynsetForGloss(Mock):
    """Mock synset for GlossParser tests."""
    
    def __init__(self, synset_name="test.n.01", definition="test definition", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = synset_name
        self.definition_ = definition
        
        # Mock common synset methods - avoid circular references by using simple mocks
        self.name = Mock(return_value=synset_name)
        self.definition = Mock(return_value=definition)
        
        # Create simple lemma mock to avoid circular reference
        simple_lemma = Mock()
        simple_lemma.name = Mock(return_value="test")
        simple_lemma.key = Mock(return_value=f"test.n.01.{synset_name}")
        simple_lemma.count = Mock(return_value=1)
        self.lemmas = Mock(return_value=[simple_lemma])
        
        self.pos = Mock(return_value=synset_name.split('.')[1][0] if '.' in synset_name else 'n')
        
        # Mock synset relationships
        self.hypernyms = Mock(return_value=[])
        self.hyponyms = Mock(return_value=[])
        self.holonyms = Mock(return_value=[])
        self.meronyms = Mock(return_value=[])


class MockLemmaForGloss(Mock):
    """Mock lemma for GlossParser tests."""
    
    def __init__(self, lemma_name="test", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lemma_name = lemma_name
        
        # Mock lemma methods and properties - avoid circular reference
        self.name = Mock(return_value=lemma_name)
        
        # Create simple synset mock to avoid circular reference
        simple_synset = Mock()
        simple_synset.name = Mock(return_value=f"{lemma_name}.n.01")
        simple_synset.definition = Mock(return_value="test definition")
        self.synset = Mock(return_value=simple_synset)
        
        self.antonyms = Mock(return_value=[])
        self.derivationally_related_forms = Mock(return_value=[])
        
        # Common properties
        self.key = Mock(return_value=f"{lemma_name}.n.01.test")
        self.count = Mock(return_value=1)