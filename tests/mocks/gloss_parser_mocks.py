"""
Mock classes for GlossParser tests.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional


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


class MockNLPForGloss(Mock):
    """Mock NLP function for GlossParser."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_value = MockDocForGloss()


class MockDocForGloss(Mock):
    """Mock document for GlossParser NLP processing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = "test gloss definition"
        self.tokens = [MockTokenForGloss("test"), MockTokenForGloss("gloss")]
        self.ents = [MockEntityForGloss("test", "NOUN")]
        self.noun_chunks = [MockChunkForGloss("test gloss")]
        self.sents = [MockSentenceForGloss("test gloss definition")]


class MockTokenForGloss(Mock):
    """Mock token for GlossParser NLP processing."""
    
    def __init__(self, text="test", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        self.lemma_ = text.lower()
        self.pos_ = "NOUN"
        self.tag_ = "NN"
        self.dep_ = "ROOT"
        self.head = self
        self.children = []
        self.is_stop = False
        self.is_alpha = True
        self.is_punct = False


class MockEntityForGloss(Mock):
    """Mock entity for GlossParser NLP processing."""
    
    def __init__(self, text="test", label="NOUN", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        self.label_ = label
        self.start = 0
        self.end = len(text)


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


class MockRealNLPForGloss(Mock):
    """Mock representing real NLP behavior for GlossParser."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # More realistic NLP processing
        self.complex_processing = Mock(return_value=MockComplexDoc())


class MockComplexDoc(Mock):
    """Mock complex document for realistic NLP processing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Complex document structure
        self.complex_tokens = self._create_complex_tokens()
        self.complex_entities = self._create_complex_entities()
        self.complex_dependencies = self._create_complex_dependencies()
    
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


class MockComplexGlosses(Mock):
    """Mock complex glosses for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Complex gloss scenarios
        self.simple_gloss = "a small furry animal"
        self.complex_gloss = "a carnivorous mammal with retractile claws that hunts small prey"
        self.technical_gloss = "a member of the family Felidae characterized by specific morphological features"