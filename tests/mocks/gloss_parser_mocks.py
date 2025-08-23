"""
Mock classes for GlossParser tests.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional


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