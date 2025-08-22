"""
Mock classes for BeamBuilder tests.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional


class MockBeamBuilder(Mock):
    """Mock for BeamBuilder class testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize with mock components
        self.embedding_helper = MockEmbeddingHelper()
        self.wn_module = MockWordNetModule()
        self.nlp_func = Mock()
        
        # Set up methods
        self.build_beam = Mock(return_value=[])
        self.get_semantic_neighbors = Mock(return_value=[])
        self.filter_candidates = Mock(return_value=[])
        self.score_candidates = Mock(return_value=[])
        self.expand_beam = Mock(return_value=[])


class MockBeamBuilderEdgeCases(Mock):
    """Mock for BeamBuilder edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up edge case scenarios
        self.empty_beam = Mock(return_value=[])
        self.invalid_synset = Mock(side_effect=Exception("Invalid synset"))
        self.no_embeddings = Mock(return_value=None)


class MockBeamBuilderIntegration(Mock):
    """Mock for BeamBuilder integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integration components
        self.mock_wordnet = MockWordNetModule()
        self.mock_embedding_helper = MockEmbeddingHelper()
        self.mock_nlp = Mock()


class MockEmbeddingHelper(Mock):
    """Mock for EmbeddingHelper used in BeamBuilder."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        self.similarity = Mock(return_value=0.8)
        self.most_similar = Mock(return_value=[])
        self.has_embedding = Mock(return_value=True)


class MockWordNetModule(Mock):
    """Mock for WordNet module used in BeamBuilder."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synset = Mock()
        self.synsets = Mock(return_value=[])
        
        # Create mock synsets
        self.mock_cat_synset = MockSynsetForBeam("cat.n.01", "feline mammal")
        self.mock_dog_synset = MockSynsetForBeam("dog.n.01", "canine mammal")


class MockSynsetForBeam(Mock):
    """Mock synset specifically for BeamBuilder tests."""
    
    def __init__(self, name="test.n.01", definition="test definition", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        self._definition = definition
        
        # Mock synset methods
        self.name = Mock(return_value=name)
        self.definition = Mock(return_value=definition)
        self.pos = Mock(return_value="n")
        self.examples = Mock(return_value=["example"])
        self.lemmas = Mock(return_value=[Mock(name=Mock(return_value="test"))])
        
        # Relations
        self.hypernyms = Mock(return_value=[])
        self.hyponyms = Mock(return_value=[])
        self.similar_tos = Mock(return_value=[])
        self.also_sees = Mock(return_value=[])
        
        # Similarity methods
        self.path_similarity = Mock(return_value=0.5)
        self.wup_similarity = Mock(return_value=0.8)
        self.lch_similarity = Mock(return_value=2.5)


class MockEmbeddingModel(Mock):
    """Mock embedding model for BeamBuilder tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_vector = Mock(return_value=[0.1, 0.2, 0.3])
        self.most_similar = Mock(return_value=[("similar_word", 0.9)])
        self.similarity = Mock(return_value=0.8)
        self.vocab = Mock()
        
        # Mock vocab
        self.vocab.__contains__ = Mock(return_value=True)


class MockNLPFunction(Mock):
    """Mock NLP function for BeamBuilder."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mock spaCy-like behavior
        self.return_value = MockDoc()


class MockDoc(Mock):
    """Mock spaCy Doc object."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = "test document"
        self.tokens = [MockToken("test"), MockToken("document")]
        self.ents = []
        self.noun_chunks = []


class MockToken(Mock):
    """Mock spaCy Token object."""
    
    def __init__(self, text="test", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        self.lemma_ = text.lower()
        self.pos_ = "NOUN"
        self.tag_ = "NN"
        self.dep_ = "ROOT"
        self.is_stop = False
        self.is_alpha = True