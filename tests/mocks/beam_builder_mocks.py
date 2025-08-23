"""
Mock classes for BeamBuilder tests.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional

# Import abstract base classes
from tests.mocks.base.nlp_doc_mock import AbstractNLPDocMock
from tests.mocks.base.nlp_token_mock import AbstractNLPTokenMock
from tests.mocks.base.nlp_function_mock import AbstractNLPFunctionMock



class BeamBuilderMockFactory:
    """Factory class for creating BeamBuilder mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockBeamBuilder': MockBeamBuilder,
            'MockBeamBuilderEdgeCases': MockBeamBuilderEdgeCases,
            'MockBeamBuilderIntegration': MockBeamBuilderIntegration,
            'MockEmbeddingHelper': MockEmbeddingHelper,
            'MockWordNetModule': MockWordNetModule,
            'MockSynsetForBeam': MockSynsetForBeam,
            'MockEmbeddingModel': MockEmbeddingModel,
            'MockNLPFunction': MockNLPFunction,
            'MockDoc': MockDoc,
            'MockToken': MockToken,
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
    
    def create_integration_synset_mocks(self):
        """Create synset mocks for integration testing."""
        mock_cat_synset = MockSynsetForBeam("cat.n.01", "feline mammal")
        mock_dog_synset = MockSynsetForBeam("dog.n.01", "canine mammal")
        return mock_cat_synset, mock_dog_synset
    
    def create_realistic_embedding_model_mock(self):
        """Create realistic embedding model mock for integration tests."""
        mock_model = MockEmbeddingModel()
        mock_model.get_vector.return_value = [0.2, 0.4, 0.6, 0.8, 1.0]
        mock_model.similarity.return_value = 0.75
        return mock_model
    
    def create_complex_synset_mocks(self):
        """Create complex synset mocks with more detailed relationships."""
        mock_cat_synset = MockSynsetForBeam("cat.n.01", "feline mammal")
        mock_dog_synset = MockSynsetForBeam("dog.n.01", "canine mammal")
        
        # Add more realistic relationships
        mock_animal_synset = MockSynsetForBeam("animal.n.01", "living organism")
        mock_mammal_synset = MockSynsetForBeam("mammal.n.01", "warm-blooded vertebrate")
        
        # Set up hypernyms
        mock_cat_synset.hypernyms.return_value = [mock_mammal_synset]
        mock_dog_synset.hypernyms.return_value = [mock_mammal_synset]
        mock_mammal_synset.hypernyms.return_value = [mock_animal_synset]
        
        # Set up hyponyms
        mock_mammal_synset.hyponyms.return_value = [mock_cat_synset, mock_dog_synset]
        mock_animal_synset.hyponyms.return_value = [mock_mammal_synset]
        
        # Set up similar relationships
        mock_cat_synset.similar_tos.return_value = [mock_dog_synset]
        mock_dog_synset.similar_tos.return_value = [mock_cat_synset]
        
        return {
            'cat': mock_cat_synset,
            'dog': mock_dog_synset,
            'animal': mock_animal_synset,
            'mammal': mock_mammal_synset
        }
    
    def setup_patch_side_effects(self, synset_mocks):
        """Setup side effects for nltk.corpus.wordnet.synset patches."""
        def synset_side_effect(name):
            if name == "cat.n.01":
                return synset_mocks['cat']
            elif name == "dog.n.01":
                return synset_mocks['dog']
            elif name == "animal.n.01":
                return synset_mocks['animal']
            elif name == "mammal.n.01":
                return synset_mocks['mammal']
            else:
                return MockSynsetForBeam(name, f"definition for {name}")
        return synset_side_effect


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


class MockNLPFunction(AbstractNLPFunctionMock):
    """Mock NLP function for BeamBuilder."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mock spaCy-like behavior
        self.return_value = MockDoc()
    
    def process_text(self, text: str, **kwargs) -> Any:
        """Process a single text and return a MockDoc object."""
        return MockDoc(text=text)
    
    def create_doc_mock(self, text: str, **kwargs) -> Any:
        """Create a mock Doc object for the given text."""
        return MockDoc(text=text)
    
    def configure_pipeline(self, components: List[str]) -> None:
        """Configure the processing pipeline."""
        self.pipe_names = components
        self.enabled_components = set(components)
        self.disabled_components = set(self.components.keys()) - set(components)


class MockDoc(AbstractNLPDocMock):
    """Mock spaCy Doc object."""
    
    def __init__(self, text="test document", *args, **kwargs):
        super().__init__(text=text, *args, **kwargs)
        # Override tokens with MockToken instances
        self.tokens = self.create_tokens()
        self.ents = self.create_entities()
        self.sents = self.create_sentences()
        # Set up noun chunks
        self.setup_basic_noun_chunks()
    
    def create_tokens(self) -> List[Any]:
        """Create token objects for the document."""
        words = self.text.split()
        tokens = []
        for i, word in enumerate(words):
            token = MockToken(text=word)
            token.set_position(i, sum(len(w) + 1 for w in words[:i]))
            tokens.append(token)
        return tokens
    
    def create_entities(self) -> List[Any]:
        """Create named entity objects for the document."""
        # Use the base class method for basic entity detection
        self.setup_basic_entities()
        return self.ents
    
    def create_sentences(self) -> List[Any]:
        """Create sentence span objects for the document."""
        # Use the base class method for basic sentence segmentation
        self.setup_basic_sentences()
        return self.sents


class MockToken(AbstractNLPTokenMock):
    """Mock spaCy Token object."""
    
    def __init__(self, text="test", *args, **kwargs):
        super().__init__(text=text, *args, **kwargs)
        # Override defaults for BeamBuilder specific behavior
        self.lemma_ = text.lower()
        self.pos_ = "NOUN"
        self.tag_ = "NN"
        self.dep_ = "ROOT"
        self.is_stop = False
        self.is_alpha = True
    
    def get_linguistic_features(self) -> Dict[str, Any]:
        """Get linguistic features specific to this token type."""
        return {
            'text': self.text,
            'lemma': self.lemma_,
            'pos': self.pos_,
            'tag': self.tag_,
            'dep': self.dep_,
            'is_stop': self.is_stop,
            'is_alpha': self.is_alpha
        }
    
    def set_linguistic_attributes(self, **kwargs) -> None:
        """Set linguistic attributes for the token."""
        if 'lemma' in kwargs:
            self.set_lemma(kwargs['lemma'])
        if 'pos' in kwargs:
            self.set_pos_tag(kwargs['pos'], kwargs.get('tag'))
        if 'dep' in kwargs:
            self.set_dependency(kwargs['dep'], kwargs.get('head'))
        if 'ent_type' in kwargs:
            self.set_entity_annotation(
                kwargs.get('ent_type', ''),
                kwargs.get('ent_iob', 'O'),
                kwargs.get('ent_id', '')
            )
    
    def validate_token_consistency(self) -> bool:
        """Validate that token attributes are consistent."""
        # Basic consistency checks
        if not self.text:
            return False
        if self.is_alpha and not self.text.isalpha():
            return False
        if self.pos_ and self.tag_ and self.pos_ == "NOUN" and not self.tag_.startswith("N"):
            return False
        return True