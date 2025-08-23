"""
Mock classes for EmbeddingHelper tests.
"""

from unittest.mock import Mock
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Import mock classes from other modules
from .smied_mocks import MockSynset

# Import abstract base classes
from .base.collection_mock import AbstractCollectionMock
from .base.entity_mock import AbstractEntityMock, EntityType
from .base.edge_case_mock import AbstractEdgeCaseMock
from .base.integration_mock import AbstractIntegrationMock


class MockLemma(AbstractEntityMock):
    """Mock lemma for EmbeddingHelper tests."""
    
    def __init__(self, lemma_name="test", *args, **kwargs):
        super().__init__(entity_type=EntityType.LEMMA, *args, **kwargs)
        self.name = Mock(return_value=lemma_name)
        self.lemma_name = lemma_name
        
        # Set entity properties
        self.set_property("lemma_name", lemma_name)
        self.label = lemma_name
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute that identifies this lemma."""
        return self.lemma_name
    
    def validate_entity(self) -> bool:
        """Validate that the lemma is consistent and valid."""
        return bool(self.lemma_name and isinstance(self.lemma_name, str))
    
    def get_entity_signature(self) -> str:
        """Get a unique signature for this lemma."""
        return f"lemma:{self.lemma_name}:{self.id}"


class EmbeddingHelperMockFactory:
    """Factory class for creating EmbeddingHelper mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockEmbeddingHelper': MockEmbeddingHelper,
            'MockEmbeddingHelperEdgeCases': MockEmbeddingHelperEdgeCases,
            'MockEmbeddingHelperIntegration': MockEmbeddingHelperIntegration,
            'MockEmbeddingModelForHelper': MockEmbeddingModelForHelper,
            'MockVocabularyForHelper': MockVocabularyForHelper,
            'MockWordVectorsForHelper': MockWordVectorsForHelper,
            'MockRealWord2Vec': MockRealWord2Vec,
            'MockRealGloVe': MockRealGloVe,
            'MockRealFastText': MockRealFastText,
            'MockClusterResult': MockClusterResult,
            'MockSimilarityMatrix': MockSimilarityMatrix,
            'MockDimensionReducer': MockDimensionReducer,
            'MockSynset': MockSynset,
            'MockLemma': MockLemma,
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


class MockEmbeddingHelper(Mock):
    """Mock for EmbeddingHelper class testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize components
        self.embedding_model = MockEmbeddingModelForHelper()
        self.vector_cache = {}
        self.dimension = 300
        
        # Set up methods
        self.get_embedding = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        self.get_similarity = Mock(return_value=0.8)
        self.get_most_similar = Mock(return_value=[("similar_word", 0.9)])
        self.has_embedding = Mock(return_value=True)
        self.compute_centroid = Mock(return_value=np.array([0.0, 0.0, 0.0]))
        self.cluster_embeddings = Mock(return_value=[])
        
        # Vector operations
        self.cosine_similarity = Mock(return_value=0.8)
        self.euclidean_distance = Mock(return_value=0.5)
        self.normalize_vector = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        
        # Caching methods
        self.cache_embedding = Mock()
        self.clear_cache = Mock()
        self.get_cache_size = Mock(return_value=0)


class MockEmbeddingHelperEdgeCases(AbstractEdgeCaseMock):
    """Mock for EmbeddingHelper edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # EmbeddingHelper-specific edge case scenarios
        self.missing_embedding = Mock(return_value=None)
        self.zero_vector = Mock(return_value=np.zeros(300))
        self.invalid_model = Mock(side_effect=AttributeError("No embedding model"))
        self.out_of_vocab = Mock(return_value=False)
        
        # Set up edge case scenarios specific to embedding helper
        self.setup_edge_case_scenario("normal")
    
    def setup_edge_case_scenario(self, scenario_name: str) -> None:
        """Set up a specific edge case scenario for EmbeddingHelper."""
        self.current_scenario = scenario_name
        
        if scenario_name == "missing_embeddings":
            self.get_embedding = self.missing_embedding
            self.has_embedding = Mock(return_value=False)
        elif scenario_name == "zero_vectors":
            self.get_embedding = self.zero_vector
            self.has_embedding = Mock(return_value=True)
        elif scenario_name == "invalid_model":
            self.embedding_model = self.invalid_model
        elif scenario_name == "out_of_vocab":
            self.has_embedding = self.out_of_vocab
            self.get_embedding = self.missing_embedding
        elif scenario_name == "normal":
            # Reset to normal behavior
            self.get_embedding = Mock(return_value=np.array([0.1, 0.2, 0.3]))
            self.has_embedding = Mock(return_value=True)
    
    def get_edge_case_scenarios(self) -> List[str]:
        """Get list of available edge case scenarios for EmbeddingHelper."""
        return [
            "normal",
            "missing_embeddings", 
            "zero_vectors",
            "invalid_model",
            "out_of_vocab"
        ]


class MockEmbeddingHelperIntegration(AbstractIntegrationMock):
    """Mock for EmbeddingHelper integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set integration mode for embedding helper
        self.integration_mode = "full"
        
        # Set up integration components
        self._setup_embedding_components()
        
        # Initialize the integration
        if self.auto_setup:
            self.initialize_integration()
    
    def _setup_embedding_components(self):
        """Set up embedding-specific integration components."""
        # Create integration components
        self.real_word2vec = MockRealWord2Vec()
        self.real_glove = MockRealGloVe()
        self.real_fasttext = MockRealFastText()
        
        # Register components
        self.register_component("word2vec", self.real_word2vec)
        self.register_component("glove", self.real_glove)
        self.register_component("fasttext", self.real_fasttext, dependencies=["word2vec"])
        
        # Add embedding helper component (defer creation to avoid forward reference)
        self.embedding_helper = None  # Will be created in setup_integration_components
    
    def setup_integration_components(self) -> Dict[str, Any]:
        """Set up all components required for EmbeddingHelper integration testing."""
        # Create the embedding helper component now that MockEmbeddingHelper is available
        if self.embedding_helper is None:
            self.embedding_helper = MockEmbeddingHelper()
            self.register_component("embedding_helper", self.embedding_helper, 
                                  dependencies=["word2vec", "glove", "fasttext"])
        
        components = {
            "word2vec": self.real_word2vec,
            "glove": self.real_glove,
            "fasttext": self.real_fasttext,
            "embedding_helper": self.embedding_helper
        }
        return components
    
    def configure_component_interactions(self) -> None:
        """Configure how EmbeddingHelper components interact with each other."""
        # Ensure embedding helper is created
        if self.embedding_helper is None:
            self.setup_integration_components()
        
        # Configure embedding helper to use different models
        if self.embedding_helper:
            self.embedding_helper.embedding_model = self.real_word2vec
            
            # Set up model switching interactions
            self.switch_to_word2vec = self.create_component_interaction(
                "embedding_helper", "word2vec", "model_switch"
            )
            self.switch_to_glove = self.create_component_interaction(
                "embedding_helper", "glove", "model_switch"
            )
            self.switch_to_fasttext = self.create_component_interaction(
                "embedding_helper", "fasttext", "model_switch"
            )
    
    def validate_integration_state(self) -> bool:
        """Validate that the EmbeddingHelper integration is in a consistent state."""
        # Ensure components are set up
        if self.embedding_helper is None:
            self.setup_integration_components()
        
        # Check that all required components are registered
        required_components = ["word2vec", "glove", "fasttext"]
        for component_name in required_components:
            if component_name not in self.components:
                return False
        
        # Check embedding helper separately since it may be created later
        if "embedding_helper" in self.components:
            required_components.append("embedding_helper")
            
            # Check that embedding helper has a valid model
            if self.embedding_helper and not hasattr(self.embedding_helper, 'embedding_model'):
                return False
        
        # Validate component states
        for component_name in required_components:
            if self.component_states.get(component_name) not in ["registered", "initialized"]:
                return False
        
        return True


class MockEmbeddingModelForHelper(Mock):
    """Mock embedding model for EmbeddingHelper."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Model properties
        self.vector_size = 300
        self.vocab = MockVocabularyForHelper()
        
        # Model methods
        self.get_vector = Mock(return_value=np.random.rand(300))
        self.most_similar = Mock(return_value=[("similar", 0.9), ("related", 0.8)])
        self.similarity = Mock(return_value=0.8)
        self.wv = MockWordVectorsForHelper()
        
        # Model info
        self.__contains__ = Mock(return_value=True)
        self.__getitem__ = Mock(return_value=np.random.rand(300))


class MockVocabularyForHelper(AbstractCollectionMock):
    """Mock vocabulary for embedding model."""
    
    def __init__(self, vocab_words=None, *args, **kwargs):
        # Don't pass initial_data to super(), we'll handle it ourselves
        super().__init__(*args, **kwargs)
        
        # Configure collection properties for vocabulary
        self.collection_type = "vocabulary"
        self.is_mutable = False  # Vocabularies are typically immutable
        self.is_ordered = False
        self.allows_duplicates = False
        self.is_indexed = False
        
        # Initialize with vocabulary words using the list-based storage
        initial_words = vocab_words or ["word1", "word2", "word3"]
        self._items = initial_words.copy()
        self._data = None  # Use None to indicate we're using list-based storage
        self._keys = set(initial_words)
        
        # Set up vocabulary-specific attributes
        self.vocab_size = len(initial_words)
        
        # Override the collection interface methods to handle None data
        self.__len__ = Mock(side_effect=lambda: len(self._items))
        self.__contains__ = Mock(side_effect=lambda item: item in self._items)
        self.__iter__ = Mock(side_effect=lambda: iter(self._items))
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about this vocabulary collection."""
        return {
            "type": "embedding_vocabulary",
            "size": len(self),
            "is_mutable": self.is_mutable,
            "allows_duplicates": self.allows_duplicates,
            "vocab_words": list(self._items[:5])  # Sample of words
        }
    
    def validate_item(self, item: Any) -> bool:
        """Validate that an item can be stored in this vocabulary."""
        return isinstance(item, str) and len(item.strip()) > 0
    
    def transform_item(self, item: Any) -> Any:
        """Transform an item before storing it in the vocabulary."""
        if isinstance(item, str):
            return item.strip().lower()
        return str(item).strip().lower()


class MockWordVectorsForHelper(Mock):
    """Mock word vectors interface for embedding model."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Word vector methods
        self.__getitem__ = Mock(return_value=np.random.rand(300))
        self.__contains__ = Mock(return_value=True)
        self.get_vector = Mock(return_value=np.random.rand(300))
        self.most_similar = Mock(return_value=[("similar", 0.9)])
        self.similarity = Mock(return_value=0.8)
        self.vocab = MockVocabularyForHelper()


class MockRealWord2Vec(Mock):
    """Mock representing real Word2Vec model behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Word2Vec specific behavior
        self.vector_size = 300
        self.window = 5
        self.min_count = 5
        self.workers = 4
        self.sg = 0  # CBOW
        
        # Training data
        self.corpus_count = 1000000
        self.corpus_total_words = 50000000
        self.epochs = 5
        
        # Methods
        self.train = Mock()
        self.save = Mock()
        self.load = Mock()


class MockRealGloVe(Mock):
    """Mock representing real GloVe model behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GloVe specific behavior
        self.vector_size = 300
        self.learning_rate = 0.05
        self.epochs = 100
        
        # Training parameters
        self.x_max = 100
        self.alpha = 0.75
        
        # Methods
        self.fit = Mock()
        self.transform = Mock(return_value=np.random.rand(300))


class MockRealFastText(Mock):
    """Mock representing real FastText model behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FastText specific behavior
        self.vector_size = 300
        self.window = 5
        self.min_count = 5
        self.min_n = 3
        self.max_n = 6
        
        # Subword information
        self.bucket = 2000000
        
        # Methods
        self.train = Mock()
        self.get_word_vector = Mock(return_value=np.random.rand(300))
        self.get_subword_vector = Mock(return_value=np.random.rand(300))


class MockClusterResult(Mock):
    """Mock clustering result for EmbeddingHelper."""
    
    def __init__(self, n_clusters=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        self.labels_ = np.random.randint(0, n_clusters, 100)
        self.cluster_centers_ = np.random.rand(n_clusters, 300)
        self.inertia_ = 123.45
        
        # Methods
        self.fit = Mock(return_value=self)
        self.predict = Mock(return_value=np.random.randint(0, n_clusters, 10))
        self.fit_predict = Mock(return_value=self.labels_)


class MockSimilarityMatrix(Mock):
    """Mock similarity matrix for EmbeddingHelper."""
    
    def __init__(self, size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create symmetric similarity matrix
        self.matrix = np.random.rand(size, size)
        self.matrix = (self.matrix + self.matrix.T) / 2  # Make symmetric
        np.fill_diagonal(self.matrix, 1.0)  # Diagonal is 1.0
        
        # Matrix properties
        self.shape = (size, size)
        self.size = size * size
        
        # Methods
        self.__getitem__ = Mock(side_effect=lambda key: self.matrix[key])
        self.get_similarity = Mock(return_value=0.8)
        self.get_top_k_similar = Mock(return_value=[(1, 0.9), (2, 0.8)])


class MockDimensionReducer(Mock):
    """Mock dimension reducer for EmbeddingHelper."""
    
    def __init__(self, n_components=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_components = n_components
        
        # Reduction methods
        self.fit = Mock(return_value=self)
        self.transform = Mock(return_value=np.random.rand(100, n_components))
        self.fit_transform = Mock(return_value=np.random.rand(100, n_components))
        self.inverse_transform = Mock(return_value=np.random.rand(100, 300))