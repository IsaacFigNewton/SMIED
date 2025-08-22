"""
Mock classes for EmbeddingHelper tests.
"""

from unittest.mock import Mock
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


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


class MockEmbeddingHelperEdgeCases(Mock):
    """Mock for EmbeddingHelper edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Edge case scenarios
        self.missing_embedding = Mock(return_value=None)
        self.zero_vector = Mock(return_value=np.zeros(300))
        self.invalid_model = Mock(side_effect=AttributeError("No embedding model"))
        self.out_of_vocab = Mock(return_value=False)


class MockEmbeddingHelperIntegration(Mock):
    """Mock for EmbeddingHelper integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integration components
        self.real_word2vec = MockRealWord2Vec()
        self.real_glove = MockRealGloVe()
        self.real_fasttext = MockRealFastText()


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


class MockVocabularyForHelper(Mock):
    """Mock vocabulary for embedding model."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Vocabulary methods
        self.__contains__ = Mock(return_value=True)
        self.__len__ = Mock(return_value=10000)
        self.__iter__ = Mock(return_value=iter(["word1", "word2", "word3"]))
        self.keys = Mock(return_value=["word1", "word2", "word3"])


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