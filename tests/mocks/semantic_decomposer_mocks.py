"""
Mock classes for SemanticDecomposer tests.
"""

from unittest.mock import Mock
import networkx as nx
from typing import List, Optional, Any, Dict, Tuple

# Import abstract base classes
from tests.mocks.base.nlp_doc_mock import AbstractNLPDocMock
from tests.mocks.base.nlp_token_mock import AbstractNLPTokenMock
from tests.mocks.base.nlp_function_mock import AbstractNLPFunctionMock
from tests.mocks.base.algorithmic_function_mock import AbstractAlgorithmicFunctionMock
from tests.mocks.base.reasoning_mock import AbstractReasoningMock, ReasoningType, InferenceStrategy


class SemanticDecomposerMockFactory:
    """Factory class for creating SemanticDecomposer mock instances.
    
    This factory follows the SMIED Testing Framework Design Specifications
    for mock creation using factory pattern with abstract base class hierarchy.
    """
    
    def __init__(self):
        self._mock_classes = {
            # Core SemanticDecomposer mocks
            'MockSemanticDecomposer': MockSemanticDecomposer,
            'MockSemanticDecomposerValidation': MockSemanticDecomposerValidation,
            'MockSemanticDecomposerEdgeCases': MockSemanticDecomposerEdgeCases,
            'MockSemanticDecomposerIntegration': MockSemanticDecomposerIntegration,
            
            # WordNet and NLP component mocks
            'MockWordNetForDecomposer': MockWordNetForDecomposer,
            'MockSynsetForDecomposer': MockSynsetForDecomposer,
            'MockNLPForDecomposer': MockNLPForDecomposer,
            'MockDocForDecomposer': MockDocForDecomposer,
            'MockTokenForDecomposer': MockTokenForDecomposer,
            
            # Algorithmic component mocks
            'MockEmbeddingModelForDecomposer': MockEmbeddingModelForDecomposer,
            'MockEmbeddingHelperForDecomposer': MockEmbeddingHelperForDecomposer,
            'MockBeamBuilderForDecomposer': MockBeamBuilderForDecomposer,
            'MockGlossParserForDecomposer': MockGlossParserForDecomposer,
            'MockPairwiseBidirectionalAStar': MockPairwiseBidirectionalAStar,
            
            # Graph and data structure mocks
            'MockNetworkXGraph': MockNetworkXGraph,
            'MockRealNetworkXGraph': MockRealNetworkXGraph,
            'MockPatternMatcher': MockPatternMatcher,
            
            # Integration and edge case mocks
            'MockRealWordNet': MockRealWordNet,
            'MockFrameNetSpacySRL': MockFrameNetSpacySRL,
            'MockDerivationalMorphology': MockDerivationalMorphology,
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


class MockSemanticDecomposer(Mock):
    """Mock for SemanticDecomposer class testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize components
        self.wn_module = MockWordNetForDecomposer()
        self.nlp_func = MockNLPForDecomposer()
        self.embedding_model = MockEmbeddingModelForDecomposer()
        self._synset_graph = None
        
        # Initialize helper components
        self.embedding_helper = MockEmbeddingHelperForDecomposer()
        self.beam_builder = MockBeamBuilderForDecomposer()
        self.gloss_parser = MockGlossParserForDecomposer()
        
        # Set up main methods
        self.find_connected_shortest_paths = Mock(return_value=(None, None, None))
        self.build_synset_graph = Mock(return_value=MockNetworkXGraph())
        self.show_path = Mock()
        self.show_connected_paths = Mock()
        
        # Internal methods
        self._find_subject_to_predicate_paths = Mock(return_value=[])
        self._find_predicate_to_object_paths = Mock(return_value=[])
        self._get_best_synset_matches = Mock(return_value=[])
        self._find_path_between_synsets = Mock(return_value=None)
        self._explore_hypernym_paths = Mock(return_value=[])


class MockSemanticDecomposerIntegration(Mock):
    """Mock for SemanticDecomposer integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integration components
        self.real_wordnet = MockRealWordNet()
        self.real_graph = MockRealNetworkXGraph()
        self.pathfinder = MockPairwiseBidirectionalAStar()
    
    def create_integration_synset_mocks(self, synset_configs):
        """Create synset mocks for integration testing.
        
        Args:
            synset_configs: List of dicts like:
                [{'name': 'cat.n.01', 'definition': 'feline mammal'}]
        """
        synsets = []
        for config in synset_configs:
            # Use the factory to create proper synset mocks
            synset_name = config.get('name', 'test.n.01')
            mock_synset = MockSynsetForDecomposer(
                name=synset_name,
                definition=config.get('definition', 'test definition'),
                pos=config.get('pos', 'n')
            )
            synsets.append(mock_synset)
        return synsets
    
    def create_integration_graph(self, node_configs, edge_configs):
        """Create NetworkX graph for integration testing.
        
        Args:
            node_configs: List of node names
            edge_configs: List of (source, target) tuples
        """
        graph = nx.DiGraph()
        for node in node_configs:
            graph.add_node(node)
        for source, target in edge_configs:
            graph.add_edge(source, target)
        return graph
    
    def create_pathfinding_scenario(self, scenario_name='simple_triple'):
        """Create complete pathfinding scenario with synsets and graph.
        
        Returns dict with synsets, graph, and configuration.
        """
        scenarios = {
            'simple_triple': {
                'synsets': [
                    {'name': 'cat.n.01', 'definition': 'a small carnivorous mammal'},
                    {'name': 'run.v.01', 'definition': 'move fast by using legs'},
                    {'name': 'park.n.01', 'definition': 'a large public garden'}
                ],
                'nodes': ['cat.n.01', 'run.v.01', 'park.n.01'],
                'edges': [('cat.n.01', 'run.v.01'), ('run.v.01', 'park.n.01')],
                'expected_paths': [(["cat.n.01", "run.v.01"], 1.0)]
            },
            'complex_network': {
                'synsets': [
                    {'name': 'cat.n.01', 'definition': 'feline mammal'},
                    {'name': 'animal.n.01', 'definition': 'living organism'},
                    {'name': 'run.v.01', 'definition': 'move fast'},
                    {'name': 'park.n.01', 'definition': 'public garden'},
                    {'name': 'mammal.n.01', 'definition': 'warm-blooded vertebrate'}
                ],
                'nodes': ['cat.n.01', 'animal.n.01', 'run.v.01', 'park.n.01', 'mammal.n.01'],
                'edges': [
                    ('cat.n.01', 'mammal.n.01'),
                    ('mammal.n.01', 'animal.n.01'),
                    ('animal.n.01', 'run.v.01'),
                    ('run.v.01', 'park.n.01')
                ],
                'expected_paths': [
                    (["cat.n.01", "mammal.n.01", "animal.n.01", "run.v.01"], 2.0),
                    (["run.v.01", "park.n.01"], 1.0)
                ]
            }
        }
        
        if scenario_name not in scenarios:
            scenario_name = 'simple_triple'
        
        scenario = scenarios[scenario_name]
        
        synsets = self.create_integration_synset_mocks(scenario['synsets'])
        graph = self.create_integration_graph(scenario['nodes'], scenario['edges'])
        
        # Create synset name lookup
        synset_lookup = {}
        for i, synset in enumerate(synsets):
            synset_lookup[scenario['synsets'][i]['name']] = synset
        
        return {
            'synsets': synsets,
            'graph': graph,
            'synset_lookup': synset_lookup,
            'expected_paths': scenario['expected_paths'],
            'scenario_config': scenario
        }
    
    def setup_wordnet_mock_responses(self, wordnet_mock, scenario):
        """Setup WordNet mock responses for a scenario."""
        synsets = scenario['synsets']
        synset_lookup = scenario['synset_lookup']
        
        # Setup synsets() calls
        wordnet_mock.synsets.side_effect = [
            [synset] for synset in synsets[:3]  # First 3 synsets for subject, predicate, object
        ]
        
        # Setup synset() calls
        wordnet_mock.synset.side_effect = lambda name: synset_lookup.get(name)
        
        # Setup constants
        wordnet_mock.NOUN = 'n'
        wordnet_mock.VERB = 'v'
        
        return wordnet_mock
    
    def create_mock_gloss_result(self, result_type='basic'):
        """Create mock gloss parsing result."""
        if result_type == 'basic':
            return {
                'subjects': [Mock()],
                'objects': [Mock()],
                'predicates': [Mock()],
                'raw_subjects': ['cat'],
                'raw_objects': ['park'],
                'raw_predicates': ['run']
            }
        elif result_type == 'empty':
            return {
                'subjects': [],
                'objects': [],
                'predicates': [],
                'raw_subjects': [],
                'raw_objects': [],
                'raw_predicates': []
            }
        else:
            return {
                'subjects': [Mock(), Mock()],
                'objects': [Mock(), Mock()],
                'predicates': [Mock()],
                'raw_subjects': ['cat', 'animal'],
                'raw_objects': ['park', 'place'],
                'raw_predicates': ['run']
            }


class MockWordNetForDecomposer(Mock):
    """Mock WordNet module for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synsets = Mock(side_effect=self._robust_mock_synsets)
        self.synset = Mock(side_effect=self._mock_synset)
        self.all_synsets = Mock(return_value=[self._create_mock_synset("test.n.01")])
        # Add constants
        self.NOUN = 'n'
        self.VERB = 'v'
        # Keep track of original side_effect for fallback
        self._original_side_effect = None
        self._call_count = 0
        
    def _robust_mock_synsets(self, word, pos=None):
        """Robust mock synsets that handles both finite and infinite calls."""
        # If a test has set a finite side_effect, use it first, then fallback
        if hasattr(self.synsets, '_mock_side_effect') and self.synsets._mock_side_effect:
            try:
                # If it's a list, try to get the next item
                if isinstance(self.synsets._mock_side_effect, list):
                    if self._call_count < len(self.synsets._mock_side_effect):
                        result = self.synsets._mock_side_effect[self._call_count]
                        self._call_count += 1
                        return result
                    else:
                        # Exhausted the list, fall back to default behavior
                        pass
            except (StopIteration, IndexError):
                # Fallback to default behavior
                pass
        
        # Use the default behavior
        return self._mock_synsets(word, pos)
    
    def _mock_synsets(self, word, pos=None):
        """Mock synsets method behavior that handles common words and returns empty for unknown."""
        word_lower = word.lower() if isinstance(word, str) else str(word).lower()
        
        # Define common synset mappings
        synset_mappings = {
            "cat": "cat.n.01",
            "run": "run.v.01", 
            "park": "park.n.01",
            "entity": "entity.n.01",
            "agent": "agent.n.01",
            "experiencer": "experiencer.n.01",
            "theme": "theme.n.01", 
            "patient": "patient.n.01",
            "goal": "goal.n.01",
            "beneficiary": "beneficiary.n.01",
            "teacher": "teacher.n.01",
            "teach": "teach.v.01",
            "student": "student.n.01",
            "emotion": "emotion.n.01",
            "connect": "connect.v.01",
            "rock": "rock.n.01",
            "mammal": "mammal.n.01",
            "relate": "relate.v.01"
        }
        
        if word_lower in synset_mappings:
            return [self._create_mock_synset(synset_mappings[word_lower])]
        
        # Return empty list for unknown words (this prevents StopIteration)
        return []
    
    def _mock_synset(self, name):
        """Mock synset method behavior."""
        return self._create_mock_synset(name)
    
    def _create_mock_synset(self, name):
        """Create a mock synset with the given name."""
        synset = Mock()
        synset.name = Mock(return_value=name)
        synset.definition = Mock(return_value=f"Definition for {name}")
        synset.pos = Mock(return_value=name.split('.')[1])
        synset.examples = Mock(return_value=[f"Example for {name}"])
        # Create proper lemma mock for synset with all necessary methods
        lemma_mock = Mock()
        lemma_mock.name = Mock(return_value=name.split('.')[0])
        lemma_mock.derivationally_related_forms = Mock(return_value=[])
        lemma_mock.pertainyms = Mock(return_value=[])
        lemma_mock.antonyms = Mock(return_value=[])
        synset.lemmas = Mock(return_value=[lemma_mock])
        
        # Relations - using correct method names
        synset.hypernyms = Mock(return_value=[])
        synset.hyponyms = Mock(return_value=[])
        synset.member_holonyms = Mock(return_value=[])
        synset.part_holonyms = Mock(return_value=[])
        synset.substance_holonyms = Mock(return_value=[])
        synset.member_meronyms = Mock(return_value=[])
        synset.part_meronyms = Mock(return_value=[])
        synset.substance_meronyms = Mock(return_value=[])
        synset.similar_tos = Mock(return_value=[])
        synset.also_sees = Mock(return_value=[])
        synset.verb_groups = Mock(return_value=[])
        synset.entailments = Mock(return_value=[])
        synset.causes = Mock(return_value=[])
        synset.attributes = Mock(return_value=[])
        
        # Similarity methods
        synset.path_similarity = Mock(return_value=0.5)
        synset.wup_similarity = Mock(return_value=0.8)
        synset.lch_similarity = Mock(return_value=2.5)
        
        return synset


class MockNLPForDecomposer(AbstractNLPFunctionMock):
    """Mock NLP function for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_value = MockDocForDecomposer()
        # Configure for semantic decomposer usage
        self.model_name = "semantic_decomposer_nlp"
        self.lang = "en"
    
    def process_text(self, text: str, **kwargs) -> Any:
        """Process a single text and return a Doc object."""
        return MockDocForDecomposer(text)
    
    def create_doc_mock(self, text: str, **kwargs) -> Any:
        """Create a mock Doc object for the given text."""
        return MockDocForDecomposer(text)
    
    def configure_pipeline(self, components: List[str]) -> None:
        """Configure the processing pipeline."""
        self.pipe_names = components
        self.enabled_components = set(components)
        self.disabled_components = set()


class MockDocForDecomposer(AbstractNLPDocMock):
    """Mock document for SemanticDecomposer NLP processing."""
    
    def __init__(self, text="test document", *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        # Preserve original behavior while using base class functionality
        self.tokens = [MockTokenForDecomposer("test")]
        self.ents = []
        self.noun_chunks = []
        
        # Set up basic processing using base class methods
        self.setup_basic_tokenization()
    
    def create_tokens(self) -> List[Any]:
        """Create token objects for the document."""
        words = self.text.split()
        tokens = []
        for word in words:
            token = MockTokenForDecomposer(word)
            tokens.append(token)
        return tokens
    
    def create_entities(self) -> List[Any]:
        """Create named entity objects for the document."""
        # Return empty list for semantic decomposer - no entities by default
        return []
    
    def create_sentences(self) -> List[Any]:
        """Create sentence span objects for the document."""
        # Simple sentence creation
        sent_mock = Mock()
        sent_mock.text = self.text
        sent_mock.tokens = self.tokens
        return [sent_mock]


class MockTokenForDecomposer(AbstractNLPTokenMock):
    """Mock token for SemanticDecomposer NLP processing."""
    
    def __init__(self, text="test", *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        # Set specific attributes for semantic decomposer
        self.lemma_ = text.lower()
        self.pos_ = "NOUN"
        self.dep_ = "ROOT"
    
    def get_linguistic_features(self) -> Dict[str, Any]:
        """Get linguistic features specific to this token type."""
        return {
            'text': self.text,
            'lemma': self.lemma_,
            'pos': self.pos_,
            'dep': self.dep_,
            'is_alpha': self.is_alpha,
            'is_stop': self.is_stop
        }
    
    def set_linguistic_attributes(self, **kwargs) -> None:
        """Set linguistic attributes for the token."""
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
    
    def validate_token_consistency(self) -> bool:
        """Validate that token attributes are consistent."""
        # Basic consistency checks
        if not isinstance(self.text, str):
            return False
        if not isinstance(self.lemma_, str):
            return False
        if not isinstance(self.pos_, str):
            return False
        return True


class MockEmbeddingModelForDecomposer(AbstractReasoningMock):
    """Mock embedding model for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for embedding model reasoning
        self.engine_name = "embedding_model_reasoning"
        self.reasoning_type = ReasoningType.SIMILARITY
        self.inference_strategy = InferenceStrategy.BEST_FIRST
        
        # Set up embedding-specific methods
        self.get_vector = Mock(return_value=[0.1, 0.2, 0.3])
        self.most_similar = Mock(return_value=[("similar", 0.9)])
        self.similarity = Mock(return_value=0.8)
        
        # Configure similarity engine for embeddings
        self.similarity_engine.get_vector = self.get_vector
        self.similarity_engine.most_similar = self.most_similar
        self.similarity_engine.similarity = self.similarity
    
    def infer(self, query: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Perform embedding-based inference on a query."""
        from tests.mocks.base.reasoning_mock import InferenceResult
        
        # Mock embedding inference - find similar embeddings
        if hasattr(query, '__iter__') and not isinstance(query, str):
            # Query is a vector
            similar_items = self.most_similar.return_value
        else:
            # Query is a string/token
            vector = self.get_vector(query) if callable(self.get_vector) else self.get_vector.return_value
            similar_items = self.most_similar.return_value
        
        return InferenceResult(
            conclusion=similar_items[0] if similar_items else ("unknown", 0.0),
            confidence=0.8,
            reasoning_path=["Embedding similarity computation"],
            evidence=[query],
            metadata={
                'inference_type': 'embedding_similarity',
                'vector_dimensions': len(self.get_vector.return_value),
                'model_type': 'embedding_model'
            }
        )
    
    def compute_similarity(self, entity1: Any, entity2: Any) -> float:
        """Compute embedding similarity between two entities."""
        # Mock embedding similarity computation
        if entity1 == entity2:
            return 1.0
        
        # Get cached similarity if available
        cache_key = f"{hash(str(entity1))}_{hash(str(entity2))}"
        if self.enable_caching and cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Compute mock similarity based on embedding vectors
        vector1 = self.get_vector(entity1) if callable(self.get_vector) else self.get_vector.return_value
        vector2 = self.get_vector(entity2) if callable(self.get_vector) else self.get_vector.return_value
        
        # Simple cosine similarity mock
        similarity_score = 0.8  # Default mock similarity
        
        if self.enable_caching:
            self.similarity_cache[cache_key] = similarity_score
        
        return similarity_score
    
    def explain_reasoning(self, inference_result) -> List[str]:
        """Generate explanation for embedding reasoning process."""
        explanations = [
            "Used embedding model to compute vector representations",
            "Performed similarity computation in embedding space",
            f"Found similarity score: {inference_result.confidence}"
        ]
        
        if 'vector_dimensions' in inference_result.metadata:
            explanations.append(f"Embedding dimensions: {inference_result.metadata['vector_dimensions']}")
        
        return explanations


class MockEmbeddingHelperForDecomposer(AbstractReasoningMock):
    """Mock EmbeddingHelper for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for embedding helper reasoning
        self.engine_name = "embedding_helper_reasoning"
        self.reasoning_type = ReasoningType.SIMILARITY
        self.inference_strategy = InferenceStrategy.BEST_FIRST
        
        # Set up embedding helper methods
        self.get_similarity = Mock(return_value=0.8)
        self.get_most_similar = Mock(return_value=[])
        self.has_embedding = Mock(return_value=True)
        
        # Configure similarity engine for embedding operations
        self.similarity_engine.get_similarity = self.get_similarity
        self.similarity_engine.get_most_similar = self.get_most_similar
        self.similarity_engine.has_embedding = self.has_embedding
    
    def infer(self, query: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Perform embedding helper inference on a query."""
        from tests.mocks.base.reasoning_mock import InferenceResult
        
        # Mock embedding helper inference - similarity reasoning
        has_emb = self.has_embedding(query) if callable(self.has_embedding) else self.has_embedding.return_value
        
        if not has_emb:
            return InferenceResult(
                conclusion=None,
                confidence=0.0,
                reasoning_path=["No embedding available for query"],
                evidence=[query],
                metadata={
                    'inference_type': 'embedding_availability_check',
                    'has_embedding': False
                }
            )
        
        # Get most similar items
        similar_items = self.get_most_similar(query) if callable(self.get_most_similar) else self.get_most_similar.return_value
        
        return InferenceResult(
            conclusion=similar_items,
            confidence=0.8,
            reasoning_path=["Embedding similarity search", "Retrieved most similar items"],
            evidence=[query],
            metadata={
                'inference_type': 'embedding_similarity_search',
                'has_embedding': True,
                'similar_items_count': len(similar_items)
            }
        )
    
    def compute_similarity(self, entity1: Any, entity2: Any) -> float:
        """Compute similarity using embedding helper."""
        # Check if both entities have embeddings
        has_emb1 = self.has_embedding(entity1) if callable(self.has_embedding) else self.has_embedding.return_value
        has_emb2 = self.has_embedding(entity2) if callable(self.has_embedding) else self.has_embedding.return_value
        
        if not (has_emb1 and has_emb2):
            return 0.0
        
        # Use the get_similarity method
        similarity_score = self.get_similarity(entity1, entity2) if callable(self.get_similarity) else self.get_similarity.return_value
        
        return similarity_score
    
    def explain_reasoning(self, inference_result) -> List[str]:
        """Generate explanation for embedding helper reasoning process."""
        explanations = [
            "Used embedding helper for similarity reasoning",
            "Checked embedding availability for entities"
        ]
        
        if inference_result.metadata.get('has_embedding', False):
            explanations.append("Embeddings found - computed similarity scores")
            if 'similar_items_count' in inference_result.metadata:
                explanations.append(f"Retrieved {inference_result.metadata['similar_items_count']} similar items")
        else:
            explanations.append("No embeddings available - unable to compute similarity")
        
        return explanations


class MockBeamBuilderForDecomposer(AbstractAlgorithmicFunctionMock):
    """Mock BeamBuilder for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for beam search algorithm
        self.algorithm_name = "beam_builder"
        self.algorithm_type = "search"
        self.complexity_class = "O(b^d)"
        self.beam_width = 10
        
        # Set up mock methods with proper return values
        self.build_beam = Mock(return_value=[])
        self.expand_candidates = Mock(return_value=[])
    
    def compute(self, *args, **kwargs) -> Any:
        """Compute the main beam building algorithm."""
        # Mock beam building computation
        if "query" in kwargs:
            return self.build_beam(kwargs["query"])
        return []
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input arguments for beam building."""
        # Basic validation for beam building
        if "beam_width" in kwargs:
            return isinstance(kwargs["beam_width"], int) and kwargs["beam_width"] > 0
        return True
    
    def get_algorithm_properties(self) -> Dict[str, Any]:
        """Get properties specific to beam building algorithm."""
        return {
            "beam_width": self.beam_width,
            "search_type": "best_first",
            "heuristic_based": True,
            "complete": False,
            "optimal": False
        }


class MockGlossParserForDecomposer(Mock):
    """Mock GlossParser for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parse_gloss = Mock(return_value={
            'subjects': [],
            'predicates': [],
            'objects': []
        })
        self.extract_relations = Mock(return_value=[])


class MockNetworkXGraph(Mock):
    """Mock NetworkX graph for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = Mock(return_value=["cat.n.01", "animal.n.01"])
        self.edges = Mock(return_value=[("animal.n.01", "cat.n.01")])
        self.number_of_nodes = Mock(return_value=100)
        self.number_of_edges = Mock(return_value=200)
        self.add_node = Mock()
        self.add_edge = Mock()
        self.has_node = Mock(return_value=True)
        self.__contains__ = Mock(return_value=True)


class MockRealWordNet(Mock):
    """Mock representing real WordNet behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simulate real WordNet with more complex behavior
        self.all_synsets = Mock(return_value=self._create_multiple_synsets())
    
    def _create_multiple_synsets(self):
        """Create multiple mock synsets for realistic testing."""
        synsets = []
        for i in range(5):
            synset = Mock()
            synset.name = Mock(return_value=f"test{i}.n.01")
            synset.definition = Mock(return_value=f"Test synset {i}")
            synset.hypernyms = Mock(return_value=[])
            synset.hyponyms = Mock(return_value=[])
            synsets.append(synset)
        return synsets


class MockRealNetworkXGraph(Mock):
    """Mock representing real NetworkX graph behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a more realistic graph structure
        self._nodes = set()
        self._edges = set()
        
        # Mock methods with realistic behavior
        self.add_node = Mock(side_effect=lambda n: self._nodes.add(n))
        self.add_edge = Mock(side_effect=lambda u, v, **attr: self._edges.add((u, v)))
        self.nodes = Mock(return_value=list(self._nodes))
        self.edges = Mock(return_value=list(self._edges))
        self.number_of_nodes = Mock(return_value=len(self._nodes))
        self.number_of_edges = Mock(return_value=len(self._edges))


class MockPairwiseBidirectionalAStar(AbstractAlgorithmicFunctionMock):
    """Mock for PairwiseBidirectionalAStar pathfinding."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for A* pathfinding algorithm
        self.algorithm_name = "pairwise_bidirectional_astar"
        self.algorithm_type = "pathfinding"
        self.complexity_class = "O(b^d)"
        
        # Set up pathfinding methods
        self.find_paths = Mock(return_value=[(["start", "end"], 1.0)])
        self.find_shortest_path = Mock(return_value=["start", "end"])
    
    def compute(self, *args, **kwargs) -> Any:
        """Compute A* pathfinding algorithm."""
        # Mock A* pathfinding computation
        if "start" in kwargs and "end" in kwargs:
            return self.find_shortest_path(kwargs["start"], kwargs["end"])
        elif "paths" in kwargs:
            return self.find_paths()
        return []
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input arguments for A* pathfinding."""
        # Basic validation for pathfinding
        if "start" in kwargs and "end" in kwargs:
            return kwargs["start"] is not None and kwargs["end"] is not None
        return True
    
    def get_algorithm_properties(self) -> Dict[str, Any]:
        """Get properties specific to A* pathfinding algorithm."""
        return {
            "bidirectional": True,
            "optimal": True,
            "complete": True,
            "heuristic_based": True,
            "memory_usage": "exponential",
            "pathfinding_type": "informed_search"
        }


class MockPatternMatcher(AbstractAlgorithmicFunctionMock):
    """Mock PatternMatcher for SemanticDecomposer tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for pattern matching algorithm
        self.algorithm_name = "pattern_matcher"
        self.algorithm_type = "matching"
        self.complexity_class = "O(nm)"
        
        # Set up mock methods with proper return values
        self.match = Mock(return_value=[])
        self.find_patterns = Mock(return_value=[])
    
    def compute(self, *args, **kwargs) -> Any:
        """Compute pattern matching."""
        if "pattern" in kwargs and "text" in kwargs:
            return self.match(kwargs["pattern"], kwargs["text"])
        elif "patterns" in kwargs:
            return self.find_patterns(kwargs["patterns"])
        return []
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input arguments for pattern matching."""
        # Check for required pattern matching inputs
        if "pattern" in kwargs:
            return kwargs["pattern"] is not None
        if "patterns" in kwargs:
            return isinstance(kwargs["patterns"], (list, tuple))
        return True
    
    def get_algorithm_properties(self) -> Dict[str, Any]:
        """Get properties specific to pattern matching algorithm."""
        return {
            "pattern_type": "regex",
            "case_sensitive": True,
            "multiline": False,
            "greedy": True,
            "supports_groups": True
        }


class MockSemanticDecomposerValidation(Mock):
    """Mock for SemanticDecomposer validation testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up validation-specific behavior
        self.validate_inputs = Mock(return_value=True)
        self.validate_synsets = Mock(return_value=True)
        self.validate_graph = Mock(return_value=True)
        self.validate_parameters = Mock(return_value=True)
        
        # Validation error methods
        self.get_validation_errors = Mock(return_value=[])
        self.is_valid_synset = Mock(return_value=True)
        self.is_valid_graph = Mock(return_value=True)


class MockSemanticDecomposerEdgeCases(Mock):
    """Mock for SemanticDecomposer edge case testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up edge case behaviors
        self.handle_empty_synsets = Mock(return_value=(None, None, None))
        self.handle_invalid_graph = Mock(return_value=(None, None, None))
        self.handle_missing_paths = Mock(return_value=(None, None, None))
        self.handle_timeout = Mock(side_effect=TimeoutError("Path finding timeout"))
        self.handle_memory_limit = Mock(side_effect=MemoryError("Memory limit exceeded"))


class MockSynsetForDecomposer(Mock):
    """Mock synset specifically designed for SemanticDecomposer testing."""
    
    def __init__(self, name="test.n.01", definition="test definition", pos="n", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        self._definition = definition
        self._pos = pos
        
        # Configure basic synset attributes
        self.name = Mock(return_value=name)
        self.definition = Mock(return_value=definition)
        self.pos = Mock(return_value=pos)
        self.examples = Mock(return_value=[f"Example for {name}"])
        # Create proper lemma mock with all necessary methods
        lemma_mock = Mock()
        lemma_mock.name = Mock(return_value=name.split('.')[0])
        lemma_mock.derivationally_related_forms = Mock(return_value=[])
        lemma_mock.pertainyms = Mock(return_value=[])
        lemma_mock.antonyms = Mock(return_value=[])
        self.lemmas = Mock(return_value=[lemma_mock])
        
        # Configure semantic relations
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
        
        # Configure similarity methods
        self.path_similarity = Mock(return_value=0.5)
        self.wup_similarity = Mock(return_value=0.8)
        self.lch_similarity = Mock(return_value=2.5)
        
        # Add missing methods required for SemanticDecomposer path finding
        self.hypernym_paths = Mock(return_value=[[self]])  # Returns list of paths, each path is a list of synsets
        self.lowest_common_hypernyms = Mock(return_value=[self])  # Returns list of common hypernym synsets


class MockFrameNetSpacySRL(Mock):
    """Mock FrameNetSpacySRL for SemanticDecomposer testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up FrameNet SRL methods
        self.extract_frames = Mock(return_value=[])
        self.find_frame_connections = Mock(return_value=[])
        self.get_semantic_roles = Mock(return_value={})
        self.analyze_frame_relations = Mock(return_value=[])
        
        # Add the critical missing process_triple method
        self.process_triple = Mock(return_value={
            "test_synset.v.01": {
                "subjects": {"Agent", "Experiencer"},
                "objects": {"Theme", "Patient"}, 
                "themes": {"Goal", "Beneficiary"}
            }
        })


class MockDerivationalMorphology(Mock):
    """Mock derivational morphology component for SemanticDecomposer testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up derivational morphology methods
        self.find_derivational_relations = Mock(return_value=[])
        self.get_morphological_variants = Mock(return_value=[])
        self.analyze_word_formation = Mock(return_value={})