"""
Mock classes for SMIED tests following the SMIED Testing Framework Design Specifications.

This module implements the MockFactory pattern and abstract base class hierarchy
for comprehensive SMIED component testing.
"""

from unittest.mock import Mock
from typing import Optional, List, Any, Dict
from tests.mocks.base.library_wrapper_mock import AbstractLibraryWrapperMock
from tests.mocks.base.entity_mock import AbstractEntityMock, EntityType


class SMIEDMockFactory:
    """Factory class for creating SMIED mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockISMIEDPipeline': MockISMIEDPipeline,
            'MockSMIED': MockSMIED,
            'MockSMIEDIntegration': MockSMIEDIntegration,
            'MockSMIEDEdgeCases': MockSMIEDEdgeCases,
            'MockNLTK': MockNLTK,
            'MockSpacy': MockSpacy,
            'MockWordNet': MockWordNet,
            'MockSynset': MockSynset,
            'MockSynsetEdgeCases': MockSynsetEdgeCases,
            'MockSemanticDecomposer': MockSemanticDecomposer,
            'MockGraph': MockGraph,
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


class MockISMIEDPipeline(Mock):
    """Mock for ISMIEDPipeline interface testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up interface methods
        self.reinitialize = Mock()
        self.analyze_triple = Mock()
        self.get_synsets = Mock()
        self.display_results = Mock()


class MockSMIED(Mock):
    """Mock for SMIED class testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize attributes
        self.nlp_model_name = kwargs.get('nlp_model', 'en_core_web_sm')
        self.embedding_model = kwargs.get('embedding_model', None)
        self.auto_download = kwargs.get('auto_download', True)
        self.nlp = None
        self.decomposer = Mock()
        self.synset_graph = None
        
        # Set up methods
        self.reinitialize = Mock()
        self.build_synset_graph = Mock()
        self.analyze_triple = Mock(return_value=(None, None, None))
        self.get_synsets = Mock(return_value=[])
        self.display_results = Mock()
        self.calculate_similarity = Mock(return_value=None)
        self.get_word_info = Mock(return_value={'word': 'test', 'synsets': [], 'total_senses': 0})
        self.demonstrate_alternative_approaches = Mock()
        self._setup_nlp = Mock(return_value=None)
        self._display_synsets = Mock()
        self._show_fallback_relationships = Mock()
        self._show_hypernym_path = Mock()
        self._show_verb_relations = Mock()


class MockSMIEDEdgeCases(Mock):
    """Mock for SMIED edge case testing."""
    
    def __init__(self, failure_mode=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_mode = failure_mode
        
        # Set up methods that may fail
        if failure_mode == 'initialization_error':
            self.reinitialize = Mock(side_effect=RuntimeError("Initialization failed"))
            self.build_synset_graph = Mock(side_effect=MemoryError("Out of memory"))
        elif failure_mode == 'analysis_error':
            self.analyze_triple = Mock(side_effect=ValueError("Invalid triple format"))
            self.get_synsets = Mock(side_effect=LookupError("Synsets not found"))
        elif failure_mode == 'resource_error':
            self.build_synset_graph = Mock(side_effect=OSError("Resource unavailable"))
            self.calculate_similarity = Mock(side_effect=TimeoutError("Processing timeout"))
        else:
            # Normal behavior with edge case configurations
            self.reinitialize = Mock()
            self.analyze_triple = Mock(return_value=([], [], None))
            self.get_synsets = Mock(return_value=[])
            self.build_synset_graph = Mock(return_value=None)
            self.calculate_similarity = Mock(return_value=None)
        
        # Common attributes
        self.nlp_model_name = kwargs.get('nlp_model', 'nonexistent_model')
        self.embedding_model = kwargs.get('embedding_model', 'unavailable_model')
        self.auto_download = kwargs.get('auto_download', False)
        self.nlp = None
        self.decomposer = None
        self.synset_graph = None


class MockSMIEDIntegration(Mock):
    """Mock for SMIED integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mock components for integration testing
        self.mock_nltk = Mock()
        self.mock_spacy = Mock()
    
    def create_full_pipeline_mocks(self):
        """Create complete set of mocks for full pipeline testing."""
        # Create NLP mock
        mock_nlp = Mock()
        
        # Create decomposer mock
        mock_decomposer = Mock()
        
        # Create graph mock with realistic data
        mock_graph = Mock()
        mock_graph.number_of_nodes.return_value = 1000
        mock_graph.number_of_edges.return_value = 5000
        mock_decomposer.build_synset_graph.return_value = mock_graph
        
        # Create synset mocks
        mock_synset = Mock()
        mock_synset.name.return_value = "test.n.01"
        mock_synset.definition.return_value = "test definition"
        
        # Setup decomposer result
        mock_decomposer.find_connected_shortest_paths.return_value = (
            [mock_synset], [mock_synset], mock_synset
        )
        
        # Create WordNet mock
        mock_wn = Mock()
        mock_wn.synsets.return_value = [mock_synset]
        mock_wn.NOUN = 'n'
        mock_wn.VERB = 'v'
        
        # Setup NLTK mock
        mock_nltk = Mock()
        mock_nltk.download = Mock()
        
        return {
            'mock_nlp': mock_nlp,
            'mock_decomposer': mock_decomposer,
            'mock_graph': mock_graph,
            'mock_synset': mock_synset,
            'mock_wn': mock_wn,
            'mock_nltk': mock_nltk
        }
    
    def setup_integration_patches(self, test_mocks):
        """Return context managers for patching in integration tests."""
        from unittest.mock import patch
        
        patches = {
            'nltk': patch('smied.SMIED.nltk', return_value=test_mocks['mock_nltk']),
            'spacy': patch('spacy.load', return_value=test_mocks['mock_nlp']),
            'wordnet': patch('smied.SMIED.wn', return_value=test_mocks['mock_wn']),
            'decomposer': patch('smied.SMIED.SemanticDecomposer', return_value=test_mocks['mock_decomposer']),
            'print': patch('builtins.print')
        }
        return patches


class MockNLTK(AbstractLibraryWrapperMock):
    """Mock for NLTK module."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library_name = "nltk"
        self.library_version = "3.8.1"
        self.minimum_required_version = "3.7.0"
        self.maximum_supported_version = "4.0.0"
        
        # Set up NLTK-specific attributes
        self.download = Mock()
        self.data = Mock()
        self.corpus = Mock()
        self.tokenize = Mock()
        self.chunk = Mock()
        self.tag = Mock()
        self.parse = Mock()
        self.sem = Mock()
        self.metrics = Mock()
        
        # Set default configuration
        self.default_config = {
            'required_data': ['punkt', 'wordnet', 'stopwords'],
            'download_dir': '/mock/nltk_data'
        }
        self.required_config_keys = []
        
        # Set available features
        self.available_features = {
            'tokenization', 'pos_tagging', 'parsing', 'corpora', 'wordnet',
            'chunking', 'classification', 'sentiment', 'metrics'
        }
        
        # Initialize if no explicit config needed
        if not kwargs.get('defer_initialization', False):
            self.initialize_library(**self.default_config)
            self.is_initialized = True
            self.is_available = True
    
    def initialize_library(self, **config) -> bool:
        """Initialize the NLTK library mock."""
        # Simulate NLTK initialization
        try:
            # Check for required data downloads
            required_data = config.get('required_data', ['punkt', 'wordnet', 'stopwords'])
            for data_name in required_data:
                self.simulate_library_download(data_name)
            
            self.is_initialized = True
            return True
        except Exception as e:
            self.initialization_error = e
            return False
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get NLTK library information."""
        return {
            'name': self.library_name,
            'version': self.library_version,
            'type': 'natural_language_toolkit',
            'capabilities': ['tokenization', 'pos_tagging', 'parsing', 'corpora', 'wordnet'],
            'data_path': '/mock/nltk_data',
            'available_corpora': ['brown', 'reuters', 'gutenberg', 'wordnet'],
            'available_models': ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker']
        }
    
    def check_compatibility(self) -> bool:
        """Check NLTK version compatibility."""
        # Mock compatibility check
        return True


class MockSpacy(AbstractLibraryWrapperMock):
    """Mock for spaCy module."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library_name = "spacy"
        self.library_version = "3.4.4"
        self.minimum_required_version = "3.2.0"
        self.maximum_supported_version = "4.0.0"
        
        # Set up spaCy-specific attributes
        self.load = Mock()
        self.util = Mock()
        self.tokens = Mock()
        self.vocab = Mock()
        self.lang = Mock()
        self.pipeline = Mock()
        self.matcher = Mock()
        self.displacy = Mock()
        
        # Set default configuration
        self.default_config = {
            'model_name': 'en_core_web_sm'
        }
        self.required_config_keys = []
        
        # Set available features
        self.available_features = {
            'tokenization', 'pos_tagging', 'ner', 'dependency_parsing', 
            'lemmatization', 'sentence_segmentation', 'similarity'
        }
        
        # Initialize if no explicit config needed
        if not kwargs.get('defer_initialization', False):
            self.initialize_library(**self.default_config)
            self.is_initialized = True
            self.is_available = True
    
    def initialize_library(self, **config) -> bool:
        """Initialize the spaCy library mock."""
        try:
            # Simulate spaCy model loading
            model_name = config.get('model_name', 'en_core_web_sm')
            if model_name not in self.get_available_models():
                raise ValueError(f"Model '{model_name}' not found")
            
            # Load the model
            self.load_model(model_name)
            self.is_initialized = True
            return True
        except Exception as e:
            self.initialization_error = e
            return False
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get spaCy library information."""
        return {
            'name': self.library_name,
            'version': self.library_version,
            'type': 'nlp_pipeline',
            'capabilities': ['tokenization', 'pos_tagging', 'ner', 'dependency_parsing', 'lemmatization'],
            'available_models': self.get_available_models(),
            'loaded_models': list(self.models.keys()),
            'pipeline_components': ['tagger', 'parser', 'ner', 'lemmatizer']
        }
    
    def check_compatibility(self) -> bool:
        """Check spaCy version compatibility."""
        # Mock compatibility check
        return True
    
    def get_available_models(self) -> List[str]:
        """Get list of available spaCy models."""
        return ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]


class MockWordNet(AbstractLibraryWrapperMock):
    """Mock for WordNet corpus."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library_name = "wordnet"
        self.library_version = "3.0"
        self.minimum_required_version = "3.0"
        self.maximum_supported_version = "3.1"
        
        # WordNet POS constants
        self.NOUN = 'n'
        self.VERB = 'v'
        self.ADJ = 'a'
        self.ADV = 'r'
        
        # Set up WordNet-specific attributes
        self.synsets = Mock(return_value=[])
        self.lemmas = Mock(return_value=[])
        self.words = Mock(return_value=[])
        self.all_synsets = Mock(return_value=[])
        self.morphy = Mock()
        self.path_similarity = Mock(return_value=0.5)
        self.wup_similarity = Mock(return_value=0.8)
        self.lch_similarity = Mock(return_value=2.5)
        self.res_similarity = Mock(return_value=1.2)
        self.jcn_similarity = Mock(return_value=0.3)
        self.lin_similarity = Mock(return_value=0.6)
        
        # Set default configuration
        self.default_config = {
            'corpus_path': '/mock/wordnet'
        }
        self.required_config_keys = []
        
        # Set available features
        self.available_features = {
            'synsets', 'lemmas', 'similarity_measures', 'semantic_relations',
            'pos_tagging', 'morphological_analysis'
        }
        
        # Initialize if no explicit config needed
        if not kwargs.get('defer_initialization', False):
            self.initialize_library(**self.default_config)
            self.is_initialized = True
            self.is_available = True
    
    def initialize_library(self, **config) -> bool:
        """Initialize the WordNet library mock."""
        try:
            # Simulate WordNet corpus availability check
            corpus_path = config.get('corpus_path', '/mock/wordnet')
            if not corpus_path:
                raise ValueError("WordNet corpus path not specified")
            
            # Simulate loading corpus data
            self.simulate_library_download('wordnet_corpus')
            self.is_initialized = True
            return True
        except Exception as e:
            self.initialization_error = e
            return False
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get WordNet library information."""
        return {
            'name': self.library_name,
            'version': self.library_version,
            'type': 'lexical_database',
            'capabilities': ['synsets', 'lemmas', 'similarity_measures', 'semantic_relations'],
            'pos_tags': [self.NOUN, self.VERB, self.ADJ, self.ADV],
            'similarity_measures': ['path', 'wup', 'lch', 'res', 'jcn', 'lin'],
            'total_synsets': 117659,  # Mock number
            'total_lemmas': 206941   # Mock number
        }
    
    def check_compatibility(self) -> bool:
        """Check WordNet version compatibility."""
        # Mock compatibility check
        return True


class MockSynset(AbstractEntityMock):
    """Mock for WordNet Synset following abstract base class hierarchy."""
    
    def __init__(self, name="test.n.01", definition="test definition", *args, **kwargs):
        super().__init__(entity_type=EntityType.SYNSET, *args, **kwargs)
        
        # Core synset attributes
        self._synset_name = name
        self._synset_definition = definition
        self.name = Mock(return_value=name)
        self.definition = Mock(return_value=definition)
        self.pos = Mock(return_value="n")
        self.examples = Mock(return_value=["test example"])
        
        # Set entity properties
        self.set_property('name', name)
        self.set_property('definition', definition)
        self.set_property('pos', 'n')
        
        # Lemma relationships
        self.lemmas = Mock(return_value=[Mock(name=Mock(return_value="test"))])
        
        # Hierarchical relationships
        self.hypernyms = Mock(return_value=[])
        self.hyponyms = Mock(return_value=[])
        
        # Holonym/Meronym relationships
        self.member_holonyms = Mock(return_value=[])
        self.part_holonyms = Mock(return_value=[])
        self.substance_holonyms = Mock(return_value=[])
        self.member_meronyms = Mock(return_value=[])
        self.part_meronyms = Mock(return_value=[])
        self.substance_meronyms = Mock(return_value=[])
        
        # Other relationships
        self.similar_tos = Mock(return_value=[])
        self.also_sees = Mock(return_value=[])
        self.verb_groups = Mock(return_value=[])
        self.entailments = Mock(return_value=[])
        self.causes = Mock(return_value=[])
        
        # Similarity methods
        self.path_similarity = Mock(return_value=0.5)
        self.wup_similarity = Mock(return_value=0.8)
        self.lch_similarity = Mock(return_value=2.5)
        
        # Initialize as valid
        self.is_valid = True
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute (synset name)."""
        return self._synset_name
    
    def validate_entity(self) -> bool:
        """Validate synset consistency."""
        return (self.is_valid and 
                bool(self._synset_name) and 
                bool(self._synset_definition))
    
    def get_entity_signature(self) -> str:
        """Get unique synset signature."""
        return f"synset:{self.id}:{self._synset_name}"


class MockSynsetEdgeCases(AbstractEntityMock):
    """Mock for WordNet Synset edge case testing."""
    
    def __init__(self, edge_case_type="empty_synset", *args, **kwargs):
        super().__init__(entity_type=EntityType.SYNSET, *args, **kwargs)
        
        self.edge_case_type = edge_case_type
        
        if edge_case_type == "empty_synset":
            # Empty synset with no data
            self._synset_name = ""
            self._synset_definition = ""
            self.is_valid = False
        elif edge_case_type == "malformed_synset":
            # Malformed synset data
            self._synset_name = "invalid..synset"
            self._synset_definition = None
            self.is_valid = False
        elif edge_case_type == "missing_relationships":
            # Valid synset but no relationships
            self._synset_name = "isolated.n.01"
            self._synset_definition = "isolated concept"
            self.is_valid = True
        elif edge_case_type == "circular_relationships":
            # Synset with circular relationships
            self._synset_name = "circular.n.01"
            self._synset_definition = "circular reference"
            self.is_valid = True
        else:
            # Default valid synset
            self._synset_name = "edge.n.01"
            self._synset_definition = "edge case definition"
            self.is_valid = True
        
        # Set up basic attributes
        self.name = Mock(return_value=self._synset_name)
        self.definition = Mock(return_value=self._synset_definition)
        self.pos = Mock(return_value="n")
        
        # Configure edge case specific behaviors
        if edge_case_type == "missing_relationships":
            self.hypernyms = Mock(return_value=[])
            self.hyponyms = Mock(return_value=[])
            self.lemmas = Mock(return_value=[])
        elif edge_case_type == "circular_relationships":
            # Create circular reference (self-referential)
            self.hypernyms = Mock(return_value=[self])
            self.hyponyms = Mock(return_value=[self])
        else:
            self.hypernyms = Mock(return_value=[])
            self.hyponyms = Mock(return_value=[])
            self.lemmas = Mock(return_value=[])
        
        # Similarity methods that may fail
        if not self.is_valid:
            self.path_similarity = Mock(side_effect=ValueError("Invalid synset"))
            self.wup_similarity = Mock(side_effect=ValueError("Invalid synset"))
            self.lch_similarity = Mock(side_effect=ValueError("Invalid synset"))
        else:
            self.path_similarity = Mock(return_value=0.0)
            self.wup_similarity = Mock(return_value=0.0)
            self.lch_similarity = Mock(return_value=0.0)
        
        # Set entity properties
        self.set_property('name', self._synset_name)
        self.set_property('definition', self._synset_definition)
        self.set_property('edge_case_type', edge_case_type)
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute (synset name)."""
        return self._synset_name
    
    def validate_entity(self) -> bool:
        """Validate synset consistency."""
        if self.edge_case_type in ["empty_synset", "malformed_synset"]:
            return False
        return (bool(self._synset_name) and 
                self._synset_definition is not None)
    
    def get_entity_signature(self) -> str:
        """Get unique synset signature."""
        return f"synset_edge:{self.id}:{self._synset_name}:{self.edge_case_type}"


class MockSemanticDecomposer(Mock):
    """Mock for SemanticDecomposer class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wn_module = kwargs.get('wn_module', MockWordNet())
        self.nlp_func = kwargs.get('nlp_func', None)
        self.embedding_model = kwargs.get('embedding_model', None)
        
        # Set up methods
        self.build_synset_graph = Mock()
        self.find_connected_shortest_paths = Mock(return_value=(None, None, None))
        self.show_path = Mock()
        self.show_connected_paths = Mock()


class MockGraph(AbstractLibraryWrapperMock):
    """Mock for NetworkX graph objects."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library_name = "networkx"
        self.library_version = "3.1"
        self.minimum_required_version = "2.8.0"
        self.maximum_supported_version = "4.0.0"
        
        # Graph structure attributes
        self.number_of_nodes = Mock(return_value=100)
        self.number_of_edges = Mock(return_value=200)
        self.nodes = Mock(return_value=[])
        self.edges = Mock(return_value=[])
        
        # Graph manipulation methods
        self.add_node = Mock()
        self.add_edge = Mock()
        self.remove_node = Mock()
        self.remove_edge = Mock()
        self.has_node = Mock(return_value=True)
        self.has_edge = Mock(return_value=True)
        
        # Graph analysis methods
        self.degree = Mock(return_value={})
        self.neighbors = Mock(return_value=[])
        self.shortest_path = Mock(return_value=[])
        self.connected_components = Mock(return_value=[])
        self.subgraph = Mock()
        self.copy = Mock()
        self.clear = Mock()
        
        # Graph properties
        self.name = Mock(return_value="MockGraph")
        self.graph = Mock(return_value={})
        self.adj = Mock(return_value={})
        
        # Set up NetworkX-specific attributes
        self.algorithms = Mock()
        self.drawing = Mock()
        self.convert = Mock()
        
        # Set default configuration
        self.default_config = {
            'graph_type': 'Graph',
            'directed': False
        }
        self.required_config_keys = []
        
        # Set available features
        self.available_features = {
            'graph_creation', 'graph_analysis', 'algorithms', 'visualization',
            'shortest_path', 'centrality', 'clustering', 'connectivity'
        }
        
        # Initialize graph data structure
        self._graph_data = {}
        
        # Initialize if no explicit config needed
        if not kwargs.get('defer_initialization', False):
            self.initialize_library(**self.default_config)
            self.is_initialized = True
            self.is_available = True
    
    def initialize_library(self, **config) -> bool:
        """Initialize the NetworkX graph mock."""
        try:
            # Simulate graph initialization
            graph_type = config.get('graph_type', 'Graph')
            directed = config.get('directed', False)
            
            # Set graph properties based on configuration
            if directed:
                self.is_directed = Mock(return_value=True)
            else:
                self.is_directed = Mock(return_value=False)
            
            # Initialize empty graph structure
            self._graph_data = {}
            self.is_initialized = True
            return True
        except Exception as e:
            self.initialization_error = e
            return False
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get NetworkX library information."""
        return {
            'name': self.library_name,
            'version': self.library_version,
            'type': 'graph_library',
            'capabilities': ['graph_creation', 'graph_analysis', 'algorithms', 'visualization'],
            'graph_types': ['Graph', 'DiGraph', 'MultiGraph', 'MultiDiGraph'],
            'algorithms': ['shortest_path', 'centrality', 'clustering', 'connectivity'],
            'node_count': self.number_of_nodes(),
            'edge_count': self.number_of_edges()
        }
    
    def check_compatibility(self) -> bool:
        """Check NetworkX version compatibility."""
        # Mock compatibility check
        return True