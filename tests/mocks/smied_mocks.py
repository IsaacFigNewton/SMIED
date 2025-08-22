"""
Mock classes for SMIED tests.
"""

from unittest.mock import Mock
from typing import Optional, List, Any, Dict


class SMIEDMockFactory:
    """Factory class for creating SMIED mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockISMIEDPipeline': MockISMIEDPipeline,
            'MockSMIED': MockSMIED,
            'MockSMIEDIntegration': MockSMIEDIntegration,
            'MockNLTK': MockNLTK,
            'MockSpacy': MockSpacy,
            'MockWordNet': MockWordNet,
            'MockSynset': MockSynset,
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


class MockSMIEDIntegration(Mock):
    """Mock for SMIED integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mock components for integration testing
        self.mock_nltk = Mock()
        self.mock_spacy = Mock()
        self.mock_wn = Mock()
        self.mock_semantic_decomposer = Mock()
        
        # Set up NLTK mock
        self.mock_nltk.download = Mock()
        
        # Set up spacy mock
        self.mock_spacy.load = Mock()
        
        # Set up WordNet mock
        self.mock_wn.synsets = Mock(return_value=[])
        self.mock_wn.NOUN = 'n'
        self.mock_wn.VERB = 'v'
        
        # Set up SemanticDecomposer mock
        self.mock_semantic_decomposer.build_synset_graph = Mock()
        self.mock_semantic_decomposer.find_connected_shortest_paths = Mock()


class MockNLTK(Mock):
    """Mock for NLTK module."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download = Mock()


class MockSpacy(Mock):
    """Mock for spaCy module."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load = Mock()


class MockWordNet(Mock):
    """Mock for WordNet corpus."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synsets = Mock(return_value=[])
        self.NOUN = 'n'
        self.VERB = 'v'
        self.ADJ = 'a'
        self.ADV = 'r'


class MockSynset(Mock):
    """Mock for WordNet Synset."""
    
    def __init__(self, name="test.n.01", definition="test definition", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = Mock(return_value=name)
        self.definition = Mock(return_value=definition)
        self.pos = Mock(return_value="n")
        self.examples = Mock(return_value=["test example"])
        self.lemmas = Mock(return_value=[Mock(name=Mock(return_value="test"))])
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
        self.path_similarity = Mock(return_value=0.5)
        self.wup_similarity = Mock(return_value=0.8)
        self.lch_similarity = Mock(return_value=2.5)


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


class MockGraph(Mock):
    """Mock for NetworkX graph objects."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_nodes = Mock(return_value=100)
        self.number_of_edges = Mock(return_value=200)
        self.nodes = Mock(return_value=[])
        self.edges = Mock(return_value=[])
        self.add_node = Mock()
        self.add_edge = Mock()
        self.has_node = Mock(return_value=True)
        self.has_edge = Mock(return_value=True)