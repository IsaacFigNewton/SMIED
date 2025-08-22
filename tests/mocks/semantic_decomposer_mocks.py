"""
Mock classes for SemanticDecomposer tests.
"""

from unittest.mock import Mock
import networkx as nx
from typing import List, Optional, Any, Dict, Tuple


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


class MockWordNetForDecomposer(Mock):
    """Mock WordNet module for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synsets = Mock(side_effect=self._mock_synsets)
        self.synset = Mock(side_effect=self._mock_synset)
        self.all_synsets = Mock(return_value=[self._create_mock_synset("test.n.01")])
        
    def _mock_synsets(self, word, pos=None):
        """Mock synsets method behavior."""
        if word == "cat":
            return [self._create_mock_synset("cat.n.01")]
        elif word == "run":
            return [self._create_mock_synset("run.v.01")]
        elif word == "park":
            return [self._create_mock_synset("park.n.01")]
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
        synset.lemmas = Mock(return_value=[Mock(name=Mock(return_value=name.split('.')[0]))])
        
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
        
        # Similarity methods
        synset.path_similarity = Mock(return_value=0.5)
        synset.wup_similarity = Mock(return_value=0.8)
        synset.lch_similarity = Mock(return_value=2.5)
        
        return synset


class MockNLPForDecomposer(Mock):
    """Mock NLP function for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_value = MockDocForDecomposer()


class MockDocForDecomposer(Mock):
    """Mock document for SemanticDecomposer NLP processing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = "test document"
        self.tokens = [MockTokenForDecomposer("test")]
        self.ents = []
        self.noun_chunks = []


class MockTokenForDecomposer(Mock):
    """Mock token for SemanticDecomposer NLP processing."""
    
    def __init__(self, text="test", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        self.lemma_ = text.lower()
        self.pos_ = "NOUN"
        self.dep_ = "ROOT"


class MockEmbeddingModelForDecomposer(Mock):
    """Mock embedding model for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_vector = Mock(return_value=[0.1, 0.2, 0.3])
        self.most_similar = Mock(return_value=[("similar", 0.9)])
        self.similarity = Mock(return_value=0.8)


class MockEmbeddingHelperForDecomposer(Mock):
    """Mock EmbeddingHelper for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_similarity = Mock(return_value=0.8)
        self.get_most_similar = Mock(return_value=[])
        self.has_embedding = Mock(return_value=True)


class MockBeamBuilderForDecomposer(Mock):
    """Mock BeamBuilder for SemanticDecomposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_beam = Mock(return_value=[])
        self.expand_candidates = Mock(return_value=[])


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


class MockPairwiseBidirectionalAStar(Mock):
    """Mock for PairwiseBidirectionalAStar pathfinding."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.find_paths = Mock(return_value=[(["start", "end"], 1.0)])
        self.find_shortest_path = Mock(return_value=["start", "end"])


class MockPatternMatcher(Mock):
    """Mock PatternMatcher for SemanticDecomposer tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.match = Mock(return_value=[])
        self.find_patterns = Mock(return_value=[])