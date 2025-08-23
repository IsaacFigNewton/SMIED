import unittest
from unittest.mock import patch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.GlossParser import GlossParser
from tests.mocks.gloss_parser_mocks import GlossParserMockFactory
from tests.config.gloss_parser_config import GlossParserMockConfig


class TestGlossParser(unittest.TestCase):
    """Test the GlossParser class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize mock factory and config
        self.mock_factory = GlossParserMockFactory()
        self.mock_config = GlossParserMockConfig()
        
        # Create NLP function using factory
        self.mock_nlp = self.mock_factory('MockNLPForGloss')
        self.parser = GlossParser(nlp_func=self.mock_nlp)
        self.parser_no_nlp = GlossParser()

    def test_initialization_with_nlp(self):
        """Test initialization with NLP function"""
        self.assertEqual(self.parser.nlp_func, self.mock_nlp)

    def test_initialization_without_nlp(self):
        """Test initialization without NLP function"""
        self.assertIsNone(self.parser_no_nlp.nlp_func)

    def test_parse_gloss_success(self):
        """Test successful gloss parsing"""
        # Create spaCy document using factory
        mock_doc = self.mock_factory('MockDocForGloss')
        
        # Get test gloss from config
        test_gloss = self.mock_config.get_test_gloss_texts()['simple_gloss']
        
        # Create tokens using factory
        mock_token1 = self.mock_factory('MockTokenForGloss', "cat")
        mock_token1.dep_ = "nsubj"
        mock_token1.pos_ = "NOUN"
        
        mock_token2 = self.mock_factory('MockTokenForGloss', "runs")
        mock_token2.dep_ = "ROOT"
        mock_token2.pos_ = "VERB"
        mock_token2.lemma_ = "run"
        
        from unittest.mock import Mock
        mock_doc.__iter__ = Mock(side_effect=lambda: iter([mock_token1, mock_token2]))
        mock_doc.noun_chunks = []
        mock_doc.text = "The cat runs"
        
        self.mock_nlp.return_value = mock_doc
        
        # Mock WordNet synsets using factory and config
        synset_data = self.mock_config.get_synset_mock_structures()['cat_synset']
        mock_synset = self.mock_factory('MockSynset', synset_data['name'])
        with patch('nltk.corpus.wordnet.synsets') as mock_synsets:
            mock_synsets.return_value = [mock_synset]
            
            result = self.parser.parse_gloss("The cat runs")
            
            self.assertIsInstance(result, dict)
            self.assertIn('subjects', result)
            self.assertIn('objects', result)
            self.assertIn('predicates', result)
            self.assertIn('raw_subjects', result)
            self.assertIn('raw_objects', result)
            self.assertIn('raw_predicates', result)

    def test_parse_gloss_no_nlp_function(self):
        """Test parse_gloss without NLP function"""
        result = self.parser_no_nlp.parse_gloss("test text")
        self.assertIsNone(result)

    def test_parse_gloss_with_provided_nlp(self):
        """Test parse_gloss with provided NLP function"""
        mock_nlp = self.mock_factory('MockNLPForGloss')
        mock_doc = self.mock_factory('MockDocForGloss')
        from unittest.mock import Mock
        mock_doc.__iter__ = Mock(side_effect=lambda: iter([]))
        mock_doc.noun_chunks = []
        mock_doc.text = "test"
        mock_nlp.return_value = mock_doc
        
        # Get test text from config
        edge_cases = self.mock_config.get_edge_case_glosses()
        test_text = edge_cases['single_word']
        
        with patch('nltk.corpus.wordnet.synsets'):
            result = self.parser_no_nlp.parse_gloss(test_text, nlp_func=mock_nlp)
            self.assertIsInstance(result, dict)

    def test_parse_gloss_exception_handling(self):
        """Test parse_gloss exception handling"""
        self.mock_nlp.side_effect = Exception("Test exception")
        result = self.parser.parse_gloss("test text")
        self.assertIsInstance(result, dict)

    def test_tokens_to_synsets(self):
        """Test token to synsets conversion"""
        # Get synset data from config
        synset_data = self.mock_config.get_synset_mock_structures()['cat_synset']
        
        # Create tokens using factory
        mock_token1 = self.mock_factory('MockTokenForGloss', "cat")
        mock_token1.is_punct = False
        mock_token1.is_stop = False
        mock_token1.lemma_ = "cat"
        
        mock_token2 = self.mock_factory('MockTokenForGloss', ".")
        mock_token2.is_punct = True
        mock_token2.is_stop = False
        
        mock_token3 = self.mock_factory('MockTokenForGloss', "the")
        mock_token3.is_punct = False
        mock_token3.is_stop = True
        
        with patch('nltk.corpus.wordnet.synsets') as mock_synsets:
            mock_synsets.return_value = [
                self.mock_factory('MockSynset', "cat.n.01"),
                self.mock_factory('MockSynset', "cat.n.02"),
                self.mock_factory('MockSynset', "cat.n.03")
            ]
            
            result = self.parser._tokens_to_synsets([mock_token1, mock_token2, mock_token3], pos='n')
            
            self.assertIsInstance(result, list)
            # Should skip punctuation and stop words
            self.assertEqual(len(result), 3)  # max_synsets_per_token=3

    def test_tokens_to_synsets_fallback(self):
        """Test token to synsets with fallback to original text"""
        # Get synset data from config
        synset_data = self.mock_config.get_synset_mock_structures()['cat_synset']
        
        mock_token = self.mock_factory('MockTokenForGloss', "cat")
        mock_token.is_punct = False
        mock_token.is_stop = False
        mock_token.lemma_ = "unknown_lemma"
        
        with patch('smied.GlossParser.wn.synsets') as mock_synsets:
            mock_synset = self.mock_factory('MockSynset', synset_data['name'])
            # First call (with lemma) raises exception, second call (with text) returns synsets
            mock_synsets.side_effect = [Exception("Not found"), [mock_synset]]
            
            result = self.parser._tokens_to_synsets([mock_token], pos='n')
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)

    def test_tokens_to_synsets_complete_failure(self):
        """Test token to synsets when both lemma and text fail"""
        mock_token = self.mock_factory('MockTokenForGloss', "unknown")
        mock_token.is_punct = False
        mock_token.is_stop = False
        mock_token.lemma_ = "unknown"
        
        with patch('nltk.corpus.wordnet.synsets') as mock_synsets:
            mock_synsets.side_effect = Exception("Not found")
            
            result = self.parser._tokens_to_synsets([mock_token], pos='n')
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 0)

    def test_extract_subjects_from_gloss(self):
        """Test subject extraction from gloss"""
        # Get dependency patterns from config
        patterns = self.mock_config.get_dependency_patterns()
        subject_patterns = patterns['subject_patterns']
        
        mock_token1 = self.mock_factory('MockTokenForGloss', "subject1")
        mock_token1.dep_ = subject_patterns[0]  # "nsubj"
        
        mock_token2 = self.mock_factory('MockTokenForGloss', "subject2")
        mock_token2.dep_ = subject_patterns[1]  # "nsubjpass"
        
        mock_token3 = self.mock_factory('MockTokenForGloss', "object")
        mock_token3.dep_ = "obj"
        
        mock_doc = self.mock_factory('MockDocForGloss')
        # Make __iter__ return a new iterator each time it's called
        from unittest.mock import Mock
        mock_doc.__iter__ = Mock(side_effect=lambda: iter([mock_token1, mock_token2, mock_token3]))
        
        subjects, passive_subjects = self.parser.extract_subjects_from_gloss(mock_doc)
        
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0], mock_token1)
        self.assertEqual(len(passive_subjects), 1)
        self.assertEqual(passive_subjects[0], mock_token2)

    def test_extract_objects_from_gloss(self):
        """Test object extraction from gloss"""
        # Get object patterns from config
        patterns = self.mock_config.get_dependency_patterns()
        object_patterns = patterns['object_patterns']
        
        mock_iobj = self.mock_factory('MockTokenForGloss', "indirect_object")
        mock_iobj.dep_ = object_patterns[1]  # "iobj"
        
        mock_dobj = self.mock_factory('MockTokenForGloss', "direct_object")
        mock_dobj.dep_ = object_patterns[0]  # "dobj"
        
        mock_pobj = self.mock_factory('MockTokenForGloss', "preposition_object")
        mock_pobj.dep_ = object_patterns[2]  # "pobj"
        
        mock_doc = self.mock_factory('MockDocForGloss')
        # Make __iter__ return a new iterator each time it's called
        from unittest.mock import Mock
        mock_doc.__iter__ = Mock(side_effect=lambda: iter([mock_iobj, mock_dobj, mock_pobj]))
        mock_doc.noun_chunks = []
        
        objects = self.parser.extract_objects_from_gloss(mock_doc)
        
        # Should include iobj and pobj, but not dobj when iobj is present
        self.assertEqual(len(objects), 2)
        self.assertIn(mock_iobj, objects)
        self.assertIn(mock_pobj, objects)
        self.assertNotIn(mock_dobj, objects)

    def test_extract_objects_no_indirect_objects(self):
        """Test object extraction when no indirect objects present"""
        # Get object patterns from config
        patterns = self.mock_config.get_dependency_patterns()
        object_patterns = patterns['object_patterns']
        
        mock_dobj = self.mock_factory('MockTokenForGloss', "direct_object")
        mock_dobj.dep_ = object_patterns[0]  # "dobj"
        
        mock_pobj = self.mock_factory('MockTokenForGloss', "preposition_object")
        mock_pobj.dep_ = object_patterns[2]  # "pobj"
        
        mock_doc = self.mock_factory('MockDocForGloss')
        # Make __iter__ return a new iterator each time it's called
        from unittest.mock import Mock
        mock_doc.__iter__ = Mock(side_effect=lambda: iter([mock_dobj, mock_pobj]))
        mock_doc.noun_chunks = []
        
        objects = self.parser.extract_objects_from_gloss(mock_doc)
        
        # Should include both dobj and pobj when no iobj
        self.assertEqual(len(objects), 2)
        self.assertIn(mock_dobj, objects)
        self.assertIn(mock_pobj, objects)

    def test_extract_objects_noun_chunks_fallback(self):
        """Test object extraction using noun chunks as fallback"""
        mock_root_verb = self.mock_factory('MockTokenForGloss', "root_verb")
        mock_root_verb.dep_ = "ROOT"
        mock_root_verb.pos_ = "VERB"
        
        mock_token = self.mock_factory('MockTokenForGloss', "token")
        mock_token.head = mock_root_verb
        
        mock_chunk = self.mock_factory('MockDocForGloss')
        mock_chunk.root = mock_token
        # Make __iter__ return a new iterator each time it's called
        from unittest.mock import Mock
        mock_chunk.__iter__ = Mock(side_effect=lambda: iter([mock_token]))
        
        mock_doc = self.mock_factory('MockDocForGloss')
        # Make __iter__ return a new iterator each time it's called
        from unittest.mock import Mock
        mock_doc.__iter__ = Mock(side_effect=lambda: iter([mock_root_verb]))
        mock_doc.noun_chunks = [mock_chunk]
        
        objects = self.parser.extract_objects_from_gloss(mock_doc)
        
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0], mock_token)

    def test_extract_verbs_from_gloss(self):
        """Test verb extraction from gloss"""
        # Get POS tag mappings from config
        pos_mappings = self.mock_config.get_pos_tag_mappings()
        verb_tags = pos_mappings['verbs']
        
        mock_verb = self.mock_factory('MockTokenForGloss', "verb")
        mock_verb.pos_ = "VERB"
        
        mock_participle = self.mock_factory('MockTokenForGloss', "participle")
        mock_participle.pos_ = "VERB"
        mock_participle.tag_ = verb_tags[3]  # "VBN"
        mock_participle.dep_ = "acl"
        
        mock_noun = self.mock_factory('MockTokenForGloss', "noun")
        mock_noun.pos_ = "NOUN"
        
        mock_doc = self.mock_factory('MockDocForGloss')
        # Make __iter__ return a new iterator each time it's called
        from unittest.mock import Mock
        mock_doc.__iter__ = Mock(side_effect=lambda: iter([mock_verb, mock_participle, mock_noun]))
        
        verbs = self.parser.extract_verbs_from_gloss(mock_doc, include_passive=True)
        
        self.assertEqual(len(verbs), 2)
        self.assertIn(mock_verb, verbs)
        self.assertIn(mock_participle, verbs)
        self.assertNotIn(mock_noun, verbs)

    def test_extract_verbs_exclude_passive(self):
        """Test verb extraction excluding passive forms"""
        # Get POS tag mappings from config
        pos_mappings = self.mock_config.get_pos_tag_mappings()
        verb_tags = pos_mappings['verbs']
        
        mock_verb = self.mock_factory('MockTokenForGloss', "verb")
        mock_verb.pos_ = "VERB"
        
        mock_participle = self.mock_factory('MockTokenForGloss', "participle")
        mock_participle.pos_ = "VERB"
        mock_participle.tag_ = verb_tags[3]  # "VBN"
        mock_participle.dep_ = "acl"
        
        mock_doc = self.mock_factory('MockDocForGloss')
        # Make __iter__ return a new iterator each time it's called
        from unittest.mock import Mock
        mock_doc.__iter__ = Mock(side_effect=lambda: iter([mock_verb, mock_participle]))
        
        verbs = self.parser.extract_verbs_from_gloss(mock_doc, include_passive=False)
        
        self.assertEqual(len(verbs), 1)
        self.assertIn(mock_verb, verbs)
        self.assertNotIn(mock_participle, verbs)

    def test_find_instrumental_verbs(self):
        """Test finding instrumental verbs"""
        mock_used = self.mock_factory('MockTokenForGloss', "used")
        mock_used.text = "used"
        
        mock_for_token = self.mock_factory('MockTokenForGloss', "for")
        mock_for_token.pos_ = "ADP"
        
        mock_verb = self.mock_factory('MockTokenForGloss', "verb")
        mock_verb.pos_ = "VERB"
        
        tokens = [mock_used, mock_for_token, mock_verb]
        
        mock_doc = self.mock_factory('MockDocForGloss')
        mock_doc.text = "This is used for running"
        from unittest.mock import Mock
        mock_doc.__iter__ = Mock(return_value=iter(tokens))
        mock_doc.__getitem__ = Mock(side_effect=lambda x: tokens[x])
        mock_doc.__len__ = Mock(return_value=3)
        
        instrumental_verbs = self.parser.find_instrumental_verbs(mock_doc)
        
        self.assertEqual(len(instrumental_verbs), 1)
        self.assertEqual(instrumental_verbs[0], mock_verb)

    def test_find_instrumental_verbs_no_used(self):
        """Test finding instrumental verbs when 'used' not present"""
        mock_doc = self.mock_factory('MockDocForGloss')
        mock_doc.text = "This is for running"
        
        instrumental_verbs = self.parser.find_instrumental_verbs(mock_doc)
        
        self.assertEqual(len(instrumental_verbs), 0)

    def test_get_all_neighbors(self):
        """Test getting all neighbors of a synset"""
        mock_synset = self.mock_factory('MockSynset', 'test.synset.01')
        mock_neighbor1 = self.mock_factory('MockSynset', 'neighbor1.synset.01')
        mock_neighbor2 = self.mock_factory('MockSynset', 'neighbor2.synset.01')
        
        mock_synset.hypernyms.return_value = [mock_neighbor1]
        mock_synset.hyponyms.return_value = [mock_neighbor2]
        mock_synset.holonyms.return_value = []
        mock_synset.meronyms.return_value = []
        mock_synset.similar_tos.return_value = []
        mock_synset.also_sees.return_value = []
        mock_synset.verb_groups.return_value = []
        mock_synset.entailments.return_value = []
        mock_synset.causes.return_value = []
        mock_synset.attributes.return_value = []
        
        neighbors = self.parser.get_all_neighbors(mock_synset)
        
        self.assertEqual(len(neighbors), 2)
        self.assertIn(mock_neighbor1, neighbors)
        self.assertIn(mock_neighbor2, neighbors)

    def test_get_all_neighbors_exception_handling(self):
        """Test get_all_neighbors with method exceptions"""
        mock_synset = self.mock_factory('MockSynset', 'test.synset.01')
        mock_neighbor = self.mock_factory('MockSynset', 'neighbor.synset.01')
        
        mock_synset.hypernyms.side_effect = Exception("Error")
        mock_synset.hyponyms.return_value = [mock_neighbor]
        mock_synset.holonyms.return_value = []
        mock_synset.meronyms.return_value = []
        mock_synset.similar_tos.return_value = []
        mock_synset.also_sees.return_value = []
        mock_synset.verb_groups.return_value = []
        mock_synset.entailments.return_value = []
        mock_synset.causes.return_value = []
        mock_synset.attributes.return_value = []
        
        neighbors = self.parser.get_all_neighbors(mock_synset)
        
        # Should handle exception and continue with other relations
        self.assertEqual(len(neighbors), 1)
        self.assertIn(mock_neighbor, neighbors)

    def test_path_syn_to_syn_same_synset(self):
        """Test pathfinding between identical synsets"""
        # Get synset data from config
        synset_data = self.mock_config.get_synset_mock_structures()['cat_synset']
        synset_name = synset_data['name']
        
        mock_synset = self.mock_factory('MockSynset', synset_name)
        
        path = self.parser.path_syn_to_syn(mock_synset, mock_synset)
        
        self.assertEqual(path, [synset_name])

    def test_path_syn_to_syn_different_pos(self):
        """Test pathfinding between synsets of different POS"""
        # Get synset data from config
        synset_data = self.mock_config.get_synset_mock_structures()
        cat_data = synset_data['cat_synset']
        run_data = synset_data['run_synset']
        
        mock_synset1 = self.mock_factory('MockSynset', cat_data['name'])
        mock_synset1.pos.return_value = cat_data['pos']
        
        mock_synset2 = self.mock_factory('MockSynset', run_data['name'])
        mock_synset2.pos.return_value = run_data['pos']
        
        path = self.parser.path_syn_to_syn(mock_synset1, mock_synset2)
        
        self.assertIsNone(path)

    def test_path_syn_to_syn_string_inputs(self):
        """Test pathfinding with string inputs instead of synset objects"""
        # Get synset data from config
        synset_data = self.mock_config.get_synset_mock_structures()
        cat_name = synset_data['cat_synset']['name']
        dog_name = synset_data['dog_synset']['name']
        
        # Test same synset case - should return path with just that synset
        path = self.parser.path_syn_to_syn(cat_name, cat_name)
        self.assertEqual(path, [cat_name])
        
        # Test different synsets - should return None because strings can't find neighbors
        path = self.parser.path_syn_to_syn(cat_name, dog_name)
        self.assertIsNone(path)

    @patch('smied.GlossParser.GlossParser.get_all_neighbors')
    def test_path_syn_to_syn_with_path(self, mock_get_neighbors):
        """Test pathfinding that finds a path through neighbors"""
        # Get synset data from config
        synset_data = self.mock_config.get_synset_mock_structures()
        cat_data = synset_data['cat_synset']
        dog_data = synset_data['dog_synset']
        
        mock_start = self.mock_factory('MockSynset', cat_data['name'])
        mock_start.pos.return_value = cat_data['pos']
        
        mock_end = self.mock_factory('MockSynset', dog_data['name'])
        mock_end.pos.return_value = dog_data['pos']
        
        mock_intermediate = self.mock_factory('MockSynset', 'animal.n.01')
        
        # Set up neighbor relationships based on input synset
        def get_neighbors_side_effect(synset, wn_module=None):
            synset_name = synset.name() if hasattr(synset, 'name') else str(synset)
            if synset_name == cat_data['name']:
                return [mock_intermediate]
            elif synset_name == "animal.n.01":
                return [mock_end]  # animal connects to dog
            elif synset_name == dog_data['name']:
                return []
            else:
                return []
        
        mock_get_neighbors.side_effect = get_neighbors_side_effect
        
        path = self.parser.path_syn_to_syn(mock_start, mock_end, max_depth=4)
        
        # Should find a path through the intermediate synset
        self.assertIsNotNone(path)
        self.assertIsInstance(path, list)


class TestGlossParserIntegration(unittest.TestCase):
    """Integration tests for GlossParser with real NLP processing"""
    
    def setUp(self):
        """Set up with mock NLP for integration testing"""
        self.mock_factory = GlossParserMockFactory()
        self.mock_config = GlossParserMockConfig()
        self.integration_mock = self.mock_factory('MockGlossParserIntegration')
        
        # Use integration mock NLP
        self.mock_nlp = self.integration_mock.create_integration_nlp_mock()
        self.parser = GlossParser(nlp_func=self.mock_nlp)

    def test_full_parsing_workflow(self):
        """Test complete parsing workflow"""
        # Setup integration parsing scenario using mock factory
        scenario = self.integration_mock.setup_integration_parsing_scenario('simple_sentence')
        
        # Configure parser with scenario NLP
        self.parser = GlossParser(nlp_func=scenario['nlp'])
        
        # Mock WordNet responses using scenario synsets
        with patch('smied.GlossParser.wn.synsets') as mock_synsets:
            # Setup synset returns for each lemma
            synset_returns = []
            for synset in scenario['synsets']:
                synset_returns.append([synset])
            mock_synsets.side_effect = synset_returns
            
            result = self.parser.parse_gloss(scenario['text'])
            
            self.assertIsNotNone(result)
            self.assertIn('subjects', result)
            self.assertIn('predicates', result)
            self.assertEqual(len(result['subjects']), 1)
            self.assertEqual(len(result['predicates']), 1)


if __name__ == '__main__':
    unittest.main()