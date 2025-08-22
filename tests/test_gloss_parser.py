import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smied.GlossParser import GlossParser


class TestGlossParser(unittest.TestCase):
    """Test the GlossParser class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_nlp = Mock()
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
        # Mock spaCy document
        mock_doc = Mock()
        mock_token1 = Mock()
        mock_token1.dep_ = "nsubj"
        mock_token1.pos_ = "NOUN"
        mock_token1.is_punct = False
        mock_token1.is_stop = False
        mock_token1.lemma_ = "cat"
        mock_token1.text = "cat"
        
        mock_token2 = Mock()
        mock_token2.dep_ = "ROOT"
        mock_token2.pos_ = "VERB"
        mock_token2.is_punct = False
        mock_token2.is_stop = False
        mock_token2.lemma_ = "run"
        mock_token2.text = "runs"
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2]))
        mock_doc.noun_chunks = []
        mock_doc.text = "The cat runs"
        
        self.mock_nlp.return_value = mock_doc
        
        # Mock WordNet synsets
        with patch('nltk.corpus.wordnet.synsets') as mock_synsets:
            mock_synsets.return_value = [Mock()]
            
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
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([]))
        mock_doc.noun_chunks = []
        mock_doc.text = "test"
        mock_nlp.return_value = mock_doc
        
        with patch('nltk.corpus.wordnet.synsets'):
            result = self.parser_no_nlp.parse_gloss("test text", nlp_func=mock_nlp)
            self.assertIsInstance(result, dict)

    def test_parse_gloss_exception_handling(self):
        """Test parse_gloss exception handling"""
        self.mock_nlp.side_effect = Exception("Test exception")
        result = self.parser.parse_gloss("test text")
        self.assertIsNone(result)

    def test_tokens_to_synsets(self):
        """Test token to synsets conversion"""
        mock_token1 = Mock()
        mock_token1.is_punct = False
        mock_token1.is_stop = False
        mock_token1.lemma_ = "cat"
        mock_token1.text = "cat"
        
        mock_token2 = Mock()
        mock_token2.is_punct = True
        mock_token2.is_stop = False
        
        mock_token3 = Mock()
        mock_token3.is_punct = False
        mock_token3.is_stop = True
        
        with patch('nltk.corpus.wordnet.synsets') as mock_synsets:
            mock_synset = Mock()
            mock_synsets.return_value = [mock_synset, Mock(), Mock()]
            
            result = self.parser._tokens_to_synsets([mock_token1, mock_token2, mock_token3], pos='n')
            
            self.assertIsInstance(result, list)
            # Should skip punctuation and stop words
            self.assertEqual(len(result), 3)  # max_synsets_per_token=3

    def test_tokens_to_synsets_fallback(self):
        """Test token to synsets with fallback to original text"""
        mock_token = Mock()
        mock_token.is_punct = False
        mock_token.is_stop = False
        mock_token.lemma_ = "unknown_lemma"
        mock_token.text = "cat"
        
        with patch('nltk.corpus.wordnet.synsets') as mock_synsets:
            mock_synset = Mock()
            # First call (with lemma) returns empty, second call (with text) returns synsets
            mock_synsets.side_effect = [[], [mock_synset]]
            
            result = self.parser._tokens_to_synsets([mock_token], pos='n')
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)

    def test_tokens_to_synsets_complete_failure(self):
        """Test token to synsets when both lemma and text fail"""
        mock_token = Mock()
        mock_token.is_punct = False
        mock_token.is_stop = False
        mock_token.lemma_ = "unknown"
        mock_token.text = "unknown"
        
        with patch('nltk.corpus.wordnet.synsets') as mock_synsets:
            mock_synsets.side_effect = Exception("Not found")
            
            result = self.parser._tokens_to_synsets([mock_token], pos='n')
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 0)

    def test_extract_subjects_from_gloss(self):
        """Test subject extraction from gloss"""
        mock_token1 = Mock()
        mock_token1.dep_ = "nsubj"
        
        mock_token2 = Mock()
        mock_token2.dep_ = "nsubjpass"
        
        mock_token3 = Mock()
        mock_token3.dep_ = "obj"
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2, mock_token3]))
        
        subjects, passive_subjects = self.parser.extract_subjects_from_gloss(mock_doc)
        
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0], mock_token1)
        self.assertEqual(len(passive_subjects), 1)
        self.assertEqual(passive_subjects[0], mock_token2)

    def test_extract_objects_from_gloss(self):
        """Test object extraction from gloss"""
        mock_iobj = Mock()
        mock_iobj.dep_ = "iobj"
        
        mock_dobj = Mock()
        mock_dobj.dep_ = "dobj"
        
        mock_pobj = Mock()
        mock_pobj.dep_ = "pobj"
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_iobj, mock_dobj, mock_pobj]))
        mock_doc.noun_chunks = []
        
        objects = self.parser.extract_objects_from_gloss(mock_doc)
        
        # Should include iobj and pobj, but not dobj when iobj is present
        self.assertEqual(len(objects), 2)
        self.assertIn(mock_iobj, objects)
        self.assertIn(mock_pobj, objects)
        self.assertNotIn(mock_dobj, objects)

    def test_extract_objects_no_indirect_objects(self):
        """Test object extraction when no indirect objects present"""
        mock_dobj = Mock()
        mock_dobj.dep_ = "dobj"
        
        mock_pobj = Mock()
        mock_pobj.dep_ = "pobj"
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_dobj, mock_pobj]))
        mock_doc.noun_chunks = []
        
        objects = self.parser.extract_objects_from_gloss(mock_doc)
        
        # Should include both dobj and pobj when no iobj
        self.assertEqual(len(objects), 2)
        self.assertIn(mock_dobj, objects)
        self.assertIn(mock_pobj, objects)

    def test_extract_objects_noun_chunks_fallback(self):
        """Test object extraction using noun chunks as fallback"""
        mock_root_verb = Mock()
        mock_root_verb.dep_ = "ROOT"
        mock_root_verb.pos_ = "VERB"
        
        mock_token = Mock()
        mock_token.head = mock_root_verb
        
        mock_chunk = Mock()
        mock_chunk.root = mock_token
        mock_chunk.__iter__ = Mock(return_value=iter([mock_token]))
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_root_verb]))
        mock_doc.noun_chunks = [mock_chunk]
        
        objects = self.parser.extract_objects_from_gloss(mock_doc)
        
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0], mock_token)

    def test_extract_verbs_from_gloss(self):
        """Test verb extraction from gloss"""
        mock_verb = Mock()
        mock_verb.pos_ = "VERB"
        
        mock_participle = Mock()
        mock_participle.pos_ = "VERB"
        mock_participle.tag_ = "VBN"
        mock_participle.dep_ = "acl"
        
        mock_noun = Mock()
        mock_noun.pos_ = "NOUN"
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_verb, mock_participle, mock_noun]))
        
        verbs = self.parser.extract_verbs_from_gloss(mock_doc, include_passive=True)
        
        self.assertEqual(len(verbs), 2)
        self.assertIn(mock_verb, verbs)
        self.assertIn(mock_participle, verbs)
        self.assertNotIn(mock_noun, verbs)

    def test_extract_verbs_exclude_passive(self):
        """Test verb extraction excluding passive forms"""
        mock_verb = Mock()
        mock_verb.pos_ = "VERB"
        
        mock_participle = Mock()
        mock_participle.pos_ = "VERB"
        mock_participle.tag_ = "VBN"
        mock_participle.dep_ = "acl"
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_verb, mock_participle]))
        
        verbs = self.parser.extract_verbs_from_gloss(mock_doc, include_passive=False)
        
        self.assertEqual(len(verbs), 1)
        self.assertIn(mock_verb, verbs)
        self.assertNotIn(mock_participle, verbs)

    def test_find_instrumental_verbs(self):
        """Test finding instrumental verbs"""
        mock_used = Mock()
        mock_used.text = "used"
        
        mock_for_token = Mock()
        mock_for_token.pos_ = "ADP"
        
        mock_verb = Mock()
        mock_verb.pos_ = "VERB"
        
        mock_doc = Mock()
        mock_doc.text = "This is used for running"
        mock_doc.__iter__ = Mock(return_value=iter([mock_used, mock_for_token, mock_verb]))
        mock_doc.__getitem__ = Mock(side_effect=[mock_used, mock_for_token, mock_verb])
        mock_doc.__len__ = Mock(return_value=3)
        
        instrumental_verbs = self.parser.find_instrumental_verbs(mock_doc)
        
        self.assertEqual(len(instrumental_verbs), 1)
        self.assertEqual(instrumental_verbs[0], mock_verb)

    def test_find_instrumental_verbs_no_used(self):
        """Test finding instrumental verbs when 'used' not present"""
        mock_doc = Mock()
        mock_doc.text = "This is for running"
        
        instrumental_verbs = self.parser.find_instrumental_verbs(mock_doc)
        
        self.assertEqual(len(instrumental_verbs), 0)

    def test_get_all_neighbors(self):
        """Test getting all neighbors of a synset"""
        mock_synset = Mock()
        mock_neighbor1 = Mock()
        mock_neighbor2 = Mock()
        
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
        mock_synset = Mock()
        mock_neighbor = Mock()
        
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
        mock_synset = Mock()
        mock_synset.name.return_value = "cat.n.01"
        
        path = self.parser.path_syn_to_syn(mock_synset, mock_synset)
        
        self.assertEqual(path, ["cat.n.01"])

    def test_path_syn_to_syn_different_pos(self):
        """Test pathfinding between synsets of different POS"""
        mock_synset1 = Mock()
        mock_synset1.name.return_value = "cat.n.01"
        mock_synset1.pos.return_value = 'n'
        
        mock_synset2 = Mock()
        mock_synset2.name.return_value = "run.v.01"
        mock_synset2.pos.return_value = 'v'
        
        path = self.parser.path_syn_to_syn(mock_synset1, mock_synset2)
        
        self.assertIsNone(path)

    def test_path_syn_to_syn_string_inputs(self):
        """Test pathfinding with string inputs instead of synset objects"""
        path = self.parser.path_syn_to_syn("cat.n.01", "dog.n.01")
        
        self.assertEqual(path, ["cat.n.01"])  # Same synset case

    @patch('smied.GlossParser.GlossParser.get_all_neighbors')
    def test_path_syn_to_syn_with_path(self, mock_get_neighbors):
        """Test pathfinding that finds a path through neighbors"""
        mock_start = Mock()
        mock_start.name.return_value = "cat.n.01"
        mock_start.pos.return_value = 'n'
        
        mock_end = Mock()
        mock_end.name.return_value = "dog.n.01"
        mock_end.pos.return_value = 'n'
        
        mock_intermediate = Mock()
        mock_intermediate.name.return_value = "animal.n.01"
        
        # Set up neighbor relationships
        mock_get_neighbors.side_effect = [
            [mock_intermediate],  # neighbors of cat
            [mock_end],           # neighbors of animal (from backward search)
            [],                   # neighbors of dog (from forward search)
            []                    # neighbors of animal (from forward search)
        ]
        
        path = self.parser.path_syn_to_syn(mock_start, mock_end, max_depth=4)
        
        # Should find a path through the intermediate synset
        self.assertIsNotNone(path)
        self.assertIsInstance(path, list)


class TestGlossParserIntegration(unittest.TestCase):
    """Integration tests for GlossParser with real NLP processing"""
    
    def setUp(self):
        """Set up with mock NLP for integration testing"""
        self.mock_nlp = Mock()
        self.parser = GlossParser(nlp_func=self.mock_nlp)

    def test_full_parsing_workflow(self):
        """Test complete parsing workflow"""
        # Create a realistic mock document
        mock_token_the = Mock()
        mock_token_the.dep_ = "det"
        mock_token_the.pos_ = "DET"
        mock_token_the.is_punct = False
        mock_token_the.is_stop = True
        
        mock_token_cat = Mock()
        mock_token_cat.dep_ = "nsubj"
        mock_token_cat.pos_ = "NOUN"
        mock_token_cat.is_punct = False
        mock_token_cat.is_stop = False
        mock_token_cat.lemma_ = "cat"
        mock_token_cat.text = "cat"
        
        mock_token_runs = Mock()
        mock_token_runs.dep_ = "ROOT"
        mock_token_runs.pos_ = "VERB"
        mock_token_runs.is_punct = False
        mock_token_runs.is_stop = False
        mock_token_runs.lemma_ = "run"
        mock_token_runs.text = "runs"
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_token_the, mock_token_cat, mock_token_runs]))
        mock_doc.noun_chunks = []
        mock_doc.text = "The cat runs"
        
        self.mock_nlp.return_value = mock_doc
        
        # Mock WordNet responses
        with patch('nltk.corpus.wordnet.synsets') as mock_synsets:
            mock_cat_synset = Mock()
            mock_run_synset = Mock()
            mock_synsets.side_effect = [[mock_cat_synset], [], [mock_run_synset]]
            
            result = self.parser.parse_gloss("The cat runs")
            
            self.assertIsNotNone(result)
            self.assertIn('subjects', result)
            self.assertIn('predicates', result)
            self.assertEqual(len(result['subjects']), 1)
            self.assertEqual(len(result['predicates']), 1)


if __name__ == '__main__':
    unittest.main()