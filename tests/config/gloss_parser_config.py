"""
Configuration class containing mock constants and test data for GlossParser tests.
"""


class GlossParserMockConfig:
    """Configuration class containing mock constants and test data for GlossParser tests."""
    
    @staticmethod
    def get_test_gloss_texts():
        """Get test gloss texts for parsing."""
        return {
            'simple_gloss': "a domesticated carnivorous mammal",
            'complex_gloss': "domesticated carnivorous mammal that typically has a barbed tongue",
            'compound_gloss': "a large domesticated mammal used for riding and racing",
            'scientific_gloss': "any of various carnivorous mammals of the family Felidae",
            'descriptive_gloss': "small domesticated carnivorous mammal with soft fur",
        }
    
    @staticmethod
    def get_expected_parsing_results():
        """Get expected parsing results for test glosses."""
        return {
            'simple_gloss': {
                'tokens': ['a', 'domesticated', 'carnivorous', 'mammal'],
                'pos_tags': ['DT', 'VBN', 'JJ', 'NN'],
                'dependencies': ['det', 'amod', 'amod', 'ROOT']
            },
            'complex_gloss': {
                'tokens': ['domesticated', 'carnivorous', 'mammal', 'that', 'typically', 'has', 'a', 'barbed', 'tongue'],
                'pos_tags': ['VBN', 'JJ', 'NN', 'WDT', 'RB', 'VBZ', 'DT', 'JJ', 'NN'],
                'dependencies': ['amod', 'amod', 'ROOT', 'nsubj', 'advmod', 'relcl', 'det', 'amod', 'dobj']
            }
        }
    
    @staticmethod
    def get_mock_token_structures():
        """Get mock token structures with consistent attributes."""
        return [
            {
                'text': 'domesticated',
                'pos_': 'VBN',
                'dep_': 'amod',
                'lemma_': 'domesticate',
                'is_alpha': True,
                'is_stop': False,
                'head_text': 'mammal'
            },
            {
                'text': 'carnivorous',
                'pos_': 'JJ',
                'dep_': 'amod',
                'lemma_': 'carnivorous',
                'is_alpha': True,
                'is_stop': False,
                'head_text': 'mammal'
            },
            {
                'text': 'mammal',
                'pos_': 'NN',
                'dep_': 'ROOT',
                'lemma_': 'mammal',
                'is_alpha': True,
                'is_stop': False,
                'head_text': 'mammal'
            }
        ]
    
    @staticmethod
    def get_dependency_patterns():
        """Get common dependency parsing patterns."""
        return {
            'subject_patterns': ['nsubj', 'nsubjpass', 'csubj'],
            'object_patterns': ['dobj', 'iobj', 'pobj'],
            'modifier_patterns': ['amod', 'advmod', 'nmod'],
            'determiner_patterns': ['det', 'poss'],
            'compound_patterns': ['compound', 'nn']
        }
    
    @staticmethod
    def get_synset_mock_structures():
        """Get WordNet synset mock structures."""
        return {
            'cat_synset': {
                'name': 'cat.n.01',
                'definition': 'feline mammal usually having thick soft fur',
                'pos': 'n',
                'examples': ['cats are often kept as pets'],
                'lemma_names': ['cat', 'true_cat']
            },
            'dog_synset': {
                'name': 'dog.n.01', 
                'definition': 'a member of the genus Canis',
                'pos': 'n',
                'examples': ['the dog barked all night'],
                'lemma_names': ['dog', 'domestic_dog', 'Canis_familiaris']
            },
            'run_synset': {
                'name': 'run.v.01',
                'definition': 'move fast by using legs',
                'pos': 'v',
                'examples': ['The children ran to the store'],
                'lemma_names': ['run']
            }
        }
    
    @staticmethod
    def get_realistic_parsing_scenarios():
        """Get realistic parsing scenarios for integration testing."""
        return {
            'scenario_1': {
                'gloss': 'large aquatic mammal with a horizontally flattened tail',
                'expected_key_terms': ['large', 'aquatic', 'mammal', 'tail'],
                'expected_relations': [('mammal', 'large'), ('mammal', 'aquatic'), ('mammal', 'tail')]
            },
            'scenario_2': {
                'gloss': 'small insectivorous mammal resembling a mouse but with a long snout',
                'expected_key_terms': ['small', 'insectivorous', 'mammal', 'mouse', 'snout'],
                'expected_relations': [('mammal', 'small'), ('mammal', 'insectivorous'), ('mammal', 'resembling')]
            },
            'scenario_3': {
                'gloss': 'tropical American bird having a very large bill surmounted by a crest',
                'expected_key_terms': ['tropical', 'American', 'bird', 'bill', 'crest'],
                'expected_relations': [('bird', 'tropical'), ('bird', 'American'), ('bird', 'bill')]
            }
        }
    
    @staticmethod
    def get_edge_case_glosses():
        """Get edge case glosses for testing error handling."""
        return {
            'empty_gloss': '',
            'single_word': 'mammal',
            'punctuation_heavy': 'a small, fast-moving, tree-dwelling mammal.',
            'parenthetical': 'mammal (especially cats and dogs)',
            'technical_terms': 'carnivorous mammal of the order Carnivora',
            'with_abbreviations': 'mammal e.g. cats, dogs, etc.'
        }
    
    @staticmethod
    def get_pos_tag_mappings():
        """Get POS tag mappings for validation."""
        return {
            'nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
            'verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'adjectives': ['JJ', 'JJR', 'JJS'],
            'adverbs': ['RB', 'RBR', 'RBS'],
            'determiners': ['DT', 'WDT'],
            'prepositions': ['IN', 'TO']
        }