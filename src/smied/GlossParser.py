from collections import deque
from typing import List, Optional

import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn


class GlossParser:
    """
    Handles parsing and extraction of semantic elements from WordNet glosses.
    
    Provides methods to extract subjects, objects, and verbs from parsed gloss documents
    using dependency parsing information.
    """
    
    def __init__(self, nlp_func=None):
        """
        Initialize GlossParser with optional NLP function.
        
        Args:
            nlp_func: spaCy NLP function for text processing. If None, parsing methods
                     will require gloss_doc to be passed as pre-processed spaCy documents.
        """
        self.nlp_func = nlp_func

    def parse_gloss(self, gloss_text: str, nlp_func=None) -> Optional[dict]:
        """
        Parse a gloss text and return a dictionary with extracted semantic elements.
        
        Args:
            gloss_text: The gloss definition text to parse
            nlp_func: Optional spaCy NLP function. Uses instance nlp_func if not provided.
            
        Returns:
            Dictionary with keys 'subjects', 'objects', 'predicates' containing lists of synsets,
            or None if parsing fails.
        """
        # Use provided nlp_func or instance nlp_func
        nlp = nlp_func or self.nlp_func
        if nlp is None:
            # Cannot parse without NLP function
            return None
            
        try:
            # Parse the gloss text
            gloss_doc = nlp(gloss_text)
            
            # Extract semantic elements using existing methods
            subjects, passive_subjects = self.extract_subjects_from_gloss(gloss_doc)
            objects = self.extract_objects_from_gloss(gloss_doc)
            predicates = self.extract_verbs_from_gloss(gloss_doc, include_passive=True)
            
            # Add instrumental verbs to predicates
            instrumental_verbs = self.find_instrumental_verbs(gloss_doc)
            predicates.extend(instrumental_verbs)
            
            # Convert tokens to synsets using WordNet
            subject_synsets = self._tokens_to_synsets(subjects, pos='n')
            object_synsets = self._tokens_to_synsets(objects + passive_subjects, pos='n')
            predicate_synsets = self._tokens_to_synsets(predicates, pos='v')
            
            return {
                'subjects': subject_synsets,
                'objects': object_synsets, 
                'predicates': predicate_synsets,
                # Keep raw tokens for debugging/analysis
                'raw_subjects': subjects,
                'raw_objects': objects,
                'raw_predicates': predicates,
                'raw_passive_subjects': passive_subjects,
                'raw_instrumental_verbs': instrumental_verbs
            }
            
        except Exception as e:
            # Return None on any parsing error
            return None
    
    def _tokens_to_synsets(self, tokens, pos='n', max_synsets_per_token=3):
        """
        Convert spaCy tokens to WordNet synsets.
        
        Args:
            tokens: List of spaCy tokens
            pos: Part of speech ('n' for noun, 'v' for verb, 'a' for adjective, 'r' for adverb)
            max_synsets_per_token: Maximum number of synsets to return per token
            
        Returns:
            List of synsets
        """
        synsets = []
        
        for token in tokens:
            # Skip punctuation and stop words
            if token.is_punct or token.is_stop:
                continue
                
            # Get lemmatized form
            lemma = token.lemma_.lower()
            
            # Get synsets for the token
            try:
                token_synsets = wn.synsets(lemma, pos=pos)
                # Limit to max_synsets_per_token to avoid explosion
                synsets.extend(token_synsets[:max_synsets_per_token])
            except:
                # Try with the original text if lemma fails
                try:
                    token_synsets = wn.synsets(token.text.lower(), pos=pos)
                    synsets.extend(token_synsets[:max_synsets_per_token])
                except:
                    # Skip tokens that can't be converted to synsets
                    continue
                    
        return synsets

    def extract_subjects_from_gloss(self, gloss_doc):
        """Extract subject tokens from a parsed gloss."""
        subjects = []

        # Direct subjects
        subjects.extend([tok for tok in gloss_doc if tok.dep_ == "nsubj"])

        # Passive subjects (which are actually objects semantically)
        # Skip these for actor identification
        passive_subjects = [tok for tok in gloss_doc if tok.dep_ == "nsubjpass"]

        # Filter out passive subjects from the main list
        subjects = [s for s in subjects if s not in passive_subjects]

        return subjects, passive_subjects

    def extract_objects_from_gloss(self, gloss_doc):
        """Extract various types of object tokens from a parsed gloss."""
        objs = []

        # Indirect objects
        iobjs = [tok for tok in gloss_doc if tok.dep_ == "iobj"]
        objs.extend(iobjs)

        # Direct objects
        # Only include if there were no indirect objects,
        #   crude, but good for MVP
        if not iobjs:
            objs.extend([tok for tok in gloss_doc if tok.dep_ == "dobj"])

        # Prepositional objects
        objs.extend([tok for tok in gloss_doc if tok.dep_ == "pobj"])

        # General objects
        objs.extend([tok for tok in gloss_doc if tok.dep_ == "obj"])

        # Check for noun chunks related to root verb
        root_verbs = [tok for tok in gloss_doc if tok.dep_ == "ROOT" and tok.pos_ == "VERB"]
        if root_verbs and not objs:
            for noun_chunk in gloss_doc.noun_chunks:
                if any(token.head == root_verbs[0] for token in noun_chunk):
                    objs.append(noun_chunk.root)

        return objs

    def extract_verbs_from_gloss(self, gloss_doc, include_passive=False):
        """Extract verb tokens from a parsed gloss."""
        verbs = [tok for tok in gloss_doc if tok.pos_ == "VERB"]

        if include_passive:
            # Past participles used as adjectives or in relative clauses
            passive_verbs = [tok for tok in gloss_doc if
                            tok.tag_ in ["VBN", "VBD"] and
                            tok.dep_ in ["acl", "relcl", "amod"]]
            verbs.extend(passive_verbs)

        return verbs

    def find_instrumental_verbs(self, gloss_doc):
        """Find verbs associated with instrumental use (e.g., 'used for')."""
        instrumental_verbs = []

        if "used" in gloss_doc.text.lower():
            for i, token in enumerate(gloss_doc):
                if token.text.lower() == "used":
                    # Check tokens after "used"
                    for j in range(i+1, min(i+4, len(gloss_doc))):
                        if gloss_doc[j].pos_ == "VERB":
                            instrumental_verbs.append(gloss_doc[j])

        return instrumental_verbs

    def get_all_neighbors(self, synset, wn_module=None):
        """Get all lexically related neighbors of a synset."""
        neighbors = set()

        # Add all types of relations
        relation_methods = [
            'hypernyms', 'hyponyms', 'holonyms', 'meronyms',
            'similar_tos', 'also_sees', 'verb_groups',
            'entailments', 'causes', 'attributes'
        ]

        for method_name in relation_methods:
            if hasattr(synset, method_name):
                try:
                    related = getattr(synset, method_name)()
                    neighbors.update(related)
                except:
                    pass

        return list(neighbors)

    def path_syn_to_syn(self, start_synset, end_synset, max_depth=6, wn_module=None):
        """
        Find shortest path between synsets of the same POS using bidirectional BFS.
        Returns a list of synset names (strings) forming the path, or None if no path found.
        """
        # Convert to names for consistency
        start_name = start_synset.name() if hasattr(start_synset, 'name') else str(start_synset)
        end_name = end_synset.name() if hasattr(end_synset, 'name') else str(end_synset)

        # Check if same POS (if we have synset objects)
        if hasattr(start_synset, 'pos') and hasattr(end_synset, 'pos'):
            if start_synset.pos() != end_synset.pos():
                return None

        # Handle the trivial case where start and end are the same
        if start_name == end_name:
            return [start_name]

        # Initialize two search frontiers
        forward_queue = deque([(start_synset, 0)])
        forward_visited = {start_name: [start_name]}

        backward_queue = deque([(end_synset, 0)])
        backward_visited = {end_name: [end_name]}

        def expand_frontier(queue, visited_from_this_side, visited_from_other_side, is_forward):
            """Expand one step of the search frontier."""
            if not queue:
                return None

            curr_synset, depth = queue.popleft()

            if depth >= (max_depth + 1) // 2:
                return None

            curr_name = curr_synset.name() if hasattr(curr_synset, 'name') else str(curr_synset)
            path_to_current = visited_from_this_side[curr_name]

            for neighbor in self.get_all_neighbors(curr_synset, wn_module):
                neighbor_name = neighbor.name() if hasattr(neighbor, 'name') else str(neighbor)

                if neighbor_name in visited_from_this_side:
                    continue

                if is_forward:
                    new_path = path_to_current + [neighbor_name]
                else:
                    new_path = [neighbor_name] + path_to_current

                if neighbor_name in visited_from_other_side:
                    other_path = visited_from_other_side[neighbor_name]

                    if is_forward:
                        full_path = path_to_current + other_path
                    else:
                        full_path = other_path + path_to_current

                    return full_path

                visited_from_this_side[neighbor_name] = new_path
                queue.append((neighbor, depth + 1))

            return None

        # Alternate between forward and backward search
        while forward_queue or backward_queue:
            if forward_queue:
                result = expand_frontier(forward_queue, forward_visited, backward_visited, True)
                if result:
                    return result

            if backward_queue:
                result = expand_frontier(backward_queue, backward_visited, forward_visited, False)
                if result:
                    return result

        return None