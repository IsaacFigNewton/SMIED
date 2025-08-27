"""
FrameNet-SpaCy Semantic Role Labeling Pipeline
A complete span-labeling based SRL system integrating NLTK's FrameNet with SpaCy
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import re
import numpy as np
import statistics

import spacy
from spacy.tokens import Doc, Span, Token
from spacy.language import Language

import nltk
nltk.download('framenet_v17')
nltk.download('wordnet')
from nltk.corpus import framenet as fn
from nltk.corpus import wordnet as wn


@dataclass
class FrameElement:
    """Represents a frame element with its span and metadata"""
    name: str
    span: Span
    frame_name: str
    confidence: float
    fe_type: str  # Core, Non-Core, Extra-Thematic
    definition: str = ""


@dataclass
class FrameInstance:
    """Represents an evoked frame with its target and elements"""
    name: str
    target: Span  # The frame-evoking element (predicate)
    elements: List[FrameElement]
    confidence: float
    definition: str = ""
    lexical_unit: str = ""


# Register the factory before the class definition
@Language.factory(
    "framenet_srl",
    default_config={
        "min_confidence": 0.5,
        "use_wordnet_expansion": True
    }
)
def create_framenet_srl_component(nlp, name, min_confidence, use_wordnet_expansion):
    """Factory to create FrameNetSpaCySRL component"""
    return FrameNetSpaCySRL(
        nlp=nlp,
        min_confidence=min_confidence,
        use_wordnet_expansion=use_wordnet_expansion
    )

@spacy.language.Language.component("wn_frame_schema_tagger")
def wn_frame_schema_tagger(doc):
    mapping = {"something": "NN", "someone": "NN"}  # Use suitable tag string
    for token in doc:
        # Save original tag
        token._.orig_tag = token.tag_
        # Retag if verb lemma mistagged as noun
        if token.text.lower() not in mapping and token.pos_ == "NOUN":
            token.tag_ = "VB"
    return doc

def sigmoid(x):
    """
    Implements the sigmoid function using NumPy.

    Parameters:
    x (np.ndarray or scalar): The input value(s) to apply the sigmoid function to.

    Returns:
    np.ndarray or scalar: The result of applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

class FrameNetSpaCySRL:
    """
    Complete FrameNet-based Semantic Role Labeling pipeline for SpaCy.

    This class identifies frames evoked by predicates in text and maps
    frame elements to syntactic spans (noun chunks, prepositional phrases, etc.)

    Usage:
        # Method 1: Use as a SpaCy pipeline component
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("framenet_srl", config={"min_confidence": 0.5})
        doc = nlp("John gave Mary a book.")

        # Method 2: Use standalone
        srl = FrameNetSpaCySRL()
        doc = srl.process("John gave Mary a book.")
    """

    # add class variables for dependency role mapping
    vect_order = ["subject", "object"]  #, "theme"]
    dep_map = {
        "subject": {"nsubj", "nsubjpass"},
        "object": {"dobj", "obj", "pobj"},
        "theme": {"iobj"}
    }

    def __init__(self,
                 nlp: Optional[Language] = None,
                 spacy_model: str = "en_core_web_sm",
                 min_confidence: float = 0.5,
                 use_wordnet_expansion: bool = True):
        """
        Initialize the FrameNet-SpaCy SRL pipeline.

        Args:
            nlp: Optional pre-loaded SpaCy Language object
            spacy_model: Name of SpaCy model to load (ignored if nlp is provided)
            min_confidence: Minimum confidence threshold for frame/element assignment
            use_wordnet_expansion: Whether to use WordNet for frame expansion
        """
        # Load or use provided SpaCy model
        if nlp is not None:
            self.nlp = nlp
        else:
            self.nlp = spacy.load(spacy_model)
        # Add custom pipeline component for fixing wordnet frame pos tagging
        self.arg_schema_nlp = spacy.load("en_core_web_sm")
        self.arg_schema_nlp.add_pipe("wn_frame_schema_tagger", after="tagger")


        self.min_confidence = min_confidence
        self.use_wordnet_expansion = use_wordnet_expansion

        # Setup SpaCy extensions
        self._setup_extensions()

        # Initialize WordNet and FrameNet interfaces for triple processing
        # Keep frame_cache and lexical_unit_cache for compatibility but deprecate others
        self.frame_cache = self._get_fn_frames_for_lemma()
        self.lexical_unit_cache = self._get_wn_synsets_for_lemma() 
        # fe_coreness_cache removed - no longer needed for triple approach

    def _setup_extensions(self):
        """Setup custom SpaCy extensions for storing frame information"""
        # Document level - keep these for compatibility
        if not Doc.has_extension("frames"):
            Doc.set_extension("frames", default=[])
        if not Doc.has_extension("frame_elements"):
            Doc.set_extension("frame_elements", default=[])

        # Token level - keep frames but remove is_predicate (not meaningful in triple approach)
        if not Token.has_extension("frames"):
            Token.set_extension("frames", default=[])
        
        # Note: Removed Token.is_predicate as it's not meaningful in triple approach
        # Note: Removed Span extensions as we no longer use spans for frame elements

    def _flatten_tok_deps(self, tok: Token) -> Set[Token]:
        deps = set()
        for child in tok.children:
            deps.add(child)
            deps.update(self._flatten_tok_deps(child))
        return deps


    # TODO: retain original arg structure and add other syntactic/lexical info for better SRL
    def _flattened_fn_arg_schema(self, doc: Doc|Token|Set[Token]) -> Dict[str, str]:
        # used as fallback if FE parsing fails later on
        schema_atom_synset_maps = {
            "something": "entity.n.01",
            "someone": "causal_agent.n.01",
        }

        # if it's a token, just flatten its children and call that set a doc
        if isinstance(doc, Token):
            doc = self._flatten_tok_deps(doc)

        arg_schema = dict()
        for tok in doc:
            if tok.lower in schema_atom_synset_maps.keys():
                arg_schema[tok.dep_] = schema_atom_synset_maps[tok.lower_]
            else:
                arg_schema[tok.dep_] = tok.lemma_
        return arg_schema


    def _get_dep_reqs(self, doc: Doc|Token) -> Tuple[bool, ...]:
        arg_schema_reqs = self._flattened_fn_arg_schema(doc)

        # restructure arg_reqs based to only evaluate core dependencies
        #   TODO: add more relationships for finer-grained SRL
        return tuple([
            len(self.dep_map[k].intersection(set(arg_schema_reqs.keys()))) > 0
            for k in self.vect_order
        ])
    


    def _get_f_id_arg_struct_dict(self, syn) -> Dict[int, Tuple[bool, ...]]:
        syn_frame_ids_strs: Dict[int, Tuple[bool, ...]] = dict()
        for lemma in syn.lemmas():

            # get all FrameNet frame IDs for this lemma
            for i, f_id in enumerate(lemma.frame_ids()):

                # get the argument structure for this frame as a string
                f_str = lemma.frame_strings()[i]
                # parse f_str into an argument structure vector
                #   use tuple to make it hashable
                f_arg_structure = f_str.split(' ')
                
                # remove any extra arguments beyond subject, object, theme
                if f_arg_structure[-1] != f_arg_structure[-1].lower():
                    f_arg_structure = f_arg_structure[:-1]
                
                # create a spacy doc for the argument structure template
                doc = self.arg_schema_nlp(' '.join(f_arg_structure))
                # # display dependency parse tree for debugging
                # displacy.render(doc, style='dep')
                syn_frame_ids_strs[f_id] = self._get_dep_reqs(doc)
        
        return syn_frame_ids_strs


    def _apply_frame_based_WSD_filter(self, pred_tok: Token) -> Tuple[Set[str], Tuple[bool, ...]]:
        # since wordnet frames only support a max of 2 args, even for ditransitive verbs,
        #   truncate to just subj, obj roles
        original_tok_dep_reqs = self._get_dep_reqs(pred_tok)
        print(f"Original pred_tok dependency structure requirements: {original_tok_dep_reqs}")

        # get candidate WordNet synsets for the predicate (verbs)
        pred_lemma = pred_tok.lemma_.lower()
        # get a dict of synset names and synset objects for quick lookup
        pred_synsets = wn.synsets(pred_lemma, pos=wn.VERB)
        if not pred_synsets or not pred_synsets[0]:
            return set(), original_tok_dep_reqs
        pred_synsets = {syn.name(): syn for syn in pred_synsets if syn}

        # filter pred_synsets by their wordnet frames
        filtered_synsets: Set[str] = set()
        for s_name, syn in pred_synsets.items():
            for frame_id, arg_struct_tuple in self._get_f_id_arg_struct_dict(syn).items():
                # ensure the proposed tuple matches the requirements of the original token
                if arg_struct_tuple == original_tok_dep_reqs:
                    filtered_synsets.add(s_name)

        return filtered_synsets, original_tok_dep_reqs


    def _get_fn_frames_for_lemma(self) -> Dict:
        """Get FrameNet frames for lemmas - builds frame cache for quick lookup"""
        cache = {}
        for frame in fn.frames():
            cache[frame.name] = frame
        return cache

    def _get_wn_synsets_for_lemma(self) -> Dict[Tuple[str, str], List[str]]:
        """Get WordNet synsets for lemmas - builds lexical unit cache mapping (lemma, pos) -> [frame_names]"""
        cache = defaultdict(list)
        for frame in fn.frames():
            for lu_name, lu_data in frame.lexUnit.items():
                # Parse lexical unit name (e.g., "run.v" -> ("run", "v"))
                parts = lu_name.split('.')
                if len(parts) >= 2:
                    lemma = '.'.join(parts[:-1])
                    pos = parts[-1]
                    cache[(lemma.lower(), pos)].append(frame.name)
        return dict(cache)

    def _get_fe_coreness(self, frame_name: str, fe_name: str) -> str:
        """Get frame element coreness type on demand (replaces cache approach)"""
        if frame_name in self.frame_cache:
            frame = self.frame_cache[frame_name]
            if fe_name in frame.FE:
                return frame.FE[fe_name].coreType
        return "Non-Core"

    def __call__(self, doc: Doc) -> Doc:
        """Make the class callable as a SpaCy pipeline component"""
        return self.process_doc(doc)

    def process_text(self, text: str) -> Optional[Doc]:
        """
        Convenience method to process raw text.

        Args:
            text: Input text string

        Returns:
            Processed SpaCy Doc with frame annotations, or None if no NLP model available
        """
        return self.process_doc(self.nlp(text))


    def _get_subj_obj_theme_tokens(self,
            pred_tok: Token
        ) -> Dict[str, Optional[Token]]:
        """
        Extract subject, object, and theme tokens for a given predicate token.

        Args:
            pred_tok: Predicate token (verb)

        Returns:
            Tuple of (subject_token, object_token, theme_token)
        """
        role_tokens: Dict[str, Optional[Token]] = { k: None for k in self.dep_map.keys() }

        for child in pred_tok.children:
            for role, dep_set in self.dep_map.items():
                if child.dep_ in dep_set:
                    role_tokens[role] = child
                    break

        return role_tokens


    def process_doc(self, doc: Doc) -> Doc:
        """
        Process a SpaCy Doc using triple-based SRL approach.
        
        This method:
        1. Extracts all predicates (verbs) from the document
        2. For each predicate, extracts subject/object/theme using dependencies
        3. Processes each triple through the triple-based engine
        4. Converts results to FrameInstance format for backward compatibility

        Args:
            doc: SpaCy Doc object

        Returns:
            Doc with frame annotations added using triple-based processing
        """
        # Clear previous annotations
        doc._.frames = []
        doc._.frame_elements = []

        # Step 1: Extract all predicates (verbs) from the document
        predicates = []
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                predicates.append(token)

        # Step 2: For each predicate, process the triple with error handling
        for pred_tok in predicates:
            try:
                # Process the triple through the core engine
                triple_results = self.process_triple(pred_tok)
                
                # Convert triple results to FrameInstance for compatibility
                if triple_results:
                    frame_instance = self._triple_to_frame_instance(pred_tok, triple_results, doc)
                    if frame_instance and frame_instance.confidence >= self.min_confidence:
                        doc._.frames.append(frame_instance)
                        # Add frame elements to doc for compatibility
                        doc._.frame_elements.extend(frame_instance.elements)
                        # Mark predicate token with frame name
                        pred_tok._.frames.append(frame_instance.name)
                        
            except Exception as e:
                # Log error but continue processing other predicates
                print(f"[WARNING] FrameNetSpaCySRL: Error processing predicate '{pred_tok.text}': {e}")
                continue

        return doc


    def get_frame_summary(self, doc: Doc) -> Dict:
        """
        Get a summary of all frames and elements in the document.
        
        Enhanced version with better error handling and additional statistics.

        Returns:
            Dictionary with frame statistics and details
        """
        # Ensure doc has the required extensions
        if not hasattr(doc._, 'frames'):
            return {"error": "Document not processed by FrameNetSpaCySRL", "frames": [], "statistics": {}}
        
        summary = {
            "text": doc.text,
            "frames": [],
            "statistics": {
                "total_frames": len(doc._.frames),
                "total_elements": sum(len(f.elements) for f in doc._.frames),
                "predicates": [],
                "avg_confidence": 0.0,
                "frame_types": {}
            }
        }
        
        total_confidence = 0.0
        
        for frame_inst in doc._.frames:
            try:
                frame_data = {
                    "frame": frame_inst.name,
                    "predicate": frame_inst.target.text,
                    "predicate_span": [frame_inst.target.start_char, frame_inst.target.end_char],
                    "confidence": round(frame_inst.confidence, 3),
                    "definition": frame_inst.definition[:100] + "..." if len(frame_inst.definition) > 100 else frame_inst.definition,
                    "lexical_unit": frame_inst.lexical_unit,
                    "elements": []
                }

                for element in frame_inst.elements:
                    element_data = {
                        "role": element.name,
                        "text": element.span.text,
                        "span": [element.span.start_char, element.span.end_char],
                        "type": element.fe_type,
                        "confidence": round(element.confidence, 3),
                        "definition": element.definition[:50] + "..." if len(element.definition) > 50 else element.definition
                    }
                    frame_data["elements"].append(element_data)

                summary["frames"].append(frame_data)
                summary["statistics"]["predicates"].append(frame_inst.target.text)
                
                # Track frame type statistics
                frame_type = frame_inst.name
                summary["statistics"]["frame_types"][frame_type] = \
                    summary["statistics"]["frame_types"].get(frame_type, 0) + 1
                
                total_confidence += frame_inst.confidence
                
            except Exception as e:
                print(f"[WARNING] Error processing frame instance: {e}")
                continue
        
        # Calculate average confidence
        if summary["statistics"]["total_frames"] > 0:
            summary["statistics"]["avg_confidence"] = round(
                total_confidence / summary["statistics"]["total_frames"], 3
            )

        return summary

    def visualize_frames(self, doc: Doc) -> str:
        """
        Create a text-based visualization of frames and elements.
        
        Enhanced version with better formatting and error handling.

        Returns:
            Formatted string representation
        """
        # Check if document has been processed
        if not hasattr(doc._, 'frames'):
            return "ERROR: Document not processed by FrameNetSpaCySRL\n" + "=" * 80
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"FRAMENET SEMANTIC ROLE LABELING RESULTS")
        lines.append("=" * 80)
        lines.append(f"Text: {doc.text}")
        lines.append(f"Total Frames: {len(doc._.frames)}")
        lines.append(f"Total Elements: {sum(len(f.elements) for f in doc._.frames)}")
        lines.append("=" * 80)

        if not doc._.frames:
            lines.append("\nNo frames identified in the text.")
            lines.append("\nPossible reasons:")
            lines.append("  - No verbs detected")
            lines.append("  - Verbs below confidence threshold")
            lines.append("  - No matching FrameNet frames found")
        else:
            for i, frame_inst in enumerate(doc._.frames, 1):
                try:
                    lines.append(f"\n[{i}] Frame: {frame_inst.name}")
                    lines.append(f"    Predicate: '{frame_inst.target.text}' "
                                f"[chars {frame_inst.target.start_char}:{frame_inst.target.end_char}]")
                    lines.append(f"    Lexical Unit: {frame_inst.lexical_unit}")
                    lines.append(f"    Confidence: {frame_inst.confidence:.3f}")
                    
                    # Show definition, truncated if too long
                    definition = frame_inst.definition
                    if len(definition) > 120:
                        definition = definition[:120] + "..."
                    lines.append(f"    Definition: {definition}")

                    if frame_inst.elements:
                        lines.append("    Frame Elements:")
                        # Sort by position in text
                        sorted_elements = sorted(frame_inst.elements, key=lambda x: x.span.start_char)
                        
                        for elem in sorted_elements:
                            # Use different markers for different element types
                            if elem.fe_type == "Core":
                                marker = "***"
                            elif elem.fe_type == "Core-Unexpressed":
                                marker = "**"
                            else:
                                marker = "*"
                            
                            lines.append(f"      {marker} {elem.name}: '{elem.span.text}' "
                                       f"(type: {elem.fe_type}, conf: {elem.confidence:.3f})")
                            
                            # Show definition if available and not too generic
                            if (elem.definition and 
                                len(elem.definition) > 10 and 
                                not elem.definition.startswith("Frame element")):
                                elem_def = elem.definition
                                if len(elem_def) > 60:
                                    elem_def = elem_def[:60] + "..."
                                lines.append(f"        -> {elem_def}")
                    else:
                        lines.append("    No frame elements identified")
                        
                    # Add separator between frames
                    if i < len(doc._.frames):
                        lines.append("    " + "-" * 60)
                        
                except Exception as e:
                    lines.append(f"    ERROR: Could not visualize frame instance: {e}")

        lines.append("=" * 80)
        lines.append("Legend: *** Core, ** Core-Unexpressed, * Non-Core/Extra-Thematic")
        lines.append("=" * 80)
        return "\n".join(lines)

    # ===== Helper Methods =====
    def _score_synset_by_idi_and_freq(self,
            synset_name: str,
            synset_roles: Dict[str, Set[str]]
        ) -> float:
        """
        Calculate confidence for a specific synset selection.
        Enhanced version that considers synset selection and role matching.
        """
        if len(synset_roles["subjects"]) == 0:
            return 0.0  # Must have at least subject role
        
        # Determine role coverage and specificity
        total_roles = sum(len(roles) for roles in synset_roles.values())
        filled_role_types = sum(1 for roles in synset_roles.values() if roles)
        mean_role_count = total_roles / filled_role_types
        # Use inverse index of dispersion of roles to balance role coverage with specificity
        #   see https://en.wikipedia.org/wiki/Index_of_dispersion
        inverse_index_of_dispersion = mean_role_count / statistics.variance([
            (len(roles) / total_roles)
            for roles in synset_roles.values()
        ])
        
        # WordNet synset frequency bonus (more common synsets tend to be more reliable)
        # This is a simplified heuristic - in practice you'd use actual frequency data
        if '.' in synset_name:
            # Parse synset sense number (lower numbers = more common)
            sense_num = int(synset_name.split('.')[-1])
        
        # Combine factors into a score
        score = inverse_index_of_dispersion / sense_num  # Penalize higher sense numbers
        
        # apply sigmoid to normalize
        return sigmoid(score)
    

    # ===== Phase 1: Core Triple Processing Infrastructure =====
    def _get_fn_fes(self, fn_frame: Any) -> Dict[str, List[str]]:
        """
        Align syntactic dependency roles with FrameNet frame roles.
        Returns: {"subjects": [...], "objects": [...], "themes": [...]}
        
        This is a basic implementation that maps common semantic roles.
        """
        aligned_roles = {
            "subjects": [],
            "objects": [], 
            "themes": []
        }
        # Subject-like roles
        all_roles = {
            "subjects": ['agent', 'experiencer', 'cognizer', 'speaker', 'actor', 'protagonist', 'donor', 'giver', 'provider'],
            "objects": ['patient', 'theme', 'stimulus', 'content', 'message', 'undergoer'], 
            "themes": ['recipient', 'beneficiary', 'goal', 'destination', 'addressee']
        }
        combined_regexes = {
            k: re.compile('|'.join([re.escape(role) for role in v]), re.IGNORECASE)
            for k, v in all_roles.items()
        }

        # Get FrameNet frame elements if available
        if hasattr(fn_frame, 'FE'):
            frame_elements = list(fn_frame.FE.keys())
            
            # Map common FrameNet roles to the categories
            for fe in frame_elements:
                for k, v in combined_regexes.items():
                    if v.search(fe.lower()):
                        aligned_roles[k].append(fe)
        
        return aligned_roles


    def process_triple(self, pred_tok: Token) -> Dict[str, Dict[str, Set[str]]]:
        """
        Core triple processing logic adapted from new_srl_pipeline.py.
        
        Args:
            pred_tok: Predicate token (verb)
            subj_tok: Subject token (can be None)
            obj_tok: Object token (can be None) 
            wn: WordNet interface
            fn: FrameNet interface
            
        Returns: 
            Dict[synset_name, Dict[dependency_role, Set[frame_element_names]]]
        """
        # get candidate WordNet synsets for the predicate (verb)
        pred_synsets, fe_type_reqs_wn = self._apply_frame_based_WSD_filter(pred_tok)
        
        # 2. Get candidate FrameNet frames for the predicate lemma
        fn_frames = []
        pos_map = {'VERB': 'v'}
        if pred_tok.pos_ in pos_map:
            fn_pos = pos_map[pred_tok.pos_]
            key = (pred_tok.lemma_, fn_pos)

            if key in self.lexical_unit_cache:
                frame_names = self.lexical_unit_cache[key]
                fn_frames = [
                    self.frame_cache[name]
                    for name in frame_names
                    if name in self.frame_cache
                ]

        # for each valid synset and its frames, align to FrameNet frames
        valid_synset_frame_roles: Dict[str, Dict[str, Set[str]]] = {}
        for synset_name in pred_synsets:
            for fn_frame in fn_frames:
                # Get FrameNet frame entities
                fes: Dict[str, List[str]] = self._get_fn_fes(fn_frame)
                # Check if frame comes with required argument types or better
                fe_type_reqs_fn = tuple([
                    len(fes.get(role, [])) > 0
                    for role in ["subjects", "objects"]
                ])
                validated_reqs = all(
                    not a or b  # a ==> b
                    for a, b in zip(fe_type_reqs_wn, fe_type_reqs_fn)
                )
                # Only keep frames that meet or exceed WordNet-based fe type requirements
                if validated_reqs:
                    # Map aligned roles to synset
                    for role_type, role_names in fes.items():
                        # Initialize entry if not present
                        if synset_name not in valid_synset_frame_roles:
                            valid_synset_frame_roles[synset_name] = {
                                "subjects": set(),
                                "objects": set(),
                                "themes": set()
                            }
                        
                        # Add all aligned roles for this type
                        valid_synset_frame_roles[synset_name][role_type].update(role_names)

        return valid_synset_frame_roles

    def _triple_to_frame_instance(self,
            pred_tok: Token,
            triple_results: Dict[str, Dict[str, Set[str]]], 
            doc: Doc,
            sample_size: int = 3
        ) -> Optional[FrameInstance]:
        """
        Convert triple processing results to FrameInstance for backward compatibility.
        
        Enhanced version that properly maps synsets and semantic roles to frame and frame elements,
        uses calculated confidence scores, and handles FrameNet frame lookup correctly.
        
        Args:
            pred_tok: The predicate token
            triple_results: Results from process_triple
            doc: The spaCy Doc
            
        Returns:
            FrameInstance object matching the original API format
        """
        if not triple_results:
            return None
                    
        # Calculate robustness score for each synset assignment
        sorted_synsets = list()
        for synset_name, synset_roles in triple_results.items():
            synset_score = self._score_synset_by_idi_and_freq(synset_name, synset_roles)
            sorted_synsets.append((synset_name, synset_roles, synset_score))
        # Sort by usefulness descending
        sorted_synsets.sort(key=lambda x: x[2], reverse=True)

        # Try to find the best matching FrameNet frame
        for synset_name, synset_roles, synset_confidence in sorted_synsets[:sample_size]:
            frame_name, frame_definition = self._find_best_framenet_frame(pred_tok, synset_roles)
        
        # Create target span (expand to include particles/auxiliaries if present)
        target_span = self._get_predicate_span(pred_tok, doc)
        
        # Create frame elements from triple roles with proper confidence and type mapping
        frame_elements = self._create_frame_elements(pred_tok, synset_roles, frame_name, doc)
        
        # Create FrameInstance with calculated confidence
        frame_instance = FrameInstance(
            name=frame_name,
            target=target_span,
            elements=frame_elements,
            confidence=synset_confidence,
            definition=frame_definition,
            lexical_unit=f"{pred_tok.lemma_}.v"
        )
        
        return frame_instance

    
    def _find_best_framenet_frame(self, pred_tok: Token, synset_roles: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Find the best matching FrameNet frame for the predicate and synset roles.
        
        Args:
            pred_tok: The predicate token
            synset_roles: Semantic roles from synset alignment
            
        Returns:
            Tuple of (frame_name, frame_definition)
        """
        pred_lemma = pred_tok.lemma_.lower()
        
        # Look up FrameNet frames for this predicate
        key = (pred_lemma, 'v')
        if key in self.lexical_unit_cache:
            candidate_frames = self.lexical_unit_cache[key]
            
            # Score each candidate frame based on role alignment
            best_frame = None
            best_score = 0
            
            for frame_name in candidate_frames:
                if frame_name in self.frame_cache:
                    frame = self.frame_cache[frame_name]
                    score = self._score_frame_alignment(frame, synset_roles)
                    if score > best_score:
                        best_score = score
                        best_frame = frame_name
                        
            if best_frame:
                frame_def = self.frame_cache[best_frame].definition if best_frame in self.frame_cache else ""
                return best_frame, frame_def
        
        # Fallback: create a generic frame name from the predicate
        fallback_name = f"Generic_{pred_lemma}_frame"
        fallback_def = f"Generic frame evoked by the verb '{pred_lemma}'"
        return fallback_name, fallback_def
    
    def _score_frame_alignment(self, frame: Any, synset_roles: Dict[str, Set[str]]) -> int:
        """
        Score how well a FrameNet frame aligns with the synset roles.
        
        Args:
            frame: FrameNet frame object
            synset_roles: Semantic roles from synset alignment
            
        Returns:
            Alignment score (higher = better match)
        """
        score = 0
        
        if hasattr(frame, 'FE'):
            frame_elements = set(frame.FE.keys())
            
            # Check overlap between frame elements and our semantic roles
            all_synset_roles = set()
            for roles in synset_roles.values():
                all_synset_roles.update(roles)
            
            # Score based on overlap
            overlap = frame_elements.intersection(all_synset_roles)
            score += len(overlap) * 2
            
            # Bonus for having core elements
            for fe_name in frame_elements:
                if fe_name in all_synset_roles:
                    fe_type = self._get_fe_coreness(frame.name, fe_name)
                    if fe_type == "Core":
                        score += 3
                    elif fe_type == "Core-Unexpressed":
                        score += 2
        
        return score
    
    def _get_predicate_span(self, pred_tok: Token, doc: Doc) -> Span:
        """
        Get span for predicate, including particles and auxiliaries.
        
        Args:
            pred_tok: The predicate token
            doc: The spaCy Doc
            
        Returns:
            Span containing the full predicate phrase
        """
        # Start with the predicate token
        start_idx = pred_tok.i
        end_idx = pred_tok.i + 1
        
        # Include particles (e.g., "give up")
        for child in pred_tok.children:
            if child.dep_ == "prt":  # particle
                start_idx = min(start_idx, child.i)
                end_idx = max(end_idx, child.i + 1)
        
        # Include auxiliary verbs that come before
        for token in doc:
            if (token.i < pred_tok.i and 
                token.dep_ in ["aux", "auxpass"] and 
                token.head == pred_tok):
                start_idx = min(start_idx, token.i)
                break
        
        return doc[start_idx:end_idx]
    
    def _create_frame_elements(self, pred_tok: Token, synset_roles: Dict[str, Set[str]], 
                              frame_name: str, doc: Doc) -> List[FrameElement]:
        """
        Create frame elements from synset roles with proper confidence and type mapping.
        
        Args:
            pred_tok: The predicate token
            synset_roles: Semantic roles from synset alignment
            frame_name: Name of the selected frame
            doc: The spaCy Doc
            
        Returns:
            List of FrameElement objects
        """
        frame_elements = []
        # Extract subject, object, theme
        role_tokens = self._get_subj_obj_theme_tokens(pred_tok)
        
        # Map tokens to frame elements with calculated confidence
        for role_type, token in role_tokens.items():
            if token and synset_roles.get(role_type):
                semantic_roles = list(synset_roles[role_type])
                if semantic_roles:
                    # Take the first (best) semantic role
                    semantic_role = semantic_roles[0]
                    
                    # Get appropriate span for this token
                    element_span = self._get_token_span(token, doc)
                    
                    # Calculate confidence based on role strength and token properties
                    confidence = self._calculate_frame_element_confidence(
                        token, semantic_role, role_type, len(semantic_roles)
                    )
                    
                    # Determine frame element type (Core/Non-Core) from FrameNet
                    fe_type = self._get_fe_coreness(frame_name, semantic_role)
                    if not fe_type or fe_type == "":
                        # Default based on role type
                        fe_type = "Core" if role_type in ["subjects", "objects"] else "Non-Core"
                    
                    # Get definition if available
                    definition = self._get_fe_definition(frame_name, semantic_role)
                    
                    element = FrameElement(
                        name=semantic_role,
                        span=element_span,
                        frame_name=frame_name,
                        confidence=confidence,
                        fe_type=fe_type,
                        definition=definition
                    )
                    frame_elements.append(element)
        
        return frame_elements
    
    def _calculate_frame_element_confidence(self, token: Token, semantic_role: str, 
                                          role_type: str, num_alternatives: int) -> float:
        """
        Calculate confidence score for a frame element assignment.
        
        Args:
            token: The token being assigned
            semantic_role: The semantic role being assigned
            role_type: Type of role (subjects/objects/themes)
            num_alternatives: Number of alternative semantic roles available
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.6
        
        # Higher confidence for subjects and objects (core arguments)
        if role_type in ["subjects", "objects"]:
            confidence += 0.2
        
        # Lower confidence if there are many alternative roles (ambiguity)
        ambiguity_penalty = min(0.2, (num_alternatives - 1) * 0.05)
        confidence -= ambiguity_penalty
        
        # Boost for proper nouns (likely to be good arguments)
        if token.ent_type_ in ["PERSON", "ORG", "GPE"]:
            confidence += 0.1
        
        # Boost for pronouns in subject position
        if role_type == "subjects" and token.pos_ == "PRON":
            confidence += 0.1
            
        return min(0.95, max(0.3, confidence))
    
    def _get_fe_definition(self, frame_name: str, fe_name: str) -> str:
        """
        Get frame element definition from FrameNet.
        
        Args:
            frame_name: Name of the frame
            fe_name: Name of the frame element
            
        Returns:
            Definition string, or empty string if not found
        """
        if frame_name in self.frame_cache:
            frame = self.frame_cache[frame_name]
            if hasattr(frame, 'FE') and fe_name in frame.FE:
                fe = frame.FE[fe_name]
                return getattr(fe, 'definition', '') or f"{fe_name} in {frame_name}"
        return f"Frame element {fe_name}"
    
    def _get_token_span(self, token: Token, doc: Doc) -> Span:
        """
        Get span for a token, preferring noun chunks when available.
        
        Args:
            token: The token to get span for
            doc: The spaCy Doc
            
        Returns:
            Span containing the token and its constituents
        """
        # Try to find noun chunk containing this token
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk
                
        # For pronouns, just return the token itself
        if token.pos_ == "PRON":
            return doc[token.i:token.i+1]
            
        # Fallback: use token's subtree (but limit expansion)
        left = token.left_edge.i
        right = token.right_edge.i + 1
        
        # Limit subtree expansion to avoid overly large spans
        max_span_length = 5
        if right - left > max_span_length:
            # Just use the token and immediate children
            left = max(token.i - 1, left)
            right = min(token.i + 2, right)
        
        return doc[left:right]