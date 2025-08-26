"""
FrameNet-SpaCy Semantic Role Labeling Pipeline
A complete span-labeling based SRL system integrating NLTK's FrameNet with SpaCy
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import re

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
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"[WARNING] FrameNetSpaCySRL: spaCy '{spacy_model}' model not found.")
                print(f"[WARNING] Install with: python -m spacy download {spacy_model}")
                print("[WARNING] FrameNetSpaCySRL will operate with limited functionality.")
                self.nlp = None

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
        if self.nlp is None:
            print("[WARNING] FrameNetSpaCySRL: Cannot process text, no spaCy model available.")
            return None
        doc = self.nlp(text)
        return self.process_doc(doc)

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
                # Extract subject, object, theme using Phase 1 helpers
                subj_tok = self._get_subject(pred_tok)
                obj_tok = self._get_object(pred_tok)
                theme_tok = self._get_theme(pred_tok)
                
                # Skip if no arguments found (intransitive verbs with no subject)
                if not any([subj_tok, obj_tok, theme_tok]):
                    continue
                
                # Process the triple through the core engine
                triple_results = self.process_triple(pred_tok, subj_tok, obj_tok, wn, fn)
                
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

    def process(self, doc: Doc) -> Doc:
        """
        Backward-compatible process method.
        
        This method maintains the original API while internally using 
        the new triple-based processing approach.
        
        Args:
            doc: SpaCy Doc object
        
        Returns:
            Doc with frame annotations added
        """
        return self.process_doc(doc)

















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

    # ===== Helper Methods (moved from deprecated span-based methods) =====
    
    def _calculate_triple_confidence(self, synset_roles: Dict[str, Set[str]]) -> float:
        """
        Calculate confidence for triple-based semantic role assignment.
        Enhanced version with better confidence modeling.
        """
        if not synset_roles:
            return 0.2
        
        # Base confidence based on number of filled roles
        filled_roles = sum(1 for roles in synset_roles.values() if roles)
        total_roles = sum(len(roles) for roles in synset_roles.values())
        
        if filled_roles == 0:
            return 0.2
        
        # Base score from role coverage
        base_confidence = min(0.8, 0.3 + (filled_roles * 0.15))
        
        # Bonus for having multiple semantic roles (indicates rich frame)
        if total_roles > filled_roles:
            base_confidence += min(0.1, (total_roles - filled_roles) * 0.02)
        
        # Bonus for having subject (most important argument)
        if synset_roles.get("subjects"):
            base_confidence += 0.1
            
        # Penalty for incomplete argument structure
        if not synset_roles.get("objects") and filled_roles < 2:
            base_confidence -= 0.05
        
        return min(0.95, max(0.2, base_confidence))
        
    def _calculate_synset_confidence(self, synset_name: str, synset_roles: Dict[str, Set[str]]) -> float:
        """
        Calculate confidence for a specific synset selection.
        Enhanced version that considers synset quality and role coherence.
        """
        if not synset_roles:
            return 0.2
        
        # Base confidence from role coverage
        total_roles = sum(len(roles) for roles in synset_roles.values())
        filled_role_types = sum(1 for roles in synset_roles.values() if roles)
        
        base_confidence = min(0.8, 0.4 + (total_roles * 0.08))
        
        # Bonus for balanced argument structure
        if filled_role_types >= 2:  # Has both subject and object/theme
            base_confidence += 0.1
        
        # WordNet synset frequency bonus (more common synsets tend to be more reliable)
        # This is a simplified heuristic - in practice you'd use actual frequency data
        if '.' in synset_name:
            # Parse synset sense number (lower numbers = more common)
            try:
                sense_num = int(synset_name.split('.')[-1])
                if sense_num <= 2:  # First two senses are typically more reliable
                    base_confidence += 0.05
            except (ValueError, IndexError):
                pass
        
        # Penalty for overly complex role assignments (might indicate overalignment)
        if total_roles > 6:
            base_confidence -= 0.1
            
        return min(0.9, max(0.2, base_confidence))
    
    # ===== Phase 1: Core Triple Processing Infrastructure =====
    
    def _get_subject(self, pred_tok: Token) -> Optional[Token]:
        """
        Extract subject token from predicate using spaCy dependencies.
        Looks for nsubj, nsubjpass dependencies.
        """
        for child in pred_tok.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return child
        return None

    def _get_object(self, pred_tok: Token) -> Optional[Token]:
        """
        Extract direct object token from predicate.
        Looks for dobj, obj dependencies.
        """
        for child in pred_tok.children:
            if child.dep_ in ["dobj", "obj"]:
                return child
        return None

    def _get_theme(self, pred_tok: Token) -> Optional[Token]:
        """
        Extract indirect object/theme token from predicate.
        Looks for iobj, dative dependencies.
        """
        for child in pred_tok.children:
            if child.dep_ in ["iobj", "dative"]:
                return child
        return None

    def _lemmatize(self, word: str) -> str:
        """
        Get lemma form of word using spaCy.
        """
        if self.nlp is None:
            return word.lower()
        
        # Process the word to get its lemma
        doc = self.nlp(word)
        if len(doc) > 0:
            return doc[0].lemma_
        return word.lower()

    def _align_wn_fn_frames(self, wn_frame: Any, fn_frame: Any) -> Dict[str, List[str]]:
        """
        Align WordNet frame with FrameNet frame.
        Returns: {"subjects": [...], "objects": [...], "themes": [...]}
        
        This is a basic implementation that maps common semantic roles.
        """
        aligned_roles = {
            "subjects": [],
            "objects": [], 
            "themes": []
        }
        
        # Get FrameNet frame elements if available
        if hasattr(fn_frame, 'FE'):
            frame_elements = list(fn_frame.FE.keys())
            
            # Map common FrameNet roles to our categories
            for fe in frame_elements:
                fe_lower = fe.lower()
                
                # Subject-like roles
                if any(subj_role in fe_lower for subj_role in 
                       ['agent', 'experiencer', 'cognizer', 'speaker', 'actor', 'protagonist', 'donor', 'giver', 'provider']):
                    aligned_roles["subjects"].append(fe)
                
                # Object-like roles  
                elif any(obj_role in fe_lower for obj_role in
                        ['patient', 'theme', 'stimulus', 'content', 'message', 'undergoer']):
                    aligned_roles["objects"].append(fe)
                
                # Theme/indirect object-like roles
                elif any(theme_role in fe_lower for theme_role in
                        ['recipient', 'beneficiary', 'goal', 'destination', 'addressee']):
                    aligned_roles["themes"].append(fe)
        
        return aligned_roles

    def process_triple(self, pred_tok: Token, subj_tok: Optional[Token], 
                      obj_tok: Optional[Token], wn: Any, fn: Any) -> Dict[str, Dict[str, Set[str]]]:
        """
        Core triple processing logic adapted from new_srl_pipeline.py.
        
        Args:
            pred_tok: Predicate token (verb)
            subj_tok: Subject token (can be None)
            obj_tok: Object token (can be None) 
            wn: WordNet interface
            fn: FrameNet interface
            
        Returns: 
            Dict[synset_name, Dict[dependency_role, Set[semantic_roles]]]
        """
        # 1. Get candidate WordNet synsets for the predicate (verbs)
        pred_lemma = pred_tok.lemma_
        pred_synsets = wn.synsets(pred_lemma, pos=wn.VERB)
        
        # 2. Get candidate FrameNet frames for the predicate lemma
        fn_frames = []
        pos_map = {'VERB': 'v'}
        if pred_tok.pos_ in pos_map:
            fn_pos = pos_map[pred_tok.pos_]
            key = (pred_lemma.lower(), fn_pos)
            if key in self.lexical_unit_cache:
                frame_names = self.lexical_unit_cache[key]
                fn_frames = [self.frame_cache[name] for name in frame_names if name in self.frame_cache]

        # 3. Extract spaCy-derived argument tokens
        # Use the helper functions we just implemented
        spacy_args = {
            "subj": self._get_subject(pred_tok),
            "obj": self._get_object(pred_tok), 
            "iobj": self._get_theme(pred_tok)
        }
        
        # Override with provided arguments if they exist
        if subj_tok is not None:
            spacy_args["subj"] = subj_tok
        if obj_tok is not None:
            spacy_args["obj"] = obj_tok

        # 4. Filter WordNet synsets by argument structure compatibility
        valid_synsets_frames: Dict[str, List[Any]] = {}

        for synset in pred_synsets:
            # Get WordNet frames (using a basic approach since frames() may not exist)
            # We'll simulate this by checking synset definitions and example patterns
            wn_frames = self._get_wn_frames_for_synset(synset, spacy_args)
            
            if wn_frames:
                valid_synsets_frames[synset.name()] = wn_frames

        # 5. For each valid synset & its frames, align to FrameNet frames
        valid_synset_frame_roles: Dict[str, Dict[str, Set[str]]] = {}

        for synset_name, valid_wn_frames in valid_synsets_frames.items():
            for wn_frame in valid_wn_frames:
                for fn_frame in fn_frames:
                    # Align WordNet frame with FrameNet frame
                    valid_args: Dict[str, List[str]] = self._align_wn_fn_frames(wn_frame, fn_frame)

                    # Only keep this synset-frame pairing if we have at least subject info
                    if valid_args["subjects"]:
                        # Initialize if not exists
                        if synset_name not in valid_synset_frame_roles:
                            valid_synset_frame_roles[synset_name] = {
                                "subjects": set(),
                                "objects": set(), 
                                "themes": set()
                            }
                        
                        # Add semantic roles to the sets
                        existing = valid_synset_frame_roles[synset_name]
                        existing["subjects"].update(valid_args["subjects"])
                        existing["objects"].update(valid_args["objects"])
                        existing["themes"].update(valid_args["themes"])

        return valid_synset_frame_roles

    def _get_wn_frames_for_synset(self, synset: Any, spacy_args: Dict[str, Optional[Token]]) -> List[Any]:
        """
        Get WordNet frames for a synset based on argument structure.
        This is a simplified implementation that creates frame-like structures
        based on the presence of arguments.
        """
        frames = []
        
        # Create a basic frame representation based on argument structure
        frame_arity = 0
        if spacy_args.get("subj"):
            frame_arity += 1
        if spacy_args.get("obj"):
            frame_arity += 1
        if spacy_args.get("iobj"):
            frame_arity += 1
            
        # Create a mock frame object with the determined arity
        # This simulates the filtering logic from new_srl_pipeline.py
        class MockWNFrame:
            def __init__(self, arity):
                self.arity = arity
            
            def __len__(self):
                return self.arity
        
        # Always include a frame that matches our argument structure
        frames.append(MockWNFrame(frame_arity))
        
        # Also include frames of different arities for flexibility
        for arity in [1, 2, 3]:
            if arity != frame_arity:
                frames.append(MockWNFrame(arity))
        
        return frames

    def _triple_to_frame_instance(self, pred_tok: Token, triple_results: Dict[str, Dict[str, Set[str]]], 
                                 doc: Doc) -> Optional[FrameInstance]:
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
        
        # Select the best synset based on confidence scores
        best_synset_name, synset_roles = self._select_best_synset(triple_results)
        if not synset_roles:
            return None
            
        # Calculate confidence for this synset assignment
        synset_confidence = self._calculate_synset_confidence(best_synset_name, synset_roles)
        
        # Try to find the best matching FrameNet frame
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
        
    def _select_best_synset(self, triple_results: Dict[str, Dict[str, Set[str]]]) -> Tuple[str, Dict[str, Set[str]]]:
        """
        Select the best synset from triple results based on role coverage and confidence.
        
        Args:
            triple_results: Results from process_triple
            
        Returns:
            Tuple of (best_synset_name, synset_roles)
        """
        if not triple_results:
            return "", {}
            
        best_synset = None
        best_score = 0
        best_roles = {}
        
        for synset_name, synset_roles in triple_results.items():
            # Score based on number of filled roles and role coverage
            score = 0
            for role_type, roles in synset_roles.items():
                if roles:  # If this role type has semantic roles assigned
                    score += len(roles)  # More roles = higher score
                    
            # Bonus for having subject (core argument)
            if synset_roles.get("subjects"):
                score += 2
                
            if score > best_score:
                best_score = score
                best_synset = synset_name
                best_roles = synset_roles
        
        return best_synset or "", best_roles
    
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
        
        # Get actual tokens for subject, object, theme
        role_tokens = {
            "subjects": self._get_subject(pred_tok),
            "objects": self._get_object(pred_tok),
            "themes": self._get_theme(pred_tok)
        }
        
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
        
    # ===== Utility Methods for External Use (REMOVE) =====
    
    def _is_animate_word(self, word: str) -> bool:
        """
        Check if word typically refers to animate entities.
        Moved here as utility method that might be useful elsewhere.
        """
        synsets = wn.synsets(word, pos=wn.NOUN)
        
        # Check direct word matches for common animate entities
        animate_words = {'cat', 'dog', 'person', 'human', 'animal', 'bird', 'fish', 'horse', 'cow'}
        if word.lower() in animate_words:
            return True
            
        for synset in synsets[:3]:  # Check top 3 senses
            # Check all hypernym paths (not just direct hypernyms)
            all_hypernyms = synset.closure(lambda s: s.hypernyms())
            for hypernym in all_hypernyms:
                hypernym_name = hypernym.name().lower()
                # Check for common animate hypernyms
                if any(animate_term in hypernym_name for animate_term in 
                       ['person', 'individual', 'animal', 'organism', 
                        'living_thing', 'being', 'creature', 'human', 'vertebrate']):
                    return True
            
            # Also check direct synset names for animate terms
            synset_name = synset.name().lower()
            if any(animate_term in synset_name for animate_term in 
                   ['person', 'animal', 'human', 'being']):
                return True
        
        return False

    def _is_concrete_word(self, word: str) -> bool:
        """
        Check if word typically refers to concrete, tangible entities.
        Moved here as utility method that might be useful elsewhere.
        """
        synsets = wn.synsets(word, pos=wn.NOUN)
        
        # Check direct word matches for common concrete objects
        concrete_words = {'book', 'car', 'table', 'chair', 'house', 'computer', 'phone', 'mouse'}
        if word.lower() in concrete_words:
            return True
            
        for synset in synsets[:3]:  # Check top 3 senses
            # Check all hypernym paths
            all_hypernyms = synset.closure(lambda s: s.hypernyms())
            for hypernym in all_hypernyms:
                hypernym_name = hypernym.name().lower()
                # Physical objects, artifacts, substances
                if any(concrete_term in hypernym_name for concrete_term in 
                       ['artifact', 'physical_object', 'whole', 'object', 
                        'instrumentality', 'device', 'structure', 'container',
                        'conveyance', 'vehicle', 'machine', 'furniture']):
                    return True
            
            # Check direct synset for concrete terms
            synset_name = synset.name().lower()
            if any(concrete_term in synset_name for concrete_term in 
                   ['object', 'thing', 'item', 'artifact']):
                return True
                
        return False