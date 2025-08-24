"""
FrameNet-SpaCy Semantic Role Labeling Pipeline
A complete span-labeling based SRL system integrating NLTK's FrameNet with SpaCy
"""

from typing import List, Dict, Tuple, Optional, Set
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
            self.nlp = spacy.load(spacy_model)

        self.min_confidence = min_confidence
        self.use_wordnet_expansion = use_wordnet_expansion

        # Setup SpaCy extensions
        self._setup_extensions()

        # Build caches for efficiency
        self.frame_cache = self._build_frame_cache()
        self.lexical_unit_cache = self._build_lexical_unit_cache()
        self.fe_coreness_cache = self._build_fe_coreness_cache()

    def _setup_extensions(self):
        """Setup custom SpaCy extensions for storing frame information"""
        # Document level
        if not Doc.has_extension("frames"):
            Doc.set_extension("frames", default=[])
        if not Doc.has_extension("frame_elements"):
            Doc.set_extension("frame_elements", default=[])

        # Token level
        if not Token.has_extension("frames"):
            Token.set_extension("frames", default=[])
        if not Token.has_extension("is_predicate"):
            Token.set_extension("is_predicate", default=False)

        # Span level
        if not Span.has_extension("frame"):
            Span.set_extension("frame", default=None)
        if not Span.has_extension("frame_element"):
            Span.set_extension("frame_element", default=None)
        if not Span.has_extension("semantic_role"):
            Span.set_extension("semantic_role", default=None)

    def _build_frame_cache(self) -> Dict:
        """Build cache of all frames for quick lookup"""
        cache = {}
        for frame in fn.frames():
            cache[frame.name] = frame
        return cache

    def _build_lexical_unit_cache(self) -> Dict[Tuple[str, str], List[str]]:
        """Build cache mapping (lemma, pos) -> [frame_names]"""
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

    def _build_fe_coreness_cache(self) -> Dict[Tuple[str, str], str]:
        """Build cache of frame element coreness types"""
        cache = {}
        for frame in fn.frames():
            for fe_name, fe_data in frame.FE.items():
                cache[(frame.name, fe_name)] = fe_data.coreType
        return cache

    def __call__(self, doc: Doc) -> Doc:
        """Make the class callable as a SpaCy pipeline component"""
        return self.process(doc)

    def process_text(self, text: str) -> Doc:
        """
        Convenience method to process raw text.

        Args:
            text: Input text string

        Returns:
            Processed SpaCy Doc with frame annotations
        """
        doc = self.nlp(text)
        return self.process(doc)

    def process(self, doc: Doc) -> Doc:
        """
        Process a SpaCy Doc to identify frames and label semantic roles.

        Args:
            doc: SpaCy Doc object

        Returns:
            Doc with frame annotations added
        """
        # Clear previous annotations
        doc._.frames = []
        doc._.frame_elements = []

        # Step 1: Identify predicates (frame-evoking elements)
        predicates = self._identify_predicates(doc)

        # Step 2: For each predicate, identify possible frames
        for pred_span in predicates:
            frames = self._get_frames_for_predicate(pred_span)

            if frames:
                # Step 3: For best frame, identify and align frame elements
                best_frame = self._select_best_frame(pred_span, frames, doc)
                if best_frame:
                    frame_instance = self._extract_frame_elements(
                        doc, pred_span, best_frame
                    )
                    if frame_instance:
                        doc._.frames.append(frame_instance)
                        # Mark predicate token
                        for token in pred_span:
                            token._.is_predicate = True
                            token._.frames.append(frame_instance.name)

        return doc

    def _identify_predicates(self, doc: Doc) -> List[Span]:
        """
        Identify potential frame-evoking elements (predicates) in the document.

        Returns verbs, relevant nouns, and adjectives that might evoke frames.
        """
        predicates = []

        # Verbs are primary frame evokers
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                # Check for phrasal verbs and multi-word expressions
                span = self._expand_predicate_span(token)
                predicates.append(span)

        # Some nouns and adjectives also evoke frames
        for chunk in doc.noun_chunks:
            head = chunk.root
            # Check if head noun is in our lexical unit cache
            if (head.lemma_.lower(), 'n') in self.lexical_unit_cache:
                predicates.append(doc[head.i:head.i+1])

        # Adjectives in predicative position
        for token in doc:
            if token.pos_ == "ADJ" and token.dep_ in ["acomp", "xcomp"]:
                if (token.lemma_.lower(), 'a') in self.lexical_unit_cache:
                    predicates.append(doc[token.i:token.i+1])

        return predicates

    def _expand_predicate_span(self, verb: Token) -> Span:
        """
        Expand verb to include particles and auxiliaries for phrasal verbs.
        E.g., "pick up", "look forward to"
        """
        doc = verb.doc
        start = verb.i
        end = verb.i + 1

        # Include particles
        for child in verb.children:
            if child.dep_ == "prt" and child.i > verb.i:
                end = max(end, child.i + 1)

        # Include auxiliary verbs
        for child in verb.children:
            if child.dep_ == "aux" and child.i < verb.i:
                start = min(start, child.i)

        return doc[start:end]

    def _get_frames_for_predicate(self, pred_span: Span) -> List[str]:
        """Get possible frames for a predicate using lexical units and WordNet."""
        frames = set()

        # Primary lookup via lexical units
        head = pred_span.root
        pos_map = {'VERB': 'v', 'NOUN': 'n', 'ADJ': 'a', 'ADV': 'adv'}
        if head.pos_ in pos_map:
            fn_pos = pos_map[head.pos_]
            key = (head.lemma_.lower(), fn_pos)
            if key in self.lexical_unit_cache:
                frames.update(self.lexical_unit_cache[key])

        # WordNet expansion if enabled
        if self.use_wordnet_expansion and head.pos_ == "VERB":
            expanded_frames = self._get_frames_via_wordnet(head.lemma_, wn.VERB)
            frames.update(expanded_frames)

        return list(frames)

    def _get_frames_via_wordnet(self, lemma: str, pos) -> Set[str]:
        """Expand frame search using WordNet synonyms"""
        frames = set()
        for synset in wn.synsets(lemma, pos=pos)[:3]:  # Limit to top 3 senses
            for syn_lemma in synset.lemma_names():
                key = (syn_lemma.lower(), 'v')
                if key in self.lexical_unit_cache:
                    frames.update(self.lexical_unit_cache[key])
        return frames

    def _select_best_frame(self, pred_span: Span,
                          frame_names: List[str],
                          doc: Doc) -> Optional[str]:
        """
        Select the best frame for a predicate based on context.
        Uses simple heuristics - could be enhanced with ML model.
        """
        if len(frame_names) == 1:
            return frame_names[0]

        scores = {}
        for frame_name in frame_names:
            frame = self.frame_cache[frame_name]
            score = 0.0

            # Score based on lexical unit match
            pred_text = pred_span.text.lower()
            for lu_name in frame.lexUnit:
                if pred_text in lu_name.lower():
                    score += 2.0

            # Score based on frame element compatibility with syntax
            score += self._score_syntactic_compatibility(pred_span, frame, doc)

            scores[frame_name] = score

        # Return highest scoring frame if above threshold
        best_frame = max(scores, key=scores.get)
        if scores[best_frame] >= self.min_confidence:
            return best_frame
        return None

    def _score_syntactic_compatibility(self, pred_span: Span,
                                      frame, doc: Doc) -> float:
        """Score how well frame elements align with syntactic structure"""
        score = 0.0
        pred_head = pred_span.root

        # Check for core frame elements in syntactic dependents
        core_fes = [fe for fe, data in frame.FE.items()
                   if data.coreType == "Core"]

        # Subject (nsubj) often maps to Agent-like FEs
        for child in pred_head.children:
            if child.dep_ == "nsubj":
                if any(fe in ["Agent", "Theme", "Experiencer"]
                      for fe in core_fes):
                    score += 1.0
            elif child.dep_ in ["dobj", "obj"]:
                if any(fe in ["Theme", "Patient", "Goal"]
                      for fe in core_fes):
                    score += 1.0

        return score

    def _extract_frame_elements(self, doc: Doc, pred_span: Span,
                               frame_name: str) -> Optional[FrameInstance]:
        """
        Extract and align frame elements with syntactic spans.
        """
        frame = self.frame_cache[frame_name]
        frame_elements = []
        pred_head = pred_span.root

        # Map syntactic dependents to frame elements
        for child in pred_head.children:
            fe = self._map_dependent_to_fe(child, frame, pred_head)
            if fe:
                # Get the span for this frame element
                span = self._get_fe_span(child, doc)
                if span:
                    element = FrameElement(
                        name=fe,
                        span=span,
                        frame_name=frame_name,
                        confidence=self._calculate_confidence(child, fe, frame),
                        fe_type=self.fe_coreness_cache.get((frame_name, fe), "Non-Core"),
                        definition=frame.FE[fe].definition if fe in frame.FE else ""
                    )
                    frame_elements.append(element)
                    # Set span extension
                    span._.frame_element = fe
                    span._.frame = frame_name

        # Also check for frame elements in prepositional phrases
        for pp in self._get_prepositional_phrases(pred_head):
            fe = self._map_pp_to_fe(pp, frame, pred_head)
            if fe:
                span = doc[pp.left_edge.i:pp.right_edge.i+1]
                element = FrameElement(
                    name=fe,
                    span=span,
                    frame_name=frame_name,
                    confidence=self._calculate_confidence(pp, fe, frame),
                    fe_type=self.fe_coreness_cache.get((frame_name, fe), "Non-Core"),
                    definition=frame.FE[fe].definition if fe in frame.FE else ""
                )
                frame_elements.append(element)
                span._.frame_element = fe
                span._.frame = frame_name

        if frame_elements or True:  # Create instance even without elements
            return FrameInstance(
                name=frame_name,
                target=pred_span,
                elements=frame_elements,
                confidence=self._calculate_frame_confidence(frame_elements),
                definition=frame.definition,
                lexical_unit=f"{pred_span.text}.{pred_head.pos_[0].lower()}"
            )

        return None

    def _map_dependent_to_fe(self, dep: Token, frame,
                            pred: Token) -> Optional[str]:
        """Map syntactic dependent to frame element"""
        dep_role = dep.dep_

        # Mapping rules based on dependency and frame
        mappings = {
            "nsubj": ["Agent", "Experiencer", "Theme", "Cognizer", "Speaker"],
            "nsubjpass": ["Patient", "Theme", "Undergoer"],
            "dobj": ["Theme", "Patient", "Goal", "Stimulus", "Content"],
            "obj": ["Theme", "Patient", "Goal", "Stimulus", "Content"],
            "iobj": ["Recipient", "Beneficiary", "Goal"],
            "xcomp": ["Event", "State", "Content"],
            "ccomp": ["Message", "Content", "Topic"],
            "advcl": ["Time", "Purpose", "Manner", "Condition"],
        }

        if dep_role in mappings:
            # Find first matching FE that exists in this frame
            for fe_candidate in mappings[dep_role]:
                if fe_candidate in frame.FE:
                    return fe_candidate

        return None

    def _map_pp_to_fe(self, pp_head: Token, frame,
                     pred: Token) -> Optional[str]:
        """Map prepositional phrase to frame element"""
        prep = pp_head.text.lower()

        # Preposition-based mappings
        prep_mappings = {
            "in": ["Place", "Location", "Time", "Manner"],
            "on": ["Place", "Topic", "Time"],
            "at": ["Place", "Location", "Time"],
            "to": ["Goal", "Recipient", "Destination"],
            "from": ["Source", "Origin"],
            "with": ["Instrument", "Manner", "Comitative"],
            "by": ["Agent", "Means", "Time"],
            "for": ["Purpose", "Beneficiary", "Duration"],
            "about": ["Topic", "Content"],
            "through": ["Path", "Means"],
            "during": ["Time", "Duration"],
        }

        if prep in prep_mappings:
            for fe_candidate in prep_mappings[prep]:
                if fe_candidate in frame.FE:
                    return fe_candidate

        return None

    def _get_fe_span(self, token: Token, doc: Doc) -> Optional[Span]:
        """Get the span for a frame element starting from a token"""
        # For noun phrases, get the full chunk
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk

        # For clauses, get the full subtree
        if token.dep_ in ["xcomp", "ccomp", "advcl"]:
            left = token.left_edge.i
            right = token.right_edge.i + 1
            return doc[left:right]

        # Default to token and its subtree
        left = token.left_edge.i
        right = token.right_edge.i + 1
        return doc[left:right]

    def _get_prepositional_phrases(self, verb: Token) -> List[Token]:
        """Get prepositional phrases attached to verb"""
        pps = []
        for child in verb.children:
            if child.dep_ == "prep":
                pps.append(child)
        return pps

    def _calculate_confidence(self, token: Token, fe: str,
                             frame) -> float:
        """Calculate confidence score for frame element assignment"""
        confidence = 0.5  # Base confidence

        # Boost for core frame elements
        if (frame.name, fe) in self.fe_coreness_cache:
            if self.fe_coreness_cache[(frame.name, fe)] == "Core":
                confidence += 0.2

        # Boost for clear syntactic mapping
        if token.dep_ in ["nsubj", "dobj", "obj"]:
            confidence += 0.15

        # Penalty for distant tokens
        if hasattr(token, 'head'):
            distance = abs(token.i - token.head.i)
            confidence -= min(0.1, distance * 0.02)

        return min(1.0, max(0.0, confidence))

    def _calculate_frame_confidence(self, elements: List[FrameElement]) -> float:
        """Calculate overall confidence for frame instance"""
        if not elements:
            return 0.3  # Low confidence if no elements found

        # Average element confidences with boost for core elements
        total_conf = 0.0
        total_weight = 0.0

        for elem in elements:
            weight = 2.0 if elem.fe_type == "Core" else 1.0
            total_conf += elem.confidence * weight
            total_weight += weight

        return total_conf / total_weight if total_weight > 0 else 0.0

    def get_frame_summary(self, doc: Doc) -> Dict:
        """
        Get a summary of all frames and elements in the document.

        Returns:
            Dictionary with frame statistics and details
        """
        summary = {
            "text": doc.text,
            "frames": [],
            "statistics": {
                "total_frames": len(doc._.frames),
                "total_elements": sum(len(f.elements) for f in doc._.frames),
                "predicates": []
            }
        }

        for frame_inst in doc._.frames:
            frame_data = {
                "frame": frame_inst.name,
                "predicate": frame_inst.target.text,
                "confidence": frame_inst.confidence,
                "elements": []
            }

            for element in frame_inst.elements:
                frame_data["elements"].append({
                    "role": element.name,
                    "text": element.span.text,
                    "type": element.fe_type,
                    "confidence": element.confidence
                })

            summary["frames"].append(frame_data)
            summary["statistics"]["predicates"].append(frame_inst.target.text)

        return summary

    def visualize_frames(self, doc: Doc) -> str:
        """
        Create a text-based visualization of frames and elements.

        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Text: {doc.text}")
        lines.append("=" * 80)

        for frame_inst in doc._.frames:
            lines.append(f"\nFrame: {frame_inst.name}")
            lines.append(f"  Predicate: '{frame_inst.target.text}' "
                        f"[{frame_inst.target.start_char}:{frame_inst.target.end_char}]")
            lines.append(f"  Confidence: {frame_inst.confidence:.2f}")
            lines.append(f"  Definition: {frame_inst.definition[:100]}...")

            if frame_inst.elements:
                lines.append("  Frame Elements:")
                for elem in sorted(frame_inst.elements,
                                 key=lambda x: x.span.start_char):
                    marker = "**" if elem.fe_type == "Core" else "  "
                    lines.append(f"    {marker}{elem.name}: '{elem.span.text}' "
                               f"(conf: {elem.confidence:.2f})")
            else:
                lines.append("  No frame elements identified")

        if not doc._.frames:
            lines.append("\nNo frames identified in the text.")

        lines.append("=" * 80)
        return "\n".join(lines)