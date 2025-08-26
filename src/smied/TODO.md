# FramenetSpacySRL Triple-Based Refactoring Plan

## Overview
Complete refactoring of `FramenetSpacySRL.py` to replace the span-labeling SRL approach with a triple-based SRL approach using the methodology from `new_srl_pipeline.py`. The refactored system will process syntactic triples (subject, predicate, object) and align them with WordNet and FrameNet semantic frames.

## Core Objectives
1. Replace span-based frame element extraction with triple-based processing
2. Maintain backward compatibility for public API methods
3. Focus on subject, object, and indirect object dependents from sentences
4. Use WordNet-FrameNet alignment as the core semantic mapping mechanism

## Implementation Steps

### Phase 1: Core Triple Processing Infrastructure

#### 1.1 Implement Helper Functions from new_srl_pipeline.py
```python
def _get_subject(self, pred_tok: Token) -> Optional[Token]:
    """Extract subject token from predicate using spaCy dependencies"""
    # Look for nsubj, nsubjpass dependencies
    
def _get_object(self, pred_tok: Token) -> Optional[Token]:
    """Extract direct object token from predicate"""
    # Look for dobj, obj dependencies
    
def _get_theme(self, pred_tok: Token) -> Optional[Token]:
    """Extract indirect object/theme token from predicate"""
    # Look for iobj, dative dependencies
    
def _lemmatize(self, word: str) -> str:
    """Get lemma form of word using spaCy or NLTK"""
    
def _align_wn_fn_frames(self, wn_frame: Any, fn_frame: Any) -> Dict[str, List[str]]:
    """Align WordNet frame with FrameNet frame"""
    # Return {"subjects": [...], "objects": [...], "themes": [...]}
```

#### 1.2 Implement process_triple as Core Engine
```python
def process_triple(self, pred_tok: Token, subj_tok: Optional[Token], 
                  obj_tok: Optional[Token], wn: Any, fn: Any) -> Dict[str, Dict[str, Set[str]]]:
    """
    Core triple processing logic from new_srl_pipeline.py
    Returns: Dict[synset_name, Dict[dependency_role, Set[semantic_roles]]]
    """
```

### Phase 2: Refactor Public API Methods

#### 2.1 Refactor __init__ Method
- Keep same parameters for backward compatibility
- Remove span-based caches (frame_cache, lexical_unit_cache, fe_coreness_cache)
- Add WordNet and FrameNet interfaces
- Initialize triple processing infrastructure

#### 2.2 Refactor process Method to process_doc
```python
def process_doc(self, doc: Doc) -> Doc:
    """
    Process document using triple-based approach
    1. Extract all predicates (verbs)
    2. For each predicate, extract subject/object/theme
    3. Process each triple through process_triple
    4. Convert results to FrameInstance format for compatibility
    """
```

#### 2.3 Maintain process_text Method
```python
def process_text(self, text: str) -> Optional[Doc]:
    """Wrapper that parses text and calls process_doc"""
```

#### 2.4 Transform get_frame_summary Method
```python
def get_frame_summary(self, doc: Doc) -> Dict:
    """
    Convert triple-based results to existing summary format
    Maintain same output structure for backward compatibility
    """
```

#### 2.5 Transform visualize_frames Method
```python
def visualize_frames(self, doc: Doc) -> str:
    """
    Convert triple-based results to text visualization
    Maintain same output format for backward compatibility
    """
```

### Phase 3: Remove Incompatible Methods

#### Methods to Remove Completely:
- `_identify_predicates()` - replaced by triple extraction
- `_expand_predicate_span()` - not needed for triple approach
- `_get_frames_for_predicate()` - replaced by process_triple
- `_score_frame_coherence()` - replaced by WordNet-FrameNet alignment
- `_select_best_frame()` - replaced by synset selection in process_triple
- `_score_syntactic_compatibility()` - not applicable to triple approach
- `_extract_frame_elements()` - replaced by triple processing
- `_map_dependent_to_fe()` - replaced by alignment logic
- `_map_pp_to_fe()` - not needed for triple approach
- `_get_fe_span()` - spans replaced by tokens
- `_get_prepositional_phrases()` - not used in triple approach
- `_is_animate_word()` - move to helper if needed
- `_is_concrete_word()` - move to helper if needed

#### Methods to Refactor as Private Helpers:
- `_build_frame_cache()` -> `_get_fn_frames_for_lemma()`
- `_build_lexical_unit_cache()` -> `_get_wn_synsets_for_lemma()`
- `_calculate_confidence()` -> `_calculate_triple_confidence()`
- `_calculate_frame_confidence()` -> `_calculate_synset_confidence()`

### Phase 4: Data Structure Transformations

#### 4.1 Triple Result to FrameInstance Conversion
```python
def _triple_to_frame_instance(self, pred_tok: Token, triple_results: Dict,
                              doc: Doc) -> FrameInstance:
    """
    Convert triple processing results to FrameInstance for compatibility
    Map synsets and semantic roles to frame and frame elements
    """
```

#### 4.2 Maintain SpaCy Extensions
- Keep Doc.frames and Doc.frame_elements extensions
- Remove Token.is_predicate (not meaningful in triple approach)
- Remove Span extensions (no longer using spans)

## Unit Test Requirements

### Test Suite 1: Triple Extraction Tests
```python
def test_get_subject():
    """Test subject extraction from various sentence structures"""
    # Test cases:
    # - Simple SVO: "John ate pizza"
    # - Passive: "Pizza was eaten by John"
    # - Complex: "The tall man quickly ate the pizza"
    # - No subject: "Eat the pizza!"
    
def test_get_object():
    """Test object extraction from various sentence structures"""
    # Test cases:
    # - Direct object: "John ate pizza"
    # - No object: "John sleeps"
    # - Multiple objects: "John gave Mary a book"
    
def test_get_theme():
    """Test indirect object extraction"""
    # Test cases:
    # - Ditransitive: "John gave Mary a book"
    # - Prepositional: "John gave a book to Mary"
    # - No theme: "John ate pizza"
```

### Test Suite 2: WordNet-FrameNet Alignment Tests
```python
def test_align_wn_fn_frames():
    """Test frame alignment logic"""
    # Test cases:
    # - Perfect alignment: matching argument structures
    # - Partial alignment: subset of arguments match
    # - No alignment: incompatible frames
    
def test_process_triple():
    """Test complete triple processing"""
    # Test cases:
    # - Verb with clear frame: "give" -> Transfer frame
    # - Ambiguous verb: "run" -> multiple possible frames
    # - Novel verb: verb not in WordNet/FrameNet
```

### Test Suite 3: API Compatibility Tests
```python
def test_process_doc_output_format():
    """Ensure process_doc returns same structure as before"""
    # Verify:
    # - Doc has .frames extension
    # - Each frame is FrameInstance object
    # - Frame elements are FrameElement objects
    
def test_get_frame_summary_format():
    """Ensure summary maintains same JSON structure"""
    # Verify all expected keys present
    # Verify data types match original
    
def test_visualize_frames_format():
    """Ensure visualization format unchanged"""
    # Compare output format with original
```

### Test Suite 4: Edge Cases and Error Handling
```python
def test_empty_input():
    """Test handling of empty or null inputs"""
    
def test_no_predicates():
    """Test sentences without verbs"""
    
def test_complex_dependencies():
    """Test complex syntactic structures"""
    # - Conjunctions: "John and Mary ate pizza"
    # - Relative clauses: "The man who ate pizza left"
    # - Nested structures: "John said Mary ate pizza"
    
def test_missing_resources():
    """Test graceful degradation when WordNet/FrameNet unavailable"""
```

### Test Suite 5: Performance and Integration Tests
```python
def test_spacy_integration():
    """Test integration as SpaCy pipeline component"""
    # Test adding to pipeline
    # Test processing through pipeline
    # Test interaction with other components
    
def test_performance_metrics():
    """Compare performance with original implementation"""
    # Measure processing time
    # Measure memory usage
    # Measure accuracy on gold standard data
```

## Validation Criteria

### Functional Requirements
1.  process_triple function fully implements new_srl_pipeline.py logic
2.  All public API methods return same output types as original
3.  Triple extraction correctly identifies subjects, objects, themes
4.  WordNet-FrameNet alignment produces valid semantic role mappings
5.  System handles edge cases gracefully (no subject, no object, etc.)

### Non-Functional Requirements
1.  Performance equal to or better than original implementation
2.  Memory footprint reduced (no large span caches)
3.  Code maintainability improved (clearer separation of concerns)
4.  Documentation updated to reflect triple-based approach

### Backward Compatibility
1.  Existing code using FrameNetSpaCySRL continues to work
2.  Output formats remain unchanged
3.  SpaCy pipeline integration preserved
4.  Configuration options maintained

## Implementation Priority

1. **High Priority (Core Functionality)**
   - Implement process_triple function
   - Implement helper functions (get_subject, get_object, get_theme)
   - Refactor process_doc method
   - Create triple-to-FrameInstance converter

2. **Medium Priority (API Compatibility)**
   - Refactor get_frame_summary
   - Refactor visualize_frames
   - Update __init__ method
   - Maintain SpaCy extensions

3. **Low Priority (Cleanup)**
   - Remove incompatible methods
   - Optimize performance
   - Add comprehensive logging
   - Update documentation

## Success Metrics

1. **Correctness**: Unit tests pass with >95% coverage
2. **Compatibility**: Existing code using the library continues to work
3. **Performance**: Processing time within 20% of original
4. **Maintainability**: Cyclomatic complexity reduced by >30%
5. **Documentation**: All public methods have updated docstrings

## Timeline Estimate

- Phase 1 (Core Infrastructure): 2-3 days
- Phase 2 (API Refactoring): 2-3 days
- Phase 3 (Method Removal): 1 day
- Phase 4 (Data Transformations): 1-2 days
- Testing & Validation: 2-3 days
- Documentation: 1 day

**Total Estimate**: 9-14 days

## Notes and Considerations

1. The triple-based approach fundamentally changes how semantic roles are identified, moving from span-labeling to dependency-based extraction
2. Some precision may be lost in complex sentences with multiple clauses
3. The system will be more focused on core argument structure (subject, object, indirect object) rather than peripheral elements
4. WordNet coverage may limit the frames that can be identified
5. Consider implementing a fallback mechanism for verbs not in WordNet/FrameNet