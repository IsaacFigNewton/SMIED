# SMIED Debugging TODO

## 🛠️ DIAGNOSTICS INFRASTRUCTURE SETUP (NEW)

### **5. Diagnostics Class Creation** (PRIORITY: HIGH)

**What's Needed:** Consolidate diagnostic tools into a proper SMIED module structure
**Where:** Create `src/smied/Diagnostics.py`

**Current State Analysis:**
- Diagnostic tools scattered in root directory files:
  - `diagnostic_tools.py` - Contains `SMIEDConnectivityAnalyzer` class (main diagnostic engine)
  - `quick_analysis.py` - Contains relation coverage audit functions 
  - `focused_analysis.py` - Contains derivational relations analysis
  - `validate_findings.py` - Contains automated validation scripts

**Implementation Tasks:**
- [ ] **Create Core Diagnostics Class** in `src/smied/Diagnostics.py`:
  - Move `SMIEDConnectivityAnalyzer` class from `diagnostic_tools.py`
  - Integrate analysis functions from `quick_analysis.py`, `focused_analysis.py`
  - Add validation methods from `validate_findings.py`
  - Maintain clean API with methods for each diagnostic category

- [ ] **Class Structure Design**:
  ```python
  class SMIEDDiagnostics:
      """Comprehensive diagnostic toolkit for SMIED semantic pathfinding analysis."""
      
      # Core Analysis Methods
      def analyze_synset_connectivity(self, synset1, synset2, max_hops=6)
      def audit_wordnet_relations(self)
      def test_parameter_sensitivity(self, test_cases)
      def validate_pathfinding_fixes(self)
      
      # Graph Analysis Methods  
      def analyze_graph_topology(self)
      def analyze_cross_pos_connectivity(self)
      def analyze_relation_density(self, synsets)
      
      # Algorithm Analysis Methods
      def analyze_beam_filtering(self, test_cases)
      def analyze_heuristic_effectiveness(self, test_cases)
      def profile_search_performance(self, test_cases)
  ```

- [ ] **API Design Considerations**:
  - Maintain backwards compatibility with existing notebook usage
  - Support both programmatic access and command-line interface
  - Include verbosity control and progress reporting
  - Provide structured output formats (JSON, dict, human-readable)

---

### **6. Diagnostics Unit Tests** (PRIORITY: HIGH)

**What's Needed:** Comprehensive test coverage for the new Diagnostics class
**Where:** Create `tests/test_diagnostics.py`

**Test Coverage Requirements:**
- [ ] **Core Functionality Tests**:
  - Test `analyze_synset_connectivity()` with known connected/disconnected synset pairs
  - Test `audit_wordnet_relations()` validates against expected relation sets
  - Test parameter sensitivity analysis produces expected parameter recommendations
  - Test validation methods correctly identify missing/present pathfinding capabilities

- [ ] **Integration Tests**:
  - Test integration with `SemanticDecomposer` for graph construction
  - Test integration with `PairwiseBidirectionalAStar` for pathfinding analysis
  - Test integration with `BeamBuilder` and `EmbeddingHelper` components
  - Verify diagnostic results match expected analysis outcomes

- [ ] **Performance Tests**:
  - Test diagnostic analysis completes within reasonable time bounds
  - Test memory usage stays within acceptable limits during large graph analysis
  - Test graceful handling of missing WordNet data or network issues

- [ ] **Mock and Fixture Setup**:
  - Create test fixtures with known graph structures and expected analysis results
  - Mock SMIED component interactions for isolated testing
  - Create synthetic test cases that exercise edge cases and error conditions

---

## 🚀 DETAILED SEMANTIC PATHFINDING IMPLEMENTATION ROADMAP (UPDATED)

*Based on comprehensive technical analysis from `SEMANTIC_PATHFINDING_ANALYSIS_REPORT.md` and `DELIVERABLES_SUMMARY.md`*

**NOTE: For now, only implement PHASE 1**
---

### **PHASE 1: CRITICAL PATH FIXES (Week 1)**

#### **Task 1.1: Integrate FrameNetSpaCySRL** (PRIORITY: CRITICAL - 80% Impact)

**Root Cause Confirmed:** `SemanticDecomposer.build_synset_graph()` missing `FrameNetSpaCySRL(use_wordnet_expansion=True)` - the PRIMARY cause of cross-POS pathfinding failures.

**DETAILED IMPLEMENTATION PLAN:**

**A. Architecture Integration Strategy:**

The FrameNetSpaCySRL class provides superior semantic role labeling compared to GlossParser by:
- Using FrameNet's rich semantic frame database to identify semantic relationships
- Mapping frame elements (Agent, Patient, Theme, Instrument, etc.) to syntactic constituents  
- Supporting WordNet expansion for broader coverage
- Identifying cross-POS connections through frame-based semantics

**B. Specific Integration Points in SemanticDecomposer:**

1. **Initialize FrameNetSpaCySRL alongside GlossParser** (line ~61):
   ```python
   # Add after GlossParser initialization
   self.frame_srl = FrameNetSpaCySRL(
       nlp=nlp_func,
       use_wordnet_expansion=True,
       min_confidence=0.5
   )
   ```

2. **Enhance `_find_subject_to_predicate_paths()` method** (lines 251-346):
   ```python
   # Strategy 0: Use FrameNet SRL for frame-based connections (NEW - HIGHEST PRIORITY)
   pred_definition = predicate_synset.definition()
   frame_doc = self.frame_srl.process_text(pred_definition)
   
   # Extract frame instances and their elements
   for frame_inst in frame_doc._.frames:
       # Look for Agent/Experiencer frame elements that could match subjects
       agent_elements = [elem for elem in frame_inst.elements 
                        if elem.name in ["Agent", "Experiencer", "Theme", "Cognizer"]]
       
       # Convert frame element spans to synsets and match with subject_synsets
       for elem in agent_elements:
           elem_synsets = self._extract_synsets_from_span(elem.span)
           matched_synsets = self._match_synsets_with_targets(elem_synsets, subject_synsets)
           
           # Build paths from subjects through frame connections to predicate
           for subj_synset in subject_synsets:
               for matched_synset in matched_synsets:
                   # Use frame semantic connection as bridge
                   path = self._build_frame_bridged_path(
                       subj_synset, matched_synset, predicate_synset, frame_inst
                   )
                   if path:
                       subject_paths.append(path)
   ```

3. **Enhance `_find_predicate_to_object_paths()` method** (lines 348-436):
   ```python
   # Strategy 0: Use FrameNet SRL for frame-based connections (NEW - HIGHEST PRIORITY)
   pred_definition = predicate_synset.definition()
   frame_doc = self.frame_srl.process_text(pred_definition)
   
   for frame_inst in frame_doc._.frames:
       # Look for Patient/Theme/Goal frame elements that could match objects
       patient_elements = [elem for elem in frame_inst.elements
                          if elem.name in ["Patient", "Theme", "Goal", "Stimulus", "Content"]]
       
       # Convert frame element spans to synsets and match with object_synsets
       for elem in patient_elements:
           elem_synsets = self._extract_synsets_from_span(elem.span)
           matched_synsets = self._match_synsets_with_targets(elem_synsets, object_synsets)
           
           # Build paths from predicate through frame connections to objects
           for matched_synset in matched_synsets:
               for obj_synset in object_synsets:
                   path = self._build_frame_bridged_path(
                       predicate_synset, matched_synset, obj_synset, frame_inst
                   )
                   if path:
                       object_paths.append(path)
   ```

4. **Add new helper methods for frame-based pathfinding**:
   ```python
   def _extract_synsets_from_span(self, span):
       """Extract WordNet synsets from a FrameNet span."""
       # Use span text to get synsets
       lemma = span.root.lemma_
       pos_map = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ}
       if span.root.pos_ in pos_map:
           return wn.synsets(lemma, pos=pos_map[span.root.pos_])
       return []
   
   def _match_synsets_with_targets(self, candidate_synsets, target_synsets):
       """Find best matching synsets between candidates and targets."""
       matches = []
       for candidate in candidate_synsets:
           for target in target_synsets:
               similarity = candidate.path_similarity(target)
               if similarity and similarity > 0.3:  # Threshold for relevance
                   matches.append((candidate, target, similarity))
       # Return top matches sorted by similarity
       matches.sort(key=lambda x: x[2], reverse=True)
       return [m[0] for m in matches[:3]]  # Top 3 matches
   
   def _build_frame_bridged_path(self, src_synset, bridge_synset, tgt_synset, frame_inst):
       """Build a semantic path using frame as conceptual bridge."""
       # Create virtual frame synset node for graph
       frame_node = f"frame:{frame_inst.name}"
       
       # Build path: src -> frame -> bridge -> tgt
       # This creates explicit cross-POS connections through semantic frames
       path = [src_synset, frame_node, bridge_synset, tgt_synset]
       return path
   ```

5. **Enhance `build_synset_graph()` to include frame-based edges** (lines 572-663):
   ```python
   # Add frame-based semantic edges (NEW)
   if self.verbosity >= 1:
       print("[build_synset_graph] Adding frame-based semantic connections...")
   
   # Process common predicate frames to add cross-POS edges
   common_predicates = ['chase', 'hunt', 'eat', 'catch', 'follow', 'pursue']
   for pred_lemma in common_predicates:
       pred_synsets = wn.synsets(pred_lemma, pos=wn.VERB)
       for pred_synset in pred_synsets[:2]:  # Top 2 senses
           # Use FrameNet to find semantic connections
           pred_def = pred_synset.definition()
           frame_doc = self.frame_srl.process_text(pred_def)
           
           for frame_inst in frame_doc._.frames:
               # Add edges from agent-like nouns to predicate
               agent_nouns = self._get_frame_agent_nouns(frame_inst)
               for agent_noun in agent_nouns:
                   if agent_noun.name() in g:
                       g.add_edge(agent_noun.name(), pred_synset.name(),
                                relation='frame_agent', weight=0.8)
               
               # Add edges from predicate to patient-like nouns  
               patient_nouns = self._get_frame_patient_nouns(frame_inst)
               for patient_noun in patient_nouns:
                   if patient_noun.name() in g:
                       g.add_edge(pred_synset.name(), patient_noun.name(),
                                relation='frame_patient', weight=0.8)
   ```

**C. Fallback Strategy - Replace GlossParser with Derivational Relations:**

**ENHANCED IMPLEMENTATION: Use WordNet's derivationally_related_forms() instead of GlossParser**

The WordNet derivationally_related_forms() method provides superior cross-POS connections compared to GlossParser by leveraging morphological relationships between words. This creates direct, reliable bridges between nouns and verbs without dependency parsing.

**Implementation Details:**

1. **Replace GlossParser fallback with derivational relations**:
   ```python
   def _get_derivational_connections(self, synset):
       """Get cross-POS synsets through derivational relationships."""
       derived_synsets = []
       
       # Get all lemmas for this synset
       for lemma in synset.lemmas():
           # Get derivationally related forms (cross-POS connections)
           for derived_lemma in lemma.derivationally_related_forms():
               # Add the synset of the derived form
               derived_synsets.append(derived_lemma.synset())
       
       return derived_synsets
   ```

2. **Enhanced cascading strategy**:
   1. **Primary**: Try FrameNetSpaCySRL first (semantic frame connections)
   2. **Secondary**: Use derivationally_related_forms() for morphological connections
   3. **Tertiary**: Hypernym/hyponym exploration for taxonomic relationships
   4. **Quaternary**: GlossParser only as last resort (if kept for backwards compatibility)

3. **Integration in _find_subject_to_predicate_paths()**:
   ```python
   # Strategy 1: Derivational relations (REPLACES GlossParser as primary fallback)
   if not subject_paths:  # Only if FrameNet didn't find paths
       if self.verbosity >= 2:
           print("[_find_subject_to_predicate_paths] Using derivational relations...")
       
       # For each predicate, find derivationally related nouns
       derived_nouns = self._get_derivational_connections(predicate_synset)
       derived_nouns = [s for s in derived_nouns if s.pos() == 'n']  # Filter nouns
       
       # Match derived nouns with subject synsets
       for subj_synset in subject_synsets:
           for derived_noun in derived_nouns:
               # Check if derived noun is semantically close to subject
               similarity = subj_synset.path_similarity(derived_noun)
               if similarity and similarity > 0.3:
                   # Build path: subject -> derived_noun -> predicate
                   path = [subj_synset, derived_noun, predicate_synset]
                   subject_paths.append(path)
                   if self.verbosity >= 2:
                       print(f"[Derivational] Found: {subj_synset.name()} -> {derived_noun.name()} -> {predicate_synset.name()}")
   ```

4. **Integration in _find_predicate_to_object_paths()**:
   ```python
   # Strategy 1: Derivational relations (REPLACES GlossParser as primary fallback)
   if not object_paths:  # Only if FrameNet didn't find paths
       if self.verbosity >= 2:
           print("[_find_predicate_to_object_paths] Using derivational relations...")
       
       # For each predicate, find derivationally related nouns
       derived_nouns = self._get_derivational_connections(predicate_synset)
       derived_nouns = [s for s in derived_nouns if s.pos() == 'n']  # Filter nouns
       
       # Match derived nouns with object synsets
       for derived_noun in derived_nouns:
           for obj_synset in object_synsets:
               similarity = derived_noun.path_similarity(obj_synset)
               if similarity and similarity > 0.3:
                   # Build path: predicate -> derived_noun -> object
                   path = [predicate_synset, derived_noun, obj_synset]
                   object_paths.append(path)
                   if self.verbosity >= 2:
                       print(f"[Derivational] Found: {predicate_synset.name()} -> {derived_noun.name()} -> {obj_synset.name()}")
   ```

5. **Enhance build_synset_graph() with derivational edges**:
   ```python
   # Add derivational relation edges for cross-POS connectivity
   if self.verbosity >= 1:
       print("[build_synset_graph] Adding derivational relation edges...")
   
   derivational_edge_count = 0
   for synset in all_synsets:
       synset_name = synset.name()
       
       # Get derivationally related forms
       for lemma in synset.lemmas():
           for derived_lemma in lemma.derivationally_related_forms():
               derived_synset_name = derived_lemma.synset().name()
               
               if derived_synset_name in g:
                   # Add bidirectional edges for derivational relations
                   g.add_edge(synset_name, derived_synset_name,
                            relation='derivational', weight=0.7)
                   g.add_edge(derived_synset_name, synset_name,
                            relation='derivational_reverse', weight=0.7)
                   derivational_edge_count += 2
   
   if self.verbosity >= 1:
       print(f"[build_synset_graph] Added {derivational_edge_count} derivational edges")
   ```

**Key Advantages of derivationally_related_forms() over GlossParser:**

1. **Direct Cross-POS Connections**: Creates explicit morphological bridges (hunt.v ↔ hunter.n, hunting.n)
2. **No NLP Processing Required**: Uses WordNet's pre-computed derivational relationships
3. **Consistent and Reliable**: Based on linguistic morphology, not variable gloss text
4. **Bidirectional**: Works both noun→verb and verb→noun directions equally well
5. **Performance**: Faster than dependency parsing, no spaCy processing needed
6. **Coverage**: Captures agent nouns (runner, hunter), gerunds (running, hunting), and result nouns

**Example Derivational Connections:**
- chase.v.01 ↔ chaser.n.01 (agent noun)
- hunt.v.01 ↔ hunter.n.01, hunting.n.01 (agent + gerund)
- eat.v.01 ↔ eater.n.01, eating.n.01 
- catch.v.01 ↔ catcher.n.01, catch.n.01
- follow.v.01 ↔ follower.n.01, following.n.01

**Migration Strategy:**
- Phase 1: Add derivational relations alongside GlossParser
- Phase 2: Prioritize derivational relations over GlossParser
- Phase 3: Deprecate GlossParser, keep only for legacy compatibility
- Phase 4: Remove GlossParser entirely in next major version

**D. Key Advantages of FrameNetSpaCySRL over GlossParser:**

1. **Richer Semantic Roles**: FrameNet provides ~1,200 frames with detailed semantic roles vs. basic dependency parsing
2. **Cross-POS Awareness**: Frames inherently connect nouns (participants) with verbs (events/actions)
3. **Conceptual Bridges**: Frames act as semantic bridges (e.g., "Hunting" frame connects hunter→hunt→prey)
4. **WordNet Integration**: Built-in WordNet expansion finds more synonym connections
5. **Confidence Scoring**: Frame confidence helps prioritize high-quality connections

**Technical Requirements:**
- [ ] **Code Implementation** in `src/smied/SemanticDecomposer.py`:
  - Add FrameNetSpaCySRL initialization in `__init__` method
  - Integrate frame-based strategies in path-finding methods
  - Add helper methods for frame element extraction and matching
  - Implement `_get_derivational_connections()` method for cross-POS morphological bridges
  - Enhance graph building with both frame-based and derivational edges
  - Replace GlossParser usage with derivationally_related_forms() as primary fallback
  - Keep GlossParser only for legacy compatibility (deprecation candidate)

- [ ] **Validation Testing**:
  - Test pathfinding on critical failing cases: `cat.n.01` → `chase.v.01` → `mouse.n.01`
  - Verify SRL connections for `chase.v.01` are now accessible in graph
  - Test derivational connections: verify `chase.v.01` ↔ `chaser.n.01`, `hunt.v.01` ↔ `hunter.n.01`
  - Validate cascading fallback: FrameNet → Derivational → Hypernym → GlossParser
  - Measure success rate improvement (expect 80%+ improvement with combined approaches)
  - Confirm path lengths remain ≤ 6 hops for conceptually related synsets
  - Test frame identification for common predicates (chase, hunt, catch, eat)
  - Validate frame element to synset mapping accuracy
  - Compare derivational relations coverage vs. GlossParser accuracy

- [ ] **Performance Impact Analysis**:
  - Benchmark graph construction time before/after SRL-based predicate relations
  - Measure derivational edge addition time (expect <5 seconds for full WordNet)
  - Profile memory usage: FrameNet data (~20% increase) + derivational edges (~10% increase)
  - Compare processing times: FrameNetSpaCySRL vs derivational vs GlossParser
  - Derivational relations expected 10x faster than GlossParser (no NLP processing)
  - Monitor combined memory footprint of frame cache + derivational edge storage
  - Profile pathfinding speed with expanded graph (acceptable degradation <50%)

---

#### **Task 1.1b: Testing and GlossParser Removal** (PRIORITY: HIGH)

**Comprehensive Testing and Cleanup for FrameNetSpaCySRL Integration**

**A. Unit Tests for FrameNetSpaCySRL:**

1. **Create `tests/test_framenet_srl.py`**:
   ```python
   import pytest
   import spacy
   from src.smied.FrameNetSpaCySRL import FrameNetSpaCySRL
   
   class TestFrameNetSpaCySRL:
       def test_frame_identification(self):
           """Test that common predicates evoke expected frames."""
           srl = FrameNetSpaCySRL(use_wordnet_expansion=True)
           
           test_cases = [
               ("The cat chased the mouse", "chase", ["Cotheme", "Pursuing"]),
               ("The hunter caught the rabbit", "catch", ["Manipulation", "Getting"]),
               ("The dog ate the food", "eat", ["Ingestion"]),
           ]
           
           for text, predicate, expected_frames in test_cases:
               doc = srl.process_text(text)
               assert doc._.frames, f"No frames found for '{text}'"
               frame_names = [f.name for f in doc._.frames]
               assert any(frame in frame_names for frame in expected_frames)
       
       def test_frame_element_extraction(self):
           """Test extraction of semantic roles from frames."""
           srl = FrameNetSpaCySRL()
           doc = srl.process_text("The cat chased the mouse quickly")
           
           for frame in doc._.frames:
               elements = {e.name: e.span.text for e in frame.elements}
               # Verify Agent/Theme roles are identified
               assert any(role in elements for role in ["Agent", "Theme", "Experiencer"])
       
       def test_cross_pos_connections(self):
           """Test that frames create cross-POS bridges."""
           srl = FrameNetSpaCySRL()
           
           # Test verb definitions that should connect to nouns
           verb_glosses = [
               "to pursue in order to catch",  # chase.v
               "to seek out and kill for food",  # hunt.v
           ]
           
           for gloss in verb_glosses:
               doc = srl.process_text(gloss)
               # Should identify noun participants in verb definitions
               assert any(elem.span.root.pos_ == "NOUN" 
                         for frame in doc._.frames 
                         for elem in frame.elements)
   ```

2. **Create `tests/test_derivational_relations.py`**:
   ```python
   import pytest
   from nltk.corpus import wordnet as wn
   
   class TestDerivationalRelations:
       def test_verb_to_noun_derivations(self):
           """Test derivational connections from verbs to nouns."""
           test_pairs = [
               ('chase.v.01', ['chaser.n.01', 'chase.n.01']),
               ('hunt.v.01', ['hunter.n.01', 'hunting.n.01']),
               ('eat.v.01', ['eater.n.01', 'eating.n.01']),
           ]
           
           for verb_name, expected_nouns in test_pairs:
               verb_synset = wn.synset(verb_name)
               derived = []
               
               for lemma in verb_synset.lemmas():
                   for derived_lemma in lemma.derivationally_related_forms():
                       if derived_lemma.synset().pos() == 'n':
                           derived.append(derived_lemma.synset().name())
               
               # Check at least one expected noun is found
               assert any(noun in derived for noun in expected_nouns), \
                   f"No expected derivations found for {verb_name}"
       
       def test_noun_to_verb_derivations(self):
           """Test derivational connections from nouns to verbs."""
           test_pairs = [
               ('hunter.n.01', 'hunt.v'),
               ('runner.n.01', 'run.v'),
           ]
           
           for noun_name, expected_verb_lemma in test_pairs:
               noun_synset = wn.synset(noun_name)
               derived_verbs = []
               
               for lemma in noun_synset.lemmas():
                   for derived_lemma in lemma.derivationally_related_forms():
                       if derived_lemma.synset().pos() == 'v':
                           derived_verbs.append(derived_lemma.name())
               
               assert any(expected_verb_lemma in verb for verb in derived_verbs)
   ```

**B. Update Existing Tests:**

1. **Modify `tests/test_semantic_decomposer.py`**:
   ```python
   # Update imports to remove GlossParser
   from src.smied.SemanticDecomposer import SemanticDecomposer
   from src.smied.FrameNetSpaCySRL import FrameNetSpaCySRL
   
   class TestSemanticDecomposer:
       def test_framenet_integration(self):
           """Test that SemanticDecomposer uses FrameNetSpaCySRL."""
           decomposer = SemanticDecomposer(wn, nlp, embedding_model=None)
           
           # Verify FrameNetSpaCySRL is initialized
           assert hasattr(decomposer, 'frame_srl')
           assert isinstance(decomposer.frame_srl, FrameNetSpaCySRL)
           
           # Test pathfinding with frame-based connections
           subject_path, object_path, predicate = decomposer.find_connected_shortest_paths(
               "cat", "chase", "mouse"
           )
           assert subject_path is not None, "Failed to find cat->chase path"
           assert object_path is not None, "Failed to find chase->mouse path"
       
       def test_derivational_fallback(self):
           """Test derivational relations as fallback."""
           decomposer = SemanticDecomposer(wn, nlp, embedding_model=None)
           
           # Mock scenario where FrameNet fails
           with mock.patch.object(decomposer.frame_srl, 'process_text', 
                                 return_value=mock_empty_doc):
               # Should fall back to derivational relations
               paths = decomposer._find_subject_to_predicate_paths(
                   [wn.synset('hunter.n.01')],
                   wn.synset('hunt.v.01'),
                   g, None, 3, 10, False, 3, 1, 5
               )
               assert paths, "Derivational fallback failed"
       
       def test_cascading_strategy(self):
           """Test the complete cascading fallback strategy."""
           # Test order: FrameNet -> Derivational -> Hypernym -> (GlossParser removed)
           pass  # Implement cascading test
   ```

2. **Update `tests/test_integration.py`**:
   ```python
   # Remove all GlossParser references
   # Update integration tests to use new pipeline
   ```

**C. GlossParser Removal Tasks:**

1. **Code Removal Checklist**:
   - [ ] Remove `from .GlossParser import GlossParser` from SemanticDecomposer.py
   - [ ] Remove `self.gloss_parser = GlossParser(nlp_func=nlp_func)` initialization
   - [ ] Replace all `self.gloss_parser.parse_gloss()` calls with derivational methods
   - [ ] Remove GlossParser-specific helper methods if no longer needed
   - [ ] Update docstrings to remove GlossParser references

2. **File Cleanup**:
   - [ ] Mark `src/smied/GlossParser.py` as deprecated with warning message
   - [ ] Create migration guide in `docs/migration_from_glossparser.md`
   - [ ] Update README.md to reflect new architecture
   - [ ] Remove GlossParser from requirements if it had specific dependencies

3. **Backwards Compatibility** (Optional - Phase 1 only):
   ```python
   # Add deprecation warning in GlossParser.__init__
   import warnings
   
   class GlossParser:
       def __init__(self, nlp_func=None):
           warnings.warn(
               "GlossParser is deprecated and will be removed in v2.0. "
               "Use FrameNetSpaCySRL or derivationally_related_forms() instead.",
               DeprecationWarning,
               stacklevel=2
           )
           self.nlp_func = nlp_func
   ```

**D. Performance Benchmarking Suite:**

1. **Create `tests/benchmarks/test_performance.py`**:
   ```python
   import time
   import memory_profiler
   
   class BenchmarkCrossPOSMethods:
       def benchmark_framenet_srl(self):
           """Benchmark FrameNetSpaCySRL performance."""
           # Time frame identification
           # Memory usage for frame cache
           # Success rate on test cases
       
       def benchmark_derivational_relations(self):
           """Benchmark derivational relation extraction."""
           # Time to extract relations
           # Memory for storing edges
           # Coverage statistics
       
       def benchmark_glossparser_legacy(self):
           """Benchmark GlossParser for comparison."""
           # Time for dependency parsing
           # Memory usage
           # Accuracy comparison
       
       def compare_methods(self):
           """Generate comparison report."""
           # Speed: Derivational (fastest) > FrameNet > GlossParser
           # Accuracy: FrameNet > Derivational > GlossParser
           # Memory: GlossParser (lowest) < Derivational < FrameNet
   ```

**E. Validation Test Suite:**

- [ ] **Cross-POS Gold Standard Tests**:
  ```python
  GOLD_STANDARD_PATHS = [
      ("cat.n.01", "chase.v.01", "mouse.n.01"),
      ("hunter.n.01", "hunt.v.01", "prey.n.01"),
      ("dog.n.01", "bark.v.01", "intruder.n.01"),
      ("bird.n.01", "fly.v.01", "nest.n.01"),
      ("fish.n.01", "swim.v.01", "ocean.n.01"),
  ]
  ```

- [ ] **Regression Tests**:
  - Ensure existing functionality not broken
  - Verify path quality remains high
  - Check that path lengths stay reasonable

- [ ] **Edge Case Tests**:
  - Synsets with no derivational forms
  - Frames not in FrameNet
  - Circular derivational relationships
  - Missing WordNet connections

**Technical Requirements:**
- [ ] Write comprehensive unit tests for FrameNetSpaCySRL
- [ ] Create tests for derivational relation extraction
- [ ] Update all existing tests to remove GlossParser dependencies
- [ ] Implement performance benchmarking suite
- [ ] Add integration tests for cascading fallback strategy
- [ ] Create migration tests to ensure smooth transition
- [ ] Document test coverage requirements (aim for >90%)
- [ ] Set up CI/CD to run all tests automatically

---

#### **Task 1.2: Algorithm Parameter Optimization** (PRIORITY: HIGH - 20% Impact)

**Root Cause Confirmed:** `PairwiseBidirectionalAStar` parameters too restrictive for cross-conceptual pathfinding.

**Technical Requirements:**
- [ ] **Parameter Tuning** in `src/smied/PairwiseBidirectionalAStar.py`:
  ```python
  # Current restrictive defaults (lines 41-43)
  beam_width: 3        → 10     # Expand search space significantly  
  max_depth: 6         → 10     # Allow slightly longer paths
  relax_beam: False    → True   # Bypass embedding-based beam constraints initially
  len_tolerance: var   → 3      # Accept longer but semantically valid paths
  ```

- [ ] **Parameter Sensitivity Analysis**:
  - Test parameter combinations systematically on failing test cases
  - Identify optimal parameter set balancing success rate vs. performance
  - Document parameter impact on search space size and algorithm termination
  - Create parameter selection guidelines for different semantic domains

- [ ] **Validation Protocol**:
  - Test on gold standard cross-POS semantic cases (noun→verb→noun paths)
  - Measure success rate improvement with parameter changes alone
  - Benchmark performance impact of increased search space
  - Validate that relaxed parameters don't introduce false positive paths

---

### **PHASE 2: HEURISTIC AND ALGORITHMIC IMPROVEMENTS (Week 2)**

#### **Task 2.1: Heuristic Function Enhancement** (PRIORITY: MEDIUM - 15% Impact)

**Root Cause Confirmed:** Embedding-based heuristics conflict with WordNet semantic structure.

**Technical Requirements:**
- [ ] **Hybrid Heuristic Implementation**:
  ```python
  # New heuristic combining embedding and WordNet distance
  def hybrid_heuristic(self, current, target):
      embedding_sim = self.embedding_helper.get_similarity(current, target)
      wordnet_distance = self.estimate_wordnet_hops(current, target)
      cross_pos_penalty = 0.2 if different_pos(current, target) else 0.0
      
      # Balanced combination
      h = (1 - embedding_sim) * 0.7 + wordnet_distance * 0.3 + cross_pos_penalty
      return h
  ```

- [ ] **WordNet-Distance Estimator**:
  - Implement fast WordNet-based distance estimation for heuristic
  - Use hypernym/hyponym hierarchy depth as distance proxy
  - Add cross-POS bridge awareness to distance calculation
  - Benchmark heuristic quality vs. embedding-only approach

- [ ] **Alternative Heuristic Testing**:
  - Test uniform cost search (h=0) to establish baseline without heuristic bias
  - Test pure WordNet-distance heuristic vs. pure embedding heuristic
  - Analyze heuristic effectiveness on different semantic relationship types
  - Document optimal heuristic selection criteria

---

#### **Task 2.2: Missing Relations Integration** (PRIORITY: MEDIUM - 10% Impact)

**Technical Requirements:**
- [ ] **Additional WordNet Relations** in `SemanticDecomposer.py`:
  ```python
  # Add to relation processing loop
  attributes_rels = synset.attributes()
  for attr_synset in attributes_rels:
      # Add noun-adjective connections
      
  # Add lemma-level antonym relations
  for lemma in synset.lemmas():
      antonyms = lemma.antonyms()
      for antonym_lemma in antonyms:
          # Add semantic opposition connections
  ```

- [ ] **Relation Impact Analysis**:
  - Measure connectivity improvement from `attributes()` relations
  - Analyze antonym relation utility for semantic pathfinding
  - Profile performance impact of additional relation types
  - Document relation type priority for graph construction

---

### **PHASE 3: TESTING AND VALIDATION INFRASTRUCTURE (Week 3)**

#### **Task 3.1: Comprehensive Test Suite Creation** (PRIORITY: HIGH)

**Technical Requirements:**
- [ ] **Gold Standard Test Cases** in `tests/test_semantic_pathfinding.py`:
  ```python
  # Cross-POS pathfinding test cases  
  CROSS_POS_TEST_CASES = [
      ("cat.n.01", "chase.v.01", "Expected via predatory behavior"),
      ("chase.v.01", "mouse.n.01", "Expected via prey/object relationship"),
      ("hunt.v.01", "prey.n.01", "Expected via direct semantic connection"),
      # ... 20+ manually curated cases
  ]
  ```

- [ ] **Automated Success Rate Measurement**:
  - Implement test harness measuring pathfinding success rates
  - Create performance benchmarking for graph construction and search
  - Add regression testing for existing functionality
  - Document test case rationale and expected path characteristics

- [ ] **Comparative Analysis Framework**:
  - Compare SMIED pathfinding results with ConceptNet.io
  - Human evaluation protocol for path quality assessment  
  - False positive/negative analysis for pathfinding quality
  - Success metric definition and measurement automation

---

#### **Task 3.2: Performance and Scalability Analysis** (PRIORITY: MEDIUM)

**Technical Requirements:**
- [ ] **Performance Profiling Suite**:
  - Memory usage profiling for large WordNet graph construction
  - Search algorithm runtime analysis across different parameter settings
  - Bottleneck identification in pathfinding pipeline components
  - Scalability analysis for graphs with 10K+ synsets

- [ ] **Optimization Implementation**:
  - Graph caching mechanisms for repeated pathfinding queries
  - Precomputed distance matrices for common concept pairs
  - Index-based fast lookup for high-frequency semantic relationships
  - Memory optimization for large-scale deployment

---

### **PHASE 4: ADVANCED FEATURES AND ROBUSTNESS (Week 4)**

#### **Task 4.1: Fallback Pathfinding Strategies** (PRIORITY: LOW - 5% Impact)

**Technical Requirements:**
- [ ] **Pure WordNet Traversal Backup**:
  - Implement breadth-first search without embedding constraints
  - Prioritize high-value semantic relations (hypernyms, entailments, causes)
  - Create relation-specific cost functions for path quality optimization
  - Add domain-specific connection strategies (animal behavior patterns, etc.)

- [ ] **Multi-Stage Search Architecture**:
  - Stage 1: Shared hypernym pathfinding
  - Stage 2: Conceptual bridge discovery (behavior, attributes)
  - Stage 3: Lexical derivation pathfinding
  - Integrate stages with confidence scoring and fallback logic

---

## **📊 SUCCESS METRICS AND VALIDATION CRITERIA**

### **Quantitative Targets:**
- [ ] **Primary Success**: `cat.n.01 → chase.v.01 → mouse.n.01` produces semantically valid paths ≤ 6 hops
- [ ] **Success Rate**: 90%+ success on manually curated cross-POS semantic pathfinding test suite  
- [ ] **Performance**: <50% degradation from baseline pathfinding performance
- [ ] **Graph Enhancement**: 80%+ increase in cross-POS connectivity after derivational relations

### **Qualitative Targets:**
- [ ] **Path Quality**: Human evaluation confirms semantic reasonableness of discovered paths
- [ ] **System Robustness**: Graceful degradation when optimal paths unavailable
- [ ] **API Consistency**: Diagnostic tools maintain backward compatibility with existing usage
- [ ] **Documentation**: Complete technical documentation for all implemented fixes

---

## **⚠️ RISK MITIGATION STRATEGIES**

### **Technical Risks:**
- **Graph Construction Performance**: Monitor memory usage and construction time with additional relations
- **Algorithm Complexity**: Validate that parameter relaxation doesn't create infinite search spaces
- **Heuristic Conflicts**: Ensure hybrid heuristics don't create local optima or poor-quality paths

### **Mitigation Approaches:**
- **Incremental Implementation**: Deploy changes incrementally with rollback capability
- **Comprehensive Testing**: Maintain full regression test coverage throughout implementation
- **Performance Monitoring**: Continuous benchmarking before/after each change
- **A/B Testing**: Compare new vs. old pathfinding on representative test cases

---

## **🎯 IMPLEMENTATION PRIORITY SEQUENCE**

1. **Week 1**: Task 1.1 (Derivational Relations) + Task 1.2 (Parameter Optimization)
2. **Week 2**: Task 2.1 (Heuristic Enhancement) + Task 2.2 (Missing Relations)  
3. **Week 3**: Task 3.1 (Test Suite) + Task 3.2 (Performance Analysis)
4. **Week 4**: Task 4.1 (Advanced Features) + Documentation + Final Validation

**Expected Total Effort**: 4 weeks for complete implementation with comprehensive testing and validation.

**Expected Impact**: 80%+ improvement in cross-conceptual pathfinding success rate, enabling robust semantic decomposition for subject-predicate-object triples in natural language processing tasks.

---

## 🧪 TEST VALIDATION STRATEGY

**For each fix:**
1. Run specific failing test in isolation
2. Add unit test for the specific bug scenario  
3. Run full test suite to ensure no regressions
4. Verify fix handles edge cases (nested structures, empty lists, etc.)