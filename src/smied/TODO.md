# SMIED Debugging TODO

## 🔍 CRITICAL BUGS TO FIX

### 1. **PatternLoader JSON-to-Set Conversion Bug** (PRIORITY: HIGH)
**What's Breaking:** `json_to_pattern()` method fails to convert lists to sets for specific keys
**Where:** `src/smied/PatternLoader.py:json_to_pattern()`

**Failed Assertions:**
- `pattern[1]["relation"]` remains `['subject']` instead of `{'subject'}`
- `pattern["text"]` remains `['surface_text']` instead of `{'surface_text'}`
- All POS tags (`pos`), dependency relations (`dep`), entity types (`ent_type`) not converting

**Files to Check:**
- `src/smied/PatternLoader.py` - Core logic for list→set conversion
- `tests/test_pattern_loader.py` - Test expectations vs implementation behavior

**Bug Hypothesis:**
- `json_to_pattern()` method missing key-specific conversion logic
- Method may only convert certain keys (like `pos`) but not others (`relation`, `text`, `dep`, `ent_type`)
- Nested pattern traversal not working correctly

---

### 2. **Test Setup Bug** (PRIORITY: MEDIUM)  
**What's Breaking:** Missing test fixture attribute
**Where:** `tests/test_pattern_loader.py:373`

**Failed Assertion:**
- `AttributeError: 'TestPatternLoaderEdgeCases' object has no attribute 'sample_patterns'`

**Files to Check:**
- `tests/test_pattern_loader.py:TestPatternLoaderEdgeCases` class
- Look for missing `setUp()` method or `sample_patterns` attribute initialization

**Bug Hypothesis:**
- Test class missing proper initialization of `sample_patterns` fixture
- `setUp()` method not called or incomplete

---

### 3. **PatternMatcher Output Format Mismatch** (PRIORITY: MEDIUM)
**What's Breaking:** `get_pattern_summary()` returns different structure than expected
**Where:** `src/smied/PatternMatcher.py:get_pattern_summary()`

**Failed Assertion:**
- Expected: `{"test_category": {"noun_pattern": [[0], [1]], "verb_pattern": []}}`
- Actual: `{"test_category": {"noun_pattern": [{"indices": [0], "metaverts": [...]}]}}`

**Files to Check:**
- `src/smied/PatternMatcher.py:get_pattern_summary()` method
- `tests/test_pattern_matcher.py` - Check mock expectations vs actual method behavior

**Bug Hypothesis:**
- Method returns enriched objects instead of simple index lists
- Test mock expectations don't match actual implementation behavior
- Return format changed but tests weren't updated

---

## 🎯 DEBUGGING VALIDATION CHECKLIST

### For PatternLoader JSON Conversion:
- [ ] Verify `json_to_pattern()` implementation in `PatternLoader.py`
- [ ] Check which keys are configured for list→set conversion
- [ ] Test nested pattern handling logic
- [ ] Confirm conversion applies to: `pos`, `dep`, `relation`, `text`, `ent_type`, `relation_type`

### For Test Setup Issues:
- [ ] Check `TestPatternLoaderEdgeCases.__init__()` or `setUp()`
- [ ] Verify `sample_patterns` attribute initialization
- [ ] Review parent class inheritance for test fixtures

### For PatternMatcher Output:
- [ ] Compare `get_pattern_summary()` implementation vs test expectations
- [ ] Check if method should return simple lists or enriched objects
- [ ] Verify if mock setup matches actual method signature

---

## 📋 SYSTEMATIC DEBUGGING APPROACH

1. **Start with PatternLoader** (highest impact - 5 failed tests)
   - Focus on `json_to_pattern()` method implementation
   - Add debug logging to see which keys are being processed
   - Test with minimal pattern structure

2. **Fix Test Setup** (blocking other tests)
   - Add missing `sample_patterns` attribute
   - Verify test class inheritance and setup methods

3. **Resolve PatternMatcher Output** (interface contract issue)
   - Determine correct return format for `get_pattern_summary()`
   - Update either implementation or test expectations

---

## 🔍 SEMANTIC PATH FINDING DEBUGGING (NEW)

### **4. PairwiseBidirectionalAStar + WordNet Graph Connectivity Analysis** (PRIORITY: HIGH)

**Problem Statement:** 
Despite fixing word sense disambiguation (correctly selecting `cat.n.01` feline and `mouse.n.01` rodent), the PairwiseBidirectionalAStar algorithm fails to find semantic paths between conceptually related synsets like "cat" → "chase" → "mouse".

**Root Cause Hypotheses:**
1. **Graph Connectivity Gap**: WordNet semantic graph lacks sufficient edge density between conceptually related synsets
2. **Beam Filtering Too Restrictive**: Algorithm's allowed node sets exclude necessary intermediate concepts  
3. **Heuristic Function Issues**: Embedding-based heuristics misdirect the search away from optimal paths
4. **Relation Type Coverage**: Missing key semantic relation types in graph construction or pathfinding
5. **Cross-POS Connection Weakness**: Insufficient noun→verb→noun bridging mechanisms
6. **Search Space Constraints**: max_depth, beam_width, or len_tolerance parameters too restrictive

---

### 🔍 **Phase 1: Graph Connectivity Analysis**

#### **Step 1.1: WordNet Relation Coverage Audit**
**Objective:** Verify all available semantic relations are being utilized in graph construction

**Tasks:**
- [ ] **Document Available Relations**: Create comprehensive mapping of NLTK WordNet relations by POS:
  - Nouns: `hypernyms()`, `hyponyms()`, `member_holonyms()`, `part_holonyms()`, `substance_holonyms()`, `member_meronyms()`, `part_meronyms()`, `substance_meronyms()`, `similar_tos()`, `attributes()`, `also_sees()`
  - Verbs: `hypernyms()`, `hyponyms()`, `entailments()`, `causes()`, `similar_tos()`, `also_sees()`, `verb_groups()`
  - Cross-POS: `lemma.derivationally_related_forms()`, `lemma.antonyms()`, `lemma.pertainyms()`

- [ ] **Audit Graph Construction** (`SemanticDecomposer.build_synset_graph()`):
  - Verify all relation types from audit are included in graph building (lines 620-634)
  - Check if derivationally_related_forms() is being added as edges (currently missing)
  - Validate edge weights and directionality for each relation type
  - Document any missing high-value relations (e.g., attributes, topic domains)

- [ ] **Cross-POS Bridge Analysis**: 
  - Investigate noun→verb connections via `lemma.derivationally_related_forms()`
  - Check verb→noun connections via `entailments()`, `causes()` 
  - Analyze if current graph supports noun→verb→noun paths effectively

#### **Step 1.2: Specific Path Connectivity Testing**  
**Objective:** Manually trace why specific conceptual paths fail to exist in the graph

**Tasks:**
- [ ] **Manual Path Tracing**: Create debugging utilities to analyze connectivity:
  ```python
  def analyze_synset_connectivity(synset1_name, synset2_name, max_hops=6):
      """Analyze all possible paths between two synsets in the graph"""
      # Report direct connections, 2-hop paths, 3-hop paths, etc.
      # Show relation types used in each path
      # Identify connectivity gaps
  ```

- [ ] **Target Analysis Cases**: Apply manual tracing to failing test cases:
  - `cat.n.01` → `chase.v.01`: Expected connection through predatory behavior concepts
  - `chase.v.01` → `mouse.n.01`: Expected connection through prey/object concepts  
  - `cat.n.01` → `mouse.n.01`: Expected connection through predator-prey relationships

- [ ] **Relation Density Analysis**: For each synset in failing test cases:
  - Count total outgoing edges by relation type
  - Identify synsets with unusually sparse connections
  - Check if key conceptual bridges exist (e.g., "feline" → "predator" → "hunt")

#### **Step 1.3: Graph Topology Metrics**
**Objective:** Quantify graph connectivity properties to identify structural issues

**Tasks:**
- [ ] **Basic Graph Metrics**: Implement analysis for the built synset graph:
  - Average node degree (in/out)
  - Connected components count (should be 1 for effective pathfinding)
  - Diameter and average path length
  - Clustering coefficient by POS type

- [ ] **POS-Specific Connectivity**: 
  - Noun-to-noun connectivity strength
  - Verb-to-verb connectivity strength  
  - Cross-POS bridge density (noun↔verb connections)
  - Identify isolated or poorly connected synset clusters

- [ ] **Embedding-Based Beam Coverage**: Analyze beam generation effectiveness:
  - What percentage of semantically related synsets appear in embedding-generated beams?
  - Are conceptually obvious connections (like cat→chase) being captured by embeddings?
  - Compare embedding similarities vs. actual WordNet relation strengths

---

### 🔍 **Phase 2: PairwiseBidirectionalAStar Algorithm Analysis**

#### **Step 2.1: Search Space Constraints Evaluation**
**Objective:** Determine if algorithm parameters are excluding valid paths

**Tasks:**
- [ ] **Parameter Sensitivity Analysis**: Test pathfinding with varying parameters:
  - `max_depth`: Current default is 10, test with 6, 8, 12, 15
  - `beam_width`: Current default is 3, test with 5, 7, 10
  - `len_tolerance`: Current default varies, test with 2, 5, 8
  - `relax_beam`: Test with `True` to bypass beam restrictions entirely

- [ ] **Allowed Node Set Analysis**: Debug the beam filtering mechanism:
  - Log which synsets are included/excluded in `src_allowed` and `tgt_allowed` sets
  - Check if conceptually important intermediate synsets are being filtered out
  - Analyze if embedding-based beam selection is missing obvious semantic bridges

- [ ] **Search Direction Bias**: Analyze bidirectional search balance:
  - Track forward vs. backward expansion ratios
  - Check if one direction consistently dominates (indicating heuristic issues)
  - Verify meet-in-the-middle behavior is working correctly

#### **Step 2.2: Heuristic Function Validation**
**Objective:** Verify embedding-based heuristics are guiding search effectively  

**Tasks:**
- [ ] **Embedding Heuristic Quality**: For target test cases:
  - Compare embedding similarity predictions vs. actual WordNet path existence
  - Check if embedding heuristics are consistent with semantic distances
  - Analyze cases where embeddings strongly suggest connections that don't exist in WordNet

- [ ] **Gloss Seed Integration**: Debug gloss-based search seeding:
  - Verify gloss parsing is extracting semantically relevant terms
  - Check if `GLOSS_BONUS` (0.15) is appropriately sized
  - Analyze if gloss seeds are actually being prioritized in search

- [ ] **Alternative Heuristics**: Test pathfinding with simpler heuristics:
  - Distance-only heuristic (h = 0 for all nodes)
  - Uniform cost search to eliminate heuristic bias
  - WordNet-path-based heuristics instead of embedding-based

#### **Step 2.3: Search Algorithm Correctness**
**Objective:** Verify the bidirectional A* implementation is functioning correctly

**Tasks:**
- [ ] **Algorithm Correctness Validation**:
  - Test against simple known-path cases (e.g., direct hypernym/hyponym relations)
  - Verify path reconstruction produces valid paths
  - Check meeting point detection and cost calculation accuracy
  - Validate priority queue ordering and cycle detection

- [ ] **Edge Weight Analysis**: 
  - Current weights are uniform (1.0) - analyze if differential weighting would help
  - Consider semantic relation type-based weights (hypernyms=1.0, meronyms=1.2, etc.)
  - Test impact of relation-specific costs on path quality

- [ ] **Memory and Performance Profiling**:
  - Check if search is terminating prematurely due to resource constraints
  - Profile queue sizes and expansion counts
  - Identify potential infinite loop or excessive branching issues

---

### 🔍 **Phase 3: Integration and Alternative Approaches**

#### **Step 3.1: Hybrid Path Finding Strategies**
**Objective:** Develop fallback strategies when embedding-based approach fails

**Tasks:**
- [ ] **Pure WordNet Relation Traversal**: Implement alternative pathfinder:
  - Breadth-first search using only direct WordNet relations
  - Prioritize high-value semantic relations (hypernyms, entailments, causes)
  - No embedding constraints, pure graph traversal with depth limits

- [ ] **Concept-Based Bridging**: Implement domain-specific connection strategies:
  - Animal behavior patterns (predator→prey, action→object)
  - Semantic role bridging (agent→action→patient)  
  - Category-based connections (mammal→behavior→small_animal)

- [ ] **Multi-Stage Search**: Implement hierarchical search approach:
  - Stage 1: Find paths through shared hypernyms
  - Stage 2: Find paths through conceptual bridges (behavior, attributes)
  - Stage 3: Find paths through lexical derivations

#### **Step 3.2: Ground Truth Validation**
**Objective:** Establish benchmark cases to validate any fixes

**Tasks:**
- [ ] **Create Gold Standard Test Cases**: 
  - Manually identify 20+ synset triples that SHOULD be connected
  - Include various domains: animals, actions, objects, abstract concepts
  - Document expected connection types and path lengths

- [ ] **Comparative Analysis**:
  - Test current system against gold standard
  - Compare results with conceptnet.io or other semantic networks
  - Analyze false positives (incorrect connections) vs false negatives (missing connections)

- [ ] **User Study Validation**: 
  - Human evaluation of path quality and semantic reasonableness
  - Comparison with human intuition about semantic relatedness

#### **Step 3.3: Performance and Scalability**
**Objective:** Ensure fixes don't degrade system performance

**Tasks:**
- [ ] **Benchmark Performance Impact**: 
  - Measure graph construction time with additional relations
  - Profile pathfinding speed with different parameter combinations
  - Memory usage analysis for larger search spaces

- [ ] **Optimization Opportunities**:
  - Graph caching and incremental updates
  - Precomputed semantic distances for common concepts  
  - Index-based fast lookup for high-frequency synset pairs

---

### 📋 **Systematic Debugging Execution Plan**

#### **Priority Order**:
1. **Start with Phase 1.2** (Specific Path Connectivity) - High impact, quick insights
2. **Phase 2.1** (Parameter Analysis) - Easy to test, immediate actionable results  
3. **Phase 1.1** (Relation Coverage) - Foundational understanding
4. **Phase 2.2** (Heuristic Validation) - Addresses core algorithm assumptions
5. **Phase 3.1** (Alternative Approaches) - Fallback solutions

#### **Success Criteria**:
- [ ] `cat.n.01 → chase.v.01 → mouse.n.01` produces valid semantic paths
- [ ] Path lengths ≤ 6 hops for conceptually related synsets  
- [ ] 90% success rate on manually curated gold standard test cases
- [ ] No significant performance degradation (<50% slower than current)

#### **Risk Mitigation**:
- Implement changes incrementally with rollback capability
- Maintain comprehensive test coverage for regression detection
- Document all parameter changes and their rationale
- Create diagnostic utilities for ongoing debugging

---

## 🧪 TEST VALIDATION STRATEGY

**For each fix:**
1. Run specific failing test in isolation
2. Add unit test for the specific bug scenario  
3. Run full test suite to ensure no regressions
4. Verify fix handles edge cases (nested structures, empty lists, etc.)