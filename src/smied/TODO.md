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

#### **Task 1.1: Add Missing Derivational Relations** (PRIORITY: CRITICAL - 80% Impact)

**Root Cause Confirmed:** `SemanticDecomposer.build_synset_graph()` missing `derivationally_related_forms()` - the PRIMARY cause of cross-POS pathfinding failures.

**Technical Requirements:**
- [ ] **Code Implementation** in `src/smied/SemanticDecomposer.py` around line 649:
  ```python
  # Add after existing relation processing in build_synset_graph()
  # Process lemma-level derivational relations for cross-POS bridges
  for lemma in synset.lemmas():
      derived_forms = lemma.derivationally_related_forms()
      for derived_lemma in derived_forms:
          derived_synset_name = derived_lemma.synset().name()
          if derived_synset_name in g:
              g.add_edge(synset_name, derived_synset_name,
                        relation='derivationally_related', weight=1.0)
              edge_count += 1
              if 'derivationally_related' not in relation_counts:
                  relation_counts['derivationally_related'] = 0
              relation_counts['derivationally_related'] += 1
  ```

- [ ] **Validation Testing**:
  - Test pathfinding on critical failing cases: `cat.n.01` → `chase.v.01` → `mouse.n.01`
  - Verify 10+ derivational connections for `chase.v.01` are now accessible in graph
  - Measure success rate improvement (expect 80%+ improvement)
  - Confirm path lengths remain ≤ 6 hops for conceptually related synsets

- [ ] **Performance Impact Analysis**:
  - Benchmark graph construction time before/after derivational relations
  - Measure memory usage increase (expect <20% increase)
  - Profile pathfinding speed impact (acceptable degradation <50%)

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