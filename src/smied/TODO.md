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

## 🧪 TEST VALIDATION STRATEGY

**For each fix:**
1. Run specific failing test in isolation
2. Add unit test for the specific bug scenario
3. Run full test suite to ensure no regressions
4. Verify fix handles edge cases (nested structures, empty lists, etc.)