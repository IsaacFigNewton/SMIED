# SMIED Bug Remediation Plan

## Executive Summary
The SMIED package has a critical blocking error preventing any semantic decomposition functionality, along with numerous warnings that affect reliability. This plan outlines the minimum necessary changes to restore core operation.

## Critical Issues Identified

### 1. BLOCKING ERROR: Token/String Type Mismatch
- **Severity**: CRITICAL - Prevents all functionality
- **Location**: `src/smied/SemanticDecomposer.py`, lines 84, 85, 86
- **Error**: `TypeError: 'int' object is not callable`
- **Root Cause**: spaCy Token objects passed to WordNet synsets() which expects strings

### 2. Division by Zero in Variance Calculations
- **Severity**: HIGH - Causes 42+ warnings, affects confidence scoring
- **Location**: `src/smied/FramenetSpacySRL.py`, lines 543-546
- **Error**: `float division by zero`
- **Root Cause**: statistics.variance() returns 0 when all role proportions are identical

### 3. Edge Case Handling
- **Severity**: MEDIUM - Causes crashes with certain inputs
- **Locations**: Multiple
- **Issues**: Empty inputs, non-English words, malformed triples

## Remediation Plan

### Phase 1: Critical Fix (IMMEDIATE - Required for Basic Functionality)

#### Fix 1.1: Token to String Conversion in SemanticDecomposer
**File**: `src/smied/SemanticDecomposer.py`
**Lines**: 84-86
**Change**:
```python
# CURRENT (BROKEN):
subject_synsets = self.wn_module.synsets(subj_tok, pos='n')
predicate_synsets = self.wn_module.synsets(pred_tok, pos='v') 
object_synsets = self.wn_module.synsets(obj_tok, pos='n')

# FIXED:
subject_synsets = self.wn_module.synsets(subj_tok.lemma_ if hasattr(subj_tok, 'lemma_') else str(subj_tok), pos='n')
predicate_synsets = self.wn_module.synsets(pred_tok.lemma_ if hasattr(pred_tok, 'lemma_') else str(pred_tok), pos='v')
object_synsets = self.wn_module.synsets(obj_tok.lemma_ if hasattr(obj_tok, 'lemma_') else str(obj_tok), pos='n')
```

### Phase 2: High Priority Fixes (Required for Stable Operation)

#### Fix 2.1: Division by Zero Protection in FramenetSpacySRL
**File**: `src/smied/FramenetSpacySRL.py`
**Lines**: 543-546
**Change**:
```python
# CURRENT (PROBLEMATIC):
idi = statistics.variance(role_proportions) / statistics.mean(role_proportions) if statistics.mean(role_proportions) != 0 else 0
inverse_idi = 1 / idi if idi != 0 else 0

# FIXED:
mean_prop = statistics.mean(role_proportions) if role_proportions else 0
var_prop = statistics.variance(role_proportions) if len(role_proportions) > 1 else 0

if mean_prop != 0 and var_prop != 0:
    idi = var_prop / mean_prop
    inverse_idi = 1 / idi
else:
    idi = 0
    inverse_idi = 0  # or use a default confidence value like 0.5
```

#### Fix 2.2: Improve Error Handling in FramenetSpacySRL._process_predicate
**File**: `src/smied/FramenetSpacySRL.py`
**Location**: Within `_process_predicate` method
**Add**: Wrap the entire method body in proper exception handling:
```python
try:
    # existing method body
except ZeroDivisionError as e:
    if self.verbose:
        print(f"[WARNING] FrameNetSpaCySRL: Division by zero in predicate '{predicate}': {e}")
    return {}  # Return empty dict instead of crashing
except Exception as e:
    if self.verbose:
        print(f"[ERROR] FrameNetSpaCySRL: Error processing predicate '{predicate}': {e}")
    return {}
```

### Phase 3: Input Validation (Required for Robustness)

#### Fix 3.1: Add Input Validation to SMIED.analyze_triple
**File**: `src/smied/SMIED.py`
**Location**: Beginning of `analyze_triple` method (around line 90)
**Add**:
```python
# Input validation
if not all([subject, predicate, obj]):
    print("[ERROR] All triple components (subject, predicate, object) must be non-empty")
    return None

# Ensure strings are not empty after stripping
subject = str(subject).strip()
predicate = str(predicate).strip()
obj = str(obj).strip()

if not all([subject, predicate, obj]):
    print("[ERROR] Triple components cannot be empty or whitespace-only")
    return None
```

### Phase 4: Testing & Verification

#### Test 4.1: Run Existing Tests
```bash
python -m pytest tests/test_hypothesis_validation.py -v
python -m pytest tests/test_edge_cases.py -v
```

#### Test 4.2: Run Example Script
```bash
python examples/smied_example.py
```

#### Test 4.3: Verify No Warnings
- Ensure no division by zero warnings appear during normal operation
- Verify all triples process without errors

### Implementation Order

1. **Day 1**: Implement Fix 1.1 (Critical Token conversion) - MUST be done first
2. **Day 1**: Test that basic functionality works with Fix 1.1
3. **Day 2**: Implement Fixes 2.1 and 2.2 (Division by zero protection)
4. **Day 2**: Implement Fix 3.1 (Input validation)
5. **Day 3**: Run full test suite and verify all fixes

## Expected Outcomes

After implementing these fixes:
1. ✅ The example script will run without crashing
2. ✅ No division by zero warnings during initialization
3. ✅ Proper error messages for invalid inputs
4. ✅ Core semantic decomposition functionality restored
5. ✅ Confidence scoring will work reliably

## Files to Modify (Minimum Set)

1. `src/smied/SemanticDecomposer.py` - 3 lines
2. `src/smied/FramenetSpacySRL.py` - ~15 lines (variance fix + error handling)
3. `src/smied/SMIED.py` - ~10 lines (input validation)

Total: ~28 lines of code changes across 3 files

## Verification Checklist

- [ ] Fix 1.1 implemented and tested
- [ ] examples/smied_example.py runs without TypeError
- [ ] Fix 2.1 implemented and tested
- [ ] No division by zero warnings in output
- [ ] Fix 2.2 implemented
- [ ] Fix 3.1 implemented
- [ ] All tests in tests/test_hypothesis_validation.py pass
- [ ] All tests in tests/test_edge_cases.py pass
- [ ] Example script produces meaningful semantic output
- [ ] No regression in existing functionality

## Notes

- These are the MINIMUM changes required for core operation
- Additional improvements (logging, documentation, optimization) can be added later
- Focus is on fixing blocking errors first, then stability issues
- All fixes maintain backward compatibility