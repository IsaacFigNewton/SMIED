# SMIED Package Hypothesis Validation Results

## Test Summary

This document summarizes the results of detailed testing to validate specific hypotheses about bugs in the SMIED package.

## Issues Validated

### 1. ✅ CRITICAL ERROR - Token/String Conversion Issue (CONFIRMED)

**Location:** `SemanticDecomposer.py`, line 84  
**Error:** `'int' object is not callable` when calling `self.wn_module.synsets(subj_tok, pos='n')`  
**Hypothesis:** The `subj_tok` parameter is a spaCy Token object but `synsets()` expects a string (lemma)

#### Test Results:
- **Test File:** `tests/test_hypothesis_validation.py`
- **Method:** `test_wordnet_synsets_with_token_object_fails()`
- **Status:** ✅ CONFIRMED

#### Key Findings:
1. Passing a spaCy Token object to `wn.synsets()` raises: `TypeError: 'int' object is not callable`
2. The SMIED class converts string inputs to Token objects using `self.nlp(string)[0]`
3. SemanticDecomposer then passes these Token objects directly to WordNet synsets()
4. **Root Cause:** Lines 84-86 in `SemanticDecomposer.py`:
   ```python
   subject_synsets = self.wn_module.synsets(subj_tok, pos='n')  # ❌ Passes Token
   object_synsets = self.wn_module.synsets(obj_tok, pos='n')    # ❌ Passes Token
   ```

#### Proposed Fix:
Replace Token objects with their lemmatized forms:
```python
subject_synsets = self.wn_module.synsets(subj_tok.lemma_, pos='n')  # ✅ Passes string
object_synsets = self.wn_module.synsets(obj_tok.lemma_, pos='n')    # ✅ Passes string
```

### 2. ✅ Division by Zero Warnings (CONFIRMED)

**Location:** `FramenetSpacySRL.py`, lines 543-546  
**Error:** `float division by zero` in `_score_synset_by_idi_and_freq()`  
**Hypothesis:** `statistics.variance()` returns 0 when all values are identical, causing division by zero

#### Test Results:
- **Test File:** `tests/test_hypothesis_validation.py`
- **Method:** `test_statistics_variance_zero_with_identical_values()` and `test_score_synset_division_by_zero()`
- **Status:** ✅ CONFIRMED

#### Key Findings:
1. When all role proportions are identical (e.g., [0.333, 0.333, 0.333]), `statistics.variance()` returns 0.0
2. Line 543-546 in `FramenetSpacySRL.py` performs division by this variance:
   ```python
   inverse_index_of_dispersion = mean_role_count / statistics.variance([
       (len(roles) / total_roles)
       for roles in synset_roles.values()
   ])
   ```
3. During testing, we observed multiple warnings like:
   ```
   [WARNING] FrameNetSpaCySRL: Error processing predicate 'chase': float division by zero
   ```

#### Proposed Fix:
Add variance check before division:
```python
role_proportions = [(len(roles) / total_roles) for roles in synset_roles.values()]
variance = statistics.variance(role_proportions)
if variance == 0:
    inverse_index_of_dispersion = mean_role_count  # or some other default
else:
    inverse_index_of_dispersion = mean_role_count / variance
```

## Test Results Summary

### Hypothesis Validation Tests (`test_hypothesis_validation.py`)
- ✅ `test_wordnet_synsets_with_token_object_fails`: Confirmed TypeError with Token objects
- ✅ `test_wordnet_synsets_with_token_text_works`: Confirmed token.text works
- ✅ `test_wordnet_synsets_with_token_lemma_works`: Confirmed token.lemma_ works (preferred)
- ✅ `test_semantic_decomposer_token_handling`: Reproduced the actual bug
- ✅ `test_fix_by_using_lemma`: Demonstrated the fix works
- ✅ `test_statistics_variance_zero_with_identical_values`: Confirmed variance=0 causes division error
- ✅ `test_score_synset_division_by_zero`: Reproduced division by zero in actual method
- ✅ `test_reproduce_smied_pipeline_error`: Confirmed error occurs in full pipeline

### Edge Cases Tests (`test_edge_cases.py`)
- ✅ `test_empty_string_inputs`: Found spaCy index errors with empty strings
- Additional tests for non-English words, malformed inputs, and boundary conditions

## Specific Error Messages Observed

### Token Conversion Error:
```
TypeError: 'int' object is not callable
```

### Division by Zero Warnings:
```
[WARNING] FrameNetSpaCySRL: Error processing predicate 'chase': float division by zero
[WARNING] FrameNetSpaCySRL: Error processing predicate 'write': float division by zero
[WARNING] FrameNetSpaCySRL: Error processing predicate 'run': float division by zero
...
```

### Empty Input Errors:
```
[E040] Attempt to access token at 0, max length 0.
```

## Impact Assessment

1. **Token Conversion Issue**: This is a blocking error that prevents the SMIED pipeline from working at all
2. **Division by Zero**: This causes warnings and potentially incorrect scoring, but doesn't completely break the pipeline
3. **Edge Cases**: Various inputs cause different types of failures, suggesting need for better input validation

## Recommendations

1. **Immediate Fix Required**: Change lines 84-86 in `SemanticDecomposer.py` to use `token.lemma_` instead of the Token object
2. **Division by Zero Protection**: Add variance checking in `FramenetSpacySRL.py` method `_score_synset_by_idi_and_freq`
3. **Input Validation**: Add proper validation for empty, None, or malformed inputs
4. **Testing**: Implement comprehensive unit tests for edge cases
5. **Error Handling**: Add try-catch blocks around critical operations with meaningful error messages

## Files Modified for Testing

1. `tests/test_hypothesis_validation.py` - Comprehensive hypothesis validation tests
2. `tests/test_edge_cases.py` - Edge case and boundary condition tests
3. `tests/hypothesis_validation_results.md` - This summary document

## Test Execution

All tests can be run with:
```bash
python -m pytest tests/test_hypothesis_validation.py -v
python -m pytest tests/test_edge_cases.py -v
```

Individual tests can be run with:
```bash
python -m pytest tests/test_hypothesis_validation.py::TestTokenStringConversionHypothesis::test_semantic_decomposer_token_handling -v -s
```