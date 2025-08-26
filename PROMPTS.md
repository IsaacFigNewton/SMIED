# Refactoring Plan Request

Please review the following files:  
- [file1]
- [file2]

## Objective
Think carefully about how the **core functionality** of [file2] can be **completely replaced** with the approach used in [file1].

## Requirements
The new implementation should:
- Return the **same output types** as the old one for:
  - `[method name here]`
- Use **existing helpers only as needed**.
- Remove or refactor all other methods so they are **incompatible with the old [file2] implementation**.

## Allowed New Functionality
- A new method:  
```python
  [new method name here]
```

* Any additional functions must be **strictly helpers** and only used within:
  * `[helper name]`

## Deliverables

1. **Comprehensive refactoring plan** for `[file2]` to enable triple-based SRL of the subject, object, and any indirect object dependents from the original sentence.
   * Do **not** propose extraneous functionality beyond what is required for this reimplementation.

2. **Unit test requirements** to validate the successful implementation of the new pipeline.

3. Write the finalized plan to `[TODO file]`