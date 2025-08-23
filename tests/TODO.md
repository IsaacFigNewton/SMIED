# Test Refactoring TODO List

## Tests Missing Config Imports

The following test files are missing the `from tests.config.` import and need to be refactored to use their corresponding configuration classes:

### 1. test_metavertex_pattern_matcher.py
- **Config File**: `tests/config/metavertex_pattern_matcher_config.py`
- **Config Class**: `MetavertexPatternMatcherMockConfig`
- **Test Classes**:
  - [ ] Refactor `test_metavertex_pattern_matcher.py` `TestMetavertexPatternMatcher` setUp(self) based on mock configuration of `MetavertexPatternMatcherMockConfig` in `config/metavertex_pattern_matcher_config.py`
  - [ ] Refactor `test_metavertex_pattern_matcher.py` `TestMetavertexPatternIntegration` setUp(self) based on mock configuration of `MetavertexPatternMatcherMockConfig` in `config/metavertex_pattern_matcher_config.py`

### 2. test_pairwise_bidirectional_astar.py
- **Config File**: `tests/config/pairwise_bidirectional_astar_config.py`
- **Config Class**: `PairwiseBidirectionalAStarMockConfig`
- **Test Classes**:
  - [ ] Refactor `test_pairwise_bidirectional_astar.py` `TestPairwiseBidirectionalAStar` setUp(self) based on mock configuration of `PairwiseBidirectionalAStarMockConfig` in `config/pairwise_bidirectional_astar_config.py`
  - [ ] Refactor `test_pairwise_bidirectional_astar.py` `TestPairwiseBidirectionalAStarEdgeCases` setUp(self) based on mock configuration of `PairwiseBidirectionalAStarMockConfig` in `config/pairwise_bidirectional_astar_config.py`
  - [ ] Refactor `test_pairwise_bidirectional_astar.py` `TestPairwiseBidirectionalAStarIntegration` setUp(self) based on mock configuration of `PairwiseBidirectionalAStarMockConfig` in `config/pairwise_bidirectional_astar_config.py`

### 3. test_pattern_matcher.py
- **Config File**: `tests/config/pattern_matcher_config.py`
- **Config Class**: `PatternMatcherMockConfig`
- **Test Classes**:
  - [ ] Refactor `test_pattern_matcher.py` `TestPatternMatcher` setUp(self) based on mock configuration of `PatternMatcherMockConfig` in `config/pattern_matcher_config.py`
  - [ ] Refactor `test_pattern_matcher.py` `TestPatternMatcherEdgeCases` setUp(self) based on mock configuration of `PatternMatcherMockConfig` in `config/pattern_matcher_config.py`
  - [ ] Refactor `test_pattern_matcher.py` `TestPatternMatcherIntegration` setUp(self) based on mock configuration of `PatternMatcherMockConfig` in `config/pattern_matcher_config.py`

### 4. test_smied.py
- **Config File**: `tests/config/smied_config.py`
- **Config Class**: `SMIEDMockConfig`
- **Test Classes**:
  - [ ] Refactor `test_smied.py` `TestISMIEDPipeline` setUp(self) based on mock configuration of `SMIEDMockConfig` in `config/smied_config.py`
  - [ ] Refactor `test_smied.py` `TestSMIED` setUp(self) based on mock configuration of `SMIEDMockConfig` in `config/smied_config.py`
  - [ ] Refactor `test_smied.py` `TestSMIEDIntegration` setUp(self) based on mock configuration of `SMIEDMockConfig` in `config/smied_config.py`

## Refactoring Steps for Each Test

For each test file and class listed above:

1. Add the import statement: `from tests.config.<config_module> import <ConfigClass>`
2. In the `setUp(self)` method, initialize the config: `self.config = <ConfigClass>()`
3. Replace direct mock instantiation with config-based mock factory calls
4. Update any mock setup to use the configuration's mock factory pattern
5. Ensure all mock dependencies are properly configured through the config class

## Notes

- Each config class provides a centralized mock factory for creating consistent test fixtures
- The config approach ensures better maintainability and reusability of test mocks
- After refactoring, run each test file to ensure functionality is preserved

## Configuration Parameter Extraction to JSON

The following config files need their nested dictionary parameters extracted to JSON files in `config/sample_params/`:

### Config Files for Parameter Extraction

- [ ] Extract parameters from `beam_builder_config.py` to `config/sample_params/beam_builder_config_params.json`
- [ ] Extract parameters from `directed_metagraph_config.py` to `config/sample_params/directed_metagraph_config_params.json`
- [ ] Extract parameters from `embedding_helper_config.py` to `config/sample_params/embedding_helper_config_params.json`
- [ ] Extract parameters from `gloss_parser_config.py` to `config/sample_params/gloss_parser_config_params.json`
- [ ] Extract parameters from `metavertex_pattern_matcher_config.py` to `config/sample_params/metavertex_pattern_matcher_config_params.json`
- [ ] Extract parameters from `pairwise_bidirectional_astar_config.py` to `config/sample_params/pairwise_bidirectional_astar_config_params.json`
- [ ] Extract parameters from `pattern_loader_config.py` to `config/sample_params/pattern_loader_config_params.json`
- [ ] Extract parameters from `pattern_matcher_config.py` to `config/sample_params/pattern_matcher_config_params.json`
- [ ] Extract parameters from `semantic_decomposer_config.py` to `config/sample_params/semantic_decomposer_config_params.json`
- [ ] Extract parameters from `semantic_metagraph_config.py` to `config/sample_params/semantic_metagraph_config_params.json`
- [ ] Extract parameters from `smied_config.py` to `config/sample_params/smied_config_params.json`

### Utility Class for JSON Configuration Loading

- [ ] Create `tests/config/config_loader.py` utility class with the following functionality:
  - Load JSON configuration files from `config/sample_params/`
  - Convert string representations of tuples to actual tuples (e.g., `"('animal.n.01', 0.9)"` → `('animal.n.01', 0.9)`)
  - Convert lists of string tuples to lists of actual tuples where appropriate
  - Handle nested dictionary structures
  - Preserve original data types from the configuration
  - Provide error handling for missing or malformed JSON files

### Parameter Extraction Guidelines

When extracting parameters to JSON files:

1. **Extract**: Simple dictionaries, lists, strings, numbers, and tuples
2. **Convert for storage**:
   - Tuples should be stored as strings or lists in JSON (e.g., `[('animal.n.01', 0.9), ('mammal.n.01', 0.85)]` → `["('animal.n.01', 0.9)", "('mammal.n.01', 0.85)"]` or as nested lists `[["animal.n.01", 0.9], ["mammal.n.01", 0.85]]`)
3. **DO NOT Extract**: 
   - Complex objects like `nx.DiGraph`
   - Library function calls like `np.random.rand()`
   - Mock objects or class instances
   - Lambda functions or callable objects

### Example Conversion Patterns

The config loader should handle these conversion patterns:

```python
# Original Python format in config
'similar_words': [('animal.n.01', 0.9), ('mammal.n.01', 0.85)]

# JSON storage format (option 1 - string representation)
"similar_words": ["('animal.n.01', 0.9)", "('mammal.n.01', 0.85)"]

# JSON storage format (option 2 - nested lists)
"similar_words": [["animal.n.01", 0.9], ["mammal.n.01", 0.85]]

# Loaded back to Python
'similar_words': [('animal.n.01', 0.9), ('mammal.n.01', 0.85)]
```

### Implementation Steps

1. Create the `config/sample_params/` directory structure
2. Analyze each config file's static methods that return dictionaries
3. Extract appropriate parameters to corresponding JSON files
4. Implement the `ConfigLoader` utility class
5. Update config files to use `ConfigLoader` for loading JSON parameters
6. Test that all configurations load correctly with proper type conversions