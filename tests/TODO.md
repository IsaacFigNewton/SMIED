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

## Proposed Abstract Base Classes for Mock Refactoring

### Abstract Base Classes to Create

#### 1. AbstractHandlerMock (tests/mocks/base/handler_mock.py)
- **Purpose**: Base class for format handler mocks (JSON, YAML, XML, etc.)
- **Common Interface**: load(), save(), validate() methods
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockJSONHandler (pattern_loader_mocks.py)
  - MockYAMLHandler (pattern_loader_mocks.py)
  - MockXMLHandler (pattern_loader_mocks.py)

#### 2. AbstractEdgeCaseMock (tests/mocks/base/edge_case_mock.py)
- **Purpose**: Base class for edge case testing scenarios
- **Common Patterns**: Error scenarios, empty returns, invalid inputs
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockPatternMatcherEdgeCases (pattern_matcher_mocks.py)
  - MockPatternLoaderEdgeCases (pattern_loader_mocks.py)
  - MockPairwiseBidirectionalAStarEdgeCases (pairwise_bidirectional_astar_mocks.py)
  - MockBeamBuilderEdgeCases (beam_builder_mocks.py)
  - MockEmbeddingHelperEdgeCases (embedding_helper_mocks.py)

#### 3. AbstractIntegrationMock (tests/mocks/base/integration_mock.py)
- **Purpose**: Base class for integration testing mocks
- **Common Pattern**: Composition of multiple mock components
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockPatternMatcherIntegration (pattern_matcher_mocks.py)
  - MockPairwiseBidirectionalAStarIntegration (pairwise_bidirectional_astar_mocks.py)
  - MockSemanticMetagraphIntegration (semantic_metagraph_mocks.py)
  - MockEmbeddingHelperIntegration (embedding_helper_mocks.py)

#### 4. AbstractGraphPatternMock (tests/mocks/base/graph_pattern_mock.py)
- **Purpose**: Base class for graph/network pattern mocks
- **Common Attributes**: nodes, size/length attributes
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockChainPattern (metavertex_pattern_matcher_mocks.py)
  - MockCyclePattern (metavertex_pattern_matcher_mocks.py)
  - MockCliquePattern (metavertex_pattern_matcher_mocks.py)
  - MockTreePattern (metavertex_pattern_matcher_mocks.py)
  - MockComplexPatterns (pattern_matcher_mocks.py)

#### 5. AbstractNLPDocMock (tests/mocks/base/nlp_doc_mock.py)
- **Purpose**: Base class for NLP document mocks
- **Common Attributes**: text, tokens, ents, noun_chunks
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockDoc (beam_builder_mocks.py)
  - MockDocForGloss (gloss_parser_mocks.py)
  - MockDocForDecomposer (semantic_decomposer_mocks.py)

#### 6. AbstractNLPTokenMock (tests/mocks/base/nlp_token_mock.py)
- **Purpose**: Base class for NLP token mocks
- **Common Attributes**: text, lemma_, pos_, tag_, dep_
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockToken (beam_builder_mocks.py)
  - MockTokenForGloss (gloss_parser_mocks.py)
  - MockTokenForDecomposer (semantic_decomposer_mocks.py)
  - MockEntityForGloss (gloss_parser_mocks.py)

#### 7. AbstractNLPFunctionMock (tests/mocks/base/nlp_function_mock.py)
- **Purpose**: Base class for NLP processing function mocks
- **Common Pattern**: Returns a doc object
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockNLPFunction (beam_builder_mocks.py)
  - MockNLPForGloss (gloss_parser_mocks.py)
  - MockNLPForDecomposer (semantic_decomposer_mocks.py)
  - MockRealNLPForGloss (gloss_parser_mocks.py)

#### 8. AbstractAlgorithmicFunctionMock (tests/mocks/base/algorithmic_function_mock.py)
- **Purpose**: Base class for algorithmic/mathematical function mocks
- **Common Pattern**: Calculation methods returning numeric values
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockHeuristicFunction (pairwise_bidirectional_astar_mocks.py)
  - MockCostFunction (pairwise_bidirectional_astar_mocks.py)
  - MockBeamBuilderForDecomposer (semantic_decomposer_mocks.py)
  - MockPatternMatcher (semantic_decomposer_mocks.py)
  - MockPairwiseBidirectionalAStar (semantic_decomposer_mocks.py)

#### 9. AbstractOperationMock (tests/mocks/base/operation_mock.py)
- **Purpose**: Base class for operation-specific mocks
- **Common Pattern**: Operation methods with appropriate return types
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockDirectedMetagraphValidation (directed_metagraph_mocks.py)
  - MockDirectedMetagraphCanonicalization (directed_metagraph_mocks.py)
  - MockDirectedMetagraphManipulation (directed_metagraph_mocks.py)
  - MockDirectedMetagraphNetworkXConversion (directed_metagraph_mocks.py)

#### 10. AbstractLibraryWrapperMock (tests/mocks/base/library_wrapper_mock.py)
- **Purpose**: Base class for external library interface mocks
- **Common Pattern**: Minimal wrapper around library methods
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockNLTK (smied_mocks.py)
  - MockSpacy (smied_mocks.py)
  - MockWordNet (smied_mocks.py, semantic_metagraph_mocks.py)
  - MockGraph (smied_mocks.py)
  - MockWordNetIntegration (semantic_metagraph_mocks.py)

#### 11. AbstractCollectionMock (tests/mocks/base/collection_mock.py)
- **Purpose**: Base class for collection-like mocks
- **Common Methods**: __contains__, __len__, __iter__, keys()
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockVocabularyForHelper (embedding_helper_mocks.py)
  - MockOntology (semantic_metagraph_mocks.py) [partial - has collection aspects]
  - MockKnowledgeBase (semantic_metagraph_mocks.py) [could benefit from collection interface]

#### 12. AbstractEntityMock (tests/mocks/base/entity_mock.py)
- **Purpose**: Base class for domain entity mocks
- **Common Pattern**: Simple attributes representing entity properties
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockLemma (embedding_helper_mocks.py)
  - MockConcept (semantic_metagraph_mocks.py)
  - MockRelation (semantic_metagraph_mocks.py)
  - MockSemanticCluster (semantic_metagraph_mocks.py)
  - MockSemanticPath (semantic_metagraph_mocks.py)

#### 13. AbstractReasoningMock (tests/mocks/base/reasoning_mock.py)
- **Purpose**: Base class for reasoning/inference engine mocks
- **Common Pattern**: Various reasoning methods returning inference results
- **Inherits From**: ABC, Mock
- **Target Classes to Refactor**:
  - MockReasoningEngine (semantic_metagraph_mocks.py)
  - MockEmbeddingModelForDecomposer (semantic_decomposer_mocks.py) [partial - has inference aspects]
  - MockEmbeddingHelperForDecomposer (semantic_decomposer_mocks.py) [partial - has similarity reasoning]

### Implementation Notes

1. **Multiple Inheritance**: Many refactored mock classes will inherit from multiple abstract base classes. For example:
   - MockWordNetIntegration could inherit from both AbstractIntegrationMock and AbstractLibraryWrapperMock
   - MockEntityForGloss could inherit from both AbstractNLPTokenMock and AbstractEntityMock

2. **Method Templates**: Each abstract base class should provide:
   - Default __init__ implementation that calls super().__init__(*args, **kwargs)
   - Abstract properties or methods that subclasses must implement
   - Common helper methods that reduce duplication

3. **Gradual Refactoring**: Start with the most commonly used patterns (AbstractNLPDocMock, AbstractNLPTokenMock, AbstractEdgeCaseMock) to maximize impact

4. **Backwards Compatibility**: Ensure refactored mocks maintain the same interface as existing ones to avoid breaking tests