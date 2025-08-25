"""
Mock classes for optimization demo tests.

This module provides mock implementations following the SMIED Testing Framework
Design Specifications with factory pattern and abstract base class hierarchy.
"""

from unittest.mock import Mock
import time
from typing import List, Optional, Any, Dict, Tuple
from dataclasses import dataclass

# Import abstract base classes
from tests.mocks.base.entity_mock import AbstractEntityMock, EntityType
from tests.mocks.base.algorithmic_function_mock import AbstractAlgorithmicFunctionMock
from tests.mocks.base.reasoning_mock import AbstractReasoningMock, ReasoningType, InferenceStrategy


class OptimizationDemoMockFactory:
    """Factory class for creating optimization demo mock instances.
    
    This factory follows the SMIED Testing Framework Design Specifications
    for mock creation using factory pattern with abstract base class hierarchy.
    """
    
    def __init__(self):
        self._mock_classes = {
            # Core optimization demo mocks
            'MockOptimizationDemo': MockOptimizationDemo,
            'MockOptimizationDemoValidation': MockOptimizationDemoValidation,
            'MockOptimizationDemoEdgeCases': MockOptimizationDemoEdgeCases,
            'MockOptimizationDemoIntegration': MockOptimizationDemoIntegration,
            
            # SMIED component mocks for optimization
            'MockSMIEDForOptimization': MockSMIEDForOptimization,
            'MockSMIEDOptimized': MockSMIEDOptimized,
            'MockSMIEDOriginal': MockSMIEDOriginal,
            
            # Parameter and configuration mocks
            'MockParameterSet': MockParameterSet,
            'MockOptimizationParameters': MockOptimizationParameters,
            'MockOriginalParameters': MockOriginalParameters,
            
            # Result and analysis mocks
            'MockOptimizationResult': MockOptimizationResult,
            'MockAnalysisResult': MockAnalysisResult,
            'MockPerformanceComparison': MockPerformanceComparison,
            
            # Test case mocks
            'MockOptimizationTestCase': MockOptimizationTestCase,
            'MockChallengingTestCase': MockChallengingTestCase,
            'MockSimpleTestCase': MockSimpleTestCase,
            
            # Summary and reporting mocks
            'MockOptimizationSummary': MockOptimizationSummary,
            'MockPerformanceMetrics': MockPerformanceMetrics,
            'MockExecutionTimer': MockExecutionTimer,
        }
    
    def __call__(self, mock_name: str, *args, **kwargs) -> Mock:
        """
        Create and return a mock instance by name.
        
        Args:
            mock_name: Name of the mock class to instantiate
            *args: Arguments to pass to the mock constructor
            **kwargs: Keyword arguments to pass to the mock constructor
            
        Returns:
            Mock instance of the specified type
            
        Raises:
            ValueError: If mock_name is not found
        """
        if mock_name not in self._mock_classes:
            available = ', '.join(self._mock_classes.keys())
            raise ValueError(f"Mock '{mock_name}' not found. Available mocks: {available}")
        
        mock_class = self._mock_classes[mock_name]
        return mock_class(*args, **kwargs)
    
    def get_available_mocks(self) -> List[str]:
        """Return list of available mock class names."""
        return list(self._mock_classes.keys())


class MockOptimizationDemo(AbstractReasoningMock):
    """Basic mock implementation for optimization demonstration functionality."""
    
    def __init__(self, verbosity: int = 1, *args, **kwargs):
        super().__init__(
            reasoning_type=ReasoningType.ANALOGICAL,
            inference_strategy=InferenceStrategy.BEST_FIRST,
            *args, **kwargs
        )
        self.verbosity = verbosity
        self.optimization_results = []
        self.execution_times = []
        
        # Mock methods
        self.demonstrate_optimization = Mock(return_value=True)
        self.show_optimization_summary = Mock()
        self.initialize_smied = Mock(return_value=Mock())
        
    def get_primary_attribute(self) -> Any:
        return "optimization_demo"
    
    def validate_entity(self) -> bool:
        return True
    
    def get_entity_signature(self) -> str:
        return f"optimization_demo:mock"
    
    def infer(self, premises: List[Any], conclusion: Any) -> bool:
        """Mock implementation of abstract infer method."""
        return True
    
    def explain_reasoning(self) -> str:
        """Mock implementation of abstract explain_reasoning method."""
        return "Mock optimization demonstration reasoning"
    
    def compute_similarity(self, other: Any) -> float:
        """Mock implementation of abstract compute_similarity method."""
        return 0.8


class MockOptimizationDemoValidation(MockOptimizationDemo):
    """Mock for optimization demo validation tests."""
    
    def __init__(self, strict_validation: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strict_validation = strict_validation
        
        # Validation-specific mocks
        self.validate_parameters = Mock(return_value=True)
        self.validate_inputs = Mock(return_value=True)
        self.validate_results = Mock(return_value=True)
        self.check_optimization_improvements = Mock(return_value=True)


class MockOptimizationDemoEdgeCases(MockOptimizationDemo):
    """Mock for optimization demo edge cases and error conditions."""
    
    def __init__(self, mock_failures: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_failures = mock_failures
        self.failure_scenarios = []
        
        # Edge case specific mocks
        self.handle_empty_inputs = Mock(side_effect=ValueError("Empty input"))
        self.handle_invalid_parameters = Mock(side_effect=ValueError("Invalid parameters"))
        self.handle_missing_dependencies = Mock(side_effect=ImportError("Missing dependency"))
        
        if mock_failures:
            self.demonstrate_optimization = Mock(return_value=False)


class MockOptimizationDemoIntegration(MockOptimizationDemo):
    """Mock for optimization demo integration testing."""
    
    def __init__(self, full_component_mocking: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_component_mocking = full_component_mocking
        self.component_interactions = []
        
        # Integration-specific mocks
        self.test_full_pipeline = Mock(return_value=True)
        self.validate_component_integration = Mock(return_value=True)
        self.check_parameter_propagation = Mock(return_value=True)


class MockSMIEDForOptimization(AbstractEntityMock):
    """Mock SMIED implementation for optimization testing."""
    
    def __init__(self, 
                 verbosity: int = 1, 
                 build_graph_on_init: bool = False,
                 optimization_enabled: bool = True,
                 *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.verbosity = verbosity
        self.build_graph_on_init = build_graph_on_init
        self.optimization_enabled = optimization_enabled
        
        # Mock SMIED methods
        self.analyze_triple = Mock(return_value=self._create_mock_analysis_result())
        self.initialize_graph = Mock()
        self.set_parameters = Mock()
        
    def _create_mock_analysis_result(self):
        """Create a mock analysis result."""
        if self.optimization_enabled:
            # Successful analysis with optimized parameters
            subject_path = ["fox", "animal", "creature"]
            object_path = ["dog", "animal", "creature"]
            connecting_predicate = "interact"
            return (subject_path, object_path, connecting_predicate)
        else:
            # Failed analysis with original parameters
            return (None, None, None)
    
    def get_primary_attribute(self) -> Any:
        return "smied_optimization"
    
    def validate_entity(self) -> bool:
        return True
    
    def get_entity_signature(self) -> str:
        return f"smied:optimization:mock"


class MockSMIEDOptimized(MockSMIEDForOptimization):
    """Mock SMIED with optimized parameters."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(optimization_enabled=True, *args, **kwargs)
        
        # Optimized parameter values
        self.max_depth = 10
        self.beam_width = 10
        self.len_tolerance = 3
        self.relax_beam = True
        self.heuristic = 'hybrid'


class MockSMIEDOriginal(MockSMIEDForOptimization):
    """Mock SMIED with original parameters."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(optimization_enabled=False, *args, **kwargs)
        
        # Original parameter values
        self.max_depth = 6
        self.beam_width = 3
        self.len_tolerance = 0
        self.relax_beam = False
        self.heuristic = 'embedding'


class MockParameterSet(AbstractEntityMock):
    """Mock parameter set for optimization testing."""
    
    def __init__(self, 
                 max_depth: int = 10,
                 beam_width: int = 10,
                 len_tolerance: int = 3,
                 relax_beam: bool = True,
                 heuristic: str = 'hybrid',
                 *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.len_tolerance = len_tolerance
        self.relax_beam = relax_beam
        self.heuristic = heuristic
        
        # Mock methods
        self.validate_parameters = Mock(return_value=True)
        self.apply_to_smied = Mock()
        
    def get_primary_attribute(self) -> Any:
        return f"params_{self.beam_width}_{self.max_depth}"
    
    def validate_entity(self) -> bool:
        return (self.max_depth > 0 and 
                self.beam_width > 0 and 
                self.len_tolerance >= 0)
    
    def get_entity_signature(self) -> str:
        return f"parameters:{self.max_depth}:{self.beam_width}:{self.len_tolerance}"


class MockOptimizationParameters(MockParameterSet):
    """Mock optimized parameter set."""
    
    def __init__(self, *args, **kwargs):
        # Set default optimized values, but allow override from kwargs
        defaults = {
            'max_depth': 10,
            'beam_width': 10,
            'len_tolerance': 3,
            'relax_beam': True,
            'heuristic': 'hybrid'
        }
        # Update defaults with any provided kwargs
        defaults.update(kwargs)
        super().__init__(*args, **defaults)


class MockOriginalParameters(MockParameterSet):
    """Mock original parameter set."""
    
    def __init__(self, *args, **kwargs):
        # Set default original values, but allow override from kwargs
        defaults = {
            'max_depth': 6,
            'beam_width': 3,
            'len_tolerance': 0,
            'relax_beam': False,
            'heuristic': 'embedding'
        }
        # Update defaults with any provided kwargs
        defaults.update(kwargs)
        super().__init__(*args, **defaults)


class MockOptimizationResult(AbstractEntityMock):
    """Mock optimization result."""
    
    def __init__(self, 
                 success: bool = True,
                 execution_time: float = 2.5,
                 subject_path: Optional[List[str]] = None,
                 object_path: Optional[List[str]] = None,
                 connecting_predicate: Optional[str] = None,
                 *args, **kwargs):
        super().__init__(entity_type=EntityType.PATH, *args, **kwargs)
        self.success = success
        self.execution_time = execution_time
        self.subject_path = subject_path or ["fox", "animal"] if success else None
        self.object_path = object_path or ["dog", "animal"] if success else None
        self.connecting_predicate = connecting_predicate or "interact" if success else None
        
        # Mock methods
        self.get_path_quality = Mock(return_value=0.8 if success else 0.0)
        self.format_result = Mock(return_value=f"Result: {success}")
        
    def get_primary_attribute(self) -> Any:
        return self.success
    
    def validate_entity(self) -> bool:
        return self.execution_time >= 0
    
    def get_entity_signature(self) -> str:
        return f"result:{self.success}:{self.execution_time}"


class MockAnalysisResult(MockOptimizationResult):
    """Mock analysis result with detailed information."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Analysis-specific attributes
        self.analysis_depth = 5
        self.nodes_explored = 150
        self.path_length = 3
        
        # Mock analysis methods
        self.get_analysis_statistics = Mock(return_value={
            'nodes_explored': self.nodes_explored,
            'path_length': self.path_length,
            'analysis_depth': self.analysis_depth
        })


class MockPerformanceComparison(AbstractEntityMock):
    """Mock performance comparison between optimized and original."""
    
    def __init__(self, 
                 improvement_percent: float = 25.0,
                 original_success_rate: float = 0.6,
                 optimized_success_rate: float = 0.85,
                 *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.improvement_percent = improvement_percent
        self.original_success_rate = original_success_rate
        self.optimized_success_rate = optimized_success_rate
        
        # Mock comparison methods
        self.calculate_improvement = Mock(return_value=improvement_percent)
        self.generate_comparison_report = Mock(return_value="Performance improved by 25%")
        
    def get_primary_attribute(self) -> Any:
        return self.improvement_percent
    
    def validate_entity(self) -> bool:
        return (0 <= self.original_success_rate <= 1 and 
                0 <= self.optimized_success_rate <= 1)
    
    def get_entity_signature(self) -> str:
        return f"comparison:{self.improvement_percent}%"


class MockOptimizationTestCase(AbstractEntityMock):
    """Mock test case for optimization demonstration."""
    
    def __init__(self, 
                 subject: str = "fox",
                 predicate: str = "jump", 
                 object: str = "dog",
                 expected_success: bool = True,
                 difficulty_level: str = "hard",
                 requires_optimization: bool = True,
                 *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.expected_success = expected_success
        self.difficulty_level = difficulty_level
        self.requires_optimization = requires_optimization
        
        # Mock test case methods
        self.run_with_parameters = Mock(return_value=Mock(success=expected_success))
        self.validate_inputs = Mock(return_value=True)
        
    def get_primary_attribute(self) -> Any:
        return f"{self.subject}_{self.predicate}_{self.object}"
    
    def validate_entity(self) -> bool:
        return bool(self.subject and self.predicate and self.object)
    
    def get_entity_signature(self) -> str:
        return f"testcase:{self.subject}:{self.predicate}:{self.object}"


class MockChallengingTestCase(MockOptimizationTestCase):
    """Mock challenging test case that requires optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            difficulty_level="very_hard",
            requires_optimization=True,
            expected_success=False,  # Typically fails without optimization
            *args, **kwargs
        )


class MockSimpleTestCase(MockOptimizationTestCase):
    """Mock simple test case that works without optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            subject="cat",
            predicate="chase",
            object="mouse",
            difficulty_level="easy",
            requires_optimization=False,
            expected_success=True,
            *args, **kwargs
        )


class MockOptimizationSummary(AbstractEntityMock):
    """Mock optimization summary for demonstration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        
        # Summary data
        self.optimization_categories = [
            "Parameter Optimization",
            "Heuristic Enhancement", 
            "Missing Relations Integration",
            "Analysis & Validation"
        ]
        
        # Mock summary methods
        self.display_summary = Mock()
        self.get_optimization_details = Mock(return_value=self.optimization_categories)
        self.format_improvements = Mock(return_value="20-30% improvement expected")
        
    def get_primary_attribute(self) -> Any:
        return "optimization_summary"
    
    def validate_entity(self) -> bool:
        return len(self.optimization_categories) > 0
    
    def get_entity_signature(self) -> str:
        return "summary:optimization:demo"


class MockPerformanceMetrics(AbstractEntityMock):
    """Mock performance metrics for optimization validation."""
    
    def __init__(self, 
                 execution_time: float = 2.5,
                 success_rate: float = 0.85,
                 path_quality: float = 0.8,
                 *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.execution_time = execution_time
        self.success_rate = success_rate
        self.path_quality = path_quality
        
        # Mock metrics methods
        self.calculate_metrics = Mock(return_value={
            'execution_time': execution_time,
            'success_rate': success_rate,
            'path_quality': path_quality
        })
        self.meets_performance_threshold = Mock(return_value=True)
        
    def get_primary_attribute(self) -> Any:
        return self.success_rate
    
    def validate_entity(self) -> bool:
        return (self.execution_time >= 0 and 
                0 <= self.success_rate <= 1 and 
                0 <= self.path_quality <= 1)
    
    def get_entity_signature(self) -> str:
        return f"metrics:{self.success_rate}:{self.execution_time}"


class MockExecutionTimer(AbstractEntityMock):
    """Mock execution timer for performance measurement."""
    
    def __init__(self, mock_execution_time: float = 2.5, *args, **kwargs):
        super().__init__(entity_type=EntityType.CONCEPT, *args, **kwargs)
        self.mock_execution_time = mock_execution_time
        self.start_time = None
        
        # Mock timer methods
        self.start = Mock(side_effect=self._start_timer)
        self.stop = Mock(side_effect=self._stop_timer, return_value=mock_execution_time)
        self.get_elapsed_time = Mock(return_value=mock_execution_time)
        
    def _start_timer(self):
        """Mock start timer implementation."""
        self.start_time = time.time()
        
    def _stop_timer(self):
        """Mock stop timer implementation."""
        return self.mock_execution_time
        
    def get_primary_attribute(self) -> Any:
        return self.mock_execution_time
    
    def validate_entity(self) -> bool:
        return self.mock_execution_time >= 0
    
    def get_entity_signature(self) -> str:
        return f"timer:{self.mock_execution_time}"