"""
Mock classes for PerformanceAnalysis tests.
"""

from unittest.mock import Mock
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Import abstract base classes
from tests.mocks.base.operation_mock import AbstractOperationMock
from tests.mocks.base.entity_mock import AbstractEntityMock, EntityType
from tests.mocks.base.integration_mock import AbstractIntegrationMock


@dataclass
class MockPerformanceProfile:
    """Mock performance profiling results for a single operation."""
    execution_time: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_usage_percent: float
    function_calls: int
    graph_nodes: int
    graph_edges: int
    search_depth_reached: int
    beam_expansions: int
    error: Optional[str] = None


@dataclass
class MockScalabilityTestResult:
    """Mock results from scalability testing."""
    graph_size: int
    test_complexity: str
    average_time: float
    memory_usage_mb: float
    success_rate: float
    bottleneck_functions: List[Tuple[str, float]]


@dataclass
class MockBottleneckAnalysis:
    """Mock analysis of performance bottlenecks."""
    slowest_functions: List[Tuple[str, float]]
    memory_hotspots: List[Tuple[str, float]]
    io_operations: List[Tuple[str, float]]
    optimization_recommendations: List[str]


class PerformanceAnalysisMockFactory:
    """Factory class for creating PerformanceAnalysis mock instances."""
    
    def __init__(self):
        self._mock_classes = {
            'MockPerformanceProfiler': MockPerformanceProfiler,
            'MockPerformanceProfilerEdgeCases': MockPerformanceProfilerEdgeCases,
            'MockPerformanceProfilerIntegration': MockPerformanceProfilerIntegration,
            'MockScalabilityTester': MockScalabilityTester,
            'MockScalabilityTesterEdgeCases': MockScalabilityTesterEdgeCases,
            'MockScalabilityTesterIntegration': MockScalabilityTesterIntegration,
            'MockMemoryProfiler': MockMemoryProfiler,
            'MockMemoryProfilerEdgeCases': MockMemoryProfilerEdgeCases,
            'MockMemoryProfilerIntegration': MockMemoryProfilerIntegration,
            'MockComprehensivePerformanceSuite': MockComprehensivePerformanceSuite,
            'MockSMIED': MockSMIED,
            'MockSMIEDEdgeCases': MockSMIEDEdgeCases,
            'MockSMIEDIntegration': MockSMIEDIntegration,
            'MockTestCase': MockTestCase,
            'MockSemanticPathfindingTestSuite': MockSemanticPathfindingTestSuite,
            'MockProcess': MockProcess,
            'MockTracemalloc': MockTracemalloc,
            'MockProfiler': MockProfiler,
            'MockPstats': MockPstats,
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


class MockPerformanceProfiler(AbstractOperationMock):
    """Mock for PerformanceProfiler class testing."""
    
    def __init__(self, verbosity: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbosity = verbosity
        self.profiling_enabled = True
        
        # Set up methods with realistic returns
        self.profile_single_operation = Mock(return_value=MockPerformanceProfile(
            execution_time=0.1,
            memory_peak_mb=10.0,
            memory_current_mb=5.0,
            cpu_usage_percent=50.0,
            function_calls=1000,
            graph_nodes=100,
            graph_edges=200,
            search_depth_reached=5,
            beam_expansions=10
        ))
        
        self.profile_batch_operations = Mock(return_value=[
            MockPerformanceProfile(0.1, 10.0, 5.0, 50.0, 1000, 100, 200, 5, 10),
            MockPerformanceProfile(0.2, 15.0, 8.0, 60.0, 2000, 100, 200, 6, 12)
        ])
        
        self.identify_bottlenecks = Mock(return_value=MockBottleneckAnalysis(
            slowest_functions=[("slow_func", 0.5)],
            memory_hotspots=[("memory_func", 50.0)],
            io_operations=[("io_func", 0.1)],
            optimization_recommendations=["Optimize algorithm", "Reduce memory usage"]
        ))
    
    def execute(self, target: Any, *args, **kwargs) -> Any:
        """Execute profiling operation on target."""
        return self.profile_single_operation(target, *args, **kwargs)
    
    def validate_target(self, target: Any) -> bool:
        """Validate that target can be profiled."""
        return callable(target) or hasattr(target, '__call__')
    
    def get_operation_metadata(self) -> Dict[str, Any]:
        """Get profiler-specific metadata."""
        return {
            'operation_type': 'performance_profiling',
            'verbosity': self.verbosity,
            'profiling_enabled': self.profiling_enabled,
            'supported_metrics': ['execution_time', 'memory_usage', 'cpu_usage', 'function_calls']
        }
    
    def get_primary_attribute(self) -> Any:
        return f"PerformanceProfiler(verbosity={self.verbosity})"
    
    def validate_entity(self) -> bool:
        return self.profiling_enabled
    
    def get_entity_signature(self) -> str:
        return f"profiler:{self.id}:{self.verbosity}"


class MockPerformanceProfilerEdgeCases(MockPerformanceProfiler):
    """Mock for PerformanceProfiler edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configure edge case scenarios
        self.profile_single_operation.side_effect = [
            # Timeout scenario
            MockPerformanceProfile(
                execution_time=60.0,
                memory_peak_mb=0,
                memory_current_mb=0,
                cpu_usage_percent=0,
                function_calls=0,
                graph_nodes=0,
                graph_edges=0,
                search_depth_reached=0,
                beam_expansions=0,
                error="Operation timed out after 60s"
            ),
            # Exception scenario
            MockPerformanceProfile(
                execution_time=0.01,
                memory_peak_mb=0,
                memory_current_mb=0,
                cpu_usage_percent=0,
                function_calls=0,
                graph_nodes=0,
                graph_edges=0,
                search_depth_reached=0,
                beam_expansions=0,
                error="Test exception occurred"
            )
        ]


class MockPerformanceProfilerIntegration(MockPerformanceProfiler):
    """Mock for PerformanceProfiler integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # More realistic integration behavior
        self.profile_single_operation.return_value = MockPerformanceProfile(
            execution_time=0.5,
            memory_peak_mb=25.0,
            memory_current_mb=15.0,
            cpu_usage_percent=75.0,
            function_calls=5000,
            graph_nodes=1000,
            graph_edges=2500,
            search_depth_reached=8,
            beam_expansions=25
        )


class MockScalabilityTester(AbstractOperationMock):
    """Mock for ScalabilityTester class testing."""
    
    def __init__(self, smied_instance=None, verbosity: int = 0, *args, **kwargs):
        super().__init__(entity_type=EntityType.UNKNOWN, *args, **kwargs)
        self.smied = smied_instance or MockSMIED()
        self.verbosity = verbosity
        self.profiler = MockPerformanceProfiler(verbosity)
        
        # Set up methods
        self.test_graph_size_scalability = Mock(return_value=[
            MockScalabilityTestResult(1000, "simple", 0.1, 10.0, 95.0, []),
            MockScalabilityTestResult(2500, "simple", 0.2, 25.0, 90.0, []),
            MockScalabilityTestResult(5000, "simple", 0.4, 50.0, 85.0, [])
        ])
        
        self.test_search_depth_scalability = Mock(return_value=[
            MockScalabilityTestResult(0, "depth_3", 0.1, 10.0, 100.0, []),
            MockScalabilityTestResult(0, "depth_5", 0.2, 15.0, 95.0, []),
            MockScalabilityTestResult(0, "depth_8", 0.5, 30.0, 80.0, [])
        ])
        
        self.test_beam_width_scalability = Mock(return_value=[
            MockScalabilityTestResult(0, "beam_1", 0.1, 8.0, 100.0, []),
            MockScalabilityTestResult(0, "beam_3", 0.3, 20.0, 95.0, []),
            MockScalabilityTestResult(0, "beam_5", 0.5, 35.0, 90.0, [])
        ])
    
    def execute(self, target: Any, *args, **kwargs) -> Any:
        """Execute scalability test on target."""
        return self.test_graph_size_scalability(target, *args, **kwargs)
    
    def validate_target(self, target: Any) -> bool:
        """Validate that target supports scalability testing."""
        return hasattr(target, 'build_synset_graph') or callable(target)
    
    def get_operation_metadata(self) -> Dict[str, Any]:
        """Get scalability tester metadata."""
        return {
            'operation_type': 'scalability_testing',
            'verbosity': self.verbosity,
            'test_types': ['graph_size', 'search_depth', 'beam_width'],
            'smied_available': self.smied is not None
        }
    
    def get_primary_attribute(self) -> Any:
        return f"ScalabilityTester(verbosity={self.verbosity})"
    
    def validate_entity(self) -> bool:
        return self.smied is not None
    
    def get_entity_signature(self) -> str:
        return f"scalability_tester:{self.id}:{self.verbosity}"


class MockScalabilityTesterEdgeCases(MockScalabilityTester):
    """Mock for ScalabilityTester edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configure edge cases
        self.test_graph_size_scalability.return_value = []
        self.test_search_depth_scalability.side_effect = Exception("Test exception")
        self.test_beam_width_scalability.return_value = [
            MockScalabilityTestResult(0, "beam_1", 0.1, 8.0, 0.0, [])  # 0% success rate
        ]


class MockScalabilityTesterIntegration(MockScalabilityTester):
    """Mock for ScalabilityTester integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # More detailed integration results
        self.test_graph_size_scalability.return_value = [
            MockScalabilityTestResult(1000, "simple", 0.05, 5.0, 100.0, [("func1", 0.02)]),
            MockScalabilityTestResult(5000, "simple", 0.25, 25.0, 95.0, [("func1", 0.15), ("func2", 0.08)]),
            MockScalabilityTestResult(10000, "simple", 0.60, 75.0, 85.0, [("func1", 0.40), ("func2", 0.15)])
        ]


class MockMemoryProfiler(AbstractOperationMock):
    """Mock for MemoryProfiler class testing."""
    
    def __init__(self, verbosity: int = 0, *args, **kwargs):
        super().__init__(entity_type=EntityType.UNKNOWN, *args, **kwargs)
        self.verbosity = verbosity
        
        # Set up methods
        self.profile_graph_construction = Mock(return_value={
            'construction_time': 2.5,
            'initial_memory_mb': 100.0,
            'final_memory_mb': 150.0,
            'peak_traced_mb': 175.0,
            'memory_increase_mb': 50.0,
            'nodes_count': 10000,
            'edges_count': 25000,
            'memory_per_node_kb': 5.0,
            'memory_per_edge_kb': 2.0
        })
        
        self.profile_pathfinding_memory = Mock(return_value={
            'total_tests': 10,
            'successful_tests': 8,
            'failed_tests': 2,
            'average_memory_delta_mb': 2.5,
            'max_memory_delta_mb': 5.0,
            'average_peak_memory_mb': 15.0,
            'max_peak_memory_mb': 25.0,
            'individual_profiles': []
        })
    
    def execute(self, target: Any, *args, **kwargs) -> Any:
        """Execute memory profiling operation on target."""
        return self.profile_graph_construction(target, *args, **kwargs)
    
    def validate_target(self, target: Any) -> bool:
        """Validate that target supports memory profiling."""
        return hasattr(target, 'build_synset_graph') or callable(target)
    
    def get_operation_metadata(self) -> Dict[str, Any]:
        """Get memory profiler metadata."""
        return {
            'operation_type': 'memory_profiling',
            'verbosity': self.verbosity,
            'profiling_types': ['graph_construction', 'pathfinding_memory'],
            'metrics': ['memory_usage', 'construction_time', 'peak_memory']
        }
    
    def get_primary_attribute(self) -> Any:
        return f"MemoryProfiler(verbosity={self.verbosity})"
    
    def validate_entity(self) -> bool:
        return True
    
    def get_entity_signature(self) -> str:
        return f"memory_profiler:{self.id}:{self.verbosity}"


class MockMemoryProfilerEdgeCases(MockMemoryProfiler):
    """Mock for MemoryProfiler edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configure edge cases
        self.profile_graph_construction.return_value = {'error': 'Graph construction failed'}
        self.profile_pathfinding_memory.side_effect = Exception("Memory profiling error")


class MockMemoryProfilerIntegration(MockMemoryProfiler):
    """Mock for MemoryProfiler integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # More detailed integration results
        self.profile_pathfinding_memory.return_value = {
            'total_tests': 20,
            'successful_tests': 18,
            'failed_tests': 2,
            'average_memory_delta_mb': 3.2,
            'max_memory_delta_mb': 8.5,
            'average_peak_memory_mb': 22.0,
            'max_peak_memory_mb': 45.0,
            'individual_profiles': [
                {
                    'test_case': 'cat->chase->mouse',
                    'success': True,
                    'execution_time': 0.15,
                    'initial_memory_mb': 120.0,
                    'final_memory_mb': 123.2,
                    'peak_traced_mb': 25.0,
                    'memory_delta_mb': 3.2
                }
            ]
        }


class MockComprehensivePerformanceSuite(AbstractIntegrationMock):
    """Mock for ComprehensivePerformanceSuite class testing."""
    
    def __init__(self, smied_instance=None, verbosity: int = 0, *args, **kwargs):
        super().__init__(entity_type=EntityType.UNKNOWN, *args, **kwargs)
        self.smied = smied_instance or MockSMIED()
        self.verbosity = verbosity
        self.profiler = MockPerformanceProfiler(verbosity)
        self.scalability_tester = MockScalabilityTester(smied_instance, verbosity)
        self.memory_profiler = MockMemoryProfiler(verbosity)
        
        # Set up methods
        self.run_comprehensive_analysis = Mock(return_value={
            'timestamp': 1234567890.0,
            'analysis_sections': {
                'graph_construction_memory': {
                    'construction_time': 2.5,
                    'memory_increase_mb': 50.0,
                    'nodes_count': 10000,
                    'edges_count': 25000
                },
                'depth_scalability': [
                    {'complexity': 'depth_3', 'average_time': 0.1, 'memory_usage_mb': 10.0, 'success_rate': 100.0},
                    {'complexity': 'depth_8', 'average_time': 0.5, 'memory_usage_mb': 30.0, 'success_rate': 80.0}
                ],
                'beam_scalability': [
                    {'complexity': 'beam_1', 'average_time': 0.1, 'memory_usage_mb': 8.0, 'success_rate': 100.0},
                    {'complexity': 'beam_5', 'average_time': 0.5, 'memory_usage_mb': 35.0, 'success_rate': 90.0}
                ],
                'pathfinding_memory': {
                    'total_tests': 10,
                    'successful_tests': 8,
                    'average_memory_delta_mb': 2.5,
                    'max_peak_memory_mb': 25.0
                }
            }
        })
        
        self.print_performance_summary = Mock()
    
    def setup_integration_components(self) -> Dict[str, Any]:
        """Set up all components required for performance analysis integration."""
        return {
            'smied': self.smied,
            'profiler': self.profiler,
            'scalability_tester': self.scalability_tester,
            'memory_profiler': self.memory_profiler
        }
    
    def configure_component_interactions(self) -> None:
        """Configure how performance analysis components interact."""
        # Mock configuration of component interactions
        self.register_component('smied', self.smied)
        self.register_component('profiler', self.profiler)
        self.register_component('scalability_tester', self.scalability_tester, dependencies=['smied'])
        self.register_component('memory_profiler', self.memory_profiler, dependencies=['smied'])
    
    def validate_integration_state(self) -> bool:
        """Validate that the performance analysis integration is in a consistent state."""
        return (self.smied is not None and 
                self.profiler is not None and
                self.scalability_tester is not None and
                self.memory_profiler is not None)
    
    def get_primary_attribute(self) -> Any:
        return f"ComprehensivePerformanceSuite(verbosity={self.verbosity})"
    
    def validate_entity(self) -> bool:
        return self.smied is not None
    
    def get_entity_signature(self) -> str:
        return f"performance_suite:{self.id}:{self.verbosity}"


class MockSMIED(AbstractEntityMock):
    """Mock for SMIED class testing."""
    
    def __init__(self, nlp_model=None, auto_download=False, verbosity: int = 0, *args, **kwargs):
        super().__init__(entity_type=EntityType.UNKNOWN, *args, **kwargs)
        self.nlp_model = nlp_model
        self.auto_download = auto_download
        self.verbosity = verbosity
        self.synset_graph = None
        
        # Set up methods
        self.build_synset_graph = Mock(return_value=MockGraph())
        self.analyze_triple = Mock(return_value=(True, True, False))
    
    def get_primary_attribute(self) -> Any:
        return f"SMIED(verbosity={self.verbosity})"
    
    def validate_entity(self) -> bool:
        return True
    
    def get_entity_signature(self) -> str:
        return f"smied:{self.id}:{self.verbosity}"


class MockSMIEDEdgeCases(MockSMIED):
    """Mock for SMIED edge cases testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configure edge cases
        self.build_synset_graph.side_effect = Exception("Graph building failed")
        self.analyze_triple.return_value = (False, False, False)


class MockSMIEDIntegration(MockSMIED):
    """Mock for SMIED integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # More realistic integration behavior
        mock_graph = MockGraph()
        mock_graph.number_of_nodes.return_value = 15000
        mock_graph.number_of_edges.return_value = 35000
        self.build_synset_graph.return_value = mock_graph


class MockTestCase(AbstractEntityMock):
    """Mock for test case data."""
    
    def __init__(self, subject: str = "cat", predicate: str = "chase", 
                 object_term: str = "mouse", expected: bool = True,
                 description: str = "test", category: str = "test",
                 difficulty_level: str = "easy", *args, **kwargs):
        super().__init__(entity_type=EntityType.UNKNOWN, *args, **kwargs)
        self.subject = subject
        self.predicate = predicate
        self.object = object_term  # Using object_term to avoid Python keyword
        self.expected = expected
        self.description = description
        self.category = category
        self.difficulty_level = difficulty_level
    
    def get_primary_attribute(self) -> Any:
        return f"{self.subject}->{self.predicate}->{self.object}"
    
    def validate_entity(self) -> bool:
        return bool(self.subject and self.predicate and self.object)
    
    def get_entity_signature(self) -> str:
        return f"test_case:{self.id}:{self.subject}_{self.predicate}_{self.object}"


class MockSemanticPathfindingTestSuite(Mock):
    """Mock for SemanticPathfindingTestSuite class."""
    
    def __init__(self, smied_instance=None, verbosity: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smied = smied_instance or MockSMIED()
        self.verbosity = verbosity
        
        # Mock test cases
        self.CROSS_POS_TEST_CASES = [
            MockTestCase("cat", "chase", "mouse", True, "predator-prey", "action", "easy"),
            MockTestCase("dog", "bark", "intruder", True, "protective", "action", "medium"),
            MockTestCase("bird", "build", "nest", True, "natural", "creation", "easy"),
            MockTestCase("fish", "swim", "ocean", True, "habitat", "movement", "easy"),
            MockTestCase("human", "write", "book", True, "creative", "creation", "medium"),
        ]
        
        # Set up methods
        self.run_single_test = Mock(return_value={'success': True, 'result': (True, True, False)})


class MockGraph(Mock):
    """Mock for graph objects."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_nodes = Mock(return_value=1000)
        self.number_of_edges = Mock(return_value=2500)


class MockProcess(Mock):
    """Mock for psutil.Process."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_times = Mock(return_value=Mock(user=1.0, system=0.5))
        self.memory_info = Mock(return_value=Mock(rss=1024*1024*100))  # 100MB


class MockTracemalloc(Mock):
    """Mock for tracemalloc module."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = Mock()
        self.stop = Mock()
        self.get_traced_memory = Mock(return_value=(1024*1024*5, 1024*1024*10))  # 5MB current, 10MB peak
        self.is_tracing = Mock(return_value=True)


class MockProfiler(Mock):
    """Mock for cProfile.Profile."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable = Mock()
        self.disable = Mock()


class MockPstats(Mock):
    """Mock for pstats.Stats."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_calls = 1000
        self.sort_stats = Mock(return_value=self)