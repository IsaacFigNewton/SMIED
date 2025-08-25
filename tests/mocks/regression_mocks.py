"""
Mock implementations for regression testing functionality.

This module provides mock factories and specialized mock classes for regression
testing components including baseline management, regression testing, and
historical tracking. Follows the SMIED Testing Framework architecture.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass
import uuid
import json
import time

from tests.mocks.base.entity_mock import AbstractEntityMock, EntityType, EntityStatus
from tests.mocks.base.operation_mock import AbstractOperationMock


class BaselineResultMock(AbstractEntityMock):
    """Mock implementation of BaselineResult for regression testing."""
    
    def __init__(self, 
                 test_id: str = "test_001",
                 subject: str = "cat",
                 predicate: str = "chase",
                 object: str = "mouse",
                 success: bool = True,
                 subject_path_length: int = 2,
                 object_path_length: int = 2,
                 connecting_predicate: Optional[str] = "chase.v.01",
                 execution_time: float = 0.1,
                 version: str = "1.0",
                 *args, **kwargs):
        """Initialize BaselineResultMock."""
        super().__init__(entity_id=test_id, entity_type=EntityType.NODE, *args, **kwargs)
        
        # Core test data
        self.test_id = test_id
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.success = success
        
        # Path information
        self.subject_path_length = subject_path_length
        self.object_path_length = object_path_length
        self.connecting_predicate = connecting_predicate
        
        # Performance data
        self.execution_time = execution_time
        self.version = version
        
        # Mock specific attributes
        self.name = f"{subject}_{predicate}_{object}"
        self.label = f"Baseline: {self.name}"
        
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute (test_id)."""
        return self.test_id
    
    def validate_entity(self) -> bool:
        """Validate baseline result entity."""
        return (bool(self.test_id) and 
                bool(self.subject) and 
                bool(self.predicate) and 
                bool(self.object) and
                self.subject_path_length >= 0 and
                self.object_path_length >= 0 and
                self.execution_time >= 0)
    
    def get_entity_signature(self) -> str:
        """Get unique signature for this baseline."""
        return f"baseline:{self.test_id}:{self.version}"


class RegressionTestResultMock(AbstractEntityMock):
    """Mock implementation of RegressionTestResult."""
    
    def __init__(self,
                 test_id: str = "test_001",
                 baseline_success: bool = True,
                 current_success: bool = True,
                 baseline_path_length: int = 4,
                 current_path_length: int = 4,
                 performance_change_percent: float = 0.0,
                 status: str = 'PASS',
                 details: str = "No changes",
                 *args, **kwargs):
        """Initialize RegressionTestResultMock."""
        super().__init__(entity_id=test_id, entity_type=EntityType.NODE, *args, **kwargs)
        
        self.test_id = test_id
        self.baseline_success = baseline_success
        self.current_success = current_success
        self.baseline_path_length = baseline_path_length
        self.current_path_length = current_path_length
        self.performance_change_percent = performance_change_percent
        self.status = status
        self.details = details
        
        self.name = f"result_{test_id}"
        self.label = f"Regression Result: {test_id}"
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute (test_id)."""
        return self.test_id
    
    def validate_entity(self) -> bool:
        """Validate regression result entity."""
        return (bool(self.test_id) and 
                self.status in ['PASS', 'FAIL', 'PERFORMANCE_REGRESSION', 'IMPROVEMENT'] and
                self.baseline_path_length >= 0 and
                self.current_path_length >= 0)
    
    def get_entity_signature(self) -> str:
        """Get unique signature for this result."""
        return f"regression_result:{self.test_id}:{self.status}"


class MockRegressionBaseline(Mock):
    """Mock implementation of RegressionBaseline."""
    
    def __init__(self, baseline_file: str = "mock_baseline.json", *args, **kwargs):
        """Initialize MockRegressionBaseline."""
        super().__init__(*args, **kwargs)
        self.baseline_file = baseline_file
        self.baseline_path = f"/mock/path/{baseline_file}"
        self._baselines = {}
        
        # Configure mock methods
        self.load_baseline = Mock(return_value=None)
        self.save_baseline = Mock(return_value=None)
        self.add_baseline = Mock(side_effect=self._add_baseline_impl)
        self.get_baseline = Mock(side_effect=self._get_baseline_impl)
        self.has_baseline = Mock(side_effect=self._has_baseline_impl)
        self.get_all_baselines = Mock(side_effect=self._get_all_baselines_impl)
        self.create_test_id = Mock(side_effect=self._create_test_id_impl)
    
    def _add_baseline_impl(self, result: BaselineResultMock):
        """Implementation for add_baseline."""
        self._baselines[result.test_id] = result
    
    def _get_baseline_impl(self, test_id: str) -> Optional[BaselineResultMock]:
        """Implementation for get_baseline."""
        return self._baselines.get(test_id)
    
    def _has_baseline_impl(self, test_id: str) -> bool:
        """Implementation for has_baseline."""
        return test_id in self._baselines
    
    def _get_all_baselines_impl(self) -> Dict[str, BaselineResultMock]:
        """Implementation for get_all_baselines."""
        return self._baselines.copy()
    
    def _create_test_id_impl(self, subject: str, predicate: str, object: str) -> str:
        """Implementation for create_test_id."""
        import hashlib
        content = f"{subject}_{predicate}_{object}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class MockRegressionTester(Mock):
    """Mock implementation of RegressionTester."""
    
    def __init__(self, smied_instance: Mock = None, verbosity: int = 0, *args, **kwargs):
        """Initialize MockRegressionTester."""
        super().__init__(*args, **kwargs)
        self.smied = smied_instance or Mock()
        self.verbosity = verbosity
        self.baseline = MockRegressionBaseline()
        self.performance_tolerance = 0.5
        
        # Configure mock methods
        self.establish_baseline = Mock(side_effect=self._establish_baseline_impl)
        self.run_regression_tests = Mock(side_effect=self._run_regression_tests_impl)
        self.generate_regression_report = Mock(side_effect=self._generate_regression_report_impl)
        self.print_regression_report = Mock(return_value=None)
        self._determine_regression_status = Mock(side_effect=self._determine_status_impl)
        self._create_status_details = Mock(side_effect=self._create_details_impl)
    
    def _establish_baseline_impl(self, test_cases: List[Mock], force_update: bool = False):
        """Implementation for establish_baseline."""
        for test_case in test_cases:
            test_id = f"test_{uuid.uuid4().hex[:8]}"
            baseline = BaselineResultMock(
                test_id=test_id,
                subject=getattr(test_case, 'subject', 'mock_subject'),
                predicate=getattr(test_case, 'predicate', 'mock_predicate'),
                object=getattr(test_case, 'object', 'mock_object')
            )
            self.baseline._add_baseline_impl(baseline)
    
    def _run_regression_tests_impl(self, test_cases: List[Mock]) -> List[RegressionTestResultMock]:
        """Implementation for run_regression_tests."""
        results = []
        for test_case in test_cases:
            test_id = f"test_{uuid.uuid4().hex[:8]}"
            result = RegressionTestResultMock(test_id=test_id)
            results.append(result)
        return results
    
    def _generate_regression_report_impl(self, results: List[RegressionTestResultMock]) -> Dict[str, Any]:
        """Implementation for generate_regression_report."""
        total_tests = len(results)
        passed = len([r for r in results if r.status == 'PASS'])
        failed = len([r for r in results if r.status == 'FAIL'])
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'performance_regressions': 0,
                'improvements': 0,
                'pass_rate': (passed / total_tests * 100) if total_tests > 0 else 0
            },
            'performance': {
                'average_performance_change': 0.0,
                'tests_with_perf_changes': 0
            },
            'regressions': {
                'functional_regressions': 0,
                'new_failures': failed
            },
            'details': {
                'failed_tests': [],
                'improved_tests': []
            }
        }
    
    def _determine_status_impl(self, baseline: Mock, current_result: Mock, perf_change: float) -> str:
        """Implementation for _determine_regression_status."""
        if hasattr(baseline, 'success') and hasattr(current_result, 'success'):
            if baseline.success and not current_result.success:
                return 'FAIL'
            elif not baseline.success and current_result.success:
                return 'IMPROVEMENT'
        
        if abs(perf_change) > 50:  # 50% threshold
            return 'PERFORMANCE_REGRESSION' if perf_change > 0 else 'IMPROVEMENT'
        
        return 'PASS'
    
    def _create_details_impl(self, baseline: Mock, current_result: Mock, perf_change: float) -> str:
        """Implementation for _create_status_details."""
        return f"Mock test details: perf_change={perf_change:.1f}%"


class MockLongTermRegressionTracker(Mock):
    """Mock implementation of LongTermRegressionTracker."""
    
    def __init__(self, history_file: str = "mock_history.json", *args, **kwargs):
        """Initialize MockLongTermRegressionTracker."""
        super().__init__(*args, **kwargs)
        self.history_file = history_file
        self.history_path = f"/mock/path/{history_file}"
        self.history = {'runs': [], 'metadata': {'created': time.time()}}
        
        # Configure mock methods
        self.load_history = Mock(return_value=self.history)
        self.save_history = Mock(return_value=None)
        self.record_regression_run = Mock(side_effect=self._record_run_impl)
        self.get_trend_analysis = Mock(side_effect=self._get_trend_analysis_impl)
    
    def _record_run_impl(self, results: List[Mock], metadata: Dict[str, Any] = None):
        """Implementation for record_regression_run."""
        run_data = {
            'timestamp': time.time(),
            'total_tests': len(results),
            'passed': len([r for r in results if getattr(r, 'status', 'PASS') == 'PASS']),
            'failed': len([r for r in results if getattr(r, 'status', 'PASS') == 'FAIL']),
            'performance_regressions': 0,
            'improvements': 0,
            'metadata': metadata or {}
        }
        self.history['runs'].append(run_data)
    
    def _get_trend_analysis_impl(self) -> Dict[str, Any]:
        """Implementation for get_trend_analysis."""
        runs = self.history['runs']
        if len(runs) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        return {
            'total_runs': len(runs),
            'recent_average_pass_rate': 95.0,
            'trend': 'stable',
            'latest_run': runs[-1] if runs else None,
            'performance_regression_frequency': 0.1
        }


class MockTestCase(AbstractEntityMock):
    """Mock implementation of test cases for regression testing."""
    
    def __init__(self,
                 subject: str = "cat",
                 predicate: str = "chase", 
                 object: str = "mouse",
                 expected_success: bool = True,
                 *args, **kwargs):
        """Initialize MockTestCase."""
        test_id = f"{subject}_{predicate}_{object}"
        super().__init__(entity_id=test_id, entity_type=EntityType.NODE, *args, **kwargs)
        
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.expected_success = expected_success
        
        self.name = test_id
        self.label = f"Test Case: {test_id}"
    
    def get_primary_attribute(self) -> Any:
        """Get the primary attribute."""
        return f"{self.subject}-{self.predicate}-{self.object}"
    
    def validate_entity(self) -> bool:
        """Validate test case entity."""
        return bool(self.subject) and bool(self.predicate) and bool(self.object)
    
    def get_entity_signature(self) -> str:
        """Get unique signature for this test case."""
        return f"test_case:{self.subject}:{self.predicate}:{self.object}"


class MockSemanticPathfindingTestSuite(Mock):
    """Mock implementation of SemanticPathfindingTestSuite."""
    
    def __init__(self, smied_instance: Mock = None, verbosity: int = 0, *args, **kwargs):
        """Initialize MockSemanticPathfindingTestSuite."""
        super().__init__(*args, **kwargs)
        self.smied = smied_instance or Mock()
        self.verbosity = verbosity
        
        # Configure mock methods
        self.run_single_test = Mock(side_effect=self._run_single_test_impl)
        
        # Mock test cases
        self.CROSS_POS_TEST_CASES = [
            MockTestCase("cat", "chase", "mouse"),
            MockTestCase("dog", "bark", "loudly"),
            MockTestCase("bird", "fly", "sky")
        ]
    
    def _run_single_test_impl(self, test_case: Mock) -> Mock:
        """Implementation for run_single_test."""
        result = Mock()
        result.success = getattr(test_case, 'expected_success', True)
        result.subject_path = ["path", "element"] if result.success else None
        result.object_path = ["path", "element"] if result.success else None
        result.connecting_predicate = Mock(name=Mock(return_value="mock.predicate.01"))
        result.execution_time = 0.1
        return result


class RegressionMockFactory:
    """Factory for creating regression testing mocks."""
    
    def __init__(self):
        """Initialize the factory."""
        self._mock_classes = {
            'BaselineResultMock': BaselineResultMock,
            'RegressionTestResultMock': RegressionTestResultMock,
            'MockRegressionBaseline': MockRegressionBaseline,
            'MockRegressionTester': MockRegressionTester,
            'MockLongTermRegressionTracker': MockLongTermRegressionTracker,
            'MockTestCase': MockTestCase,
            'MockSemanticPathfindingTestSuite': MockSemanticPathfindingTestSuite
        }
    
    def __call__(self, mock_name: str, *args, **kwargs) -> Mock:
        """
        Create a mock instance.
        
        Args:
            mock_name: Name of the mock class to create
            *args: Positional arguments for mock creation
            **kwargs: Keyword arguments for mock creation
            
        Returns:
            Mock instance
            
        Raises:
            ValueError: If mock_name is not recognized
        """
        if mock_name not in self._mock_classes:
            available_mocks = ', '.join(self._mock_classes.keys())
            raise ValueError(f"Unknown mock: {mock_name}. Available mocks: {available_mocks}")
        
        mock_class = self._mock_classes[mock_name]
        return mock_class(*args, **kwargs)
    
    def get_available_mocks(self) -> List[str]:
        """Get list of available mock names."""
        return list(self._mock_classes.keys())
    
    def create_baseline_result(self, **kwargs) -> BaselineResultMock:
        """Create a BaselineResultMock instance."""
        return self('BaselineResultMock', **kwargs)
    
    def create_regression_test_result(self, **kwargs) -> RegressionTestResultMock:
        """Create a RegressionTestResultMock instance."""
        return self('RegressionTestResultMock', **kwargs)
    
    def create_regression_baseline(self, **kwargs) -> MockRegressionBaseline:
        """Create a MockRegressionBaseline instance."""
        return self('MockRegressionBaseline', **kwargs)
    
    def create_regression_tester(self, **kwargs) -> MockRegressionTester:
        """Create a MockRegressionTester instance."""
        return self('MockRegressionTester', **kwargs)
    
    def create_trend_tracker(self, **kwargs) -> MockLongTermRegressionTracker:
        """Create a MockLongTermRegressionTracker instance."""
        return self('MockLongTermRegressionTracker', **kwargs)
    
    def create_test_case(self, **kwargs) -> MockTestCase:
        """Create a MockTestCase instance."""
        return self('MockTestCase', **kwargs)
    
    def create_test_suite(self, **kwargs) -> MockSemanticPathfindingTestSuite:
        """Create a MockSemanticPathfindingTestSuite instance."""
        return self('MockSemanticPathfindingTestSuite', **kwargs)


# Global factory instance
regression_mock_factory = RegressionMockFactory()