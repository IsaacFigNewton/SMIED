"""
Mock classes for Comparative Analysis tests.

This module follows the SMIED Testing Framework Design Specifications
for the Mock Layer with MockFactory pattern and abstract base class hierarchy.
"""

from unittest.mock import Mock
import time
import requests
from typing import List, Optional, Any, Dict, Tuple
from dataclasses import dataclass

# Import abstract base classes
from tests.mocks.base.entity_mock import AbstractEntityMock, EntityType
from tests.mocks.base.operation_mock import AbstractOperationMock
from tests.mocks.base.integration_mock import AbstractIntegrationMock


@dataclass
class ConceptNetResult:
    """Results from ConceptNet API query."""
    success: bool
    relation_found: bool
    relation_type: Optional[str]
    confidence_score: float
    path_exists: bool
    response_time: float
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


@dataclass
class SMIEDResult:
    """Results from SMIED pathfinding."""
    success: bool
    path_found: bool
    subject_path_length: int
    object_path_length: int
    total_path_length: int
    connecting_predicate: Optional[str]
    response_time: float
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Comparison results between SMIED and ConceptNet."""
    test_case_id: str
    subject: str
    predicate: str
    object: str
    smied_result: SMIEDResult
    conceptnet_result: ConceptNetResult
    winner: str  # 'SMIED', 'ConceptNet', 'Tie', 'Both_Failed'
    performance_comparison: str  # 'SMIED_Faster', 'ConceptNet_Faster', 'Similar'
    semantic_quality: str  # 'SMIED_Better', 'ConceptNet_Better', 'Similar', 'Unknown'
    notes: str


class ComparativeAnalysisMockFactory:
    """Factory class for creating Comparative Analysis mock instances.
    
    This factory follows the SMIED Testing Framework Design Specifications
    for mock creation using factory pattern with abstract base class hierarchy.
    """
    
    def __init__(self):
        self._mock_classes = {
            # Core ComparativeAnalysis mocks
            'MockComparativeAnalyzer': MockComparativeAnalyzer,
            'MockComparativeAnalyzerValidation': MockComparativeAnalyzerValidation,
            'MockComparativeAnalyzerEdgeCases': MockComparativeAnalyzerEdgeCases,
            'MockComparativeAnalyzerIntegration': MockComparativeAnalyzerIntegration,
            
            # ConceptNet interface mocks
            'MockConceptNetInterface': MockConceptNetInterface,
            'MockConceptNetInterfaceEdgeCases': MockConceptNetInterfaceEdgeCases,
            'MockConceptNetIntegration': MockConceptNetIntegration,
            
            # Result and data structure mocks
            'MockConceptNetResult': MockConceptNetResult,
            'MockSMIEDResult': MockSMIEDResult,
            'MockComparisonResult': MockComparisonResult,
            'MockTestCase': MockTestCase,
            
            # External service mocks
            'MockRequestsSession': MockRequestsSession,
            'MockSMIEDInstance': MockSMIEDInstance,
            'MockSemanticPathfindingTestSuite': MockSemanticPathfindingTestSuite,
        }
    
    def __call__(self, mock_name: str, *args, **kwargs) -> Mock:
        """
        Create and return a mock instance.
        
        Args:
            mock_name: Name of the mock to create
            *args: Positional arguments to pass to mock constructor
            **kwargs: Keyword arguments to pass to mock constructor
            
        Returns:
            Mock instance of the specified type
            
        Raises:
            ValueError: If mock_name is not recognized
        """
        if mock_name not in self._mock_classes:
            available_mocks = ', '.join(self._mock_classes.keys())
            raise ValueError(f"Unknown mock type: {mock_name}. Available mocks: {available_mocks}")
        
        mock_class = self._mock_classes[mock_name]
        return mock_class(*args, **kwargs)
    
    def get_available_mocks(self) -> List[str]:
        """Get list of all available mock types."""
        return list(self._mock_classes.keys())
    
    def create_mock_set(self, mock_names: List[str], *args, **kwargs) -> Dict[str, Mock]:
        """Create multiple mocks at once."""
        return {name: self(name, *args, **kwargs) for name in mock_names}
    
    def create_configured_mock(self, mock_name: str, config_dict: Dict[str, Any]) -> Mock:
        """
        Create a mock instance with specific configuration.
        
        Args:
            mock_name: Name of the mock to create
            config_dict: Configuration dictionary to apply to the mock
            
        Returns:
            Configured mock instance
        """
        mock_instance = self(mock_name)
        
        # Apply configuration to the mock
        for attr_name, attr_value in config_dict.items():
            if hasattr(mock_instance, attr_name):
                setattr(mock_instance, attr_name, attr_value)
            else:
                # For methods that don't exist, add them as Mock objects
                setattr(mock_instance, attr_name, Mock(return_value=attr_value))
        
        return mock_instance
    
    def get_mock_info(self, mock_name: str) -> Dict[str, Any]:
        """
        Get information about a specific mock type.
        
        Args:
            mock_name: Name of the mock to get info for
            
        Returns:
            Dictionary with mock information
        """
        if mock_name not in self._mock_classes:
            return {'exists': False, 'error': f'Mock {mock_name} not found'}
        
        mock_class = self._mock_classes[mock_name]
        return {
            'exists': True,
            'class_name': mock_class.__name__,
            'module': mock_class.__module__,
            'docstring': mock_class.__doc__,
            'base_classes': [base.__name__ for base in mock_class.__bases__]
        }


class MockComparativeAnalyzer(AbstractOperationMock):
    """Mock for ComparativeAnalyzer basic functionality."""
    
    def __init__(self, smied_instance=None, verbosity=1):
        super().__init__(
            operation_id="mock_comparative_analyzer",
            operation_type="comparative_analysis"
        )
        
        self.smied = smied_instance or MockSMIEDInstance()
        self.verbosity = verbosity
        self.conceptnet = MockConceptNetInterface(verbosity=verbosity)
        self.conceptnet_available = True
        
        # Configure mock methods to use actual input data
        self.run_smied_test = Mock(side_effect=self._run_smied_test_with_input)
        self.compare_single_test = Mock(side_effect=self._compare_single_test_with_input)
        self.run_comparative_analysis = Mock(side_effect=self._run_comparative_analysis_with_input)
        self.generate_comparative_report = Mock(return_value={})
        self.print_comparative_report = Mock()
        
        # Private method mocks with proper logic
        self._determine_winner = self._determine_winner_logic
        self._compare_performance = self._compare_performance_logic
        self._assess_semantic_quality = Mock(return_value="Similar")
        self._generate_comparison_notes = Mock(return_value="No significant differences")
    
    def _create_default_smied_result(self) -> SMIEDResult:
        """Create a default SMIED result for mocking."""
        return SMIEDResult(
            success=True,
            path_found=True,
            subject_path_length=2,
            object_path_length=2,
            total_path_length=4,
            connecting_predicate="test.v.01",
            response_time=0.1
        )
    
    def _create_default_comparison_result(self) -> ComparisonResult:
        """Create a default comparison result for mocking."""
        return ComparisonResult(
            test_case_id="test_test_test",
            subject="test",
            predicate="test",
            object="test",
            smied_result=self._create_default_smied_result(),
            conceptnet_result=MockConceptNetResult()._create_default_result(),
            winner="SMIED",
            performance_comparison="Similar",
            semantic_quality="Similar",
            notes="No significant differences"
        )
    
    def _run_smied_test_with_input(self, test_case) -> SMIEDResult:
        """Run SMIED test with actual input data."""
        if not hasattr(test_case, 'subject') or not test_case.subject:
            return SMIEDResult(
                success=False,
                path_found=False,
                subject_path_length=0,
                object_path_length=0,
                total_path_length=0,
                connecting_predicate=None,
                response_time=0.1,
                error="Invalid test case"
            )
        
        return SMIEDResult(
            success=True,
            path_found=True,
            subject_path_length=2,
            object_path_length=2,
            total_path_length=4,
            connecting_predicate="test.v.01",
            response_time=0.1
        )
    
    def _compare_single_test_with_input(self, test_case) -> ComparisonResult:
        """Compare single test with actual input data."""
        if not hasattr(test_case, 'subject'):
            return self._create_default_comparison_result()
        
        # Use actual test case data
        smied_result = self._run_smied_test_with_input(test_case)
        conceptnet_result = self.conceptnet.query_relation(
            test_case.subject, test_case.predicate, test_case.object
        )
        
        return ComparisonResult(
            test_case_id=f"{test_case.subject}_{test_case.predicate}_{test_case.object}",
            subject=test_case.subject,
            predicate=test_case.predicate,
            object=test_case.object,
            smied_result=smied_result,
            conceptnet_result=conceptnet_result,
            winner="SMIED",
            performance_comparison="Similar",
            semantic_quality="Similar",
            notes=f"Comparison for {test_case.subject}->{test_case.object}"
        )
    
    def _run_comparative_analysis_with_input(self, test_cases) -> List[ComparisonResult]:
        """Run comparative analysis with actual input data."""
        results = []
        for test_case in test_cases:
            comparison = self._compare_single_test_with_input(test_case)
            results.append(comparison)
        return results
    
    def get_primary_attribute(self) -> Any:
        return self.smied
    
    def validate_entity(self) -> bool:
        return self.smied is not None
    
    def get_entity_signature(self) -> str:
        return f"ComparativeAnalyzer:{self.operation_id}"
    
    def execute(self, target: Any, *args, **kwargs) -> Any:
        """
        Execute the comparative analysis operation.
        
        Args:
            target: The target object to analyze (typically test cases)
            *args: Additional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Analysis results
        """
        if hasattr(target, '__iter__'):
            # If target is iterable (list of test cases), run comparative analysis
            return self.run_comparative_analysis(target)
        else:
            # If target is a single test case, run single comparison
            return self.compare_single_test(target)
    
    def get_operation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata specific to comparative analysis operation.
        
        Returns:
            Dictionary containing operation metadata
        """
        return {
            'operation_name': 'comparative_analysis',
            'operation_type': self.operation_type,
            'operation_id': self.operation_id,
            'supports_smied': True,
            'supports_conceptnet': self.conceptnet_available,
            'verbosity_level': self.verbosity,
            'smied_instance': self.smied is not None,
            'conceptnet_interface': self.conceptnet is not None,
            'comparison_methods': ['run_smied_test', 'compare_single_test', 'run_comparative_analysis'],
            'result_formats': ['SMIEDResult', 'ConceptNetResult', 'ComparisonResult']
        }
    
    def validate_target(self, target: Any) -> bool:
        """
        Validate that the target is suitable for comparative analysis.
        
        Args:
            target: Target object to validate
            
        Returns:
            True if target is valid, False otherwise
        """
        if hasattr(target, '__iter__'):
            # If target is iterable, validate each item
            for item in target:
                if not self._validate_single_target(item):
                    return False
            return True
        else:
            # If target is a single item, validate it directly
            return self._validate_single_target(target)
    
    def _validate_single_target(self, target: Any) -> bool:
        """
        Validate a single target object.
        
        Args:
            target: Single target object to validate
            
        Returns:
            True if target is valid, False otherwise
        """
        # Check if target has required attributes for a test case
        required_attrs = ['subject', 'predicate', 'object']
        
        for attr in required_attrs:
            if not hasattr(target, attr):
                return False
            
            # Check that the attribute is not None or empty
            attr_value = getattr(target, attr)
            if not attr_value or (isinstance(attr_value, str) and not attr_value.strip()):
                return False
        
        return True
    
    def _determine_winner_logic(self, smied_result, conceptnet_result):
        """Determine winner based on actual result comparison."""
        smied_success = getattr(smied_result, 'path_found', getattr(smied_result, 'success', False))
        conceptnet_success = getattr(conceptnet_result, 'relation_found', getattr(conceptnet_result, 'success', False))
        
        if smied_success and conceptnet_success:
            return "Tie"
        elif smied_success:
            return "SMIED"
        elif conceptnet_success:
            return "ConceptNet"
        else:
            return "Both_Failed"
    
    def _compare_performance_logic(self, smied_result, conceptnet_result):
        """Compare performance based on response times."""
        smied_time = getattr(smied_result, 'response_time', 0.0)
        conceptnet_time = getattr(conceptnet_result, 'response_time', 0.0)
        
        # Consider times similar if within 0.1 seconds of each other
        time_diff = abs(smied_time - conceptnet_time)
        if time_diff <= 0.1:
            return "Similar"
        elif smied_time < conceptnet_time:
            return "SMIED_Faster"
        else:
            return "ConceptNet_Faster"


class MockComparativeAnalyzerValidation(MockComparativeAnalyzer):
    """Mock for ComparativeAnalyzer validation tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override for validation scenarios
        self.run_smied_test.side_effect = self._validate_test_case_input
        self.compare_single_test.side_effect = self._validate_comparison_input
    
    def _validate_test_case_input(self, test_case):
        """Validate test case input and return appropriate result."""
        if not hasattr(test_case, 'subject') or not test_case.subject:
            return SMIEDResult(
                success=False,
                path_found=False,
                subject_path_length=0,
                object_path_length=0,
                total_path_length=0,
                connecting_predicate=None,
                response_time=0.0,
                error="Invalid test case: missing subject"
            )
        return self._create_default_smied_result()
    
    def _validate_comparison_input(self, test_case):
        """Validate comparison input and return appropriate result."""
        if not hasattr(test_case, 'predicate') or not test_case.predicate:
            return ComparisonResult(
                test_case_id="invalid_test",
                subject=getattr(test_case, 'subject', ''),
                predicate="",
                object=getattr(test_case, 'object', ''),
                smied_result=SMIEDResult(False, False, 0, 0, 0, None, 0.0, "Invalid predicate"),
                conceptnet_result=ConceptNetResult(False, False, None, 0.0, False, 0.0, "Invalid predicate"),
                winner="Both_Failed",
                performance_comparison="Similar",
                semantic_quality="Unknown",
                notes="Invalid test case"
            )
        return self._create_default_comparison_result()


class MockComparativeAnalyzerEdgeCases(MockComparativeAnalyzer):
    """Mock for ComparativeAnalyzer edge cases and error conditions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configure for edge cases
        self.conceptnet_available = False
        self.conceptnet.test_connection = Mock(return_value=False)
        # Override ConceptNet query to always return failure when unavailable
        self.conceptnet.query_relation = Mock(side_effect=self._conceptnet_unavailable_response)
        
        # Override methods for edge case scenarios
        self.run_smied_test.side_effect = self._handle_edge_case_smied_test
        self.compare_single_test.side_effect = self._handle_edge_case_comparison
        # Override _determine_winner to use edge case logic
        self._determine_winner = self._handle_edge_case_winner
    
    def _handle_edge_case_smied_test(self, test_case):
        """Handle edge cases in SMIED testing."""
        # Simulate timeout or error conditions
        if hasattr(test_case, 'subject') and test_case.subject == "timeout_test":
            return SMIEDResult(
                success=False,
                path_found=False,
                subject_path_length=0,
                object_path_length=0,
                total_path_length=0,
                connecting_predicate=None,
                response_time=10.0,
                error="Request timeout"
            )
        
        # Simulate very long paths
        if hasattr(test_case, 'subject') and test_case.subject == "long_path_test":
            return SMIEDResult(
                success=True,
                path_found=True,
                subject_path_length=10,
                object_path_length=12,
                total_path_length=22,
                connecting_predicate="complex.v.01",
                response_time=5.0
            )
        
        return self._create_default_smied_result()
    
    def _handle_edge_case_winner(self, smied_result, conceptnet_result):
        """Handle edge cases in winner determination."""
        if not smied_result.success and not conceptnet_result.success:
            return "Both_Failed"
        return "SMIED"
    
    def _handle_edge_case_comparison(self, test_case):
        """Handle edge case comparisons for impossible relationships."""
        if not hasattr(test_case, 'subject'):
            return ComparisonResult(
                test_case_id="invalid_test",
                subject="",
                predicate="",
                object="",
                smied_result=SMIEDResult(False, False, 0, 0, 0, None, 0.0, "Invalid test case"),
                conceptnet_result=ConceptNetResult(False, False, None, 0.0, False, 0.0, "Invalid test case"),
                winner="Both_Failed",
                performance_comparison="Similar",
                semantic_quality="Unknown",
                notes="Invalid test case"
            )
        
        # Get results using edge case handlers
        smied_result = self._handle_edge_case_smied_test(test_case)
        conceptnet_result = self.conceptnet.query_relation(
            test_case.subject, test_case.predicate, test_case.object
        )
        
        # Check for impossible relationships based on subject
        impossible_subjects = ['rock', 'color', 'silence', 'abstract_concept']
        is_impossible = any(impossible in test_case.subject.lower() for impossible in impossible_subjects)
        
        if is_impossible:
            # Force both systems to fail for impossible relationships
            smied_result = SMIEDResult(
                success=False,
                path_found=False,
                subject_path_length=0,
                object_path_length=0,
                total_path_length=0,
                connecting_predicate=None,
                response_time=0.1,
                error="Impossible relationship detected"
            )
            conceptnet_result = ConceptNetResult(
                success=False,
                relation_found=False,
                relation_type=None,
                confidence_score=0.0,
                path_exists=False,
                response_time=0.1,
                error="Impossible relationship detected"
            )
        
        # Determine winner
        winner = self._handle_edge_case_winner(smied_result, conceptnet_result)
        
        return ComparisonResult(
            test_case_id=f"{test_case.subject}_{test_case.predicate}_{test_case.object}",
            subject=test_case.subject,
            predicate=test_case.predicate,
            object=test_case.object,
            smied_result=smied_result,
            conceptnet_result=conceptnet_result,
            winner=winner,
            performance_comparison="Similar",
            semantic_quality="Unknown" if winner == "Both_Failed" else "Similar",
            notes=f"Edge case result for {test_case.subject}->{test_case.object}"
        )
    
    def _conceptnet_unavailable_response(self, subject, predicate, obj, timeout=10.0):
        """Return ConceptNet unavailable response for edge case testing."""
        return ConceptNetResult(
            success=False,
            relation_found=False,
            relation_type=None,
            confidence_score=0.0,
            path_exists=False,
            response_time=0.0,
            error="ConceptNet API is unavailable"
        )


class MockComparativeAnalyzerIntegration(AbstractIntegrationMock):
    """Mock for ComparativeAnalyzer integration tests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            integration_id="comparative_analyzer_integration",
            components=["SMIED", "ConceptNet", "SemanticPathfindingTestSuite"]
        )
        
        self.smied = MockSMIEDInstance()
        self.conceptnet = MockConceptNetIntegration()
        self.test_suite = MockSemanticPathfindingTestSuite()
        
        # Configure realistic integration behavior
        self.run_comparative_analysis = Mock(side_effect=self._realistic_comparative_analysis)
        self.generate_comparative_report = Mock(side_effect=self._realistic_report_generation)
        self.run_smied_test = Mock(side_effect=self._realistic_smied_test)
        self.compare_single_test = Mock(side_effect=self._realistic_single_comparison)
    
    def _realistic_comparative_analysis(self, test_cases):
        """Simulate realistic comparative analysis with multiple components."""
        results = []
        for test_case in test_cases:
            # Simulate varying performance between systems with better distribution
            # Use abs() to handle negative hash values and ensure more balanced results
            subject_hash = abs(hash(test_case.subject))
            object_hash = abs(hash(test_case.object))
            smied_success = (subject_hash % 3) > 0  # 2/3 success rate
            conceptnet_success = (object_hash % 4) > 1  # 1/2 success rate
            
            smied_result = SMIEDResult(
                success=smied_success,
                path_found=smied_success,
                subject_path_length=2 if smied_success else 0,
                object_path_length=3 if smied_success else 0,
                total_path_length=5 if smied_success else 0,
                connecting_predicate="test.v.01" if smied_success else None,
                response_time=0.1 + (hash(test_case.subject) % 100) / 1000,
                error=None if smied_success else "Path not found"
            )
            
            conceptnet_result = ConceptNetResult(
                success=conceptnet_success,
                relation_found=conceptnet_success,
                relation_type="RelatedTo" if conceptnet_success else None,
                confidence_score=2.5 if conceptnet_success else 0.0,
                path_exists=conceptnet_success,
                response_time=0.2 + (hash(test_case.object) % 100) / 1000,
                error=None if conceptnet_success else "No relation found"
            )
            
            # Determine winner
            if smied_success and conceptnet_success:
                winner = "Tie"
            elif smied_success:
                winner = "SMIED"
            elif conceptnet_success:
                winner = "ConceptNet"
            else:
                winner = "Both_Failed"
            
            comparison = ComparisonResult(
                test_case_id=f"{test_case.subject}_{test_case.predicate}_{test_case.object}",
                subject=test_case.subject,
                predicate=test_case.predicate,
                object=test_case.object,
                smied_result=smied_result,
                conceptnet_result=conceptnet_result,
                winner=winner,
                performance_comparison="SMIED_Faster" if smied_result.response_time < conceptnet_result.response_time else "ConceptNet_Faster",
                semantic_quality="Similar",
                notes=f"Integration test result for {test_case.subject}->{test_case.object}"
            )
            results.append(comparison)
        
        return results
    
    def _realistic_report_generation(self, results):
        """Generate realistic comparative report."""
        total_tests = len(results)
        smied_successes = len([r for r in results if r.smied_result.path_found])
        conceptnet_successes = len([r for r in results if r.conceptnet_result.relation_found])
        
        return {
            'summary': {
                'total_tests': total_tests,
                'smied_success_rate': (smied_successes / total_tests * 100) if total_tests > 0 else 0,
                'conceptnet_success_rate': (conceptnet_successes / total_tests * 100) if total_tests > 0 else 0,
                'conceptnet_available': True
            },
            'winner_distribution': {'SMIED': smied_successes, 'ConceptNet': conceptnet_successes, 'Tie': 0, 'Both_Failed': total_tests - smied_successes - conceptnet_successes},
            'performance_comparison': {'SMIED_Faster': total_tests // 2, 'ConceptNet_Faster': total_tests // 3, 'Similar': total_tests // 6},
            'response_times': {
                'smied_avg': 0.15,
                'smied_median': 0.12,
                'conceptnet_avg': 0.25,
                'conceptnet_median': 0.22
            },
            'smied_path_analysis': {
                'avg_path_length': 4.5,
                'median_path_length': 4.0,
                'max_path_length': 10,
                'min_path_length': 2
            },
            'semantic_quality_distribution': {'Similar': total_tests, 'SMIED_Better': 0, 'ConceptNet_Better': 0, 'Unknown': 0},
            'detailed_failures': []
        }
    
    def get_primary_attribute(self) -> Any:
        return self.components
    
    def validate_entity(self) -> bool:
        return len(self.components) > 0
    
    def get_entity_signature(self) -> str:
        return f"ComparativeAnalyzerIntegration:{self.integration_id}"
    
    def setup_integration_components(self) -> Dict[str, Any]:
        """
        Set up all components required for integration testing.
        
        Returns:
            Dictionary mapping component names to component instances
        """
        components = {
            'SMIED': self.smied,
            'ConceptNet': self.conceptnet,
            'SemanticPathfindingTestSuite': self.test_suite
        }
        
        # Register components with the integration mock
        for name, component in components.items():
            self.register_component(name, component)
        
        return components
    
    def configure_component_interactions(self) -> None:
        """
        Configure how components interact with each other.
        
        This sets up the relationships, dependencies, and communication 
        patterns between SMIED, ConceptNet, and the test suite.
        """
        # Set up component dependencies
        # SMIED doesn't depend on other components
        # ConceptNet doesn't depend on other components
        # Test suite might depend on both SMIED and ConceptNet
        
        # Create interactions between components
        if 'SMIED' in self.components and 'ConceptNet' in self.components:
            # Create interaction for comparative analysis
            self.create_component_interaction('SMIED', 'ConceptNet', 'comparison')
        
        if 'SemanticPathfindingTestSuite' in self.components:
            # Test suite interacts with both SMIED and ConceptNet
            if 'SMIED' in self.components:
                self.create_component_interaction('SemanticPathfindingTestSuite', 'SMIED', 'test_execution')
            if 'ConceptNet' in self.components:
                self.create_component_interaction('SemanticPathfindingTestSuite', 'ConceptNet', 'test_execution')
        
        # Configure shared context for data flow
        self.shared_context.update({
            'comparison_mode': 'full',
            'timeout_settings': {'smied': 30.0, 'conceptnet': 10.0},
            'result_format': 'detailed',
            'error_handling': 'continue'
        })
    
    def validate_integration_state(self) -> bool:
        """
        Validate that the integration is in a consistent state.
        
        Returns:
            True if integration state is valid, False otherwise
        """
        # Check that all required components are present and initialized
        required_components = ['SMIED', 'ConceptNet', 'SemanticPathfindingTestSuite']
        
        for component_name in required_components:
            if component_name not in self.components:
                return False
            
            component = self.components[component_name]
            if component is None:
                return False
            
            # Check component state if it has a validation method
            if hasattr(component, 'validate_entity'):
                if not component.validate_entity():
                    return False
            elif hasattr(component, 'test_connection'):
                # For ConceptNet, check connection
                if not component.test_connection():
                    # Allow ConceptNet to be unavailable in some test scenarios
                    if component_name == 'ConceptNet' and not getattr(component, 'required', True):
                        continue
                    return False
        
        # Verify component interactions are configured
        if not self.interaction_log and len(self.components) > 1:
            # If we have multiple components but no interactions recorded yet,
            # that's okay - interactions happen during execution
            pass
        
        # Check shared context has required settings
        required_context_keys = ['comparison_mode', 'timeout_settings']
        for key in required_context_keys:
            if key not in self.shared_context:
                return False
        
        return True
    
    def _realistic_smied_test(self, test_case) -> SMIEDResult:
        """Simulate realistic SMIED test with varying results."""
        if not hasattr(test_case, 'subject') or not test_case.subject:
            return SMIEDResult(
                success=False,
                path_found=False,
                subject_path_length=0,
                object_path_length=0,
                total_path_length=0,
                connecting_predicate=None,
                response_time=0.1,
                error="Invalid test case"
            )
        
        # Simulate varying success based on hash of subject with better distribution
        subject_hash = abs(hash(test_case.subject))
        success = (subject_hash % 3) > 0  # 2/3 success rate
        
        if success:
            return SMIEDResult(
                success=True,
                path_found=True,
                subject_path_length=2,
                object_path_length=3,
                total_path_length=5,
                connecting_predicate="test.v.01",
                response_time=0.1 + (hash(test_case.subject) % 100) / 1000
            )
        else:
            return SMIEDResult(
                success=False,
                path_found=False,
                subject_path_length=0,
                object_path_length=0,
                total_path_length=0,
                connecting_predicate=None,
                response_time=0.05,
                error="Path not found"
            )
    
    def _realistic_single_comparison(self, test_case) -> ComparisonResult:
        """Simulate realistic single comparison with varying results."""
        if not hasattr(test_case, 'subject'):
            return ComparisonResult(
                test_case_id="invalid_test",
                subject="",
                predicate="",
                object="",
                smied_result=SMIEDResult(False, False, 0, 0, 0, None, 0.0, "Invalid test case"),
                conceptnet_result=ConceptNetResult(False, False, None, 0.0, False, 0.0, "Invalid test case"),
                winner="Both_Failed",
                performance_comparison="Similar",
                semantic_quality="Unknown",
                notes="Invalid test case"
            )
        
        # Get realistic results from components
        smied_result = self._realistic_smied_test(test_case)
        conceptnet_result = self.conceptnet.query_relation(
            test_case.subject, test_case.predicate, test_case.object
        )
        
        # Determine winner based on success
        if smied_result.path_found and conceptnet_result.relation_found:
            winner = "Tie"
        elif smied_result.path_found:
            winner = "SMIED"
        elif conceptnet_result.relation_found:
            winner = "ConceptNet"
        else:
            winner = "Both_Failed"
        
        # Determine performance comparison
        if smied_result.response_time < conceptnet_result.response_time:
            perf_comparison = "SMIED_Faster"
        elif conceptnet_result.response_time < smied_result.response_time:
            perf_comparison = "ConceptNet_Faster"
        else:
            perf_comparison = "Similar"
        
        return ComparisonResult(
            test_case_id=f"{test_case.subject}_{test_case.predicate}_{test_case.object}",
            subject=test_case.subject,
            predicate=test_case.predicate,
            object=test_case.object,
            smied_result=smied_result,
            conceptnet_result=conceptnet_result,
            winner=winner,
            performance_comparison=perf_comparison,
            semantic_quality="Similar",
            notes=f"Integration test result for {test_case.subject}->{test_case.object}"
        )


class MockConceptNetInterface(AbstractOperationMock):
    """Mock for ConceptNet interface."""
    
    def __init__(self, base_url="http://api.conceptnet.io", verbosity=1):
        super().__init__(
            operation_id="mock_conceptnet_interface",
            operation_type="external_api"
        )
        
        self.base_url = base_url
        self.verbosity = verbosity
        self.session = MockRequestsSession()
        
        # Configure mock methods
        self.test_connection = Mock(return_value=True)
        self.query_relation = Mock(return_value=self._create_default_conceptnet_result())
        self._query_direct_relation = Mock(return_value=self._create_default_conceptnet_result())
        self._query_broader_search = Mock(return_value=self._create_default_conceptnet_result())
    
    def _create_default_conceptnet_result(self) -> ConceptNetResult:
        """Create a default ConceptNet result for mocking."""
        return ConceptNetResult(
            success=True,
            relation_found=True,
            relation_type="RelatedTo",
            confidence_score=2.5,
            path_exists=True,
            response_time=0.2,
            raw_response={"edges": []}
        )
    
    def get_primary_attribute(self) -> Any:
        return self.base_url
    
    def validate_entity(self) -> bool:
        return self.session is not None
    
    def get_entity_signature(self) -> str:
        return f"ConceptNetInterface:{self.operation_id}:{self.base_url}"
    
    def execute(self, target: Any, *args, **kwargs) -> Any:
        """
        Execute ConceptNet query operation.
        
        Args:
            target: The target query (typically a tuple of subject, predicate, object)
            *args: Additional arguments
            **kwargs: Keyword arguments including timeout
            
        Returns:
            ConceptNet query results
        """
        if isinstance(target, (tuple, list)) and len(target) >= 3:
            # Unpack subject, predicate, object from target
            subject, predicate, obj = target[0], target[1], target[2]
            timeout = kwargs.get('timeout', 10.0)
            return self.query_relation(subject, predicate, obj, timeout)
        elif hasattr(target, 'subject') and hasattr(target, 'predicate') and hasattr(target, 'object'):
            # Target is a test case object
            timeout = kwargs.get('timeout', 10.0)
            return self.query_relation(target.subject, target.predicate, target.object, timeout)
        else:
            # Invalid target format
            return ConceptNetResult(
                success=False,
                relation_found=False,
                relation_type=None,
                confidence_score=0.0,
                path_exists=False,
                response_time=0.0,
                error="Invalid target format for ConceptNet query"
            )
    
    def get_operation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata specific to ConceptNet interface operation.
        
        Returns:
            Dictionary containing operation metadata
        """
        return {
            'operation_name': 'conceptnet_interface',
            'operation_type': self.operation_type,
            'operation_id': self.operation_id,
            'base_url': self.base_url,
            'verbosity_level': self.verbosity,
            'session_available': self.session is not None,
            'connection_methods': ['test_connection', 'query_relation'],
            'query_methods': ['_query_direct_relation', '_query_broader_search'],
            'supported_formats': ['subject-predicate-object', 'test_case_object'],
            'timeout_support': True,
            'result_format': 'ConceptNetResult'
        }
    
    def validate_target(self, target: Any) -> bool:
        """
        Validate that the target is suitable for ConceptNet queries.
        
        Args:
            target: Target object to validate
            
        Returns:
            True if target is valid, False otherwise
        """
        # Check if target is a tuple/list with 3 elements (subject, predicate, object)
        if isinstance(target, (tuple, list)):
            if len(target) >= 3:
                # Check that all three elements are non-empty strings
                for element in target[:3]:
                    if not isinstance(element, str) or not element.strip():
                        return False
                return True
            return False
        
        # Check if target is an object with subject, predicate, object attributes
        elif hasattr(target, 'subject') and hasattr(target, 'predicate') and hasattr(target, 'object'):
            # Validate that all attributes exist and are non-empty
            required_attrs = ['subject', 'predicate', 'object']
            for attr in required_attrs:
                attr_value = getattr(target, attr)
                if not attr_value or (isinstance(attr_value, str) and not attr_value.strip()):
                    return False
            return True
        
        # Target format not supported
        return False


class MockConceptNetInterfaceEdgeCases(MockConceptNetInterface):
    """Mock for ConceptNet interface edge cases."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configure for edge cases
        self.test_connection.return_value = False
        self.query_relation.side_effect = self._handle_edge_case_query
    
    def _handle_edge_case_query(self, subject, predicate, object, timeout=10.0):
        """Handle edge case queries."""
        # Simulate network timeout
        if subject == "timeout_test":
            return ConceptNetResult(
                success=False,
                relation_found=False,
                relation_type=None,
                confidence_score=0.0,
                path_exists=False,
                response_time=timeout,
                error="Request timeout"
            )
        
        # Simulate API error
        if subject == "error_test":
            return ConceptNetResult(
                success=False,
                relation_found=False,
                relation_type=None,
                confidence_score=0.0,
                path_exists=False,
                response_time=0.1,
                error="HTTP 500 Server Error"
            )
        
        return self._create_default_conceptnet_result()


class MockConceptNetIntegration(MockConceptNetInterface):
    """Mock for ConceptNet integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configure for realistic integration behavior
        self.query_relation.side_effect = self._realistic_query_relation
    
    def _realistic_query_relation(self, subject, predicate, object, timeout=10.0):
        """Simulate realistic ConceptNet query behavior."""
        # Base success probability on hash of terms
        success_probability = (hash(subject) + hash(object)) % 100 / 100.0
        
        if success_probability > 0.7:
            return ConceptNetResult(
                success=True,
                relation_found=True,
                relation_type=self._get_realistic_relation_type(predicate),
                confidence_score=1.0 + success_probability * 3.0,
                path_exists=True,
                response_time=0.1 + (hash(object) % 50) / 1000.0,
                raw_response={"edges": [{"rel": {"label": self._get_realistic_relation_type(predicate)}}]}
            )
        else:
            return ConceptNetResult(
                success=True,
                relation_found=False,
                relation_type=None,
                confidence_score=0.0,
                path_exists=False,
                response_time=0.2 + (hash(subject) % 30) / 1000.0
            )
    
    def _get_realistic_relation_type(self, predicate):
        """Get realistic relation type based on predicate."""
        relation_map = {
            "chase": "RelatedTo",
            "is": "IsA",
            "has": "HasProperty",
            "can": "CapableOf",
            "loves": "RelatedTo",
            "eats": "UsedFor"
        }
        return relation_map.get(predicate, "RelatedTo")


class MockConceptNetResult(AbstractEntityMock):
    """Mock for ConceptNet result objects."""
    
    def __init__(self, success=True, relation_found=True, **kwargs):
        super().__init__(
            entity_id="mock_conceptnet_result",
            entity_type=EntityType.UNKNOWN
        )
        
        self.success = success
        self.relation_found = relation_found
        self.relation_type = kwargs.get('relation_type', "RelatedTo")
        self.confidence_score = kwargs.get('confidence_score', 2.5)
        self.path_exists = kwargs.get('path_exists', relation_found)
        self.response_time = kwargs.get('response_time', 0.2)
        self.error = kwargs.get('error', None)
        self.raw_response = kwargs.get('raw_response', {"edges": []})
    
    def _create_default_result(self) -> ConceptNetResult:
        """Create a default ConceptNet result."""
        return ConceptNetResult(
            success=self.success,
            relation_found=self.relation_found,
            relation_type=self.relation_type,
            confidence_score=self.confidence_score,
            path_exists=self.path_exists,
            response_time=self.response_time,
            error=self.error,
            raw_response=self.raw_response
        )
    
    def get_primary_attribute(self) -> Any:
        return self.relation_type
    
    def validate_entity(self) -> bool:
        return self.success is not None
    
    def get_entity_signature(self) -> str:
        return f"ConceptNetResult:{self.success}:{self.relation_found}"


class MockSMIEDResult(AbstractEntityMock):
    """Mock for SMIED result objects."""
    
    def __init__(self, success=True, path_found=True, **kwargs):
        super().__init__(
            entity_id="mock_smied_result",
            entity_type=EntityType.PATH
        )
        
        self.success = success
        self.path_found = path_found
        self.subject_path_length = kwargs.get('subject_path_length', 2)
        self.object_path_length = kwargs.get('object_path_length', 2)
        self.total_path_length = kwargs.get('total_path_length', 4)
        self.connecting_predicate = kwargs.get('connecting_predicate', "test.v.01")
        self.response_time = kwargs.get('response_time', 0.1)
        self.error = kwargs.get('error', None)
    
    def get_primary_attribute(self) -> Any:
        return self.connecting_predicate
    
    def validate_entity(self) -> bool:
        return self.success is not None
    
    def get_entity_signature(self) -> str:
        return f"SMIEDResult:{self.success}:{self.total_path_length}"


class MockComparisonResult(AbstractEntityMock):
    """Mock for comparison result objects."""
    
    def __init__(self, test_case_id="test_test_test", **kwargs):
        super().__init__(
            entity_id=test_case_id,
            entity_type=EntityType.UNKNOWN
        )
        
        self.test_case_id = test_case_id
        self.subject = kwargs.get('subject', 'test_subject')
        self.predicate = kwargs.get('predicate', 'test_predicate')
        self.object = kwargs.get('object', 'test_object')
        self.smied_result = kwargs.get('smied_result', MockSMIEDResult())
        self.conceptnet_result = kwargs.get('conceptnet_result', MockConceptNetResult())
        self.winner = kwargs.get('winner', 'SMIED')
        self.performance_comparison = kwargs.get('performance_comparison', 'Similar')
        self.semantic_quality = kwargs.get('semantic_quality', 'Similar')
        self.notes = kwargs.get('notes', 'No significant differences')
    
    def get_primary_attribute(self) -> Any:
        return self.test_case_id
    
    def validate_entity(self) -> bool:
        return self.test_case_id is not None
    
    def get_entity_signature(self) -> str:
        return f"ComparisonResult:{self.test_case_id}:{self.winner}"


class MockTestCase(AbstractEntityMock):
    """Mock for test case objects."""
    
    def __init__(self, subject="test", predicate="test", object="test", **kwargs):
        super().__init__(
            entity_id=f"{subject}_{predicate}_{object}",
            entity_type=EntityType.UNKNOWN
        )
        
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.expected = kwargs.get('expected', True)
        self.description = kwargs.get('description', "Test case")
        self.category = kwargs.get('category', "test")
        self.difficulty = kwargs.get('difficulty', "easy")
    
    def get_primary_attribute(self) -> Any:
        return f"{self.subject}->{self.predicate}->{self.object}"
    
    def validate_entity(self) -> bool:
        return all([self.subject, self.predicate, self.object])
    
    def get_entity_signature(self) -> str:
        return f"TestCase:{self.subject}:{self.predicate}:{self.object}"


class MockRequestsSession(Mock):
    """Mock for requests Session objects."""
    
    def __init__(self):
        super().__init__()
        
        self.headers = {}
        self.get = Mock(return_value=MockResponse())
        self.post = Mock(return_value=MockResponse())
        self.put = Mock(return_value=MockResponse())
        self.delete = Mock(return_value=MockResponse())


class MockResponse(Mock):
    """Mock for HTTP response objects."""
    
    def __init__(self, status_code=200, json_data=None):
        super().__init__()
        
        self.status_code = status_code
        self._json_data = json_data or {"edges": []}
        self.text = str(self._json_data)
        
        self.json = Mock(return_value=self._json_data)
        self.raise_for_status = Mock()


class MockSMIEDInstance(Mock):
    """Mock for SMIED instance objects."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.analyze_triple = Mock(return_value=(
            ["subject", "path"],  # subject_path
            ["object", "path"],   # object_path
            Mock(name=Mock(return_value="connecting.v.01"))  # connecting_predicate
        ))


class MockSemanticPathfindingTestSuite(Mock):
    """Mock for SemanticPathfindingTestSuite."""
    
    def __init__(self):
        super().__init__()
        
        # Create mock test cases
        self.CROSS_POS_TEST_CASES = [
            MockTestCase("cat", "chase", "mouse"),
            MockTestCase("dog", "bark", "loudly"),
            MockTestCase("bird", "fly", "sky"),
            MockTestCase("fish", "swim", "water"),
            MockTestCase("person", "read", "book")
        ]