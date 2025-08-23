"""
Abstract base class for edge case testing mocks.

This module provides the AbstractEdgeCaseMock class that serves as a base
for all edge case testing scenarios including error conditions, empty returns,
and invalid inputs.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock, MagicMock
from typing import Any, Dict, List, Optional, Union, Callable, Type
import json


class AbstractEdgeCaseMock(ABC, Mock):
    """
    Abstract base class for edge case testing mocks.
    
    This class provides a common interface for mocks that handle edge cases,
    error scenarios, and boundary conditions in testing.
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the AbstractEdgeCaseMock.
        
        Args:
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes()
        self._setup_common_edge_cases()
        self._setup_error_scenarios()
    
    def _setup_common_attributes(self):
        """Set up common attributes for edge case handling."""
        # Edge case configuration
        self.edge_case_enabled = True
        self.error_probability = 0.0  # 0.0 = never, 1.0 = always
        self.current_scenario = "normal"
        
        # Error tracking
        self.error_count = 0
        self.error_history = []
        self.max_errors = None  # Unlimited by default
        
        # Response configuration  
        self.empty_responses = True
        self.null_responses = True
        self.invalid_type_responses = True
        
        # Scenario tracking
        self.executed_scenarios = []
        self.scenario_counters = {}
    
    def _setup_common_edge_cases(self):
        """Set up common edge case scenarios."""
        # Empty/null response scenarios
        self.return_empty_list = Mock(return_value=[])
        self.return_empty_dict = Mock(return_value={})
        self.return_empty_string = Mock(return_value="")
        self.return_none = Mock(return_value=None)
        self.return_zero = Mock(return_value=0)
        self.return_false = Mock(return_value=False)
        
        # Invalid data scenarios
        self.return_invalid_type = Mock(return_value="invalid_type_placeholder")
        self.return_malformed_data = Mock(return_value={"malformed": "data"})
        self.return_large_data = Mock(return_value=["x"] * 10000)  # Large list
        self.return_unicode_data = Mock(return_value="🚀 Üñíçødé téxt 测试")
        
        # Boundary value scenarios
        self.return_max_int = Mock(return_value=2**31 - 1)
        self.return_min_int = Mock(return_value=-2**31)
        self.return_max_float = Mock(return_value=float('inf'))
        self.return_min_float = Mock(return_value=float('-inf'))
        self.return_nan = Mock(return_value=float('nan'))
    
    def _setup_error_scenarios(self):
        """Set up common error scenarios."""
        # File system errors
        self.file_not_found_error = Mock(side_effect=FileNotFoundError("Test file not found"))
        self.permission_error = Mock(side_effect=PermissionError("Test permission denied"))
        self.is_directory_error = Mock(side_effect=IsADirectoryError("Test is a directory"))
        
        # Network/IO errors
        self.connection_error = Mock(side_effect=ConnectionError("Test connection failed"))
        self.timeout_error = Mock(side_effect=TimeoutError("Test operation timed out"))
        self.io_error = Mock(side_effect=IOError("Test IO operation failed"))
        
        # Data parsing errors
        self.json_decode_error = Mock(side_effect=json.JSONDecodeError("Test JSON decode error", "", 0))
        self.value_error = Mock(side_effect=ValueError("Test value error"))
        self.type_error = Mock(side_effect=TypeError("Test type error"))
        self.key_error = Mock(side_effect=KeyError("test_key"))
        self.index_error = Mock(side_effect=IndexError("Test index error"))
        self.attribute_error = Mock(side_effect=AttributeError("Test attribute error"))
        
        # Memory/resource errors
        self.memory_error = Mock(side_effect=MemoryError("Test memory error"))
        self.overflow_error = Mock(side_effect=OverflowError("Test overflow error"))
        self.recursion_error = Mock(side_effect=RecursionError("Test recursion limit exceeded"))
        
        # Custom application errors
        self.runtime_error = Mock(side_effect=RuntimeError("Test runtime error"))
        self.not_implemented_error = Mock(side_effect=NotImplementedError("Test not implemented"))
    
    @abstractmethod
    def setup_edge_case_scenario(self, scenario_name: str) -> None:
        """
        Set up a specific edge case scenario.
        
        Args:
            scenario_name: Name of the edge case scenario to set up
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def get_edge_case_scenarios(self) -> List[str]:
        """
        Get list of available edge case scenarios.
        
        Returns:
            List of scenario names supported by this mock
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def enable_scenario(self, scenario_name: str) -> None:
        """
        Enable a specific edge case scenario.
        
        Args:
            scenario_name: Name of the scenario to enable
        """
        self.current_scenario = scenario_name
        self.setup_edge_case_scenario(scenario_name)
        self._track_scenario_execution(scenario_name)
    
    def disable_edge_cases(self) -> None:
        """Disable all edge case scenarios and return to normal behavior."""
        self.edge_case_enabled = False
        self.current_scenario = "normal"
    
    def enable_edge_cases(self) -> None:
        """Enable edge case scenarios."""
        self.edge_case_enabled = True
    
    def set_error_probability(self, probability: float) -> None:
        """
        Set the probability of errors occurring.
        
        Args:
            probability: Probability between 0.0 and 1.0
            
        Raises:
            ValueError: If probability is not between 0.0 and 1.0
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
        self.error_probability = probability
    
    def should_trigger_error(self) -> bool:
        """
        Determine if an error should be triggered based on probability.
        
        Returns:
            True if error should be triggered, False otherwise
        """
        if not self.edge_case_enabled:
            return False
        
        if self.max_errors and self.error_count >= self.max_errors:
            return False
        
        import random
        return random.random() < self.error_probability
    
    def track_error(self, error: Exception) -> None:
        """
        Track an error occurrence.
        
        Args:
            error: The exception that was raised
        """
        self.error_count += 1
        self.error_history.append({
            'error_type': type(error).__name__,
            'error_message': str(error),
            'scenario': self.current_scenario,
            'timestamp': Mock()  # Would be actual timestamp in real implementation
        })
    
    def reset_error_tracking(self) -> None:
        """Reset error tracking counters and history."""
        self.error_count = 0
        self.error_history.clear()
        self.executed_scenarios.clear()
        self.scenario_counters.clear()
    
    def _track_scenario_execution(self, scenario_name: str) -> None:
        """Track execution of a scenario."""
        self.executed_scenarios.append(scenario_name)
        self.scenario_counters[scenario_name] = self.scenario_counters.get(scenario_name, 0) + 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about error occurrences.
        
        Returns:
            Dictionary containing error statistics
        """
        error_types = {}
        for error_info in self.error_history:
            error_type = error_info['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': self.error_count,
            'error_types': error_types,
            'scenario_counters': self.scenario_counters.copy(),
            'current_scenario': self.current_scenario,
            'edge_cases_enabled': self.edge_case_enabled,
            'error_probability': self.error_probability
        }
    
    def create_conditional_mock(self, 
                              normal_return: Any = None, 
                              edge_case_return: Any = None,
                              error_class: Type[Exception] = None) -> Mock:
        """
        Create a mock that behaves differently based on edge case settings.
        
        Args:
            normal_return: Value to return in normal conditions
            edge_case_return: Value to return in edge case conditions
            error_class: Exception class to raise if error should be triggered
            
        Returns:
            Mock object with conditional behavior
        """
        def conditional_behavior(*args, **kwargs):
            if self.should_trigger_error() and error_class:
                error = error_class("Test error triggered by edge case mock")
                self.track_error(error)
                raise error
            
            if self.edge_case_enabled and self.current_scenario != "normal":
                return edge_case_return
            
            return normal_return
        
        return Mock(side_effect=conditional_behavior)
    
    def simulate_intermittent_failure(self, 
                                    success_return: Any = None,
                                    failure_count: int = 1,
                                    error_class: Type[Exception] = RuntimeError) -> Mock:
        """
        Create a mock that fails for a specified number of calls then succeeds.
        
        Args:
            success_return: Value to return on successful calls
            failure_count: Number of times to fail before succeeding
            error_class: Exception class to raise on failures
            
        Returns:
            Mock object with intermittent failure behavior
        """
        call_count = [0]  # Use list for closure
        
        def intermittent_behavior(*args, **kwargs):
            call_count[0] += 1
            
            if call_count[0] <= failure_count:
                error = error_class(f"Intermittent failure #{call_count[0]}")
                self.track_error(error)
                raise error
            
            return success_return
        
        return Mock(side_effect=intermittent_behavior)