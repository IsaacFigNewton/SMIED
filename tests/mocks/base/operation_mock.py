"""
Abstract base class for operation-specific mocks.

This module provides the AbstractOperationMock class that serves as a base
for operation-specific mocks including validation, canonicalization, manipulation,
and conversion operations.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from enum import Enum


class OperationStatus(Enum):
    """Enumeration of operation statuses."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class OperationType(Enum):
    """Enumeration of operation types."""
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CONVERSION = "conversion"
    MANIPULATION = "manipulation"
    CANONICALIZATION = "canonicalization"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"


class AbstractOperationMock(ABC, Mock):
    """
    Abstract base class for operation-specific mocks.
    
    This class provides a common interface for mocks that represent various
    operations like validation, transformation, conversion, manipulation, etc.
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the AbstractOperationMock.
        
        Args:
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes()
        self._setup_operation_tracking()
        self._setup_execution_context()
    
    def _setup_common_attributes(self):
        """Set up common attributes for operations."""
        # Operation identification
        self.operation_name = "unknown_operation"
        self.operation_type = OperationType.VALIDATION
        self.operation_version = "1.0"
        self.operation_id = None
        
        # Operation properties
        self.is_reversible = False
        self.is_idempotent = True
        self.is_atomic = True
        self.supports_batch = False
        
        # State management
        self.status = OperationStatus.PENDING
        self.progress = 0.0  # 0.0 to 1.0
        self.result = None
        self.error = None
        
        # Configuration
        self.strict_mode = True
        self.dry_run_mode = False
        self.validation_enabled = True
        self.auto_fix_enabled = False
        
        # Dependencies
        self.prerequisites = []
        self.dependencies = []
        self.postconditions = []
    
    def _setup_operation_tracking(self):
        """Set up operation tracking and metrics."""
        # Execution tracking
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.skip_count = 0
        
        # Performance metrics
        self.total_execution_time = 0.0
        self.average_execution_time = 0.0
        self.min_execution_time = float('inf')
        self.max_execution_time = 0.0
        
        # Operation history
        self.execution_history = []
        self.error_history = []
        self.performance_history = []
        
        # Resource usage
        self.memory_usage = 0
        self.cpu_usage = 0.0
        self.io_operations = 0
    
    def _setup_execution_context(self):
        """Set up execution context and environment."""
        # Context information
        self.execution_context = {}
        self.environment_variables = {}
        self.configuration = {}
        
        # Callbacks and hooks
        self.pre_execution_hooks = []
        self.post_execution_hooks = []
        self.error_handlers = []
        self.progress_callbacks = []
        
        # Rollback support
        self.rollback_data = None
        self.checkpoint_data = {}
        self.transaction_id = None
    
    @abstractmethod
    def execute(self, target: Any, *args, **kwargs) -> Any:
        """
        Execute the operation on the target.
        
        Args:
            target: The target object to operate on
            *args: Additional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def validate_target(self, target: Any) -> bool:
        """
        Validate that the target is suitable for this operation.
        
        Args:
            target: Target object to validate
            
        Returns:
            True if target is valid, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def get_operation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata specific to this operation type.
        
        Returns:
            Dictionary containing operation-specific metadata
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def __call__(self, target: Any, *args, **kwargs) -> Any:
        """
        Make the mock callable to execute the operation.
        
        Args:
            target: Target object for the operation
            *args: Additional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Operation result
        """
        # Check prerequisites
        if not self._check_prerequisites():
            raise RuntimeError("Prerequisites not met for operation")
        
        # Validate target if validation is enabled
        if self.validation_enabled and not self.validate_target(target):
            raise ValueError("Target validation failed")
        
        # Set status to running
        self.status = OperationStatus.RUNNING
        self.progress = 0.0
        
        try:
            # Run pre-execution hooks
            self._run_pre_execution_hooks(target, *args, **kwargs)
            
            # Execute the operation
            if self.dry_run_mode:
                result = self._dry_run_execute(target, *args, **kwargs)
            else:
                result = self.execute(target, *args, **kwargs)
            
            # Update state
            self.result = result
            self.status = OperationStatus.SUCCESS
            self.progress = 1.0
            self.success_count += 1
            
            # Run post-execution hooks
            self._run_post_execution_hooks(target, result, *args, **kwargs)
            
            return result
            
        except Exception as e:
            # Handle error
            self.error = e
            self.status = OperationStatus.FAILED
            self.failure_count += 1
            self.error_history.append({
                'error': e,
                'target': target,
                'args': args,
                'kwargs': kwargs,
                'timestamp': Mock()  # Would be actual timestamp
            })
            
            # Run error handlers
            self._run_error_handlers(e, target, *args, **kwargs)
            
            raise
        finally:
            # Update execution tracking
            self.execution_count += 1
            execution_time = 0.001  # Mock execution time
            self._update_performance_metrics(execution_time)
    
    def _check_prerequisites(self) -> bool:
        """
        Check if all prerequisites are met.
        
        Returns:
            True if prerequisites are met, False otherwise
        """
        for prerequisite in self.prerequisites:
            if callable(prerequisite):
                if not prerequisite():
                    return False
            elif hasattr(prerequisite, 'is_satisfied'):
                if not prerequisite.is_satisfied():
                    return False
        
        return True
    
    def _dry_run_execute(self, target: Any, *args, **kwargs) -> Any:
        """
        Execute operation in dry-run mode (simulation).
        
        Args:
            target: Target object
            *args: Additional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Simulated result
        """
        # Create a mock result based on operation type
        if self.operation_type == OperationType.VALIDATION:
            return Mock(is_valid=True, issues=[])
        elif self.operation_type == OperationType.TRANSFORMATION:
            return Mock(transformed_target=target)
        elif self.operation_type == OperationType.CONVERSION:
            return Mock(converted_target=target)
        else:
            return Mock(success=True, message="Dry run completed")
    
    def _run_pre_execution_hooks(self, target: Any, *args, **kwargs) -> None:
        """Run pre-execution hooks."""
        for hook in self.pre_execution_hooks:
            try:
                hook(self, target, *args, **kwargs)
            except Exception as e:
                # Log hook errors but don't fail the operation
                pass
    
    def _run_post_execution_hooks(self, target: Any, result: Any, *args, **kwargs) -> None:
        """Run post-execution hooks."""
        for hook in self.post_execution_hooks:
            try:
                hook(self, target, result, *args, **kwargs)
            except Exception as e:
                # Log hook errors but don't fail the operation
                pass
    
    def _run_error_handlers(self, error: Exception, target: Any, *args, **kwargs) -> None:
        """Run error handlers."""
        for handler in self.error_handlers:
            try:
                handler(self, error, target, *args, **kwargs)
            except Exception:
                # Error handlers should not raise exceptions
                pass
    
    def _update_performance_metrics(self, execution_time: float) -> None:
        """
        Update performance tracking metrics.
        
        Args:
            execution_time: Time taken for this execution
        """
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.execution_count
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        
        # Record performance data
        self.performance_history.append({
            'execution_time': execution_time,
            'memory_usage': self.memory_usage,
            'timestamp': Mock()
        })
    
    def add_prerequisite(self, prerequisite: Union[Callable, Any]) -> None:
        """
        Add a prerequisite for this operation.
        
        Args:
            prerequisite: Callable or object with is_satisfied() method
        """
        self.prerequisites.append(prerequisite)
    
    def add_pre_execution_hook(self, hook: Callable) -> None:
        """
        Add a pre-execution hook.
        
        Args:
            hook: Function to call before execution
        """
        self.pre_execution_hooks.append(hook)
    
    def add_post_execution_hook(self, hook: Callable) -> None:
        """
        Add a post-execution hook.
        
        Args:
            hook: Function to call after execution
        """
        self.post_execution_hooks.append(hook)
    
    def add_error_handler(self, handler: Callable) -> None:
        """
        Add an error handler.
        
        Args:
            handler: Function to call on error
        """
        self.error_handlers.append(handler)
    
    def create_checkpoint(self, name: str) -> None:
        """
        Create a checkpoint for rollback purposes.
        
        Args:
            name: Name of the checkpoint
        """
        self.checkpoint_data[name] = {
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'timestamp': Mock()
        }
    
    def rollback_to_checkpoint(self, name: str) -> bool:
        """
        Rollback to a specific checkpoint.
        
        Args:
            name: Name of the checkpoint
            
        Returns:
            True if rollback successful, False otherwise
        """
        if name not in self.checkpoint_data:
            return False
        
        checkpoint = self.checkpoint_data[name]
        self.status = checkpoint['status']
        self.progress = checkpoint['progress']
        self.result = checkpoint['result']
        
        return True
    
    def cancel_operation(self) -> None:
        """Cancel the operation if it's currently running."""
        if self.status == OperationStatus.RUNNING:
            self.status = OperationStatus.CANCELLED
    
    def reset_operation(self) -> None:
        """Reset operation state to initial conditions."""
        self.status = OperationStatus.PENDING
        self.progress = 0.0
        self.result = None
        self.error = None
        self.rollback_data = None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive execution summary.
        
        Returns:
            Dictionary containing execution statistics and status
        """
        return {
            'operation_info': {
                'name': self.operation_name,
                'type': self.operation_type.value,
                'version': self.operation_version,
                'id': self.operation_id
            },
            'current_state': {
                'status': self.status.value,
                'progress': self.progress,
                'has_result': self.result is not None,
                'has_error': self.error is not None
            },
            'execution_stats': {
                'total_executions': self.execution_count,
                'successes': self.success_count,
                'failures': self.failure_count,
                'skips': self.skip_count,
                'success_rate': (self.success_count / self.execution_count 
                               if self.execution_count > 0 else 0.0)
            },
            'performance_stats': {
                'total_time': self.total_execution_time,
                'average_time': self.average_execution_time,
                'min_time': (self.min_execution_time 
                           if self.min_execution_time != float('inf') else 0),
                'max_time': self.max_execution_time
            },
            'configuration': {
                'strict_mode': self.strict_mode,
                'dry_run_mode': self.dry_run_mode,
                'validation_enabled': self.validation_enabled,
                'auto_fix_enabled': self.auto_fix_enabled
            },
            'dependencies': {
                'prerequisites': len(self.prerequisites),
                'dependencies': len(self.dependencies),
                'postconditions': len(self.postconditions)
            }
        }
    
    def simulate_batch_operation(self, targets: List[Any], **kwargs) -> List[Any]:
        """
        Simulate batch operation on multiple targets.
        
        Args:
            targets: List of target objects
            **kwargs: Operation parameters
            
        Returns:
            List of operation results
        """
        if not self.supports_batch:
            # Process individually
            return [self(target, **kwargs) for target in targets]
        
        # Batch processing simulation
        results = []
        batch_size = kwargs.get('batch_size', 10)
        
        for i in range(0, len(targets), batch_size):
            batch = targets[i:i + batch_size]
            
            # Simulate batch processing
            batch_results = []
            for target in batch:
                result = self(target, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Update progress
            self.progress = min(1.0, (i + len(batch)) / len(targets))
        
        return results
    
    def create_operation_pipeline(self, operations: List['AbstractOperationMock']) -> 'OperationPipeline':
        """
        Create a pipeline with this and other operations.
        
        Args:
            operations: List of operations to include in pipeline
            
        Returns:
            Operation pipeline mock
        """
        pipeline = Mock()
        pipeline.operations = [self] + operations
        pipeline.execute = Mock(return_value=Mock(success=True))
        pipeline.validate = Mock(return_value=True)
        
        return pipeline