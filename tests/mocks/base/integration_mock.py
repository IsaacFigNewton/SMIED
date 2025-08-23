"""
Abstract base class for integration testing mocks.

This module provides the AbstractIntegrationMock class that serves as a base
for integration testing mocks that compose multiple mock components together.
"""

from abc import ABC, abstractmethod
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from contextlib import contextmanager


class AbstractIntegrationMock(ABC, Mock):
    """
    Abstract base class for integration testing mocks.
    
    This class provides a common interface for mocks that combine multiple
    components to simulate integrated system behavior in tests.
    
    Inherits from both ABC and Mock to provide abstract method enforcement
    while maintaining Mock functionality for testing.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the AbstractIntegrationMock.
        
        Args:
            *args: Variable length argument list passed to Mock
            **kwargs: Arbitrary keyword arguments passed to Mock
        """
        super().__init__(*args, **kwargs)
        self._setup_common_attributes()
        self._setup_component_registry()
        self._setup_interaction_tracking()
    
    def _setup_common_attributes(self):
        """Set up common attributes for integration testing."""
        # Component management
        self.components = {}
        self.component_dependencies = {}
        self.component_call_order = []
        
        # Integration configuration
        self.integration_mode = "full"  # full, partial, minimal
        self.auto_setup = True
        self.strict_ordering = False
        
        # State management
        self.integration_state = {}
        self.shared_context = {}
        self.transaction_id = None
        
        # Monitoring and debugging
        self.interaction_log = []
        self.component_states = {}
        self.call_graph = {}
    
    def _setup_component_registry(self):
        """Set up component registry for managing mock components."""
        self.component_registry = {}
        self.component_factories = {}
        self.component_configs = {}
        
        # Component lifecycle hooks
        self.pre_component_setup_hooks = []
        self.post_component_setup_hooks = []
        self.pre_integration_hooks = []
        self.post_integration_hooks = []
    
    def _setup_interaction_tracking(self):
        """Set up interaction tracking between components."""
        self.interactions = {}
        self.call_sequences = {}
        self.data_flow = {}
        self.timing_constraints = {}
        
        # Performance tracking
        self.call_counts = {}
        self.execution_times = {}
        self.memory_usage = {}
    
    @abstractmethod
    def setup_integration_components(self) -> Dict[str, Any]:
        """
        Set up all components required for integration testing.
        
        Returns:
            Dictionary mapping component names to component instances
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def configure_component_interactions(self) -> None:
        """
        Configure how components interact with each other.
        
        This method should set up the relationships, dependencies, and
        communication patterns between components.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def validate_integration_state(self) -> bool:
        """
        Validate that the integration is in a consistent state.
        
        Returns:
            True if integration state is valid, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def register_component(self, 
                          name: str, 
                          component: Any, 
                          dependencies: Optional[List[str]] = None,
                          config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a component in the integration.
        
        Args:
            name: Unique name for the component
            component: The component instance or mock
            dependencies: List of component names this component depends on
            config: Configuration for the component
        """
        self.components[name] = component
        self.component_dependencies[name] = dependencies or []
        self.component_configs[name] = config or {}
        
        # Initialize tracking for this component
        self.call_counts[name] = 0
        self.execution_times[name] = []
        self.component_states[name] = "registered"
    
    def get_component(self, name: str) -> Any:
        """
        Get a registered component by name.
        
        Args:
            name: Name of the component to retrieve
            
        Returns:
            The component instance
            
        Raises:
            KeyError: If component is not registered
        """
        if name not in self.components:
            raise KeyError(f"Component '{name}' not registered")
        return self.components[name]
    
    def initialize_integration(self) -> None:
        """Initialize the integration by setting up all components."""
        # Run pre-integration hooks
        for hook in self.pre_integration_hooks:
            hook(self)
        
        # Set up components in dependency order
        setup_order = self._resolve_dependency_order()
        
        for component_name in setup_order:
            self._initialize_component(component_name)
        
        # Configure component interactions
        self.configure_component_interactions()
        
        # Run post-integration hooks
        for hook in self.post_integration_hooks:
            hook(self)
        
        # Validate integration state
        if not self.validate_integration_state():
            raise RuntimeError("Integration validation failed")
    
    def _resolve_dependency_order(self) -> List[str]:
        """
        Resolve the order in which components should be initialized.
        
        Returns:
            List of component names in dependency order
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(component_name: str):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{component_name}'")
            
            if component_name not in visited:
                temp_visited.add(component_name)
                
                for dependency in self.component_dependencies.get(component_name, []):
                    visit(dependency)
                
                temp_visited.remove(component_name)
                visited.add(component_name)
                order.append(component_name)
        
        for component_name in self.components:
            visit(component_name)
        
        return order
    
    def _initialize_component(self, component_name: str) -> None:
        """
        Initialize a specific component.
        
        Args:
            component_name: Name of the component to initialize
        """
        component = self.components[component_name]
        config = self.component_configs.get(component_name, {})
        
        # Run pre-component setup hooks
        for hook in self.pre_component_setup_hooks:
            hook(component_name, component, config)
        
        # Initialize component if it has an init method
        if hasattr(component, 'initialize'):
            component.initialize(config)
        
        # Update component state
        self.component_states[component_name] = "initialized"
        
        # Run post-component setup hooks
        for hook in self.post_component_setup_hooks:
            hook(component_name, component, config)
    
    def simulate_integration_workflow(self, workflow_steps: List[str]) -> List[Any]:
        """
        Simulate a complete integration workflow.
        
        Args:
            workflow_steps: List of method names or workflow step identifiers
            
        Returns:
            List of results from each workflow step
        """
        results = []
        
        for step in workflow_steps:
            self._log_interaction("workflow_step", step, None, None)
            
            if hasattr(self, step):
                result = getattr(self, step)()
                results.append(result)
            else:
                # Assume it's a component method call
                component_name, method_name = step.split('.', 1) if '.' in step else (step, 'execute')
                component = self.get_component(component_name)
                result = getattr(component, method_name)()
                results.append(result)
        
        return results
    
    def create_component_interaction(self, 
                                   source_component: str, 
                                   target_component: str,
                                   interaction_type: str = "call",
                                   data_transform: Optional[Callable] = None) -> Mock:
        """
        Create a mock interaction between two components.
        
        Args:
            source_component: Name of the source component
            target_component: Name of the target component
            interaction_type: Type of interaction (call, data_flow, event, etc.)
            data_transform: Optional function to transform data between components
            
        Returns:
            Mock object representing the interaction
        """
        def interaction_handler(*args, **kwargs):
            # Log the interaction
            self._log_interaction(interaction_type, source_component, target_component, args, kwargs)
            
            # Transform data if transformer provided
            if data_transform:
                args, kwargs = data_transform(args, kwargs)
            
            # Get target component and call it
            target = self.get_component(target_component)
            if hasattr(target, 'handle_interaction'):
                return target.handle_interaction(source_component, interaction_type, *args, **kwargs)
            
            return Mock(return_value="interaction_result")
        
        return Mock(side_effect=interaction_handler)
    
    def _log_interaction(self, 
                        interaction_type: str, 
                        source: str, 
                        target: Optional[str], 
                        args: Optional[Tuple] = None,
                        kwargs: Optional[Dict] = None) -> None:
        """Log an interaction between components."""
        interaction_record = {
            'type': interaction_type,
            'source': source,
            'target': target,
            'args': args,
            'kwargs': kwargs,
            'timestamp': Mock(),  # Would be actual timestamp in real implementation
            'transaction_id': self.transaction_id
        }
        
        self.interaction_log.append(interaction_record)
        
        # Update call counts
        if source in self.call_counts:
            self.call_counts[source] += 1
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all interactions that occurred.
        
        Returns:
            Dictionary containing interaction statistics and patterns
        """
        interaction_types = {}
        component_interactions = {}
        
        for interaction in self.interaction_log:
            # Count interaction types
            interaction_type = interaction['type']
            interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
            
            # Count component interactions
            source = interaction['source']
            target = interaction['target']
            if source not in component_interactions:
                component_interactions[source] = {}
            if target:
                component_interactions[source][target] = component_interactions[source].get(target, 0) + 1
        
        return {
            'total_interactions': len(self.interaction_log),
            'interaction_types': interaction_types,
            'component_interactions': component_interactions,
            'call_counts': self.call_counts.copy(),
            'component_states': self.component_states.copy()
        }
    
    @contextmanager
    def integration_transaction(self, transaction_id: Optional[str] = None):
        """
        Context manager for grouping integration operations into a transaction.
        
        Args:
            transaction_id: Optional ID for the transaction
            
        Yields:
            The transaction ID
        """
        old_transaction_id = self.transaction_id
        self.transaction_id = transaction_id or f"txn_{len(self.interaction_log)}"
        
        try:
            yield self.transaction_id
        finally:
            self.transaction_id = old_transaction_id
    
    def reset_integration(self) -> None:
        """Reset the integration state and clear all tracking data."""
        # Clear tracking data
        self.interaction_log.clear()
        self.call_counts = {name: 0 for name in self.components}
        self.execution_times.clear()
        self.component_states = {name: "registered" for name in self.components}
        
        # Clear state
        self.integration_state.clear()
        self.shared_context.clear()
        self.transaction_id = None
        
        # Reset components if they have reset methods
        for name, component in self.components.items():
            if hasattr(component, 'reset'):
                component.reset()
    
    def create_realistic_integration_flow(self) -> List[Mock]:
        """
        Create a realistic flow of component interactions.
        
        Returns:
            List of mock objects representing the integration flow
        """
        flow = []
        
        # Initialize all components
        for name in self.components:
            init_mock = Mock(return_value=f"{name}_initialized")
            flow.append(init_mock)
        
        # Create typical interaction patterns
        component_names = list(self.components.keys())
        for i, source in enumerate(component_names):
            for j, target in enumerate(component_names[i+1:], i+1):
                interaction_mock = self.create_component_interaction(source, target)
                flow.append(interaction_mock)
        
        return flow