"""
DAG - A dependency tracking and memoization framework for Python.

This framework provides:
- Automatic dependency tracking between functions
- Memoization with intelligent cache invalidation
- Support for scenarios via overrides and branches
- Parse-time dependency detection via AST analysis

Basic Usage:
    import dag

    class SimpleOption(dag.Model):
        @dag.computed
        def Pair(self):
            return 'EURUSD'

        @dag.computed(dag.Input)
        def Strike(self):
            return 1.0  # default value

        @dag.computed(dag.Overridable)
        def Spot(self):
            return 1.1

        @dag.computed
        def Payoff(self):
            return max(0, self.Spot() - self.Strike())

    # Create an option
    opt = SimpleOption()
    print(opt.Payoff())  # Computes and caches: 0.1

    # Change strike permanently
    opt.Strike = 1.05
    print(opt.Payoff())  # Recomputes: 0.05

    # Temporary override
    with dag.scenario():
        opt.Spot.override(1.2)
        print(opt.Payoff())  # Computes with overridden spot: 0.15
    # Spot reverts here
    print(opt.Payoff())  # Back to: 0.05

Key Concepts:
    - Model: Base class for objects with tracked methods
    - computed: Decorator to mark methods as computed functions
    - Input: Flag allowing permanent value changes
    - Overridable: Flag allowing temporary value overrides
    - scenario(): Context manager for temporary overrides
    - branch(): Creates parallel graph states

See the individual module documentation for more details.
"""

from .model import Model, Registry, RegistryMixin
from .core import Scenario, DagManager, Node, NodeKey, scenario, flush
from .decorators import ComputedFunctionAccessor, ComputedFunctionDescriptor, NodeChange, computed
from .exceptions import (
    ModelError,
    ConstructorError,
    ScenarioError,
    CycleError,
    DagError,
    DependencyError,
    EvaluationError,
    InvalidationError,
    UntrackedError,
    ParseError,
    SetValueError,
    OverrideError,
)
from .flags import (
    NO_VALUE,
    CanChange,
    Input,
    Overridable,
    Remote,
    Flags,
    Serialized,
    Persisted,
    Optional,
)
from .parser import parse_dependencies, parse_dependencies_detailed
from .state import Branch, Override, OverrideSet, apply_overrides, get_overrides, branch, untracked

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Model",
    "ComputedFunctionDescriptor",
    "ComputedFunctionAccessor",
    "Scenario",
    "DagManager",
    "Node",
    "NodeKey",
    "NodeChange",
    "Registry",
    "RegistryMixin",
    # Decorators
    "computed",
    # Flags
    "Flags",
    "Serialized",
    "Input",
    "Overridable",
    "Persisted",
    "Optional",
    "Remote",
    "CanChange",
    "NO_VALUE",
    # State management
    "scenario",
    "branch",
    "Branch",
    "Override",
    "OverrideSet",
    "get_overrides",
    "apply_overrides",
    "untracked",
    "flush",
    # Parser
    "parse_dependencies",
    "parse_dependencies_detailed",
    # Exceptions
    "DagError",
    "DependencyError",
    "UntrackedError",
    "CycleError",
    "InvalidationError",
    "SetValueError",
    "OverrideError",
    "ScenarioError",
    "EvaluationError",
    "ParseError",
    "ModelError",
    "ConstructorError",
]


def reset() -> None:
    """Reset the DAG manager (for testing)."""
    DagManager.reset()


# Convenience function for browsing the DAG (like the spec's qzdev browser)
def browse(computed_accessor: ComputedFunctionAccessor) -> None:
    """
    Browse a computed function's dependencies and value.

    Similar to the graph browser described in the spec, this helps
    inspect how a calculation arrived at its result.
    """
    node = computed_accessor._node
    if node is None:
        print(f"Node not yet created for {computed_accessor._descriptor.name}")
        return

    print(f"Computed: {node.method_name}")
    print(f"State: {node.state.name}")
    print(f"Value: {node.value}")
    print(f"Flags: {node.flags}")
    print(f"Static deps: {node.static_deps}")
    print(f"Runtime inputs: {[k.method_name for k in node.inputs]}")
    print(f"Runtime outputs: {[k.method_name for k in node.outputs]}")


def filter_nodes(
    model_type: type,
    root_accessor: ComputedFunctionAccessor,
) -> list:
    """
    Filter nodes in the DAG by type.

    Returns all settable/overridable nodes belonging to objects of the
    given type underneath the graph of root_accessor.

    Similar to dag.filter(sandra.cells.MarketInterface, p.Price) from the spec.
    """
    # Simplified implementation - full version would walk the graph
    dag = DagManager.get_instance()
    results = []

    for node in dag._nodes.values():
        obj = node.obj_ref()
        if obj is not None and isinstance(obj, model_type):
            if node.flags & (Input | Overridable):
                results.append(node)

    return results
