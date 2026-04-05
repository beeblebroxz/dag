"""
Core DAG infrastructure: Node class, DAG manager, and evaluation engine.

The DAG tracks dependencies between computed functions and manages:
- Caching of computed values
- Invalidation propagation when values change
- Bottom-up evaluation of the dependency graph
- Branch and scenario management for overrides
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)

from .exceptions import CycleError, EvaluationError, UntrackedError
from .flags import NO_VALUE, Flags

if TYPE_CHECKING:
    from .model import Model


class NodeState(Enum):
    """State of a node in the DAG."""
    INVALID = auto()       # Value needs recalculation
    VALID = auto()         # Value is cached and valid
    EVALUATING = auto()    # Currently being evaluated (cycle detection)
    ERROR = auto()         # Evaluation resulted in an error


@dataclass
class NodeKey:
    """
    Unique identifier for a node in the DAG.

    A node is identified by:
    - The object instance it belongs to
    - The method name
    - The arguments (for parameterized computed functions)
    """
    obj_id: int              # id() of the Model instance
    method_name: str         # Name of the computed function
    args: Tuple[Any, ...]    # Hashable arguments

    def __hash__(self) -> int:
        return hash((self.obj_id, self.method_name, self.args))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeKey):
            return NotImplemented
        return (
            self.obj_id == other.obj_id
            and self.method_name == other.method_name
            and self.args == other.args
        )


@dataclass
class Node:
    """
    A node in the dependency graph representing a computed function invocation.

    Nodes store:
    - The cached value
    - Input edges (nodes this depends on)
    - Output edges (nodes that depend on this)
    - State (valid/invalid/evaluating)
    - Flags from the decorator
    """
    key: NodeKey
    obj_ref: weakref.ref     # Weak reference to the Model
    method_name: str
    func: Callable           # The original computed function
    flags: int = Flags.NONE

    # Cached value and state
    _value: Any = field(default=NO_VALUE, repr=False)
    _state: NodeState = field(default=NodeState.INVALID)
    _error: Optional[Exception] = field(default=None, repr=False)

    # Dependency tracking
    # Static dependencies detected at parse time
    static_deps: FrozenSet[str] = field(default_factory=frozenset)
    # Runtime input edges (nodes we depend on)
    inputs: Set[NodeKey] = field(default_factory=set)
    # Runtime output edges (nodes that depend on us)
    outputs: Set[NodeKey] = field(default_factory=set)

    # Branch tracking (which branches have overrides for this node)
    layer_set: Set[int] = field(default_factory=set)

    # Set/override value (if applicable)
    _set_value: Any = field(default=NO_VALUE, repr=False)
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        repr=False,
        compare=False,
    )
    _evaluating_thread_id: Optional[int] = field(default=None, repr=False, compare=False)

    @property
    def value(self) -> Any:
        return self._value

    @property
    def state(self) -> NodeState:
        return self._state

    @property
    def is_valid(self) -> bool:
        return self._state == NodeState.VALID

    @property
    def has_set_value(self) -> bool:
        return self._set_value is not NO_VALUE

    def invalidate(self) -> None:
        """Mark this node as needing recalculation."""
        if self._state == NodeState.VALID:
            self._state = NodeState.INVALID
            self._value = NO_VALUE

    def set_valid(self, value: Any) -> None:
        """Mark this node as valid with the given value."""
        self._value = value
        self._state = NodeState.VALID
        self._error = None

    def set_error(self, error: Exception) -> None:
        """Mark this node as having an error."""
        self._error = error
        self._state = NodeState.ERROR
        self._value = NO_VALUE

    def get_effective_value(self) -> Tuple[bool, Any]:
        """
        Get the effective value considering permanent set values.

        Returns (has_override, value) where has_override indicates if
        the value came from an explicit set rather than computation.
        """
        if self._set_value is not NO_VALUE:
            return (True, self._set_value)
        return (False, self._value)


@dataclass
class _ExecutionState:
    """Per-thread DAG execution state."""

    eval_stack: List[NodeKey] = field(default_factory=list)
    evaluating_node: Optional[Node] = None
    context_stack: List["Scenario"] = field(default_factory=list)
    dependency_stack: List[Set[NodeKey]] = field(default_factory=list)
    tweaks: Dict[NodeKey, Any] = field(default_factory=dict)
    untracked_callers: Dict[NodeKey, int] = field(default_factory=dict)


class DagManager:
    """
    Singleton manager for the DAG.

    Handles:
    - Node registry
    - Evaluation scenario stack
    - Dependency tracking during evaluation
    - Invalidation propagation
    """

    _instance: Optional[DagManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> DagManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Initialize the DAG manager."""
        self._nodes: Dict[NodeKey, Node] = {}
        self._nodes_lock = threading.RLock()
        self._thread_state = threading.local()
        self._layer_counter: int = 0
        self._layer_lock = threading.Lock()
        self._subscriptions: Dict[NodeKey, List[weakref.ref]] = {}
        self._subscriptions_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> DagManager:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the DAG manager (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._init()

    def _get_execution_state(self) -> _ExecutionState:
        """Get the execution state for the current thread."""
        state = getattr(self._thread_state, "state", None)
        if state is None:
            state = _ExecutionState()
            self._thread_state.state = state
        return state

    def get_tweak_value(self, node_key: NodeKey) -> Any:
        """Get the active scenario override for a node in the current thread."""
        return self._get_execution_state().tweaks.get(node_key, NO_VALUE)

    def set_tweak_value(self, node_key: NodeKey, value: Any) -> None:
        """Set or clear the active scenario override for a node in the current thread."""
        state = self._get_execution_state()
        if value is NO_VALUE:
            state.tweaks.pop(node_key, None)
        else:
            state.tweaks[node_key] = value

    def _get_effective_value(self, node: Node) -> Tuple[bool, Any]:
        """Resolve the value visible to the current thread."""
        tweak_value = self.get_tweak_value(node.key)
        if tweak_value is not NO_VALUE:
            return (True, tweak_value)
        return node.get_effective_value()

    def enter_untracked(self) -> Optional[NodeKey]:
        """Disable dependency tracking for the current caller scope."""
        state = self._get_execution_state()
        caller = state.evaluating_node
        if caller is None:
            return None

        state.untracked_callers[caller.key] = state.untracked_callers.get(caller.key, 0) + 1
        return caller.key

    def exit_untracked(self, caller_key: Optional[NodeKey]) -> None:
        """Restore dependency tracking for a caller scope."""
        if caller_key is None:
            return

        state = self._get_execution_state()
        depth = state.untracked_callers.get(caller_key, 0)
        if depth <= 1:
            state.untracked_callers.pop(caller_key, None)
        else:
            state.untracked_callers[caller_key] = depth - 1

    def _is_untracked_for(self, caller_key: NodeKey) -> bool:
        """Return whether dependency tracking is disabled for the given caller."""
        return self._get_execution_state().untracked_callers.get(caller_key, 0) > 0

    def _record_dependency(self, from_node: Node, to_node: Node) -> None:
        """Record a runtime dependency for the currently evaluating node."""
        state = self._get_execution_state()
        if not state.dependency_stack or state.evaluating_node is None:
            return

        if state.evaluating_node.key != from_node.key:
            return

        state.dependency_stack[-1].add(to_node.key)

    def _prepare_dependencies(self, node: Node) -> None:
        """Clear the current runtime dependency edges for a node before recomputing."""
        with self._nodes_lock:
            previous_inputs = list(node.inputs)
            node.inputs.clear()

            for input_key in previous_inputs:
                input_node = self._nodes.get(input_key)
                if input_node is not None:
                    input_node.outputs.discard(node.key)

    def _finalize_dependencies(self, node: Node, new_inputs: Set[NodeKey]) -> None:
        """Install the runtime dependency edges discovered during evaluation."""
        with self._nodes_lock:
            node.inputs = set(new_inputs)
            for input_key in new_inputs:
                input_node = self._nodes.get(input_key)
                if input_node is not None:
                    input_node.outputs.add(node.key)

    def _validate_dependency(self, from_node: Node, to_node: Node) -> None:
        """Ensure a runtime dependency was declared statically."""
        if to_node.method_name in from_node.static_deps:
            return

        raise UntrackedError(
            f"Undeclared dependency from '{from_node.method_name}' to '{to_node.method_name}'. "
            "Wrap the call in dag.untracked(...) if this is intentional."
        )

    def _cleanup_object_nodes(self, obj_id: int) -> None:
        """Remove nodes for an object that has been garbage-collected."""
        stale_nodes: List[Node] = []
        stale_outputs: List[Node] = []

        with self._nodes_lock:
            keys_to_remove = [key for key in self._nodes if key.obj_id == obj_id]
            for key in keys_to_remove:
                node = self._nodes.pop(key, None)
                if node is None:
                    continue

                stale_nodes.append(node)

                for input_key in list(node.inputs):
                    input_node = self._nodes.get(input_key)
                    if input_node is not None:
                        input_node.outputs.discard(node.key)

                for output_key in list(node.outputs):
                    output_node = self._nodes.get(output_key)
                    if output_node is not None:
                        output_node.inputs.discard(node.key)
                        stale_outputs.append(output_node)

        with self._subscriptions_lock:
            for node in stale_nodes:
                self._subscriptions.pop(node.key, None)

        for output_node in stale_outputs:
            self.invalidate_node(output_node)

    def get_or_create_node(
        self,
        obj: Model,
        method_name: str,
        func: Callable,
        flags: int = Flags.NONE,
        static_deps: FrozenSet[str] = frozenset(),
        args: Tuple[Any, ...] = (),
    ) -> Node:
        """Get an existing node or create a new one."""
        key = NodeKey(id(obj), method_name, args)
        with self._nodes_lock:
            if key not in self._nodes:
                manager_ref = weakref.ref(self)

                def _on_object_gc(_ref: weakref.ReferenceType[Model], *, obj_id: int = id(obj)) -> None:
                    manager = manager_ref()
                    if manager is not None:
                        manager._cleanup_object_nodes(obj_id)

                node = Node(
                    key=key,
                    obj_ref=weakref.ref(obj, _on_object_gc),
                    method_name=method_name,
                    func=func,
                    flags=flags,
                    static_deps=static_deps,
                )
                self._nodes[key] = node

            return self._nodes[key]

    def get_node(self, key: NodeKey) -> Optional[Node]:
        """Get a node by its key."""
        with self._nodes_lock:
            return self._nodes.get(key)

    def evaluate(self, node: Node, args: Tuple[Any, ...] = ()) -> Any:
        """
        Evaluate a node, computing its value if necessary.

        Uses bottom-up evaluation: dependencies are evaluated first.
        Results are cached for valid nodes.

        Runtime dependencies are tracked: when node A calls node B during
        evaluation, B is recorded as a dependency of A.
        """
        state = self._get_execution_state()

        # Track runtime dependency: if we're evaluating another node,
        # this node becomes a dependency of that node
        caller = state.evaluating_node
        if caller is not None and caller.key != node.key:
            if not self._is_untracked_for(caller.key):
                self._validate_dependency(caller, node)
                self._record_dependency(caller, node)

        current_thread_id = threading.get_ident()
        cycle_error: Optional[CycleError] = None

        with node._condition:
            has_override, override_value = self._get_effective_value(node)
            if has_override:
                return override_value

            if node.is_valid:
                return node.value

            if node.key in state.eval_stack:
                cycle_path = state.eval_stack[state.eval_stack.index(node.key):]
                cycle_str = " -> ".join(k.method_name for k in cycle_path)
                cycle_error = CycleError(
                    f"Cyclic dependency detected: {cycle_str} -> {node.key.method_name}"
                )
            else:
                while (
                    node.state == NodeState.EVALUATING
                    and node._evaluating_thread_id != current_thread_id
                ):
                    node._condition.wait()
                    has_override, override_value = self._get_effective_value(node)
                    if has_override:
                        return override_value
                    if node.is_valid:
                        return node.value

                if node.key in state.eval_stack:
                    cycle_path = state.eval_stack[state.eval_stack.index(node.key):]
                    cycle_str = " -> ".join(k.method_name for k in cycle_path)
                    cycle_error = CycleError(
                        f"Cyclic dependency detected: {cycle_str} -> {node.key.method_name}"
                    )
                else:
                    has_override, override_value = self._get_effective_value(node)
                    if has_override:
                        return override_value

                    if node.is_valid:
                        return node.value

                    node._state = NodeState.EVALUATING
                    node._evaluating_thread_id = current_thread_id

        if cycle_error is not None:
            raise cycle_error

        self._prepare_dependencies(node)
        state.eval_stack.append(node.key)
        state.dependency_stack.append(set())
        previous_evaluating = state.evaluating_node
        state.evaluating_node = node

        try:
            # Get the object
            obj = node.obj_ref()
            if obj is None:
                raise EvaluationError(node.method_name, RuntimeError("Object has been garbage collected"))

            # Evaluate the function
            # Dependencies will be tracked as computed functions are called
            result = node.func(obj, *args)

            # Cache the result
            with node._condition:
                node.set_valid(result)
                node._evaluating_thread_id = None
                node._condition.notify_all()
            return result

        except Exception as e:
            with node._condition:
                if node.flags & Flags.Optional:
                    node.set_valid(NO_VALUE)
                    node._evaluating_thread_id = None
                    node._condition.notify_all()
                    return NO_VALUE

                node.set_error(e)
                node._evaluating_thread_id = None
                node._condition.notify_all()

            if isinstance(e, UntrackedError):
                raise

            raise EvaluationError(node.method_name, e) from e

        finally:
            discovered_inputs = state.dependency_stack.pop()
            self._finalize_dependencies(node, discovered_inputs)
            state.eval_stack.pop()
            state.evaluating_node = previous_evaluating

    def invalidate_node(self, node: Node, propagate_only: bool = False) -> None:
        """
        Invalidate a node and propagate to dependents.

        When a node is invalidated, all nodes that depend on it
        must also be invalidated.

        Args:
            node: The node to invalidate
            propagate_only: If True, only propagate to dependents without
                           invalidating this node (used when set changes)
        """
        if not propagate_only:
            if not node.is_valid:
                return
            node.invalidate()

        # Propagate to outputs (dependents)
        for output_key in list(node.outputs):  # list() to avoid mutation during iteration
            output_node = self.get_node(output_key)
            if output_node is not None:
                self.invalidate_node(output_node)

        # Queue subscription notifications
        self._queue_subscription(node.key)

    def invalidate_dependents(self, node: Node) -> None:
        """
        Invalidate only the dependents of a node, not the node itself.

        Used when a node's value is explicitly set - the node stays valid
        with its set value, but dependents need to recompute.
        """
        for output_key in list(node.outputs):
            output_node = self.get_node(output_key)
            if output_node is not None:
                self.invalidate_node(output_node)

    def add_dependency(self, from_node: Node, to_node: Node) -> None:
        """Add a dependency edge: from_node depends on to_node."""
        from_node.inputs.add(to_node.key)
        to_node.outputs.add(from_node.key)

    def remove_dependency(self, from_node: Node, to_node: Node) -> None:
        """Remove a dependency edge."""
        from_node.inputs.discard(to_node.key)
        to_node.outputs.discard(from_node.key)

    # Scenario management
    def push_context(self, ctx: Scenario) -> None:
        """Push a new scenario onto the stack."""
        state = self._get_execution_state()
        state.context_stack.append(ctx)

    def pop_context(self) -> Optional[Scenario]:
        """Pop the current scenario from the stack."""
        state = self._get_execution_state()
        if state.context_stack:
            ctx = state.context_stack.pop()
            return ctx
        return None

    @property
    def current_context(self) -> Optional[Scenario]:
        """Get the current scenario."""
        state = self._get_execution_state()
        if state.context_stack:
            return state.context_stack[-1]
        return None

    def next_layer_id(self) -> int:
        """Get the next branch ID."""
        with self._layer_lock:
            self._layer_counter += 1
            return self._layer_counter

    # Subscriptions
    def subscribe(self, node_key: NodeKey, callback: Callable) -> None:
        """Subscribe to notifications when a node is invalidated."""
        with self._subscriptions_lock:
            if node_key not in self._subscriptions:
                self._subscriptions[node_key] = []
            self._subscriptions[node_key].append(weakref.ref(callback))

    def _queue_subscription(self, node_key: NodeKey) -> None:
        """Queue a subscription notification (lazy dispatch)."""
        # For now, we don't dispatch immediately - use flush()
        pass

    def flush(self) -> None:
        """Dispatch all queued subscription notifications."""
        # Clean up dead references and invoke callbacks
        with self._subscriptions_lock:
            subscriptions = list(self._subscriptions.items())

        for node_key, callbacks in subscriptions:
            node = self.get_node(node_key)
            if node is None or node.is_valid:
                continue

            live_callbacks = []
            for cb_ref in callbacks:
                cb = cb_ref()
                if cb is not None:
                    live_callbacks.append(cb_ref)
                    try:
                        cb(node)
                    except Exception:
                        pass  # Don't let callback errors propagate

            with self._subscriptions_lock:
                self._subscriptions[node_key] = live_callbacks



class Scenario:
    """
    A scenario for temporary overrides.

    Overrides made within a scenario are automatically reverted when
    the scenario exits.
    """

    def __init__(self):
        self._dag = DagManager.get_instance()
        self._tweaks: List[Tuple[Node, Any]] = []  # (node, old_override_value)
        self._layer_id = self._dag.next_layer_id()

    def __enter__(self) -> Scenario:
        self._dag.push_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        # Revert all overrides
        for node, old_value in reversed(self._tweaks):
            self._dag.set_tweak_value(node.key, old_value)
            # Always invalidate dependents when reverting an override
            # The node's dependents need to recompute with the original value
            self._dag.invalidate_dependents(node)
            if old_value is NO_VALUE:
                # Also invalidate this node since we're reverting to computed value
                node.invalidate()

        self._dag.pop_context()
        return False

    def add_tweak(self, node: Node, new_value: Any) -> None:
        """Record an override for later reversion."""
        old_value = self._dag.get_tweak_value(node.key)
        self._tweaks.append((node, old_value))
        self._dag.set_tweak_value(node.key, new_value)
        # Must invalidate dependents even if this node is already invalid
        # (e.g., when node has a set_value but state is INVALID)
        self._dag.invalidate_dependents(node)

    @property
    def layer_id(self) -> int:
        return self._layer_id


# Convenience function for creating scenarios
def scenario() -> Scenario:
    """Create a new DAG scenario for temporary overrides."""
    return Scenario()


# Module-level function for flushing subscriptions
def flush() -> None:
    """Dispatch all pending subscription notifications."""
    DagManager.get_instance().flush()
