"""
Core DAG infrastructure: Node class, DAG manager, and evaluation engine.

The DAG tracks dependencies between computed functions and manages:
- Caching of computed values
- Invalidation propagation when values change
- Bottom-up evaluation of the dependency graph
- Branch and scenario management for overrides
"""

from __future__ import annotations

import inspect
import logging
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
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)

from .exceptions import (
    ConcurrentScenarioError,
    CycleError,
    EvaluationError,
    OverrideError,
    UntrackedError,
)
from .flags import NO_VALUE, Flags

if TYPE_CHECKING:
    from .model import Model


logger = logging.getLogger(__name__)


class NodeState(Enum):
    """State of a node in the DAG."""
    INVALID = auto()       # Value needs recalculation
    VALID = auto()         # Value is cached and valid
    EVALUATING = auto()    # Currently being evaluated (cycle detection)
    ERROR = auto()         # Evaluation resulted in an error


@dataclass(frozen=True)
class NodeKey:
    """
    Unique identifier for a node in the DAG.

    A node is identified by:
    - The object instance it belongs to
    - The method name
    - The arguments (for parameterized computed functions)

    Frozen so it is hashable and usable as a dict key; equality and hashing
    are derived from (obj_id, method_name, args).
    """
    obj_id: int              # id() of the Model instance
    method_name: str         # Name of the computed function
    args: Tuple[Any, ...]    # Hashable arguments


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
    # Static dependencies detected at parse time (None = source unparseable,
    # so dependencies are unknown and the untracked check is skipped).
    static_deps: Optional[FrozenSet[str]] = field(default_factory=frozenset)
    # Runtime input edges (nodes we depend on)
    inputs: Set[NodeKey] = field(default_factory=set)
    # Runtime output edges (nodes that depend on us)
    outputs: Set[NodeKey] = field(default_factory=set)

    # Set/override value (if applicable)
    _set_value: Any = field(default=NO_VALUE, repr=False)
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        repr=False,
        compare=False,
    )
    _evaluating_thread_id: Optional[int] = field(default=None, repr=False, compare=False)
    # Bumped whenever an invalidation reaches this node from a thread other
    # than its evaluator. An evaluation compares the count before and after
    # to detect that an input changed underneath it (see DagManager._settle).
    _invalidation_count: int = field(default=0, repr=False, compare=False)

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

    @property
    def has_result(self) -> bool:
        """Whether the last evaluation finished, with a value or an error.

        Dependents may have consumed either outcome (a dependent can catch a
        dependency's exception and cache a fallback), so invalidation must
        treat an ERROR node like a VALID one.
        """
        return self._state in (NodeState.VALID, NodeState.ERROR)

    def invalidate(self) -> None:
        """Mark this node as needing recalculation, discarding a cached
        value or a cached error.

        May be called from any thread, with or without the node's lock.
        The state and the value must change together: if they were written
        one after the other with no lock, another thread could re-evaluate
        the node in between and be left VALID with its value wiped. So the
        change is made under the node's lock when that is free. The lock is
        never waited for, because invalidation can run from a garbage
        collection callback while this thread holds other locks. If it is
        busy, only the state is flipped (one write, nothing to tear) and the
        stale value stays until the next evaluation replaces it.
        """
        if not self._condition.acquire(blocking=False):
            if self.has_result:
                self._state = NodeState.INVALID
            return
        try:
            if self.has_result:
                self._state = NodeState.INVALID
                self._value = NO_VALUE
                self._error = None
        finally:
            self._condition.release()

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
        # Per-object index, so releasing an object touches only its own
        # nodes: obj id -> keys of its nodes, and obj id -> the one weak
        # reference (shared by all of the object's nodes) whose callback
        # cleans them up. Both are guarded by _nodes_lock.
        self._object_nodes: Dict[int, Set[NodeKey]] = {}
        self._object_refs: Dict[int, weakref.ref] = {}
        self._nodes_lock = threading.RLock()
        self._thread_state = threading.local()
        self._layer_counter: int = 0
        self._layer_lock = threading.Lock()
        self._subscriptions: Dict[NodeKey, List[weakref.ref]] = {}
        self._subscriptions_lock = threading.RLock()
        self._pending_notifications: Set[NodeKey] = set()
        self._scenario_owner: Optional[int] = None
        self._scenario_depth: int = 0
        self._scenario_lock = threading.Lock()
        # Wait-for graph for cross-thread deadlock detection:
        # thread id -> key of the node that thread is blocked waiting on.
        self._threads_waiting: Dict[int, NodeKey] = {}
        self._waiting_lock = threading.Lock()

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

        # Install the output edge now, before the caller reads to_node's
        # value, rather than when the caller finishes. A change to to_node
        # made by another thread while the caller is still evaluating then
        # finds the caller and marks its result stale.
        if from_node.key not in to_node.outputs:
            with self._nodes_lock:
                to_node.outputs.add(from_node.key)

    def _finalize_dependencies(self, node: Node, new_inputs: Set[NodeKey]) -> None:
        """Make a node's input edges exactly those its evaluation discovered.

        The edges from the previous evaluation stay in place while the node
        recomputes, so an invalidation arriving mid-evaluation still reaches
        it; only here, once the new inputs are known, are the edges to inputs
        it no longer reads removed.
        """
        with self._nodes_lock:
            stale_inputs = node.inputs - new_inputs
            node.inputs = new_inputs
            for input_key in stale_inputs:
                input_node = self._nodes.get(input_key)
                if input_node is not None:
                    input_node.outputs.discard(node.key)
            for input_key in new_inputs:
                input_node = self._nodes.get(input_key)
                if input_node is not None:
                    input_node.outputs.add(node.key)

    def _settle(self, node: Node, started_at: int) -> None:
        """Finish an evaluation whose outcome was just stored on the node.

        If an invalidation from another thread reached the node while it was
        evaluating, the outcome was computed from an input that has since
        changed: it is discarded and the node left INVALID, so the next read
        recomputes. Storing the outcome first and checking the count second
        pairs with ``_mark_stale``, which bumps the count first and reads the
        state second, so one of the two always sees the other.

        Caller must hold ``node._condition``.
        """
        if node._invalidation_count != started_at:
            node.invalidate()
        node._evaluating_thread_id = None
        node._condition.notify_all()

    def _validate_dependency(self, from_node: Node, to_node: Node) -> None:
        """Ensure a runtime dependency was declared statically.

        Best-effort safety net, called only for same-object calls (cross-object
        calls cannot be resolved by the AST parser; see ``evaluate``). It is
        intentionally **name-based**: it confirms the callee's method name is
        among the names the parser detected being called on ``self``. A missing
        name raises so the author can declare the dependency; a coincidental
        match (e.g. a name seen only inside a ``self.x().Method()`` chain) is
        accepted. That looseness is safe: ``_record_dependency`` records the
        runtime edge regardless, so invalidation stays correct even when this
        check lets a call through. Tightening it to only direct ``self.X()``
        calls would risk false positives on chains whose receiver returns
        ``self``, for no correctness gain.
        """
        # None means the parser could not read from_node's source, so its
        # dependencies are unknown: track the call at runtime without rejecting.
        if from_node.static_deps is None:
            return
        if to_node.method_name in from_node.static_deps:
            return

        raise UntrackedError(
            f"Undeclared dependency from '{from_node.method_name}' to '{to_node.method_name}'. "
            "Wrap the call in dag.untracked(...) if this is intentional."
        )

    def _cleanup_object_nodes(
        self,
        obj_id: int,
        dead_ref: Optional[weakref.ref] = None,
    ) -> None:
        """Remove nodes for an object that has been garbage-collected.

        ``dead_ref`` is the weak reference whose callback triggered the
        cleanup. If it is no longer the one registered for ``obj_id`` (the
        manager was reset, or the id now belongs to another object), the
        call is stale and does nothing.
        """
        dead: Dict[NodeKey, Node] = {}
        stale_outputs: List[Node] = []
        thread_id = threading.get_ident()
        exec_state = self._get_execution_state()
        # Dependents this thread is evaluating, with the dead nodes each one
        # read. Only possible when the object died inside an evaluation.
        orphaned: Dict[NodeKey, Tuple[Node, List[Node]]] = {}

        with self._nodes_lock:
            if dead_ref is not None and self._object_refs.get(obj_id) is not dead_ref:
                return
            self._object_refs.pop(obj_id, None)
            # Unregister every node first, so the lookups below find only
            # surviving nodes and the dead nodes' own edges stay intact.
            for key in self._object_nodes.pop(obj_id, ()):
                node = self._nodes.pop(key, None)
                if node is not None:
                    dead[key] = node

            for node in dead.values():
                for input_key in node.inputs:
                    input_node = self._nodes.get(input_key)
                    if input_node is not None:
                        input_node.outputs.discard(node.key)

                for output_key in node.outputs:
                    output_node = self._nodes.get(output_key)
                    if output_node is None:
                        continue
                    output_node.inputs.discard(node.key)
                    stale_outputs.append(output_node)
                    if exec_state.eval_stack and self._is_evaluating_here(output_node, thread_id):
                        orphaned.setdefault(output_key, (output_node, []))[1].append(node)

            # A temporary model created and released inside an evaluation:
            # the evaluating node keeps its result, so it takes over what the
            # dead nodes depended on, or it would never hear of a change.
            for target, consumed in orphaned.values():
                donors: Dict[NodeKey, Node] = {}
                pending = list(consumed)
                while pending:
                    node = pending.pop()
                    if node.key in donors:
                        continue
                    donors[node.key] = node
                    pending.extend(
                        dead[input_key] for input_key in node.inputs if input_key in dead
                    )
                self._adopt_inputs(target, donors.values(), exec_state)

        with self._subscriptions_lock:
            for key in dead:
                self._subscriptions.pop(key, None)

        for output_node in stale_outputs:
            self.invalidate_node(output_node)

    def get_or_create_node(
        self,
        obj: Model,
        method_name: str,
        func: Callable,
        flags: int = Flags.NONE,
        static_deps: Optional[FrozenSet[str]] = frozenset(),
        args: Tuple[Any, ...] = (),
    ) -> Node:
        """Get an existing node or create a new one."""
        key = NodeKey(id(obj), method_name, args)
        with self._nodes_lock:
            node = self._nodes.get(key)
            if node is None:
                node = Node(
                    key=key,
                    obj_ref=self._object_ref(obj),
                    method_name=method_name,
                    func=func,
                    flags=flags,
                    static_deps=static_deps,
                )
                self._nodes[key] = node
                self._object_nodes.setdefault(key.obj_id, set()).add(key)

            return node

    def _object_ref(self, obj: Model) -> weakref.ref:
        """Return the weak reference shared by all of obj's nodes, creating
        it (with the cleanup callback) on first use.

        One reference per object means one callback when the object dies,
        rather than one per node. Caller must hold ``_nodes_lock``.
        """
        obj_id = id(obj)
        obj_ref = self._object_refs.get(obj_id)
        if obj_ref is None or obj_ref() is not obj:
            manager_ref = weakref.ref(self)

            def _on_object_gc(dead_ref: weakref.ref) -> None:
                manager = manager_ref()
                if manager is not None:
                    manager._cleanup_object_nodes(obj_id, dead_ref)

            obj_ref = weakref.ref(obj, _on_object_gc)
            self._object_refs[obj_id] = obj_ref
        return obj_ref

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
        self._check_scenario_owner()
        state = self._get_execution_state()

        # Track runtime dependency: if we're evaluating another node,
        # this node becomes a dependency of that node
        caller = state.evaluating_node
        if caller is not None and caller.key != node.key:
            if not self._is_untracked_for(caller.key):
                # The parser only detects calls rooted at `self`, so the
                # static-dependency check can only apply to same-object calls.
                # Cross-object calls (e.g. iterating a collection of other
                # models) are impossible to resolve statically; track them at
                # runtime so invalidation still propagates, but don't reject them.
                if caller.key.obj_id == node.key.obj_id:
                    self._validate_dependency(caller, node)
                self._record_dependency(caller, node)

        current_thread_id = threading.get_ident()
        cycle_error: Optional[CycleError] = None
        started_at = 0

        with node._condition:
            # Re-checked on every iteration: after waking from wait() the node
            # may have become valid/overridden, failed into ERROR, or revealed a
            # cycle, so all conditions are re-evaluated from the top.
            while True:
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
                    break

                # Another thread is computing this node: release the lock and
                # wait, then loop back to re-check the conditions above.
                if (
                    node.state == NodeState.EVALUATING
                    and node._evaluating_thread_id != current_thread_id
                ):
                    # Registering the wait detects wait-for cycles (thread A
                    # waiting on a node evaluated by thread B that is waiting
                    # on a node evaluated by A) and raises CycleError instead
                    # of deadlocking.
                    self._register_wait(current_thread_id, node)
                    try:
                        node._condition.wait()
                    finally:
                        self._unregister_wait(current_thread_id)
                    continue

                node._evaluating_thread_id = current_thread_id
                node._state = NodeState.EVALUATING
                # Read after the state is published: an invalidation that
                # arrives from here on is seen by _settle.
                started_at = node._invalidation_count
                break

        if cycle_error is not None:
            raise cycle_error

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
                self._settle(node, started_at)
            return result

        except Exception as e:
            with node._condition:
                # Optional covers failed calculations. An undeclared
                # dependency or a cycle is a mistake in the model, here or in
                # anything this node called, and has to surface.
                if node.flags & Flags.Optional and not self._is_model_error(e):
                    node.set_valid(NO_VALUE)
                    self._settle(node, started_at)
                    return NO_VALUE

                node.set_error(e)
                self._settle(node, started_at)

            if isinstance(e, UntrackedError):
                raise

            raise EvaluationError(node.method_name, e) from e

        finally:
            discovered_inputs = state.dependency_stack.pop()
            self._finalize_dependencies(node, discovered_inputs)
            state.eval_stack.pop()
            state.evaluating_node = previous_evaluating

    @staticmethod
    def _is_model_error(error: BaseException) -> bool:
        """Whether error reports a mistake in the model rather than a failed
        calculation: an UntrackedError or CycleError, however many
        EvaluationErrors it was wrapped in on the way up."""
        while isinstance(error, EvaluationError):
            error = error.original_error
        return isinstance(error, (UntrackedError, CycleError))

    def invalidate_node(self, node: Node, propagate_only: bool = False) -> None:
        """
        Invalidate a node and propagate to dependents.

        When a node is invalidated, all nodes that depend on it must also be
        invalidated. Traversal is iterative (an explicit stack) so that very
        deep dependency chains do not exceed Python's recursion limit.

        Args:
            node: The node to invalidate
            propagate_only: If True, only propagate to dependents without
                           invalidating this node (used when set changes)
        """
        if propagate_only:
            self._queue_subscription(node.key)
            self._walk_invalidation(node, reset_root=False, always_propagate=True)
        else:
            self._walk_invalidation(node, reset_root=True, always_propagate=False)

    def _walk_invalidation(
        self,
        root: Node,
        *,
        reset_root: bool,
        always_propagate: bool,
    ) -> None:
        """Walk the dependents of root, invalidating them.

        Args:
            root: Where the change happened.
            reset_root: Invalidate root itself, if it holds a result.
            always_propagate: Visit root's dependents even when root was not
                reset. Needed when root's visible value changed without its
                state changing (a set value or an override served from an
                INVALID node).

        A walk started from inside an evaluation on this thread (an on-graph
        scenario applying or reverting an override) can invalidate nodes
        that an evaluating node has already consumed. That node will finish
        VALID above INVALID inputs, and an INVALID node stops later walks, so
        the evaluating node would never hear about changes further up. To
        prevent that, the walk records which nodes it reset on the way to
        each evaluating node, and the evaluating node inherits their inputs
        as direct dependencies (see ``_inherit_inputs``).
        """
        thread_id = threading.get_ident()
        exec_state = self._get_execution_state()
        # Only then can the walk reach a node this thread is evaluating.
        tracking = bool(exec_state.eval_stack)
        # Nodes passed through while another thread evaluates them. They do
        # not change state, so unlike invalidated nodes they need an explicit
        # visited set to keep a cyclic edge from looping the walk forever.
        in_flight: Set[NodeKey] = set()
        # Used only when tracking: nodes this walk reset, the walk's edges
        # in reverse (child -> parents), and the evaluating nodes it reached.
        reset: Dict[NodeKey, Node] = {}
        parents: Dict[NodeKey, List[NodeKey]] = {}
        reached: Dict[NodeKey, Node] = {}
        stack: List[NodeKey] = []

        def visit_dependents(node: Node, was_reset: bool) -> None:
            outputs = list(node.outputs)
            if tracking and was_reset:
                reset[node.key] = node
                for output_key in outputs:
                    parents.setdefault(output_key, []).append(node.key)
            stack.extend(outputs)

        root_reset = (
            reset_root
            and not self._is_evaluating_here(root, thread_id)
            and self._mark_stale(root, thread_id, in_flight)
        )
        if not (root_reset or always_propagate):
            return
        visit_dependents(root, root_reset)

        while stack:
            output_node = self.get_node(stack.pop())
            if output_node is None:
                continue
            if tracking and self._is_evaluating_here(output_node, thread_id):
                reached[output_node.key] = output_node
                continue
            if self._mark_stale(output_node, thread_id, in_flight):
                visit_dependents(output_node, True)

        for target in reached.values():
            self._inherit_inputs(target, reset, parents, exec_state)

    @staticmethod
    def _is_evaluating_here(node: Node, thread_id: int) -> bool:
        """Whether node is being evaluated by the given (current) thread."""
        return (
            node._state is NodeState.EVALUATING
            and node._evaluating_thread_id == thread_id
        )

    def _inherit_inputs(
        self,
        target: Node,
        reset: Dict[NodeKey, Node],
        parents: Dict[NodeKey, List[NodeKey]],
        exec_state: _ExecutionState,
    ) -> None:
        """Make target, a node this thread is evaluating, depend directly on
        the inputs of every node a walk reset on its way to target.

        Those nodes are what target consumed and has now lost: they are
        INVALID, so they will not forward later invalidations, and when they
        are next evaluated (outside the scenario, possibly along a different
        code path) their own input edges may change. Their inputs as of now
        are exactly what target's result depends on.
        """
        # Reset nodes from which the walk reached target.
        upstream: Dict[NodeKey, Node] = {}
        pending = list(parents.get(target.key, ()))
        while pending:
            key = pending.pop()
            node = reset.get(key)
            if node is None or key in upstream:
                continue
            upstream[key] = node
            pending.extend(parents.get(key, ()))

        # Inputs that were themselves reset are adopted too: a settable or
        # overridable one can still change directly, and that walk starts
        # from its own dependents.
        self._adopt_inputs(target, upstream.values(), exec_state)

    def _adopt_inputs(
        self,
        target: Node,
        donors: Iterable[Node],
        exec_state: _ExecutionState,
    ) -> None:
        """Add the inputs of donors to the dependencies of target, a node
        this thread is evaluating, installing the edges immediately."""
        try:
            depth = exec_state.eval_stack.index(target.key)
        except ValueError:
            return
        discovered = exec_state.dependency_stack[depth]

        with self._nodes_lock:
            for node in donors:
                for input_key in node.inputs:
                    input_node = self._nodes.get(input_key)
                    if input_node is None or input_key == target.key:
                        continue
                    discovered.add(input_key)
                    input_node.outputs.add(target.key)

    def _mark_stale(self, node: Node, thread_id: int, in_flight: Set[NodeKey]) -> bool:
        """Apply an invalidation to one node; return whether its dependents
        must be visited too.

        - INVALID: nothing to do. Its dependents were invalidated when it
          became INVALID.
        - VALID or ERROR: reset to INVALID. ERROR is included because a
          dependent that caught the error is VALID with a fallback and must
          recompute once the error may clear.
        - EVALUATING on this thread: the change comes from inside the node's
          own evaluation (an on-graph scenario, or the release of a temporary
          model), and the evaluation already accounts for it. Left alone.
        - EVALUATING on another thread: that evaluation may have read the
          old input. The count is bumped so ``_settle`` discards its result,
          and the walk continues: the callers waiting for this result up
          that thread's stack are EVALUATING dependents and get the same
          treatment.

        Never waits for a lock. Correctness against a concurrent
        ``_settle`` relies on ordering: the count is bumped before the state
        is read.
        """
        if self._is_evaluating_here(node, thread_id):
            return False

        node._invalidation_count += 1
        state = node._state
        if state is NodeState.INVALID:
            return False
        if state is NodeState.EVALUATING:
            if node.key in in_flight:
                return False
            in_flight.add(node.key)
        else:
            node.invalidate()
        self._queue_subscription(node.key)
        return True

    def invalidate_dependents(self, node: Node, reset: bool = False) -> None:
        """
        Invalidate the dependents of a node, whatever the node's own state.

        Used when a node's visible value changes through a set value or an
        override: the node may be INVALID (it serves the set value without
        being evaluated), so its state says nothing about its dependents.

        Args:
            node: The node whose value changed.
            reset: Also discard the node's own cached result, for a change
                that reverts it to its computed value.
        """
        self._walk_invalidation(node, reset_root=reset, always_propagate=True)

    def set_node_value(self, node: Node, value: Any) -> None:
        """Permanently set a node's value, invalidating dependents and
        notifying the node's own watchers.

        Single choke point for every permanent-write path (``accessor.set``
        and ``NodeChange.apply``), so their invalidation and notification
        semantics cannot drift apart.
        """
        self._check_scenario_owner()
        node._set_value = value
        self.invalidate_dependents(node)
        self._queue_subscription(node.key)

    def clear_node_value(self, node: Node) -> None:
        """Clear a node's permanent set value, reverting it to its computed
        value.

        Dependents are invalidated explicitly rather than via
        ``invalidate_node``: a node that was set before ever being evaluated
        is still INVALID (``evaluate`` serves the set value without validating
        the node), so ``invalidate_node``'s already-invalid short-circuit
        would skip dependents that consumed the set value.
        """
        self._check_scenario_owner()
        node._set_value = NO_VALUE
        self.invalidate_dependents(node, reset=True)
        self._queue_subscription(node.key)

    def add_dependency(self, from_node: Node, to_node: Node) -> None:
        """Add a dependency edge: from_node depends on to_node."""
        from_node.inputs.add(to_node.key)
        to_node.outputs.add(from_node.key)

    def remove_dependency(self, from_node: Node, to_node: Node) -> None:
        """Remove a dependency edge."""
        from_node.inputs.discard(to_node.key)
        to_node.outputs.discard(from_node.key)

    def _claim_scenario_ownership(self) -> None:
        """Claim scenario/branch ownership for the current thread (fail-fast on conflict)."""
        tid = threading.get_ident()
        with self._scenario_lock:
            if self._scenario_owner is None:
                self._scenario_owner = tid
                self._scenario_depth = 1
            elif self._scenario_owner == tid:
                self._scenario_depth += 1
            else:
                raise ConcurrentScenarioError(self._scenario_owner, tid)

    def _release_scenario_ownership(self) -> None:
        """Release one level of scenario/branch ownership for the current thread."""
        with self._scenario_lock:
            if self._scenario_owner != threading.get_ident():
                return  # only the owning thread may release; ignore foreign/no-op calls
            if self._scenario_depth > 0:
                self._scenario_depth -= 1
                if self._scenario_depth == 0:
                    self._scenario_owner = None

    def _register_wait(self, thread_id: int, node: Node) -> None:
        """Record that thread_id is about to block waiting for node.

        Walks the wait-for chain (node's evaluator -> the node that thread
        waits on -> its evaluator -> ...) under one lock; check and
        registration are atomic, so for any two threads about to deadlock,
        whichever registers second sees the other and raises CycleError.
        """
        with self._waiting_lock:
            evaluator = node._evaluating_thread_id
            while evaluator is not None:
                if evaluator == thread_id:
                    raise CycleError(
                        f"Cross-thread cyclic dependency detected while "
                        f"waiting for '{node.method_name}': the threads "
                        "evaluating these nodes are waiting on each other."
                    )
                waited_key = self._threads_waiting.get(evaluator)
                if waited_key is None:
                    break
                waited_node = self.get_node(waited_key)
                evaluator = (
                    waited_node._evaluating_thread_id
                    if waited_node is not None
                    else None
                )
            self._threads_waiting[thread_id] = node.key

    def _unregister_wait(self, thread_id: int) -> None:
        """Remove thread_id from the wait-for graph after it wakes."""
        with self._waiting_lock:
            self._threads_waiting.pop(thread_id, None)

    def _check_scenario_owner(self) -> None:
        """Reject DAG access from a thread other than the active scenario owner."""
        owner = self._scenario_owner  # lock-free single-attribute read (GIL-atomic)
        if owner is not None and owner != threading.get_ident():
            raise ConcurrentScenarioError(owner, threading.get_ident())

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

    @property
    def active_scenarios(self) -> Tuple[Scenario, ...]:
        """The current thread's active scenarios, outermost first."""
        return tuple(self._get_execution_state().context_stack)

    def next_layer_id(self) -> int:
        """Get the next branch ID."""
        with self._layer_lock:
            self._layer_counter += 1
            return self._layer_counter

    # Subscriptions
    def subscribe(self, node_key: NodeKey, callback: Callable) -> None:
        """Subscribe to notifications when a node is invalidated.

        Callbacks are weakly held: a callback dies with its owner. Bound
        methods are held via ``weakref.WeakMethod`` (a plain ``weakref.ref``
        would die immediately, since each ``obj.method`` access creates a
        transient bound-method object).
        """
        ref: weakref.ref
        if inspect.ismethod(callback):
            ref = weakref.WeakMethod(callback)
        else:
            ref = weakref.ref(callback)
        with self._subscriptions_lock:
            if node_key not in self._subscriptions:
                self._subscriptions[node_key] = []
            self._subscriptions[node_key].append(ref)

    def unsubscribe(self, node_key: NodeKey, callback: Callable) -> None:
        """Remove a previously subscribed callback (matched by equality, so a
        fresh bound-method object identifies its registration)."""
        with self._subscriptions_lock:
            refs = self._subscriptions.get(node_key)
            if not refs:
                return
            kept = [
                ref for ref in refs
                if ref() is not None and ref() != callback
            ]
            if kept:
                self._subscriptions[node_key] = kept
            else:
                self._subscriptions.pop(node_key, None)
                self._pending_notifications.discard(node_key)

    def _queue_subscription(self, node_key: NodeKey) -> None:
        """Enqueue a pending notification for a watched node (dispatched on flush)."""
        with self._subscriptions_lock:
            if node_key in self._subscriptions:
                self._pending_notifications.add(node_key)

    def flush(self) -> None:
        """Dispatch pending notifications (one per watched node that changed
        since the last flush) and clear the pending set."""
        with self._subscriptions_lock:
            pending = list(self._pending_notifications)
            self._pending_notifications.clear()

        for node_key in pending:
            node = self.get_node(node_key)
            if node is None:
                continue

            with self._subscriptions_lock:
                callbacks = list(self._subscriptions.get(node_key, []))

            for cb_ref in callbacks:
                cb = cb_ref()
                if cb is None:
                    continue
                # An earlier callback in this dispatch may have unwatched it.
                with self._subscriptions_lock:
                    if cb_ref not in self._subscriptions.get(node_key, ()):
                        continue
                try:
                    cb(node)
                except Exception:
                    # Watch callbacks must not interrupt the DAG, but their
                    # failures must be visible rather than silently dropped.
                    logger.exception(
                        "Watch callback for %r failed", node.method_name
                    )

            # Drop dead references from the list as it stands now. Writing
            # back the snapshot taken before dispatch would undo any watch or
            # unwatch a callback (or another thread) made in the meantime.
            with self._subscriptions_lock:
                current = self._subscriptions.get(node_key)
                if current is not None:
                    alive = [ref for ref in current if ref() is not None]
                    if alive:
                        self._subscriptions[node_key] = alive
                    else:
                        del self._subscriptions[node_key]


class Scenario:
    """
    A scenario for temporary overrides.

    Overrides made within a scenario are automatically reverted when
    the scenario exits.
    """

    def __init__(self):
        self._dag = DagManager.get_instance()
        self._tweaks: List[Tuple[Node, Any]] = []  # (node, old_override_value)
        # Hard references to every overridden object, keyed by id. Overrides
        # are stored under NodeKey, which carries id(obj): if the object died
        # mid-scenario, a new object allocated at the same address would
        # inherit its overrides. Released when the scenario exits.
        self._held_objects: Dict[int, Any] = {}
        self._layer_id = self._dag.next_layer_id()

    def __enter__(self) -> Scenario:
        self._dag._claim_scenario_ownership()
        self._dag.push_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        try:
            # Revert all overrides
            for node, old_value in reversed(self._tweaks):
                self._dag.set_tweak_value(node.key, old_value)
                # Always invalidate dependents when reverting an override:
                # they need to recompute with the original value. When the
                # node reverts to its computed value, its own cached result
                # is discarded too, as part of the same walk, so that a node
                # evaluating on this thread inherits its inputs.
                self._dag.invalidate_dependents(node, reset=old_value is NO_VALUE)
                # The reverted node changed too: notify its own watchers.
                self._dag._queue_subscription(node.key)
        finally:
            # The overrides are gone, so the objects no longer need pinning
            # and a later re-entry must not revert them a second time.
            self._tweaks.clear()
            self._held_objects.clear()
            self._dag.pop_context()
            self._dag._release_scenario_ownership()
        return False

    def add_tweak(self, node: Node, new_value: Any) -> None:
        """Record an override for later reversion."""
        self._dag._check_scenario_owner()
        obj = node.obj_ref()
        if obj is None:
            raise OverrideError(
                node.method_name,
                f"Cannot override '{node.method_name}': its object has been "
                "garbage collected",
            )
        self._held_objects[node.key.obj_id] = obj
        old_value = self._dag.get_tweak_value(node.key)
        self._tweaks.append((node, old_value))
        self._dag.set_tweak_value(node.key, new_value)
        # Must invalidate dependents even if this node is already invalid
        # (e.g., when node has a set_value but state is INVALID)
        self._dag.invalidate_dependents(node)
        # The overridden node itself changed: notify its own watchers,
        # mirroring how set() notifies the node it sets.
        self._dag._queue_subscription(node.key)

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
