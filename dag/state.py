"""
State management for the DAG framework.

This module provides:
- scenario() - Context manager for temporary overrides
- branch() - Branch management for parallel graph states
- apply_overrides() - Apply a set of overrides
- get_overrides() - Get current overrides in a scenario

The key distinction between set and override:
- set is permanent - you can't "take off" a set value
- override is temporary - reverts when scenario exits
- overrides can be nested arbitrarily
- overrides hold hard references to objects (prevents GC)
- override sets can be serialized independently
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Optional, Tuple

from .core import Scenario, DagManager, scenario as create_scenario
from .decorators import ComputedFunctionAccessor
from .exceptions import OverrideError, ScenarioError
from .flags import Overridable

if TYPE_CHECKING:
    from .model import Model


def _overridable_accessor(obj: Model, method_name: str) -> ComputedFunctionAccessor:
    """Resolve obj.method_name to an Overridable computed-function accessor.

    Shared by every name-based override path (Branch.override,
    OverrideSet.apply) so they enforce the same Overridable rule as
    ``accessor.override()`` and create nodes through the accessor (which
    carries the descriptor's parsed static dependencies).
    """
    accessor = getattr(obj, method_name, None)
    if not isinstance(accessor, ComputedFunctionAccessor):
        raise OverrideError(
            method_name, f"'{method_name}' is not a computed function"
        )
    if not (accessor._descriptor.flags & Overridable):
        raise OverrideError(method_name)
    return accessor


@dataclass
class Override:
    """
    Represents a single override (temporary value override).

    Overrides hold hard references to the object to prevent garbage collection,
    which would cause the object to be reloaded from the database without
    the override applied.
    """
    obj: Model                          # Hard reference to the object
    method_name: str                    # Name of the computed function
    value: Any                          # The overridden value
    args: Tuple[Any, ...] = ()          # Arguments (for parameterized functions)


@dataclass
class OverrideSet:
    """
    A collection of overrides that can be serialized and reapplied.

    Useful for:
    - Distributing computations with scenario data
    - Expressing your scenario as a change to the base state
    """
    overrides: List[Override] = field(default_factory=list)

    def add(self, obj: Model, method_name: str, value: Any, args: Tuple = ()) -> None:
        """Add an override to this set."""
        self.overrides.append(Override(obj=obj, method_name=method_name, value=value, args=args))

    def apply(self, ctx: Scenario) -> None:
        """Apply all overrides in this set to the given scenario.

        Every target must be a computed function with the Overridable flag —
        the same rule ``accessor.override()`` enforces.
        """
        for override in self.overrides:
            accessor = _overridable_accessor(override.obj, override.method_name)
            ctx.add_tweak(accessor._get_or_create_node(override.args), override.value)


class Branch:
    """
    A branch provides parallel graph states.

    Unlike scenarios which are transient, branches persist and allow
    multiple simultaneous views of the graph.

    Branches share nodes with their parent where possible (copy-on-write),
    making them memory-efficient.
    """

    def __init__(self, parent: Optional[Branch] = None):
        self._dag = DagManager.get_instance()
        self._branch_id = self._dag.next_layer_id()
        self._parent = parent
        self._overrides: Dict[Tuple[int, str, Tuple[Any, ...]], Override] = {}
        # Stack of active scenarios: branches are re-entrant, including
        # nested re-entry while already active (with b: ... with b: ...).
        self._scenarios: List[BranchScenario] = []

    def __enter__(self) -> Branch:
        # Create a scenario for this entry and replay persisted overrides.
        scenario = BranchScenario(self)
        scenario.__enter__()
        self._scenarios.append(scenario)
        self._apply_persisted_overrides(scenario)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if not self._scenarios:
            raise ScenarioError("Branch exited without a matching entry")
        self._scenarios.pop().__exit__(exc_type, exc_val, exc_tb)
        return False

    def override(self, obj: Model, method_name: str, value: Any, args: Tuple = ()) -> None:
        """Add an override to this branch.

        The target must be a computed function with the Overridable flag,
        mirroring ``accessor.override()``. Overrides recorded before the
        branch is entered are applied on entry.
        """
        accessor = _overridable_accessor(obj, method_name)
        self._overrides[(id(obj), method_name, args)] = Override(
            obj=obj, method_name=method_name, value=value, args=args
        )

        # Also apply immediately when this branch's scenario is active.
        if self._scenarios and self._dag.current_context is self._scenarios[-1]:
            self._scenarios[-1].add_tweak(accessor._get_or_create_node(args), value)

    def _remember_override(self, node, value: Any) -> None:
        """Persist an override made while this branch is active."""
        obj = node.obj_ref()
        if obj is None:
            return

        self._overrides[(id(obj), node.method_name, node.key.args)] = Override(
            obj=obj,
            method_name=node.method_name,
            value=value,
            args=node.key.args,
        )

    def _iter_overrides(self) -> Generator[Override, None, None]:
        """Yield overrides inherited from parent branches first."""
        if self._parent is not None:
            yield from self._parent._iter_overrides()
        yield from self._overrides.values()

    def _apply_persisted_overrides(self, ctx: Scenario) -> None:
        """Replay the branch's persisted overrides into the active scenario.

        Flag validation happened when each override was recorded, so the
        replay is lenient: entries that no longer resolve to a computed
        function are skipped. Nodes are resolved through the accessor so they
        are created with the descriptor's parsed static dependencies.
        """
        for override in self._iter_overrides():
            accessor = getattr(override.obj, override.method_name, None)
            if isinstance(accessor, ComputedFunctionAccessor):
                ctx.add_tweak(accessor._get_or_create_node(override.args), override.value)

    @property
    def branch_id(self) -> int:
        return self._branch_id


@contextmanager
def branch() -> Generator[Branch, None, None]:
    """
    Create a new branch context.

    Branches allow multiple parallel states of the graph to exist.
    Unlike scenarios, branches can be nested and share nodes efficiently.

    Usage:
        with dag.branch() as b1:
            o.Strike.override(1.4)
            with b1: print(o.Price())  # uses overridden Strike

        with dag.branch() as b2:
            o.Strike.override(1.5)
            with b2: print(o.Price())  # uses different overridden Strike
    """
    b = Branch()
    with b:
        yield b


class BranchScenario(Scenario):
    """Scenario implementation that persists overrides onto a branch."""

    def __init__(self, branch: Branch):
        super().__init__()
        self._branch = branch

    def add_tweak(self, node, new_value: Any) -> None:
        self._branch._remember_override(node, new_value)
        super().add_tweak(node, new_value)


def get_overrides() -> OverrideSet:
    """
    Get the current overrides as an OverrideSet.

    Useful for serializing scenario state for distributed computation.
    """
    dag = DagManager.get_instance()
    ctx = dag.current_context

    override_set = OverrideSet()

    if ctx is None:
        return override_set

    # A node overridden several times appears once with its current value.
    unique_nodes = {node.key: node for node, _old_value in ctx._tweaks}
    for node in unique_nodes.values():
        obj = node.obj_ref()
        if obj is not None:
            override_set.add(
                obj=obj,
                method_name=node.method_name,
                value=dag.get_tweak_value(node.key),
                args=node.key.args,
            )

    return override_set


def apply_overrides(override_set: OverrideSet) -> Scenario:
    """
    Apply an OverrideSet within a new scenario.

    Returns the scenario for use with 'with' statement:
        with dag.apply_overrides(overrides):
            result = obj.Price()
    """
    ctx = create_scenario()
    override_set.apply(ctx)
    return ctx

def untracked(func: Callable[[], Any]) -> Any:
    """
    Execute a function without strict dependency checking.

    The DAG normally throws an exception if you call a computed function
    that wasn't detected at parse time. Use untracked() to suppress
    this check when you're sure you don't need the dependency tracked.

    Warning: Use sparingly! Missing dependencies can cause stale cache issues.

    Usage:
        result = dag.untracked(lambda: self.SomeMethod())
    """
    dag = DagManager.get_instance()
    caller_key = dag.enter_untracked()
    try:
        return func()
    finally:
        dag.exit_untracked(caller_key)
