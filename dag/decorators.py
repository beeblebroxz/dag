"""
Computed function decorator (@computed).

The @computed decorator transforms a method into a computed function that:
- Has dependencies detected at parse time (AST analysis)
- Caches computed values
- Invalidates dependents when changed
- Supports set and override operations

Usage:
    @dag.computed
    def Price(self):
        return self.Spot() - self.Strike()

    @dag.computed(dag.Input)
    def Strike(self):
        return 1.0  # default value

    @dag.computed(dag.Overridable)
    def Spot(self):
        return self.PairObject().Spot()

    @dag.computed(inverse=spotChange)
    def Spot(self):
        return self.FwdCurve()[self.SpotDate()]
"""

from __future__ import annotations

import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    FrozenSet,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from .core import DagManager, Node
from .exceptions import SetValueError, OverrideError
from .flags import Input, Overridable, Flags
from .parser import Dependency, DependencyParseResult, parse_dependency_result

if TYPE_CHECKING:
    from .model import Model


F = TypeVar('F', bound=Callable[..., Any])


class ComputedFunctionDescriptor:
    """
    Descriptor that wraps a computed function method.

    When accessed as an attribute, returns a ComputedFunctionAccessor
    that provides the computed function interface (.set, .override, etc.)

    When called, evaluates the computed function.
    """

    def __init__(
        self,
        func: Callable,
        flags: int = Flags.NONE,
        inverse: Optional[Callable] = None,
        static_deps: Optional[FrozenSet[str]] = frozenset(),
        dependency_parse_result: Optional[DependencyParseResult] = None,
    ):
        self.func = func
        self.flags = flags
        self.inverse = inverse
        self.static_deps = static_deps
        self.dependency_parse_result = dependency_parse_result
        self.dependency_paths: Tuple[Dependency, ...] = (
            dependency_parse_result.dependencies
            if dependency_parse_result is not None and dependency_parse_result.succeeded
            else ()
        )
        self.name = func.__name__
        try:
            # Used to validate the args passed to set/override/watch against
            # the function's parameters. None when unavailable (validation is
            # then skipped, mirroring how unparseable source is handled).
            self.signature: Optional[inspect.Signature] = inspect.signature(func)
        except (ValueError, TypeError):
            self.signature = None

        # Copy function metadata
        functools.update_wrapper(cast(Any, self), func)

    def __get__(self, obj: Optional[Model], objtype=None) -> Union[ComputedFunctionDescriptor, ComputedFunctionAccessor]:
        if obj is None:
            # Accessed on class, return the descriptor
            return self
        # Accessed on instance, return an accessor
        return ComputedFunctionAccessor(obj, self)

    def __set__(self, obj: Model, value: Any) -> None:
        """
        Allow setting via assignment: obj.Strike = 1.4

        This is syntactic sugar for obj.Strike.set(1.4)
        """
        if not (self.flags & Input):
            raise SetValueError(self.name)

        accessor = ComputedFunctionAccessor(obj, self)
        accessor.set(value)


class ComputedFunctionAccessor:
    """
    Accessor for a computed function on a specific object instance.

    Provides:
    - __call__() - evaluate the computed function
    - set() - permanently set the value (if Input)
    - override() - temporarily override (if Overridable)
    - watch() - register a callback for invalidation
    """

    def __init__(self, obj: Model, descriptor: ComputedFunctionDescriptor):
        self._obj = obj
        self._descriptor = descriptor
        self._dag = DagManager.get_instance()

    def __call__(self, *args, **kwargs) -> Any:
        """Evaluate the computed function."""
        # Handle keyword arguments by converting to positional
        # (For simplicity, we don't support kwargs in caching key)
        if kwargs:
            raise ValueError("Computed functions do not support keyword arguments for caching")

        # Get or create the node
        node = self._dag.get_or_create_node(
            obj=self._obj,
            method_name=self._descriptor.name,
            func=self._descriptor.func,
            flags=self._descriptor.flags,
            static_deps=self._descriptor.static_deps,
            args=args,
        )

        # Evaluate
        return self._dag.evaluate(node, args)

    def set(self, value: Any, *args: Any) -> None:
        """
        Permanently set the value of this computed function.

        The computed function must have the Input flag. For parameterized
        computed functions, pass the same arguments used to call the
        function: ``obj.Rate.set(0.10, '1Y')`` targets the
        ``obj.Rate('1Y')`` node.
        """
        if not (self._descriptor.flags & Input):
            raise SetValueError(self._descriptor.name)
        self._validate_args(args)

        # Handle inverse if configured
        if self._descriptor.inverse is not None:
            if args:
                raise TypeError(
                    f"'{self._descriptor.name}' routes set() through an inverse "
                    "handler and does not support parameterized set"
                )
            changes = self._descriptor.inverse(self._obj, value)
            # Apply the NodeChange operations returned by the inverse
            self._apply_inverse_changes(changes)
            return

        node = self._get_or_create_node(args)
        self._dag.set_node_value(node, value)

    def override(self, value: Any, *args: Any) -> None:
        """
        Temporarily override the value of this computed function.

        The computed function must have the Overridable flag.
        Must be called within a dag.scenario(). For parameterized computed
        functions, pass the same arguments used to call the function:
        ``obj.Rate.override(0.10, '1Y')`` targets the ``obj.Rate('1Y')`` node.
        """
        if not (self._descriptor.flags & Overridable):
            raise OverrideError(self._descriptor.name)
        self._validate_args(args)

        ctx = self._dag.current_context
        if ctx is None:
            raise OverrideError(
                self._descriptor.name,
                "override must be called within a dag.scenario()"
            )

        # Inverse handlers redirect the change to other nodes (mutual
        # dependencies). Apply those as temporary tweaks so they revert with
        # the scenario, mirroring how set() routes through the inverse.
        if self._descriptor.inverse is not None:
            if args:
                raise TypeError(
                    f"'{self._descriptor.name}' routes override() through an "
                    "inverse handler and does not support parameterized override"
                )
            changes = self._descriptor.inverse(self._obj, value)
            self._apply_inverse_overrides(changes, ctx)
            return

        node = self._get_or_create_node(args)
        ctx.add_tweak(node, value)

    def watch(self, callback: Callable[[Node], None], *args: Any) -> None:
        """
        Watch for notifications when this computed function is invalidated.

        The callback receives the Node object and is called when
        dag.flush() is invoked after the node
        transitions from valid to invalid. For parameterized computed
        functions, pass the invocation arguments to watch that node:
        ``obj.Rate.watch(callback, '1Y')``.
        """
        self._validate_args(args)
        node = self._get_or_create_node(args)
        self._dag.subscribe(node.key, callback)

    def unwatch(self, callback: Callable[[Node], None], *args: Any) -> None:
        """Remove a callback registered with watch() for the same args."""
        self._validate_args(args)
        node = self._get_or_create_node(args)
        self._dag.unsubscribe(node.key, callback)

    def invalidate(self, *args: Any) -> None:
        """Force this computed function to recompute on next evaluation.

        Invalidates the node and its dependents and queues watch
        notifications. This is the explicit refresh hook for cells whose
        body is not perfectly pure — e.g. re-reading external data, or
        retrying an Optional cell that cached NO_VALUE after a transient
        failure.
        """
        self._validate_args(args)
        self._dag._check_scenario_owner()
        node = self._get_or_create_node(args)
        self._dag.invalidate_node(node)

    def clearValue(self, *args: Any) -> None:
        """Clear any set value, reverting to computed value."""
        if not (self._descriptor.flags & Input):
            raise SetValueError(self._descriptor.name)
        self._validate_args(args)

        node = self._get_or_create_node(args)
        self._dag.clear_node_value(node)

    def clear_value(self, *args: Any) -> None:
        """Pythonic alias for clearValue()."""
        self.clearValue(*args)

    def _validate_args(self, args: Tuple[Any, ...]) -> None:
        """Reject argument lists no invocation of this function could produce.

        Without this, set/override/watch on a parameterized cell would
        silently target the ``()``-node, which no call ever reads.
        """
        signature = self._descriptor.signature
        if signature is None:
            return
        try:
            signature.bind(self._obj, *args)
        except TypeError as exc:
            raise TypeError(
                f"Arguments {args!r} do not match the signature of "
                f"'{self._descriptor.name}': {exc}. Pass the same arguments "
                "used when calling the computed function, e.g. "
                f"obj.{self._descriptor.name}.set(value, *args)."
            ) from None

    def _get_or_create_node(self, args: Tuple = ()) -> Node:
        """Get or create the node for this computed function."""
        return self._dag.get_or_create_node(
            obj=self._obj,
            method_name=self._descriptor.name,
            func=self._descriptor.func,
            flags=self._descriptor.flags,
            static_deps=self._descriptor.static_deps,
            args=args,
        )

    def _apply_inverse_changes(self, changes: Any) -> None:
        """Apply changes from an inverse handler permanently (the set() path).

        Accepts NodeChange instances (or anything with .apply()) and
        (node_getter, value) tuples. Anything else raises rather than being
        silently dropped.
        """
        for change in self._iter_inverse_changes(changes, SetValueError):
            if hasattr(change, 'apply'):
                change.apply()
            else:
                node_getter, value = change[0], change[1]
                node_getter().set(value)

    def _apply_inverse_overrides(self, changes: Any, ctx: Any) -> None:
        """Apply changes from an inverse handler as temporary scenario tweaks
        (the override counterpart of _apply_inverse_changes; accepts the same
        change formats)."""
        for change in self._iter_inverse_changes(changes, OverrideError):
            if isinstance(change, NodeChange):
                node = change.node_accessor._get_or_create_node()
                ctx.add_tweak(node, change.value)
            elif hasattr(change, 'apply'):
                raise OverrideError(
                    self._descriptor.name,
                    f"Inverse handler for '{self._descriptor.name}' returned "
                    f"{change!r}, which cannot be applied as a temporary "
                    "override (only NodeChange and (node_getter, value) "
                    "tuples can)."
                )
            else:
                node_getter, value = change[0], change[1]
                node = node_getter()._get_or_create_node()
                ctx.add_tweak(node, value)

    def _iter_inverse_changes(self, changes: Any, error_cls: type) -> list:
        """Normalize an inverse handler's return value to a list of supported
        changes, raising error_cls on unsupported entries."""
        if changes is None:
            return []

        if not isinstance(changes, (list, tuple)):
            changes = [changes]

        for change in changes:
            is_tuple_change = (
                isinstance(change, tuple)
                and len(change) >= 2
                and callable(change[0])
            )
            if not is_tuple_change and not hasattr(change, 'apply'):
                raise error_cls(
                    self._descriptor.name,
                    f"Inverse handler for '{self._descriptor.name}' returned "
                    f"an unsupported change: {change!r}"
                )

        return list(changes)

    @property
    def _node(self) -> Optional[Node]:
        """Get the node if it exists."""
        from .core import NodeKey
        key = NodeKey(id(self._obj), self._descriptor.name, ())
        return self._dag.get_node(key)


def computed(
    func_or_flags: Union[Callable, int, None] = None,
    inverse: Optional[Callable] = None,
) -> Union[ComputedFunctionDescriptor, Callable[[F], ComputedFunctionDescriptor]]:
    """
    Decorator for computed functions.

    Usage:
        @computed
        def Price(self): ...

        @computed(Input)
        def Strike(self): ...

        @computed(Overridable, inverse=handler)
        def Spot(self): ...
    """
    if func_or_flags is None:
        # @computed() with no arguments
        def wrap_without_args(func: F) -> ComputedFunctionDescriptor:
            parse_result = parse_dependency_result(func)
            return ComputedFunctionDescriptor(
                func=func,
                flags=Flags.NONE,
                inverse=inverse,
                static_deps=parse_result.names if parse_result.succeeded else None,
                dependency_parse_result=parse_result,
            )
        return wrap_without_args

    if callable(func_or_flags):
        # @computed without parentheses
        func = func_or_flags
        parse_result = parse_dependency_result(func)
        return ComputedFunctionDescriptor(
            func=func,
            flags=Flags.NONE,
            inverse=None,
            static_deps=parse_result.names if parse_result.succeeded else None,
            dependency_parse_result=parse_result,
        )

    # @computed(flags) or @computed(flags, inverse=...)
    flags = func_or_flags

    def wrap_with_flags(func: F) -> ComputedFunctionDescriptor:
        parse_result = parse_dependency_result(func)
        return ComputedFunctionDescriptor(
            func=func,
            flags=flags,
            inverse=inverse,
            static_deps=parse_result.names if parse_result.succeeded else None,
            dependency_parse_result=parse_result,
        )

    return wrap_with_flags


class NodeChange:
    """
    Represents a change to be applied to a node.

    Used by inverse handlers to express mutual dependencies.
    """

    def __init__(self, node_accessor: ComputedFunctionAccessor, value: Any):
        self.node_accessor = node_accessor
        self.value = value

    def apply(self) -> None:
        """Apply this change permanently, with the same checks and
        invalidation/notification semantics as set()."""
        descriptor = self.node_accessor._descriptor
        if not (descriptor.flags & Input):
            raise SetValueError(descriptor.name)
        node = self.node_accessor._get_or_create_node()
        self.node_accessor._dag.set_node_value(node, self.value)
