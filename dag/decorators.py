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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    FrozenSet,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from .core import DagManager, Node
from .exceptions import UntrackedError, SetValueError, OverrideError
from .flags import NO_VALUE, Input, Overridable, Flags
from .parser import parse_dependencies

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
        static_deps: FrozenSet[str] = frozenset(),
    ):
        self.func = func
        self.flags = flags
        self.inverse = inverse
        self.static_deps = static_deps
        self.name = func.__name__

        # Copy function metadata
        functools.update_wrapper(self, func)

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

    def set(self, value: Any) -> None:
        """
        Permanently set the value of this computed function.

        The computed function must have the Input flag.
        """
        if not (self._descriptor.flags & Input):
            raise SetValueError(self._descriptor.name)

        node = self._get_or_create_node()

        # Handle inverse if configured
        if self._descriptor.inverse is not None:
            changes = self._descriptor.inverse(self._obj, value)
            # Apply the NodeChange operations returned by the inverse
            self._apply_inverse_changes(changes)
            return

        # Direct set
        old_value = node._set_value
        node._set_value = value
        # Invalidate dependents (not this node, since it now has a set value)
        self._dag.invalidate_dependents(node)

    def override(self, value: Any) -> None:
        """
        Temporarily override the value of this computed function.

        The computed function must have the Overridable flag.
        Must be called within a dag.scenario().
        """
        if not (self._descriptor.flags & Overridable):
            raise OverrideError(self._descriptor.name)

        ctx = self._dag.current_context
        if ctx is None:
            raise OverrideError(
                self._descriptor.name,
                f"override must be called within a dag.scenario()"
            )

        node = self._get_or_create_node()
        ctx.add_tweak(node, value)

    def watch(self, callback: Callable[[Node], None]) -> None:
        """
        Watch for notifications when this computed function is invalidated.

        The callback receives the Node object and is called when
        dag.flush() is invoked after the node
        transitions from valid to invalid.
        """
        node = self._get_or_create_node()
        self._dag.subscribe(node.key, callback)

    def clearValue(self) -> None:
        """Clear any set value, reverting to computed value."""
        if not (self._descriptor.flags & Input):
            raise SetValueError(self._descriptor.name)

        node = self._get_or_create_node()
        node._set_value = NO_VALUE
        self._dag.invalidate_node(node)

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
        """Apply NodeChange operations from an inverse handler."""
        # NodeChange is a tuple of (node_accessor, new_value)
        # This allows mutual dependencies to be expressed
        if changes is None:
            return

        if not isinstance(changes, (list, tuple)):
            changes = [changes]

        for change in changes:
            if hasattr(change, 'apply'):
                change.apply()
            elif isinstance(change, tuple) and len(change) >= 2:
                # (node_getter, value) format
                node_accessor, value = change[0], change[1]
                if callable(node_accessor):
                    node_accessor().set(value)

    @property
    def _node(self) -> Optional[Node]:
        """Get the node if it exists."""
        from .core import NodeKey
        key = NodeKey(id(self._obj), self._descriptor.name, ())
        return self._dag.get_node(key)



# Type-preserving overloads for the decorator
@overload
def computed(func: F) -> F: ...

@overload
def computed(
    flags: int = Flags.NONE,
    *,
    inverse: Optional[Callable] = None,
) -> Callable[[F], F]: ...

@overload
def computed(
    flags: int,
    inverse: Optional[Callable] = None,
) -> Callable[[F], F]: ...


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
        def decorator(func: F) -> ComputedFunctionDescriptor:
            static_deps = parse_dependencies(func)
            return ComputedFunctionDescriptor(
                func=func,
                flags=Flags.NONE,
                inverse=inverse,
                static_deps=static_deps,
            )
        return decorator

    if callable(func_or_flags):
        # @computed without parentheses
        func = func_or_flags
        static_deps = parse_dependencies(func)
        return ComputedFunctionDescriptor(
            func=func,
            flags=Flags.NONE,
            inverse=None,
            static_deps=static_deps,
        )

    # @computed(flags) or @computed(flags, inverse=...)
    flags = func_or_flags

    def decorator(func: F) -> ComputedFunctionDescriptor:
        static_deps = parse_dependencies(func)
        return ComputedFunctionDescriptor(
            func=func,
            flags=flags,
            inverse=inverse,
            static_deps=static_deps,
        )

    return decorator


class NodeChange:
    """
    Represents a change to be applied to a node.

    Used by inverse handlers to express mutual dependencies.
    """

    def __init__(self, node_accessor: ComputedFunctionAccessor, value: Any):
        self.node_accessor = node_accessor
        self.value = value

    def apply(self) -> None:
        """Apply this change."""
        # Get the node and set its value directly (bypassing inverse)
        node = self.node_accessor._get_or_create_node()
        node._set_value = self.value
        self.node_accessor._dag.invalidate_node(node)
