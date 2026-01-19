"""
Model base class for DAG-tracked objects.

Models are the containers for computed functions. They follow these rules:
- No constructor with side effects (use default calculated values)
- No member variables (use Input computed functions instead)
- Everything is a function - use set/override to mutate state

Usage:
    class SimpleOption(dag.Model):
        @dag.computed
        def Pair(self):
            return 'EURUSD'

        @dag.computed(dag.Input)
        def Strike(self):
            return 1.0  # default value

        @dag.computed
        def Price(self):
            return max(0, self.Spot() - self.Strike())
"""

from __future__ import annotations

import weakref
from typing import Any, Dict, Optional, Set, Type

from .decorators import ComputedFunctionDescriptor


class ModelMeta(type):
    """
    Metaclass for Model that:
    - Collects all computed functions defined on the class
    - Validates Model constraints
    - Sets up the class for DAG tracking
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple,
        namespace: Dict[str, Any],
        **kwargs
    ) -> ModelMeta:
        # Collect computed functions from this class and bases
        computed_functions: Dict[str, ComputedFunctionDescriptor] = {}

        # Inherit computed functions from bases
        for base in bases:
            if hasattr(base, '_computed_functions_'):
                computed_functions.update(base._computed_functions_)

        # Find computed functions in this class
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, ComputedFunctionDescriptor):
                computed_functions[attr_name] = attr_value

        # Store computed functions registry
        namespace['_computed_functions_'] = computed_functions

        # Create the class
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        return cls


class Model(metaclass=ModelMeta):
    """
    Base class for objects with computed functions.

    Subclasses define computed functions using the @computed decorator.
    These functions are automatically tracked in the DAG for
    dependency management and memoization.

    Rules:
    - Don't override __init__ with side effects
    - Don't use instance variables directly (use Input computed functions)
    - Computed functions must be pure (same inputs -> same outputs)
    """

    # Registry of computed functions (populated by metaclass)
    _computed_functions_: Dict[str, ComputedFunctionDescriptor]

    # Store-awareness attributes (set by Store when object is stored/loaded)
    _store_ref: Optional[weakref.ref] = None  # Weak reference to Store
    _store_path: Optional[str] = None  # Path in the store

    def __init__(self):
        """
        Initialize the Model.

        Subclasses should generally not override this.
        If you need initialization, use Input computed functions with defaults.
        """
        # Initialize instance-level store attributes
        self._store_ref = None
        self._store_path = None

    def __init_subclass__(cls, **kwargs):
        """Called when a subclass is created."""
        super().__init_subclass__(**kwargs)

    @classmethod
    def get_computed_function_names(cls) -> Set[str]:
        """Get the names of all computed functions on this class."""
        return set(cls._computed_functions_.keys())

    @classmethod
    def get_computed_function(cls, name: str) -> Optional[ComputedFunctionDescriptor]:
        """Get a computed function descriptor by name."""
        return cls._computed_functions_.get(name)

    def _get_computed_methods(self) -> Dict[str, ComputedFunctionDescriptor]:
        """Get all computed function descriptors for this instance."""
        return self._computed_functions_

    # Store-awareness methods

    def path(self) -> Optional[str]:
        """Get this object's path in its store.

        Returns:
            The path if this object is stored, None otherwise.

        Example:
            option = db['/Instruments/AAPL_C_150']
            print(option.path())  # '/Instruments/AAPL_C_150'
        """
        return self._store_path

    def save(self) -> None:
        """Save this object to its store.

        The object must have been retrieved from or stored in a Store
        so we know its path and store.

        Raises:
            RuntimeError: If object is not associated with a store.

        Example:
            option = db['/Instruments/AAPL_C_150']
            option.Strike.set(160.0)
            option.save()  # Persists the change
        """
        if self._store_ref is None or self._store_path is None:
            raise RuntimeError(
                "Object is not associated with a store. "
                "Use db[path] = obj to store it first."
            )
        store = self._store_ref()
        if store is None:
            raise RuntimeError("Store has been garbage collected.")
        store.save(self)

    @property
    def store(self) -> Any:
        """Access the store this object belongs to.

        Returns:
            The Store instance, or raises RuntimeError if not stored.

        Example:
            option = db['/Instruments/AAPL_C_150']
            other = option.store['/Instruments/GOOGL_C_100']
        """
        if self._store_ref is None:
            raise RuntimeError("Object is not associated with a store.")
        store = self._store_ref()
        if store is None:
            raise RuntimeError("Store has been garbage collected.")
        return store


class RegistryMixin:
    """
    Mixin for Models that need registry access.

    Provides a 'db' property for accessing a registry
    of other Models, enabling indirection patterns like:
        self.db['/FX/Pairs/EURUSD'].Spot()
    """

    _registry: Optional[weakref.ref] = None

    @property
    def db(self) -> Any:
        """
        Access the registry.

        Returns a dict-like object for looking up other Models.
        """
        if self._registry is not None:
            registry = self._registry()
            if registry is not None:
                return registry
        raise RuntimeError("No registry attached to this Model")

    @classmethod
    def set_registry(cls, registry: Any) -> None:
        """Set the registry for all instances of this class."""
        cls._registry = weakref.ref(registry)


class Registry(dict):
    """
    A simple registry for Models.

    Allows lookup of Models by path:
        db['/FX/Pairs/EURUSD']
        db.new('FXSettledEuropean')

    This is a simplified version - production implementations would
    connect to actual persistence layers.
    """

    def __init__(self):
        super().__init__()
        self._factories: Dict[str, Type[Model]] = {}

    def register(self, name: str, cls: Type[Model]) -> None:
        """Register a Model class for creation."""
        self._factories[name] = cls

    def new(self, class_name: str, path: Optional[str] = None) -> Model:
        """
        Create a new Model instance.

        Args:
            class_name: The registered class name
            path: Optional path to store the object at

        Returns:
            A new Model instance
        """
        if class_name not in self._factories:
            raise KeyError(f"Unknown class: {class_name}")

        obj = self._factories[class_name]()

        if path is not None:
            self[path] = obj

        return obj
