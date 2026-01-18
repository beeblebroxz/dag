"""
Custom exceptions for the DAG framework.
"""


class DagError(Exception):
    """Base exception for all DAG-related errors."""
    pass


class DependencyError(DagError):
    """Raised when there's an issue with dependency tracking."""
    pass


class UntrackedError(DependencyError):
    """
    Raised when a computed function is called that wasn't detected at parse time.

    The DAG evaluator throws this if you make a call to a computed function which
    it did not detect at parse time. Without this check, you would miss a
    dependency and a change to the function input would result in an invalid cached result.

    Use dag.untracked() to suppress this check if you're really sure you don't
    want the dependency tracked.
    """
    pass


class CycleError(DagError):
    """Raised when a cyclic dependency is detected in the DAG."""
    pass


class InvalidationError(DagError):
    """Raised when there's an issue during invalidation propagation."""
    pass


class SetValueError(DagError):
    """Raised when trying to set a value on a computed function that doesn't support it."""

    def __init__(self, func_name: str, message: str = None):
        self.func_name = func_name
        msg = message or f"Computed function '{func_name}' does not have Input flag"
        super().__init__(msg)


class OverrideError(DagError):
    """Raised when trying to override a value on a computed function that doesn't support it."""

    def __init__(self, func_name: str, message: str = None):
        self.func_name = func_name
        msg = message or f"Computed function '{func_name}' does not have Overridable flag"
        super().__init__(msg)


class ScenarioError(DagError):
    """Raised when there's an issue with DAG scenario management."""
    pass


class EvaluationError(DagError):
    """Raised when there's an error during computed function evaluation."""

    def __init__(self, func_name: str, original_error: Exception):
        self.func_name = func_name
        self.original_error = original_error
        super().__init__(f"Error evaluating '{func_name}': {original_error}")


class ParseError(DagError):
    """Raised when there's an error parsing a computed function for dependencies."""

    def __init__(self, func_name: str, message: str):
        self.func_name = func_name
        super().__init__(f"Error parsing '{func_name}': {message}")


class ModelError(DagError):
    """Raised when there's a violation of Model constraints."""
    pass


class ConstructorError(ModelError):
    """Raised when a Model has an invalid constructor."""

    def __init__(self, class_name: str):
        self.class_name = class_name
        super().__init__(
            f"Model '{class_name}' should not have a constructor with side effects. "
            "Use default calculated values via computed functions instead."
        )
