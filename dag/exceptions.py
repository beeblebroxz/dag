"""
Custom exceptions for the DAG framework.
"""

from typing import Optional


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


class SetValueError(DagError):
    """Raised when trying to set a value on a computed function that doesn't support it."""

    def __init__(self, func_name: str, message: Optional[str] = None):
        self.func_name = func_name
        msg = message or f"Computed function '{func_name}' does not have Input flag"
        super().__init__(msg)


class OverrideError(DagError):
    """Raised when trying to override a value on a computed function that doesn't support it."""

    def __init__(self, func_name: str, message: Optional[str] = None):
        self.func_name = func_name
        msg = message or f"Computed function '{func_name}' does not have Overridable flag"
        super().__init__(msg)


class ScenarioError(DagError):
    """Raised when there's an issue with DAG scenario management."""
    pass


class ConcurrentScenarioError(ScenarioError):
    """Raised when a scenario or branch is used concurrently across threads.

    Scenarios and branches are single-threaded. While one is active on a
    thread, the DAG must not be evaluated or mutated from another thread.
    """

    def __init__(self, owner_thread: int, current_thread: int):
        self.owner_thread = owner_thread
        self.current_thread = current_thread
        super().__init__(
            f"A scenario/branch is active on thread {owner_thread}; the DAG "
            f"cannot be used concurrently from thread {current_thread}. "
            "Scenarios and branches are single-threaded."
        )


class EvaluationError(DagError):
    """Raised when there's an error during computed function evaluation."""

    def __init__(self, func_name: str, original_error: Exception):
        self.func_name = func_name
        self.original_error = original_error
        super().__init__(f"Error evaluating '{func_name}': {original_error}")


