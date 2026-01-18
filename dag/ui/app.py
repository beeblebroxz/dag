"""
DagApp: Main application class for DAG-bound Tkinter UIs.

Provides event loop integration with automatic subscription dispatching.
"""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import dag
from .bindings import (
    Binding,
    InputBinding,
    OutputBinding,
    TwoWayBinding,
    Formatter,
    Parser,
    ErrorHandler,
    default_formatter,
    float_parser,
)

if TYPE_CHECKING:
    from ..decorators import ComputedFunctionAccessor


class DagApp:
    """
    Main application class with integrated DAG event loop.

    DagApp wraps a Tkinter root window and provides:
    - Automatic dispatching of DAG subscriptions
    - Convenient binding creation methods
    - Debounced updates for performance

    Example:
        app = DagApp("My Calculator")

        calc = Calculator()

        # Create bindings
        entry = tk.Entry(app.root)
        app.bind_twoway(calc.Value, entry)

        label = tk.Label(app.root)
        app.bind_output(calc.Result, label)

        app.run()
    """

    def __init__(
        self,
        title: str = "DAG Application",
        width: int = 400,
        height: int = 300,
        root: Optional[tk.Tk] = None,
    ):
        """
        Initialize the DAG application.

        Args:
            title: Window title
            width: Initial window width
            height: Initial window height
            root: Optional existing Tk root (creates new one if None)
        """
        if root is not None:
            self.root = root
            self._owns_root = False
        else:
            self.root = tk.Tk()
            self._owns_root = True

        self.root.title(title)
        self.root.geometry(f"{width}x{height}")

        self._bindings: List[Binding] = []
        self._update_pending = False
        self._running = False

    def bind_output(
        self,
        cell_accessor: ComputedFunctionAccessor,
        widget: tk.Widget,
        formatter: Optional[Formatter] = None,
        on_error: Optional[ErrorHandler] = None,
    ) -> OutputBinding:
        """
        Create a one-way output binding: computed function -> widget.

        The widget will update whenever the computed value changes.

        Args:
            cell_accessor: The computed function accessor (e.g., obj.Price)
            widget: The Tkinter widget to update
            formatter: Optional function to format values for display
            on_error: Optional error handler

        Returns:
            The created OutputBinding
        """
        binding = OutputBinding(
            cell_accessor=cell_accessor,
            widget=widget,
            app=self,
            formatter=formatter,
            on_error=on_error,
        )
        self._bindings.append(binding)
        return binding

    def bind_input(
        self,
        cell_accessor: ComputedFunctionAccessor,
        widget: tk.Widget,
        parser: Optional[Parser] = None,
        on_error: Optional[ErrorHandler] = None,
        update_on: str = 'focusout',
    ) -> InputBinding:
        """
        Create a one-way input binding: widget -> computed function.

        The computed function will update whenever the widget value changes.

        Args:
            cell_accessor: The computed function accessor (e.g., obj.Strike)
            widget: The Tkinter widget to read from
            parser: Optional function to parse widget text to value
            on_error: Optional error handler
            update_on: When to update ('focusout', 'key', or 'both')

        Returns:
            The created InputBinding
        """
        binding = InputBinding(
            cell_accessor=cell_accessor,
            widget=widget,
            app=self,
            parser=parser,
            on_error=on_error,
            update_on=update_on,
        )
        self._bindings.append(binding)
        return binding

    def bind_twoway(
        self,
        cell_accessor: ComputedFunctionAccessor,
        widget: tk.Widget,
        formatter: Optional[Formatter] = None,
        parser: Optional[Parser] = None,
        on_error: Optional[ErrorHandler] = None,
        update_on: str = 'focusout',
    ) -> TwoWayBinding:
        """
        Create a two-way binding: widget <-> computed function.

        The widget and computed function stay synchronized in both directions.

        Args:
            cell_accessor: The computed function accessor (e.g., obj.Strike)
            widget: The Tkinter widget to bind
            formatter: Optional function to format values for display
            parser: Optional function to parse widget text to value
            on_error: Optional error handler
            update_on: When to update from widget ('focusout', 'key', or 'both')

        Returns:
            The created TwoWayBinding
        """
        binding = TwoWayBinding(
            cell_accessor=cell_accessor,
            widget=widget,
            app=self,
            formatter=formatter,
            parser=parser,
            on_error=on_error,
            update_on=update_on,
        )
        self._bindings.append(binding)
        return binding

    def schedule_update(self) -> None:
        """
        Schedule a DAG subscription dispatch.

        This is debounced so multiple rapid changes result in
        a single dispatch on the next event loop tick.
        """
        if not self._update_pending:
            self._update_pending = True
            self.root.after_idle(self._do_update)

    def _do_update(self) -> None:
        """Perform the actual subscription dispatch."""
        self._update_pending = False
        dag.flush()

    def run(self) -> None:
        """
        Start the application main loop.

        This blocks until the window is closed.
        """
        self._running = True

        # Initial dispatch to sync all bindings
        dag.flush()

        try:
            self.root.mainloop()
        finally:
            self._running = False

    def quit(self) -> None:
        """Stop the application and close the window."""
        self._running = False
        self.root.quit()

    def destroy(self) -> None:
        """Destroy the application and clean up resources."""
        for binding in self._bindings:
            binding.destroy()
        self._bindings.clear()

        if self._owns_root:
            self.root.destroy()

    @property
    def bindings(self) -> List[Binding]:
        """Get all registered bindings."""
        return self._bindings.copy()

    def remove_binding(self, binding: Binding) -> bool:
        """
        Remove a binding from the application.

        Args:
            binding: The binding to remove

        Returns:
            True if the binding was found and removed
        """
        try:
            self._bindings.remove(binding)
            binding.destroy()
            return True
        except ValueError:
            return False


class DagFrame(tk.Frame):
    """
    A Frame that integrates with a DagApp.

    Convenience class for creating modular UI components
    that participate in DAG binding.
    """

    def __init__(
        self,
        master: tk.Widget,
        app: DagApp,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.app = app

    def bind_output(self, *args, **kwargs) -> OutputBinding:
        """Delegate to app.bind_output()."""
        return self.app.bind_output(*args, **kwargs)

    def bind_input(self, *args, **kwargs) -> InputBinding:
        """Delegate to app.bind_input()."""
        return self.app.bind_input(*args, **kwargs)

    def bind_twoway(self, *args, **kwargs) -> TwoWayBinding:
        """Delegate to app.bind_twoway()."""
        return self.app.bind_twoway(*args, **kwargs)
