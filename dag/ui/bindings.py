"""
Binding classes for connecting DAG computed functions to Tkinter widgets.

This module provides three types of bindings:
- OutputBinding: One-way binding from computed function to widget (display only)
- InputBinding: One-way binding from widget to computed function (input only)
- TwoWayBinding: Bidirectional binding between widget and computed function
"""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from ..decorators import ComputedFunctionAccessor
    from ..core import Node
    from .app import DagApp


# Type aliases
Formatter = Callable[[Any], str]
Parser = Callable[[str], Any]
ErrorHandler = Callable[[Exception, str], None]


def default_formatter(value: Any) -> str:
    """Default formatter: convert value to string."""
    if value is None:
        return ""
    if isinstance(value, float):
        # Format floats nicely
        if value == int(value):
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def float_parser(text: str) -> float:
    """Parse text as float, returning 0.0 for empty/invalid input."""
    text = text.strip()
    if not text:
        return 0.0
    return float(text)


def int_parser(text: str) -> int:
    """Parse text as int, returning 0 for empty/invalid input."""
    text = text.strip()
    if not text:
        return 0
    return int(float(text))  # Allow "3.0" -> 3


def str_parser(text: str) -> str:
    """Parse text as string (identity function)."""
    return text


class Binding:
    """
    Base class for all bindings.

    A binding connects a DAG computed function to a Tkinter widget,
    managing the synchronization between the two.
    """

    def __init__(
        self,
        cell_accessor: ComputedFunctionAccessor,
        widget: tk.Widget,
        app: DagApp,
        formatter: Optional[Formatter] = None,
        parser: Optional[Parser] = None,
        on_error: Optional[ErrorHandler] = None,
    ):
        self.cell_accessor = cell_accessor
        self.widget = widget
        self.app = app
        self.formatter = formatter or default_formatter
        self.parser = parser or float_parser
        self.on_error = on_error or self._default_error_handler
        self._updating = False  # Prevent feedback loops
        self._original_bg = None  # For error highlighting

    def _default_error_handler(self, error: Exception, context: str) -> None:
        """Default error handler: highlight widget and print error."""
        print(f"Binding error ({context}): {error}")
        # Highlight entry widgets with red background
        if isinstance(self.widget, (tk.Entry, tk.Spinbox)):
            if self._original_bg is None:
                self._original_bg = self.widget.cget('bg')
            self.widget.config(bg='#ffcccc')

    def _clear_error(self) -> None:
        """Clear error highlighting."""
        if self._original_bg is not None:
            if isinstance(self.widget, (tk.Entry, tk.Spinbox)):
                self.widget.config(bg=self._original_bg)
            self._original_bg = None

    def destroy(self) -> None:
        """Clean up the binding."""
        pass  # Subclasses can override


class OutputBinding(Binding):
    """
    One-way binding: computed function -> widget.

    Updates the widget display when the computed value changes.
    Used for labels, readonly displays, and computed outputs.
    """

    def __init__(
        self,
        cell_accessor: ComputedFunctionAccessor,
        widget: tk.Widget,
        app: DagApp,
        formatter: Optional[Formatter] = None,
        on_error: Optional[ErrorHandler] = None,
    ):
        super().__init__(
            cell_accessor=cell_accessor,
            widget=widget,
            app=app,
            formatter=formatter,
            on_error=on_error,
        )

        # Keep a strong reference to the callback to prevent garbage collection
        # (DAG uses weakref for subscriptions)
        self._cell_change_callback = self._on_cell_change

        # Subscribe to computed function invalidation
        self.cell_accessor.watch(self._cell_change_callback)

        # Initial update
        self._update_widget()

    def _on_cell_change(self, node: Node) -> None:
        """Called when the computed function is invalidated."""
        # Schedule widget update on the main thread
        self.widget.after_idle(self._update_widget)

    def _update_widget(self) -> None:
        """Update the widget with the current computed value."""
        if self._updating:
            return

        self._updating = True
        try:
            # Get the current value
            value = self.cell_accessor()
            display_text = self.formatter(value)

            # Update based on widget type
            if isinstance(self.widget, tk.Label):
                self.widget.config(text=display_text)
            elif isinstance(self.widget, tk.Entry):
                state = self.widget.cget('state')
                self.widget.config(state='normal')
                self.widget.delete(0, tk.END)
                self.widget.insert(0, display_text)
                self.widget.config(state=state)
            elif isinstance(self.widget, tk.Text):
                state = self.widget.cget('state')
                self.widget.config(state='normal')
                self.widget.delete('1.0', tk.END)
                self.widget.insert('1.0', display_text)
                self.widget.config(state=state)
            elif hasattr(self.widget, 'set'):
                # Scale, Spinbox variable, etc.
                self.widget.set(display_text)

            self._clear_error()

        except Exception as e:
            self.on_error(e, "updating widget")
        finally:
            self._updating = False


class InputBinding(Binding):
    """
    One-way binding: widget -> computed function.

    Updates the computed function value when the widget changes.
    Used for input fields that write to settable computed functions.
    """

    def __init__(
        self,
        cell_accessor: ComputedFunctionAccessor,
        widget: tk.Widget,
        app: DagApp,
        parser: Optional[Parser] = None,
        on_error: Optional[ErrorHandler] = None,
        update_on: str = 'focusout',  # 'focusout', 'key', or 'both'
    ):
        super().__init__(
            cell_accessor=cell_accessor,
            widget=widget,
            app=app,
            parser=parser,
            on_error=on_error,
        )
        self.update_on = update_on
        self._bind_events()

    def _bind_events(self) -> None:
        """Bind widget events for input detection."""
        if isinstance(self.widget, tk.Entry):
            if self.update_on in ('focusout', 'both'):
                self.widget.bind('<FocusOut>', self._on_widget_change)
                self.widget.bind('<Return>', self._on_widget_change)
            if self.update_on in ('key', 'both'):
                self.widget.bind('<KeyRelease>', self._on_widget_change)

        elif isinstance(self.widget, tk.Scale):
            self.widget.config(command=self._on_scale_change)

        elif isinstance(self.widget, tk.Spinbox):
            if self.update_on in ('focusout', 'both'):
                self.widget.bind('<FocusOut>', self._on_widget_change)
                self.widget.bind('<Return>', self._on_widget_change)
            if self.update_on in ('key', 'both'):
                self.widget.bind('<KeyRelease>', self._on_widget_change)
            # Also bind spinbox buttons
            self.widget.config(command=self._on_spinbox_change)

    def _on_widget_change(self, event: Optional[tk.Event] = None) -> None:
        """Called when the widget value changes."""
        if self._updating:
            return

        self._updating = True
        try:
            # Get widget value
            if isinstance(self.widget, tk.Entry):
                text = self.widget.get()
            elif isinstance(self.widget, tk.Spinbox):
                text = self.widget.get()
            elif hasattr(self.widget, 'get'):
                text = str(self.widget.get())
            else:
                return

            # Parse and set
            value = self.parser(text)
            self.cell_accessor.set(value)
            self.app.schedule_update()
            self._clear_error()

        except Exception as e:
            self.on_error(e, "parsing input")
        finally:
            self._updating = False

    def _on_scale_change(self, value: str) -> None:
        """Called when a Scale widget changes."""
        if self._updating:
            return

        self._updating = True
        try:
            parsed_value = self.parser(value)
            self.cell_accessor.set(parsed_value)
            self.app.schedule_update()
            self._clear_error()
        except Exception as e:
            self.on_error(e, "parsing scale value")
        finally:
            self._updating = False

    def _on_spinbox_change(self) -> None:
        """Called when Spinbox buttons are clicked."""
        self._on_widget_change()


class TwoWayBinding(Binding):
    """
    Two-way binding: widget <-> computed function.

    Synchronizes the widget and computed function in both directions.
    The computed function value is updated when the widget changes,
    and the widget is updated when the computed function changes.
    """

    def __init__(
        self,
        cell_accessor: ComputedFunctionAccessor,
        widget: tk.Widget,
        app: DagApp,
        formatter: Optional[Formatter] = None,
        parser: Optional[Parser] = None,
        on_error: Optional[ErrorHandler] = None,
        update_on: str = 'focusout',
    ):
        super().__init__(
            cell_accessor=cell_accessor,
            widget=widget,
            app=app,
            formatter=formatter,
            parser=parser,
            on_error=on_error,
        )
        self.update_on = update_on

        # Keep a strong reference to the callback to prevent garbage collection
        # (DAG uses weakref for subscriptions)
        self._cell_change_callback = self._on_cell_change

        # Subscribe to computed function changes
        self.cell_accessor.watch(self._cell_change_callback)

        # Bind widget events
        self._bind_events()

        # Initial sync
        self._update_widget()

    def _bind_events(self) -> None:
        """Bind widget events for input detection."""
        if isinstance(self.widget, tk.Entry):
            if self.update_on in ('focusout', 'both'):
                self.widget.bind('<FocusOut>', self._on_widget_change)
                self.widget.bind('<Return>', self._on_widget_change)
            if self.update_on in ('key', 'both'):
                self.widget.bind('<KeyRelease>', self._on_widget_change)

        elif isinstance(self.widget, tk.Scale):
            self.widget.config(command=self._on_scale_change)

        elif isinstance(self.widget, tk.Spinbox):
            if self.update_on in ('focusout', 'both'):
                self.widget.bind('<FocusOut>', self._on_widget_change)
                self.widget.bind('<Return>', self._on_widget_change)
            if self.update_on in ('key', 'both'):
                self.widget.bind('<KeyRelease>', self._on_widget_change)
            self.widget.config(command=self._on_spinbox_change)

    def _on_cell_change(self, node: Node) -> None:
        """Called when the computed function is invalidated."""
        self.widget.after_idle(self._update_widget)

    def _update_widget(self) -> None:
        """Update the widget with the current computed value."""
        if self._updating:
            return

        self._updating = True
        try:
            value = self.cell_accessor()
            display_text = self.formatter(value)

            if isinstance(self.widget, tk.Entry):
                # Preserve cursor position if possible
                cursor_pos = self.widget.index(tk.INSERT)
                self.widget.delete(0, tk.END)
                self.widget.insert(0, display_text)
                try:
                    self.widget.icursor(min(cursor_pos, len(display_text)))
                except tk.TclError:
                    pass

            elif isinstance(self.widget, tk.Spinbox):
                self.widget.delete(0, tk.END)
                self.widget.insert(0, display_text)

            elif isinstance(self.widget, tk.Scale):
                self.widget.set(float(value) if value is not None else 0)

            elif hasattr(self.widget, 'set'):
                self.widget.set(display_text)

            self._clear_error()

        except Exception as e:
            self.on_error(e, "updating widget")
        finally:
            self._updating = False

    def _on_widget_change(self, event: Optional[tk.Event] = None) -> None:
        """Called when the widget value changes."""
        if self._updating:
            return

        self._updating = True
        try:
            if isinstance(self.widget, (tk.Entry, tk.Spinbox)):
                text = self.widget.get()
            elif hasattr(self.widget, 'get'):
                text = str(self.widget.get())
            else:
                return

            value = self.parser(text)
            self.cell_accessor.set(value)
            self.app.schedule_update()
            self._clear_error()

        except Exception as e:
            self.on_error(e, "parsing input")
        finally:
            self._updating = False

    def _on_scale_change(self, value: str) -> None:
        """Called when a Scale widget changes."""
        if self._updating:
            return

        self._updating = True
        try:
            parsed_value = self.parser(value)
            self.cell_accessor.set(parsed_value)
            self.app.schedule_update()
            self._clear_error()
        except Exception as e:
            self.on_error(e, "parsing scale value")
        finally:
            self._updating = False

    def _on_spinbox_change(self) -> None:
        """Called when Spinbox buttons are clicked."""
        self._on_widget_change()
