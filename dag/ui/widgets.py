"""
Pre-built reactive widgets for DAG-bound UIs.

These widgets have built-in bindings, making it easy to create
reactive interfaces with minimal code.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any, Callable, Optional

from .bindings import (
    Formatter,
    Parser,
    ErrorHandler,
    default_formatter,
    float_parser,
    int_parser,
    str_parser,
)

if TYPE_CHECKING:
    from ..decorators import ComputedFunctionAccessor
    from .app import DagApp


class BoundLabel(tk.Label):
    """
    A Label widget with built-in output binding.

    Automatically updates when the bound computed function changes.

    Example:
        label = BoundLabel(parent, cell=calc.Result, app=app)
        label.pack()
    """

    def __init__(
        self,
        master: tk.Widget,
        cell: ComputedFunctionAccessor,
        app: DagApp,
        formatter: Optional[Formatter] = None,
        on_error: Optional[ErrorHandler] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.cell = cell
        self.app = app
        self.formatter = formatter or default_formatter
        self.on_error = on_error
        self._binding = None

        # Create the binding
        self._binding = app.bind_output(
            cell_accessor=cell,
            widget=self,
            formatter=self.formatter,
            on_error=self.on_error,
        )

    def destroy(self) -> None:
        """Clean up binding when widget is destroyed."""
        if self._binding:
            self.app.remove_binding(self._binding)
        super().destroy()


class BoundEntry(tk.Entry):
    """
    An Entry widget with built-in two-way binding.

    Synchronizes with the bound computed function in both directions.

    Example:
        entry = BoundEntry(parent, cell=calc.Value, app=app)
        entry.pack()
    """

    def __init__(
        self,
        master: tk.Widget,
        cell: ComputedFunctionAccessor,
        app: DagApp,
        formatter: Optional[Formatter] = None,
        parser: Optional[Parser] = None,
        on_error: Optional[ErrorHandler] = None,
        update_on: str = 'focusout',
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.cell = cell
        self.app = app
        self.formatter = formatter or default_formatter
        self.parser = parser or float_parser
        self.on_error = on_error
        self._binding = None

        # Create the binding
        self._binding = app.bind_twoway(
            cell_accessor=cell,
            widget=self,
            formatter=self.formatter,
            parser=self.parser,
            on_error=self.on_error,
            update_on=update_on,
        )

    def destroy(self) -> None:
        """Clean up binding when widget is destroyed."""
        if self._binding:
            self.app.remove_binding(self._binding)
        super().destroy()


class BoundSpinbox(tk.Spinbox):
    """
    A Spinbox widget with built-in two-way binding.

    Synchronizes with the bound computed function, including spinbox button clicks.

    Example:
        spinbox = BoundSpinbox(
            parent, cell=model.Count, app=app,
            from_=0, to=100, increment=1
        )
        spinbox.pack()
    """

    def __init__(
        self,
        master: tk.Widget,
        cell: ComputedFunctionAccessor,
        app: DagApp,
        formatter: Optional[Formatter] = None,
        parser: Optional[Parser] = None,
        on_error: Optional[ErrorHandler] = None,
        update_on: str = 'focusout',
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.cell = cell
        self.app = app
        self.formatter = formatter or default_formatter
        self.parser = parser or float_parser
        self.on_error = on_error
        self._binding = None

        # Create the binding
        self._binding = app.bind_twoway(
            cell_accessor=cell,
            widget=self,
            formatter=self.formatter,
            parser=self.parser,
            on_error=self.on_error,
            update_on=update_on,
        )

    def destroy(self) -> None:
        """Clean up binding when widget is destroyed."""
        if self._binding:
            self.app.remove_binding(self._binding)
        super().destroy()


class BoundScale(tk.Scale):
    """
    A Scale (slider) widget with built-in two-way binding.

    Synchronizes with the bound computed function as the slider moves.

    Example:
        scale = BoundScale(
            parent, cell=model.Volume, app=app,
            from_=0, to=100, orient=tk.HORIZONTAL
        )
        scale.pack()
    """

    def __init__(
        self,
        master: tk.Widget,
        cell: ComputedFunctionAccessor,
        app: DagApp,
        formatter: Optional[Formatter] = None,
        parser: Optional[Parser] = None,
        on_error: Optional[ErrorHandler] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.cell = cell
        self.app = app
        self.formatter = formatter or default_formatter
        self.parser = parser or float_parser
        self.on_error = on_error
        self._binding = None

        # Create the binding
        self._binding = app.bind_twoway(
            cell_accessor=cell,
            widget=self,
            formatter=self.formatter,
            parser=self.parser,
            on_error=self.on_error,
        )

    def destroy(self) -> None:
        """Clean up binding when widget is destroyed."""
        if self._binding:
            self.app.remove_binding(self._binding)
        super().destroy()


class CellDisplay(tk.Frame):
    """
    A frame that displays a computed function's name and value.

    Shows a label with the name and a read-only display of its value.

    Example:
        display = CellDisplay(parent, "Price", calc.Price, app)
        display.pack()
    """

    def __init__(
        self,
        master: tk.Widget,
        label_text: str,
        cell: ComputedFunctionAccessor,
        app: DagApp,
        formatter: Optional[Formatter] = None,
        label_width: int = 10,
        value_width: int = 15,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.cell = cell
        self.app = app

        # Label for the name
        self.name_label = tk.Label(
            self,
            text=label_text,
            width=label_width,
            anchor='e',
        )
        self.name_label.pack(side=tk.LEFT, padx=(0, 5))

        # Label for the value
        self.value_label = BoundLabel(
            self,
            cell=cell,
            app=app,
            formatter=formatter,
            width=value_width,
            anchor='w',
            relief='sunken',
            bg='white',
        )
        self.value_label.pack(side=tk.LEFT)


class CellInput(tk.Frame):
    """
    A frame with a label and bound entry for a settable computed function.

    Shows a label with the name and an editable entry.

    Example:
        input_widget = CellInput(parent, "Strike", model.Strike, app)
        input_widget.pack()
    """

    def __init__(
        self,
        master: tk.Widget,
        label_text: str,
        cell: ComputedFunctionAccessor,
        app: DagApp,
        formatter: Optional[Formatter] = None,
        parser: Optional[Parser] = None,
        label_width: int = 10,
        entry_width: int = 15,
        update_on: str = 'focusout',
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.cell = cell
        self.app = app

        # Label for the name
        self.name_label = tk.Label(
            self,
            text=label_text,
            width=label_width,
            anchor='e',
        )
        self.name_label.pack(side=tk.LEFT, padx=(0, 5))

        # Entry for the value
        self.entry = BoundEntry(
            self,
            cell=cell,
            app=app,
            formatter=formatter,
            parser=parser,
            width=entry_width,
            update_on=update_on,
        )
        self.entry.pack(side=tk.LEFT)


class CellSlider(tk.Frame):
    """
    A frame with a label, slider, and value display for a settable computed function.

    Example:
        slider = CellSlider(
            parent, "Volume", model.Volume, app,
            from_=0, to=100
        )
        slider.pack()
    """

    def __init__(
        self,
        master: tk.Widget,
        label_text: str,
        cell: ComputedFunctionAccessor,
        app: DagApp,
        from_: float = 0,
        to: float = 100,
        resolution: float = 1,
        formatter: Optional[Formatter] = None,
        parser: Optional[Parser] = None,
        label_width: int = 10,
        scale_length: int = 150,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.cell = cell
        self.app = app

        # Label for the name
        self.name_label = tk.Label(
            self,
            text=label_text,
            width=label_width,
            anchor='e',
        )
        self.name_label.pack(side=tk.LEFT, padx=(0, 5))

        # Scale widget
        self.scale = BoundScale(
            self,
            cell=cell,
            app=app,
            formatter=formatter,
            parser=parser,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            length=scale_length,
            showvalue=True,
        )
        self.scale.pack(side=tk.LEFT)


class ModelInspector(tk.Frame):
    """
    A frame that displays all computed functions of a Model.

    Useful for debugging or quick prototyping.

    Example:
        inspector = ModelInspector(parent, model, app)
        inspector.pack(fill=tk.BOTH, expand=True)
    """

    def __init__(
        self,
        master: tk.Widget,
        model: Any,  # Model
        app: DagApp,
        show_inputs: bool = True,
        show_outputs: bool = True,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.model = model
        self.app = app

        # Get computed function names
        computed_names = model.get_computed_function_names()

        row = 0
        for name in sorted(computed_names):
            descriptor = model.get_computed_function(name)
            cell = getattr(model, name)

            # Check if it's settable
            from ..flags import Input
            is_settable = descriptor.flags & Input

            if is_settable and show_inputs:
                widget = CellInput(self, name, cell, app)
                widget.grid(row=row, column=0, sticky='w', pady=2)
                row += 1
            elif not is_settable and show_outputs:
                widget = CellDisplay(self, name, cell, app)
                widget.grid(row=row, column=0, sticky='w', pady=2)
                row += 1
