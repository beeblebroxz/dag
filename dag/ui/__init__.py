"""
DAG UI Binding Framework for Tkinter.

This module provides reactive UI bindings that connect DAG computed functions
to Tkinter widgets. When computed values change, bound widgets automatically
update. When users edit input widgets, the underlying values update
and trigger dependent recalculations.

Example:
    import dag
    from dag.ui import DagApp, BoundEntry, BoundLabel

    class Calculator(dag.Model):
        @dag.computed(dag.Input)
        def A(self):
            return 0.0

        @dag.computed(dag.Input)
        def B(self):
            return 0.0

        @dag.computed
        def Sum(self):
            return self.A() + self.B()

    app = DagApp("Calculator")
    calc = Calculator()

    # Inputs
    entry_a = BoundEntry(app.root, calc.A, app)
    entry_b = BoundEntry(app.root, calc.B, app)

    # Output
    label_sum = BoundLabel(app.root, calc.Sum, app)

    app.run()
"""

# Application
from .app import DagApp, DagFrame

# Bindings
from .bindings import (
    Binding,
    OutputBinding,
    InputBinding,
    TwoWayBinding,
    # Formatters and parsers
    default_formatter,
    float_parser,
    int_parser,
    str_parser,
)

# Widgets
from .widgets import (
    BoundLabel,
    BoundEntry,
    BoundSpinbox,
    BoundScale,
    CellDisplay,
    CellInput,
    CellSlider,
    ModelInspector,
)

__all__ = [
    # Application
    'DagApp',
    'DagFrame',
    # Bindings
    'Binding',
    'OutputBinding',
    'InputBinding',
    'TwoWayBinding',
    # Formatters/Parsers
    'default_formatter',
    'float_parser',
    'int_parser',
    'str_parser',
    # Widgets
    'BoundLabel',
    'BoundEntry',
    'BoundSpinbox',
    'BoundScale',
    'CellDisplay',
    'CellInput',
    'CellSlider',
    'ModelInspector',
]
