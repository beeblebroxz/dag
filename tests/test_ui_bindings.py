"""
Tests for the UI binding framework.

Note: These tests use Tkinter but avoid mainloop() for automated testing.
Event simulation is done by directly calling binding methods since
event_generate() doesn't work reliably in headless testing.
"""

import pytest
import tkinter as tk
import dag
from dag.ui import (
    DagApp,
    OutputBinding,
    InputBinding,
    TwoWayBinding,
    BoundLabel,
    BoundEntry,
    default_formatter,
    float_parser,
    int_parser,
    str_parser,
)


@pytest.fixture
def tk_root():
    """Create a Tk root for testing."""
    root = tk.Tk()
    root.withdraw()  # Hide the window
    yield root
    root.destroy()


@pytest.fixture
def dag_app(tk_root):
    """Create a DagApp for testing."""
    dag.reset()
    app = DagApp("Test App", root=tk_root)
    yield app
    app.destroy()


class TestFormattersAndParsers:
    """Test the formatter and parser functions."""

    def test_default_formatter_int(self):
        assert default_formatter(42) == "42"

    def test_default_formatter_float(self):
        assert default_formatter(3.14159) == "3.14159"

    def test_default_formatter_float_as_int(self):
        assert default_formatter(5.0) == "5"

    def test_default_formatter_string(self):
        assert default_formatter("hello") == "hello"

    def test_default_formatter_none(self):
        assert default_formatter(None) == ""

    def test_float_parser_valid(self):
        assert float_parser("3.14") == 3.14

    def test_float_parser_int(self):
        assert float_parser("42") == 42.0

    def test_float_parser_empty(self):
        assert float_parser("") == 0.0

    def test_float_parser_whitespace(self):
        assert float_parser("  ") == 0.0

    def test_int_parser_valid(self):
        assert int_parser("42") == 42

    def test_int_parser_float_string(self):
        assert int_parser("3.7") == 3

    def test_int_parser_empty(self):
        assert int_parser("") == 0

    def test_str_parser(self):
        assert str_parser("hello world") == "hello world"


class TestOutputBinding:
    """Test one-way output bindings (computed value -> widget)."""

    def setup_method(self):
        dag.reset()

    def test_output_binding_initial_value(self, dag_app):
        """Test that output binding sets initial value."""

        class Model(dag.Model):
            @dag.computed
            def Value(self):
                return 42

        model = Model()
        label = tk.Label(dag_app.root)

        binding = dag_app.bind_output(model.Value, label)

        assert label.cget('text') == "42"

    def test_output_binding_updates_on_change(self, dag_app):
        """Test that output binding updates when computed value changes."""

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Source(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Source() * 2

        model = Model()
        label = tk.Label(dag_app.root)

        binding = dag_app.bind_output(model.Derived, label)
        assert label.cget('text') == "2"

        # Change source
        model.Source = 5
        dag.flush()
        dag_app.root.update_idletasks()

        # Force the binding to update
        binding._update_widget()

        assert label.cget('text') == "10"

    def test_output_binding_custom_formatter(self, dag_app):
        """Test output binding with custom formatter."""

        class Model(dag.Model):
            @dag.computed
            def Price(self):
                return 123.456

        model = Model()
        label = tk.Label(dag_app.root)

        def currency_formatter(value):
            return f"${value:.2f}"

        binding = dag_app.bind_output(model.Price, label, formatter=currency_formatter)

        assert label.cget('text') == "$123.46"

    def test_output_binding_entry_widget(self, dag_app):
        """Test output binding with Entry widget (readonly)."""

        class Model(dag.Model):
            @dag.computed
            def Value(self):
                return "test value"

        model = Model()
        entry = tk.Entry(dag_app.root, state='readonly')

        binding = dag_app.bind_output(model.Value, entry, formatter=str)

        assert entry.get() == "test value"


class TestInputBinding:
    """Test one-way input bindings (widget -> computed value)."""

    def setup_method(self):
        dag.reset()

    def test_input_binding_updates_cell(self, dag_app):
        """Test that input binding updates computed value on widget change."""

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 0.0

        model = Model()
        entry = tk.Entry(dag_app.root)

        binding = dag_app.bind_input(model.Value, entry)

        # Simulate user input by directly calling the binding method
        entry.insert(0, "42.5")
        binding._on_widget_change()

        assert model.Value() == 42.5

    def test_input_binding_custom_parser(self, dag_app):
        """Test input binding with custom parser."""

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Count(self):
                return 0

        model = Model()
        entry = tk.Entry(dag_app.root)

        binding = dag_app.bind_input(model.Count, entry, parser=int_parser)

        entry.insert(0, "42")
        binding._on_widget_change()

        assert model.Count() == 42
        assert isinstance(model.Count(), int)


class TestTwoWayBinding:
    """Test two-way bindings (widget <-> computed value)."""

    def setup_method(self):
        dag.reset()

    def test_twoway_binding_initial_value(self, dag_app):
        """Test that two-way binding sets initial value."""

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 3.14

        model = Model()
        entry = tk.Entry(dag_app.root)

        binding = dag_app.bind_twoway(model.Value, entry)

        assert entry.get() == "3.14"

    def test_twoway_binding_widget_to_cell(self, dag_app):
        """Test two-way binding: widget -> computed value."""

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 0.0

        model = Model()
        entry = tk.Entry(dag_app.root)

        binding = dag_app.bind_twoway(model.Value, entry)

        # Clear and type new value
        entry.delete(0, tk.END)
        entry.insert(0, "99.9")
        binding._on_widget_change()

        assert model.Value() == 99.9

    def test_twoway_binding_cell_to_widget(self, dag_app):
        """Test two-way binding: computed value -> widget."""

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 0.0

        model = Model()
        entry = tk.Entry(dag_app.root)

        binding = dag_app.bind_twoway(model.Value, entry)
        assert entry.get() == "0"

        # Change computed value
        model.Value = 42.0
        dag.flush()

        # Force widget update
        binding._update_widget()

        assert entry.get() == "42"

    def test_twoway_binding_no_feedback_loop(self, dag_app):
        """Test that two-way binding doesn't create feedback loops."""
        update_count = {'cell': 0}

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                update_count['cell'] += 1
                return 0.0

        model = Model()
        entry = tk.Entry(dag_app.root)

        binding = dag_app.bind_twoway(model.Value, entry)

        # Initial evaluation
        initial_count = update_count['cell']

        # Change widget
        entry.delete(0, tk.END)
        entry.insert(0, "10")
        binding._on_widget_change()

        # Verify computed value was updated
        assert model.Value() == 10.0


class TestBoundWidgets:
    """Test the pre-built bound widgets."""

    def setup_method(self):
        dag.reset()

    def test_bound_label(self, dag_app):
        """Test BoundLabel widget."""

        class Model(dag.Model):
            @dag.computed
            def Message(self):
                return "Hello, World!"

        model = Model()
        label = BoundLabel(dag_app.root, cell=model.Message, app=dag_app)

        assert label.cget('text') == "Hello, World!"

    def test_bound_entry(self, dag_app):
        """Test BoundEntry widget."""

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 42.0

        model = Model()
        entry = BoundEntry(dag_app.root, cell=model.Value, app=dag_app)

        # Check initial value
        assert entry.get() == "42"

        # Test input by directly triggering binding
        entry.delete(0, tk.END)
        entry.insert(0, "100")
        entry._binding._on_widget_change()

        assert model.Value() == 100.0


class TestDagApp:
    """Test the DagApp class."""

    def setup_method(self):
        dag.reset()

    def test_app_creation(self, tk_root):
        """Test basic app creation."""
        app = DagApp("Test", root=tk_root)
        assert app.root is tk_root
        app.destroy()

    def test_app_bindings_list(self, dag_app):
        """Test that bindings are tracked."""

        class Model(dag.Model):
            @dag.computed
            def Value(self):
                return 1

        model = Model()
        label1 = tk.Label(dag_app.root)
        label2 = tk.Label(dag_app.root)

        dag_app.bind_output(model.Value, label1)
        dag_app.bind_output(model.Value, label2)

        assert len(dag_app.bindings) == 2

    def test_app_remove_binding(self, dag_app):
        """Test removing a binding."""

        class Model(dag.Model):
            @dag.computed
            def Value(self):
                return 1

        model = Model()
        label = tk.Label(dag_app.root)

        binding = dag_app.bind_output(model.Value, label)
        assert len(dag_app.bindings) == 1

        result = dag_app.remove_binding(binding)
        assert result is True
        assert len(dag_app.bindings) == 0

    def test_app_schedule_update(self, dag_app):
        """Test update scheduling."""

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Source(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Source() * 2

        model = Model()
        label = tk.Label(dag_app.root)

        binding = dag_app.bind_output(model.Derived, label)
        assert label.cget('text') == "2"

        # Change and schedule update
        model.Source = 10
        dag_app.schedule_update()
        dag_app.root.update_idletasks()

        # Force binding update
        binding._update_widget()

        assert label.cget('text') == "20"


class TestIntegration:
    """Integration tests with complete DAG models."""

    def setup_method(self):
        dag.reset()

    def test_calculator_model(self, dag_app):
        """Test a simple calculator model with bindings."""

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

            @dag.computed
            def Product(self):
                return self.A() * self.B()

        calc = Calculator()

        # Create widgets
        entry_a = BoundEntry(dag_app.root, cell=calc.A, app=dag_app)
        entry_b = BoundEntry(dag_app.root, cell=calc.B, app=dag_app)
        label_sum = BoundLabel(dag_app.root, cell=calc.Sum, app=dag_app)
        label_product = BoundLabel(dag_app.root, cell=calc.Product, app=dag_app)

        # Initial state
        assert label_sum.cget('text') == "0"
        assert label_product.cget('text') == "0"

        # Set A = 3
        entry_a.delete(0, tk.END)
        entry_a.insert(0, "3")
        entry_a._binding._on_widget_change()
        dag.flush()
        label_sum._binding._update_widget()
        label_product._binding._update_widget()

        assert label_sum.cget('text') == "3"
        assert label_product.cget('text') == "0"

        # Set B = 4
        entry_b.delete(0, tk.END)
        entry_b.insert(0, "4")
        entry_b._binding._on_widget_change()
        dag.flush()
        label_sum._binding._update_widget()
        label_product._binding._update_widget()

        assert label_sum.cget('text') == "7"
        assert label_product.cget('text') == "12"

    def test_chained_dependencies(self, dag_app):
        """Test bindings with chained computed function dependencies."""

        class Chain(dag.Model):
            @dag.computed(dag.Input)
            def Input(self):
                return 1.0

            @dag.computed
            def Step1(self):
                return self.Input() * 2

            @dag.computed
            def Step2(self):
                return self.Step1() + 10

            @dag.computed
            def Output(self):
                return self.Step2() ** 2

        chain = Chain()

        entry = BoundEntry(dag_app.root, cell=chain.Input, app=dag_app)
        label = BoundLabel(dag_app.root, cell=chain.Output, app=dag_app)

        # Initial: (1*2 + 10)^2 = 144
        assert label.cget('text') == "144"

        # Change to 5: (5*2 + 10)^2 = 400
        entry.delete(0, tk.END)
        entry.insert(0, "5")
        entry._binding._on_widget_change()
        dag.flush()
        label._binding._update_widget()

        assert label.cget('text') == "400"
