"""
Tests for inverse handlers.
"""

import pytest
import dag
from dag.decorators import NodeChange


class TestDelegateChange:
    """Test inverse handlers for mutual dependencies."""

    def setup_method(self):
        dag.reset()

    def test_basic_delegate_change(self):
        """Test basic inverse handler."""
        changes_applied = []

        class Linked(dag.Model):
            def _on_a_change(self, new_value):
                changes_applied.append(('A', new_value))
                # Return a NodeChange to update B
                return NodeChange(self.B, new_value * 2)

            @dag.computed(dag.Input, inverse=lambda self, v: self._on_a_change(v))
            def A(self):
                return 1

            @dag.computed(dag.Input)
            def B(self):
                return 2

        obj = Linked()
        assert obj.A() == 1
        assert obj.B() == 2

        # Setting A should trigger the delegate and update B
        obj.A.set(5)

        assert ('A', 5) in changes_applied
        # B should have been updated by the delegate
        assert obj.B() == 10

    def test_delegate_change_returns_none(self):
        """Test inverse handler that returns None (no-op)."""

        class Optional(dag.Model):
            def _on_change(self, new_value):
                if new_value < 0:
                    return None  # Don't propagate negative values
                return NodeChange(self.Other, new_value)

            @dag.computed(dag.Input, inverse=lambda self, v: self._on_change(v))
            def Value(self):
                return 0

            @dag.computed(dag.Input)
            def Other(self):
                return 0

        obj = Optional()

        # Positive value should propagate
        obj.Value.set(5)
        assert obj.Other() == 5

        # Negative value should not propagate
        obj.Value.set(-1)
        assert obj.Other() == 5  # Unchanged

    def test_delegate_change_returns_list(self):
        """Test inverse handler that returns multiple changes."""

        class Multi(dag.Model):
            def _on_value_change(self, new_value):
                return [
                    NodeChange(self.Double, new_value * 2),
                    NodeChange(self.Triple, new_value * 3),
                ]

            @dag.computed(dag.Input, inverse=lambda self, v: self._on_value_change(v))
            def Value(self):
                return 1

            @dag.computed(dag.Input)
            def Double(self):
                return 2

            @dag.computed(dag.Input)
            def Triple(self):
                return 3

        obj = Multi()
        assert obj.Value() == 1

        obj.Value.set(10)

        assert obj.Double() == 20
        assert obj.Triple() == 30

    def test_delegate_change_with_derived_cells(self):
        """Test that inverse handlers properly invalidate derived cells."""
        compute_count = {'Sum': 0}

        class WithDerived(dag.Model):
            def _on_a_change(self, new_value):
                # Delegate sets B based on the new value for A
                return NodeChange(self.B, new_value + 1)

            @dag.computed(dag.Input, inverse=lambda self, v: self._on_a_change(v))
            def A(self):
                return 1

            @dag.computed(dag.Input)
            def B(self):
                return 2

            @dag.computed
            def Sum(self):
                compute_count['Sum'] += 1
                return self.A() + self.B()

        obj = WithDerived()

        # Initial
        assert obj.Sum() == 3  # 1 + 2
        assert compute_count['Sum'] == 1

        # When using inverse, the inverse handler is called
        # and the value is NOT set on A itself - only the delegate changes are applied
        obj.A.set(5)

        # B should be updated by the delegate
        assert obj.B() == 6  # 5 + 1

        # A still returns default since delegate doesn't set A's value
        assert obj.A() == 1

        # Sum recomputes: A() returns 1 (default), B() returns 6 (from delegate)
        assert obj.Sum() == 7  # 1 + 6
        assert compute_count['Sum'] == 2

    def test_node_change_apply(self):
        """Test NodeChange.apply() method directly."""

        class Simple(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 0

        obj = Simple()
        assert obj.Value() == 0

        # Apply a NodeChange directly
        change = NodeChange(obj.Value, 42)
        change.apply()

        assert obj.Value() == 42


class TestNodeChangeClass:
    """Test the NodeChange class in isolation."""

    def setup_method(self):
        dag.reset()

    def test_node_change_stores_value(self):
        """Test that NodeChange stores the accessor and value."""

        class Simple(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 0

        obj = Simple()
        accessor = obj.Value

        change = NodeChange(accessor, 100)
        assert change.value == 100
        assert change.node_accessor is accessor

    def test_multiple_node_changes_independent(self):
        """Test that multiple NodeChanges are independent."""

        class Simple(dag.Model):
            @dag.computed(dag.Input)
            def A(self):
                return 0

            @dag.computed(dag.Input)
            def B(self):
                return 0

        obj = Simple()

        change_a = NodeChange(obj.A, 10)
        change_b = NodeChange(obj.B, 20)

        change_a.apply()
        assert obj.A() == 10
        assert obj.B() == 0  # Unchanged

        change_b.apply()
        assert obj.A() == 10
        assert obj.B() == 20
