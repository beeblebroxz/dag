"""
Tests for the watch/notification system.
"""

import pytest
import dag


class TestSubscriptions:
    """Test the watch system for invalidation notifications."""

    def setup_method(self):
        dag.reset()

    def test_watch_basic(self):
        """Test basic watch and notification."""
        notifications = []

        class Observable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Value() * 2

        obj = Observable()

        # Watch the derived computed function
        def on_invalidate(node):
            notifications.append(node.method_name)

        obj.Derived.watch(on_invalidate)

        # Initial evaluation
        assert obj.Derived() == 2
        assert len(notifications) == 0

        # Change the source value
        obj.Value = 5

        # Dispatch notifications
        dag.flush()

        # Should have been notified
        assert 'Derived' in notifications

    def test_watch_multiple_callbacks(self):
        """Test multiple callbacks on same computed function.

        Note: Watches fire when a node becomes INVALID (needs recomputation).
        For computed functions with set values, the node stays valid. We test with derived computed functions.
        """
        notifications = []

        class Observable(dag.Model):
            @dag.computed(dag.Input)
            def Source(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Source() * 2

        obj = Observable()

        def callback1(node):
            notifications.append('cb1')

        def callback2(node):
            notifications.append('cb2')

        # Watch the derived computed function (which becomes invalid when Source changes)
        obj.Derived.watch(callback1)
        obj.Derived.watch(callback2)

        # Evaluate first to establish dependency
        assert obj.Derived() == 2

        # Change source value - this invalidates Derived
        obj.Source = 10

        # Dispatch
        dag.flush()

        # Both callbacks should have been called
        assert 'cb1' in notifications
        assert 'cb2' in notifications

    def test_watch_chain_invalidation(self):
        """Test that watches work through dependency chains."""
        notifications = []

        class Chain(dag.Model):
            @dag.computed(dag.Input)
            def A(self):
                return 1

            @dag.computed
            def B(self):
                return self.A() + 10

            @dag.computed
            def C(self):
                return self.B() + 100

        obj = Chain()

        def on_c_invalidate(node):
            notifications.append('C invalidated')

        obj.C.watch(on_c_invalidate)

        # Initial evaluation
        assert obj.C() == 111

        # Change A (should propagate to C)
        obj.A = 5

        dag.flush()

        assert 'C invalidated' in notifications

    def test_watch_does_not_fire_when_valid(self):
        """Test that callbacks don't fire when node stays valid."""
        notifications = []

        class Multi(dag.Model):
            @dag.computed(dag.Input)
            def A(self):
                return 1

            @dag.computed(dag.Input)
            def B(self):
                return 2

            @dag.computed
            def SumA(self):
                return self.A() * 2

        obj = Multi()

        def on_sum_invalidate(node):
            notifications.append('SumA invalidated')

        obj.SumA.watch(on_sum_invalidate)

        # Evaluate
        assert obj.SumA() == 2

        # Change B (SumA doesn't depend on B)
        obj.B = 10

        dag.flush()

        # SumA should not have been notified
        assert 'SumA invalidated' not in notifications

    def test_watch_with_overrides(self):
        """Test watches work with overrides."""
        notifications = []

        class Overridable(dag.Model):
            @dag.computed(dag.Overridable)
            def Value(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Value() * 2

        obj = Overridable()

        def on_invalidate(node):
            notifications.append(node.method_name)

        obj.Derived.watch(on_invalidate)

        # Evaluate
        assert obj.Derived() == 2

        with dag.scenario():
            obj.Value.override(5)
            dag.flush()
            assert 'Derived' in notifications

    def test_callback_exception_does_not_propagate(self):
        """Test that exceptions in callbacks don't propagate."""

        class Observable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

        obj = Observable()

        def bad_callback(node):
            raise RuntimeError("Callback error")

        obj.Value.watch(bad_callback)

        # Evaluate and change
        assert obj.Value() == 1
        obj.Value = 5

        # Should not raise
        dag.flush()

    def test_watch_before_evaluation(self):
        """Test watching before first evaluation."""
        notifications = []

        class Observable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

        obj = Observable()

        def callback(node):
            notifications.append('called')

        # Watch before evaluation
        obj.Value.watch(callback)

        # Now set value (which creates node and invalidates it)
        obj.Value = 10

        dag.flush()

        # Should be notified
        assert 'called' in notifications
