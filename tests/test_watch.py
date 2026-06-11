"""
Tests for the watch/notification system.
"""

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

    def test_callback_exception_is_logged_not_swallowed(self, caplog):
        """A raising callback must not crash flush, but its error must be
        surfaced (logged) rather than silently swallowed, and other callbacks
        must still run."""
        import logging

        class Observable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Value() * 2

        obj = Observable()
        ran = []

        def bad_callback(node):
            raise RuntimeError("boom")

        def good_callback(node):
            ran.append('good')

        obj.Derived.watch(bad_callback)
        obj.Derived.watch(good_callback)

        assert obj.Derived() == 2
        obj.Value = 5  # invalidates Derived -> callbacks fire on flush

        with caplog.at_level(logging.ERROR):
            dag.flush()  # must not raise

        # The bad callback's failure is surfaced, not silently swallowed.
        assert any('Watch callback' in r.getMessage() for r in caplog.records)
        # A failing callback does not prevent others from running.
        assert 'good' in ran

    def test_flush_notifies_once_per_change(self):
        """A change enqueues one notification; flushing twice fires the callback
        only once (the pending set is cleared on dispatch)."""

        class Observable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Value() * 2

        obj = Observable()
        calls = []

        def cb(node):
            calls.append(1)

        obj.Derived.watch(cb)
        assert obj.Derived() == 2
        obj.Value = 5  # invalidates Derived -> enqueued once
        dag.flush()
        dag.flush()  # pending is now empty

        assert len(calls) == 1

    def test_flush_fires_only_changed_watched_nodes(self):
        """Only nodes that changed since the last flush are notified; an
        unrelated, still-invalid watched node is not swept in."""

        class M(dag.Model):
            @dag.computed(dag.Input)
            def A(self):
                return 1

            @dag.computed(dag.Input)
            def B(self):
                return 1

            @dag.computed
            def DA(self):
                return self.A() + 1

            @dag.computed
            def DB(self):
                return self.B() + 1

        obj = M()
        fired = []

        def cb_da(node):
            fired.append('DA')

        def cb_db(node):
            fired.append('DB')

        obj.DA.watch(cb_da)
        obj.DB.watch(cb_db)

        assert obj.DA() == 2  # DA evaluated (valid); DB never evaluated (invalid)
        obj.A = 9             # changes only DA's input
        dag.flush()

        assert fired == ['DA']  # DB (invalid but unchanged) must NOT fire

    def test_watcher_rearms_after_reevaluation(self):
        """After a notification the watcher fires again only once the node is
        re-evaluated (re-armed) and then changed again."""

        class Observable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Value() * 2

        obj = Observable()
        calls = []

        def cb(node):
            calls.append(1)

        obj.Derived.watch(cb)
        assert obj.Derived() == 2
        obj.Value = 5
        dag.flush()
        assert len(calls) == 1

        assert obj.Derived() == 10  # re-evaluate -> re-arm
        obj.Value = 7
        dag.flush()
        assert len(calls) == 2


class TestWatchOnChangedNodeItself:
    """Watchers of the node being overridden/set/cleared must be notified,
    not only watchers of its dependents (audit 2026-06 #1/#7/#9)."""

    def setup_method(self):
        dag.reset()

    def test_override_notifies_watchers_of_overridden_node(self):
        class Market(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

        m = Market()
        events = []

        def on_spot(node):
            events.append(m.Spot())  # evaluate to re-arm

        m.Spot.watch(on_spot)
        m.Spot()  # prime

        with dag.scenario():
            m.Spot.override(120.0)
            dag.flush()
            assert events == [120.0]

        dag.flush()
        assert events == [120.0, 100.0]  # revert also notifies

    def test_clear_value_notifies_watchers(self):
        class Market(dag.Model):
            @dag.computed(dag.Input)
            def Spot(self):
                return 100.0

        m = Market()
        events = []

        def on_spot(node):
            events.append(m.Spot())

        m.Spot.watch(on_spot)
        m.Spot.set(120.0)  # set before any evaluation
        dag.flush()
        assert events == [120.0]

        m.Spot.clearValue()
        dag.flush()
        assert events == [120.0, 100.0]

    def test_node_change_apply_notifies_watchers(self):
        from dag.decorators import NodeChange

        class Market(dag.Model):
            @dag.computed(dag.Input)
            def Spot(self):
                return 100.0

        m = Market()
        events = []

        def on_spot(node):
            events.append(m.Spot())

        m.Spot.watch(on_spot)
        NodeChange(m.Spot, 120.0).apply()  # node never evaluated before
        dag.flush()
        assert events == [120.0]
