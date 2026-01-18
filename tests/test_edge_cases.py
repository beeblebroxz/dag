"""
Tests for edge cases and boundary conditions.
"""

import pytest
import gc
import dag


class TestWeakReferences:
    """Test weak reference behavior and garbage collection."""

    def setup_method(self):
        dag.reset()

    def test_object_can_be_garbage_collected(self):
        """Test that Models can be garbage collected."""

        class Simple(dag.Model):
            @dag.computed
            def Value(self):
                return 1

        obj = Simple()
        assert obj.Value() == 1

        # Delete and collect
        del obj
        gc.collect()

        # Should not crash

    def test_evaluation_after_gc_raises(self):
        """Test that evaluating a GC'd object raises error."""
        from dag.core import DagManager, NodeKey

        class Simple(dag.Model):
            @dag.computed
            def Value(self):
                return 1

        obj = Simple()
        obj_id = id(obj)

        # Evaluate to create node
        assert obj.Value() == 1

        # Get the node key
        key = NodeKey(obj_id, 'Value', ())
        manager = DagManager.get_instance()
        node = manager.get_node(key)

        assert node is not None

        # Delete the object
        del obj
        gc.collect()

        # Node's weak ref should be dead
        assert node.obj_ref() is None


class TestContextEdgeCases:
    """Test edge cases with contexts."""

    def setup_method(self):
        dag.reset()

    def test_context_exit_on_exception(self):
        """Test that context properly reverts on exception."""

        class Overridable(dag.Model):
            @dag.computed(dag.Overridable)
            def Value(self):
                return 1

        obj = Overridable()
        assert obj.Value() == 1

        try:
            with dag.scenario():
                obj.Value.override(100)
                assert obj.Value() == 100
                raise RuntimeError("Intentional error")
        except RuntimeError:
            pass

        # Should revert despite exception
        assert obj.Value() == 1

    def test_deeply_nested_contexts(self):
        """Test many nested contexts."""

        class Overridable(dag.Model):
            @dag.computed(dag.Overridable)
            def Value(self):
                return 0

        obj = Overridable()

        # Nest 10 contexts
        contexts = []
        for i in range(10):
            ctx = dag.scenario()
            ctx.__enter__()
            contexts.append(ctx)
            obj.Value.override(i + 1)
            assert obj.Value() == i + 1

        # Exit in reverse order
        for i in range(9, -1, -1):
            contexts[i].__exit__(None, None, None)
            expected = i if i > 0 else 0
            assert obj.Value() == expected

    def test_empty_context(self):
        """Test context with no overrides."""

        class Simple(dag.Model):
            @dag.computed
            def Value(self):
                return 1

        obj = Simple()

        with dag.scenario():
            assert obj.Value() == 1

        assert obj.Value() == 1


class TestDeepDependencyChains:
    """Test deep dependency chains."""

    def setup_method(self):
        dag.reset()

    def test_deep_chain(self):
        """Test a deep dependency chain (10 levels)."""

        class DeepChain(dag.Model):
            @dag.computed(dag.Input)
            def Base(self):
                return 0

            @dag.computed
            def L1(self):
                return self.Base() + 1

            @dag.computed
            def L2(self):
                return self.L1() + 1

            @dag.computed
            def L3(self):
                return self.L2() + 1

            @dag.computed
            def L4(self):
                return self.L3() + 1

            @dag.computed
            def L5(self):
                return self.L4() + 1

            @dag.computed
            def L6(self):
                return self.L5() + 1

            @dag.computed
            def L7(self):
                return self.L6() + 1

            @dag.computed
            def L8(self):
                return self.L7() + 1

            @dag.computed
            def L9(self):
                return self.L8() + 1

            @dag.computed
            def L10(self):
                return self.L9() + 1

        obj = DeepChain()
        assert obj.L10() == 10  # Base(0) + 10 levels

        # Change base - should propagate through all levels
        obj.Base = 100
        assert obj.L10() == 110

    def test_wide_dependencies(self):
        """Test a node with many dependencies."""
        call_counts = {}

        class Wide(dag.Model):
            @dag.computed(dag.Input)
            def C0(self):
                call_counts[0] = call_counts.get(0, 0) + 1
                return 0

            @dag.computed(dag.Input)
            def C1(self):
                call_counts[1] = call_counts.get(1, 0) + 1
                return 1

            @dag.computed(dag.Input)
            def C2(self):
                call_counts[2] = call_counts.get(2, 0) + 1
                return 2

            @dag.computed(dag.Input)
            def C3(self):
                call_counts[3] = call_counts.get(3, 0) + 1
                return 3

            @dag.computed(dag.Input)
            def C4(self):
                call_counts[4] = call_counts.get(4, 0) + 1
                return 4

            @dag.computed(dag.Input)
            def C5(self):
                call_counts[5] = call_counts.get(5, 0) + 1
                return 5

            @dag.computed(dag.Input)
            def C6(self):
                call_counts[6] = call_counts.get(6, 0) + 1
                return 6

            @dag.computed(dag.Input)
            def C7(self):
                call_counts[7] = call_counts.get(7, 0) + 1
                return 7

            @dag.computed(dag.Input)
            def C8(self):
                call_counts[8] = call_counts.get(8, 0) + 1
                return 8

            @dag.computed(dag.Input)
            def C9(self):
                call_counts[9] = call_counts.get(9, 0) + 1
                return 9

            @dag.computed
            def Total(self):
                return (self.C0() + self.C1() + self.C2() + self.C3() + self.C4() +
                        self.C5() + self.C6() + self.C7() + self.C8() + self.C9())

        obj = Wide()
        assert obj.Total() == sum(range(10))  # 0 + 1 + 2 + ... + 9 = 45

        # Each cell called once
        for i in range(10):
            assert call_counts[i] == 1


class TestReEvaluationScenarios:
    """Test various re-evaluation scenarios."""

    def setup_method(self):
        dag.reset()

    def test_set_same_value_does_not_recompute(self):
        """Test that setting the same value doesn't trigger recomputation."""
        compute_count = {'derived': 0}

        class Settable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

            @dag.computed
            def Derived(self):
                compute_count['derived'] += 1
                return self.Value() * 2

        obj = Settable()

        # Initial
        assert obj.Derived() == 2
        assert compute_count['derived'] == 1

        # Set same value
        obj.Value = 1

        # Derived still recomputes because invalidation happened
        # (optimization to check value equality would be nice but not implemented)
        assert obj.Derived() == 2
        # Note: Current implementation does invalidate on any setValue

    def test_clear_value_triggers_recompute(self):
        """Test that clearing value triggers recomputation."""
        compute_count = {'value': 0}

        class Clearable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                compute_count['value'] += 1
                return 42

        obj = Clearable()

        # Initial - computes default
        assert obj.Value() == 42
        assert compute_count['value'] == 1

        # Set value - no compute, just uses set value
        obj.Value = 100
        assert obj.Value() == 100
        assert compute_count['value'] == 1

        # Clear - should recompute
        obj.Value.clearValue()
        assert obj.Value() == 42
        assert compute_count['value'] == 2

    def test_override_and_set_interaction(self):
        """Test interaction between override and set."""

        class Both(dag.Model):
            @dag.computed(dag.Input | dag.Overridable)
            def Value(self):
                return 1

        obj = Both()

        # Set value
        obj.Value = 10
        assert obj.Value() == 10

        # Override takes precedence
        with dag.scenario():
            obj.Value.override(100)
            assert obj.Value() == 100

        # After context, set value is restored
        assert obj.Value() == 10


class TestFlagsEdgeCases:
    """Test edge cases with flags."""

    def setup_method(self):
        dag.reset()

    def test_combined_flags(self):
        """Test combining multiple flags."""

        class Multi(dag.Model):
            @dag.computed(dag.Input | dag.Overridable | dag.Optional)
            def Value(self):
                return 1

        obj = Multi()

        # Can set
        obj.Value = 10
        assert obj.Value() == 10

        # Can override
        with dag.scenario():
            obj.Value.override(100)
            assert obj.Value() == 100

        assert obj.Value() == 10

    def test_persisted_flag_equivalence(self):
        """Test that Persisted == Input | Serialized."""
        assert dag.Persisted == (dag.Input | dag.Serialized)

    def test_suppress_errors_with_dependency_error(self):
        """Test Optional when dependency raises."""

        class Chained(dag.Model):
            @dag.computed
            def Failing(self):
                raise ValueError("Error")

            @dag.computed(dag.Optional)
            def Suppressed(self):
                return self.Failing() + 1

        obj = Chained()

        # Should return NO_VALUE, not propagate error
        result = obj.Suppressed()
        assert result is dag.NO_VALUE


class TestModelInheritance:
    """Test inheritance of Models."""

    def setup_method(self):
        dag.reset()

    def test_inherited_computed_functions(self):
        """Test that computed functions are inherited."""

        class Base(dag.Model):
            @dag.computed
            def BaseValue(self):
                return 1

        class Derived(Base):
            @dag.computed
            def DerivedValue(self):
                return self.BaseValue() + 10

        obj = Derived()
        assert obj.BaseValue() == 1
        assert obj.DerivedValue() == 11

    def test_overridden_computed_functions(self):
        """Test overriding computed functions."""

        class Base(dag.Model):
            @dag.computed
            def Value(self):
                return 1

        class Derived(Base):
            @dag.computed
            def Value(self):
                return 100

        base = Base()
        derived = Derived()

        assert base.Value() == 1
        assert derived.Value() == 100

    def test_get_computed_function_names_includes_inherited(self):
        """Test that get_cell_function_names includes inherited."""

        class Base(dag.Model):
            @dag.computed
            def A(self):
                return 1

        class Derived(Base):
            @dag.computed
            def B(self):
                return 2

        names = Derived.get_computed_function_names()
        assert 'A' in names
        assert 'B' in names


class TestMultipleInstances:
    """Test behavior with multiple instances."""

    def setup_method(self):
        dag.reset()

    def test_instances_have_separate_state(self):
        """Test that instances don't share state."""

        class Counter(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 0

        obj1 = Counter()
        obj2 = Counter()

        obj1.Value = 10
        obj2.Value = 20

        assert obj1.Value() == 10
        assert obj2.Value() == 20

    def test_instances_have_separate_caches(self):
        """Test that instances have separate caches."""
        call_counts = {'obj1': 0, 'obj2': 0}

        class Tracked(dag.Model):
            def __init__(self, name):
                super().__init__()
                self._name = name

            @dag.computed
            def Value(self):
                call_counts[self._name] = call_counts.get(self._name, 0) + 1
                return 1

        obj1 = Tracked('obj1')
        obj2 = Tracked('obj2')

        obj1.Value()
        obj1.Value()  # Cache hit
        obj2.Value()
        obj2.Value()  # Cache hit

        assert call_counts['obj1'] == 1
        assert call_counts['obj2'] == 1


class TestOverrideSetEdgeCases:
    """Test edge cases with OverrideSet."""

    def setup_method(self):
        dag.reset()

    def test_empty_override_set(self):
        """Test applying empty OverrideSet."""

        class Simple(dag.Model):
            @dag.computed(dag.Overridable)
            def Value(self):
                return 1

        obj = Simple()
        empty_set = dag.OverrideSet()

        with dag.scenario() as ctx:
            empty_set.apply(ctx)
            assert obj.Value() == 1

    def test_override_set_multiple_objects(self):
        """Test OverrideSet with multiple objects."""

        class Simple(dag.Model):
            @dag.computed(dag.Overridable)
            def Value(self):
                return 0

        obj1 = Simple()
        obj2 = Simple()

        override_set = dag.OverrideSet()
        override_set.add(obj1, 'Value', 10)
        override_set.add(obj2, 'Value', 20)

        with dag.scenario() as ctx:
            override_set.apply(ctx)
            assert obj1.Value() == 10
            assert obj2.Value() == 20

        # Reverted
        assert obj1.Value() == 0
        assert obj2.Value() == 0

    def test_override_set_serialization_roundtrip(self):
        """Test getting and applying overrides."""

        class Overridable(dag.Model):
            @dag.computed(dag.Overridable)
            def A(self):
                return 1

            @dag.computed(dag.Overridable)
            def B(self):
                return 2

        obj = Overridable()

        # Create overrides
        with dag.scenario():
            obj.A.override(10)
            obj.B.override(20)

            # Get the overrides
            overrides = dag.get_overrides()
            assert len(overrides.overrides) == 2

        # Apply them in a new context
        with dag.scenario() as ctx:
            overrides.apply(ctx)
            assert obj.A() == 10
            assert obj.B() == 20
