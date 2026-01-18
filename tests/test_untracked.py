"""
Tests for untracked mode.
"""

import pytest
import dag


class TestUntrackedMode:
    """Test the untracked() function for bypassing dependency checks."""

    def setup_method(self):
        dag.reset()

    def test_untracked_basic(self):
        """Test basic untracked usage."""

        class Simple(dag.Model):
            @dag.computed
            def A(self):
                return 1

            @dag.computed
            def B(self):
                return 2

        obj = Simple()

        # Call A within untracked context
        result = dag.untracked(lambda: obj.A())
        assert result == 1

    def test_untracked_returns_value(self):
        """Test that untracked returns the function's return value."""

        class Math(dag.Model):
            @dag.computed
            def Value(self):
                return 42

        obj = Math()
        result = dag.untracked(lambda: obj.Value() * 2)
        assert result == 84

    def test_untracked_with_complex_expression(self):
        """Test untracked with complex lambda."""

        class Multi(dag.Model):
            @dag.computed
            def A(self):
                return 10

            @dag.computed
            def B(self):
                return 20

        obj = Multi()
        result = dag.untracked(lambda: obj.A() + obj.B())
        assert result == 30

    def test_untracked_exception_propagates(self):
        """Test that exceptions in untracked propagate correctly."""

        class Faulty(dag.Model):
            @dag.computed
            def Broken(self):
                raise ValueError("Expected error")

        obj = Faulty()

        with pytest.raises(dag.EvaluationError):
            dag.untracked(lambda: obj.Broken())

    def test_untracked_caching_still_works(self):
        """Test that caching works within untracked."""
        call_count = {'A': 0}

        class Cached(dag.Model):
            @dag.computed
            def A(self):
                call_count['A'] += 1
                return 1

        obj = Cached()

        # First call computes
        dag.untracked(lambda: obj.A())
        assert call_count['A'] == 1

        # Second call uses cache
        dag.untracked(lambda: obj.A())
        assert call_count['A'] == 1

    def test_untracked_with_tweaks(self):
        """Test untracked respects active tweaks."""

        class Tweakable(dag.Model):
            @dag.computed(dag.Overridable)
            def Value(self):
                return 1

        obj = Tweakable()

        with dag.scenario():
            obj.Value.override(100)
            result = dag.untracked(lambda: obj.Value())
            assert result == 100

    def test_untracked_with_set_values(self):
        """Test untracked respects set values."""

        class Settable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

        obj = Settable()
        obj.Value = 50

        result = dag.untracked(lambda: obj.Value())
        assert result == 50

    def test_nested_untracked_calls(self):
        """Test nested untracked calls."""

        class Nested(dag.Model):
            @dag.computed
            def Inner(self):
                return 5

            @dag.computed
            def Outer(self):
                return 10

        obj = Nested()

        result = dag.untracked(lambda: (
            dag.untracked(lambda: obj.Inner()) +
            dag.untracked(lambda: obj.Outer())
        ))
        assert result == 15
