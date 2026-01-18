"""
Tests for parameterized computed functions (computed functions with arguments).
"""

import pytest
import dag


class TestParameterizedComputedFunctions:
    """Test computed functions that take arguments."""

    def setup_method(self):
        dag.reset()

    def test_single_arg(self):
        """Test computed function with single argument."""

        class Math(dag.Model):
            @dag.computed
            def Double(self, x):
                return x * 2

        obj = Math()
        assert obj.Double(5) == 10
        assert obj.Double(0) == 0
        assert obj.Double(-3) == -6

    def test_multiple_args(self):
        """Test computed function with multiple arguments."""

        class Math(dag.Model):
            @dag.computed
            def Add(self, a, b, c):
                return a + b + c

        obj = Math()
        assert obj.Add(1, 2, 3) == 6
        assert obj.Add(0, 0, 0) == 0

    def test_arg_caching_separate_values(self):
        """Test that different args produce separate cached values."""
        call_count = {}

        class Expensive(dag.Model):
            @dag.computed
            def Compute(self, x):
                call_count[x] = call_count.get(x, 0) + 1
                return x ** 2

        obj = Expensive()

        # First calls compute
        assert obj.Compute(2) == 4
        assert obj.Compute(3) == 9
        assert call_count[2] == 1
        assert call_count[3] == 1

        # Second calls use cache
        assert obj.Compute(2) == 4
        assert obj.Compute(3) == 9
        assert call_count[2] == 1  # Still 1
        assert call_count[3] == 1  # Still 1

    def test_mixed_arg_types(self):
        """Test caching with different argument types."""
        call_count = {'calls': 0}

        class Flexible(dag.Model):
            @dag.computed
            def Process(self, x):
                call_count['calls'] += 1
                return str(x)

        obj = Flexible()

        # Different values should be cached separately
        assert obj.Process(1) == "1"
        assert obj.Process("hello") == "hello"
        assert obj.Process(2.5) == "2.5"

        # Each unique arg was computed once
        assert call_count['calls'] == 3

        # Calling with same values uses cache
        obj.Process(1)
        obj.Process("hello")
        assert call_count['calls'] == 3  # Still 3

    def test_tuple_args(self):
        """Test computed function with tuple arguments."""

        class Container(dag.Model):
            @dag.computed
            def Sum(self, items):
                return sum(items)

        obj = Container()
        assert obj.Sum((1, 2, 3)) == 6
        assert obj.Sum((10, 20)) == 30

    def test_parameterized_with_dependency(self):
        """Test parameterized cells that depend on other cells."""

        class Scaled(dag.Model):
            @dag.computed(dag.Input)
            def Factor(self):
                return 2

            @dag.computed
            def Scale(self, value):
                return value * self.Factor()

        obj = Scaled()
        assert obj.Scale(5) == 10
        assert obj.Scale(3) == 6

        # Change factor
        obj.Factor = 3
        assert obj.Scale(5) == 15
        assert obj.Scale(3) == 9

    def test_parameterized_invalidation(self):
        """Test that parameterized cells properly invalidate."""
        call_count = {'scale': 0}

        class Scaled(dag.Model):
            @dag.computed(dag.Input)
            def Factor(self):
                return 2

            @dag.computed
            def Scale(self, value):
                call_count['scale'] += 1
                return value * self.Factor()

        obj = Scaled()

        # Initial computation
        assert obj.Scale(5) == 10
        assert call_count['scale'] == 1

        # Cache hit
        assert obj.Scale(5) == 10
        assert call_count['scale'] == 1

        # Different arg
        assert obj.Scale(3) == 6
        assert call_count['scale'] == 2

        # Change factor - should invalidate all Scale nodes
        obj.Factor = 3

        # Both should recompute
        assert obj.Scale(5) == 15
        assert obj.Scale(3) == 9
        assert call_count['scale'] == 4

    def test_nested_parameterized_calls(self):
        """Test nested calls to parameterized cells."""

        class Recursive(dag.Model):
            @dag.computed
            def Fib(self, n):
                if n <= 1:
                    return n
                return self.Fib(n - 1) + self.Fib(n - 2)

        obj = Recursive()
        assert obj.Fib(0) == 0
        assert obj.Fib(1) == 1
        assert obj.Fib(5) == 5
        assert obj.Fib(10) == 55

    def test_parameterized_with_override(self):
        """Test overriding affects parameterized cells."""

        class Scaled(dag.Model):
            @dag.computed(dag.Overridable)
            def Factor(self):
                return 2

            @dag.computed
            def Scale(self, value):
                return value * self.Factor()

        obj = Scaled()
        assert obj.Scale(5) == 10

        with dag.scenario():
            obj.Factor.override(10)
            assert obj.Scale(5) == 50
            assert obj.Scale(3) == 30

        # Reverted
        assert obj.Scale(5) == 10

    def test_kwargs_not_supported(self):
        """Test that keyword arguments raise an error."""

        class Math(dag.Model):
            @dag.computed
            def Add(self, a, b):
                return a + b

        obj = Math()
        with pytest.raises(ValueError) as exc_info:
            obj.Add(a=1, b=2)

        assert "keyword arguments" in str(exc_info.value)

    def test_none_arg(self):
        """Test computed function with None argument."""

        class Nullable(dag.Model):
            @dag.computed
            def Process(self, x):
                if x is None:
                    return "null"
                return str(x)

        obj = Nullable()
        assert obj.Process(None) == "null"
        assert obj.Process(1) == "1"

    def test_large_number_of_arg_variations(self):
        """Test many different argument values."""
        call_count = {'calls': 0}

        class Counter(dag.Model):
            @dag.computed
            def Count(self, x):
                call_count['calls'] += 1
                return x

        obj = Counter()

        # Create 100 different cached values
        for i in range(100):
            assert obj.Count(i) == i

        assert call_count['calls'] == 100

        # All should be cached now
        for i in range(100):
            assert obj.Count(i) == i

        assert call_count['calls'] == 100  # No new calls


class TestParameterizedSet:
    """Test set with parameterized cells."""

    def setup_method(self):
        dag.reset()

    def test_set_not_supported_for_parameterized(self):
        """Test that set on parameterized cells works for no-arg case."""

        class Settable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

        obj = Settable()
        obj.Value.set(10)
        assert obj.Value() == 10
