"""
Basic tests for computed functions and memoization.
"""

import pytest
import dag


class TestBasicComputedFunctions:
    """Test basic computed function behavior."""

    def setup_method(self):
        """Reset DAG before each test."""
        dag.reset()

    def test_simple_computed_function(self):
        """Test that a simple computed function works."""

        class Simple(dag.Model):
            @dag.computed
            def Value(self):
                return 42

        obj = Simple()
        assert obj.Value() == 42

    def test_computed_function_with_dependency(self):
        """Test computed functions that depend on each other."""

        class Calculator(dag.Model):
            @dag.computed
            def A(self):
                return 10

            @dag.computed
            def B(self):
                return 20

            @dag.computed
            def Sum(self):
                return self.A() + self.B()

        calc = Calculator()
        assert calc.Sum() == 30

    def test_memoization(self):
        """Test that computed functions are memoized."""
        call_count = 0

        class Counter(dag.Model):
            @dag.computed
            def Value(self):
                nonlocal call_count
                call_count += 1
                return 42

        obj = Counter()

        # First call computes
        assert obj.Value() == 42
        assert call_count == 1

        # Second call uses cache
        assert obj.Value() == 42
        assert call_count == 1  # Still 1

    def test_default_value(self):
        """Test computed functions with default values."""

        class WithDefault(dag.Model):
            @dag.computed(dag.Input)
            def Strike(self):
                return 1.0  # default

        obj = WithDefault()
        assert obj.Strike() == 1.0

    def test_multiple_instances(self):
        """Test that multiple instances have separate state."""

        class Counter(dag.Model):
            _instance_count = 0

            def __init__(self):
                super().__init__()
                Counter._instance_count += 1
                self._id = Counter._instance_count

            @dag.computed
            def Id(self):
                return self._id

        obj1 = Counter()
        obj2 = Counter()

        assert obj1.Id() == 1
        assert obj2.Id() == 2


class TestComputedFunctionWithArguments:
    """Test computed functions that take arguments."""

    def setup_method(self):
        dag.reset()

    def test_computed_function_with_args(self):
        """Test computed function with arguments."""

        class Math(dag.Model):
            @dag.computed
            def Add(self, a, b):
                return a + b

        obj = Math()
        assert obj.Add(2, 3) == 5
        assert obj.Add(10, 20) == 30

    def test_argument_caching(self):
        """Test that different arguments get different cached values."""
        call_count = {}

        class Expensive(dag.Model):
            @dag.computed
            def Compute(self, x):
                call_count[x] = call_count.get(x, 0) + 1
                return x * 2

        obj = Expensive()

        assert obj.Compute(5) == 10
        assert obj.Compute(5) == 10  # cached
        assert call_count[5] == 1

        assert obj.Compute(10) == 20  # different args
        assert call_count[10] == 1


class TestChainedDependencies:
    """Test chains of dependencies."""

    def setup_method(self):
        dag.reset()

    def test_dependency_chain(self):
        """Test A -> B -> C dependency chain."""
        calls = []

        class Chain(dag.Model):
            @dag.computed
            def A(self):
                calls.append('A')
                return 1

            @dag.computed
            def B(self):
                calls.append('B')
                return self.A() + 1

            @dag.computed
            def C(self):
                calls.append('C')
                return self.B() + 1

        obj = Chain()
        assert obj.C() == 3
        # Should have evaluated in order (bottom-up during recursion)
        assert set(calls) == {'A', 'B', 'C'}

    def test_diamond_dependency(self):
        """Test diamond dependency pattern: A <- B, A <- C, B <- D, C <- D."""
        call_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        class Diamond(dag.Model):
            @dag.computed
            def A(self):
                call_counts['A'] += 1
                return 1

            @dag.computed
            def B(self):
                call_counts['B'] += 1
                return self.A() + 10

            @dag.computed
            def C(self):
                call_counts['C'] += 1
                return self.A() + 100

            @dag.computed
            def D(self):
                call_counts['D'] += 1
                return self.B() + self.C()

        obj = Diamond()
        result = obj.D()
        assert result == 112  # (1+10) + (1+100)

        # A should only be computed once due to memoization
        assert call_counts['A'] == 1
        assert call_counts['B'] == 1
        assert call_counts['C'] == 1
        assert call_counts['D'] == 1


class TestErrorHandling:
    """Test error handling in computed functions."""

    def setup_method(self):
        dag.reset()

    def test_error_propagation(self):
        """Test that errors propagate correctly."""

        class Faulty(dag.Model):
            @dag.computed
            def Failing(self):
                raise ValueError("Test error")

        obj = Faulty()
        with pytest.raises(dag.EvaluationError) as exc_info:
            obj.Failing()

        assert "Test error" in str(exc_info.value)

    def test_suppress_errors(self):
        """Test Optional flag."""

        class Suppressable(dag.Model):
            @dag.computed(dag.Optional)
            def MayFail(self):
                raise ValueError("Suppressed")

        obj = Suppressable()
        result = obj.MayFail()
        assert result is dag.NO_VALUE
