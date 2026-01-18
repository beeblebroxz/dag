"""
Tests for cycle detection in the DAG.

Note: CycleError is wrapped in EvaluationError when it propagates up the call stack.
These tests verify that cycles are detected by checking for either exception type.
"""

import pytest
import dag


def raises_cycle_error(func):
    """Helper to check if a function raises CycleError (possibly wrapped in EvaluationError)."""
    try:
        func()
        return False
    except dag.CycleError:
        return True
    except dag.EvaluationError as e:
        # Check if the cause chain contains a CycleError
        cause = e.__cause__
        while cause is not None:
            if isinstance(cause, dag.CycleError):
                return True
            cause = getattr(cause, '__cause__', None)
        # Also check the error message
        return "Cyclic dependency" in str(e)
    except Exception:
        return False


class TestCycleDetection:
    """Test that cycles are properly detected and reported."""

    def setup_method(self):
        dag.reset()

    def test_direct_self_cycle(self):
        """Test that a computed function calling itself is detected as a cycle."""

        class SelfCycle(dag.Model):
            @dag.computed
            def A(self):
                return self.A() + 1  # Direct self-reference

        obj = SelfCycle()
        assert raises_cycle_error(lambda: obj.A())

    def test_two_node_cycle(self):
        """Test A -> B -> A cycle."""

        class TwoCycle(dag.Model):
            @dag.computed
            def A(self):
                return self.B() + 1

            @dag.computed
            def B(self):
                return self.A() + 1

        obj = TwoCycle()
        assert raises_cycle_error(lambda: obj.A())

    def test_three_node_cycle(self):
        """Test A -> B -> C -> A cycle."""

        class ThreeCycle(dag.Model):
            @dag.computed
            def A(self):
                return self.B() + 1

            @dag.computed
            def B(self):
                return self.C() + 1

            @dag.computed
            def C(self):
                return self.A() + 1

        obj = ThreeCycle()
        assert raises_cycle_error(lambda: obj.A())

    def test_conditional_cycle_taken(self):
        """Test cycle in conditional branch that is taken."""

        class ConditionalCycle(dag.Model):
            @dag.computed
            def A(self):
                if True:  # Always taken
                    return self.B() + 1
                return 0

            @dag.computed
            def B(self):
                return self.A() + 1

        obj = ConditionalCycle()
        assert raises_cycle_error(lambda: obj.A())

    def test_diamond_no_cycle(self):
        """Test that diamond dependencies don't cause false cycle detection."""
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
        # Should not raise CycleError
        result = obj.D()
        assert result == 112  # (1+10) + (1+100)

    def test_cycle_with_args(self):
        """Test cycle detection with parameterized computed functions."""

        class ParamCycle(dag.Model):
            @dag.computed
            def Compute(self, n):
                if n <= 0:
                    return 0
                return self.Compute(n - 1) + 1

        obj = ParamCycle()
        # Different args should NOT be detected as a cycle
        result = obj.Compute(3)
        assert result == 3

    def test_same_arg_cycle(self):
        """Test cycle when same args are used."""

        class SameArgCycle(dag.Model):
            @dag.computed
            def Compute(self, n):
                # Always calls with same arg - creates cycle
                return self.Compute(n) + 1

        obj = SameArgCycle()
        assert raises_cycle_error(lambda: obj.Compute(5))

    def test_indirect_cycle_through_multiple_objects(self):
        """Test cycle detection across multiple object instances."""

        class NodeA(dag.Model):
            def __init__(self, node_b=None):
                super().__init__()
                self._node_b = node_b

            def set_node_b(self, node_b):
                self._node_b = node_b

            @dag.computed
            def Value(self):
                if self._node_b:
                    return self._node_b.Value() + 1
                return 0

        class NodeB(dag.Model):
            def __init__(self, node_a=None):
                super().__init__()
                self._node_a = node_a

            def set_node_a(self, node_a):
                self._node_a = node_a

            @dag.computed
            def Value(self):
                if self._node_a:
                    return self._node_a.Value() + 1
                return 0

        a = NodeA()
        b = NodeB()
        a.set_node_b(b)
        b.set_node_a(a)

        # This should detect cycle across objects
        assert raises_cycle_error(lambda: a.Value())
