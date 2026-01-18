"""
Tests for dependency tracking and AST parsing.
"""

import pytest
import dag
from dag.parser import parse_dependencies, parse_dependencies_detailed


class TestASTParser:
    """Test AST-based dependency detection."""

    def test_simple_dependency(self):
        """Test detection of simple self.X() calls."""

        class Sample(dag.Model):
            @dag.computed
            def A(self):
                return 1

            @dag.computed
            def B(self):
                return self.A() + 1

        # Check static deps were detected
        b_descriptor = Sample._computed_functions_['B']
        assert 'A' in b_descriptor.static_deps

    def test_multiple_dependencies(self):
        """Test detection of multiple dependencies."""

        class Sample(dag.Model):
            @dag.computed
            def X(self):
                return 1

            @dag.computed
            def Y(self):
                return 2

            @dag.computed
            def Z(self):
                return self.X() + self.Y()

        z_descriptor = Sample._computed_functions_['Z']
        assert 'X' in z_descriptor.static_deps
        assert 'Y' in z_descriptor.static_deps

    def test_chained_calls(self):
        """Test detection of chained calls like self.A().B()."""

        def sample_method(self):
            return self.PairObject().Spot()

        deps = parse_dependencies(sample_method)
        assert 'PairObject' in deps
        assert 'Spot' in deps

    def test_nested_calls(self):
        """Test detection of nested calls."""

        def sample_method(self):
            return max(0, self.Spot() - self.Strike())

        deps = parse_dependencies(sample_method)
        assert 'Spot' in deps
        assert 'Strike' in deps

    def test_conditional_dependencies(self):
        """Test that conditional dependencies are detected."""

        def sample_method(self):
            if self.UseA():
                return self.A()
            else:
                return self.B()

        deps = parse_dependencies(sample_method)
        assert 'UseA' in deps
        assert 'A' in deps
        assert 'B' in deps

    def test_loop_dependencies(self):
        """Test dependencies inside loops."""

        def sample_method(self):
            total = 0
            for item in self.Items():
                total += item.Price()
            return total

        deps = parse_dependencies(sample_method)
        assert 'Items' in deps
        # Note: 'Price' is called on 'item', not 'self', so it's not detected
        # as a static dependency. This is a known limitation - the parser
        # only tracks self.X() calls. Runtime tracking handles this.

    def test_self_method_in_loop(self):
        """Test self method calls inside loops."""

        def sample_method(self):
            total = 0
            for i in range(3):
                total += self.GetValue(i)
            return total

        deps = parse_dependencies(sample_method)
        assert 'GetValue' in deps


class TestDependencyTracking:
    """Test runtime dependency tracking."""

    def setup_method(self):
        dag.reset()

    def test_invalidation_propagates(self):
        """Test that invalidation propagates to dependents."""
        compute_count = {'B': 0}

        class Propagation(dag.Model):
            @dag.computed(dag.Input)
            def A(self):
                return 1

            @dag.computed
            def B(self):
                compute_count['B'] += 1
                return self.A() * 2

        obj = Propagation()

        # Initial computation
        assert obj.B() == 2
        assert compute_count['B'] == 1

        # Change A
        obj.A = 5
        # B should recompute
        assert obj.B() == 10
        assert compute_count['B'] == 2

    def test_deep_invalidation(self):
        """Test invalidation through multiple levels."""
        counts = {'A': 0, 'B': 0, 'C': 0}

        class Deep(dag.Model):
            @dag.computed(dag.Input)
            def A(self):
                counts['A'] += 1
                return 1

            @dag.computed
            def B(self):
                counts['B'] += 1
                return self.A() + 10

            @dag.computed
            def C(self):
                counts['C'] += 1
                return self.B() + 100

        obj = Deep()

        # Initial
        assert obj.C() == 111
        assert counts == {'A': 1, 'B': 1, 'C': 1}

        # Change A
        obj.A = 5

        # C should recompute (and so should B)
        assert obj.C() == 115
        assert counts == {'A': 1, 'B': 2, 'C': 2}  # A not recomputed (set value)


class TestStaticDependencies:
    """Test static vs runtime dependency detection."""

    def setup_method(self):
        dag.reset()

    def test_static_deps_stored(self):
        """Test that static dependencies are stored on descriptors."""

        class Sample(dag.Model):
            @dag.computed
            def A(self):
                return 1

            @dag.computed
            def B(self):
                return self.A()

        assert 'A' in Sample._computed_functions_['B'].static_deps

    def test_detailed_dependency_info(self):
        """Test detailed dependency information."""

        def complex_method(self):
            # Using chained call so both are detected
            spot = self.PairObject().Spot()
            strike = self.Strike()
            return max(0, spot - strike)

        details = parse_dependencies_detailed(complex_method)

        names = [d.name for d in details]
        assert 'PairObject' in names
        assert 'Spot' in names  # Detected from self.PairObject().Spot()
        assert 'Strike' in names
