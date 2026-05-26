"""
Tests for dependency tracking and AST parsing.
"""

import textwrap

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
        # 'Price' is called on 'item', not 'self', so the parser cannot detect
        # it statically (it only sees self.X() chains). Such cross-object calls
        # are tracked at runtime and evaluate correctly -- see
        # TestDependencyTracking.test_cross_object_iteration_evaluates.

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

    def test_runtime_dependencies_are_rebuilt_on_recompute(self):
        """Test stale runtime dependencies are removed after recomputation."""
        counts = {'C': 0}

        class Conditional(dag.Model):
            @dag.computed(dag.Input)
            def UseA(self):
                return 1

            @dag.computed(dag.Input)
            def A(self):
                return 10

            @dag.computed(dag.Input)
            def B(self):
                return 20

            @dag.computed
            def C(self):
                counts['C'] += 1
                if self.UseA():
                    return self.A()
                return self.B()

        obj = Conditional()

        assert obj.C() == 10
        assert counts['C'] == 1

        obj.UseA = 0
        assert obj.C() == 20
        assert counts['C'] == 2

        # A should no longer be a live dependency after the branch changes.
        obj.A = 100
        assert obj.C() == 20
        assert counts['C'] == 2

    def test_cross_object_iteration_evaluates(self):
        """A computed function may call computed functions on OTHER models
        (e.g. summing over a collection). Such cross-object calls cannot be
        resolved statically by the parser, so they must be tracked at runtime
        without being rejected by the untracked check."""

        class Instrument(dag.Model):
            @dag.computed(dag.Input)
            def Price(self):
                return 0.0

        class Portfolio(dag.Model):
            @dag.computed(dag.Input)
            def Instruments(self):
                return []

            @dag.computed
            def TotalValue(self):
                return sum(inst.Price() for inst in self.Instruments())

        a, b = Instrument(), Instrument()
        a.Price.set(10.0)
        b.Price.set(20.0)

        p = Portfolio()
        p.Instruments.set([a, b])

        assert p.TotalValue() == 30.0

    def test_cross_object_dependency_invalidates(self):
        """Changing a child model's input invalidates a parent that consumed it
        via a cross-object call, proving the runtime edge was recorded."""

        class Instrument(dag.Model):
            @dag.computed(dag.Input)
            def Price(self):
                return 0.0

        class Portfolio(dag.Model):
            @dag.computed(dag.Input)
            def Instruments(self):
                return []

            @dag.computed
            def TotalValue(self):
                return sum(inst.Price() for inst in self.Instruments())

        a, b = Instrument(), Instrument()
        a.Price.set(10.0)
        b.Price.set(20.0)

        p = Portfolio()
        p.Instruments.set([a, b])
        assert p.TotalValue() == 30.0

        b.Price.set(50.0)
        assert p.TotalValue() == 60.0

    def test_deep_invalidation_does_not_overflow(self):
        """Invalidation must propagate through very deep dependency chains
        without exceeding Python's recursion limit."""

        class Chain(dag.Model):
            @dag.computed(dag.Input)
            def Base(self):
                return 0

            @dag.computed
            def Link(self, i):
                if i == 0:
                    return self.Base()
                return self.Link(i - 1) + 1

        obj = Chain()
        depth = 2000  # exceeds the default recursion limit

        # Build the chain bottom-up so evaluation itself stays shallow.
        for i in range(depth + 1):
            obj.Link(i)
        assert obj.Link(depth) == depth

        # Changing Base invalidates the entire chain; this must not overflow.
        obj.Base = 1

        # Re-evaluate bottom-up (keeps evaluation shallow) and confirm new value.
        for i in range(depth + 1):
            obj.Link(i)
        assert obj.Link(depth) == depth + 1


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

    def test_undeclared_runtime_dependency_raises(self):
        """Test runtime dependencies must be declared statically."""

        class Dynamic(dag.Model):
            @dag.computed(dag.Input)
            def A(self):
                return 1

            @dag.computed
            def B(self):
                return getattr(self, 'A')()

        obj = Dynamic()

        with pytest.raises(dag.UntrackedError):
            obj.B()

    def test_unparseable_source_is_tracked_not_rejected(self):
        """If a computed function's source can't be retrieved (e.g. exec'd code
        with no file backing), its dependencies can't be parsed. Such calls must
        be tracked at runtime rather than rejected as undeclared."""
        source = textwrap.dedent('''
            import dag

            class Dynamic(dag.Model):
                @dag.computed(dag.Input)
                def A(self):
                    return 1

                @dag.computed
                def B(self):
                    return self.A() + 1
        ''')
        namespace = {}
        exec(compile(source, '<dynamic-test>', 'exec'), namespace)
        obj = namespace['Dynamic']()

        # getsource('<dynamic-test>') fails, so B's deps are unknown; calling
        # self.A() must still work (tracked, not rejected with UntrackedError).
        assert obj.B() == 2

        # The runtime dependency edge is still recorded, so invalidation works.
        obj.A.set(5)
        assert obj.B() == 6

    def test_parse_dependencies_returns_none_when_source_unavailable(self):
        """parse_dependencies signals 'unknown' (None) when it cannot read the
        source, distinct from an empty set meaning 'parsed, no dependencies'."""
        source = textwrap.dedent('''
            def standalone(self):
                return self.A() + self.B()
        ''')
        namespace = {}
        exec(compile(source, '<dynamic-test>', 'exec'), namespace)

        assert parse_dependencies(namespace['standalone']) is None

        # A normally-defined function with no detectable deps returns an empty
        # set (parsed successfully), NOT None.
        def no_deps(self):
            return 42

        assert parse_dependencies(no_deps) == frozenset()
