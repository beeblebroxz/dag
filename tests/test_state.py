"""
Tests for state management: set, override, scenarios, and branches.
"""

import pytest
import dag


class TestSetValue:
    """Test permanent value changes via set."""

    def setup_method(self):
        dag.reset()

    def test_set_value_basic(self):
        """Test basic set functionality."""

        class Option(dag.Model):
            @dag.computed(dag.Input)
            def Strike(self):
                return 1.0

        opt = Option()
        assert opt.Strike() == 1.0

        opt.Strike.set(1.5)
        assert opt.Strike() == 1.5

    def test_set_value_via_assignment(self):
        """Test set via property assignment syntax."""

        class Option(dag.Model):
            @dag.computed(dag.Input)
            def Strike(self):
                return 1.0

        opt = Option()
        opt.Strike = 1.5
        assert opt.Strike() == 1.5

    def test_set_value_without_flag_raises(self):
        """Test that set without Input flag raises error."""

        class Option(dag.Model):
            @dag.computed
            def Strike(self):
                return 1.0

        opt = Option()
        with pytest.raises(dag.SetValueError):
            opt.Strike.set(1.5)

    def test_set_value_invalidates_dependents(self):
        """Test that set invalidates dependent cells."""
        count = {'price': 0}

        class Option(dag.Model):
            @dag.computed(dag.Input)
            def Strike(self):
                return 1.0

            @dag.computed(dag.Input)
            def Spot(self):
                return 1.1

            @dag.computed
            def Price(self):
                count['price'] += 1
                return max(0, self.Spot() - self.Strike())

        opt = Option()
        assert abs(opt.Price() - 0.1) < 1e-9  # 1.1 - 1.0 (floating point)
        assert count['price'] == 1

        opt.Strike = 1.05
        assert abs(opt.Price() - 0.05) < 1e-9  # 1.1 - 1.05
        assert count['price'] == 2

    def test_clear_value(self):
        """Test clearing a set value."""

        class Option(dag.Model):
            @dag.computed(dag.Input)
            def Strike(self):
                return 1.0

        opt = Option()
        opt.Strike = 2.0
        assert opt.Strike() == 2.0

        opt.Strike.clearValue()
        assert opt.Strike() == 1.0  # back to default


class TestOverride:
    """Test temporary value overrides via override."""

    def setup_method(self):
        dag.reset()

    def test_override_basic(self):
        """Test basic override functionality."""

        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 1.0

        opt = Option()
        assert opt.Spot() == 1.0

        with dag.scenario():
            opt.Spot.override(1.5)
            assert opt.Spot() == 1.5

        # Reverts after scenario
        assert opt.Spot() == 1.0

    def test_override_without_flag_raises(self):
        """Test that override without Overridable flag raises error."""

        class Option(dag.Model):
            @dag.computed
            def Spot(self):
                return 1.0

        opt = Option()
        with dag.scenario():
            with pytest.raises(dag.OverrideError):
                opt.Spot.override(1.5)

    def test_override_outside_scenario_raises(self):
        """Test that override outside scenario raises error."""

        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 1.0

        opt = Option()
        with pytest.raises(dag.OverrideError):
            opt.Spot.override(1.5)

    def test_nested_overrides(self):
        """Test nested override scenarios."""

        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 1.0

        opt = Option()

        with dag.scenario():
            opt.Spot.override(2.0)
            assert opt.Spot() == 2.0

            with dag.scenario():
                opt.Spot.override(3.0)
                assert opt.Spot() == 3.0

            # Inner scenario ended
            assert opt.Spot() == 2.0

        # Outer scenario ended
        assert opt.Spot() == 1.0

    def test_override_propagates(self):
        """Test that overrides propagate to dependents."""

        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 1.0

            @dag.computed(dag.Input)
            def Strike(self):
                return 0.9

            @dag.computed
            def Price(self):
                return max(0, self.Spot() - self.Strike())

        opt = Option()
        assert abs(opt.Price() - 0.1) < 1e-9

        with dag.scenario():
            opt.Spot.override(1.5)
            assert abs(opt.Price() - 0.6) < 1e-9  # 1.5 - 0.9

        assert abs(opt.Price() - 0.1) < 1e-9  # back to original


class TestScenario:
    """Test scenario management."""

    def setup_method(self):
        dag.reset()

    def test_scenario_as_manager(self):
        """Test scenario works as context manager."""

        class Simple(dag.Model):
            @dag.computed(dag.Overridable)
            def Value(self):
                return 1

        obj = Simple()

        ctx = dag.scenario()
        with ctx:
            obj.Value.override(2)
            assert obj.Value() == 2

        assert obj.Value() == 1

    def test_multiple_overrides_in_scenario(self):
        """Test multiple overrides in same scenario."""

        class Multi(dag.Model):
            @dag.computed(dag.Overridable)
            def A(self):
                return 1

            @dag.computed(dag.Overridable)
            def B(self):
                return 2

            @dag.computed
            def Sum(self):
                return self.A() + self.B()

        obj = Multi()
        assert obj.Sum() == 3

        with dag.scenario():
            obj.A.override(10)
            obj.B.override(20)
            assert obj.Sum() == 30

        assert obj.Sum() == 3


class TestOverrideSets:
    """Test OverrideSet serialization."""

    def setup_method(self):
        dag.reset()

    def test_get_overrides(self):
        """Test getting current overrides as an OverrideSet."""

        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 1.0

        opt = Option()

        with dag.scenario():
            opt.Spot.override(1.5)
            overrides = dag.get_overrides()
            assert len(overrides.overrides) == 1
            assert overrides.overrides[0].value == 1.5

    def test_apply_overrides(self):
        """Test applying an OverrideSet."""

        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 1.0

            @dag.computed
            def Price(self):
                return self.Spot() * 2

        opt = Option()

        # Create an override set
        override_set = dag.OverrideSet()
        override_set.add(opt, 'Spot', 2.0)

        assert opt.Price() == 2.0  # 1.0 * 2

        # Apply overrides
        with dag.scenario() as ctx:
            override_set.apply(ctx)
            assert opt.Price() == 4.0  # 2.0 * 2

        assert opt.Price() == 2.0  # reverted


class TestBranches:
    """Test branch functionality."""

    def setup_method(self):
        dag.reset()

    def test_branch_basic(self):
        """Test basic branch functionality."""

        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Strike(self):
                return 1.0

        opt = Option()

        with dag.branch():
            opt.Strike.override(1.4)
            assert opt.Strike() == 1.4

        # After branch exits
        assert opt.Strike() == 1.0

    def test_branch_reentry_preserves_overrides(self):
        """Test a branch remembers overrides across re-entry."""

        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Strike(self):
                return 1.0

        opt = Option()
        stressed = dag.Branch()

        with stressed:
            opt.Strike.override(1.4)
            assert opt.Strike() == 1.4

        assert opt.Strike() == 1.0

        with stressed:
            assert opt.Strike() == 1.4

        assert opt.Strike() == 1.0

    def test_branches_are_isolated(self):
        """Test sibling branches keep separate override state."""

        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Strike(self):
                return 1.0

        opt = Option()
        branch_a = dag.Branch()
        branch_b = dag.Branch()

        with branch_a:
            opt.Strike.override(1.4)

        with branch_b:
            opt.Strike.override(1.5)

        with branch_a:
            assert opt.Strike() == 1.4

        with branch_b:
            assert opt.Strike() == 1.5

        assert opt.Strike() == 1.0


class TestPersisted:
    """Test Persisted flag (Input | Serialized)."""

    def setup_method(self):
        dag.reset()

    def test_persisted_flag(self):
        """Test that Persisted combines Input and Serialized."""
        assert dag.Persisted == (dag.Input | dag.Serialized)

    def test_persisted_can_be_set(self):
        """Test that Persisted cells can be set."""

        class Data(dag.Model):
            @dag.computed(dag.Persisted)
            def Value(self):
                return "default"

        obj = Data()
        assert obj.Value() == "default"

        obj.Value = "changed"
        assert obj.Value() == "changed"


class TestClearValue:
    """Regression tests for clearValue invalidation (audit 2026-06 #1)."""

    def setup_method(self):
        dag.reset()

    def test_clear_value_before_first_evaluation_invalidates_dependents(self):
        """A node set before it was ever evaluated never becomes VALID, but its
        dependents still consume the set value. clearValue must invalidate them."""

        class Option(dag.Model):
            @dag.computed(dag.Input)
            def Strike(self):
                return 1.0

            @dag.computed
            def Price(self):
                return self.Strike() * 2

        opt = Option()
        opt.Strike.set(5.0)        # set BEFORE first evaluation
        assert opt.Price() == 10.0

        opt.Strike.clearValue()
        assert opt.Strike() == 1.0
        assert opt.Price() == 2.0  # must not be stale 10.0

    def test_clear_value_after_evaluation_invalidates_dependents(self):
        """Control: clearValue after the node has been VALID also works."""

        class Option(dag.Model):
            @dag.computed(dag.Input)
            def Strike(self):
                return 1.0

            @dag.computed
            def Price(self):
                return self.Strike() * 2

        opt = Option()
        assert opt.Price() == 2.0  # Strike evaluated -> VALID
        opt.Strike.set(5.0)
        assert opt.Price() == 10.0
        opt.Strike.clearValue()
        assert opt.Price() == 2.0


class TestBranchReentry:
    """Regression tests for Branch re-entrancy (audit 2026-06 #2)."""

    def setup_method(self):
        dag.reset()

    def _model(self):
        class Option(dag.Model):
            @dag.computed(dag.Overridable)
            def Strike(self):
                return 1.0

            @dag.computed
            def Price(self):
                return self.Strike() * 2

        return Option()

    def test_branch_nested_reentry(self):
        """The nested re-entry pattern documented in CLAUDE.md must work."""
        opt = self._model()

        with dag.branch() as b1:
            opt.Strike.override(1.4)
            with b1:
                assert opt.Price() == 2.8

        assert opt.Price() == 2.0

    def test_branch_nested_reentry_releases_scenario_ownership(self):
        """After nested branch use, the DAG must not stay locked to one thread."""
        opt = self._model()

        with dag.branch() as b1:
            opt.Strike.override(1.4)
            with b1:
                opt.Price()

        mgr = dag.DagManager.get_instance()
        assert mgr._scenario_owner is None
        assert mgr.current_context is None

    def test_branch_override_before_entry(self):
        """Branch.override() before the branch is first entered must not crash;
        the override applies when the branch is entered."""
        opt = self._model()

        b = dag.Branch()
        b.override(opt, "Strike", 1.5)
        with b:
            assert opt.Price() == 3.0
        assert opt.Price() == 2.0

    def test_branch_exit_without_enter_raises(self):
        b = dag.Branch()
        with pytest.raises(dag.ScenarioError):
            b.__exit__(None, None, None)


class TestOverridePathsPreserveStaticDeps:
    """Nodes created via override plumbing must carry the descriptor's parsed
    static dependencies (audit 2026-06 #3)."""

    def setup_method(self):
        dag.reset()

    def _model(self):
        class Market(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return self.Base() + 1

            @dag.computed
            def Base(self):
                return 10.0

        return Market()

    def test_override_set_apply_preserves_static_deps(self):
        m = self._model()
        ovs = dag.OverrideSet()
        ovs.add(m, "Spot", 99.0)

        with dag.apply_overrides(ovs):
            assert m.Spot() == 99.0

        # Spot's node was first created via OverrideSet.apply; evaluating it
        # normally must not raise UntrackedError for its real self-deps.
        assert m.Spot() == 11.0

    def test_branch_override_preserves_static_deps(self):
        m = self._model()
        b = dag.Branch()
        with b:
            b.override(m, "Spot", 99.0)
            assert m.Spot() == 99.0
        assert m.Spot() == 11.0


class TestOverrideFlagEnforcement:
    """Branch.override and OverrideSet.apply must enforce the Overridable flag
    like accessor.override() does (audit 2026-06 #5)."""

    def setup_method(self):
        dag.reset()

    def _model(self):
        class Fixed(dag.Model):
            @dag.computed  # NOT Overridable
            def Value(self):
                return 42.0

        return Fixed()

    def test_branch_override_requires_overridable_flag(self):
        m = self._model()
        b = dag.Branch()
        with pytest.raises(dag.OverrideError):
            b.override(m, "Value", 0.0)

    def test_override_set_apply_requires_overridable_flag(self):
        m = self._model()
        ovs = dag.OverrideSet()
        ovs.add(m, "Value", 0.0)
        with pytest.raises(dag.OverrideError):
            with dag.apply_overrides(ovs):
                pass


class TestGetOverridesDedup:
    """get_overrides() must not duplicate re-overridden nodes (audit 2026-06 #8)."""

    def setup_method(self):
        dag.reset()

    def test_reoverridden_node_reported_once(self):
        class Market(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

        m = Market()
        with dag.scenario():
            m.Spot.override(1.0)
            m.Spot.override(2.0)
            ovs = dag.get_overrides()

        assert len(ovs.overrides) == 1
        assert ovs.overrides[0].value == 2.0


class TestScenarioHoldsOverriddenObjects:
    """A scenario keeps its overridden objects alive, so an override can
    never land on a new object allocated at a freed object's address
    (review 2026-11-09, correctness 3)."""

    def setup_method(self):
        dag.reset()

    @staticmethod
    def _market():
        class Market(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

        return Market

    def test_overridden_object_survives_until_scenario_exit(self):
        import gc
        import weakref

        m = self._market()()
        ref = weakref.ref(m)

        with dag.scenario():
            m.Spot.override(1.0)
            del m
            gc.collect()
            assert ref() is not None
            assert ref().Spot() == 1.0

        gc.collect()
        assert ref() is None

    def test_new_object_never_inherits_a_dead_objects_override(self):
        import gc

        Market = self._market()
        with dag.scenario():
            m = Market()
            m.Spot.override(1.0)
            address = id(m)
            del m
            gc.collect()

            others = [Market() for _ in range(50)]
            assert address not in {id(other) for other in others}
            assert all(other.Spot() == 100.0 for other in others)

    def test_override_set_scenario_releases_objects_on_exit(self):
        import gc
        import weakref

        m = self._market()()
        ref = weakref.ref(m)
        overrides = dag.OverrideSet()
        overrides.add(m, 'Spot', 1.0)

        ctx = dag.apply_overrides(overrides)
        with ctx:
            assert m.Spot() == 1.0

        del m, overrides
        gc.collect()
        assert ref() is None

    def test_reentering_an_exited_scenario_does_not_restore_old_overrides(self):
        m = self._market()()
        outer, inner = dag.scenario(), dag.scenario()

        with outer:
            m.Spot.override(1.0)
            with inner:
                m.Spot.override(2.0)
                assert m.Spot() == 2.0
            assert m.Spot() == 1.0
        assert m.Spot() == 100.0

        with inner:
            pass
        assert m.Spot() == 100.0


class TestOnGraphScenario:
    """A cell that computes inside its own dag.scenario() must still hear
    about later changes to everything its result depended on (review
    2026-11-09, "found while fixing")."""

    def setup_method(self):
        dag.reset()

    def test_change_reached_only_through_a_recomputed_cell(self):
        calls = []

        class Model(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

            @dag.computed(dag.Input)
            def Vol(self):
                return 2

            @dag.computed
            def Price(self):
                return self.Spot() * self.Vol()

            @dag.computed
            def Up(self):
                calls.append(1)
                with dag.scenario():
                    self.Spot.override(101.0)
                    return self.Price()

        m = Model()
        assert m.Up() == 202.0
        assert m.Up() == 202.0
        assert len(calls) == 1  # still cached

        m.Vol.set(3)
        assert m.Up() == 303.0
        assert m.Price() == 300.0
        assert m.Up() == 303.0
        assert len(calls) == 2

    def test_change_several_levels_below_the_scenario(self):
        class Model(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

            @dag.computed(dag.Input)
            def RateQuote(self):
                return 1.0

            @dag.computed
            def Rate(self):
                return self.RateQuote() * 2

            @dag.computed
            def Forward(self):
                return self.Spot() + self.Rate()

            @dag.computed
            def Price(self):
                return self.Forward() * 10

            @dag.computed
            def Up(self):
                with dag.scenario():
                    self.Spot.override(101.0)
                    return self.Price()

            @dag.computed
            def Report(self):
                return self.Up() + 0.5

        m = Model()
        assert m.Report() == 1030.5

        m.RateQuote.set(2.0)
        assert m.Up() == 1050.0
        assert m.Report() == 1050.5

    def test_value_read_before_the_scenario_and_not_recomputed_in_it(self):
        class Model(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

            @dag.computed(dag.Input)
            def Vol(self):
                return 2

            @dag.computed(dag.Input)
            def Rate(self):
                return 5

            @dag.computed
            def Price(self):
                return self.Spot() * self.Vol()

            @dag.computed
            def Forward(self):
                return self.Spot() + self.Rate()

            @dag.computed
            def Spread(self):
                base = self.Price()
                with dag.scenario():
                    self.Spot.override(101.0)
                    forward = self.Forward()
                return forward - base

        m = Model()
        assert m.Spread() == 106.0 - 200.0

        m.Vol.set(3)
        assert m.Spread() == 106.0 - 300.0

        # Spot's own base value fed Price before the scenario replaced it,
        # so a later override of Spot from outside must reach Spread too.
        with dag.scenario():
            m.Spot.override(50.0)
            assert m.Spread() == 106.0 - 150.0
        assert m.Spread() == 106.0 - 300.0

    def test_dependency_taken_only_under_the_override(self):
        """Price reads Vol only when Spot is high. Recomputing Price outside
        the scenario drops its edge to Vol, but Up's result still depends
        on it."""

        class Model(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

            @dag.computed(dag.Input)
            def Vol(self):
                return 2

            @dag.computed(dag.Input)
            def OtherVol(self):
                return 7

            @dag.computed
            def Price(self):
                if self.Spot() > 100.5:
                    return self.Spot() * self.Vol()
                return self.Spot() * self.OtherVol()

            @dag.computed
            def Up(self):
                with dag.scenario():
                    self.Spot.override(101.0)
                    return self.Price()

        m = Model()
        assert m.Up() == 202.0
        assert m.Price() == 700.0  # recomputed along the other path

        m.Vol.set(3)
        assert m.Up() == 303.0

    def test_overridden_cell_with_inputs_of_its_own(self):
        class Model(dag.Model):
            @dag.computed(dag.Input)
            def Quote(self):
                return 10.0

            @dag.computed(dag.Overridable)
            def Spot(self):
                return self.Quote()

            @dag.computed
            def Price(self):
                return self.Spot() ** 2

            @dag.computed
            def Delta(self):
                spot = self.Spot()
                base = self.Price()
                with dag.scenario():
                    self.Spot.override(spot + 1)
                    up = self.Price()
                return up - base

        m = Model()
        assert m.Delta() == 21.0

        m.Quote.set(20.0)
        assert m.Delta() == 41.0

    def test_cells_the_scenario_did_not_consume_are_not_inherited(self):
        """Overriding Spot invalidates every dependent of Spot, including
        ones the scenario's owner never read. Their inputs must not become
        the owner's dependencies."""
        calls = []

        class Model(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

            @dag.computed(dag.Input)
            def Vol(self):
                return 2

            @dag.computed(dag.Input)
            def Unrelated(self):
                return 1

            @dag.computed
            def Price(self):
                return self.Spot() * self.Vol()

            @dag.computed
            def OtherPrice(self):
                return self.Spot() + self.Unrelated()

            @dag.computed
            def Up(self):
                calls.append(1)
                with dag.scenario():
                    self.Spot.override(101.0)
                    return self.Price()

        m = Model()
        assert m.OtherPrice() == 101.0
        assert m.Up() == 202.0

        m.Unrelated.set(5)
        assert m.Up() == 202.0
        assert len(calls) == 1
        assert m.OtherPrice() == 105.0


class TestGetOverridesNested:
    """get_overrides() reports every override in effect, not only the
    innermost scenario's (review 2026-11-09, correctness 7)."""

    def setup_method(self):
        dag.reset()

    @staticmethod
    def _model():
        class Model(dag.Model):
            @dag.computed(dag.Overridable)
            def A(self):
                return 1

            @dag.computed(dag.Overridable)
            def B(self):
                return 2

            @dag.computed
            def Sum(self):
                return self.A() + self.B()

        return Model()

    @staticmethod
    def _as_dict(override_set):
        return {o.method_name: o.value for o in override_set.overrides}

    def test_outer_scenario_overrides_are_included(self):
        m = self._model()
        with dag.scenario():
            m.A.override(10)
            with dag.scenario():
                m.B.override(20)
                assert self._as_dict(dag.get_overrides()) == {'A': 10, 'B': 20}
            assert self._as_dict(dag.get_overrides()) == {'A': 10}
        assert self._as_dict(dag.get_overrides()) == {}

    def test_node_overridden_in_both_scenarios_reports_the_visible_value(self):
        m = self._model()
        with dag.scenario():
            m.A.override(10)
            with dag.scenario():
                m.A.override(11)
                captured = dag.get_overrides()

        assert len(captured.overrides) == 1
        assert self._as_dict(captured) == {'A': 11}

    def test_captured_nested_state_can_be_reapplied(self):
        m = self._model()
        with dag.scenario():
            m.A.override(10)
            with dag.scenario():
                m.B.override(20)
                expected = m.Sum()
                captured = dag.get_overrides()

        assert m.Sum() == 3
        with dag.apply_overrides(captured):
            assert m.Sum() == expected == 30
        assert m.Sum() == 3
