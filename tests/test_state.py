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

        with dag.branch() as l1:
            opt.Strike.override(1.4)
            assert opt.Strike() == 1.4

        # After branch exits
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
