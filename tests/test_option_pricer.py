"""
Tests for the Black-Scholes option pricing example.

These tests verify:
- Correct pricing calculations
- Greeks calculations
- Put-call parity
- Edge cases (expiry, zero vol, etc.)
- DAG reactivity for option model
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dag
from examples.option_pricer import BlackScholesOption, norm_cdf, norm_pdf


@pytest.fixture(autouse=True)
def reset_dag():
    """Reset DAG state before each test."""
    dag.reset()
    yield
    dag.reset()


class TestNormalDistribution:
    """Test the normal distribution helper functions."""

    def test_norm_cdf_zero(self):
        """N(0) = 0.5"""
        assert abs(norm_cdf(0) - 0.5) < 1e-10

    def test_norm_cdf_large_positive(self):
        """N(large) -> 1"""
        assert norm_cdf(10) > 0.9999

    def test_norm_cdf_large_negative(self):
        """N(-large) -> 0"""
        assert norm_cdf(-10) < 0.0001

    def test_norm_cdf_symmetry(self):
        """N(x) + N(-x) = 1"""
        for x in [0.5, 1.0, 2.0]:
            assert abs(norm_cdf(x) + norm_cdf(-x) - 1.0) < 1e-10

    def test_norm_pdf_zero(self):
        """n(0) = 1/sqrt(2*pi)"""
        expected = 1.0 / math.sqrt(2 * math.pi)
        assert abs(norm_pdf(0) - expected) < 1e-10

    def test_norm_pdf_symmetry(self):
        """n(x) = n(-x)"""
        for x in [0.5, 1.0, 2.0]:
            assert abs(norm_pdf(x) - norm_pdf(-x)) < 1e-10


class TestBlackScholesDefaults:
    """Test default values and basic setup."""

    def test_default_spot(self):
        option = BlackScholesOption()
        assert option.Spot() == 100.0

    def test_default_strike(self):
        option = BlackScholesOption()
        assert option.Strike() == 100.0

    def test_default_volatility(self):
        option = BlackScholesOption()
        assert option.Volatility() == 0.20

    def test_default_rate(self):
        option = BlackScholesOption()
        assert option.RiskFreeRate() == 0.05

    def test_default_dividend(self):
        option = BlackScholesOption()
        assert option.DividendYield() == 0.0

    def test_default_time(self):
        option = BlackScholesOption()
        assert option.TimeToExpiry() == 1.0

    def test_default_is_call(self):
        option = BlackScholesOption()
        assert option.IsCall() is True


class TestBlackScholesPricing:
    """Test option pricing calculations."""

    def test_atm_call_price(self):
        """ATM call with standard params should be around $10.45."""
        option = BlackScholesOption()
        # S=K=100, vol=20%, r=5%, T=1
        price = option.CallPrice()
        assert 10.0 < price < 11.0

    def test_atm_put_price(self):
        """ATM put with standard params should be around $5.57."""
        option = BlackScholesOption()
        price = option.PutPrice()
        assert 5.0 < price < 6.0

    def test_put_call_parity(self):
        """Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)"""
        option = BlackScholesOption()
        S = option.Spot()
        K = option.Strike()
        r = option.RiskFreeRate()
        q = option.DividendYield()
        T = option.TimeToExpiry()

        call = option.CallPrice()
        put = option.PutPrice()

        lhs = call - put
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

        assert abs(lhs - rhs) < 1e-10

    def test_itm_call_higher_than_atm(self):
        """ITM call (S > K) should be worth more than ATM."""
        option = BlackScholesOption()
        atm_price = option.CallPrice()

        option.Spot.set(120)
        itm_price = option.CallPrice()

        assert itm_price > atm_price

    def test_otm_call_lower_than_atm(self):
        """OTM call (S < K) should be worth less than ATM."""
        option = BlackScholesOption()
        atm_price = option.CallPrice()

        option.Spot.set(80)
        otm_price = option.CallPrice()

        assert otm_price < atm_price

    def test_higher_vol_higher_price(self):
        """Higher volatility should increase option price."""
        option = BlackScholesOption()
        low_vol_price = option.CallPrice()

        option.Volatility.set(0.40)
        high_vol_price = option.CallPrice()

        assert high_vol_price > low_vol_price

    def test_longer_expiry_higher_price(self):
        """Longer time to expiry should increase option price (for non-dividend)."""
        option = BlackScholesOption()
        short_price = option.CallPrice()

        option.TimeToExpiry.set(2.0)
        long_price = option.CallPrice()

        assert long_price > short_price

    def test_intrinsic_value_itm_call(self):
        """ITM call intrinsic = S - K."""
        option = BlackScholesOption()
        option.Spot.set(110)
        option.Strike.set(100)

        assert option.IntrinsicValue() == 10.0

    def test_intrinsic_value_otm_call(self):
        """OTM call intrinsic = 0."""
        option = BlackScholesOption()
        option.Spot.set(90)
        option.Strike.set(100)

        assert option.IntrinsicValue() == 0.0

    def test_intrinsic_value_itm_put(self):
        """ITM put intrinsic = K - S."""
        option = BlackScholesOption()
        option.IsCall.set(False)
        option.Spot.set(90)
        option.Strike.set(100)

        assert option.IntrinsicValue() == 10.0

    def test_time_value_positive(self):
        """Time value should be positive for options with time remaining."""
        option = BlackScholesOption()
        assert option.TimeValue() > 0


class TestGreeks:
    """Test Greeks calculations."""

    def test_call_delta_between_0_and_1(self):
        """Call delta should be between 0 and 1."""
        option = BlackScholesOption()
        delta = option.Delta()
        assert 0 < delta < 1

    def test_put_delta_between_minus1_and_0(self):
        """Put delta should be between -1 and 0."""
        option = BlackScholesOption()
        option.IsCall.set(False)
        delta = option.Delta()
        assert -1 < delta < 0

    def test_atm_call_delta_around_half(self):
        """ATM call delta should be around 0.5-0.6."""
        option = BlackScholesOption()
        delta = option.Delta()
        assert 0.5 < delta < 0.7

    def test_itm_call_delta_near_one(self):
        """Deep ITM call delta should approach 1."""
        option = BlackScholesOption()
        option.Spot.set(150)
        delta = option.Delta()
        assert delta > 0.9

    def test_otm_call_delta_near_zero(self):
        """Deep OTM call delta should approach 0."""
        option = BlackScholesOption()
        option.Spot.set(50)
        delta = option.Delta()
        assert delta < 0.1

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        option = BlackScholesOption()
        assert option.Gamma() > 0

    def test_gamma_highest_atm(self):
        """Gamma should be highest for ATM options."""
        option = BlackScholesOption()
        atm_gamma = option.Gamma()

        option.Spot.set(80)
        otm_gamma = option.Gamma()

        option.Spot.set(120)
        itm_gamma = option.Gamma()

        assert atm_gamma > otm_gamma
        assert atm_gamma > itm_gamma

    def test_vega_positive(self):
        """Vega should always be positive."""
        option = BlackScholesOption()
        assert option.Vega() > 0

    def test_theta_negative_for_long(self):
        """Theta should be negative for long options (time decay)."""
        option = BlackScholesOption()
        assert option.Theta() < 0

    def test_call_rho_positive(self):
        """Call rho should be positive (higher rates help calls)."""
        option = BlackScholesOption()
        assert option.Rho() > 0

    def test_put_rho_negative(self):
        """Put rho should be negative (higher rates hurt puts)."""
        option = BlackScholesOption()
        option.IsCall.set(False)
        assert option.Rho() < 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_at_expiry_itm_call(self):
        """At expiry, ITM call should equal intrinsic value."""
        option = BlackScholesOption()
        option.TimeToExpiry.set(0)
        option.Spot.set(110)
        option.Strike.set(100)

        assert option.CallPrice() == 10.0

    def test_at_expiry_otm_call(self):
        """At expiry, OTM call should be worth 0."""
        option = BlackScholesOption()
        option.TimeToExpiry.set(0)
        option.Spot.set(90)
        option.Strike.set(100)

        assert option.CallPrice() == 0.0

    def test_at_expiry_itm_put(self):
        """At expiry, ITM put should equal intrinsic value."""
        option = BlackScholesOption()
        option.TimeToExpiry.set(0)
        option.Spot.set(90)
        option.Strike.set(100)

        assert option.PutPrice() == 10.0

    def test_at_expiry_delta_call_itm(self):
        """At expiry, ITM call delta = 1."""
        option = BlackScholesOption()
        option.TimeToExpiry.set(0)
        option.Spot.set(110)

        assert option.Delta() == 1.0

    def test_at_expiry_delta_call_otm(self):
        """At expiry, OTM call delta = 0."""
        option = BlackScholesOption()
        option.TimeToExpiry.set(0)
        option.Spot.set(90)

        assert option.Delta() == 0.0

    def test_zero_vol_itm_call(self):
        """With zero vol, ITM call = discounted intrinsic."""
        option = BlackScholesOption()
        option.Volatility.set(0.0001)  # Near zero to avoid div by zero
        option.Spot.set(110)
        option.Strike.set(100)

        # Should be close to S - K*e^(-rT)
        expected = 110 - 100 * math.exp(-0.05)
        assert abs(option.CallPrice() - expected) < 0.5


class TestMoneyness:
    """Test moneyness classification."""

    def test_atm_call(self):
        option = BlackScholesOption()
        option.Spot.set(100)
        option.Strike.set(100)
        assert option.Moneyness() == "ATM"

    def test_itm_call(self):
        option = BlackScholesOption()
        option.Spot.set(110)
        option.Strike.set(100)
        assert option.Moneyness() == "ITM"

    def test_otm_call(self):
        option = BlackScholesOption()
        option.Spot.set(90)
        option.Strike.set(100)
        assert option.Moneyness() == "OTM"

    def test_itm_put(self):
        option = BlackScholesOption()
        option.IsCall.set(False)
        option.Spot.set(90)
        option.Strike.set(100)
        assert option.Moneyness() == "ITM"

    def test_otm_put(self):
        option = BlackScholesOption()
        option.IsCall.set(False)
        option.Spot.set(110)
        option.Strike.set(100)
        assert option.Moneyness() == "OTM"


class TestDAGReactivity:
    """Test DAG-specific reactivity behavior."""

    def test_price_updates_on_spot_change(self):
        """Price should update when spot changes."""
        option = BlackScholesOption()
        initial_price = option.Price()

        option.Spot.set(110)
        new_price = option.Price()

        assert new_price != initial_price

    def test_greeks_update_on_vol_change(self):
        """Greeks should update when volatility changes."""
        option = BlackScholesOption()
        initial_vega = option.Vega()

        option.Volatility.set(0.30)
        new_vega = option.Vega()

        assert new_vega != initial_vega

    def test_subscription_fires_on_input_change(self):
        """Subscriptions should fire when inputs change."""
        option = BlackScholesOption()

        callback_count = [0]
        callback_ref = lambda node: callback_count.__setitem__(0, callback_count[0] + 1)

        # Watch Price for changes
        option.Price.watch(callback_ref)

        # Trigger initial evaluation
        option.Price()

        # Change spot
        option.Spot.set(110)

        # Flush pending notifications
        dag.flush()

        assert callback_count[0] >= 1

    def test_multiple_outputs_update(self):
        """Multiple dependent computed functions should all update."""
        option = BlackScholesOption()

        # Get initial values
        initial_call = option.CallPrice()
        initial_put = option.PutPrice()
        initial_delta = option.Delta()

        # Change spot
        option.Spot.set(120)

        # All should have changed
        assert option.CallPrice() != initial_call
        assert option.PutPrice() != initial_put
        assert option.Delta() != initial_delta

    def test_independent_options(self):
        """Multiple option instances should be independent."""
        option1 = BlackScholesOption()
        option2 = BlackScholesOption()

        option1.Spot.set(110)

        assert option1.Spot() == 110
        assert option2.Spot() == 100  # Unchanged


class TestDividends:
    """Test dividend yield effects."""

    def test_dividend_reduces_call_price(self):
        """Dividend yield should reduce call price."""
        option = BlackScholesOption()
        no_div_price = option.CallPrice()

        option.DividendYield.set(0.03)
        with_div_price = option.CallPrice()

        assert with_div_price < no_div_price

    def test_dividend_increases_put_price(self):
        """Dividend yield should increase put price."""
        option = BlackScholesOption()
        no_div_price = option.PutPrice()

        option.DividendYield.set(0.03)
        with_div_price = option.PutPrice()

        assert with_div_price > no_div_price

    def test_put_call_parity_with_dividend(self):
        """Put-call parity should hold with dividends."""
        option = BlackScholesOption()
        option.DividendYield.set(0.02)

        S = option.Spot()
        K = option.Strike()
        r = option.RiskFreeRate()
        q = option.DividendYield()
        T = option.TimeToExpiry()

        call = option.CallPrice()
        put = option.PutPrice()

        lhs = call - put
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

        assert abs(lhs - rhs) < 1e-10
