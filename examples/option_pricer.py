#!/usr/bin/env python3
"""
Option Pricer Example - DAG UI Binding Demo for Finance

A Black-Scholes option pricing calculator that demonstrates:
- Reactive computation of option prices and Greeks
- Automatic propagation of parameter changes
- Real-time updates as market data changes

Run with: python examples/option_pricer.py
"""

import math
import tkinter as tk
from tkinter import ttk
import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dag
from dag.ui import (
    DagApp,
    BoundEntry,
    BoundLabel,
    BoundScale,
    CellInput,
    CellDisplay,
    CellSlider,
)


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class BlackScholesOption(dag.Model):
    """
    Black-Scholes option pricing model with Greeks.

    This demonstrates how DAG computed functions naturally model financial
    derivatives where outputs (prices, Greeks) depend on market inputs
    (spot, volatility, rates) and contract parameters (strike, expiry).
    """

    # ==================== Market Inputs ====================

    @dag.computed(dag.Input)
    def Spot(self) -> float:
        """Current underlying price."""
        return 100.0

    @dag.computed(dag.Input)
    def Volatility(self) -> float:
        """Annualized volatility (as decimal, e.g., 0.20 = 20%)."""
        return 0.20

    @dag.computed(dag.Input)
    def RiskFreeRate(self) -> float:
        """Risk-free interest rate (as decimal)."""
        return 0.05

    @dag.computed(dag.Input)
    def DividendYield(self) -> float:
        """Continuous dividend yield (as decimal)."""
        return 0.0

    # ==================== Contract Parameters ====================

    @dag.computed(dag.Input)
    def Strike(self) -> float:
        """Option strike price."""
        return 100.0

    @dag.computed(dag.Input)
    def TimeToExpiry(self) -> float:
        """Time to expiration in years."""
        return 1.0

    @dag.computed(dag.Input)
    def IsCall(self) -> bool:
        """True for call option, False for put option."""
        return True

    # ==================== Intermediate Calculations ====================

    @dag.computed
    def D1(self) -> float:
        """d1 parameter in Black-Scholes formula."""
        S = self.Spot()
        K = self.Strike()
        T = self.TimeToExpiry()
        r = self.RiskFreeRate()
        q = self.DividendYield()
        sigma = self.Volatility()

        if T <= 0 or sigma <= 0:
            return 0.0

        numerator = math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T
        denominator = sigma * math.sqrt(T)
        return numerator / denominator

    @dag.computed
    def D2(self) -> float:
        """d2 parameter in Black-Scholes formula."""
        T = self.TimeToExpiry()
        sigma = self.Volatility()

        if T <= 0:
            return 0.0

        return self.D1() - sigma * math.sqrt(T)

    # ==================== Option Price ====================

    @dag.computed
    def CallPrice(self) -> float:
        """Black-Scholes call option price."""
        S = self.Spot()
        K = self.Strike()
        T = self.TimeToExpiry()
        r = self.RiskFreeRate()
        q = self.DividendYield()
        d1 = self.D1()
        d2 = self.D2()

        if T <= 0:
            # At expiry, intrinsic value
            return max(S - K, 0)

        discount_factor = math.exp(-r * T)
        dividend_factor = math.exp(-q * T)

        return S * dividend_factor * norm_cdf(d1) - K * discount_factor * norm_cdf(d2)

    @dag.computed
    def PutPrice(self) -> float:
        """Black-Scholes put option price (via put-call parity)."""
        S = self.Spot()
        K = self.Strike()
        T = self.TimeToExpiry()
        r = self.RiskFreeRate()
        q = self.DividendYield()

        if T <= 0:
            return max(K - S, 0)

        # Put-Call Parity: P = C - S*e^(-qT) + K*e^(-rT)
        call = self.CallPrice()
        return call - S * math.exp(-q * T) + K * math.exp(-r * T)

    @dag.computed
    def Price(self) -> float:
        """Current option price based on IsCall flag."""
        if self.IsCall():
            return self.CallPrice()
        return self.PutPrice()

    @dag.computed
    def IntrinsicValue(self) -> float:
        """Intrinsic value of the option."""
        S = self.Spot()
        K = self.Strike()
        if self.IsCall():
            return max(S - K, 0)
        return max(K - S, 0)

    @dag.computed
    def TimeValue(self) -> float:
        """Time value (extrinsic value) of the option."""
        return self.Price() - self.IntrinsicValue()

    # ==================== Greeks ====================

    @dag.computed
    def Delta(self) -> float:
        """
        Delta: Rate of change of option price with respect to spot.

        For calls: e^(-qT) * N(d1)
        For puts: e^(-qT) * (N(d1) - 1)
        """
        T = self.TimeToExpiry()
        q = self.DividendYield()
        d1 = self.D1()

        if T <= 0:
            # At expiry
            S, K = self.Spot(), self.Strike()
            if self.IsCall():
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0

        dividend_factor = math.exp(-q * T)

        if self.IsCall():
            return dividend_factor * norm_cdf(d1)
        return dividend_factor * (norm_cdf(d1) - 1)

    @dag.computed
    def Gamma(self) -> float:
        """
        Gamma: Rate of change of delta with respect to spot.

        Same for calls and puts: e^(-qT) * n(d1) / (S * sigma * sqrt(T))
        """
        S = self.Spot()
        T = self.TimeToExpiry()
        sigma = self.Volatility()
        q = self.DividendYield()
        d1 = self.D1()

        if T <= 0 or sigma <= 0:
            return 0.0

        dividend_factor = math.exp(-q * T)
        return dividend_factor * norm_pdf(d1) / (S * sigma * math.sqrt(T))

    @dag.computed
    def Vega(self) -> float:
        """
        Vega: Rate of change of option price with respect to volatility.

        Same for calls and puts: S * e^(-qT) * n(d1) * sqrt(T)
        Returns value per 1% move (divide by 100).
        """
        S = self.Spot()
        T = self.TimeToExpiry()
        q = self.DividendYield()
        d1 = self.D1()

        if T <= 0:
            return 0.0

        dividend_factor = math.exp(-q * T)
        # Per 1% vol move
        return S * dividend_factor * norm_pdf(d1) * math.sqrt(T) / 100

    @dag.computed
    def Theta(self) -> float:
        """
        Theta: Rate of change of option price with respect to time.

        Returns daily theta (divide annual by 365).
        Negative for long options (time decay).
        """
        S = self.Spot()
        K = self.Strike()
        T = self.TimeToExpiry()
        r = self.RiskFreeRate()
        q = self.DividendYield()
        sigma = self.Volatility()
        d1 = self.D1()
        d2 = self.D2()

        if T <= 0:
            return 0.0

        dividend_factor = math.exp(-q * T)
        discount_factor = math.exp(-r * T)
        sqrt_T = math.sqrt(T)

        # First term (same for call and put)
        term1 = -S * dividend_factor * norm_pdf(d1) * sigma / (2 * sqrt_T)

        if self.IsCall():
            term2 = -r * K * discount_factor * norm_cdf(d2)
            term3 = q * S * dividend_factor * norm_cdf(d1)
        else:
            term2 = r * K * discount_factor * norm_cdf(-d2)
            term3 = -q * S * dividend_factor * norm_cdf(-d1)

        # Return daily theta
        return (term1 + term2 + term3) / 365

    @dag.computed
    def Rho(self) -> float:
        """
        Rho: Rate of change of option price with respect to interest rate.

        Returns value per 1% rate move (divide by 100).
        """
        K = self.Strike()
        T = self.TimeToExpiry()
        r = self.RiskFreeRate()
        d2 = self.D2()

        if T <= 0:
            return 0.0

        discount_factor = math.exp(-r * T)

        if self.IsCall():
            return K * T * discount_factor * norm_cdf(d2) / 100
        return -K * T * discount_factor * norm_cdf(-d2) / 100

    # ==================== Moneyness ====================

    @dag.computed
    def Moneyness(self) -> str:
        """Describe the moneyness of the option."""
        S = self.Spot()
        K = self.Strike()
        ratio = S / K

        if self.IsCall():
            if ratio > 1.05:
                return "ITM"
            elif ratio < 0.95:
                return "OTM"
            return "ATM"
        else:
            if ratio < 0.95:
                return "ITM"
            elif ratio > 1.05:
                return "OTM"
            return "ATM"


def create_input_section(parent: tk.Frame, option: BlackScholesOption, app: DagApp) -> tk.Frame:
    """Create the market inputs section of the UI."""
    frame = ttk.LabelFrame(parent, text="Market Data", padding=10)

    row = 0

    # Spot Price
    ttk.Label(frame, text="Spot Price:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    spot_entry = BoundEntry(frame, cell=option.Spot, app=app, width=12)
    spot_entry.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    # Volatility (as percentage)
    ttk.Label(frame, text="Volatility (%):").grid(row=row, column=0, sticky='e', padx=5, pady=3)

    # Custom formatter/parser for percentage display
    def vol_formatter(v):
        return f"{v * 100:.1f}"

    def vol_parser(s):
        return float(s) / 100

    vol_entry = BoundEntry(frame, cell=option.Volatility, app=app, width=12,
                          formatter=vol_formatter, parser=vol_parser)
    vol_entry.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    # Risk-Free Rate (as percentage)
    ttk.Label(frame, text="Risk-Free Rate (%):").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    rate_entry = BoundEntry(frame, cell=option.RiskFreeRate, app=app, width=12,
                           formatter=vol_formatter, parser=vol_parser)
    rate_entry.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    # Dividend Yield (as percentage)
    ttk.Label(frame, text="Dividend Yield (%):").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    div_entry = BoundEntry(frame, cell=option.DividendYield, app=app, width=12,
                          formatter=vol_formatter, parser=vol_parser)
    div_entry.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    return frame


def create_contract_section(parent: tk.Frame, option: BlackScholesOption, app: DagApp) -> tk.Frame:
    """Create the contract parameters section of the UI."""
    frame = ttk.LabelFrame(parent, text="Contract", padding=10)

    row = 0

    # Strike Price
    ttk.Label(frame, text="Strike Price:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    strike_entry = BoundEntry(frame, cell=option.Strike, app=app, width=12)
    strike_entry.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    # Time to Expiry
    ttk.Label(frame, text="Time to Expiry (years):").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    time_entry = BoundEntry(frame, cell=option.TimeToExpiry, app=app, width=12)
    time_entry.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    # Call/Put Toggle
    ttk.Label(frame, text="Option Type:").grid(row=row, column=0, sticky='e', padx=5, pady=3)

    type_frame = ttk.Frame(frame)
    type_frame.grid(row=row, column=1, sticky='w', padx=5, pady=3)

    type_var = tk.StringVar(value="Call")

    def on_type_change():
        is_call = type_var.get() == "Call"
        option.IsCall.set(is_call)
        app.schedule_update()

    call_rb = ttk.Radiobutton(type_frame, text="Call", variable=type_var, value="Call",
                              command=on_type_change)
    call_rb.pack(side=tk.LEFT)

    put_rb = ttk.Radiobutton(type_frame, text="Put", variable=type_var, value="Put",
                             command=on_type_change)
    put_rb.pack(side=tk.LEFT, padx=10)
    row += 1

    # Moneyness display
    ttk.Label(frame, text="Moneyness:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    moneyness_label = BoundLabel(frame, cell=option.Moneyness, app=app,
                                 width=12, anchor='w')
    moneyness_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    return frame


def create_pricing_section(parent: tk.Frame, option: BlackScholesOption, app: DagApp) -> tk.Frame:
    """Create the pricing output section of the UI."""
    frame = ttk.LabelFrame(parent, text="Pricing", padding=10)

    def price_formatter(v):
        return f"${v:.4f}"

    row = 0

    # Option Price
    ttk.Label(frame, text="Option Price:", font=('Helvetica', 10, 'bold')).grid(
        row=row, column=0, sticky='e', padx=5, pady=3)
    price_label = BoundLabel(frame, cell=option.Price, app=app,
                            formatter=price_formatter,
                            width=14, anchor='w', font=('Helvetica', 10, 'bold'))
    price_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    # Intrinsic Value
    ttk.Label(frame, text="Intrinsic Value:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    intrinsic_label = BoundLabel(frame, cell=option.IntrinsicValue, app=app,
                                formatter=price_formatter, width=14, anchor='w')
    intrinsic_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    # Time Value
    ttk.Label(frame, text="Time Value:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    time_val_label = BoundLabel(frame, cell=option.TimeValue, app=app,
                               formatter=price_formatter, width=14, anchor='w')
    time_val_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                   sticky='ew', pady=5)
    row += 1

    # Call and Put prices for comparison
    ttk.Label(frame, text="Call Price:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    call_label = BoundLabel(frame, cell=option.CallPrice, app=app,
                           formatter=price_formatter, width=14, anchor='w')
    call_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    ttk.Label(frame, text="Put Price:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    put_label = BoundLabel(frame, cell=option.PutPrice, app=app,
                          formatter=price_formatter, width=14, anchor='w')
    put_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)
    row += 1

    return frame


def create_greeks_section(parent: tk.Frame, option: BlackScholesOption, app: DagApp) -> tk.Frame:
    """Create the Greeks output section of the UI."""
    frame = ttk.LabelFrame(parent, text="Greeks", padding=10)

    row = 0

    # Delta
    ttk.Label(frame, text="Delta:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    delta_label = BoundLabel(frame, cell=option.Delta, app=app,
                            formatter=lambda v: f"{v:.4f}",
                            width=12, anchor='w')
    delta_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)

    ttk.Label(frame, text="(per $1 spot move)", font=('Helvetica', 8)).grid(
        row=row, column=2, sticky='w', padx=5)
    row += 1

    # Gamma
    ttk.Label(frame, text="Gamma:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    gamma_label = BoundLabel(frame, cell=option.Gamma, app=app,
                            formatter=lambda v: f"{v:.6f}",
                            width=12, anchor='w')
    gamma_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)

    ttk.Label(frame, text="(delta change per $1)", font=('Helvetica', 8)).grid(
        row=row, column=2, sticky='w', padx=5)
    row += 1

    # Vega
    ttk.Label(frame, text="Vega:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    vega_label = BoundLabel(frame, cell=option.Vega, app=app,
                           formatter=lambda v: f"${v:.4f}",
                           width=12, anchor='w')
    vega_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)

    ttk.Label(frame, text="(per 1% vol move)", font=('Helvetica', 8)).grid(
        row=row, column=2, sticky='w', padx=5)
    row += 1

    # Theta
    ttk.Label(frame, text="Theta:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    theta_label = BoundLabel(frame, cell=option.Theta, app=app,
                            formatter=lambda v: f"${v:.4f}",
                            width=12, anchor='w')
    theta_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)

    ttk.Label(frame, text="(daily decay)", font=('Helvetica', 8)).grid(
        row=row, column=2, sticky='w', padx=5)
    row += 1

    # Rho
    ttk.Label(frame, text="Rho:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
    rho_label = BoundLabel(frame, cell=option.Rho, app=app,
                          formatter=lambda v: f"${v:.4f}",
                          width=12, anchor='w')
    rho_label.grid(row=row, column=1, sticky='w', padx=5, pady=3)

    ttk.Label(frame, text="(per 1% rate move)", font=('Helvetica', 8)).grid(
        row=row, column=2, sticky='w', padx=5)
    row += 1

    return frame


def create_vol_slider(parent: tk.Frame, option: BlackScholesOption, app: DagApp) -> tk.Frame:
    """Create a volatility slider for interactive adjustment."""
    frame = ttk.LabelFrame(parent, text="Volatility Slider", padding=10)

    ttk.Label(frame, text="5%").pack(side=tk.LEFT, padx=5)

    # Custom formatter/parser for the slider
    def vol_formatter(v):
        return f"{v * 100:.1f}"

    def vol_parser(s):
        return float(s) / 100

    vol_scale = BoundScale(
        frame,
        cell=option.Volatility,
        app=app,
        from_=0.05,
        to=1.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        length=300,
        formatter=vol_formatter,
        parser=vol_parser,
    )
    vol_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    ttk.Label(frame, text="100%").pack(side=tk.LEFT, padx=5)

    return frame


def main():
    # Reset DAG state
    dag.reset()

    # Create the application
    app = DagApp("Black-Scholes Option Pricer", width=700, height=580)

    # Create the option model
    option = BlackScholesOption()

    # Configure main window
    app.root.columnconfigure(0, weight=1)
    app.root.columnconfigure(1, weight=1)

    # Header
    header = ttk.Label(
        app.root,
        text="Black-Scholes Option Pricer",
        font=('Helvetica', 14, 'bold')
    )
    header.grid(row=0, column=0, columnspan=2, pady=10)

    # Left column: Inputs
    left_frame = ttk.Frame(app.root)
    left_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)

    market_frame = create_input_section(left_frame, option, app)
    market_frame.pack(fill=tk.X, pady=5)

    contract_frame = create_contract_section(left_frame, option, app)
    contract_frame.pack(fill=tk.X, pady=5)

    # Right column: Outputs
    right_frame = ttk.Frame(app.root)
    right_frame.grid(row=1, column=1, sticky='nsew', padx=10, pady=5)

    pricing_frame = create_pricing_section(right_frame, option, app)
    pricing_frame.pack(fill=tk.X, pady=5)

    greeks_frame = create_greeks_section(right_frame, option, app)
    greeks_frame.pack(fill=tk.X, pady=5)

    # Volatility slider (full width)
    vol_slider_frame = create_vol_slider(app.root, option, app)
    vol_slider_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=10)

    # Instructions
    instructions = ttk.Label(
        app.root,
        text="Edit any parameter and press Tab/Enter to update. Use the slider to adjust volatility.",
        font=('Helvetica', 9, 'italic'),
    )
    instructions.grid(row=3, column=0, columnspan=2, pady=5)

    # Footer with formulas
    formula_text = "Black-Scholes: C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)"
    formula_label = ttk.Label(
        app.root,
        text=formula_text,
        font=('Courier', 8),
    )
    formula_label.grid(row=4, column=0, columnspan=2, pady=5)

    # Run the application
    print("Starting Black-Scholes Option Pricer...")
    print("Adjust market data and contract parameters to see real-time pricing updates.")
    print("\nDefault: ATM call option with S=K=100, vol=20%, r=5%, T=1 year")
    app.run()


if __name__ == '__main__':
    main()
