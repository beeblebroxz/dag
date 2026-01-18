#!/usr/bin/env python3
"""
Option Pricer Web Application - DAG UI Demo for Finance

A modern web-based Black-Scholes option pricing calculator.
Demonstrates how the DAG framework can power a reactive web UI.

Run with: python examples/option_pricer_web.py
Then open: http://localhost:8000

No external dependencies required - uses Python's built-in http.server.
"""

import math
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dag


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class BlackScholesOption(dag.Model):
    """Black-Scholes option pricing model with Greeks."""

    @dag.computed(dag.Input)
    def Spot(self) -> float:
        return 100.0

    @dag.computed(dag.Input)
    def Volatility(self) -> float:
        return 0.20

    @dag.computed(dag.Input)
    def RiskFreeRate(self) -> float:
        return 0.05

    @dag.computed(dag.Input)
    def DividendYield(self) -> float:
        return 0.0

    @dag.computed(dag.Input)
    def Strike(self) -> float:
        return 100.0

    @dag.computed(dag.Input)
    def TimeToExpiry(self) -> float:
        return 1.0

    @dag.computed(dag.Input)
    def IsCall(self) -> bool:
        return True

    @dag.computed
    def D1(self) -> float:
        S, K = self.Spot(), self.Strike()
        T, r, q = self.TimeToExpiry(), self.RiskFreeRate(), self.DividendYield()
        sigma = self.Volatility()
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

    @dag.computed
    def D2(self) -> float:
        T, sigma = self.TimeToExpiry(), self.Volatility()
        if T <= 0:
            return 0.0
        return self.D1() - sigma * math.sqrt(T)

    @dag.computed
    def CallPrice(self) -> float:
        S, K = self.Spot(), self.Strike()
        T, r, q = self.TimeToExpiry(), self.RiskFreeRate(), self.DividendYield()
        if T <= 0:
            return max(S - K, 0)
        d1, d2 = self.D1(), self.D2()
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

    @dag.computed
    def PutPrice(self) -> float:
        S, K = self.Spot(), self.Strike()
        T, r, q = self.TimeToExpiry(), self.RiskFreeRate(), self.DividendYield()
        if T <= 0:
            return max(K - S, 0)
        call = self.CallPrice()
        return call - S * math.exp(-q * T) + K * math.exp(-r * T)

    @dag.computed
    def Price(self) -> float:
        return self.CallPrice() if self.IsCall() else self.PutPrice()

    @dag.computed
    def IntrinsicValue(self) -> float:
        S, K = self.Spot(), self.Strike()
        return max(S - K, 0) if self.IsCall() else max(K - S, 0)

    @dag.computed
    def TimeValue(self) -> float:
        return self.Price() - self.IntrinsicValue()

    @dag.computed
    def Delta(self) -> float:
        T, q, d1 = self.TimeToExpiry(), self.DividendYield(), self.D1()
        if T <= 0:
            S, K = self.Spot(), self.Strike()
            if self.IsCall():
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        df = math.exp(-q * T)
        return df * norm_cdf(d1) if self.IsCall() else df * (norm_cdf(d1) - 1)

    @dag.computed
    def Gamma(self) -> float:
        S, T = self.Spot(), self.TimeToExpiry()
        sigma, q, d1 = self.Volatility(), self.DividendYield(), self.D1()
        if T <= 0 or sigma <= 0:
            return 0.0
        return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))

    @dag.computed
    def Vega(self) -> float:
        S, T, q, d1 = self.Spot(), self.TimeToExpiry(), self.DividendYield(), self.D1()
        if T <= 0:
            return 0.0
        return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) / 100

    @dag.computed
    def Theta(self) -> float:
        S, K = self.Spot(), self.Strike()
        T, r, q = self.TimeToExpiry(), self.RiskFreeRate(), self.DividendYield()
        sigma, d1, d2 = self.Volatility(), self.D1(), self.D2()
        if T <= 0:
            return 0.0
        df, disc = math.exp(-q * T), math.exp(-r * T)
        term1 = -S * df * norm_pdf(d1) * sigma / (2 * math.sqrt(T))
        if self.IsCall():
            term2 = -r * K * disc * norm_cdf(d2)
            term3 = q * S * df * norm_cdf(d1)
        else:
            term2 = r * K * disc * norm_cdf(-d2)
            term3 = -q * S * df * norm_cdf(-d1)
        return (term1 + term2 + term3) / 365

    @dag.computed
    def Rho(self) -> float:
        K, T, r, d2 = self.Strike(), self.TimeToExpiry(), self.RiskFreeRate(), self.D2()
        if T <= 0:
            return 0.0
        disc = math.exp(-r * T)
        return K * T * disc * norm_cdf(d2) / 100 if self.IsCall() else -K * T * disc * norm_cdf(-d2) / 100

    @dag.computed
    def Moneyness(self) -> str:
        ratio = self.Spot() / self.Strike()
        if self.IsCall():
            return "ITM" if ratio > 1.05 else ("OTM" if ratio < 0.95 else "ATM")
        return "ITM" if ratio < 0.95 else ("OTM" if ratio > 1.05 else "ATM")

    @dag.computed
    def Breakeven(self) -> float:
        """Breakeven price at expiry."""
        if self.IsCall():
            return self.Strike() + self.Price()
        return self.Strike() - self.Price()

    def to_dict(self) -> dict:
        """Export all values as a dictionary."""
        return {
            "inputs": {
                "spot": self.Spot(),
                "strike": self.Strike(),
                "volatility": self.Volatility() * 100,
                "riskFreeRate": self.RiskFreeRate() * 100,
                "dividendYield": self.DividendYield() * 100,
                "timeToExpiry": self.TimeToExpiry(),
                "isCall": self.IsCall(),
            },
            "pricing": {
                "price": self.Price(),
                "callPrice": self.CallPrice(),
                "putPrice": self.PutPrice(),
                "intrinsicValue": self.IntrinsicValue(),
                "timeValue": self.TimeValue(),
                "breakeven": self.Breakeven(),
            },
            "greeks": {
                "delta": self.Delta(),
                "gamma": self.Gamma(),
                "vega": self.Vega(),
                "theta": self.Theta(),
                "rho": self.Rho(),
            },
            "analysis": {
                "moneyness": self.Moneyness(),
                "d1": self.D1(),
                "d2": self.D2(),
            }
        }


# HTML Template with modern CSS
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Black-Scholes Option Pricer</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-purple: #a371f7;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        header p {
            color: var(--text-secondary);
            margin-top: 8px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }

        @media (max-width: 900px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }

        .card-header h2 {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .card-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }

        .icon-market { background: rgba(88, 166, 255, 0.15); }
        .icon-contract { background: rgba(163, 113, 247, 0.15); }
        .icon-price { background: rgba(63, 185, 80, 0.15); }
        .icon-greeks { background: rgba(210, 153, 34, 0.15); }

        .form-group {
            margin-bottom: 16px;
        }

        .form-group label {
            display: block;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        .form-group input[type="number"],
        .form-group input[type="range"] {
            width: 100%;
        }

        input[type="number"] {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            color: var(--text-primary);
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input[type="number"]:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
        }

        input[type="range"] {
            -webkit-appearance: none;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            margin-top: 8px;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--accent-blue);
            border-radius: 50%;
            cursor: pointer;
            transition: transform 0.1s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.1);
        }

        .toggle-group {
            display: flex;
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 4px;
        }

        .toggle-btn {
            flex: 1;
            padding: 10px 20px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
        }

        .toggle-btn.active {
            background: var(--accent-blue);
            color: white;
        }

        .toggle-btn:hover:not(.active) {
            color: var(--text-primary);
        }

        .output-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .output-row:last-child {
            border-bottom: none;
        }

        .output-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .output-value {
            font-size: 1.1rem;
            font-weight: 600;
            font-family: 'SF Mono', 'Consolas', monospace;
        }

        .price-main {
            font-size: 2rem;
            color: var(--accent-green);
        }

        .value-positive { color: var(--accent-green); }
        .value-negative { color: var(--accent-red); }
        .value-neutral { color: var(--text-primary); }

        .moneyness-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .moneyness-itm { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); }
        .moneyness-atm { background: rgba(210, 153, 34, 0.15); color: var(--accent-yellow); }
        .moneyness-otm { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); }

        .greek-bar {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .greek-visual {
            flex: 1;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
        }

        .greek-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }

        .greek-fill.positive { background: var(--accent-green); }
        .greek-fill.negative { background: var(--accent-red); }

        .slider-container {
            margin-top: 30px;
            padding: 24px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .slider-value {
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 1.2rem;
            color: var(--accent-blue);
        }

        .scenario-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
            margin-top: 24px;
        }

        .scenario-item {
            text-align: center;
            padding: 12px 8px;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }

        .scenario-spot {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }

        .scenario-pnl {
            font-size: 1rem;
            font-weight: 600;
            font-family: 'SF Mono', 'Consolas', monospace;
        }

        footer {
            text-align: center;
            padding: 30px 0;
            margin-top: 30px;
            border-top: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        footer code {
            background: var(--bg-tertiary);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'SF Mono', 'Consolas', monospace;
        }

        .hint {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Black-Scholes Option Pricer</h1>
            <p>Real-time option pricing powered by the DAG framework</p>
        </header>

        <div class="grid">
            <!-- Market Data -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon icon-market">📈</div>
                    <h2>Market Data</h2>
                </div>

                <div class="form-group">
                    <label>Spot Price ($)</label>
                    <input type="number" id="spot" value="100" step="0.5" min="0.01">
                </div>

                <div class="form-group">
                    <label>Volatility (%)</label>
                    <input type="number" id="volatility" value="20" step="0.5" min="0.1" max="200">
                    <input type="range" id="volatilitySlider" value="20" min="5" max="100" step="0.5">
                </div>

                <div class="form-group">
                    <label>Risk-Free Rate (%)</label>
                    <input type="number" id="riskFreeRate" value="5" step="0.1" min="0" max="50">
                </div>

                <div class="form-group">
                    <label>Dividend Yield (%)</label>
                    <input type="number" id="dividendYield" value="0" step="0.1" min="0" max="50">
                </div>
            </div>

            <!-- Contract Parameters -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon icon-contract">📋</div>
                    <h2>Contract</h2>
                </div>

                <div class="form-group">
                    <label>Strike Price ($)</label>
                    <input type="number" id="strike" value="100" step="0.5" min="0.01">
                </div>

                <div class="form-group">
                    <label>Time to Expiry (years)</label>
                    <input type="number" id="timeToExpiry" value="1" step="0.05" min="0.01" max="10">
                    <p class="hint" id="daysHint">≈ 365 days</p>
                </div>

                <div class="form-group">
                    <label>Option Type</label>
                    <div class="toggle-group">
                        <button class="toggle-btn active" id="btnCall" onclick="setOptionType(true)">Call</button>
                        <button class="toggle-btn" id="btnPut" onclick="setOptionType(false)">Put</button>
                    </div>
                </div>

                <div class="output-row" style="margin-top: 20px;">
                    <span class="output-label">Moneyness</span>
                    <span class="moneyness-badge moneyness-atm" id="moneyness">ATM</span>
                </div>
            </div>

            <!-- Pricing -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon icon-price">💰</div>
                    <h2>Pricing</h2>
                </div>

                <div class="output-row">
                    <span class="output-label">Option Price</span>
                    <span class="output-value price-main" id="price">$0.00</span>
                </div>

                <div class="output-row">
                    <span class="output-label">Intrinsic Value</span>
                    <span class="output-value" id="intrinsic">$0.00</span>
                </div>

                <div class="output-row">
                    <span class="output-label">Time Value</span>
                    <span class="output-value" id="timeValue">$0.00</span>
                </div>

                <div class="output-row">
                    <span class="output-label">Breakeven at Expiry</span>
                    <span class="output-value" id="breakeven">$0.00</span>
                </div>

                <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                    <div class="output-row">
                        <span class="output-label">Call Price</span>
                        <span class="output-value" id="callPrice">$0.00</span>
                    </div>
                    <div class="output-row">
                        <span class="output-label">Put Price</span>
                        <span class="output-value" id="putPrice">$0.00</span>
                    </div>
                </div>
            </div>

            <!-- Greeks -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon icon-greeks">Δ</div>
                    <h2>Greeks</h2>
                </div>

                <div class="output-row">
                    <span class="output-label">Delta (Δ)</span>
                    <div class="greek-bar">
                        <span class="output-value" id="delta">0.00</span>
                        <div class="greek-visual">
                            <div class="greek-fill positive" id="deltaBar" style="width: 50%;"></div>
                        </div>
                    </div>
                </div>

                <div class="output-row">
                    <span class="output-label">Gamma (Γ)</span>
                    <span class="output-value" id="gamma">0.0000</span>
                </div>

                <div class="output-row">
                    <span class="output-label">Vega (ν)</span>
                    <span class="output-value" id="vega">$0.00</span>
                    <span class="hint">per 1% vol</span>
                </div>

                <div class="output-row">
                    <span class="output-label">Theta (Θ)</span>
                    <span class="output-value value-negative" id="theta">-$0.00</span>
                    <span class="hint">daily</span>
                </div>

                <div class="output-row">
                    <span class="output-label">Rho (ρ)</span>
                    <span class="output-value" id="rho">$0.00</span>
                    <span class="hint">per 1% rate</span>
                </div>
            </div>
        </div>

        <!-- Spot Price Slider -->
        <div class="slider-container">
            <div class="slider-header">
                <span>Spot Price Scenario</span>
                <span class="slider-value" id="spotSliderValue">$100.00</span>
            </div>
            <input type="range" id="spotSlider" value="100" min="50" max="150" step="0.5" style="width: 100%;">

            <div class="scenario-grid" id="scenarioGrid">
                <!-- Filled by JS -->
            </div>
        </div>

        <footer>
            <p>Powered by the <code>DAG</code> reactive computation framework</p>
            <p style="margin-top: 8px;">C = S·e<sup>-qT</sup>·N(d₁) - K·e<sup>-rT</sup>·N(d₂)</p>
        </footer>
    </div>

    <script>
        let isCall = true;
        let debounceTimer = null;

        // Sync volatility slider with input
        document.getElementById('volatility').addEventListener('input', function() {
            document.getElementById('volatilitySlider').value = this.value;
            debouncedUpdate();
        });

        document.getElementById('volatilitySlider').addEventListener('input', function() {
            document.getElementById('volatility').value = this.value;
            debouncedUpdate();
        });

        // Sync spot slider
        document.getElementById('spotSlider').addEventListener('input', function() {
            document.getElementById('spot').value = this.value;
            document.getElementById('spotSliderValue').textContent = '$' + parseFloat(this.value).toFixed(2);
            debouncedUpdate();
        });

        // Time to expiry days hint
        document.getElementById('timeToExpiry').addEventListener('input', function() {
            const days = Math.round(parseFloat(this.value) * 365);
            document.getElementById('daysHint').textContent = '≈ ' + days + ' days';
            debouncedUpdate();
        });

        // Add event listeners to all inputs
        document.querySelectorAll('input[type="number"]').forEach(input => {
            input.addEventListener('input', debouncedUpdate);
        });

        function setOptionType(call) {
            isCall = call;
            document.getElementById('btnCall').classList.toggle('active', call);
            document.getElementById('btnPut').classList.toggle('active', !call);
            updatePricing();
        }

        function debouncedUpdate() {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(updatePricing, 50);
        }

        async function updatePricing() {
            const params = {
                spot: parseFloat(document.getElementById('spot').value) || 100,
                strike: parseFloat(document.getElementById('strike').value) || 100,
                volatility: parseFloat(document.getElementById('volatility').value) || 20,
                riskFreeRate: parseFloat(document.getElementById('riskFreeRate').value) || 5,
                dividendYield: parseFloat(document.getElementById('dividendYield').value) || 0,
                timeToExpiry: parseFloat(document.getElementById('timeToExpiry').value) || 1,
                isCall: isCall
            };

            try {
                const response = await fetch('/api/price', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
                const data = await response.json();
                updateDisplay(data);
                updateScenarios(params);
            } catch (error) {
                console.error('Error:', error);
            }
        }

        function updateDisplay(data) {
            // Pricing
            document.getElementById('price').textContent = '$' + data.pricing.price.toFixed(4);
            document.getElementById('intrinsic').textContent = '$' + data.pricing.intrinsicValue.toFixed(4);
            document.getElementById('timeValue').textContent = '$' + data.pricing.timeValue.toFixed(4);
            document.getElementById('breakeven').textContent = '$' + data.pricing.breakeven.toFixed(2);
            document.getElementById('callPrice').textContent = '$' + data.pricing.callPrice.toFixed(4);
            document.getElementById('putPrice').textContent = '$' + data.pricing.putPrice.toFixed(4);

            // Greeks
            const delta = data.greeks.delta;
            document.getElementById('delta').textContent = delta.toFixed(4);
            document.getElementById('delta').className = 'output-value ' + (delta >= 0 ? 'value-positive' : 'value-negative');

            const deltaBar = document.getElementById('deltaBar');
            const deltaWidth = Math.abs(delta) * 100;
            deltaBar.style.width = deltaWidth + '%';
            deltaBar.className = 'greek-fill ' + (delta >= 0 ? 'positive' : 'negative');

            document.getElementById('gamma').textContent = data.greeks.gamma.toFixed(6);
            document.getElementById('vega').textContent = '$' + data.greeks.vega.toFixed(4);

            const theta = data.greeks.theta;
            document.getElementById('theta').textContent = (theta >= 0 ? '$' : '-$') + Math.abs(theta).toFixed(4);
            document.getElementById('theta').className = 'output-value ' + (theta >= 0 ? 'value-positive' : 'value-negative');

            const rho = data.greeks.rho;
            document.getElementById('rho').textContent = (rho >= 0 ? '$' : '-$') + Math.abs(rho).toFixed(4);
            document.getElementById('rho').className = 'output-value ' + (rho >= 0 ? 'value-positive' : 'value-negative');

            // Moneyness
            const moneyness = data.analysis.moneyness;
            const badge = document.getElementById('moneyness');
            badge.textContent = moneyness;
            badge.className = 'moneyness-badge moneyness-' + moneyness.toLowerCase();

            // Update spot slider
            document.getElementById('spotSlider').value = data.inputs.spot;
            document.getElementById('spotSliderValue').textContent = '$' + data.inputs.spot.toFixed(2);
        }

        async function updateScenarios(params) {
            const spots = [
                params.spot * 0.9,
                params.spot * 0.95,
                params.spot,
                params.spot * 1.05,
                params.spot * 1.1
            ];

            const grid = document.getElementById('scenarioGrid');
            grid.innerHTML = '';

            const optionPrice = parseFloat(document.getElementById('price').textContent.replace('$', ''));

            for (const spot of spots) {
                const scenarioParams = {...params, spot: spot};
                try {
                    const response = await fetch('/api/price', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(scenarioParams)
                    });
                    const data = await response.json();

                    // Calculate P&L at expiry (intrinsic - premium paid)
                    const pnl = data.pricing.intrinsicValue - optionPrice;

                    const item = document.createElement('div');
                    item.className = 'scenario-item';
                    item.innerHTML = `
                        <div class="scenario-spot">$${spot.toFixed(0)}</div>
                        <div class="scenario-pnl ${pnl >= 0 ? 'value-positive' : 'value-negative'}">
                            ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}
                        </div>
                    `;
                    grid.appendChild(item);
                } catch (error) {
                    console.error('Scenario error:', error);
                }
            }
        }

        // Initial load
        updatePricing();
    </script>
</body>
</html>
'''


# Global option instance
option = None


class OptionPricerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the option pricer API."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_html(self, html):
        """Send HTML response."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.send_html(HTML_TEMPLATE)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests."""
        global option

        if self.path == '/api/price':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            # Update option parameters
            option.Spot.set(data.get('spot', 100))
            option.Strike.set(data.get('strike', 100))
            option.Volatility.set(data.get('volatility', 20) / 100)
            option.RiskFreeRate.set(data.get('riskFreeRate', 5) / 100)
            option.DividendYield.set(data.get('dividendYield', 0) / 100)
            option.TimeToExpiry.set(data.get('timeToExpiry', 1))
            option.IsCall.set(data.get('isCall', True))

            # Return computed values
            self.send_json(option.to_dict())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global option

    # Reset DAG and create option model
    dag.reset()
    option = BlackScholesOption()

    port = 8000
    server = HTTPServer(('0.0.0.0', port), OptionPricerHandler)

    print("\n" + "="*60)
    print("  Black-Scholes Option Pricer - Web UI")
    print("="*60)
    print(f"\n  Open in browser: http://localhost:{port}")
    print("  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.shutdown()


if __name__ == '__main__':
    main()
