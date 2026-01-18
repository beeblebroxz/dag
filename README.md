# DAG

A dependency tracking and memoization framework for Python, inspired by functional reactive programming patterns used in quantitative finance.

## Features

- **Automatic Dependency Tracking**: Dependencies between functions are tracked automatically at runtime
- **Intelligent Memoization**: Results are cached and automatically invalidated when dependencies change
- **Reactive Updates**: Watch computed functions and get notified when values change
- **Scenario Analysis**: Use scenarios and branches for temporary "what-if" calculations
- **UI Bindings**: Built-in support for Tkinter and web-based reactive UIs

## Installation

```bash
pip install dag
```

Or install from source:

```bash
git clone https://github.com/beeblebroxz/dag.git
cd dag
pip install -e .
```

## Quick Start

```python
import dag

class PriceCalculator(dag.Model):
    @dag.computed(dag.Input)
    def Quantity(self):
        return 100

    @dag.computed(dag.Input)
    def UnitPrice(self):
        return 9.99

    @dag.computed
    def Total(self):
        return self.Quantity() * self.UnitPrice()

    @dag.computed
    def TotalWithTax(self):
        return self.Total() * 1.08

# Create calculator
calc = PriceCalculator()

print(calc.Total())          # 999.0
print(calc.TotalWithTax()) # 1078.92

# Change a value - dependents automatically recompute
calc.Quantity.set(200)
print(calc.Total())          # 1998.0
print(calc.TotalWithTax()) # 2157.84
```

## Core Concepts

### Computed Functions

Computed functions are methods decorated with `@dag.computed`. They:
- Automatically track which other computed functions they depend on
- Cache their results until dependencies change
- Can be marked with flags for special behavior

```python
@dag.computed              # Basic computed function - computed and cached
@dag.computed(dag.Input)   # Can be permanently changed with set()
@dag.computed(dag.Overridable) # Can be temporarily changed in a scenario
@dag.computed(dag.Optional)  # Returns None instead of raising
```

### Dependency Graph

When computed functions call other computed functions, dependencies are automatically tracked:

```python
class Option(dag.Model):
    @dag.computed(dag.Input)
    def Spot(self):
        return 100.0

    @dag.computed(dag.Input)
    def Strike(self):
        return 100.0

    @dag.computed
    def Intrinsic(self):
        # Automatically depends on Spot() and Strike()
        return max(0, self.Spot() - self.Strike())
```

When `Spot` or `Strike` changes, `Intrinsic` is automatically invalidated and will recompute on next access.

### Scenarios and Overrides

Use scenarios for temporary "what-if" calculations:

```python
opt = Option()
print(opt.Intrinsic())  # 0.0

with dag.scenario():
    opt.Spot.override(120.0)
    print(opt.Intrinsic())  # 20.0 (temporary)

print(opt.Intrinsic())  # 0.0 (reverted)
```

### Watches

Watch computed functions to be notified when they change:

```python
def on_change(node):
    print(f"{node.method_name} changed!")

calc.Total.watch(on_change)
calc.Quantity.set(50)
dag.flush()  # Triggers: "Total changed!"
```

## UI Bindings

### Tkinter

```python
from dag.ui import DagApp, BoundEntry, BoundLabel

app = DagApp("Calculator")
calc = PriceCalculator()

# Two-way binding: entry <-> computed function
entry = BoundEntry(app.root, cell=calc.Quantity, app=app)

# One-way binding: computed function -> label (auto-updates)
label = BoundLabel(app.root, cell=calc.Total, app=app)

app.run()
```

### Web UI

```python
# Run the option pricer example
python examples/option_pricer_web.py
# Open http://localhost:8000
```

## Examples

### Calculator
```bash
python examples/calculator.py
```

### Black-Scholes Option Pricer (Tkinter)
```bash
python examples/option_pricer.py
```

### Black-Scholes Option Pricer (Web)
```bash
python examples/option_pricer_web.py
```

## Advanced Features

### Parameterized Computed Functions

Computed functions can take parameters:

```python
class Portfolio(dag.Model):
    @dag.computed(dag.Input)
    def Price(self, ticker):
        return {"AAPL": 150, "GOOGL": 140}.get(ticker, 0)

    @dag.computed
    def TotalValue(self):
        return self.Price("AAPL") + self.Price("GOOGL")
```

### Registry Pattern

Create indexed collections of objects:

```python
class Instrument(dag.Model):
    @dag.computed(dag.Input)
    def Price(self):
        return 0.0

class InstrumentRegistry(dag.Registry[str, Instrument]):
    pass

registry = InstrumentRegistry()
registry["AAPL"].Price.set(150.0)
registry["GOOGL"].Price.set(140.0)
```

### Branches

Branches provide independent graph states for parallel scenario analysis:

```python
base_branch = dag.branch()
stressed_branch = dag.branch()

with stressed_branch:
    model.Volatility.override(0.5)
    stressed_price = model.Price()

with base_branch:
    base_price = model.Price()
```

### Untracked Mode

Skip dependency tracking in parts of the graph:

```python
result = dag.untracked(lambda: model.Compute())
# Dependencies from this call are not tracked
```

## API Reference

### Decorators
- `@dag.computed` - Mark a method as a computed function
- `@dag.computed(dag.Input)` - Allow permanent value changes
- `@dag.computed(dag.Overridable)` - Allow temporary overrides
- `@dag.computed(dag.Optional)` - Return None on errors

### Computed Function Methods
- `func.set(value)` - Set a permanent value (requires Input)
- `func.override(value)` - Set a temporary value (requires Overridable)
- `func.watch(callback)` - Watch for changes
- `func()` - Get the current value

### Context Managers
- `dag.scenario()` - Create a scenario for temporary overrides
- `dag.branch()` - Create an independent graph branch
- `dag.untracked(fn)` - Execute without dependency tracking

### Functions
- `dag.flush()` - Dispatch pending notifications
- `dag.reset()` - Reset the DAG (for testing)

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
