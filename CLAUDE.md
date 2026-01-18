# DAG Framework - Project Context for Claude

## Overview

This is a DAG (Directed Acyclic Graph) based dependency tracking and memoization framework for Python. It's inspired by functional reactive programming patterns used in quantitative finance for pricing derivatives and managing complex calculation graphs.

## Project Structure

```
dag/
├── __init__.py          # Public API exports
├── core.py              # DagManager, Node, NodeKey, Scenario
├── decorators.py        # @computed decorator, ComputedFunctionDescriptor
├── model.py             # Model base class, Registry pattern
├── flags.py             # Input, Overridable, Optional, etc.
├── state.py             # Branches, Overrides, OverrideSets, untracked mode
├── parser.py            # AST-based dependency parsing
├── exceptions.py        # Custom exception hierarchy
└── ui/                  # UI binding framework
    ├── __init__.py
    ├── app.py           # DagApp for Tkinter
    ├── bindings.py      # OutputBinding, InputBinding, TwoWayBinding
    └── widgets.py       # BoundLabel, BoundEntry, etc.

examples/
├── calculator.py        # Simple Tkinter calculator
├── option_pricer.py     # Black-Scholes pricer (Tkinter)
└── option_pricer_web.py # Black-Scholes pricer (Web UI)

tests/
├── test_basic.py        # Core functionality tests
├── test_cycles.py       # Cycle detection tests
├── test_watch.py        # Watch/notification tests
├── test_ui_bindings.py  # UI framework tests
└── ...                  # Additional test files
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_basic.py
```

## Coding Conventions

- Type hints are used throughout
- Docstrings follow Google style
- Tests use pytest with fixtures
- Each module has a clear single responsibility

---

## Original Design Specification

*The following is derived from the original DAG framework specification document.*

### Motivation and Philosophy

**Core Principles:**
- Object-oriented, lazy, memoization framework
- Dependency tracking and memoization - cache expensive computations or I/O operations
- Track inputs via parse-time constructs
- Lazy evaluation - compute only when needed
- Repeatable computations/Pure functions - stateless functions like Excel formulas
- Scenarios - captured as deltas or absolute overrides to nodes

**Architectural Goals:**
- **Model/View abstraction** - DAG models the dataflow/interaction between objects; UIs watch nodes
- **Distributed computing** - State is externalized from the computation
- **Abstraction** - Implementation is delegated (e.g., pricers use market models)
- **Database integration** - Objects can be persisted and loaded on demand

### Models

Models are the fundamental building blocks:
- Plain old Python objects that inherit from `dag.Model`
- Implicitly associated with a registry for persistence
- **Lazy and externalized state** with these consequences:
  - No constructor arguments allowed - use default calculated values
  - No member variables - use `Input` functions instead
  - Everything is a function - use `set`/`override` to mutate state
- Standard procedural member functions are supported alongside computed functions
- `@dag.computed` decorator defines a computed function
- Mutually dependent values can be handled via `inverse` functions
- Persistence is built in - anything marked as `Persisted` will be automatically serialized
- Watches notify when recalculations are required (but cannot intercept them)

### Computed Functions

**Evaluation Model:**
- The DAG parser transforms the syntax tree at parse time to detect dependencies
- The DAG caches valid invocations
- Stepping through a DAG function is **bottom up** rather than top down
- A computed function can be set or overridden to replace its computed value

**Purity Requirements - Computed functions must be pure:**
- Cannot change other computed functions during a calculation
- Never depend upon global variables or external state
- Must be repeatable - same inputs always produce same outputs

**Control Flow:**
Given computed functions where `C = self.A() + self.B()`, `A = self.B() + 1`, and `B = 2 + 3`:
- Calling `self.C()` executes in order: B → A → C

**Exception Handling:**
- Bottom-up control flow means you cannot catch exceptions from dependencies
- Lazy memoization means we cannot raise an exception when a dependency fails (we'd have to remember every possible invocation chain)
- Use `Optional` flag to indicate you will handle errors gracefully

### Computed Function Flags

```python
@dag.computed(dag.Input)        # This function can be permanently set
@dag.computed(dag.Overridable)  # This function can be temporarily overridden
@dag.computed(dag.Serialized)   # The result will be serialized
@dag.computed(dag.Persisted)    # == dag.Input | dag.Serialized
@dag.computed(dag.Optional)     # Return None instead of raising exceptions
```

### Parser and Dependency Detection

The parser guesses at references to other computed functions starting from known roots (`self` and arguments):

- **Method invocations:** `self.Spot()`
- **Indirection:** `self.PairObject().Spot()`
- **Registry lookups:** `self.registry['FX/MarketEnv'].CalculationDate()`
- **Expressions:** `self.registry['FXPairs/EURUSD'].Spot.override(1.4)`
- **Loops:** `[inst.Price() for inst in self.Instruments()]`
- **Predication:** `self.A() if self.UseA() else self.UseB()`
- **Star args:** `self.A(*self.Args())`

**Runtime Dependencies (untracked mode):**
- The DAG throws an exception if you call a computed function not detected at parse time
- Without this check, you'd miss a dependency and get invalid cached results
- Use `dag.untracked()` only if you're certain you don't want a dependency tracked
- Do not use it to work around exceptions

### State Management

**Set Values (enabled by `dag.Input`):**
- The DAG tracks state changes to decide what nodes need recalculation
- Instead of assigning to member variables, you set computed functions:
  ```python
  o.Strike.set(1.4)
  ```

**Overrides (enabled by `dag.Overridable`):**
- An override is a temporary change or expresses a change from the original state
- Designed for computing risk, editing objects, what-if scenarios
- A `dag.scenario()` call defines the scope of an override:
  ```python
  with dag.scenario():
      o.Spot.override(spot + eps)
      priceUp = o.Price()
  # Override automatically reverts here
  ```

**Why differentiate set values and overrides - Intent:**
- `set` implies a **permanent** change - you can't take it off
- `override` indicates a **temporary** change - reverts when scenario exits
- Overrides can be nested arbitrarily
- Overrides hold a hard reference to the object (prevents garbage collection during override)
- Override sets can be captured, serialized, and reapplied independently

### Inverse Functions

For expressing mutual dependencies (e.g., Spot == ForwardCurve[SpotDate]):

```python
def spotChange(self, newSpot):
    return [dag.NodeChange(self.FwdCurve, shift(self.FwdCurve(), newSpot - self.Spot()))]

@dag.computed(inverse=spotChange)
def Spot(self):
    return self.FwdCurve()[self.SpotDate()]
```

- Instead of applying the value directly, DAG calls the inverse function expecting NodeChange operations
- Think of it as the inverse function
- Must be a pure function - can inspect but not change state
- Never override or set computed functions during its execution

### Watches

- Watches **lazily notify** about recalculation needs
- Cannot be used to intercept the flow of control
- Notification mechanism to enable updates to external views (like UIs)
- Are lazy and cumulative - only one notification per traversal
- **Callbacks are weakly held** - keep strong references to prevent garbage collection
- Dispatched on demand using `dag.flush()`
- Callbacks must evaluate the node if they expect to be called again (otherwise node stays dirty)

### Branches

**Scenarios vs Branches:**
- **Scenarios** allow you to temporarily change the current state of the graph
- **Branches** allow multiple parallel states of the graph to exist simultaneously

**Branch Characteristics:**
- The DAG implements branches efficiently by sharing nodes across branches where possible
- Branches behave like scenarios - they can be nested
- A branch inherits overrides from its parent (search path for finding nodes)
- When calculating a node, it's automatically copied into a branch if any dependency is modified (**copy on write** semantics)

**Imperative Use:**
```python
with dag.branch() as b1:
    o.Strike.override(1.4)
    with b1: print(o.Price())  # Uses overridden strike

with dag.branch() as b2:
    o.Strike.override(1.5)
    with b2: print(o.Price())  # Uses different override
```

**On-Graph Branches:**
- Allow changing state while computing in a safe fashion
- A computed function can create a branch (new world state) automatically inheriting current state + modifications

### When to Use the DAG

**Memory Overhead:**
- DAG computed functions have overhead (~40 bytes per function + hash table entry)
- Persisted functions have additional overhead (~100-150 bytes)

**Best Practices:**
- **Don't use DAG computed functions for trivial computations** - use plain Python functions for abstraction
- **Bad:** A computed function to get an attribute: `def attr(self, a): return self.Data()[a]`
- **Good:** Return the data and access directly, or return a lambda
- **Instead of many small related attributes,** create an object and have "fat nodes":
  ```python
  @dag.computed(dag.Persisted)
  def Data(self):
      return DataType(self.a, self.b)  # One node instead of two
  ```
- **Computed functions cache based on arguments** - each unique value creates a node
- **Use lambdas as return values** to avoid creating excessive parameterized nodes

---

## Important Implementation Details

### Weak References
The watch system uses `weakref.ref()` for callbacks. When creating bindings, keep strong references to callback methods to prevent garbage collection:
```python
self._callback_ref = self._on_change  # Keep strong reference
computed_func.watch(self._callback_ref)
```

### Node States
Nodes can be: `INVALID`, `EVALUATING`, `VALID`, `ERROR`

### Cycle Detection
Cycles are detected during evaluation and raise `CycleError` (wrapped in `EvaluationError`)
