# Design: Single-Threaded Scenario/Branch Guard

- **Date:** 2026-05-26
- **Status:** Approved (pending implementation plan)
- **Issue:** #1 — Concurrent scenario cache poisoning

## Problem

Scenario overrides live in **thread-local** state (`_ExecutionState.tweaks`), but the
memoized value (`Node._value`), node validity, and invalidation are **global** (shared
across threads). As a result, a derived value computed under one thread's override is
written to the shared node and becomes visible to other threads.

Concrete failure (verified):

1. Thread A enters `dag.scenario()` and overrides node `X` (a thread-local tweak), which
   globally invalidates `X`'s dependents.
2. Thread A evaluates a derived node `D` that depends on `X`. `D` is computed using the
   overridden `X` and the result is written to the **shared** `D._value` (state `VALID`).
3. Thread B (no scenario, or a different scenario) reads `D`, hits the valid shared cache,
   and receives thread A's overridden-derived value.

The result is a silent, intermittent wrong answer — the worst failure mode for a pricing
library. `Branch` uses the same mechanism (`BranchScenario` subclasses `Scenario`) and has
the same flaw.

## Decision

Scenarios and branches are a **single-threaded** construct. Concurrent scenario/branch use
across threads is **not supported** and must **fail fast** with a clear exception rather
than silently returning wrong values.

We do **not** build per-scenario value isolation (parallel scenario evaluation on shared
objects). That was explicitly ruled out as a non-goal.

Rationale: the poisoning requires three steps (override → cache derived value globally →
foreign read). The cheapest, fully-correct break is to forbid the foreign access while a
scenario is active in another thread. This is a guard, not a redesign: O(1) hot-path cost,
zero overhead when no scenario is active, and it matches the single-threaded contract.

## Design

### Scenario ownership state

Add to `DagManager` (initialized in `_init`, reset by `reset()`):

- `_scenario_owner: Optional[int]` — thread id holding active scenario(s), or `None`.
- `_scenario_depth: int` — nesting count for the owner thread.
- `_scenario_lock: threading.Lock` — guards owner/depth transitions.

### Three operations on `DagManager`

**claim** (called on scenario/branch *enter*):

```python
def _claim_scenario_ownership(self) -> None:
    tid = threading.get_ident()
    with self._scenario_lock:
        if self._scenario_owner is None:
            self._scenario_owner = tid
            self._scenario_depth = 1
        elif self._scenario_owner == tid:
            self._scenario_depth += 1          # nested scenario/branch, same thread
        else:
            raise ConcurrentScenarioError(self._scenario_owner, tid)
```

**release** (called on scenario/branch *exit*):

```python
def _release_scenario_ownership(self) -> None:
    with self._scenario_lock:
        if self._scenario_depth > 0:
            self._scenario_depth -= 1
            if self._scenario_depth == 0:
                self._scenario_owner = None
```

**guard** (called on every read/mutation):

```python
def _check_scenario_owner(self) -> None:
    owner = self._scenario_owner            # lock-free single-attribute read (GIL-atomic)
    if owner is not None and owner != threading.get_ident():
        raise ConcurrentScenarioError(owner, threading.get_ident())
```

The guard reads `_scenario_owner` without the lock. This is a safety net, not a hard
invariant; a momentarily stale read is acceptable and keeps the hot path to a single
attribute read + comparison (and `get_ident()` is only called when an owner exists).

### Where the checks sit

- `Scenario.__enter__` calls `_claim_scenario_ownership()` **first** (before `push_context`).
  If it raises, the `with` body never runs and `__exit__` is not called — nothing to release.
- `Scenario.__exit__` calls `_release_scenario_ownership()` in a `finally`, so ownership is
  released even if the body raised or the revert logic fails.
- `BranchScenario` subclasses `Scenario` and `Branch` drives it via `BranchScenario.__enter__/
  __exit__`, so **branches are covered with no additional code**.
- `DagManager.evaluate()` calls `_check_scenario_owner()` as its **first statement** (before
  dependency tracking and before the cached-value check — a cached value may itself be poisoned).
- `ComputedFunctionAccessor.set()` and `clearValue()` call `self._dag._check_scenario_owner()`
  at the top (permanent mutations of shared state).
- `Scenario.add_tweak()` calls the guard before applying. This covers `override()` and also
  `apply_overrides()`, which calls `add_tweak()` *before* the returned scenario is entered
  (i.e., before ownership is claimed) — without this, that pre-enter `invalidate_dependents`
  would be unguarded. Within the owner thread the guard is a no-op (owner == current).

### New exception

In `exceptions.py`:

```python
class ConcurrentScenarioError(ScenarioError):
    """Raised when a scenario or branch is used concurrently across threads.

    Scenarios and branches are single-threaded. While one is active on a thread,
    the DAG must not be evaluated or mutated from another thread.
    """
    def __init__(self, owner_thread: int, current_thread: int):
        self.owner_thread = owner_thread
        self.current_thread = current_thread
        super().__init__(
            f"A scenario/branch is active on thread {owner_thread}; the DAG cannot be "
            f"used concurrently from thread {current_thread}. Scenarios and branches "
            "are single-threaded."
        )
```

Exported from `__init__.py`.

### Folded-in cleanup: dedup `evaluate()` cycle/wait logic

While editing the top of `evaluate()`, collapse the triplicated override/valid/cycle checks
(currently checked once before the wait loop, once inside it, and once after) into a single
re-check loop. This is a behavior-preserving refactor; the cycle condition
(`node.key in state.eval_stack`) is invariant across `wait()` because this thread's
`eval_stack` cannot change while the thread is blocked.

Target structure (inside `with node._condition:`):

```python
while True:
    has_override, override_value = self._get_effective_value(node)
    if has_override:
        return override_value
    if node.is_valid:
        return node.value

    if node.key in state.eval_stack:                 # re-entrant call from this thread
        cycle_path = state.eval_stack[state.eval_stack.index(node.key):]
        cycle_str = " -> ".join(k.method_name for k in cycle_path)
        cycle_error = CycleError(
            f"Cyclic dependency detected: {cycle_str} -> {node.key.method_name}"
        )
        break

    if (node.state == NodeState.EVALUATING
            and node._evaluating_thread_id != current_thread_id):
        node._condition.wait()                       # another thread is computing it
        continue

    node._state = NodeState.EVALUATING               # claim evaluation for this thread
    node._evaluating_thread_id = current_thread_id
    break
```

`cycle_error` is still raised after releasing the lock, exactly as today.

## Semantics & edge cases

- **Nesting / same-thread reentry:** the depth counter allows arbitrarily nested
  scenarios/branches in one thread — no false positives.
- **Exceptions inside a scenario body:** `__exit__` runs (context-manager contract) and
  releases in a `finally`.
- **No scenario active:** `_scenario_owner is None`, so the guard is a no-op. All existing
  non-scenario concurrency behavior (concurrent evaluation/sets without scenarios) is
  unchanged.
- **`flush()`** is not guarded directly (it only reads validity flags); any callback that
  evaluates a node is guarded transitively via `evaluate()`.
- **`reset()`** re-initializes ownership state (test isolation).

## Testing

In `tests/test_concurrency.py`:

- **Rewrite `test_concurrent_context_creation`** to assert the new contract deterministically:
  thread A enters a scenario and holds it behind a `threading.Event`; a second thread that
  tries to enter a scenario raises `ConcurrentScenarioError`. (Today this test asserts the
  now-unsupported parallel-scenario pattern; it passes only because it reads the directly
  overridden node, never a derived one.)
- **New — the repro, now loud:** thread A holds a scenario (behind an `Event`) with an
  override on `X`; a no-scenario thread B evaluating a derived node raises
  `ConcurrentScenarioError` (instead of silently returning A's overridden value).
- **New — nested same-thread OK:** nested `scenario()`/`branch()` in a single thread evaluate
  without error.
- **New — ownership released after exit:** after thread A's scenario exits, thread B can enter
  a scenario / evaluate normally.
- **New — branch parity:** a second thread entering a `branch()` while another holds one raises.
- **Regression:** the full single-threaded scenario/branch suites (`test_state.py`,
  `test_basic.py`, `test_inverse.py`, etc.) stay green, plus the existing non-scenario
  concurrency tests in `test_concurrency.py`.

## Files touched

- `dag/exceptions.py` — add `ConcurrentScenarioError(ScenarioError)`.
- `dag/core.py` — ownership fields in `_init`; `_claim_scenario_ownership`,
  `_release_scenario_ownership`, `_check_scenario_owner`; call claim/release in
  `Scenario.__enter__/__exit__`; guard in `Scenario.add_tweak`; guard + cycle-dedup in
  `evaluate()`.
- `dag/decorators.py` — guard call in `set()` and `clearValue()`.
- `dag/__init__.py` — export `ConcurrentScenarioError`.
- `tests/test_concurrency.py` — rewrite one test, add the new tests above.

## Out of scope

- Per-scenario/per-branch value isolation (parallel scenario evaluation on shared objects).
- Subscription `flush()` performance / queue redesign.
- General (non-scenario) concurrent set/read race hardening beyond current behavior.
