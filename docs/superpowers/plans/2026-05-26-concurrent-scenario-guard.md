# Single-Threaded Scenario/Branch Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make concurrent cross-thread use of scenarios/branches fail fast with a clear `ConcurrentScenarioError` instead of silently returning cache-poisoned values.

**Architecture:** The `DagManager` singleton tracks a single scenario-owner thread id plus a nesting depth, guarded by a lock. Scenario/branch entry *claims* ownership (conflicting thread → raise), exit *releases* it, and a lock-free O(1) guard rejects DAG reads/mutations from any other thread while a scenario is active. Branches are covered for free because `BranchScenario` subclasses `Scenario`.

**Tech Stack:** Python 3.9+, `threading`, `pytest`. Run tools with `python3 -m <tool>`.

**Conventions:**
- TDD throughout: write the failing test, watch it fail, implement minimally, watch it pass, commit.
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- After each task, the full suite must stay green: `python3 -m pytest -q`, plus `python3 -m ruff check dag/ tests/` and `python3 -m mypy`.
- Work happens on the existing branch `fix/correctness-ci-cleanup`.

---

## File Structure

- `dag/exceptions.py` — add `ConcurrentScenarioError(ScenarioError)`.
- `dag/__init__.py` — export `ConcurrentScenarioError`.
- `dag/core.py` — ownership state + `_claim_scenario_ownership`/`_release_scenario_ownership`/`_check_scenario_owner`; wire claim/release into `Scenario.__enter__`/`__exit__`; guard `Scenario.add_tweak` and `evaluate`; dedup the `evaluate` cycle/wait loop.
- `dag/decorators.py` — guard `ComputedFunctionAccessor.set()` and `clearValue()`.
- `tests/test_concurrency.py` — new `TestScenarioThreadGuard` class; remove the obsolete `test_concurrent_context_creation`.

---

## Task 1: Add `ConcurrentScenarioError`

**Files:**
- Modify: `dag/exceptions.py` (after `ScenarioError`, ~line 62)
- Modify: `dag/__init__.py` (exceptions import block ~lines 59-72; `__all__` ~lines 125-136)
- Test: `tests/test_concurrency.py` (new `TestScenarioThreadGuard` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_concurrency.py`:

```python
class TestScenarioThreadGuard:
    """Scenarios/branches are single-threaded; concurrent use fails fast."""

    def setup_method(self):
        dag.reset()

    def test_concurrent_scenario_error_is_scenario_error(self):
        err = dag.ConcurrentScenarioError(111, 222)
        assert isinstance(err, dag.ScenarioError)
        assert err.owner_thread == 111
        assert err.current_thread == 222
        assert "111" in str(err)
        assert "222" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest "tests/test_concurrency.py::TestScenarioThreadGuard::test_concurrent_scenario_error_is_scenario_error" -q`
Expected: FAIL with `AttributeError: module 'dag' has no attribute 'ConcurrentScenarioError'`.

- [ ] **Step 3: Add the exception**

In `dag/exceptions.py`, immediately after the `ScenarioError` class (which ends at the `pass` on ~line 62), add:

```python
class ConcurrentScenarioError(ScenarioError):
    """Raised when a scenario or branch is used concurrently across threads.

    Scenarios and branches are single-threaded. While one is active on a
    thread, the DAG must not be evaluated or mutated from another thread.
    """

    def __init__(self, owner_thread: int, current_thread: int):
        self.owner_thread = owner_thread
        self.current_thread = current_thread
        super().__init__(
            f"A scenario/branch is active on thread {owner_thread}; the DAG "
            f"cannot be used concurrently from thread {current_thread}. "
            "Scenarios and branches are single-threaded."
        )
```

- [ ] **Step 4: Export it**

In `dag/__init__.py`, add `ConcurrentScenarioError,` to the `from .exceptions import (...)` block (right after `ScenarioError,`), and add `"ConcurrentScenarioError",` to the `__all__` list (next to the other exception names, e.g. after `"ScenarioError",`).

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest "tests/test_concurrency.py::TestScenarioThreadGuard::test_concurrent_scenario_error_is_scenario_error" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add dag/exceptions.py dag/__init__.py tests/test_concurrency.py
git commit -m "Add ConcurrentScenarioError exception" \
  -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Scenario ownership + claim/release on enter/exit

**Files:**
- Modify: `dag/core.py` — import (`~line 30`), `DagManager._init` (~lines 195-203), new helpers near scenario management, `Scenario.__enter__`/`__exit__` (~lines 636-652)
- Test: `tests/test_concurrency.py::TestScenarioThreadGuard`
- Remove: `tests/test_concurrency.py::TestThreadSafety::test_concurrent_context_creation`

- [ ] **Step 1: Write the failing tests**

Add these methods to `TestScenarioThreadGuard` in `tests/test_concurrency.py`:

```python
    def test_concurrent_scenario_enter_raises(self):
        import threading

        class M(dag.Model):
            @dag.computed(dag.Overridable)
            def V(self):
                return 1

        obj = M()
        a_in = threading.Event()
        release_a = threading.Event()
        errors = []

        def thread_a():
            with dag.scenario():
                obj.V.override(5)
                a_in.set()
                release_a.wait(timeout=2)

        def thread_b():
            a_in.wait(timeout=2)
            try:
                with dag.scenario():
                    pass
            except dag.ConcurrentScenarioError as e:
                errors.append(e)
            finally:
                release_a.set()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start(); tb.start(); ta.join(); tb.join()

        assert len(errors) == 1

    def test_nested_scenarios_same_thread_ok(self):
        class M(dag.Model):
            @dag.computed(dag.Overridable)
            def V(self):
                return 1

        obj = M()
        with dag.scenario():
            obj.V.override(2)
            with dag.scenario():
                obj.V.override(3)
                assert obj.V() == 3
            assert obj.V() == 2
        assert obj.V() == 1

    def test_scenario_ownership_released_after_exit(self):
        import threading

        class M(dag.Model):
            @dag.computed(dag.Overridable)
            def V(self):
                return 1

        obj = M()
        with dag.scenario():
            obj.V.override(5)

        result = []

        def worker():
            with dag.scenario():
                obj.V.override(9)
                result.append(obj.V())

        t = threading.Thread(target=worker)
        t.start(); t.join()
        assert result == [9]

    def test_concurrent_branch_enter_raises(self):
        import threading

        class M(dag.Model):
            @dag.computed(dag.Overridable)
            def V(self):
                return 1

        obj = M()
        a_in = threading.Event()
        release_a = threading.Event()
        errors = []

        def thread_a():
            with dag.branch():
                obj.V.override(5)
                a_in.set()
                release_a.wait(timeout=2)

        def thread_b():
            a_in.wait(timeout=2)
            try:
                with dag.branch():
                    pass
            except dag.ConcurrentScenarioError as e:
                errors.append(e)
            finally:
                release_a.set()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start(); tb.start(); ta.join(); tb.join()

        assert len(errors) == 1
```

Also **delete** the existing `test_concurrent_context_creation` method from `TestThreadSafety` (in the same file, ~lines 198-237). It asserts the now-unsupported parallel-scenario pattern; the new contract is covered by `test_concurrent_scenario_enter_raises`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest "tests/test_concurrency.py::TestScenarioThreadGuard" -q`
Expected: `test_concurrent_scenario_enter_raises` and `test_concurrent_branch_enter_raises` FAIL (`assert 0 == 1` — no error raised, both threads enter). The nested/released tests PASS already (no guard yet).

- [ ] **Step 3: Add the import**

In `dag/core.py`, change the exceptions import (~line 30) from:

```python
from .exceptions import CycleError, EvaluationError, UntrackedError
```

to:

```python
from .exceptions import ConcurrentScenarioError, CycleError, EvaluationError, UntrackedError
```

- [ ] **Step 4: Add ownership state to `_init`**

In `DagManager._init` (~lines 195-203), append after `self._subscriptions_lock = threading.RLock()`:

```python
        self._scenario_owner: Optional[int] = None
        self._scenario_depth: int = 0
        self._scenario_lock = threading.Lock()
```

- [ ] **Step 5: Add claim/release helpers**

In `dag/core.py`, add these two methods to `DagManager` immediately before the `# Scenario management` comment / `push_context` method:

```python
    def _claim_scenario_ownership(self) -> None:
        """Claim scenario/branch ownership for the current thread (fail-fast on conflict)."""
        tid = threading.get_ident()
        with self._scenario_lock:
            if self._scenario_owner is None:
                self._scenario_owner = tid
                self._scenario_depth = 1
            elif self._scenario_owner == tid:
                self._scenario_depth += 1
            else:
                raise ConcurrentScenarioError(self._scenario_owner, tid)

    def _release_scenario_ownership(self) -> None:
        """Release one level of scenario/branch ownership for the current thread."""
        with self._scenario_lock:
            if self._scenario_depth > 0:
                self._scenario_depth -= 1
                if self._scenario_depth == 0:
                    self._scenario_owner = None
```

- [ ] **Step 6: Wire claim/release into `Scenario`**

Replace `Scenario.__enter__` and `Scenario.__exit__` (~lines 636-652) with:

```python
    def __enter__(self) -> Scenario:
        self._dag._claim_scenario_ownership()
        self._dag.push_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        try:
            # Revert all overrides
            for node, old_value in reversed(self._tweaks):
                self._dag.set_tweak_value(node.key, old_value)
                # Always invalidate dependents when reverting an override
                # The node's dependents need to recompute with the original value
                self._dag.invalidate_dependents(node)
                if old_value is NO_VALUE:
                    # Also invalidate this node since we're reverting to computed value
                    node.invalidate()
        finally:
            self._dag.pop_context()
            self._dag._release_scenario_ownership()
        return False
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python3 -m pytest "tests/test_concurrency.py::TestScenarioThreadGuard" tests/test_state.py -q`
Expected: PASS (all guard tests pass; single-threaded scenario tests in `test_state.py` unaffected).

- [ ] **Step 8: Run full suite + lint + types**

Run: `python3 -m pytest -q && python3 -m ruff check dag/ tests/ && python3 -m mypy`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add dag/core.py tests/test_concurrency.py
git commit -m "Track scenario/branch ownership; fail-fast on concurrent enter" \
  -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Guard reads in `evaluate()`

**Files:**
- Modify: `dag/core.py` — new `_check_scenario_owner` helper; first line of `evaluate()` (~line 387)
- Test: `tests/test_concurrency.py::TestScenarioThreadGuard`

- [ ] **Step 1: Write the failing test**

Add to `TestScenarioThreadGuard`:

```python
    def test_evaluate_during_foreign_scenario_raises(self):
        import threading

        class M(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

            @dag.computed
            def Price(self):
                return self.Spot() * 2

        obj = M()
        assert obj.Price() == 200.0  # cache it with the base value

        a_in = threading.Event()
        release_a = threading.Event()
        errors = []
        observed = []

        def thread_a():
            with dag.scenario():
                obj.Spot.override(500.0)
                observed.append(obj.Price())  # computes 1000 under the override
                a_in.set()
                release_a.wait(timeout=2)

        def thread_b():
            a_in.wait(timeout=2)
            try:
                obj.Price()  # no scenario: must not silently see A's overridden value
            except dag.ConcurrentScenarioError as e:
                errors.append(e)
            finally:
                release_a.set()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start(); tb.start(); ta.join(); tb.join()

        assert observed == [1000.0]
        assert len(errors) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest "tests/test_concurrency.py::TestScenarioThreadGuard::test_evaluate_during_foreign_scenario_raises" -q`
Expected: FAIL — `assert 0 == 1` because thread B silently receives `1000.0` (the poisoned value) instead of raising.

- [ ] **Step 3: Add the guard helper**

In `dag/core.py`, add this method to `DagManager` next to `_claim_scenario_ownership`/`_release_scenario_ownership`:

```python
    def _check_scenario_owner(self) -> None:
        """Reject DAG access from a thread other than the active scenario owner."""
        owner = self._scenario_owner  # lock-free single-attribute read (GIL-atomic)
        if owner is not None and owner != threading.get_ident():
            raise ConcurrentScenarioError(owner, threading.get_ident())
```

- [ ] **Step 4: Call the guard at the top of `evaluate()`**

In `DagManager.evaluate()`, insert the guard as the first statement of the method body (before `state = self._get_execution_state()` ~line 387):

```python
        self._check_scenario_owner()
        state = self._get_execution_state()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest "tests/test_concurrency.py::TestScenarioThreadGuard::test_evaluate_during_foreign_scenario_raises" -q`
Expected: PASS.

- [ ] **Step 6: Run full suite + lint + types**

Run: `python3 -m pytest -q && python3 -m ruff check dag/ tests/ && python3 -m mypy`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add dag/core.py tests/test_concurrency.py
git commit -m "Guard evaluate() against reads during a foreign scenario" \
  -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Guard mutations (`set`, `clearValue`, `add_tweak`)

**Files:**
- Modify: `dag/core.py` — `Scenario.add_tweak` (~line 654)
- Modify: `dag/decorators.py` — `set()` (~line 137) and `clearValue()` (~line 198)
- Test: `tests/test_concurrency.py::TestScenarioThreadGuard`

- [ ] **Step 1: Write the failing test**

Add to `TestScenarioThreadGuard`:

```python
    def test_set_during_foreign_scenario_raises(self):
        import threading

        class M(dag.Model):
            @dag.computed(dag.Input)
            def X(self):
                return 1

            @dag.computed(dag.Overridable)
            def Y(self):
                return 2

        obj = M()
        a_in = threading.Event()
        release_a = threading.Event()
        errors = []

        def thread_a():
            with dag.scenario():
                obj.Y.override(5)
                a_in.set()
                release_a.wait(timeout=2)

        def thread_b():
            a_in.wait(timeout=2)
            try:
                obj.X.set(99)  # permanent mutation must not run during a foreign scenario
            except dag.ConcurrentScenarioError as e:
                errors.append(e)
            finally:
                release_a.set()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start(); tb.start(); ta.join(); tb.join()

        assert len(errors) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest "tests/test_concurrency.py::TestScenarioThreadGuard::test_set_during_foreign_scenario_raises" -q`
Expected: FAIL — `assert 0 == 1` (the `set` succeeds silently).

- [ ] **Step 3: Guard `set()` and `clearValue()`**

In `dag/decorators.py`, in `set()` add the guard right after the Input-flag check:

```python
        if not (self._descriptor.flags & Input):
            raise SetValueError(self._descriptor.name)
        self._dag._check_scenario_owner()
```

In `clearValue()` add the guard right after its Input-flag check:

```python
        if not (self._descriptor.flags & Input):
            raise SetValueError(self._descriptor.name)
        self._dag._check_scenario_owner()
```

- [ ] **Step 4: Guard `Scenario.add_tweak()`**

In `dag/core.py`, make `add_tweak` the guard's first action (this also covers `apply_overrides`, which applies tweaks before the scenario is entered):

```python
    def add_tweak(self, node: Node, new_value: Any) -> None:
        """Record an override for later reversion."""
        self._dag._check_scenario_owner()
        old_value = self._dag.get_tweak_value(node.key)
        self._tweaks.append((node, old_value))
        self._dag.set_tweak_value(node.key, new_value)
        # Must invalidate dependents even if this node is already invalid
        # (e.g., when node has a set_value but state is INVALID)
        self._dag.invalidate_dependents(node)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest "tests/test_concurrency.py::TestScenarioThreadGuard::test_set_during_foreign_scenario_raises" -q`
Expected: PASS.

- [ ] **Step 6: Run full suite + lint + types**

Run: `python3 -m pytest -q && python3 -m ruff check dag/ tests/ && python3 -m mypy`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add dag/core.py dag/decorators.py tests/test_concurrency.py
git commit -m "Guard set/clearValue/add_tweak against foreign-scenario mutation" \
  -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Dedup the `evaluate()` cycle/wait logic (refactor)

Behavior-preserving cleanup. The override/valid/cycle checks are currently written three times (before the wait loop, inside it, and after). Collapse them into one re-check loop. The cycle condition (`node.key in state.eval_stack`) is invariant across `wait()` because this thread's `eval_stack` cannot change while the thread is blocked, so checking it each iteration is equivalent to checking it once.

**Files:**
- Modify: `dag/core.py` — `evaluate()` (the `with node._condition:` block through the `if cycle_error is not None: raise cycle_error`)
- Tests: existing `tests/test_cycles.py` and `tests/test_concurrency.py` guard the behavior (no new test).

- [ ] **Step 1: Confirm the guarding tests pass before refactoring**

Run: `python3 -m pytest tests/test_cycles.py tests/test_concurrency.py -q`
Expected: PASS (establishes the green baseline that must be preserved).

- [ ] **Step 2: Replace the block**

In `dag/core.py`, replace this current block inside `evaluate()`:

```python
        with node._condition:
            has_override, override_value = self._get_effective_value(node)
            if has_override:
                return override_value

            if node.is_valid:
                return node.value

            if node.key in state.eval_stack:
                cycle_path = state.eval_stack[state.eval_stack.index(node.key):]
                cycle_str = " -> ".join(k.method_name for k in cycle_path)
                cycle_error = CycleError(
                    f"Cyclic dependency detected: {cycle_str} -> {node.key.method_name}"
                )
            else:
                while (
                    node.state == NodeState.EVALUATING
                    and node._evaluating_thread_id != current_thread_id
                ):
                    node._condition.wait()
                    has_override, override_value = self._get_effective_value(node)
                    if has_override:
                        return override_value
                    if node.is_valid:
                        return node.value

                if node.key in state.eval_stack:
                    cycle_path = state.eval_stack[state.eval_stack.index(node.key):]
                    cycle_str = " -> ".join(k.method_name for k in cycle_path)
                    cycle_error = CycleError(
                        f"Cyclic dependency detected: {cycle_str} -> {node.key.method_name}"
                    )
                else:
                    has_override, override_value = self._get_effective_value(node)
                    if has_override:
                        return override_value

                    if node.is_valid:
                        return node.value

                    node._state = NodeState.EVALUATING
                    node._evaluating_thread_id = current_thread_id

        if cycle_error is not None:
            raise cycle_error
```

with:

```python
        with node._condition:
            while True:
                has_override, override_value = self._get_effective_value(node)
                if has_override:
                    return override_value
                if node.is_valid:
                    return node.value

                if node.key in state.eval_stack:
                    cycle_path = state.eval_stack[state.eval_stack.index(node.key):]
                    cycle_str = " -> ".join(k.method_name for k in cycle_path)
                    cycle_error = CycleError(
                        f"Cyclic dependency detected: {cycle_str} -> {node.key.method_name}"
                    )
                    break

                if (
                    node.state == NodeState.EVALUATING
                    and node._evaluating_thread_id != current_thread_id
                ):
                    node._condition.wait()
                    continue

                node._state = NodeState.EVALUATING
                node._evaluating_thread_id = current_thread_id
                break

        if cycle_error is not None:
            raise cycle_error
```

- [ ] **Step 3: Run the guarding tests**

Run: `python3 -m pytest tests/test_cycles.py tests/test_concurrency.py -q`
Expected: PASS (identical behavior).

- [ ] **Step 4: Run full suite + lint + types**

Run: `python3 -m pytest -q && python3 -m ruff check dag/ tests/ && python3 -m mypy`
Expected: all green (`192 passed` expected: 191 prior + the net effect of the new guard tests minus the removed `test_concurrent_context_creation`).

- [ ] **Step 5: Commit**

```bash
git add dag/core.py
git commit -m "Collapse triplicated cycle/wait checks in evaluate() into one loop" \
  -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Ownership state + lock → Task 2 (Steps 4-5). ✓
- claim/release + enter/exit + `finally` release → Task 2 (Steps 5-6). ✓
- Guard in `evaluate()` (first statement, before cache check) → Task 3. ✓
- Guard in `set()`/`clearValue()` → Task 4 (Step 3). ✓
- Guard in `add_tweak()` (covers `apply_overrides`) → Task 4 (Step 4). ✓
- Branches covered via `BranchScenario`/`Scenario` → Task 2 `test_concurrent_branch_enter_raises`. ✓
- New `ConcurrentScenarioError(ScenarioError)` + export → Task 1. ✓
- evaluate() cycle/wait dedup → Task 5. ✓
- Tests: rewrite `test_concurrent_context_creation` (removed; replaced by enter-raises test), repro-now-loud, nested-OK, released-after-exit → Tasks 2-4. ✓
- Edge cases: nesting (depth counter), exception-safe release (`finally`), no-scenario no-op, `flush()` not guarded directly → covered by design + tests. ✓

**Placeholder scan:** none — every code/command step has concrete content.

**Type consistency:** `_scenario_owner: Optional[int]`, `_scenario_depth: int`, `_scenario_lock`; `_claim_scenario_ownership()`, `_release_scenario_ownership()`, `_check_scenario_owner()`; `ConcurrentScenarioError(owner_thread, current_thread)` — names/signatures match across Tasks 1-5. `Optional` is already imported in `dag/core.py`.

**Note:** Tasks 1→5 are ordered so each leaves the suite green and is independently committable. Task 5 (refactor) must come after Tasks 3-4 so the new concurrency tests exist to guard the change.
