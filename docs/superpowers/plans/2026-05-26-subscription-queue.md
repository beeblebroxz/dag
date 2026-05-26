# Pending-Queue Watch Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the O(all-watchers) scan-all `flush()` with a real pending-notification set so watchers fire once per change (O(pending)), per the spec's lazy/cumulative/evaluate-to-rearm model.

**Architecture:** `DagManager` keeps a `_pending_notifications` set. A watched node is enqueued when it changes (`invalidate_node` already calls `_queue_subscription`; `set()` gains one line for the never-evaluated-Input case). `flush()` snapshots-and-clears the set and dispatches each node's callbacks outside the lock.

**Tech Stack:** Python 3.9+, `threading`, `weakref`, `pytest`. Run tools with `python3 -m <tool>`.

**Conventions:** TDD (write failing test, watch it fail, implement, watch it pass, commit). Every commit message ends with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`. Work on the existing branch `redesign/subscription-flush`.

---

## File Structure

- `dag/core.py` — `DagManager._init` (add the pending set), `_queue_subscription` (real enqueue), `flush` (dispatch-and-clear).
- `dag/decorators.py` — one line in `ComputedFunctionAccessor.set()` to enqueue the set node.
- `tests/test_watch.py` — three new tests appended to the existing `TestSubscriptions` class.

The change is one cohesive unit: the `flush` rewrite, the real `_queue_subscription`, and the `set()` enqueue must land together, because intermediate states would break existing watch tests. Hence a single task driven by two discriminating RED tests.

---

## Task 1: Pending-queue dispatch

**Files:**
- Modify: `dag/core.py` — `_init` (~line 199), `_queue_subscription` (~lines 614-617), `flush` (~lines 619-645)
- Modify: `dag/decorators.py` — `set()` (~line 158)
- Test: `tests/test_watch.py` (append to `class TestSubscriptions`)

- [ ] **Step 1: Write the first failing test (notify-once-per-change)**

Append to `class TestSubscriptions` in `tests/test_watch.py`:

```python
    def test_flush_notifies_once_per_change(self):
        """A change enqueues one notification; flushing twice fires the callback
        only once (the pending set is cleared on dispatch)."""

        class Observable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Value() * 2

        obj = Observable()
        calls = []

        def cb(node):
            calls.append(1)

        obj.Derived.watch(cb)
        assert obj.Derived() == 2
        obj.Value = 5  # invalidates Derived -> enqueued once
        dag.flush()
        dag.flush()  # pending is now empty

        assert len(calls) == 1
```

- [ ] **Step 2: Write the second failing test (only-changed-fires)**

Append to `class TestSubscriptions`:

```python
    def test_flush_fires_only_changed_watched_nodes(self):
        """Only nodes that changed since the last flush are notified; an
        unrelated, still-invalid watched node is not swept in."""

        class M(dag.Model):
            @dag.computed(dag.Input)
            def A(self):
                return 1

            @dag.computed(dag.Input)
            def B(self):
                return 1

            @dag.computed
            def DA(self):
                return self.A() + 1

            @dag.computed
            def DB(self):
                return self.B() + 1

        obj = M()
        fired = []

        def cb_da(node):
            fired.append('DA')

        def cb_db(node):
            fired.append('DB')

        obj.DA.watch(cb_da)
        obj.DB.watch(cb_db)

        assert obj.DA() == 2  # DA evaluated (valid); DB never evaluated (invalid)
        obj.A = 9             # changes only DA's input
        dag.flush()

        assert fired == ['DA']  # DB (invalid but unchanged) must NOT fire
```

- [ ] **Step 3: Run both tests to verify they FAIL**

Run: `python3 -m pytest "tests/test_watch.py::TestSubscriptions::test_flush_notifies_once_per_change" "tests/test_watch.py::TestSubscriptions::test_flush_fires_only_changed_watched_nodes" -q`

Expected: both FAIL.
- `test_flush_notifies_once_per_change`: `assert 2 == 1` — the current scan-all `flush` re-fires the still-invalid node on the second flush.
- `test_flush_fires_only_changed_watched_nodes`: `fired` contains both `'DA'` and `'DB'` — the current `flush` sweeps in `DB` because it is invalid (never evaluated), even though `B` didn't change.

- [ ] **Step 4: Add the pending set to `_init`**

In `dag/core.py`, `DagManager._init`, add this line immediately after `self._subscriptions_lock = threading.RLock()` (~line 199):

```python
        self._pending_notifications: Set[NodeKey] = set()
```

(`Set` and `NodeKey` are already imported/defined in `dag/core.py`.)

- [ ] **Step 5: Make `_queue_subscription` enqueue (only for watched nodes)**

In `dag/core.py`, replace the no-op `_queue_subscription`:

```python
    def _queue_subscription(self, node_key: NodeKey) -> None:
        """Queue a subscription notification (lazy dispatch)."""
        # For now, we don't dispatch immediately - use flush()
        pass
```

with:

```python
    def _queue_subscription(self, node_key: NodeKey) -> None:
        """Enqueue a pending notification for a watched node (dispatched on flush)."""
        with self._subscriptions_lock:
            if node_key in self._subscriptions:
                self._pending_notifications.add(node_key)
```

- [ ] **Step 6: Rewrite `flush` to dispatch-and-clear the pending set**

In `dag/core.py`, replace the current `flush` body:

```python
    def flush(self) -> None:
        """Dispatch all queued subscription notifications."""
        # Clean up dead references and invoke callbacks
        with self._subscriptions_lock:
            subscriptions = list(self._subscriptions.items())

        for node_key, callbacks in subscriptions:
            node = self.get_node(node_key)
            if node is None or node.is_valid:
                continue

            live_callbacks = []
            for cb_ref in callbacks:
                cb = cb_ref()
                if cb is not None:
                    live_callbacks.append(cb_ref)
                    try:
                        cb(node)
                    except Exception:
                        # Watch callbacks must not interrupt the DAG, but their
                        # failures must be visible rather than silently dropped.
                        logger.exception(
                            "Watch callback for %r failed", node.method_name
                        )

            with self._subscriptions_lock:
                self._subscriptions[node_key] = live_callbacks
```

with:

```python
    def flush(self) -> None:
        """Dispatch pending notifications (one per watched node that changed
        since the last flush) and clear the pending set."""
        with self._subscriptions_lock:
            pending = list(self._pending_notifications)
            self._pending_notifications.clear()

        for node_key in pending:
            node = self.get_node(node_key)
            if node is None:
                continue

            with self._subscriptions_lock:
                callbacks = list(self._subscriptions.get(node_key, []))

            live_callbacks = []
            for cb_ref in callbacks:
                cb = cb_ref()
                if cb is None:
                    continue
                live_callbacks.append(cb_ref)
                try:
                    cb(node)
                except Exception:
                    # Watch callbacks must not interrupt the DAG, but their
                    # failures must be visible rather than silently dropped.
                    logger.exception(
                        "Watch callback for %r failed", node.method_name
                    )

            with self._subscriptions_lock:
                if node_key in self._subscriptions:
                    self._subscriptions[node_key] = live_callbacks
```

- [ ] **Step 7: Enqueue the node on `set()`**

In `dag/decorators.py`, `ComputedFunctionAccessor.set()`, the direct-set branch currently ends:

```python
        # Direct set
        node = self._get_or_create_node()
        node._set_value = value
        # Invalidate dependents (not this node, since it now has a set value)
        self._dag.invalidate_dependents(node)
```

Add one line so watchers of the set node fire even when it has no dependents / was never evaluated:

```python
        # Direct set
        node = self._get_or_create_node()
        node._set_value = value
        # Invalidate dependents (not this node, since it now has a set value)
        self._dag.invalidate_dependents(node)
        # Notify watchers of this node itself (it changed).
        self._dag._queue_subscription(node.key)
```

- [ ] **Step 8: Run the two new tests to verify they PASS**

Run: `python3 -m pytest "tests/test_watch.py::TestSubscriptions::test_flush_notifies_once_per_change" "tests/test_watch.py::TestSubscriptions::test_flush_fires_only_changed_watched_nodes" -q`
Expected: both PASS.

- [ ] **Step 9: Run all existing watch tests to verify no regression**

Run: `python3 -m pytest tests/test_watch.py -q`
Expected: PASS (the 8 pre-existing tests + the 2 new = 10 passed). In particular `test_watch_before_evaluation` (set a never-evaluated Input → notified) passes because of Step 7, and `test_watch_does_not_fire_when_valid` passes because the unaffected node is never enqueued.

- [ ] **Step 10: Add the re-arm coverage test**

This locks the evaluate-to-rearm contract. Append to `class TestSubscriptions`:

```python
    def test_watcher_rearms_after_reevaluation(self):
        """After a notification the watcher fires again only once the node is
        re-evaluated (re-armed) and then changed again."""

        class Observable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 1

            @dag.computed
            def Derived(self):
                return self.Value() * 2

        obj = Observable()
        calls = []

        def cb(node):
            calls.append(1)

        obj.Derived.watch(cb)
        assert obj.Derived() == 2
        obj.Value = 5
        dag.flush()
        assert len(calls) == 1

        assert obj.Derived() == 10  # re-evaluate -> re-arm
        obj.Value = 7
        dag.flush()
        assert len(calls) == 2
```

Run: `python3 -m pytest "tests/test_watch.py::TestSubscriptions::test_watcher_rearms_after_reevaluation" -q`
Expected: PASS. (Note: this test also passes on the pre-change code, which re-fired liberally; it documents the intended re-arm behavior rather than driving the change.)

- [ ] **Step 11: Full suite + lint + types**

Run: `python3 -m pytest -q && python3 -m ruff check dag/ tests/ && python3 -m mypy`
Expected: all green (expect 205 passed: 202 + 3 new).

- [ ] **Step 12: Commit**

```bash
git add dag/core.py dag/decorators.py tests/test_watch.py
git commit -m "Dispatch watch notifications from a pending queue, not a full scan" \
  -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Pending set in `_init` (cleared by `reset()`) → Step 4. ✓
- `_queue_subscription` enqueues only watched nodes → Step 5. ✓
- `invalidate_node` already enqueues invalidated nodes → unchanged, relied upon (covers derived/dependent + `clearValue`). ✓
- `set()` enqueues its node → Step 7. ✓
- `override` unchanged (dependents via `invalidate_dependents`) → no task needed. ✓
- `flush` snapshot-and-clear, fire outside lock, no `is_valid` re-check, log exceptions → Step 6. ✓
- Behavior changes (notify-once, only-changed-fires, O(pending)) → Steps 1-2 tests. ✓
- Edge: never-evaluated set Input preserved → Step 9 (`test_watch_before_evaluation`). ✓
- Edge: unrelated invalid node not swept → Step 2 test. ✓
- Edge: callback exceptions logged not propagated → Step 6 (unchanged) + existing tests in Step 9. ✓
- New tests: dispatch-and-clear, only-changed-fires, re-arm → Steps 1, 2, 10. ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `_pending_notifications: Set[NodeKey]`; `_queue_subscription(node_key: NodeKey)`; `flush(self) -> None`. `Set` and `NodeKey` already imported in `dag/core.py`; `_subscriptions_lock` is the existing `RLock`. Names consistent across steps.
