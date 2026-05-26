# Design: Pending-Queue Watch Dispatch

- **Date:** 2026-05-26
- **Status:** Approved (pending implementation plan)
- **Issue:** Subscription `flush()` performance + dispatch semantics (deferred from the 2026-05 audit)

## Problem

The watch/notification system has three defects:

1. **`flush()` is O(all watchers).** It iterates *every* subscribed node on every call and fires callbacks for any that are currently invalid, regardless of whether anything changed.
2. **`_queue_subscription` is a no-op** (`pass`). The "lazy, cumulative queue" the spec describes was never implemented; `flush` relies entirely on scanning `node.is_valid`.
3. **Re-fires on every flush while dirty.** A watched node that stays invalid (its callback didn't re-evaluate it) re-fires its callbacks on *every* subsequent `flush`.

The framework spec intends an event-queue model: "watches are lazy and cumulative — only one notification per traversal; callbacks must evaluate the node if they expect to be called again."

## Decision

Implement a real pending-notification set: enqueue a watched node when it changes, and have `flush()` dispatch-and-clear that set. This makes `flush` O(changed) and adopts the spec's notify-once / evaluate-to-rearm semantics.

We do **not** change `override()` to self-notify the overridden node (it keeps notifying dependents only, as today), and we do not add async/batched dispatch beyond the pending set.

## Design

### State

`DagManager` gains, in `_init` (so `reset()` clears it):

```python
self._pending_notifications: Set[NodeKey] = set()
```

Guarded by the existing `self._subscriptions_lock` (a reentrant `RLock`).

### What enqueues

A node is enqueued only if it currently has subscribers (keeps the pending set bounded by *watched* nodes):

```python
def _queue_subscription(self, node_key: NodeKey) -> None:
    with self._subscriptions_lock:
        if node_key in self._subscriptions:
            self._pending_notifications.add(node_key)
```

Enqueue points:
- **`invalidate_node`** already calls `_queue_subscription(node.key)` for every node it invalidates. This covers all derived/dependent nodes and `clearValue()` (which invalidates its own node).
- **`set()`** (in `dag/decorators.py`) gains one line after the direct-set branch's `invalidate_dependents`: `self._dag._queue_subscription(node.key)`. This ensures watchers of a set `Input` fire even when the node was never evaluated and has no dependents (the `test_watch_before_evaluation` case). The inverse path of `set()` enqueues via `NodeChange.apply()` → `invalidate_node`.
- **`override()`** is unchanged: `add_tweak` → `invalidate_dependents` enqueues the overridden node's *dependents*. The overridden node itself is not enqueued (matches current behavior — the main use case watches derived output nodes).

### flush

```python
def flush(self) -> None:
    with self._subscriptions_lock:
        pending = list(self._pending_notifications)
        self._pending_notifications.clear()

    for node_key in pending:
        node = self.get_node(node_key)
        if node is None:
            continue
        with self._subscriptions_lock:
            callbacks = list(self._subscriptions.get(node_key, []))
        live = []
        for cb_ref in callbacks:
            cb = cb_ref()
            if cb is None:
                continue
            live.append(cb_ref)
            try:
                cb(node)
            except Exception:
                # Watch callbacks must not interrupt the DAG, but their
                # failures must be visible rather than silently dropped.
                logger.exception("Watch callback for %r failed", node.method_name)
        with self._subscriptions_lock:
            if node_key in self._subscriptions:
                self._subscriptions[node_key] = live
```

Notes:
- The pending set is the filter — there is no `node.is_valid` re-check. A queued node fires regardless of its current validity (the event means "this node changed since you were last notified").
- Callbacks are invoked **outside** the lock so a callback may re-evaluate, subscribe, or otherwise re-enter without deadlock. (`_subscriptions_lock` is reentrant, so even nested acquisition from a callback is safe.)
- Dead weakrefs are pruned per node during dispatch, as today.

## Observable behavior changes (all spec-aligned)

1. **Notify once per change.** A watcher fires once when its node changes; it re-arms only when the node is re-evaluated and then changed again. (Was: re-fired on every `flush` while the node stayed invalid.)
2. **Only *changed* watched nodes fire.** An unrelated, never-evaluated (hence invalid) watched node no longer fires on an unrelated `flush`. (Was: the scan-all-invalid sweep included it.)
3. **`flush` is O(pending)**, not O(all watchers).

None of these break the existing watch tests (see Testing).

## Edge cases & semantics

- **Never-evaluated set Input** (`test_watch_before_evaluation`): preserved because `set()` explicitly enqueues its node.
- **Unrelated invalid watched node** (`test_watch_does_not_fire_when_valid`): preserved — the unaffected node is never enqueued, so it never fires (previously it didn't fire because it stayed valid; now because it isn't pending).
- **Callback raising** (`test_callback_exception_*`): unchanged — logged, never propagated.
- **Callback subscribes/evaluates during dispatch:** safe — callbacks run outside the lock and `RLock` tolerates re-entry. Subscriptions to *other* nodes are unaffected. (Subscribing a new callback to the *same* node currently being dispatched is an unsupported corner that the existing code also drops on the post-dispatch overwrite; out of scope.)
- **`flush` is not scenario-guarded** (consistent with the prior #1 decision): it reads validity and dispatches; any callback that evaluates a node hits the scenario guard via `evaluate`.
- **`reset()`** clears `_pending_notifications` via `_init`.

## Testing

Keep all 8 existing tests in `tests/test_watch.py` (verified compatible). Add:

- **dispatch-and-clear:** invalidate a watched node once, call `flush()` twice → the callback fires exactly once (the second flush has an empty pending set).
- **re-arm by re-evaluation:** after the first notify, re-evaluate the node (valid again), change its input again, `flush()` → the callback fires a second time.
- **only-changed-fires:** two watched derived nodes on independent inputs; change one input, `flush()` → only that node's callback fires (the other, though watched, is not pending).

## Files touched

- `dag/core.py` — `_init` (add `_pending_notifications`), `_queue_subscription` (real enqueue), `flush` (dispatch-and-clear).
- `dag/decorators.py` — one line in `set()` to enqueue the set node.
- `tests/test_watch.py` — three new tests.

## Out of scope

- Notifying the overridden node's own watchers on `override()` (keeps current behavior).
- Async, batched, or coalesced dispatch beyond the per-flush pending set.
- Any change to how invalidation propagates through the graph.
