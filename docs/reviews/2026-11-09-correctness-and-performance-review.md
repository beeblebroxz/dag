# Correctness and performance review, 2026-11-09

A read of the whole library at commit `15c76ad`, followed by a reproduction of
every finding. The review itself changed nothing in the library.

- **Status:** fixed after the review, in `dag/core.py` and `dag/state.py`,
  each with regression tests: correctness 1, 2, 3, 5, 6 and 7, performance
  1, and all three findings under "Found while fixing" at the end.
  Still open: correctness 4 and 8, performance 2 and 3.
  - Correctness 7 was fixed by walking the whole scenario stack.
  - Performance 2 and 3 improved as a side effect, to 2,666 bytes per node
    and about 4,000 ns per cached read.

## Suggested order

1. Correctness 1 and 3: silent wrong answers in single-threaded use, each a
   contained change in `dag/core.py`.
2. Performance 1: releasing 8,000 objects takes about 20 seconds.
3. Correctness 2 if cross-thread use is supported, which
   `tests/test_concurrency.py` suggests it is.

## Correctness

### 1. A cell that catches a dependency's error never refreshes

```python
import dag

class M(dag.Model):
    @dag.computed(dag.Input)
    def Denominator(self):
        return 0

    @dag.computed
    def Ratio(self):
        return 10 / self.Denominator()

    @dag.computed
    def Safe(self):
        try:
            return self.Ratio()
        except Exception:
            return "n/a"

m = M()
print(m.Safe())            # n/a
m.Denominator.set(5)
print(m.Ratio())           # 2.0
print(m.Safe())            # n/a  (expected 2.0, and it stays n/a)
```

- **Cause:** `invalidate_node` returns early for a node that is not valid
  (`core.py:534`) and its traversal skips such nodes (`core.py:546`). A node
  whose evaluation raised is in the ERROR state, which is not valid. So when
  `Denominator` changes, invalidation stops at `Ratio` and never reaches
  `Safe`, which is VALID with its fallback.
- **The comment at `core.py:544` states the assumption that fails:** "an
  invalid node's dependents were invalidated when it first became invalid".
  A dependent can become valid *after* its dependency entered ERROR.
- **Direction:** propagate through ERROR nodes, resetting them to INVALID.
  The design notes say exceptions from dependencies cannot be caught, but
  Python lets a cell catch them, so either support it or reject it loudly.

### 2. A `set()` during another thread's evaluation is lost

```python
import threading
import dag

started, release = threading.Event(), threading.Event()

class M(dag.Model):
    @dag.computed(dag.Input)
    def X(self):
        return 1

    @dag.computed
    def Slow(self):
        x = self.X()
        started.set()
        release.wait(5)
        return x * 100

m = M()
worker = threading.Thread(target=m.Slow)
worker.start()
started.wait(5)
m.X.set(2)                 # Slow is mid-evaluation on the other thread
release.set()
worker.join()
print(m.X(), m.Slow())     # 2 100  (expected 2 200, and it stays 100)
```

- **Cause:** `_prepare_dependencies` removes the node from its inputs'
  `outputs` when evaluation starts (`core.py:285`), and
  `_finalize_dependencies` reinstalls the edges when it ends. In between,
  `X.outputs` does not contain `Slow`, so the invalidation from `set()` cannot
  see it. Even with the edge present, the node is EVALUATING, not valid, and
  would be skipped. It then caches a result computed from the old input.
- **Direction:** keep the old edges until the new ones are known, and mark a
  node "dirtied while evaluating" so its result is discarded or the node is
  left INVALID when the evaluation finishes.

### 3. A scenario override outlives its object and lands on a new one

```python
import gc
import dag

class M(dag.Model):
    @dag.computed(dag.Overridable)
    def Spot(self):
        return 100.0

reused = stale = 0
for _ in range(300):
    with dag.scenario():
        a = M()
        a.Spot.override(1.0)
        address = id(a)
        del a
        gc.collect()
        keep = []
        for _ in range(50):          # allocate until the address is reused
            b = M()
            if id(b) == address:
                reused += 1
                stale += b.Spot() == 1.0
                break
            keep.append(b)
        del b, keep
print(reused, stale)       # 298 298 in one run: every reuse read the dead override
```

- **Cause:** scenario overrides live in a per-thread dict keyed by `NodeKey`,
  which holds `id(obj)` (`core.py:163`, `core.py:229`). `Scenario._tweaks`
  keeps the `Node`, and a `Node` holds only a weak reference to its object.
  When the object is freed, its nodes are cleaned up but the override entry
  stays until the scenario exits. A new object at the same address has the
  same key.
- **`CLAUDE.md` says** "Overrides hold a hard reference to the object". Only
  `Branch` does that, through `Override.obj`.
- **Direction:** have `Scenario` hold the object for each override, as the
  documentation says, or drop the object's override entries in
  `_cleanup_object_nodes`.

### 4. Default arguments create two nodes for one call

```python
import dag

class M(dag.Model):
    @dag.computed(dag.Input)
    def Rate(self, tenor="1Y"):
        return 0.05

    @dag.computed
    def Show(self, x):
        return repr(x)

m = M()
m.Rate.set(0.10)
print(m.Rate(), m.Rate("1Y"))             # 0.1 0.05  (the same call)
print(m.Show(1), m.Show(1.0), m.Show(True))   # 1 1 1
```

- **Cause:** the node key is the literal positional tuple
  (`decorators.py:140`). `Rate()` is keyed `()` and `Rate("1Y")` is keyed
  `("1Y",)`. `_validate_args` (`decorators.py:263`) binds the signature but
  does not apply defaults.
- **The second line** is the usual equal-hash pitfall: `1`, `1.0` and `True`
  are equal, so they share a node and the first result is returned for all.
- **Direction:** normalise arguments with `signature.bind(...)` and
  `apply_defaults()` before building the key. Decide whether keys should
  also carry argument types, as `functools.lru_cache(typed=True)` does.

### 5. A watch registered inside a callback is dropped

```python
import dag

class M(dag.Model):
    @dag.computed(dag.Input)
    def A(self):
        return 1

m = M()
m.A()
calls = []

def second(node):
    calls.append("second")

def first(node):
    calls.append("first")
    m.A.watch(second)

m.A.watch(first)
m.A.set(2); dag.flush()
m.A.set(3); dag.flush()
print(calls)               # ['first', 'first']  (expected 'second' too)
```

- **Cause:** `flush` snapshots a node's callbacks, calls them, then writes
  the surviving snapshot back (`core.py:764`). A subscription added during
  dispatch is overwritten. An `unwatch` during dispatch is undone the same
  way, and so is a subscription added by another thread.
- **Direction:** remove only the dead references from the current list, under
  the lock, instead of replacing the list.

### 6. `Optional` hides programming errors

```python
import dag

class M(dag.Model):
    @dag.computed
    def A(self):
        return 1

    def helper(self):
        return self.A()

    @dag.computed(dag.Optional)
    def B(self):
        return self.helper() + 1      # undeclared dependency

m = M()
print(repr(m.B()))         # NO_VALUE  (without Optional: UntrackedError)
```

- **Cause:** the `except Exception` branch turns any exception from an
  `Optional` node into `NO_VALUE` (`core.py:498`), including
  `UntrackedError` and `CycleError`, which report mistakes in the model
  rather than failed calculations.
- **Direction:** re-raise those two before the `Optional` check.

### 7. `get_overrides()` returns only the innermost scenario's overrides

```python
import dag

class M(dag.Model):
    @dag.computed(dag.Overridable)
    def A(self):
        return 1

    @dag.computed(dag.Overridable)
    def B(self):
        return 2

m = M()
with dag.scenario():
    m.A.override(10)
    with dag.scenario():
        m.B.override(20)
        print(sorted(o.method_name for o in dag.get_overrides().overrides))
        # ['B']  (A is also in effect)
```

- **Cause:** it reads `ctx._tweaks` of the current scenario only
  (`state.py:228`).
- **Why it matters:** the stated use is to capture a scenario and reapply it
  elsewhere. A captured inner scenario silently loses the outer overrides.
- **Direction:** walk the whole scenario stack, or document that the result
  is the innermost delta.

### 8. Dependency chains fail at a depth of 200 to 300

```python
import dag

class M(dag.Model):
    @dag.computed
    def F(self, n):
        return 0 if n == 0 else self.F(n - 1) + 1

M().F(200)                 # ok
M().F(300)                 # EvaluationError caused by RecursionError
```

- **Cause:** evaluation recurses through Python calls, about four frames per
  node, against the default recursion limit of 1,000. `invalidate_node` was
  made iterative for deep chains; evaluation was not.
- **The error is also wrapped once per level,** so the message is a chain of
  several hundred nested `EvaluationError`s.
- **Direction:** this follows from top-down evaluation, so the realistic
  options are to document the limit, or to raise one clear error that names
  it.

## Performance

### 1. Releasing objects is quadratic

Three nodes per object (`A` an input, `B = A + 1`, `C = B + 1`), all evaluated,
then the objects are dropped and collected:

| Objects | Evaluate | Release |
|---:|---:|---:|
| 2,000 | 88 ms | 1.1 s |
| 4,000 | 188 ms | 4.6 s |
| 8,000 | 438 ms | 19.7 s |

- **Cause:** `get_or_create_node` gives every node its own
  `weakref.ref(obj, callback)` (`core.py:379`). When an object dies, one
  callback fires per node, and each callback scans every key in the graph
  for that object's id (`core.py:338`). The cost is nodes-per-object times
  total nodes, for every object.
- **Direction:** one weak reference per object, and an index from object id
  to its node keys, so cleanup touches only that object's nodes.

### 2. A node costs about 3.4 KB

Measured with `tracemalloc` over 20,000 parameterized nodes on one object:
3,354 bytes each. `CLAUDE.md` says about 40 bytes plus a hash table entry.

- **Cause:** each `Node` owns a `threading.Condition` (`core.py:102`), two
  sets, a weak reference and a closure, on top of the dataclass and its key.
- **Direction:** the condition is only needed while a node is being
  evaluated by one thread and awaited by another; one shared condition, or
  one created on first contention, would remove most of it. Correct the
  figure in `CLAUDE.md` either way, since it drives the "fat nodes" advice.

### 3. Reading a cached value costs about 4.6 µs

A cached `m.V()` took 4,631 ns against 65 ns for a plain method call, 71
times as long.

- **Cause:** each read allocates a `ComputedFunctionAccessor`
  (`decorators.py:96`), builds a `NodeKey` and hashes it several times (the
  frozen dataclass does not cache its hash), takes the node registry lock
  and the node's condition, and looks up thread-local state three or four
  times.
- **Direction:** cache the key's hash, look up thread state once per
  `evaluate`, and add a lock-free fast path for a valid node with no active
  override.

## Not reviewed

- `dag/ui/` and the three examples.
- The parser beyond reading it: no false negative or false positive was
  tested.

## Found while fixing

### A cached read racing an invalidation could return `NO_VALUE` (fixed)

`evaluate` checked `node.is_valid` and then read `node.value`. Invalidation
takes no lock, so it could clear the node between the two and the reader got
the cleared `NO_VALUE`. The stress test added for correctness 2
(`test_concurrent_writes_and_reads_converge`) hit it about one run in three.

- **First fix, incomplete:** reading the value before checking the state.
  The test still failed about one run in fifty.
- **Actual cause:** invalidation wrote the state and then the value as two
  unlocked steps. Another thread could re-evaluate the node between them
  and be left VALID with its value wiped, permanently.
- **Fix:** `Node.invalidate` changes both under the node's lock when the
  lock is free. It never waits for it, because invalidation can run from a
  garbage collection callback; if the lock is busy it flips only the state
  and leaves the stale value for the next evaluation to replace. 200 full
  runs and 400 runs of the stress test then passed.

### An on-graph scenario leaves its node deaf to later changes (fixed)

```python
import dag

class M(dag.Model):
    @dag.computed(dag.Overridable)
    def Spot(self):
        return 100.0

    @dag.computed(dag.Input)
    def Vol(self):
        return 2

    @dag.computed
    def Price(self):
        return self.Spot() * self.Vol()

    @dag.computed
    def Up(self):
        with dag.scenario():
            self.Spot.override(101.0)
            return self.Price()

m = M()
print(m.Up())              # 202.0
m.Vol.set(3)
print(m.Up())              # 202.0  (expected 303.0, and it stays 202.0)
```

- **Cause:** when the scenario inside `Up` exits, reverting the override
  invalidates `Price`. `Up` then finishes and becomes VALID above an INVALID
  `Price`. A later change to `Vol` stops at `Price`, which is already
  INVALID, and never reaches `Up`.
- **Single-threaded, and present at `15c76ad`.** It only shows when the
  later change is to an input the node reaches solely through a cell the
  scenario recomputed; a node that also reads `Vol` directly is fine.
- **Fix:** when an invalidation walk started inside an evaluation resets
  nodes on its way to the evaluating node, that node inherits their inputs
  as direct dependencies (`DagManager._inherit_inputs`). Inheriting at that
  moment, rather than letting the walk pass through INVALID nodes later,
  also covers a cell whose inputs differ under the override: once it is
  recomputed outside the scenario its own edges no longer describe what the
  scenario's owner consumed.
- **Cost:** a cell that bumps and reprices keeps one direct edge per input
  of every cell it repriced.

### A temporary model created inside a cell has the same problem (fixed)

```python
import dag

class Market(dag.Model):
    @dag.computed(dag.Input)
    def Spot(self):
        return 100.0

market = Market()

class Pricer(dag.Model):
    @dag.computed
    def Price(self):
        return market.Spot() * 2

class Holder(dag.Model):
    @dag.computed
    def Total(self):
        return Pricer().Price() + 1

h = Holder()
print(h.Total())           # 201.0
market.Spot.set(50.0)
print(h.Total())           # 201.0  (expected 101.0, and it stays 201.0)
```

- **Cause:** the `Pricer` dies when `Total` returns. Cleanup removes its
  nodes and their edges, so nothing connects `market.Spot` to `Total` any
  more. Present at `15c76ad`.
- **Fix:** the same inheritance: in `_cleanup_object_nodes`, a dependent
  that this thread is still evaluating takes over the inputs of the dead
  nodes it read.
