"""
Tests for thread safety and concurrency.
"""

import threading
import time
import dag


class TestThreadSafety:
    """Test thread-safe access to the DAG."""

    def setup_method(self):
        dag.reset()

    def test_concurrent_reads_after_cached(self):
        """Test concurrent reads from multiple threads after value is cached."""
        results = {}
        errors = []

        class Simple(dag.Model):
            @dag.computed
            def Value(self):
                return 42

        obj = Simple()

        # Pre-cache the value
        assert obj.Value() == 42

        def reader(thread_id):
            try:
                for _ in range(10):
                    result = obj.Value()
                    results[thread_id] = result
                    assert result == 42
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(v == 42 for v in results.values())

    def test_concurrent_first_reads_share_evaluation(self):
        """Test concurrent uncached reads of the same node."""
        results = []
        errors = []
        start = threading.Barrier(5)
        call_count = {'value': 0}
        call_count_lock = threading.Lock()

        class Simple(dag.Model):
            @dag.computed
            def Value(self):
                time.sleep(0.01)
                with call_count_lock:
                    call_count['value'] += 1
                return 42

        obj = Simple()

        def reader():
            try:
                start.wait()
                results.append(obj.Value())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert results == [42] * 5
        assert call_count['value'] == 1

    def test_concurrent_different_objects(self):
        """Test concurrent access to different objects."""
        results = {}
        errors = []

        class Counter(dag.Model):
            def __init__(self, value):
                super().__init__()
                self._value = value

            @dag.computed
            def Value(self):
                time.sleep(0.001)
                return self._value

        objects = [Counter(i) for i in range(10)]

        def accessor(obj, thread_id):
            try:
                result = obj.Value()
                results[thread_id] = result
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=accessor, args=(objects[i], i))
            for i in range(10)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        for i in range(10):
            assert results[i] == i

    def test_concurrent_set_and_read(self):
        """Test concurrent set and reads."""
        errors = []
        read_values = []

        class Mutable(dag.Model):
            @dag.computed(dag.Input)
            def Value(self):
                return 0

        obj = Mutable()

        def setter():
            try:
                for i in range(100):
                    obj.Value = i
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    value = obj.Value()
                    read_values.append(value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        setter_thread = threading.Thread(target=setter)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        setter_thread.start()
        for t in reader_threads:
            t.start()

        setter_thread.join()
        for t in reader_threads:
            t.join()

        assert len(errors) == 0
        # All read values should be valid integers
        assert all(isinstance(v, int) for v in read_values)

    def test_dag_manager_singleton_thread_safe(self):
        """Test that DagManager singleton is thread-safe."""
        from dag.core import DagManager

        instances = []
        errors = []

        def get_instance():
            try:
                instance = DagManager.get_instance()
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(20)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        # All should be the same instance
        assert all(inst is instances[0] for inst in instances)


class TestConcurrentInvalidation:
    """Test concurrent invalidation scenarios."""

    def setup_method(self):
        dag.reset()

    def test_concurrent_invalidation_and_evaluation(self):
        """Test invalidating while evaluating."""
        errors = []
        results = []

        class Chain(dag.Model):
            @dag.computed(dag.Input)
            def Base(self):
                return 1

            @dag.computed
            def Derived(self):
                time.sleep(0.01)  # Slow computation
                return self.Base() * 2

        obj = Chain()

        def evaluator():
            try:
                for _ in range(10):
                    result = obj.Derived()
                    results.append(result)
            except Exception as e:
                errors.append(e)

        def invalidator():
            try:
                for i in range(10):
                    obj.Base = i
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        eval_thread = threading.Thread(target=evaluator)
        inv_thread = threading.Thread(target=invalidator)

        eval_thread.start()
        inv_thread.start()

        eval_thread.join()
        inv_thread.join()

        # Should not crash - results may vary due to race conditions
        assert len(errors) == 0


class TestParallelComputation:
    """Test parallel computation patterns."""

    def setup_method(self):
        dag.reset()

    def test_independent_computation_chains(self):
        """Test multiple independent computation chains in parallel."""
        results = {}
        errors = []

        class Independent(dag.Model):
            def __init__(self, base):
                super().__init__()
                self._base = base

            @dag.computed
            def Base(self):
                return self._base

            @dag.computed
            def Step1(self):
                time.sleep(0.01)
                return self.Base() + 1

            @dag.computed
            def Step2(self):
                time.sleep(0.01)
                return self.Step1() + 1

            @dag.computed
            def Final(self):
                return self.Step2() + 1

        objects = [Independent(i * 10) for i in range(5)]

        def compute(obj, idx):
            try:
                result = obj.Final()
                results[idx] = result
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=compute, args=(objects[i], i))
            for i in range(5)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        for i in range(5):
            expected = i * 10 + 3  # base + 3 steps
            assert results[i] == expected

    def test_shared_dependency_concurrent_access(self):
        """Test multiple objects sharing a dependency."""
        errors = []
        results = []

        class Shared(dag.Model):
            @dag.computed(dag.Input)
            def Config(self):
                return 100

        class Consumer(dag.Model):
            def __init__(self, shared, multiplier):
                super().__init__()
                self._shared = shared
                self._mult = multiplier

            @dag.computed
            def Result(self):
                time.sleep(0.01)
                return self._shared.Config() * self._mult

        shared = Shared()
        consumers = [Consumer(shared, i) for i in range(1, 6)]

        def compute(consumer, idx):
            try:
                result = consumer.Result()
                results.append((idx, result))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=compute, args=(consumers[i], i))
            for i in range(5)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5


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

    def test_concurrent_scenario_enter_raises(self):
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
        ta.start()
        tb.start()
        ta.join()
        tb.join()

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
        t.start()
        t.join()
        assert result == [9]

    def test_concurrent_branch_enter_raises(self):
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
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert len(errors) == 1

    def test_evaluate_during_foreign_scenario_raises(self):
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
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert observed == [1000.0]
        assert len(errors) == 1

    def test_set_during_foreign_scenario_raises(self):
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
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert len(errors) == 1

    def test_node_change_apply_during_foreign_scenario_raises(self):
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
                dag.NodeChange(obj.X, 99).apply()
            except dag.ConcurrentScenarioError as e:
                errors.append(e)
            finally:
                release_a.set()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert len(errors) == 1

    def test_clearvalue_during_foreign_scenario_raises(self):
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
                obj.X.clearValue()
            except dag.ConcurrentScenarioError as e:
                errors.append(e)
            finally:
                release_a.set()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert len(errors) == 1

    def test_release_ownership_ignores_non_owner_thread(self):
        from dag.core import DagManager

        mgr = DagManager.get_instance()
        mgr._claim_scenario_ownership()  # main thread owns it
        try:
            def foreign_release():
                mgr._release_scenario_ownership()

            t = threading.Thread(target=foreign_release)
            t.start()
            t.join()

            # A non-owning thread must not be able to clear/decrement ownership.
            assert mgr._scenario_owner == threading.get_ident()
            assert mgr._scenario_depth == 1
        finally:
            mgr._release_scenario_ownership()  # owner releases
        assert mgr._scenario_owner is None


class TestCrossThreadDeadlock:
    """Mutually-dependent nodes evaluated from two threads must raise
    CycleError instead of deadlocking on each other's condition variables
    (audit 2026-06)."""

    def setup_method(self):
        dag.reset()

    def test_mutual_dependency_across_threads_raises_instead_of_hanging(self):
        sync = threading.Barrier(2)

        def synced():
            try:
                sync.wait(timeout=1.0)
            except threading.BrokenBarrierError:
                pass

        class Cyclic(dag.Model):
            @dag.computed
            def A(self):
                synced()
                return self.B() + 1

            @dag.computed
            def B(self):
                synced()
                return self.A() + 1

        m = Cyclic()
        errors = {}

        def run(name, accessor):
            try:
                accessor()
            except Exception as exc:  # noqa: BLE001 - recording for assertions
                errors[name] = exc

        t1 = threading.Thread(target=run, args=("A", m.A), daemon=True)
        t2 = threading.Thread(target=run, args=("B", m.B), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not t1.is_alive() and not t2.is_alive(), \
            "threads deadlocked on mutually-dependent nodes"
        # Both evaluations fail: the dependency really is cyclic.
        assert set(errors) == {"A", "B"}
        for exc in errors.values():
            assert isinstance(exc, (dag.CycleError, dag.EvaluationError))


class TestChangeDuringEvaluation:
    """A change made by one thread while another thread is evaluating a
    dependent must not be lost (review 2026-11-09, correctness 2)."""

    def setup_method(self):
        dag.reset()

    @staticmethod
    def _slow_model():
        started, release = threading.Event(), threading.Event()

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def X(self):
                return 1

            @dag.computed
            def Slow(self):
                x = self.X()
                started.set()
                release.wait(5)
                return x * 100

            @dag.computed
            def Outer(self):
                return self.Slow() + 1

        return Model(), started, release

    @staticmethod
    def _run(target):
        """Start target on a thread; return (thread, results list)."""
        results = []
        thread = threading.Thread(target=lambda: results.append(target()))
        thread.start()
        return thread, results

    def test_set_during_first_evaluation_is_not_lost(self):
        m, started, release = self._slow_model()
        worker, results = self._run(m.Slow)
        assert started.wait(5)

        m.X.set(2)  # Slow is mid-evaluation on the worker thread
        release.set()
        worker.join(5)

        # The in-flight call returns what it computed from the old input,
        # but that result must not be cached.
        assert results == [100]
        assert m.Slow() == 200

    def test_set_during_reevaluation_is_not_lost(self):
        m, started, release = self._slow_model()
        release.set()
        assert m.Slow() == 100
        m.X.set(5)  # invalidate, so the worker recomputes
        started.clear()
        release.clear()

        worker, _results = self._run(m.Slow)
        assert started.wait(5)
        m.X.set(2)
        release.set()
        worker.join(5)

        assert m.Slow() == 200

    def test_callers_up_the_evaluating_stack_are_not_cached_either(self):
        m, started, release = self._slow_model()
        worker, results = self._run(m.Outer)
        assert started.wait(5)

        m.X.set(2)
        release.set()
        worker.join(5)

        assert results == [101]
        assert m.Outer() == 201

    def test_thread_waiting_on_the_evaluation_recomputes(self):
        from dag.core import DagManager

        m, started, release = self._slow_model()
        first, first_results = self._run(m.Slow)
        assert started.wait(5)

        second, second_results = self._run(m.Slow)
        manager = DagManager.get_instance()
        deadline = time.time() + 5
        while second.ident not in manager._threads_waiting:
            assert time.time() < deadline, "second thread never blocked"
            time.sleep(0.001)

        m.X.set(2)
        release.set()
        first.join(5)
        second.join(5)

        assert first_results == [100]
        assert second_results == [200]
        assert m.Slow() == 200

    def test_watcher_is_notified_when_an_evaluation_is_discarded(self):
        m, started, release = self._slow_model()
        events = []

        def on_change(node):
            events.append(node.method_name)

        m.Slow.watch(on_change)
        worker, _results = self._run(m.Slow)
        assert started.wait(5)

        m.X.set(2)
        release.set()
        worker.join(5)
        dag.flush()

        # Slow ended up INVALID again, so its watcher must hear about it.
        assert events == ['Slow']

    def test_on_graph_scenario_does_not_discard_its_own_evaluation(self):
        """Changes made on the evaluating thread itself (here an on-graph
        scenario) must leave the node cacheable."""
        calls = []

        class Model(dag.Model):
            @dag.computed(dag.Overridable)
            def Spot(self):
                return 100.0

            @dag.computed
            def Price(self):
                return self.Spot() * 2

            @dag.computed
            def Delta(self):
                calls.append(1)
                spot = self.Spot()
                base = self.Price()
                with dag.scenario():
                    self.Spot.override(spot + 1)
                    up = self.Price()
                return up - base

        m = Model()
        assert m.Delta() == 2.0
        assert m.Delta() == 2.0
        assert len(calls) == 1

    def test_temporary_model_does_not_discard_its_creators_evaluation(self):
        """A model created and released inside an evaluation invalidates its
        dependents when it dies; the creator must still be cached."""
        calls = []

        class Leaf(dag.Model):
            @dag.computed
            def Value(self):
                return 7

        class Holder(dag.Model):
            @dag.computed
            def Total(self):
                calls.append(1)
                return Leaf().Value() + 1

        holder = Holder()
        assert holder.Total() == 8
        assert holder.Total() == 8
        assert len(calls) == 1

    def test_invalidation_terminates_on_cyclic_in_flight_edges(self):
        """Nodes being evaluated by another thread are passed through without
        changing state, so a cycle among them must not loop forever."""
        from dag.core import DagManager, NodeState

        class Model(dag.Model):
            @dag.computed
            def A(self):
                return 1

            @dag.computed
            def B(self):
                return 2

        m = Model()
        a, b = m.A._get_or_create_node(), m.B._get_or_create_node()
        for node in (a, b):
            node._state = NodeState.EVALUATING
            node._evaluating_thread_id = -1  # some other thread
        a.outputs.add(b.key)
        b.outputs.add(a.key)

        DagManager.get_instance().invalidate_node(a)  # must return

        assert a._invalidation_count >= 1
        assert b._invalidation_count >= 1

    def test_concurrent_writes_and_reads_converge(self):
        """After writers and readers race, the cache must agree with the
        final input."""
        import sys

        class Model(dag.Model):
            @dag.computed(dag.Input)
            def X(self):
                return 0

            @dag.computed
            def Double(self):
                return self.X() * 2

            @dag.computed
            def Total(self):
                return self.Double() + self.X()

        m = Model()
        errors = []
        done = threading.Event()

        def reader():
            try:
                while not done.is_set():
                    m.Total()
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-5)
        try:
            readers = [threading.Thread(target=reader) for _ in range(4)]
            for t in readers:
                t.start()
            stale = 0
            for i in range(1, 1501):
                m.X.set(i)
                if i % 50 == 0:
                    # No write is in flight now, so whatever the readers
                    # cache from here on must reflect X == i.
                    time.sleep(0.001)
                    stale += m.Total() != 3 * i
            done.set()
            for t in readers:
                t.join(5)
        finally:
            sys.setswitchinterval(old_interval)

        assert errors == []
        assert stale == 0
        assert m.Total() == 3 * 1500
