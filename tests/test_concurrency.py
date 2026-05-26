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
