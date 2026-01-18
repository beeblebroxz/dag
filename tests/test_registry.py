"""
Tests for Registry and RegistryMixin functionality.
"""

import pytest
import dag
from dag.model import Registry, RegistryMixin


class TestRegistry:
    """Test the Registry class."""

    def setup_method(self):
        dag.reset()

    def test_registry_basic_storage(self):
        """Test basic object storage and retrieval."""

        class Entity(dag.Model):
            @dag.computed(dag.Input)
            def Name(self):
                return "default"

        db = Registry()
        entity = Entity()
        entity.Name = "test"

        db['/entities/test'] = entity

        assert db['/entities/test'] is entity
        assert db['/entities/test'].Name() == "test"

    def test_registry_register_and_new(self):
        """Test factory registration and object creation."""

        class Product(dag.Model):
            @dag.computed(dag.Input)
            def Price(self):
                return 0.0

        db = Registry()
        db.register('Product', Product)

        product = db.new('Product', '/products/item1')

        assert isinstance(product, Product)
        assert db['/products/item1'] is product

    def test_registry_new_without_path(self):
        """Test creating object without storing in registry."""

        class Item(dag.Model):
            @dag.computed
            def Value(self):
                return 1

        db = Registry()
        db.register('Item', Item)

        item = db.new('Item')  # No path

        assert isinstance(item, Item)
        assert len(db) == 0  # Not stored

    def test_registry_unknown_class_raises(self):
        """Test that creating unknown class raises KeyError."""
        db = Registry()

        with pytest.raises(KeyError) as exc_info:
            db.new('Unknown')

        assert "Unknown" in str(exc_info.value)

    def test_registry_overwrite(self):
        """Test overwriting existing path."""

        class Simple(dag.Model):
            @dag.computed(dag.Input)
            def Id(self):
                return 0

        db = Registry()

        obj1 = Simple()
        obj1.Id = 1

        obj2 = Simple()
        obj2.Id = 2

        db['/path'] = obj1
        assert db['/path'].Id() == 1

        db['/path'] = obj2
        assert db['/path'].Id() == 2


class TestRegistryMixin:
    """Test the RegistryMixin functionality."""

    def setup_method(self):
        dag.reset()

    def test_registry_mixin_access(self):
        """Test accessing registry via mixin."""

        class WithDb(dag.Model, RegistryMixin):
            @dag.computed
            def Value(self):
                return 1

        class Other(dag.Model):
            @dag.computed
            def OtherValue(self):
                return 100

        db = Registry()
        db.register('Other', Other)
        other = db.new('Other', '/other')

        WithDb.set_registry(db)

        obj = WithDb()
        assert obj.db['/other'].OtherValue() == 100

    def test_registry_mixin_without_registry_raises(self):
        """Test accessing registry without setting one."""

        class Orphan(dag.Model, RegistryMixin):
            @dag.computed
            def Value(self):
                return 1

        # Reset registry reference
        Orphan._database = None

        obj = Orphan()
        with pytest.raises(RuntimeError) as exc_info:
            _ = obj.db

        assert "No registry" in str(exc_info.value)

    def test_registry_mixin_cross_object_dependency(self):
        """Test computed functions that depend on registry objects."""

        class PairObject(dag.Model, RegistryMixin):
            @dag.computed(dag.Input)
            def Spot(self):
                return 1.0

        class Option(dag.Model, RegistryMixin):
            @dag.computed(dag.Input)
            def Strike(self):
                return 1.0

            @dag.computed
            def Price(self):
                spot = self.db['/pairs/EURUSD'].Spot()
                return max(0, spot - self.Strike())

        db = Registry()
        db.register('PairObject', PairObject)
        db.register('Option', Option)

        # Set registry for both classes
        PairObject.set_registry(db)
        Option.set_registry(db)

        pair = db.new('PairObject', '/pairs/EURUSD')
        pair.Spot = 1.1

        option = db.new('Option', '/options/opt1')
        option.Strike = 1.0

        # Option price depends on pair spot
        assert abs(option.Price() - 0.1) < 1e-9

        # Change spot
        pair.Spot = 1.2
        assert abs(option.Price() - 0.2) < 1e-9

    def test_registry_weak_reference(self):
        """Test that registry uses weak reference."""

        class WithDb(dag.Model, RegistryMixin):
            @dag.computed
            def Value(self):
                return 1

        db = Registry()
        WithDb.set_registry(db)

        obj = WithDb()
        assert obj.db is db

        # After deleting db, weak ref should be dead
        del db

        # Access should fail
        with pytest.raises(RuntimeError):
            _ = obj.db


class TestIndirectionPattern:
    """Test the indirection pattern through registry."""

    def setup_method(self):
        dag.reset()

    def test_chained_registry_lookup(self):
        """Test chained lookups like self.db['path'].Method()."""

        class Config(dag.Model, RegistryMixin):
            @dag.computed(dag.Input)
            def Value(self):
                return 10

        class Consumer(dag.Model, RegistryMixin):
            @dag.computed
            def ComputedValue(self):
                return self.db['/config'].Value() * 2

        db = Registry()
        Config.set_registry(db)
        Consumer.set_registry(db)

        config = Config()
        db['/config'] = config

        consumer = Consumer()

        assert consumer.ComputedValue() == 20

        # Change config
        config.Value = 50
        assert consumer.ComputedValue() == 100

    def test_multiple_registry_lookups(self):
        """Test computed function depending on multiple registry objects."""

        class Rate(dag.Model, RegistryMixin):
            @dag.computed(dag.Input)
            def Value(self):
                return 0.0

        class Calculator(dag.Model, RegistryMixin):
            @dag.computed
            def Result(self):
                rate_a = self.db['/rates/A'].Value()
                rate_b = self.db['/rates/B'].Value()
                return rate_a + rate_b

        db = Registry()
        Rate.set_registry(db)
        Calculator.set_registry(db)

        rate_a = Rate()
        rate_a.Value = 1.5
        db['/rates/A'] = rate_a

        rate_b = Rate()
        rate_b.Value = 2.5
        db['/rates/B'] = rate_b

        calc = Calculator()
        assert calc.Result() == 4.0

        # Change one rate
        rate_a.Value = 3.0
        assert calc.Result() == 5.5
