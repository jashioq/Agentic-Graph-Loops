"""The API is five abstract methods, a JSON layer over them, and two errors."""

import inspect
import json

import pytest

from agl.core.store import InvalidKeyError, MissingKeyError, Store
from tests.fakes import MemoryStore


def test_store_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Store()  # type: ignore[abstract]


def test_an_incomplete_implementation_fails_at_instantiation() -> None:
    class Partial(Store):
        def read(self, key: str) -> str:
            return ""

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_exactly_five_methods_are_abstract() -> None:
    assert Store.__abstractmethods__ == frozenset(
        {"read", "write", "delete", "exists", "list"}
    )


def test_json_helpers_are_concrete_on_the_base_class() -> None:
    assert not getattr(Store.read_json, "__isabstractmethod__", False)
    assert not getattr(Store.write_json, "__isabstractmethod__", False)


def test_the_errors_are_exceptions() -> None:
    assert issubclass(MissingKeyError, Exception)
    assert issubclass(InvalidKeyError, Exception)


def test_write_json_stores_text_through_write() -> None:
    store = MemoryStore()
    store.write_json("tickets.json", {"id": "T-03"})
    assert json.loads(store.read("tickets.json")) == {"id": "T-03"}


def test_read_json_round_trips_nested_structures() -> None:
    store = MemoryStore()
    value = {"tickets": [{"id": "T-03", "blockers": ["T-01"]}], "count": 1}
    store.write_json("tickets.json", value)
    assert store.read_json("tickets.json") == value


def test_read_json_reports_a_missing_key_through_read() -> None:
    with pytest.raises(MissingKeyError):
        MemoryStore().read_json("nope.json")


def test_json_helpers_use_only_the_abstract_methods() -> None:
    # If they reached for a filesystem, a fake backed by a dict could not serve them.
    source = inspect.getsource(Store.read_json) + inspect.getsource(Store.write_json)
    assert "open(" not in source
    assert "Path" not in source
