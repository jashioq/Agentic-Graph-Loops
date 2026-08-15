"""The run's two documents, over a real `FileStore` in `tmp_path`.

A store is a directory of files and a directory is cheap, so nothing here is
faked: the tests write the documents the way a run writes them and read them
back the way `agl resume` will. The interesting cases are the ones a real store
can produce and a fake would have to be taught — an empty store, a document
from a build that is not this one, a document that is not JSON at all.
"""

from pathlib import Path
from typing import Any

import pytest

from agl.core.store import Store
from agl.core.store.impl.file_store import FileStore
from agl.runtime.record import (
    RUN_KEY,
    STATE_KEY,
    VERSION,
    RecordError,
    RunRecord,
    StateError,
    StateFile,
    read_record,
    write_record,
)

RECORD = RunRecord(
    workflow="tickets",
    label="add-auth",
    request="Add authentication",
    base_branch="feature",
    project="demo",
    max_concurrent=3,
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    """An empty store, as a run has before it has written anything."""
    return FileStore(tmp_path / "run")


# -- the record -----------------------------------------------------------


def test_a_record_survives_the_round_trip(store: Store) -> None:
    write_record(store, RECORD)
    assert read_record(store) == RECORD


def test_a_written_record_carries_this_build_s_version(store: Store) -> None:
    write_record(store, RECORD)
    assert store.read_json(RUN_KEY)["version"] == VERSION


def test_no_record_raises(store: Store) -> None:
    with pytest.raises(RecordError):
        read_record(store)


def test_a_version_this_build_cannot_read_raises(store: Store) -> None:
    store.write_json(RUN_KEY, {"version": VERSION + 1, "workflow": "tickets"})
    with pytest.raises(RecordError):
        read_record(store)


def test_a_record_missing_a_field_raises(store: Store) -> None:
    store.write_json(RUN_KEY, {"version": VERSION, "workflow": "tickets"})
    with pytest.raises(RecordError):
        read_record(store)


def test_a_record_that_is_not_json_raises(store: Store) -> None:
    store.write(RUN_KEY, "{not json")
    with pytest.raises(RecordError):
        read_record(store)


def test_a_record_that_is_not_an_object_raises(store: Store) -> None:
    store.write_json(RUN_KEY, ["tickets"])
    with pytest.raises(RecordError):
        read_record(store)


def test_a_record_field_of_the_wrong_type_raises(store: Store) -> None:
    written = {"version": VERSION, "workflow": "tickets", "label": "add-auth"}
    store.write_json(RUN_KEY, {**written, "max_concurrent": "three"})
    with pytest.raises(RecordError):
        read_record(store)


# -- the state document ---------------------------------------------------


def test_an_unwritten_state_does_not_exist(store: Store) -> None:
    state = StateFile(store)
    assert not state.exists()
    assert state.load() is None


def test_a_saved_payload_comes_back_as_it_was(store: Store) -> None:
    state = StateFile(store)
    payload = {"halt": None, "tickets": [{"id": "T-01", "status": "pending"}]}

    state.save(payload)

    assert state.exists()
    assert state.load() == payload


def test_a_saved_state_carries_this_build_s_version(store: Store) -> None:
    StateFile(store).save({"tickets": []})
    assert store.read_json(STATE_KEY)["version"] == VERSION


def test_saving_replaces_the_whole_document(store: Store) -> None:
    state = StateFile(store)
    state.save({"tickets": ["T-01"], "halt": "conflict"})
    state.save({"tickets": []})
    assert state.load() == {"tickets": []}


def test_a_state_from_a_version_this_build_cannot_read_raises(store: Store) -> None:
    store.write_json(STATE_KEY, {"version": VERSION + 1, "tickets": []})
    with pytest.raises(StateError):
        StateFile(store).load()


def test_a_state_that_is_not_json_raises(store: Store) -> None:
    store.write(STATE_KEY, "{not json")
    with pytest.raises(StateError):
        StateFile(store).load()


def test_a_state_that_is_not_an_object_raises(store: Store) -> None:
    store.write_json(STATE_KEY, ["T-01"])
    with pytest.raises(StateError):
        StateFile(store).load()


def test_update_hands_the_callable_none_when_nothing_is_written(store: Store) -> None:
    seen: list[dict[str, Any] | None] = []

    def change(current: dict[str, Any] | None) -> dict[str, Any]:
        seen.append(current)
        return {"tickets": []}

    StateFile(store).update(change)

    assert seen == [None]


def test_update_hands_the_callable_the_current_payload(store: Store) -> None:
    state = StateFile(store)
    state.save({"tickets": ["T-01"]})
    seen: list[dict[str, Any] | None] = []

    def change(current: dict[str, Any] | None) -> dict[str, Any]:
        seen.append(current)
        return {"tickets": [*(current or {}).get("tickets", []), "T-02"]}

    state.update(change)

    assert seen == [{"tickets": ["T-01"]}]


def test_update_stores_and_returns_what_the_callable_returned(store: Store) -> None:
    state = StateFile(store)
    state.save({"tickets": ["T-01"]})

    returned = state.update(lambda current: {"tickets": ["T-01", "T-02"]})

    assert returned == {"tickets": ["T-01", "T-02"]}
    assert state.load() == {"tickets": ["T-01", "T-02"]}


def test_a_state_file_can_be_given_another_key(store: Store) -> None:
    state = StateFile(store, key="rounds/first/state.json")
    state.save({"tickets": []})
    assert not store.exists(STATE_KEY)
    assert state.load() == {"tickets": []}
