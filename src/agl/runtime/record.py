"""A run's two documents: the record it was started from, and its state.

Layer: runtime. Imports `agl.core.store` and nothing else.

The **record** is written once after preflight and never again — what a run was
asked for, which `agl resume` must read before it can build a `RunContext`. The
**state** is the run's only mutable truth, rewritten whole on every move; this
module owns reading, writing and the version stamp, and leaves the shape to the
workflow. A document from another build is refused, never guessed at. The stamp
is invisible to callers: `save` puts it on, `load` takes it off.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, NoReturn

from agl.core.store import MissingKeyError, Store
from agl.runtime.json_fields import OnError, as_text, as_whole_number

__all__ = [
    "RUN_KEY",
    "STATE_KEY",
    "VERSION",
    "RecordError",
    "RunRecord",
    "StateError",
    "StateFile",
    "read_record",
    "write_record",
]

VERSION = 1
RUN_KEY = "run.json"
STATE_KEY = "state.json"

_VERSION_FIELD = "version"


class RecordError(Exception):
    """Raised when there is no record, or one this version cannot read."""


class StateError(Exception):
    """Raised when the state document is unreadable."""


def _bad_record(message: str) -> NoReturn:
    """This module's `on_error` for the record: every way of failing is one error."""
    raise RecordError(message)


def _bad_state(message: str) -> NoReturn:
    """This module's `on_error` for the state document."""
    raise StateError(message)


@dataclass(frozen=True)
class RunRecord:
    """What a run was asked for, fixed at preflight and true for its whole life.

    Every field is one `agl resume` needs before it has a context at all.
    """

    workflow: str
    label: str
    request: str
    base_branch: str
    project: str
    max_concurrent: int # TODO this is workflow specific. it should be metadata. in cli also run should take different params for each workflow.


def write_record(store: Store, record: RunRecord) -> None:
    """Writes the record under `RUN_KEY`, stamped with this build's version.

    Called once, after preflight: the record is the thing that does not change.
    """
    store.write_json(RUN_KEY, {_VERSION_FIELD: VERSION, **asdict(record)})


def read_record(store: Store) -> RunRecord:
    """The record a run was started from.

    return: RunRecord - every way of failing is one `RecordError`: there is no run to resume
    """
    payload = _read_object(store, RUN_KEY, _bad_record, "record")
    _require_version(payload, _bad_record, "record")
    return RunRecord(
        workflow=as_text(payload.get("workflow"), "workflow", "record", _bad_record),
        label=as_text(payload.get("label"), "label", "record", _bad_record),
        request=as_text(payload.get("request"), "request", "record", _bad_record),
        base_branch=as_text(payload.get("base_branch"), "base_branch", "record", _bad_record),
        project=as_text(payload.get("project"), "project", "record", _bad_record),
        max_concurrent=as_whole_number(
            payload.get("max_concurrent"), "max_concurrent", "record", _bad_record
        ),
    )


class StateFile:
    """The document that is a workflow's state, and the only way to move it."""

    def __init__(self, store: Store, key: str = STATE_KEY) -> None:
        """Bind to one key in one store. Nothing is read or written until asked."""
        self._store = store
        self._key = key

    def exists(self) -> bool:
        """Whether a state document has been written yet."""
        return self._store.exists(self._key)

    def load(self) -> dict[str, Any] | None:
        """The stored payload, or `None` when the run has never written one.

        `None` is not an error; anything else unreadable raises `StateError`.
        """
        if not self._store.exists(self._key):
            return None
        payload = _read_object(self._store, self._key, _bad_state, "state")
        _require_version(payload, _bad_state, "state")
        return {key: value for key, value in payload.items() if key != _VERSION_FIELD}

    def save(self, payload: Mapping[str, Any]) -> None:
        """Replaces the document with `payload`, whole, stamped with this build's version.

        The store writes atomically, so a reader never meets half a document.
        """
        self._store.write_json(self._key, {_VERSION_FIELD: VERSION, **payload})

    def update(
        self, change: Callable[[dict[str, Any] | None], Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Read, hand the payload to `change`, store what it returns, and return that."""
        # `StateFile.update` must stay synchronous, with no `await` in it or in
        # `change`: an await between the read and the write lets two tasks each
        # save a state built from the same stale read.
        updated = dict(change(self.load()))
        self.save(updated)
        return updated


def _read_object(store: Store, key: str, on_error: OnError, noun: str) -> Mapping[str, Any]:
    """The document at `key` as a JSON object, or `on_error` saying why it is not one.

    `on_error` is called from inside the `except` blocks, so what it raises still
    carries the underlying store error as its context.
    """
    try:
        payload = store.read_json(key)
    except MissingKeyError:
        on_error(f"no {noun} at {key!r}")
    except ValueError:
        on_error(f"{noun} at {key!r} is not JSON")
    if not isinstance(payload, dict):
        on_error(f"{noun} at {key!r} is not a JSON object")
    return payload


def _require_version(payload: Mapping[str, Any], on_error: OnError, noun: str) -> None:
    """Refuse a document this build does not claim to understand."""
    version = payload.get(_VERSION_FIELD)
    if version != VERSION:
        on_error(f"{noun} is version {version!r}, and this build reads version {VERSION}")
