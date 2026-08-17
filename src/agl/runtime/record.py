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
from typing import Any

from agl.core.store import MissingKeyError, Store
from agl.runtime.json_fields import InvalidFieldError, as_text, as_whole_number

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
    try:
        return _read_record(store)
    except InvalidFieldError as invalid:
        raise RecordError(str(invalid)) from invalid


def _read_record(store: Store) -> RunRecord:
    """The read itself, whose `InvalidFieldError`s `read_record` renames."""
    payload = _read_object(store, RUN_KEY, "record")
    _require_version(payload, "record")
    return RunRecord(
        workflow=as_text(payload.get("workflow"), "workflow", "record"),
        label=as_text(payload.get("label"), "label", "record"),
        request=as_text(payload.get("request"), "request", "record"),
        base_branch=as_text(payload.get("base_branch"), "base_branch", "record"),
        project=as_text(payload.get("project"), "project", "record"),
        max_concurrent=as_whole_number(payload.get("max_concurrent"), "max_concurrent", "record"),
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
        try:
            payload = _read_object(self._store, self._key, "state")
            _require_version(payload, "state")
        except InvalidFieldError as invalid:
            raise StateError(str(invalid)) from invalid
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


def _read_object(store: Store, key: str, noun: str) -> Mapping[str, Any]:
    """The document at `key` as a JSON object, or `InvalidFieldError` saying why not."""
    try:
        payload = store.read_json(key)
    except MissingKeyError as missing:
        raise InvalidFieldError(f"no {noun} at {key!r}") from missing
    except ValueError as invalid:
        raise InvalidFieldError(f"{noun} at {key!r} is not JSON") from invalid
    if not isinstance(payload, dict):
        raise InvalidFieldError(f"{noun} at {key!r} is not a JSON object")
    return payload


def _require_version(payload: Mapping[str, Any], noun: str) -> None:
    """Refuse a document this build does not claim to understand."""
    version = payload.get(_VERSION_FIELD)
    if version != VERSION:
        raise InvalidFieldError(
            f"{noun} is version {version!r}, and this build reads version {VERSION}"
        )
