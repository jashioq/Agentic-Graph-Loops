"""A run's two documents: the record it was started from, and its state.

Layer: runtime. Imports `agl.core.store` and nothing else — a run's documents
are keys in a store, and where that store puts them is the store's business.

The two differ in when they are written, and that is the whole design. The
**record** is written once, after preflight, and never again: it is what a run
was asked for — workflow, label, request, base branch, project, concurrency —
and none of that can change without the run being a different run. `agl resume`
reads it before it can build a `RunContext` at all, which is why it is separate
from the state rather than a corner of it.

The **state** is the run's only mutable truth, rewritten whole every time it
moves. This module says nothing about what is in it; it owns the reading, the
writing and the version stamp, and leaves the shape to the workflow whose state
it is. Everything a run could derive instead — the stage it has reached, the
dependency graph, the merge queue's contents — is derived, because a stored copy
is a second truth waiting to disagree with the first.

Both documents carry `version`. A document from a build that is not this one is
refused rather than guessed at: half-understanding a state document is how a run
resumes into a shape nobody wrote. The stamp is this module's concern alone, so
a caller neither writes it nor sees it — `save` puts it on and `load` takes it
off, and what a caller gets back is exactly the payload it gave.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from agl.core.store import MissingKeyError, Store
from agl.runtime.json_fields import as_text, as_whole_number

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

    Every field is something `agl resume` needs before it has a context to ask
    anything else with: which workflow to hand the run back to, which label
    names it, and the request, branch, project and concurrency the original
    invocation settled on.
    """

    workflow: str
    label: str
    request: str
    base_branch: str
    project: str
    max_concurrent: int


def write_record(store: Store, record: RunRecord) -> None:
    """Write the record under `RUN_KEY`, stamped with this build's version.

    Called once, after preflight. Writing it twice is not an error the store can
    see, but it is one the run has: the record is the thing that does not change.
    """
    store.write_json(RUN_KEY, {_VERSION_FIELD: VERSION, **asdict(record)})


def read_record(store: Store) -> RunRecord:
    """The record a run was started from. Raises `RecordError`.

    Every way of failing is one `RecordError`, because the caller does the same
    thing about all of them — there is no run here it can resume. That covers a
    missing document, one that is not JSON, one from a version this build does
    not know, and one whose fields are absent or of the wrong type.
    """
    payload = _read_object(store, RUN_KEY, RecordError, "record")
    _require_version(payload, RecordError, "record")
    return RunRecord(
        workflow=as_text(payload.get("workflow"), "workflow", "record", RecordError),
        label=as_text(payload.get("label"), "label", "record", RecordError),
        request=as_text(payload.get("request"), "request", "record", RecordError),
        base_branch=as_text(payload.get("base_branch"), "base_branch", "record", RecordError),
        project=as_text(payload.get("project"), "project", "record", RecordError),
        max_concurrent=as_whole_number(
            payload.get("max_concurrent"), "max_concurrent", "record", RecordError
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

        `None` is not an error: a run that has not saved its state yet is the
        ordinary first moment of a run, and it is the caller that knows what an
        absent state means. Anything else that stops the document from being
        understood — not JSON, not an object, a version this build does not know
        — raises `StateError`.
        """
        if not self._store.exists(self._key):
            return None
        payload = _read_object(self._store, self._key, StateError, "state")
        _require_version(payload, StateError, "state")
        return {key: value for key, value in payload.items() if key != _VERSION_FIELD}

    def save(self, payload: Mapping[str, Any]) -> None:
        """Replace the document with `payload`, stamped with this build's version.

        The whole state, every time. The store writes through a temp file and
        `os.replace`, so a reader meets the old document or the new one and
        never half of either.
        """
        self._store.write_json(self._key, {_VERSION_FIELD: VERSION, **payload})

    def update(
        self, change: Callable[[dict[str, Any] | None], Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Read, hand the payload to `change`, store what it returns, and return that.

        Synchronous on purpose, and `change` must be too. Load, decide and save
        contain no `await`, so on a single-threaded event loop no other task can
        run between the read and the write and no mutation can be interleaved
        with another. An `await` anywhere in here — or in `change` — gives that
        up, and two tasks each save a state built from the same stale read.
        """
        updated = dict(change(self.load()))
        self.save(updated)
        return updated


def _read_object(
    store: Store, key: str, error: type[Exception], noun: str
) -> Mapping[str, Any]:
    """The document at `key` as a JSON object, or `error` saying why it is not one."""
    try:
        payload = store.read_json(key)
    except MissingKeyError as missing:
        raise error(f"no {noun} at {key!r}") from missing
    except ValueError as invalid:
        raise error(f"{noun} at {key!r} is not JSON") from invalid
    if not isinstance(payload, dict):
        raise error(f"{noun} at {key!r} is not a JSON object")
    return payload


def _require_version(payload: Mapping[str, Any], error: type[Exception], noun: str) -> None:
    """Refuse a document this build does not claim to understand."""
    version = payload.get(_VERSION_FIELD)
    if version != VERSION:
        raise error(f"{noun} is version {version!r}, and this build reads version {VERSION}")
