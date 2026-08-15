"""The state document, as this workflow's `Run`: the codec and the file over it.

Layer: workflows. The one place that knows what a ticket run's `state.json`
looks like. `runtime.record` owns the document — the key, the atomic write, the
version stamp — and deliberately says nothing about what is inside it; this says
what is inside it and nothing about how it gets to disk.

`from_json` re-checks every field and ends with `check`, for the same reason
`tickets_from_json` re-checks a schema it handed to a model: the document is a
*request*, not a guarantee. Here the requester is a person with an editor rather
than an agent, which if anything makes it likelier to be wrong in ways worth a
message they can act on.

`load` and `latest` differ only in what they do about a document that will not
parse, and the difference is which caller is asking. A run about to act on its
state has to hear that it is unreadable. The render loop draws four frames a
second and must never raise over a file somebody is halfway through editing, so
it gets the last `Run` that did parse and keeps drawing.
"""

from collections.abc import Callable, Mapping
from typing import Any

from agl.runtime.record import StateError, StateFile
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.state import Halt, InvalidStateError, Run, check

__all__ = ["RunFile", "from_json", "to_json"]

_TICKETS = "tickets"
_HALT = "halt"

_TICKET_FIELDS = (
    "id",
    "title",
    "status",
    "deliverables",
    "blocked_by",
    "parent",
    "review_round",
    "resume_to",
    "base_sha",
)
_HALT_FIELDS = ("reason", "detail", "resumable")


def to_json(run: Run) -> dict[str, Any]:
    """`run` as the document's payload — every field, every time.

    Nothing is omitted for being empty or defaulted. The document is written
    whole on every change and read by people diffing one run against another,
    and a field that appears only sometimes is a field they have to remember the
    default of.
    """
    return {
        _HALT: None if run.halt is None else _halt_json(run.halt),
        _TICKETS: [_ticket_json(ticket) for ticket in run.tickets],
    }


def from_json(payload: Mapping[str, Any]) -> Run:
    """The `Run` a payload describes, raising `InvalidStateError` if it describes none.

    Every field is checked here rather than trusted, and `check` has the last
    word: a document that parses into tickets which cannot all be true at once
    is refused whole, before a run acts on half of it.
    """
    _known_fields(payload, _TICKETS, _HALT, where="state")
    raw = payload.get(_TICKETS, [])
    if not isinstance(raw, list):
        raise InvalidStateError(f"{_TICKETS!r} must be an array, got {_kind(raw)}")
    run = Run(
        tickets=tuple(_ticket(item, index) for index, item in enumerate(raw)),
        halt=_halt(payload.get(_HALT)),
    )
    check(run)
    return run


class RunFile:
    """The state document, as this workflow's `Run`."""

    def __init__(self, file: StateFile) -> None:
        """Bind to one state document. Nothing is read until asked."""
        self._file = file
        self._latest = Run()

    def load(self) -> Run:
        """The stored state, strictly. An absent document is an empty `Run`.

        Nothing written yet is the ordinary first moment of a run, not a
        failure — the stage a run is at is read off what it has produced, and a
        run that has produced nothing is one about to interview. Anything else
        that stops the document being understood raises `InvalidStateError`.
        """
        try:
            payload = self._file.load()
        except StateError as unreadable:
            raise InvalidStateError(str(unreadable)) from unreadable
        run = Run() if payload is None else from_json(payload)
        self._latest = run
        return run

    def latest(self) -> Run:
        """The stored state, or the last one that parsed if this one will not.

        For the render loop alone. A frame is a picture of the run, and a
        picture that is a second out of date is better than a traceback where
        the dashboard was.
        """
        try:
            return self.load()
        except InvalidStateError:
            return self._latest

    def write(self, run: Run) -> None:
        """Replace the document with `run`, whole."""
        self._file.save(to_json(run))
        self._latest = run

    def update(self, change: Callable[[Run], Run]) -> Run:
        """Load, hand the `Run` to `change`, write what comes back, and return it.

        Synchronous on purpose, and `change` must be too. Load, decide and write
        contain no `await`, so on a single-threaded event loop no other task can
        run between the read and the write and no mutation can interleave with
        another. An `await` anywhere in here — or in `change` — gives that up,
        and two tasks each write a state built from the same stale read.
        """
        updated = change(self.load())
        self.write(updated)
        return updated


# -- one ticket -----------------------------------------------------------


def _ticket_json(ticket: Ticket) -> dict[str, Any]:
    return {
        "id": ticket.id,
        "title": ticket.title,
        "status": ticket.status.value,
        "deliverables": list(ticket.deliverables),
        "blocked_by": list(ticket.blocked_by),
        "parent": ticket.parent,
        "review_round": ticket.review_round,
        "resume_to": None if ticket.resume_to is None else ticket.resume_to.value,
        "base_sha": ticket.base_sha,
    }


def _ticket(item: Any, index: int) -> Ticket:
    """One ticket out of one array entry, checking every field it names."""
    where = f"ticket {index}"
    fields = _object(item, where)
    _known_fields(fields, *_TICKET_FIELDS, where=where)
    ticket_id = _text(fields.get("id"), "id", where)
    where = f"ticket {ticket_id!r}"
    return Ticket(
        id=ticket_id,
        title=_text(fields.get("title"), "title", where),
        status=_status(fields.get("status"), "status", where),
        deliverables=_texts(fields.get("deliverables", []), "deliverables", where),
        blocked_by=_texts(fields.get("blocked_by", []), "blocked_by", where),
        parent=_optional_text(fields.get("parent"), "parent", where),
        review_round=_count(fields.get("review_round", 0), "review_round", where),
        resume_to=_optional_status(fields.get("resume_to"), "resume_to", where),
        base_sha=_optional_text(fields.get("base_sha"), "base_sha", where),
    )


def _halt_json(halt: Halt) -> dict[str, Any]:
    return {"reason": halt.reason, "detail": halt.detail, "resumable": halt.resumable}


def _halt(value: Any) -> Halt | None:
    """The halt a payload describes, or `None` for a run that is not stopped."""
    if value is None:
        return None
    fields = _object(value, "halt")
    _known_fields(fields, *_HALT_FIELDS, where="halt")
    resumable = fields.get("resumable", True)
    if not isinstance(resumable, bool):
        raise InvalidStateError(f"halt: resumable must be true or false, got {resumable!r}")
    return Halt(
        reason=_text(fields.get("reason"), "reason", "halt"),
        detail=_optional_text(fields.get("detail"), "detail", "halt") or "",
        resumable=resumable,
    )


# -- narrowing ------------------------------------------------------------


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidStateError(f"{where} must be an object, got {_kind(value)}")
    return value


def _known_fields(fields: Mapping[str, Any], *allowed: str, where: str) -> None:
    """Refuse a field this build does not know, rather than ignoring it.

    A misspelled key that is quietly dropped is a person's edit that appeared to
    take and did not, which is worse than being told.
    """
    for name in fields:
        if name not in allowed:
            raise InvalidStateError(f"{where}: unknown field {name!r}")


def _text(value: Any, name: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidStateError(f"{where}: {name} must be non-empty text, got {value!r}")
    return value


def _optional_text(value: Any, name: str, where: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, where)


def _texts(value: Any, name: str, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise InvalidStateError(f"{where}: {name} must be an array, got {_kind(value)}")
    return tuple(_text(entry, name, where) for entry in value)


def _count(value: Any, name: str, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InvalidStateError(f"{where}: {name} must be a whole number, got {value!r}")
    return value


def _status(value: Any, name: str, where: str) -> Status:
    try:
        return Status(value)
    except ValueError as unknown:
        known = ", ".join(status.value for status in Status)
        raise InvalidStateError(
            f"{where}: unknown {name} {value!r}, expected one of {known}"
        ) from unknown


def _optional_status(value: Any, name: str, where: str) -> Status | None:
    if value is None:
        return None
    return _status(value, name, where)


def _kind(value: Any) -> str:
    """What something is, for an error message a person has to act on."""
    return type(value).__name__
