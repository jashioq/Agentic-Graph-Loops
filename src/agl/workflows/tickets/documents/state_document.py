"""The `state.json` codec, and the document over it, as this workflow's `Run`.

Layer: workflows. The one place that knows what a ticket run's state document
looks like; `runtime.record` owns how it reaches disk and says nothing about
what is inside it. Every field is re-checked on the way in, because the writer
may be a person with an editor.
"""

from collections.abc import Callable, Mapping
from typing import Any

from agl.runtime.json_fields import (
    as_object,
    as_optional_text,
    as_text,
    as_text_list,
    as_whole_number,
    reject_unknown_fields,
    type_name,
)
from agl.runtime.record import StateError, StateFile
from agl.workflows.tickets.errors import Halt, InvalidStateError
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import Run, check

__all__ = ["StateDocument", "from_json", "to_json"]

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

    Nothing is omitted for being empty or defaulted: people diff one run against
    another, and a field that appears only sometimes is a default to remember.
    """
    return {
        _HALT: None if run.halt is None else _halt_json(run.halt),
        _TICKETS: [_ticket_json(ticket) for ticket in run.tickets],
    }


def from_json(payload: Mapping[str, Any]) -> Run:
    """The `Run` a payload describes, raising `InvalidStateError` if it describes none.

    `check` has the last word: a document that parses into tickets which cannot
    all be true at once is refused whole, before a run acts on half of it.
    """
    reject_unknown_fields(payload, [_TICKETS, _HALT], "state", InvalidStateError)
    raw = payload.get(_TICKETS, [])
    if not isinstance(raw, list):
        raise InvalidStateError(f"{_TICKETS!r} must be an array, got {type_name(raw)}")
    run = Run(
        tickets=tuple(_ticket(item, index) for index, item in enumerate(raw)),
        halt=_halt(payload.get(_HALT)),
    )
    check(run)
    return run


class StateDocument:
    """The state document, as this workflow's `Run`."""

    def __init__(self, file: StateFile) -> None:
        """Bind to one state document. Nothing is read until asked."""
        self._file = file
        self._latest = Run()

    def load(self) -> Run:
        """The stored state, strictly. An absent document is an empty `Run`.

        Nothing written yet is the ordinary first moment of a run; anything else
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

        For the render loop alone: a picture that is a second out of date is
        better than a traceback where the dashboard was.
        """
        try:
            return self.load()
        except InvalidStateError:
            return self._latest

    def write(self, run: Run) -> None:
        """Replace the document with `run`, whole."""
        self._file.save(to_json(run))
        self._latest = run

    # `StateDocument.update` must stay synchronous, with no `await` in it or in
    # `change`: an await between the read and the write lets two tasks each write
    # a state built from the same stale read.
    def update(self, change: Callable[[Run], Run]) -> Run:
        """Load, hand the `Run` to `change`, write what comes back, and return it."""
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
    fields = as_object(item, where, InvalidStateError)
    reject_unknown_fields(fields, _TICKET_FIELDS, where, InvalidStateError)
    ticket_id = as_text(fields.get("id"), "id", where, InvalidStateError)
    where = f"ticket {ticket_id!r}"
    return Ticket(
        id=ticket_id,
        title=as_text(fields.get("title"), "title", where, InvalidStateError),
        status=_status(fields.get("status"), "status", where),
        deliverables=as_text_list(
            fields.get("deliverables", []), "deliverables", where, InvalidStateError
        ),
        blocked_by=as_text_list(
            fields.get("blocked_by", []), "blocked_by", where, InvalidStateError
        ),
        parent=as_optional_text(fields.get("parent"), "parent", where, InvalidStateError),
        review_round=as_whole_number(
            fields.get("review_round", 0), "review_round", where, InvalidStateError
        ),
        resume_to=_optional_status(fields.get("resume_to"), "resume_to", where),
        base_sha=as_optional_text(fields.get("base_sha"), "base_sha", where, InvalidStateError),
    )


def _halt_json(halt: Halt) -> dict[str, Any]:
    return {"reason": halt.reason, "detail": halt.detail, "resumable": halt.resumable}


def _halt(value: Any) -> Halt | None:
    """The halt a payload describes, or `None` for a run that is not stopped."""
    if value is None:
        return None
    fields = as_object(value, "halt", InvalidStateError)
    reject_unknown_fields(fields, _HALT_FIELDS, "halt", InvalidStateError)
    resumable = fields.get("resumable", True)
    if not isinstance(resumable, bool):
        raise InvalidStateError(f"halt: resumable must be true or false, got {resumable!r}")
    # `detail` is the one text field that may be empty: a halt whose reason says
    # everything carries no detail.
    detail = fields.get("detail") or ""
    if not isinstance(detail, str):
        raise InvalidStateError(f"halt: detail must be text, got {type_name(detail)}")
    return Halt(
        reason=as_text(fields.get("reason"), "reason", "halt", InvalidStateError),
        detail=detail,
        resumable=resumable,
    )


# -- the one narrowing that is this document's own --------------------------


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
