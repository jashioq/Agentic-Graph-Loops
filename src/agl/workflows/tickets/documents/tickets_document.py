"""What the decompose agent produces: the schema it is handed, and the parser.

Layer: workflows. Narrows untrusted JSON with `runtime.json_fields` and raises
`InvalidTicketsError`. `TICKETS_KEY` here is the payload key inside the
document, not the store key of the same name in `store_keys`.
"""

import re
from typing import Any

from agl.runtime.json_fields import (
    InvalidFieldError,
    as_object,
    as_text,
    as_text_list,
    reject_unknown_fields,
    require_fields,
    type_name,
)
from agl.workflows.tickets.errors import InvalidTicketsError
from agl.workflows.tickets.models import Status, Ticket

__all__ = ["TICKETS_KEY", "TICKETS_SCHEMA", "tickets_from_json"]

# The same shape `paths.validate_node_id` enforces, spelled out rather than
# imported so this file stays free of `agl.core`.
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9-]*$"
_ID_RE = re.compile(_ID_PATTERN)

TICKETS_KEY = "tickets"
_REQUIRED = ["id", "title", "deliverables"]
_OPTIONAL = ["blocked_by"]
_ALLOWED = [*_REQUIRED, *_OPTIONAL]

TICKETS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        TICKETS_KEY: {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": _ID_PATTERN},
                    "title": {"type": "string", "minLength": 1},
                    "deliverables": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "blocked_by": {
                        "type": "array",
                        "items": {"type": "string", "pattern": _ID_PATTERN},
                    },
                },
                "required": _REQUIRED,
                "additionalProperties": False,
            },
        }
    },
    "required": [TICKETS_KEY],
    "additionalProperties": False,
}
"""The schema `tools.save_tickets` hands the model. Read-only.

Describes what an agent supplies, not what a `Ticket` holds: `status`,
`review_round` and `parent` are the orchestrator's."""


def tickets_from_json(data: Any) -> tuple[Ticket, ...]:
    """Parses decompose-agent output into tickets, raising `InvalidTicketsError`.

    Re-checks the whole schema — a schema handed to a model is a request, not a
    guarantee — plus unique ids and resolvable blockers, which it cannot state.
    """
    try:
        return _read(data)
    except InvalidFieldError as invalid:
        raise InvalidTicketsError(str(invalid)) from invalid


def _read(data: Any) -> tuple[Ticket, ...]:
    """The read itself, whose `InvalidFieldError`s the caller above renames."""
    payload = as_object(data, "output")
    require_fields(payload, [TICKETS_KEY], "output")
    reject_unknown_fields(payload, [TICKETS_KEY], "output")
    raw = payload[TICKETS_KEY]
    if not isinstance(raw, list):
        raise InvalidTicketsError(f"{TICKETS_KEY!r} must be an array, got {type_name(raw)}")
    if not raw:
        raise InvalidTicketsError(f"{TICKETS_KEY!r} is empty: a run needs at least one ticket")

    tickets = tuple(_one_ticket(item, index) for index, item in enumerate(raw))
    _check_ids(tickets)
    return tickets


def _one_ticket(item: Any, index: int) -> Ticket:
    """Builds one ticket from one array entry, checking every field it names."""
    where = f"ticket {index}"
    fields = as_object(item, where)
    require_fields(fields, _REQUIRED, where)
    reject_unknown_fields(fields, _ALLOWED, where)

    ticket_id = fields["id"]
    if not isinstance(ticket_id, str) or not _ID_RE.fullmatch(ticket_id):
        raise InvalidTicketsError(
            f"{where}: id {ticket_id!r} must be letters, digits and hyphens, "
            "starting with a letter or digit"
        )
    where = f"ticket {ticket_id!r}"

    return Ticket(
        id=ticket_id,
        title=as_text(fields["title"], "title", where),
        status=Status.PENDING,
        deliverables=as_text_list(fields["deliverables"], "deliverables", where, allow_empty=False),
        blocked_by=as_text_list(fields.get("blocked_by", []), "blocked_by", where),
    )


def _check_ids(tickets: tuple[Ticket, ...]) -> None:
    """The two rules a JSON schema cannot state: unique ids, resolvable blockers."""
    seen: set[str] = set()
    for ticket in tickets:
        if ticket.id in seen:
            raise InvalidTicketsError(f"duplicate ticket id {ticket.id!r}")
        seen.add(ticket.id)
    for ticket in tickets:
        for blocker in ticket.blocked_by:
            if blocker == ticket.id:
                raise InvalidTicketsError(f"ticket {ticket.id!r} is blocked by itself")
            if blocker not in seen:
                raise InvalidTicketsError(
                    f"ticket {ticket.id!r} is blocked by unknown ticket {blocker!r}"
                )
