"""What each role may reach, built as closures over the store.

Layer: workflows. Composes `agent` and `store`, and imports both from their
package roots.

**Scoping is the closure, not a permission check.** A factory closes over
exactly what a role may touch — one store, one key, one ticket id — and the tool
it hands back has no parameter that could widen it. `get_ticket` for `T-03` has
an empty schema, so a reviewer holding it has no argument it could pass to reach
`T-05`. There is no policy object here, no allow list, and no check inside a
handler that could be bypassed by an argument nobody thought of: what is not in
the closure is not reachable, full stop.

Writes are the same fact in the other direction. A save tool takes the content
and nothing else; the key belongs to the closure, so an agent cannot choose
where its output lands or overwrite another role's document.

Invalid input to `save_tickets` comes back as a string the agent reads, rather
than as an exception, and nothing is written. That in-session correction is the
whole reason these are tools rather than an `output_schema`: a run that produced
an unusable set of tickets can be told what was wrong and try again inside the
same session, instead of failing and being started over.

Every description here is written for the model that will read it — what the
tool answers with and when to reach for it, not what it is called.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from agl.core.agent import NO_PARAMS, Tool
from agl.core.store import MissingKeyError, Store
from agl.workflows.tickets.models import (
    TICKETS_KEY as TICKETS_FIELD,  # the field inside the payload, not a store key
)
from agl.workflows.tickets.models import TICKETS_SCHEMA, InvalidTicketsError, tickets_from_json

__all__ = [
    "SPEC_KEY",
    "STANDARDS_KEY",
    "TICKETS_KEY",
    "decompose_tools",
    "get_ticket",
    "implement_tools",
    "interview_tools",
    "read_spec",
    "read_standards",
    "review_quality_tools",
    "review_spec_tools",
    "save_spec",
    "save_tickets",
]

SPEC_KEY = "spec.md"
STANDARDS_KEY = "standards.md"
TICKETS_KEY = "tickets.json"

_CONTENT = "content"

_CONTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {_CONTENT: {"type": "string"}},
    "required": [_CONTENT],
    "additionalProperties": False,
}
"""The schema for a write: the document, and nothing about where it goes.

Read-only by convention, like `agent.NO_PARAMS` — hand it to a `Tool` rather
than mutating it."""


# -- reads ----------------------------------------------------------------


def read_spec(store: Store) -> Tool:
    """The spec, for a role that is allowed to know what was agreed."""
    return Tool(
        name="read_spec",
        description=(
            "Returns the agreed specification for this run: what the user asked for and "
            "the decisions taken with them while it was written. Call it before you plan "
            "or write anything, and again whenever you are about to guess at intent."
        ),
        schema=NO_PARAMS,
        handler=_reader(store, SPEC_KEY, "the specification for this run"),
    )


def read_standards(store: Store) -> Tool:
    """The project's coding standards."""
    return Tool(
        name="read_standards",
        description=(
            "Returns this project's coding standards: the conventions its code is held "
            "to. Call it before you write code, and before you judge code someone else "
            "wrote — the standards decide, not your own habits."
        ),
        schema=NO_PARAMS,
        handler=_reader(store, STANDARDS_KEY, "the project's coding standards"),
    )


def get_ticket(store: Store, ticket_id: str) -> Tool:
    """One ticket, bound at build time. The model is given no way to say which.

    Looked up at call time rather than captured when the tool was built, so a
    ticket the run has since edited reads as it is now.
    """

    async def handler(arguments: dict[str, Any]) -> str:
        try:
            payload = store.read_json(TICKETS_KEY)
        except MissingKeyError:
            return f"This run holds no {TICKETS_KEY}: no tickets have been stored yet."
        except ValueError:
            return f"The tickets in {TICKETS_KEY} could not be read as JSON."
        entry = _entry(payload, ticket_id)
        if entry is None:
            return (
                f"There is no ticket {ticket_id!r} in {TICKETS_KEY}. "
                "Nothing you can pass to this tool changes which ticket it answers with, "
                "so this is something for the run to fix, not you."
            )
        return json.dumps(entry, indent=2, ensure_ascii=False)

    return Tool(
        name="get_ticket",
        description=(
            "Returns the one ticket you are working on: its title, everything it has to "
            "deliver, and the tickets it waits for. Call it before you start and again "
            "whenever you are unsure of the scope. It always answers with the same "
            "ticket — it takes no arguments, and there is no other ticket to ask for."
        ),
        schema=NO_PARAMS,
        handler=handler,
    )


# -- writes ---------------------------------------------------------------


def save_spec(store: Store) -> Tool:
    """Stores the spec at the key this closure chose."""

    async def handler(arguments: dict[str, Any]) -> str:
        content = arguments.get(_CONTENT)
        if not isinstance(content, str):
            return f"Nothing was saved: pass the document as {_CONTENT!r}, a string."
        store.write(SPEC_KEY, content)
        return "Saved the specification."

    return Tool(
        name="save_spec",
        description=(
            "Stores the specification you have agreed with the user. Call it once the "
            "interview has settled, and again to correct it. Pass the whole document "
            "every time: it replaces what is stored, it does not add to it."
        ),
        schema=_CONTENT_SCHEMA,
        handler=handler,
    )


def save_tickets(store: Store) -> Tool:
    """Stores the decomposition, refusing anything that is not a usable set.

    Validated by `tickets_from_json` before a byte is written, so what reaches
    the store is always something the workflow can read back.
    """

    async def handler(arguments: dict[str, Any]) -> str:
        try:
            tickets = tickets_from_json(arguments)
        except InvalidTicketsError as error:
            return (
                f"Nothing was saved: {error}. "
                "Fix that and call this tool again with the whole set of tickets."
            )
        store.write_json(TICKETS_KEY, arguments)
        return f"Saved {len(tickets)} tickets."

    return Tool(
        name="save_tickets",
        description=(
            "Stores the tickets you have broken the work into. Each one needs an id, a "
            "title, and at least one deliverable, and anything in `blocked_by` has to "
            "name another ticket in the same call. If something is wrong the call comes "
            "back saying what, nothing is stored, and you can call again with it fixed."
        ),
        schema=TICKETS_SCHEMA,
        handler=handler,
    )


# -- what each role is handed ---------------------------------------------


def interview_tools(store: Store) -> tuple[Tool, ...]:
    """Interviewing the user: the spec is the only thing it produces."""
    return (save_spec(store),)


def decompose_tools(store: Store) -> tuple[Tool, ...]:
    """Breaking the spec into tickets: read the one, write the other."""
    return (read_spec(store), save_tickets(store))


def implement_tools(store: Store, ticket_id: str) -> tuple[Tool, ...]:
    """Doing one ticket's work: everything to read, nothing to write.

    What it produces is a commit in its own worktree, not a document.
    """
    return (get_ticket(store, ticket_id), read_spec(store), read_standards(store))


def review_quality_tools(store: Store, ticket_id: str) -> tuple[Tool, ...]:
    """Reviewing one ticket against the standards. **No spec access.**

    It judges the code as code. Handed the spec it starts re-arguing design
    decisions that were settled with the user, which is not its job and which
    would file findings nobody can act on.
    """
    return (get_ticket(store, ticket_id), read_standards(store))


def review_spec_tools(store: Store, ticket_id: str) -> tuple[Tool, ...]:
    """Reviewing one ticket against what was agreed — the reviewer that does hold the spec."""
    return (get_ticket(store, ticket_id), read_spec(store))


# -- internals ------------------------------------------------------------


def _reader(store: Store, key: str, what: str) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """A no-argument handler over one document, missing or not.

    A store with no `standards.md` is a project that has not written any, which
    is a fact the agent should work around rather than a reason to end a run
    that had already started.
    """

    async def handler(arguments: dict[str, Any]) -> str:
        try:
            return store.read(key)
        except MissingKeyError:
            return f"This run holds no {key}: {what} was never written."

    return handler


def _entry(payload: Any, ticket_id: str) -> dict[str, Any] | None:
    """One ticket out of the stored payload, by id, tolerating a payload it cannot parse."""
    if not isinstance(payload, dict):
        return None
    entries = payload.get(TICKETS_FIELD)
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") == ticket_id:
            return entry
    return None
