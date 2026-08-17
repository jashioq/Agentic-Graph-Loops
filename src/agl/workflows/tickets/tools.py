"""What each role may reach, built as closures over the store.

Layer: workflows. Scoping is the closure: a factory closes over what a role may
touch, and the tool it returns has no parameter that could widen it. Invalid
input comes back as a string and nothing is written. Every `description` is a
prompt written for the model.
"""

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from agl.core.agent import NO_PARAMS, Tool
from agl.core.store import MissingKeyError, Store
from agl.runtime.record import STATE_KEY, StateFile
from agl.workflows.tickets.documents.review_documents import (
    FINDINGS_SCHEMA,
    TRIAGE_SCHEMA,
    bug_groups_from_json,
    findings_from_json,
)
from agl.workflows.tickets.documents.state_document import StateDocument
from agl.workflows.tickets.documents.store_keys import (
    SPEC_KEY,
    STANDARDS_KEY,
    TICKETS_KEY,
    review_key,
)
from agl.workflows.tickets.documents.tickets_document import TICKETS_SCHEMA, tickets_from_json
from agl.workflows.tickets.errors import (
    CoverageError,
    InvalidFindingsError,
    InvalidGroupsError,
    InvalidStateError,
    InvalidTicketsError,
)
from agl.workflows.tickets.findings import Finding, check_coverage
from agl.workflows.tickets.models import Ticket

__all__ = [
    "decompose_tools",
    "get_ticket",
    "implement_tools",
    "interview_tools",
    "read_spec",
    "read_standards",
    "review_quality_tools",
    "review_spec_tools",
    "save_findings",
    "save_spec",
    "save_tickets",
    "save_triage",
    "triage_tools",
]

_CONTENT = "content"

_CONTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {_CONTENT: {"type": "string"}},
    "required": [_CONTENT],
    "additionalProperties": False,
}
"""The schema for a write: the document, and nothing about where it goes. Read-only."""


# -- reads ----------------------------------------------------------------


def read_spec(store: Store) -> Tool:
    """Builds a no-argument tool returning the run's spec, or a note if none was written."""
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
    """Builds a no-argument tool returning the coding standards, or a note if unwritten."""
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
    """Builds a tool answering with one ticket, read from the state at call time.

    param: ticket_id - the only ticket it will ever answer with; the model cannot widen it
    return: Tool - the ticket as JSON, or why it could not be read
    """

    state = StateDocument(StateFile(store))

    async def handler(arguments: dict[str, Any]) -> str:
        try:
            run = state.load()
        except InvalidStateError as unreadable:
            return f"The run's state in {STATE_KEY} could not be read: {unreadable}."
        if not run.tickets:
            return f"This run holds no {STATE_KEY}: no tickets have been stored yet."
        ticket = run.get(ticket_id)
        if ticket is None:
            return (
                f"There is no ticket {ticket_id!r} in this run. "
                "Nothing you can pass to this tool changes which ticket it answers with, "
                "so this is something for the run to fix, not you."
            )
        return json.dumps(_entry(ticket), indent=2, ensure_ascii=False)

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
    """Builds a tool storing the spec at the key this closure chose."""

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
    """Builds a tool storing the decomposition, validated by `tickets_from_json` first."""

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


def save_findings(store: Store, ticket_id: str, round_: int, source: str) -> Tool:
    """Builds a tool storing one reviewer's findings under a closed-over key.

    param: source - which reviewer, one of `REVIEWERS`
    return: Tool - validates, then writes; nothing an agent passes can move the key
    """

    key = review_key(ticket_id, round_, source)

    async def handler(arguments: dict[str, Any]) -> str:
        try:
            findings = findings_from_json(arguments)
        except InvalidFindingsError as error:
            return (
                f"Nothing was saved: {error}. "
                "Fix that and call this tool again with the whole set of findings."
            )
        store.write_json(key, arguments)
        if not findings:
            return "Saved: no findings."
        return f"Saved {len(findings)} finding(s)."

    return Tool(
        name="save_findings",
        description=(
            "Stores the findings from this review. This is the only way findings leave "
            "this session — the review is not finished until this has been called, even "
            "when it finds nothing. An empty list is a valid and expected result on a "
            "clean review: call this tool with an empty `findings` array rather than "
            "writing a summary. Each finding needs an id, a severity, a title, and a "
            "detail that says both what is wrong and what would fix it, plus at least one "
            "file. If something is wrong the call comes back saying what, nothing is "
            "stored, and you can call again with it fixed."
        ),
        schema=FINDINGS_SCHEMA,
        handler=handler,
    )


def save_triage(store: Store, ticket_id: str, round_: int, highs: Sequence[Finding]) -> Tool:
    """Builds a tool storing triage groups, checked for shape then for coverage.

    param: highs - the `HIGH` findings the groups must cover exactly
    return: Tool - refuses and writes nothing unless coverage holds
    """

    key = review_key(ticket_id, round_, "triage")

    async def handler(arguments: dict[str, Any]) -> str:
        try:
            groups = bug_groups_from_json(arguments)
        except InvalidGroupsError as error:
            return (
                f"Nothing was saved: {error}. "
                "Fix that and call this tool again with the whole set of groups."
            )
        try:
            check_coverage(groups, highs)
        except CoverageError as error:
            return (
                f"Nothing was saved: {error}. "
                "Fix that and call this tool again with the whole set of groups."
            )
        store.write_json(key, arguments)
        return f"Saved {len(groups)} group(s)."

    return Tool(
        name="save_triage",
        description=(
            "Stores the groups you have triaged the HIGH findings into. Every HIGH "
            "finding listed above must appear in exactly one group's `findings` list. If "
            "one is missing, named in two groups, or a group names a finding that is not "
            "HIGH, the call comes back saying which id and nothing is stored — fix that "
            "and call this tool again with the whole set of groups."
        ),
        schema=TRIAGE_SCHEMA,
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
    """Doing one ticket's work: everything to read, nothing to write."""
    return (get_ticket(store, ticket_id), read_spec(store), read_standards(store))


def review_quality_tools(store: Store, ticket_id: str, round_: int) -> tuple[Tool, ...]:
    """Reviewing one ticket against the standards. No spec access: it judges code as code."""
    return (
        get_ticket(store, ticket_id),
        read_standards(store),
        save_findings(store, ticket_id, round_, "quality"),
    )


def review_spec_tools(store: Store, ticket_id: str, round_: int) -> tuple[Tool, ...]:
    """Reviewing one ticket against what was agreed — the reviewer that does hold the spec."""
    return (
        get_ticket(store, ticket_id),
        read_spec(store),
        save_findings(store, ticket_id, round_, "spec"),
    )


def triage_tools(
    store: Store, ticket_id: str, round_: int, highs: Sequence[Finding]
) -> tuple[Tool, ...]:
    """Grouping the `HIGH` findings into bug tickets: nothing to read, one tool to write."""
    return (save_triage(store, ticket_id, round_, highs),)


# -- internals ------------------------------------------------------------


def _reader(store: Store, key: str, what: str) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """A no-argument handler reading one key, answering with `what` when it is missing."""

    async def handler(arguments: dict[str, Any]) -> str:
        try:
            return store.read(key)
        except MissingKeyError:
            return f"This run holds no {key}: {what} was never written."

    return handler


def _entry(ticket: Ticket) -> dict[str, Any]:
    """One ticket as an agent reads it: what to build and what it waits for.

    Omits the run's own bookkeeping — status, round, base sha — which no agent decides.
    """
    return {
        "id": ticket.id,
        "title": ticket.title,
        "deliverables": list(ticket.deliverables),
        "blocked_by": list(ticket.blocked_by),
        "parent": ticket.parent,
    }
