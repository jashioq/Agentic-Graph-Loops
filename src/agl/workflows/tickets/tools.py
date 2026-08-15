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
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from agl.core.agent import NO_PARAMS, Tool
from agl.core.store import MissingKeyError, Store
from agl.runtime.record import STATE_KEY, StateFile
from agl.workflows.tickets.models import (
    TICKETS_SCHEMA,
    InvalidTicketsError,
    Ticket,
    tickets_from_json,
)
from agl.workflows.tickets.reviews import (
    FINDINGS_SCHEMA,
    TRIAGE_SCHEMA,
    CoverageError,
    Finding,
    InvalidFindingsError,
    InvalidGroupsError,
    bug_groups_from_json,
    check_coverage,
    findings_from_json,
    review_key,
)
from agl.workflows.tickets.snapshot import RunFile
from agl.workflows.tickets.state import InvalidStateError

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
    "save_findings",
    "save_spec",
    "save_tickets",
    "save_triage",
    "triage_tools",
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

    Answered out of the run's state rather than the decompose agent's output,
    because the state is where every ticket is. A bug ticket is filed into the
    state and never appears in `tickets.json`, so a tool reading the
    decomposition had nothing to say to the bug-fix agents whose prompt opens
    "Call `get_ticket` first" — it told them their own ticket did not exist.

    Read at call time rather than captured when the tool was built, so a ticket
    the run has since edited — another round of bugs to wait on, say — reads as
    it is now.
    """

    state = RunFile(StateFile(store))

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


def save_findings(store: Store, ticket_id: str, round_: int, source: str) -> Tool:
    """Stores one reviewer's findings, refusing anything that is not a usable set.

    The key is closed over — `review_key(ticket_id, round_, source)` — so
    nothing an agent passes can land its findings anywhere else.
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
    """Stores the triage groups, refusing anything that does not cover every `HIGH`.

    Validated in two stages: `bug_groups_from_json` for shape, then
    `check_coverage` against the `highs` this closure holds — the same check
    the caller runs again after reading back, as a backstop.
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
    """Doing one ticket's work: everything to read, nothing to write.

    What it produces is a commit in its own worktree, not a document.
    """
    return (get_ticket(store, ticket_id), read_spec(store), read_standards(store))


def review_quality_tools(store: Store, ticket_id: str, round_: int) -> tuple[Tool, ...]:
    """Reviewing one ticket against the standards. **No spec access.**

    It judges the code as code. Handed the spec it starts re-arguing design
    decisions that were settled with the user, which is not its job and which
    would file findings nobody can act on.
    """
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


def _entry(ticket: Ticket) -> dict[str, Any]:
    """One ticket as the role holding it reads it: what to build, and what it waits for.

    The run's own bookkeeping — the status it is parked at, which review round
    it is on, the sha its branch stood at — is deliberately not here. None of it
    is anything an agent decides, and a field a role cannot act on is one it can
    only be misled by. `parent` stays, because on a bug ticket it names the work
    being fixed, and it is rendered on every ticket rather than only on bugs so
    that the answer has one shape.
    """
    return {
        "id": ticket.id,
        "title": ticket.title,
        "deliverables": list(ticket.deliverables),
        "blocked_by": list(ticket.blocked_by),
        "parent": ticket.parent,
    }
