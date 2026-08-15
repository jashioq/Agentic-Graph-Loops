"""The shapes review findings take, and the transformation into bug tickets.

Layer: workflows. Pure — no agents, no I/O. Imports this workflow's `models`
and nothing from `agl.core`.

`Finding.files` is required: a finding that cannot name a file is too vague to
act on, and requiring it is what makes grouping checkable.

Triage returns `BugGroup`s, not tickets — ids and parentage belong to Python,
the same as commits and branches do. A group's `findings` field is what lets
`check_coverage` assert every `HIGH` finding landed in exactly one group and
nothing else did, so a triage agent that quietly drops a finding fails loudly
instead of shipping a bug; it is also the traceability from a bug ticket back
to the findings that caused it.

`MEDIUM` and `LOW` findings are stored but never acted on here. That pile is
the interesting one to read later — twenty `MEDIUM`s about the same pattern
belong in `standards.md`, not in a bug ticket.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agl.workflows.tickets.models import Status, Ticket

__all__ = [
    "FINDINGS_SCHEMA",
    "GROUPS_KEY",
    "REVIEWERS",
    "TRIAGE_SCHEMA",
    "BugGroup",
    "CoverageError",
    "Finding",
    "InvalidFindingsError",
    "InvalidGroupsError",
    "Severity",
    "bug_groups_from_json",
    "check_coverage",
    "findings_from_json",
    "high",
    "next_bug_start",
    "review_key",
    "to_bug_tickets",
]


REVIEWERS: tuple[str, ...] = ("quality", "spec")
"""The two reviewers, named by the findings document each one writes.

Every other name a reviewer has is this one with something around it — its role
is `review-<source>`, its prompt is `review_<source>`, its activity is prefixed
with it — so the document a role is checked for and the role that would write it
cannot drift apart. It lives here, beside `review_key`, because two callers ask
about those documents: `agents` runs the roles that write them, and `steps`
reads their presence to decide whether a ticket's review is finished.
"""


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Finding:
    """One thing a reviewer found, and where it lives."""

    id: str  # "Q-1" from quality, "S-1" from spec
    severity: Severity
    title: str
    detail: str
    files: tuple[str, ...]  # at least one


@dataclass(frozen=True)
class BugGroup:
    """A set of `HIGH` findings one agent can fix in a single pass."""

    title: str
    deliverables: tuple[str, ...]
    findings: tuple[str, ...]  # the finding ids this group covers


class InvalidFindingsError(Exception):
    """Raised when reviewer output does not describe a usable set of findings."""


class InvalidGroupsError(Exception):
    """Raised when triage-agent output does not describe usable bug groups."""


class CoverageError(Exception):
    """Raised when a set of groups does not account for every `HIGH` finding."""


# -- what a reviewer produces -----------------------------------------------

FINDINGS_KEY = "findings"
_FINDING_REQUIRED = ["id", "severity", "title", "detail", "files"]

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        FINDINGS_KEY: {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "severity": {"type": "string", "enum": [s.value for s in Severity]},
                    "title": {"type": "string", "minLength": 1},
                    "detail": {"type": "string", "minLength": 1},
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": _FINDING_REQUIRED,
                "additionalProperties": False,
            },
        }
    },
    "required": [FINDINGS_KEY],
    "additionalProperties": False,
}
"""The schema `tools.save_findings` hands the model for its `findings` argument.

Read-only by convention, like `agent.NO_PARAMS` — hand it to a `Tool` rather
than mutating it."""


def findings_from_json(data: Any) -> tuple[Finding, ...]:
    """Parse reviewer output into findings, raising `InvalidFindingsError`.

    Re-checks everything `FINDINGS_SCHEMA` states, because a schema handed to a
    model is a request rather than a guarantee, plus the one rule JSON schema
    cannot express: ids are unique. A review with no findings at all is valid —
    an empty array is not `minItems`-constrained, unlike each finding's `files`.
    """
    payload = _object(data, "output", InvalidFindingsError)
    _known_fields(payload, [FINDINGS_KEY], (), "output", InvalidFindingsError)
    raw = payload[FINDINGS_KEY]
    if not isinstance(raw, list):
        raise InvalidFindingsError(f"{FINDINGS_KEY!r} must be an array, got {_kind(raw)}")

    findings = tuple(_one_finding(item, index) for index, item in enumerate(raw))
    _check_finding_ids(findings)
    return findings


def _one_finding(item: Any, index: int) -> Finding:
    where = f"finding {index}"
    fields = _object(item, where, InvalidFindingsError)
    _known_fields(fields, _FINDING_REQUIRED, (), where, InvalidFindingsError)

    finding_id = fields["id"]
    if not isinstance(finding_id, str) or not finding_id.strip():
        raise InvalidFindingsError(f"{where}: id must be non-empty text, got {finding_id!r}")
    where = f"finding {finding_id!r}"

    severity_raw = fields["severity"]
    try:
        severity = Severity(severity_raw)
    except ValueError as error:
        known = ", ".join(s.value for s in Severity)
        raise InvalidFindingsError(
            f"{where}: unknown severity {severity_raw!r}, expected one of {known}"
        ) from error

    title = _text(fields["title"], "title", where, InvalidFindingsError)
    detail = _text(fields["detail"], "detail", where, InvalidFindingsError)
    files = _text_list(
        fields["files"], "files", where, allow_empty=False, error=InvalidFindingsError
    )

    return Finding(id=finding_id, severity=severity, title=title, detail=detail, files=files)


def _check_finding_ids(findings: tuple[Finding, ...]) -> None:
    seen: set[str] = set()
    for finding in findings:
        if finding.id in seen:
            raise InvalidFindingsError(f"duplicate finding id {finding.id!r}")
        seen.add(finding.id)


# -- triage -------------------------------------------------------------


def high(findings: Sequence[Finding]) -> tuple[Finding, ...]:
    """Every `HIGH` finding, in the order it was given."""
    return tuple(finding for finding in findings if finding.severity is Severity.HIGH)


def check_coverage(groups: Sequence[BugGroup], highs: Sequence[Finding]) -> None:
    """Raise `CoverageError` unless every finding in `highs` is in exactly one group.

    A group naming an id outside `highs` — because it does not exist, or
    because it names a `MEDIUM` or `LOW` finding — fails the same check: `highs`
    is the only set a group is allowed to draw from.
    """
    high_ids = {finding.id for finding in highs}
    covered: set[str] = set()
    for group in groups:
        for finding_id in group.findings:
            if finding_id not in high_ids:
                raise CoverageError(
                    f"group {group.title!r} names finding {finding_id!r}, "
                    "which is not a HIGH finding in this review"
                )
            if finding_id in covered:
                raise CoverageError(
                    f"finding {finding_id!r} is covered by more than one group"
                )
            covered.add(finding_id)

    missing = high_ids - covered
    if missing:
        raise CoverageError(
            "the following HIGH findings are not covered by any group: "
            + ", ".join(sorted(missing))
        )


def to_bug_tickets(parent: Ticket, groups: Sequence[BugGroup], start: int) -> tuple[Ticket, ...]:
    """One bug ticket per group, ids assembled from `parent.id` starting at `start`.

    `start` exists because a second review round must not reuse ids from the
    first — the caller passes one past the highest bug id already in use.
    """
    return tuple(
        Ticket(
            id=f"{parent.id}-bug-{n}",
            title=group.title,
            status=Status.PENDING,
            deliverables=group.deliverables,
            parent=parent.id,
            review_round=0,
        )
        for n, group in enumerate(groups, start=start)
    )


def next_bug_start(ticket_ids: Iterable[str], parent_id: str) -> int:
    """One past the highest `<parent_id>-bug-N` id already among `ticket_ids`.

    A second review round must not reuse the first round's ids — the caller
    passes this to `to_bug_tickets` as `start`. Ids alone, because that is all
    the answer depends on: the caller has whatever it holds them in.
    """
    prefix = f"{parent_id}-bug-"
    used = [
        int(ticket_id[len(prefix) :])
        for ticket_id in ticket_ids
        if ticket_id.startswith(prefix) and ticket_id[len(prefix) :].isdigit()
    ]
    return max(used, default=0) + 1


# -- what the triage agent produces ------------------------------------------

GROUPS_KEY = "groups"
_GROUP_REQUIRED = ["title", "deliverables", "findings"]

TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        GROUPS_KEY: {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "deliverables": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "findings": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": _GROUP_REQUIRED,
                "additionalProperties": False,
            },
        }
    },
    "required": [GROUPS_KEY],
    "additionalProperties": False,
}
"""The schema `tools.save_triage` hands the model for its `groups` argument.

Read-only by convention, like `agent.NO_PARAMS` — hand it to a `Tool` rather
than mutating it."""


def bug_groups_from_json(data: Any, *, allow_empty: bool = False) -> tuple[BugGroup, ...]:
    """Parse triage-agent output into groups, raising `InvalidGroupsError`.

    Re-checks everything `TRIAGE_SCHEMA` states, the same reasoning as
    `findings_from_json`: a schema handed to a model is a request, not a
    guarantee. Whether the groups it describes actually cover every `HIGH`
    finding is `check_coverage`'s question, not this one's.

    `allow_empty` separates the two readers. An agent that produced no groups
    was asked to group findings and did not, so the default refuses it. A
    triage document read back off disk is a *recorded* outcome, and "this round
    left nothing to fix" is one of the outcomes there is to record — the caller
    reading its own writing passes `allow_empty=True`.
    """
    payload = _object(data, "output", InvalidGroupsError)
    _known_fields(payload, [GROUPS_KEY], (), "output", InvalidGroupsError)
    raw = payload[GROUPS_KEY]
    if not isinstance(raw, list):
        raise InvalidGroupsError(f"{GROUPS_KEY!r} must be an array, got {_kind(raw)}")
    if not raw and not allow_empty:
        raise InvalidGroupsError(f"{GROUPS_KEY!r} is empty: triage must produce at least one group")

    return tuple(_one_group(item, index) for index, item in enumerate(raw))


def _one_group(item: Any, index: int) -> BugGroup:
    where = f"group {index}"
    fields = _object(item, where, InvalidGroupsError)
    _known_fields(fields, _GROUP_REQUIRED, (), where, InvalidGroupsError)

    title = _text(fields["title"], "title", where, InvalidGroupsError)
    deliverables = _text_list(
        fields["deliverables"], "deliverables", where, allow_empty=False, error=InvalidGroupsError
    )
    findings = _text_list(
        fields["findings"], "findings", where, allow_empty=False, error=InvalidGroupsError
    )

    return BugGroup(title=title, deliverables=deliverables, findings=findings)


# -- storage keys -----------------------------------------------------------


def review_key(ticket_id: str, round_: int, source: str) -> str:
    """Where one reviewer's findings for one ticket and round are stored.

    Round is in the key because a ticket is reviewed again after its bugs
    merge, and round 1 must not be overwritten.

    `review_round` counts *completed* reviews and is bumped on leaving
    `IN_REVIEW`, so during a ticket's first review it is still `0` and that
    review's findings land at `round-0`. That is correct, not an off-by-one:
    the counter is tracking how many reviews have finished, not which review
    is in flight.
    """
    return f"reviews/{ticket_id}/round-{round_}/{source}.json"


# -- validation helpers -------------------------------------------------------


def _object(value: Any, where: str, error: type[Exception]) -> dict[str, Any]:
    """Narrow `value` to a JSON object or raise `error`."""
    if not isinstance(value, dict):
        raise error(f"{where} must be an object, got {_kind(value)}")
    return value


def _known_fields(
    fields: dict[str, Any],
    required: list[str],
    optional: tuple[str, ...] | list[str],
    where: str,
    error: type[Exception],
) -> None:
    """Require every `required` key and refuse anything outside the two lists."""
    for name in required:
        if name not in fields:
            raise error(f"{where}: missing required field {name!r}")
    allowed = {*required, *optional}
    for name in fields:
        if name not in allowed:
            raise error(f"{where}: unknown field {name!r}")


def _text(value: Any, name: str, where: str, error: type[Exception]) -> str:
    """Narrow `value` to non-empty text or raise `error`."""
    if not isinstance(value, str) or not value.strip():
        raise error(f"{where}: {name} must be non-empty text, got {value!r}")
    return value


def _text_list(
    value: Any, name: str, where: str, *, allow_empty: bool, error: type[Exception]
) -> tuple[str, ...]:
    """Narrow `value` to a tuple of non-empty strings or raise `error`."""
    if not isinstance(value, list):
        raise error(f"{where}: {name} must be an array, got {_kind(value)}")
    if not value and not allow_empty:
        raise error(f"{where}: {name} is empty")
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise error(f"{where}: every {name} entry must be non-empty text")
    return tuple(value)


def _kind(value: Any) -> str:
    """What something is, for an error message a person has to act on."""
    return type(value).__name__
