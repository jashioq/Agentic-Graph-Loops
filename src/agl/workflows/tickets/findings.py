"""What a reviewer found, and the transformation from findings into bug tickets.

Layer: workflows. Pure — no agents, no I/O. Imports this workflow's `models`
and `errors` only; the reviewer and triage schemas and their parsers live in
`documents/review_documents.py`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from agl.workflows.tickets.errors import CoverageError
from agl.workflows.tickets.models import Status, Ticket

__all__ = [
    "BugGroup",
    "Finding",
    "Severity",
    "check_coverage",
    "high",
    "next_bug_start",
    "to_bug_tickets",
]


class Severity(Enum):
    """How much a finding matters. Only `HIGH` becomes a bug ticket."""

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


def high(findings: Sequence[Finding]) -> tuple[Finding, ...]:
    """Every `HIGH` finding, in the order it was given."""
    return tuple(finding for finding in findings if finding.severity is Severity.HIGH)


def check_coverage(groups: Sequence[BugGroup], highs: Sequence[Finding]) -> None:
    """Raise `CoverageError` unless every finding in `highs` is in exactly one group.

    A group naming an id outside `highs` — because it does not exist, or because
    it names a `MEDIUM` or `LOW` finding — fails the same check.
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
    """One bug ticket per group, ids assembled from `parent.id` starting at `start`."""
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
    passes this to `to_bug_tickets` as `start`.
    """
    prefix = f"{parent_id}-bug-"
    used = [
        int(ticket_id[len(prefix) :])
        for ticket_id in ticket_ids
        if ticket_id.startswith(prefix) and ticket_id[len(prefix) :].isdigit()
    ]
    return max(used, default=0) + 1
