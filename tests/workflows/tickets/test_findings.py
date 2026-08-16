"""Review findings, and the transformation from findings into bug tickets."""

from typing import Any

import pytest

from agl.workflows.tickets.errors import CoverageError
from agl.workflows.tickets.findings import (
    BugGroup,
    Finding,
    Severity,
    check_coverage,
    high,
    to_bug_tickets,
)
from agl.workflows.tickets.models import Status, Ticket

# -- fixtures as data -------------------------------------------------------


def a_finding(**overrides: Any) -> Finding:
    """A parsed `Finding`, with any field overridable."""
    fields: dict[str, Any] = {
        "id": "Q-1",
        "severity": Severity.HIGH,
        "title": "Missing null check",
        "detail": "auth() does not check for a None token.",
        "files": ("src/auth.py",),
    }
    fields.update(overrides)
    return Finding(**fields)


def parent_ticket(**overrides: Any) -> Ticket:
    fields: dict[str, Any] = {
        "id": "T-03",
        "title": "Add auth",
        "status": Status.IN_REVIEW,
        "deliverables": ("src/auth.py",),
    }
    fields.update(overrides)
    return Ticket(**fields)


# -- high ----------------------------------------------------------------


def test_high_filters_and_preserves_order() -> None:
    findings = (
        a_finding(id="Q-1", severity=Severity.MEDIUM),
        a_finding(id="Q-2", severity=Severity.HIGH),
        a_finding(id="Q-3", severity=Severity.LOW),
        a_finding(id="Q-4", severity=Severity.HIGH),
    )

    assert high(findings) == (findings[1], findings[3])


# -- check_coverage --------------------------------------------------------


def test_check_coverage_passes_when_every_high_is_covered_once() -> None:
    highs = (a_finding(id="Q-1"), a_finding(id="Q-2"))
    groups = (BugGroup(title="fix", deliverables=("x",), findings=("Q-1", "Q-2")),)

    check_coverage(groups, highs)  # does not raise


def test_check_coverage_raises_when_a_high_is_missing() -> None:
    highs = (a_finding(id="Q-1"), a_finding(id="Q-2"))
    groups = (BugGroup(title="fix", deliverables=("x",), findings=("Q-1",)),)

    with pytest.raises(CoverageError, match="Q-2"):
        check_coverage(groups, highs)


def test_check_coverage_raises_when_a_high_appears_in_two_groups() -> None:
    highs = (a_finding(id="Q-1"),)
    groups = (
        BugGroup(title="a", deliverables=("x",), findings=("Q-1",)),
        BugGroup(title="b", deliverables=("y",), findings=("Q-1",)),
    )

    with pytest.raises(CoverageError, match="Q-1"):
        check_coverage(groups, highs)


def test_check_coverage_raises_when_a_group_names_an_unknown_id() -> None:
    highs = (a_finding(id="Q-1"),)
    groups = (BugGroup(title="a", deliverables=("x",), findings=("Q-1", "Q-99")),)

    with pytest.raises(CoverageError, match="Q-99"):
        check_coverage(groups, highs)


def test_check_coverage_raises_when_a_group_names_a_medium_or_low() -> None:
    # Q-2 is a real finding id, but it is not HIGH, so it is absent from `highs`.
    highs = (a_finding(id="Q-1"),)
    groups = (BugGroup(title="a", deliverables=("x",), findings=("Q-1", "Q-2")),)

    with pytest.raises(CoverageError, match="Q-2"):
        check_coverage(groups, highs)


# -- to_bug_tickets --------------------------------------------------------


def test_to_bug_tickets_produces_right_ids_parent_status_and_deliverables() -> None:
    parent = parent_ticket()
    groups = (
        BugGroup(title="Fix null check", deliverables=("d1", "d2"), findings=("Q-1",)),
        BugGroup(title="Fix other", deliverables=("d3",), findings=("Q-2",)),
    )

    tickets = to_bug_tickets(parent, groups, start=1)

    assert tickets == (
        Ticket(
            id="T-03-bug-1",
            title="Fix null check",
            status=Status.PENDING,
            deliverables=("d1", "d2"),
            parent="T-03",
            review_round=0,
        ),
        Ticket(
            id="T-03-bug-2",
            title="Fix other",
            status=Status.PENDING,
            deliverables=("d3",),
            parent="T-03",
            review_round=0,
        ),
    )


def test_start_offsets_ids_so_a_second_round_does_not_collide_with_the_first() -> None:
    parent = parent_ticket()
    groups = (BugGroup(title="Fix", deliverables=("d",), findings=("S-1",)),)

    tickets = to_bug_tickets(parent, groups, start=3)

    assert tickets[0].id == "T-03-bug-3"
