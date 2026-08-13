"""Review findings, and the transformation from findings into bug tickets."""

from typing import Any

import pytest

from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.reviews import (
    FINDINGS_SCHEMA,
    TRIAGE_SCHEMA,
    BugGroup,
    CoverageError,
    Finding,
    InvalidFindingsError,
    InvalidGroupsError,
    Severity,
    bug_groups_from_json,
    check_coverage,
    findings_from_json,
    high,
    review_key,
    to_bug_tickets,
)

# -- fixtures as data -------------------------------------------------------


def finding(**overrides: Any) -> dict[str, Any]:
    """A single findings-schema entry, with any field overridable."""
    payload: dict[str, Any] = {
        "id": "Q-1",
        "severity": "high",
        "title": "Missing null check",
        "detail": "auth() does not check for a None token.",
        "files": ["src/auth.py"],
    }
    payload.update(overrides)
    return payload


def findings_payload(*items: dict[str, Any]) -> dict[str, Any]:
    return {"findings": list(items) if items else [finding()]}


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


def group_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": "Fix null check",
        "deliverables": ["Guard against a None token in auth()"],
        "findings": ["Q-1"],
    }
    payload.update(overrides)
    return payload


# -- findings: round trip ----------------------------------------------------


def test_findings_round_trip() -> None:
    payload = findings_payload(finding(id="Q-1"), finding(id="Q-2", severity="medium"))

    result = findings_from_json(payload)

    assert result == (
        a_finding(id="Q-1"),
        a_finding(id="Q-2", severity=Severity.MEDIUM),
    )


# -- findings: rejections -----------------------------------------------------


def test_rejects_duplicate_id() -> None:
    payload = findings_payload(finding(id="Q-1"), finding(id="Q-1"))
    with pytest.raises(InvalidFindingsError, match="duplicate"):
        findings_from_json(payload)


def test_rejects_unknown_severity() -> None:
    payload = findings_payload(finding(severity="critical"))
    with pytest.raises(InvalidFindingsError):
        findings_from_json(payload)


def test_rejects_empty_files() -> None:
    payload = findings_payload(finding(files=[]))
    with pytest.raises(InvalidFindingsError):
        findings_from_json(payload)


def test_rejects_missing_field() -> None:
    entry = finding()
    del entry["detail"]
    with pytest.raises(InvalidFindingsError, match="detail"):
        findings_from_json({"findings": [entry]})


def test_rejects_unknown_extra_field() -> None:
    payload = findings_payload(finding(extra="nope"))
    with pytest.raises(InvalidFindingsError, match="extra"):
        findings_from_json(payload)


def test_rejects_non_object_entry() -> None:
    with pytest.raises(InvalidFindingsError):
        findings_from_json({"findings": ["not an object"]})


def test_rejects_non_array() -> None:
    with pytest.raises(InvalidFindingsError):
        findings_from_json({"findings": "not an array"})


def test_rejects_non_object_payload() -> None:
    with pytest.raises(InvalidFindingsError):
        findings_from_json(["not", "an", "object"])


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


# -- review_key -------------------------------------------------------------


def test_review_key_shape() -> None:
    assert review_key("T-03", 1, "quality") == "reviews/T-03/round-1/quality.json"


def test_review_key_is_distinct_per_ticket_round_and_source() -> None:
    keys = {
        review_key("T-03", 1, "quality"),
        review_key("T-03", 2, "quality"),
        review_key("T-03", 1, "spec"),
        review_key("T-04", 1, "quality"),
    }

    assert len(keys) == 4


# -- triage groups (agent output) --------------------------------------------


def test_bug_groups_round_trip() -> None:
    payload = {"groups": [group_payload()]}

    assert bug_groups_from_json(payload) == (
        BugGroup(
            title="Fix null check",
            deliverables=("Guard against a None token in auth()",),
            findings=("Q-1",),
        ),
    )


def test_bug_groups_rejects_missing_field() -> None:
    entry = group_payload()
    del entry["findings"]
    with pytest.raises(InvalidGroupsError, match="findings"):
        bug_groups_from_json({"groups": [entry]})


def test_bug_groups_rejects_unknown_field() -> None:
    with pytest.raises(InvalidGroupsError, match="extra"):
        bug_groups_from_json({"groups": [group_payload(extra="nope")]})


def test_bug_groups_rejects_non_array() -> None:
    with pytest.raises(InvalidGroupsError):
        bug_groups_from_json({"groups": "nope"})


# -- schemas are what they claim to be ---------------------------------------


def test_findings_schema_requires_the_findings_field() -> None:
    assert FINDINGS_SCHEMA["required"] == ["findings"]


def test_triage_schema_requires_the_groups_field() -> None:
    assert TRIAGE_SCHEMA["required"] == ["groups"]
