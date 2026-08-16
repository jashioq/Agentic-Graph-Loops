"""The two review documents: what a reviewer and triage have to produce.

The schemas are what the agents are asked for and the parsers are what the run
trusts, so the two are checked against each other rather than separately.
"""

from typing import Any

import pytest

from agl.workflows.tickets.documents.review_documents import (
    FINDINGS_SCHEMA,
    TRIAGE_SCHEMA,
    bug_groups_from_json,
    findings_from_json,
)
from agl.workflows.tickets.errors import InvalidFindingsError, InvalidGroupsError
from agl.workflows.tickets.findings import BugGroup, Finding, Severity

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


def test_bug_groups_rejects_an_empty_array_by_default() -> None:
    with pytest.raises(InvalidGroupsError, match="at least one group"):
        bug_groups_from_json({"groups": []})


def test_bug_groups_accepts_an_empty_array_when_asked_to() -> None:
    # Reading a recorded outcome back, not judging an agent's answer: a round
    # that produced nothing to fix is written down as an empty `groups` array.
    assert bug_groups_from_json({"groups": []}, allow_empty=True) == ()


def test_bug_groups_still_checks_shape_when_empty_is_allowed() -> None:
    with pytest.raises(InvalidGroupsError):
        bug_groups_from_json({"groups": "nope"}, allow_empty=True)


# -- schemas are what they claim to be ---------------------------------------


def test_findings_schema_requires_the_findings_field() -> None:
    assert FINDINGS_SCHEMA["required"] == ["findings"]


def test_triage_schema_requires_the_groups_field() -> None:
    assert TRIAGE_SCHEMA["required"] == ["groups"]
