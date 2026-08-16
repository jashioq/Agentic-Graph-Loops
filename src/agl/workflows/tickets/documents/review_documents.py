"""What a reviewer and the triage agent produce: the schemas, and the parsers.

Layer: workflows. Narrows untrusted JSON with `runtime.json_fields` and raises
`InvalidFindingsError` / `InvalidGroupsError`. Whether a set of groups covers
every `HIGH` finding is `findings.check_coverage`'s question, not this module's.
"""

from typing import Any

from agl.runtime.json_fields import (
    as_object,
    as_text,
    as_text_list,
    reject_unknown_fields,
    require_fields,
    type_name,
)
from agl.workflows.tickets.errors import InvalidFindingsError, InvalidGroupsError
from agl.workflows.tickets.findings import BugGroup, Finding, Severity

__all__ = [
    "FINDINGS_KEY",
    "FINDINGS_SCHEMA",
    "GROUPS_KEY",
    "TRIAGE_SCHEMA",
    "bug_groups_from_json",
    "findings_from_json",
]

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
"""The schema `tools.save_findings` hands the model. Read-only."""


def findings_from_json(data: Any) -> tuple[Finding, ...]:
    """Parses reviewer output into findings, raising `InvalidFindingsError`.

    Re-checks the whole schema, plus unique ids. No findings at all is valid.
    """
    payload = as_object(data, "output", InvalidFindingsError)
    require_fields(payload, [FINDINGS_KEY], "output", InvalidFindingsError)
    reject_unknown_fields(payload, [FINDINGS_KEY], "output", InvalidFindingsError)
    raw = payload[FINDINGS_KEY]
    if not isinstance(raw, list):
        raise InvalidFindingsError(f"{FINDINGS_KEY!r} must be an array, got {type_name(raw)}")

    findings = tuple(_one_finding(item, index) for index, item in enumerate(raw))
    _check_finding_ids(findings)
    return findings


def _one_finding(item: Any, index: int) -> Finding:
    where = f"finding {index}"
    fields = as_object(item, where, InvalidFindingsError)
    require_fields(fields, _FINDING_REQUIRED, where, InvalidFindingsError)
    reject_unknown_fields(fields, _FINDING_REQUIRED, where, InvalidFindingsError)

    finding_id = as_text(fields["id"], "id", where, InvalidFindingsError)
    where = f"finding {finding_id!r}"

    severity_raw = fields["severity"]
    try:
        severity = Severity(severity_raw)
    except ValueError as error:
        known = ", ".join(s.value for s in Severity)
        raise InvalidFindingsError(
            f"{where}: unknown severity {severity_raw!r}, expected one of {known}"
        ) from error

    return Finding(
        id=finding_id,
        severity=severity,
        title=as_text(fields["title"], "title", where, InvalidFindingsError),
        detail=as_text(fields["detail"], "detail", where, InvalidFindingsError),
        files=as_text_list(
            fields["files"], "files", where, InvalidFindingsError, allow_empty=False
        ),
    )


def _check_finding_ids(findings: tuple[Finding, ...]) -> None:
    seen: set[str] = set()
    for finding in findings:
        if finding.id in seen:
            raise InvalidFindingsError(f"duplicate finding id {finding.id!r}")
        seen.add(finding.id)


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
"""The schema `tools.save_triage` hands the model. Read-only."""


def bug_groups_from_json(data: Any, *, allow_empty: bool = False) -> tuple[BugGroup, ...]:
    """Parses triage-agent output into groups, raising `InvalidGroupsError`.

    param: allow_empty - `True` when reading a recorded outcome back off disk,
        where "nothing left to fix" is a result; `False` for fresh agent output
    return: tuple[BugGroup, ...] - shape only; coverage is `check_coverage`'s question
    """
    payload = as_object(data, "output", InvalidGroupsError)
    require_fields(payload, [GROUPS_KEY], "output", InvalidGroupsError)
    reject_unknown_fields(payload, [GROUPS_KEY], "output", InvalidGroupsError)
    raw = payload[GROUPS_KEY]
    if not isinstance(raw, list):
        raise InvalidGroupsError(f"{GROUPS_KEY!r} must be an array, got {type_name(raw)}")
    if not raw and not allow_empty:
        raise InvalidGroupsError(f"{GROUPS_KEY!r} is empty: triage must produce at least one group")

    return tuple(_one_group(item, index) for index, item in enumerate(raw))


def _one_group(item: Any, index: int) -> BugGroup:
    where = f"group {index}"
    fields = as_object(item, where, InvalidGroupsError)
    require_fields(fields, _GROUP_REQUIRED, where, InvalidGroupsError)
    reject_unknown_fields(fields, _GROUP_REQUIRED, where, InvalidGroupsError)

    return BugGroup(
        title=as_text(fields["title"], "title", where, InvalidGroupsError),
        deliverables=as_text_list(
            fields["deliverables"], "deliverables", where, InvalidGroupsError, allow_empty=False
        ),
        findings=as_text_list(
            fields["findings"], "findings", where, InvalidGroupsError, allow_empty=False
        ),
    )
