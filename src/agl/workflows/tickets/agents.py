"""The five roles: which prompt each runs, which tools it holds, what it reads back.

Layer: workflows. The only file here that calls an agent. Every role reports
through a tool, not an `output_schema`; one that ends without calling it raises
`RoleIncompleteError`, and a report already on disk is read rather than re-run.
Judgement roles run on opus, execution roles on sonnet.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agl.core.agent import Model, Tool
from agl.runtime.agents import Activity, Ask, Prompts, call
from agl.runtime.context import RunContext
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.documents.review_documents import (
    GROUPS_KEY,
    bug_groups_from_json,
    findings_from_json,
)
from agl.workflows.tickets.documents.store_keys import REVIEWERS, review_key
from agl.workflows.tickets.errors import RoleIncompleteError
from agl.workflows.tickets.findings import BugGroup, Finding, check_coverage, high
from agl.workflows.tickets.models import Ticket

__all__ = [
    "GIT_WRITES",
    "PROMPTS",
    "decompose",
    "implement",
    "interview",
    "review",
    "triage",
]

PROMPTS = Prompts(Path(__file__).parent / "prompts")
"""This workflow's own prompt files."""

# `GIT_WRITES` is a scoped deny, so it holds under `bypassPermissions` too.
GIT_WRITES: tuple[str, ...] = (
    "Bash(git commit:*)",
    "Bash(git checkout:*)",
    "Bash(git push:*)",
    "Bash(git merge:*)",
    "Bash(git worktree:*)",
    "Bash(git rebase:*)",
    "Bash(git reset:*)",
)
"""Denies the git writes Python owns, on every role that runs in a worktree."""

_NO_FILE_ACCESS: tuple[str, ...] = ("Read", "Write", "Edit", "Glob", "Grep", "Bash", "NotebookEdit")
"""Denies triage the file and shell tools entirely: it works from the prompt alone."""


# -- the roles ----------------------------------------------------------------


async def interview(
    ctx: RunContext, user_input: str, on_activity: Activity | None = None, ask: Ask | None = None
) -> None:
    """Interrogate the user about what to build. Writes the spec through its tools."""
    await _call(
        ctx,
        role="interview",
        prompt=PROMPTS.render("interview", user_input=user_input),
        cwd=ctx.project.repo,
        model=Model.OPUS,
        agent_tools=ticket_tools.interview_tools(ctx.store),
        permission_mode="plan",
        on_activity=on_activity,
        ask=ask,
    )


async def decompose(
    ctx: RunContext, on_activity: Activity | None = None, ask: Ask | None = None
) -> None:
    """Break the spec into tickets. Writes them through its tools."""
    await _call(
        ctx,
        role="decompose",
        prompt=PROMPTS.render("decompose"),
        cwd=ctx.project.repo,
        model=Model.OPUS,
        agent_tools=ticket_tools.decompose_tools(ctx.store),
        on_activity=on_activity,
        ask=ask,
    )


async def implement(
    ctx: RunContext,
    ticket: Ticket,
    tree: Path,
    on_activity: Activity | None = None,
    ask: Ask | None = None,
) -> None:
    """Do one ticket's work in its worktree. What it produces is a commit, not a document."""
    prompt_name = "implement_bug" if ticket.is_bug else "implement"
    await _call(
        ctx,
        role="implement",
        prompt=PROMPTS.render(prompt_name),
        cwd=tree,
        model=Model.SONNET,
        agent_tools=ticket_tools.implement_tools(ctx.store, ticket.id),
        disallowed_tools=GIT_WRITES,
        on_activity=on_activity,
        ask=ask,
    )


async def review(
    ctx: RunContext,
    ticket: Ticket,
    tree: Path,
    base_branch: str,
    on_activity: Activity | None = None,
    ask: Ask | None = None,
) -> tuple[Finding, ...]:
    """Runs whichever reviewers are still owed, returns both reviewers' findings.

    param: tree - the worktree each reviewer runs in
    param: base_branch - substituted into the prompts for `git diff $base...HEAD`
    return: tuple[Finding, ...] - quality's findings then spec's, read back off disk
    """
    # A reviewer skips itself when its findings document already exists.
    owed = tuple(
        source
        for source in REVIEWERS
        if not ctx.store.exists(review_key(ticket.id, ticket.review_round, source))
    )
    prompts = {
        source: PROMPTS.render(f"review_{source}", base_branch=base_branch) for source in owed
    }

    # Both reviewers run as top-level calls, never subagents: `AskUserQuestion`
    # is unavailable to a subagent, and a reviewer that cannot ask is guessing.
    await asyncio.gather(
        *(_review(ctx, source, prompts[source], ticket, tree, on_activity, ask) for source in owed)
    )

    findings: tuple[Finding, ...] = ()
    for source in REVIEWERS:
        findings += _read_findings(ctx, ticket, source)
    return findings


async def triage(
    ctx: RunContext,
    ticket: Ticket,
    findings: Sequence[Finding],
    on_activity: Activity | None = None,
    ask: Ask | None = None,
) -> tuple[BugGroup, ...]:
    """Groups the `HIGH` findings into bug tickets one agent can fix in a pass.

    Always leaves a triage document behind, and skips the agent call when zero or
    one `HIGH` finding leaves nothing to decide.

    param: findings - both reviewers' findings; only the `HIGH` ones are grouped
    return: tuple[BugGroup, ...] - parsed back out of the document just written
    """
    highs = high(findings)
    key = review_key(ticket.id, ticket.review_round, "triage")

    # Triage skips itself when its document already exists.
    if ctx.store.exists(key):
        return _read_groups(ctx, key, highs)

    if len(highs) <= 1:
        decided = {GROUPS_KEY: [_as_group(finding) for finding in highs]}
        ctx.store.write_json(key, decided)
        return bug_groups_from_json(decided, allow_empty=True)

    await _call(
        ctx,
        role="triage",
        prompt=PROMPTS.render(
            "triage",
            findings=_render_findings(highs),
            deliverables=_render_list(ticket.deliverables),
        ),
        cwd=ctx.project.repo,
        model=Model.SONNET,
        agent_tools=ticket_tools.triage_tools(ctx.store, ticket.id, ticket.review_round, highs),
        disallowed_tools=_NO_FILE_ACCESS,
        on_activity=_prefixed(on_activity, "triage"),
        ask=ask,
    )

    if not ctx.store.exists(key):
        raise RoleIncompleteError(f"triage on {ticket.id!r} ended without calling save_triage")
    return _read_groups(ctx, key, highs)


# -- internals ------------------------------------------------------------


async def _review(
    ctx: RunContext,
    source: str,
    prompt: str,
    ticket: Ticket,
    tree: Path,
    on_activity: Activity | None,
    ask: Ask | None,
) -> None:
    factory = (
        ticket_tools.review_quality_tools
        if source == "quality"
        else ticket_tools.review_spec_tools
    )
    await _call(
        ctx,
        role=f"review-{source}",
        prompt=prompt,
        cwd=tree,
        model=Model.OPUS,
        agent_tools=factory(ctx.store, ticket.id, ticket.review_round),
        disallowed_tools=GIT_WRITES,
        on_activity=_prefixed(on_activity, source),
        ask=ask,
    )


def _read_findings(ctx: RunContext, ticket: Ticket, source: str) -> tuple[Finding, ...]:
    """One reviewer's findings, read back from where its `save_findings` wrote them."""
    key = review_key(ticket.id, ticket.review_round, source)
    if not ctx.store.exists(key):
        raise RoleIncompleteError(
            f"review-{source} on {ticket.id!r} ended without calling save_findings"
        )
    return findings_from_json(ctx.store.read_json(key))


def _read_groups(ctx: RunContext, key: str, highs: Sequence[Finding]) -> tuple[BugGroup, ...]:
    """The triage document, parsed and re-checked against the findings it covers."""
    groups = bug_groups_from_json(ctx.store.read_json(key), allow_empty=True)
    check_coverage(groups, highs)
    return groups


def _as_group(finding: Finding) -> dict[str, Any]:
    """One finding turned straight into a one-group document entry."""
    return {"title": finding.title, "deliverables": [finding.detail], "findings": [finding.id]}


async def _call(
    ctx: RunContext,
    *,
    role: str,
    prompt: str,
    cwd: Path,
    model: Model,
    agent_tools: tuple[Tool, ...],
    disallowed_tools: tuple[str, ...] = (),
    permission_mode: str = "default",
    on_activity: Activity | None = None,
    ask: Ask | None = None,
) -> None:
    """One role's call. Returns nothing: a role reports by writing through a tool.

    `model` has no default, so a new role cannot silently inherit another's.
    """
    await call(
        ctx.agent,
        role=role,
        prompt=prompt,
        cwd=cwd,
        tools=agent_tools,
        disallowed=disallowed_tools,
        permission_mode=permission_mode,
        model=model,
        on_activity=on_activity,
        ask=ask,
    )


def _prefixed(on_activity: Activity | None, label: str) -> Activity | None:
    """Wraps `on_activity` to prefix `label · `, or `None` when nobody is watching."""
    if on_activity is None:
        return None

    def wrapped(activity: str) -> None:
        on_activity(f"{label} · {activity}")

    return wrapped


def _render_findings(findings: Sequence[Finding]) -> str:
    """Findings rendered into prompt text — triage has no tool to fetch them itself."""
    return "\n".join(
        f"- {finding.id}: {finding.title}\n"
        f"  {finding.detail}\n"
        f"  files: {', '.join(finding.files)}"
        for finding in findings
    )


def _render_list(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
