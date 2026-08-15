"""Per-role prompt assembly and the calls themselves.

Layer: workflows. The only file in the workflow that calls an agent at all.
Imports `agl.runtime.agents` for the call itself and for prompt rendering,
`agl.core.agent` and `agl.core.store` for the types it holds, and this
workflow's `models`, `tools`, and `reviews`. What is left here is per-role
knowledge: which prompt, which tools, what is denied, and what to read back.

Every role reports through a tool, never through `output_schema`: a tool call
the workflow can validate lets the agent read the error and correct itself in
the same session, where a bad `output_schema` result fails the whole call and
sends it back round the retry ladder to redo work it already did. `interview`
and `decompose` have always worked this way; `review` and `triage` write their
result through `save_findings` and `save_triage` and this file reads it back
from the store afterward. A role that ends without calling its tool raises
`RoleIncompleteError` naming the role and the ticket — a missing key is a
failure to surface, not an empty result to return quietly.

**`review` runs both reviewers as parallel top-level calls**, never as
subagents: `AskUserQuestion` is unavailable to subagents spawned via the Agent
tool, and a reviewer that cannot ask anything is a reviewer working from
guesses. Both are awaited together: `asyncio.gather` without
`return_exceptions` raises on the first failure, so a reviewer that never
finished never reaches the read-back below. Each reviewer persists its own
findings itself, through its own tool call, as soon as it makes one — so a
review that fails after the other reviewer already saved leaves that one
write standing; `RoleIncompleteError` is what still stops the ticket from
moving on with half a review.

**Triage gets the findings and the parent ticket's deliverables, and no
code.** It is denied the file and shell tools entirely: if it needed to read
source to write a deliverable, the finding was too vague, and that is a
reviewer-prompt problem worth surfacing rather than papering over. Zero `HIGH`
findings skips the call and returns nothing to fix; exactly one skips it too,
turning that finding directly into a group, because there is nothing left for
an agent to decide.

Git writes are denied, not discouraged: Python owns commits and branches, and
`GIT_WRITES` holds under every permission mode, including `bypassPermissions`,
so it still applies when a run is unattended.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from agl.core.agent import AgentQuestion, AgentRunner, Tool
from agl.core.store import Store
from agl.runtime.agents import Limits, PromptError, Prompts, call
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.models import Ticket
from agl.workflows.tickets.reviews import (
    BugGroup,
    Finding,
    bug_groups_from_json,
    check_coverage,
    findings_from_json,
    high,
    review_key,
)

__all__ = [
    "GIT_WRITES",
    "AgentContext",
    "Limits",
    "PromptError",
    "RoleIncompleteError",
    "decompose",
    "implement",
    "interview",
    "review",
    "triage",
]

GIT_WRITES: tuple[str, ...] = (
    "Bash(git commit:*)",
    "Bash(git checkout:*)",
    "Bash(git push:*)",
    "Bash(git merge:*)",
    "Bash(git worktree:*)",
    "Bash(git rebase:*)",
    "Bash(git reset:*)",
)
"""Denies the writes Python owns, on every role that runs in a worktree.

Not a policy that discourages — a scoped deny that holds in every permission
mode, including `bypassPermissions`. An agent running `git checkout main`
inside its worktree silently moves it off the ticket branch, and a stray
`git commit` breaks the one-commit guarantee.
"""

_NO_FILE_ACCESS: tuple[str, ...] = ("Read", "Write", "Edit", "Glob", "Grep", "Bash", "NotebookEdit")
"""Denies triage the file and shell tools entirely — it works from the prompt
and its own judgement, nothing else."""

_ActivityCallback = Callable[[str], None] | None


@dataclass(frozen=True)
class AgentContext:
    """What every role needs to run: where things are, and what to run them with."""

    runner: AgentRunner
    store: Store
    repo: Path
    prompts: Path
    limits: Limits
    settings: Path | None = None
    ask: Callable[[AgentQuestion], Awaitable[str]] | None = None


class RoleIncompleteError(Exception):
    """Raised when a role's run ended without calling the tool that reports its result.

    Not treated as "found nothing": a missing key is a run that stopped short,
    and the caller has no way to tell that apart from a genuine empty result
    unless this is raised instead of returned.
    """


# -- the roles ----------------------------------------------------------------


async def interview(ctx: AgentContext, user_input: str, on_activity: _ActivityCallback) -> None:
    """Interrogate the user about what to build. Writes the spec through its tools."""
    await _call(
        ctx,
        role="interview",
        prompt=_prompt(ctx, "interview", user_input=user_input),
        cwd=ctx.repo,
        agent_tools=ticket_tools.interview_tools(ctx.store),
        permission_mode="plan",
        on_activity=on_activity,
    )


async def decompose(ctx: AgentContext, on_activity: _ActivityCallback) -> None:
    """Break the spec into tickets. Writes them through its tools."""
    await _call(
        ctx,
        role="decompose",
        prompt=_prompt(ctx, "decompose"),
        cwd=ctx.repo,
        agent_tools=ticket_tools.decompose_tools(ctx.store),
        on_activity=on_activity,
    )


async def implement(
    ctx: AgentContext, ticket: Ticket, tree: Path, on_activity: _ActivityCallback
) -> None:
    """Do one ticket's work in its worktree. What it produces is a commit, not a document."""
    prompt_name = "implement_bug" if ticket.is_bug else "implement"
    await _call(
        ctx,
        role="implement",
        prompt=_prompt(ctx, prompt_name),
        cwd=tree,
        agent_tools=ticket_tools.implement_tools(ctx.store, ticket.id),
        disallowed_tools=GIT_WRITES,
        on_activity=on_activity,
    )


async def review(
    ctx: AgentContext,
    ticket: Ticket,
    tree: Path,
    base_branch: str,
    on_activity: _ActivityCallback,
) -> tuple[Finding, ...]:
    """Both reviewers, run concurrently, persisted and returned together.

    `base_branch` is substituted into both reviewer prompts so each can run
    its own `git diff $base_branch...HEAD` — three dots, so it shows only
    what this ticket's branch added since diverging.

    Either one failing raises before either is persisted: `asyncio.gather`
    without `return_exceptions` propagates the first failure immediately, and
    nothing below it runs.
    """
    # Both prompts are rendered before either call starts, so a template bug
    # fails the review without having run half of it.
    quality_prompt = _prompt(ctx, "review_quality", base_branch=base_branch)
    spec_prompt = _prompt(ctx, "review_spec", base_branch=base_branch)

    await asyncio.gather(
        _review(ctx, "review-quality", quality_prompt, ticket, tree, on_activity, "quality"),
        _review(ctx, "review-spec", spec_prompt, ticket, tree, on_activity, "spec"),
    )

    quality_findings = _read_findings(ctx, "review-quality", ticket, "quality")
    spec_findings = _read_findings(ctx, "review-spec", ticket, "spec")

    return quality_findings + spec_findings


async def triage(
    ctx: AgentContext,
    ticket: Ticket,
    findings: Sequence[Finding],
    on_activity: _ActivityCallback,
) -> tuple[BugGroup, ...]:
    """Group the `HIGH` findings into bug tickets one agent can fix in a pass.

    Skips the call entirely when there is nothing to decide: zero `HIGH`
    findings means merge with no call at all, and exactly one means a single
    group built directly from that finding.
    """
    highs = high(findings)
    if not highs:
        return ()
    if len(highs) == 1:
        only = highs[0]
        return (BugGroup(title=only.title, deliverables=(only.detail,), findings=(only.id,)),)

    await _call(
        ctx,
        role="triage",
        prompt=_prompt(
            ctx,
            "triage",
            findings=_render_findings(highs),
            deliverables=_render_list(ticket.deliverables),
        ),
        cwd=ctx.repo,
        agent_tools=ticket_tools.triage_tools(ctx.store, ticket.id, ticket.review_round, highs),
        disallowed_tools=_NO_FILE_ACCESS,
        on_activity=_prefixed(on_activity, "triage"),
    )

    key = review_key(ticket.id, ticket.review_round, "triage")
    if not ctx.store.exists(key):
        raise RoleIncompleteError(f"triage on {ticket.id!r} ended without calling save_triage")
    groups = bug_groups_from_json(ctx.store.read_json(key))
    # A backstop, not the enforcement: `save_triage` already refused to store
    # groups that fail this, so a failure here means the tool was bypassed or
    # the store was written to by something else.
    check_coverage(groups, highs)
    return groups


# -- spec assembly --------------------------------------------------------


async def _review(
    ctx: AgentContext,
    role: str,
    prompt: str,
    ticket: Ticket,
    tree: Path,
    on_activity: _ActivityCallback,
    label: str,
) -> None:
    factory = (
        ticket_tools.review_quality_tools
        if role == "review-quality"
        else ticket_tools.review_spec_tools
    )
    await _call(
        ctx,
        role=role,
        prompt=prompt,
        cwd=tree,
        agent_tools=factory(ctx.store, ticket.id, ticket.review_round),
        disallowed_tools=GIT_WRITES,
        on_activity=_prefixed(on_activity, label),
    )


def _read_findings(
    ctx: AgentContext, role: str, ticket: Ticket, source: str
) -> tuple[Finding, ...]:
    """One reviewer's findings, read back from where its `save_findings` wrote them."""
    key = review_key(ticket.id, ticket.review_round, source)
    if not ctx.store.exists(key):
        raise RoleIncompleteError(f"{role} on {ticket.id!r} ended without calling save_findings")
    return findings_from_json(ctx.store.read_json(key))


async def _call(
    ctx: AgentContext,
    *,
    role: str,
    prompt: str,
    cwd: Path,
    agent_tools: tuple[Tool, ...],
    disallowed_tools: tuple[str, ...] = (),
    permission_mode: str = "default",
    on_activity: _ActivityCallback = None,
) -> None:
    """One role's call, with everything the context carries threaded onto it.

    Nothing is returned: every role here reports through a tool, and what it
    wrote is read back from the store rather than from a result.
    """
    await call(
        ctx.runner,
        role=role,
        prompt=prompt,
        cwd=cwd,
        tools=agent_tools,
        disallowed=disallowed_tools,
        permission_mode=permission_mode,
        limits=ctx.limits,
        on_activity=on_activity,
        ask=ctx.ask,
    )


def _prefixed(on_activity: _ActivityCallback, label: str) -> _ActivityCallback:
    """`label · activity`, or `None` when nobody is watching.

    All three review roles write to the same ticket's row and last writer
    wins, so each is told apart by what it says rather than where it says it.
    """
    if on_activity is None:
        return None

    def wrapped(activity: str) -> None:
        on_activity(f"{label} · {activity}")

    return wrapped


# -- prompts ----------------------------------------------------------------


def _prompt(ctx: AgentContext, name: str, **substitutions: str) -> str:
    """Load `prompts/<name>.md` and substitute it, raising on anything left unfilled."""
    return Prompts(ctx.prompts).render(name, **substitutions)


def _render_findings(findings: Sequence[Finding]) -> str:
    """`HIGH` findings, rendered for a prompt — triage has no tool to fetch them itself."""
    return "\n".join(
        f"- {finding.id}: {finding.title}\n"
        f"  {finding.detail}\n"
        f"  files: {', '.join(finding.files)}"
        for finding in findings
    )


def _render_list(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
