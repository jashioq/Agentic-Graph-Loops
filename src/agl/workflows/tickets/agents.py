"""Per-role prompt assembly and the calls themselves.

Layer: workflows. The only file in the workflow that constructs an
`AgentSpec`. Imports `agl.core.agent`, `agl.core.store`, and this workflow's
`models`, `tools`, and `reviews`.

`interview` and `decompose` write through their tools and return nothing.
`review` and `triage` return data the workflow consumes, so they use
`output_schema` — tools for what persists, `output_schema` for what the
workflow reads and acts on.

**`review` runs both reviewers as parallel top-level calls**, never as
subagents: `AskUserQuestion` is unavailable to subagents spawned via the Agent
tool, and a reviewer that cannot ask anything is a reviewer working from
guesses. Both are awaited together, so a failure from either one raises before
anything is persisted — there is no half-reported review.

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
from string import Template

from agl.core.agent import AgentQuestion, AgentRunner, AgentSpec, Tool
from agl.core.store import Store
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.models import Ticket
from agl.workflows.tickets.reviews import (
    FINDINGS_SCHEMA,
    TRIAGE_SCHEMA,
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
class Limits:
    """Ceilings threaded onto every `AgentSpec` this file builds."""

    model: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None


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


class PromptError(Exception):
    """Raised when a prompt could not be assembled — usually a missing substitution."""


# -- the roles ----------------------------------------------------------------


async def interview(ctx: AgentContext, user_input: str) -> None:
    """Interrogate the user about what to build. Writes the spec through its tools."""
    spec = _spec(
        ctx,
        role="interview",
        prompt=_prompt(ctx, "interview", user_input=user_input),
        cwd=ctx.repo,
        agent_tools=ticket_tools.interview_tools(ctx.store),
        permission_mode="plan",
    )
    await ctx.runner.run(spec, None, ctx.ask)


async def decompose(ctx: AgentContext) -> None:
    """Break the spec into tickets. Writes them through its tools."""
    spec = _spec(
        ctx,
        role="decompose",
        prompt=_prompt(ctx, "decompose"),
        cwd=ctx.repo,
        agent_tools=ticket_tools.decompose_tools(ctx.store),
    )
    await ctx.runner.run(spec, None, ctx.ask)


async def implement(
    ctx: AgentContext, ticket: Ticket, tree: Path, on_activity: _ActivityCallback
) -> None:
    """Do one ticket's work in its worktree. What it produces is a commit, not a document."""
    prompt_name = "implement_bug" if ticket.is_bug else "implement"
    spec = _spec(
        ctx,
        role="implement",
        prompt=_prompt(ctx, prompt_name),
        cwd=tree,
        agent_tools=ticket_tools.implement_tools(ctx.store, ticket.id),
        disallowed_tools=GIT_WRITES,
    )
    await ctx.runner.run(spec, on_activity, ctx.ask)


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
    quality_spec = _review_spec(ctx, "review-quality", "review_quality", ticket, tree, base_branch)
    spec_spec = _review_spec(ctx, "review-spec", "review_spec", ticket, tree, base_branch)

    quality_result, spec_result = await asyncio.gather(
        ctx.runner.run(quality_spec, _prefixed(on_activity, "quality"), ctx.ask),
        ctx.runner.run(spec_spec, _prefixed(on_activity, "spec"), ctx.ask),
    )

    quality_findings = findings_from_json(quality_result.structured)
    spec_findings = findings_from_json(spec_result.structured)

    ctx.store.write_json(
        review_key(ticket.id, ticket.review_round, "quality"), quality_result.structured
    )
    ctx.store.write_json(
        review_key(ticket.id, ticket.review_round, "spec"), spec_result.structured
    )

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

    spec = _spec(
        ctx,
        role="triage",
        prompt=_prompt(
            ctx,
            "triage",
            findings=_render_findings(highs),
            deliverables=_render_list(ticket.deliverables),
        ),
        cwd=ctx.repo,
        agent_tools=(),
        disallowed_tools=_NO_FILE_ACCESS,
        output_schema=TRIAGE_SCHEMA,
    )
    result = await ctx.runner.run(spec, _prefixed(on_activity, "triage"), ctx.ask)
    groups = bug_groups_from_json(result.structured)
    check_coverage(groups, highs)
    return groups


# -- spec assembly --------------------------------------------------------


def _review_spec(
    ctx: AgentContext, role: str, prompt_name: str, ticket: Ticket, tree: Path, base_branch: str
) -> AgentSpec:
    factory = (
        ticket_tools.review_quality_tools
        if role == "review-quality"
        else ticket_tools.review_spec_tools
    )
    return _spec(
        ctx,
        role=role,
        prompt=_prompt(ctx, prompt_name, base_branch=base_branch),
        cwd=tree,
        agent_tools=factory(ctx.store, ticket.id),
        disallowed_tools=GIT_WRITES,
        output_schema=FINDINGS_SCHEMA,
    )


def _spec(
    ctx: AgentContext,
    *,
    role: str,
    prompt: str,
    cwd: Path,
    agent_tools: tuple[Tool, ...],
    disallowed_tools: tuple[str, ...] = (),
    permission_mode: str = "default",
    output_schema: dict[str, object] | None = None,
) -> AgentSpec:
    """Every field `Limits` owns, threaded onto one spec, once."""
    return AgentSpec(
        prompt=prompt,
        cwd=cwd,
        role=role,
        tools=agent_tools,
        disallowed_tools=disallowed_tools,
        permission_mode=permission_mode,
        model=ctx.limits.model,
        max_turns=ctx.limits.max_turns,
        max_budget_usd=ctx.limits.max_budget_usd,
        output_schema=output_schema,
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
    path = ctx.prompts / f"{name}.md"
    template = Template(path.read_text())
    try:
        return template.substitute(**substitutions)
    except KeyError as error:
        raise PromptError(f"{path}: missing a substitution for {error}") from error


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
