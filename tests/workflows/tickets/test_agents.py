"""Per-role prompt assembly and the calls themselves.

`FakeAgentRunner` throughout — no test here calls a real model. What is under
test is the `AgentSpec` each function builds and the parsing of what comes
back, so most assertions read `runner.specs` rather than a return value.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import AgentError, AgentQuestion, AgentResult, AgentRunner, AgentSpec
from agl.workflows.tickets.agents import (
    GIT_WRITES,
    AgentContext,
    Limits,
    PromptError,
    decompose,
    implement,
    interview,
    review,
    triage,
)
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.reviews import CoverageError, Finding, Severity, review_key
from tests.fakes import FakeAgentRunner, MemoryStore, ScriptedRun

# -- prompts fixture ---------------------------------------------------------

_PROMPTS: dict[str, str] = {
    "interview": "Talk to the user about: $user_input",
    "decompose": "Break the spec into tickets.",
    "implement": "Build exactly the ticket's deliverables.",
    "implement_bug": "Fix exactly the findings on this bug ticket.",
    "review_quality": "Review against the standards document only.",
    "review_spec": "Review against the ticket's deliverables only.",
    "triage": "Findings:\n$findings\n\nDeliverables:\n$deliverables",
}


def write_prompts(tmp_path: Path, **overrides: str) -> Path:
    """A `prompts/` directory with one markdown file per role."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    texts = {**_PROMPTS, **overrides}
    for name, text in texts.items():
        (prompts / f"{name}.md").write_text(text)
    return prompts


# -- fixtures as data --------------------------------------------------------


def feature_ticket(**overrides: Any) -> Ticket:
    fields: dict[str, Any] = {
        "id": "T-03",
        "title": "Add auth",
        "status": Status.IN_PROGRESS,
        "deliverables": ("TokenStore in data/auth/",),
    }
    fields.update(overrides)
    return Ticket(**fields)


def bug_ticket(**overrides: Any) -> Ticket:
    fields: dict[str, Any] = {
        "id": "T-03-bug-1",
        "title": "Fix null check",
        "status": Status.IN_PROGRESS,
        "deliverables": ("Guard against a None token",),
        "parent": "T-03",
    }
    fields.update(overrides)
    return Ticket(**fields)


def context(
    tmp_path: Path,
    runner: AgentRunner,
    *,
    prompts: Path | None = None,
    limits: Limits | None = None,
    ask: Callable[[AgentQuestion], Awaitable[str]] | None = None,
) -> AgentContext:
    return AgentContext(
        runner=runner,
        store=MemoryStore(),
        repo=tmp_path / "repo",
        prompts=prompts if prompts is not None else write_prompts(tmp_path),
        limits=limits if limits is not None else Limits(),
        ask=ask,
    )


def findings_result(*findings: dict[str, Any]) -> AgentResult:
    return AgentResult(
        text="done",
        structured={"findings": list(findings)},
        session_id="s-1",
        cost_usd=0.0,
        num_turns=1,
        duration_ms=0,
        terminal_reason="completed",
    )


def finding(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "Q-1",
        "severity": "high",
        "title": "Missing null check",
        "detail": "auth() does not check for a None token.",
        "files": ["src/auth.py"],
    }
    payload.update(overrides)
    return payload


def a_finding(**overrides: Any) -> Finding:
    """A parsed `Finding` — what `triage` takes, as opposed to `finding()`'s raw JSON."""
    fields: dict[str, Any] = {
        "id": "Q-1",
        "severity": Severity.HIGH,
        "title": "Missing null check",
        "detail": "auth() does not check for a None token.",
        "files": ("src/auth.py",),
    }
    fields.update(overrides)
    return Finding(**fields)


def groups_result(*groups: dict[str, Any]) -> AgentResult:
    return AgentResult(
        text="done",
        structured={"groups": list(groups)},
        session_id="s-1",
        cost_usd=0.0,
        num_turns=1,
        duration_ms=0,
        terminal_reason="completed",
    )


def group(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": "Fix null checks",
        "deliverables": ["Guard against a None token in auth()"],
        "findings": ["Q-1", "Q-2"],
    }
    payload.update(overrides)
    return payload


class _FailingRunner(AgentRunner):
    """Wraps a real fake, raising for one named role — a real `AgentRunner`.

    Used only to prove `review` propagates a failure rather than
    half-reporting; `FakeAgentRunner` itself has no way to make a whole call
    raise, only a tool inside one.
    """

    def __init__(self, ok: AgentRunner, fails_role: str) -> None:
        self._ok = ok
        self._fails_role = fails_role

    async def run(
        self,
        spec: AgentSpec,
        on_activity: Callable[[str], None] | None = None,
        on_question: Callable[[AgentQuestion], Awaitable[str]] | None = None,
    ) -> AgentResult:
        if spec.role == self._fails_role:
            raise AgentError(f"{spec.role} blew up")
        return await self._ok.run(spec, on_activity, on_question)


# -- each role's spec --------------------------------------------------------


async def test_interview_spec(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"interview": "ok"})
    ctx = context(tmp_path, runner)

    await interview(ctx, "Add password login")

    spec = runner.specs[0]
    assert spec.role == "interview"
    assert spec.cwd == ctx.repo
    assert [tool.name for tool in spec.tools] == ["save_spec"]
    assert spec.permission_mode == "plan"
    assert "Add password login" in spec.prompt


async def test_decompose_spec(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"decompose": "ok"})
    ctx = context(tmp_path, runner)

    await decompose(ctx)

    spec = runner.specs[0]
    assert spec.role == "decompose"
    assert spec.cwd == ctx.repo
    assert [tool.name for tool in spec.tools] == ["read_spec", "save_tickets"]


async def test_implement_spec(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"implement": "ok"})
    ctx = context(tmp_path, runner)
    tree = tmp_path / "tree"

    await implement(ctx, feature_ticket(), tree, None)

    spec = runner.specs[0]
    assert spec.role == "implement"
    assert spec.cwd == tree
    assert [tool.name for tool in spec.tools] == ["get_ticket", "read_spec", "read_standards"]
    assert spec.disallowed_tools == GIT_WRITES


async def test_review_specs(tmp_path: Path) -> None:
    runner = FakeAgentRunner(
        {"review-quality": findings_result(), "review-spec": findings_result()}
    )
    ctx = context(tmp_path, runner)
    tree = tmp_path / "tree"

    await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    by_role = {spec.role: spec for spec in runner.specs}
    assert by_role["review-quality"].cwd == tree
    assert [tool.name for tool in by_role["review-quality"].tools] == [
        "get_ticket",
        "read_standards",
    ]
    assert by_role["review-quality"].disallowed_tools == GIT_WRITES
    assert by_role["review-quality"].output_schema is not None

    assert by_role["review-spec"].cwd == tree
    assert [tool.name for tool in by_role["review-spec"].tools] == ["get_ticket", "read_spec"]
    assert by_role["review-spec"].disallowed_tools == GIT_WRITES
    assert by_role["review-spec"].output_schema is not None


async def test_triage_spec(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"triage": groups_result(group())})
    ctx = context(tmp_path, runner)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    await triage(ctx, feature_ticket(), findings, None)

    spec = runner.specs[0]
    assert spec.role == "triage"
    assert spec.tools == ()
    assert spec.output_schema is not None


# -- GIT_WRITES ---------------------------------------------------------------


async def test_git_writes_denied_on_every_worktree_role(tmp_path: Path) -> None:
    runner = FakeAgentRunner(
        {
            "implement": "ok",
            "review-quality": findings_result(),
            "review-spec": findings_result(),
        }
    )
    ctx = context(tmp_path, runner)
    tree = tmp_path / "tree"

    await implement(ctx, feature_ticket(), tree, None)
    await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    for spec in runner.specs:
        assert set(GIT_WRITES) <= set(spec.disallowed_tools)


def test_git_writes_are_scoped_bash_deny_rules() -> None:
    assert all(rule.startswith("Bash(git ") for rule in GIT_WRITES)
    assert "Bash(git commit:*)" in GIT_WRITES
    assert "Bash(git checkout:*)" in GIT_WRITES
    assert "Bash(git push:*)" in GIT_WRITES
    assert "Bash(git merge:*)" in GIT_WRITES
    assert "Bash(git worktree:*)" in GIT_WRITES
    assert "Bash(git rebase:*)" in GIT_WRITES
    assert "Bash(git reset:*)" in GIT_WRITES


# -- AskUserQuestion is never pre-allowed -------------------------------------


async def test_ask_user_question_never_appears_as_a_tool(tmp_path: Path) -> None:
    runner = FakeAgentRunner(
        {
            "interview": "ok",
            "decompose": "ok",
            "implement": "ok",
            "review-quality": findings_result(),
            "review-spec": findings_result(),
            "triage": groups_result(group()),
        }
    )
    ctx = context(tmp_path, runner)
    tree = tmp_path / "tree"

    await interview(ctx, "hello")
    await decompose(ctx)
    await implement(ctx, feature_ticket(), tree, None)
    await review(ctx, feature_ticket(review_round=1), tree, "main", None)
    await triage(ctx, feature_ticket(), (a_finding(id="Q-1"), a_finding(id="Q-2")), None)

    for spec in runner.specs:
        assert "AskUserQuestion" not in [tool.name for tool in spec.tools]


# -- implement: bug vs feature prompt -----------------------------------------


async def test_implement_uses_a_different_prompt_for_a_bug_ticket(tmp_path: Path) -> None:
    runner = FakeAgentRunner(["ok", "ok"])
    ctx = context(tmp_path, runner)
    tree = tmp_path / "tree"

    await implement(ctx, feature_ticket(), tree, None)
    await implement(ctx, bug_ticket(), tree, None)

    feature_prompt, bug_prompt = runner.specs[0].prompt, runner.specs[1].prompt
    assert feature_prompt != bug_prompt
    assert feature_prompt == _PROMPTS["implement"]
    assert bug_prompt == _PROMPTS["implement_bug"]


# -- review: two calls, concatenated, persisted -------------------------------


async def test_review_issues_two_calls_and_returns_both_lists_concatenated(
    tmp_path: Path,
) -> None:
    runner = FakeAgentRunner(
        {
            "review-quality": findings_result(finding(id="Q-1")),
            "review-spec": findings_result(finding(id="S-1", severity="medium")),
        }
    )
    ctx = context(tmp_path, runner)
    tree = tmp_path / "tree"

    findings = await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    assert {f.id for f in findings} == {"Q-1", "S-1"}


async def test_review_persists_both_under_the_right_review_key(tmp_path: Path) -> None:
    runner = FakeAgentRunner(
        {
            "review-quality": findings_result(finding(id="Q-1")),
            "review-spec": findings_result(finding(id="S-1")),
        }
    )
    ctx = context(tmp_path, runner)
    tree = tmp_path / "tree"

    await review(ctx, feature_ticket(review_round=2), tree, "main", None)

    assert ctx.store.read_json(review_key("T-03", 2, "quality")) == {
        "findings": [finding(id="Q-1")]
    }
    assert ctx.store.read_json(review_key("T-03", 2, "spec")) == {"findings": [finding(id="S-1")]}


async def test_review_propagates_a_failure_rather_than_half_reporting(tmp_path: Path) -> None:
    ok = FakeAgentRunner({"review-spec": findings_result(finding(id="S-1"))})
    runner = _FailingRunner(ok, fails_role="review-quality")
    ctx = context(tmp_path, runner)
    tree = tmp_path / "tree"

    with pytest.raises(AgentError):
        await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    assert ctx.store.list() == ()


# -- activity prefixes ---------------------------------------------------------


async def test_review_activity_is_prefixed_and_implement_is_not(tmp_path: Path) -> None:
    runner = FakeAgentRunner(
        {
            "implement": ScriptedRun("done", activity=("wrote a.py",)),
            "review-quality": ScriptedRun(result=findings_result(), activity=("read a.py",)),
            "review-spec": ScriptedRun(result=findings_result(), activity=("read spec",)),
        }
    )
    ctx = context(tmp_path, runner)
    tree = tmp_path / "tree"
    seen: list[str] = []

    await implement(ctx, feature_ticket(), tree, seen.append)
    await review(ctx, feature_ticket(review_round=1), tree, "main", seen.append)

    assert "wrote a.py" in seen
    assert "quality · read a.py" in seen
    assert "spec · read spec" in seen


async def test_triage_activity_is_prefixed(tmp_path: Path) -> None:
    runner = FakeAgentRunner(
        {"triage": ScriptedRun(result=groups_result(group()), activity=("thinking",))}
    )
    ctx = context(tmp_path, runner)
    seen: list[str] = []

    await triage(ctx, feature_ticket(), (a_finding(id="Q-1"), a_finding(id="Q-2")), seen.append)

    assert seen == ["triage · thinking"]


# -- triage: skip when there is nothing to decide -----------------------------


async def test_triage_skips_the_call_for_zero_high_findings(tmp_path: Path) -> None:
    runner = FakeAgentRunner([])  # no script needed: nothing should be called
    ctx = context(tmp_path, runner)
    findings = (
        a_finding(id="Q-1", severity=Severity.MEDIUM),
        a_finding(id="Q-2", severity=Severity.LOW),
    )

    groups = await triage(ctx, feature_ticket(), findings, None)

    assert groups == ()
    assert runner.specs == []


async def test_triage_skips_the_call_for_exactly_one_high_finding(tmp_path: Path) -> None:
    runner = FakeAgentRunner([])
    ctx = context(tmp_path, runner)
    only = a_finding(id="Q-1", title="Missing null check", detail="auth() can NPE")
    findings = (only, a_finding(id="Q-2", severity=Severity.LOW))

    groups = await triage(ctx, feature_ticket(), findings, None)

    assert runner.specs == []
    assert len(groups) == 1
    assert groups[0].findings == ("Q-1",)


async def test_triage_calls_the_agent_for_two_or_more_high_findings(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"triage": groups_result(group(findings=["Q-1", "Q-2"]))})
    ctx = context(tmp_path, runner)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    groups = await triage(ctx, feature_ticket(), findings, None)

    assert len(runner.specs) == 1
    assert [g.findings for g in groups] == [("Q-1", "Q-2")]


# -- triage: no file or shell tools -------------------------------------------


async def test_triage_has_no_file_or_shell_tools(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"triage": groups_result(group())})
    ctx = context(tmp_path, runner)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    await triage(ctx, feature_ticket(), findings, None)

    spec = runner.specs[0]
    assert spec.tools == ()
    assert "Bash" in spec.disallowed_tools
    assert "Read" in spec.disallowed_tools
    assert "Write" in spec.disallowed_tools
    assert "Edit" in spec.disallowed_tools


# -- triage: a coverage failure raises rather than returning ------------------


async def test_triage_result_failing_coverage_raises(tmp_path: Path) -> None:
    # Two HIGHs go in; the scripted triage only covers one of them.
    runner = FakeAgentRunner({"triage": groups_result(group(findings=["Q-1"]))})
    ctx = context(tmp_path, runner)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    with pytest.raises(CoverageError):
        await triage(ctx, feature_ticket(), findings, None)


# -- limits reach every spec ---------------------------------------------------


async def test_limits_reach_every_spec(tmp_path: Path) -> None:
    limits = Limits(model="claude-opus-5", max_turns=12, max_budget_usd=3.5)
    runner = FakeAgentRunner(
        {
            "interview": "ok",
            "decompose": "ok",
            "implement": "ok",
            "review-quality": findings_result(),
            "review-spec": findings_result(),
            "triage": groups_result(group()),
        }
    )
    ctx = context(tmp_path, runner, limits=limits)
    tree = tmp_path / "tree"

    await interview(ctx, "hello")
    await decompose(ctx)
    await implement(ctx, feature_ticket(), tree, None)
    await review(ctx, feature_ticket(review_round=1), tree, "main", None)
    await triage(ctx, feature_ticket(), (a_finding(id="Q-1"), a_finding(id="Q-2")), None)

    for spec in runner.specs:
        assert spec.model == "claude-opus-5"
        assert spec.max_turns == 12
        assert spec.max_budget_usd == 3.5


# -- prompt substitution -------------------------------------------------------


async def test_a_prompt_with_an_unsubstituted_placeholder_raises(tmp_path: Path) -> None:
    prompts = write_prompts(tmp_path, decompose="Break the spec into $unfilled tickets.")
    runner = FakeAgentRunner({"decompose": "ok"})
    ctx = context(tmp_path, runner, prompts=prompts)

    with pytest.raises(PromptError):
        await decompose(ctx)


async def test_interview_substitutes_user_input(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"interview": "ok"})
    ctx = context(tmp_path, runner)

    await interview(ctx, "Add password login")

    assert runner.specs[0].prompt == "Talk to the user about: Add password login"
