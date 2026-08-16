"""Per-role prompt assembly and the calls themselves.

`FakeAgentRunner` throughout — no test here calls a real model. What is under
test is the `AgentSpec` each function builds and the parsing of what comes
back, so most assertions read `runner.specs` rather than a return value.

Each role takes a `RunContext`, so every test here is one call to the harness
in `tests/runtime/conftest.py` with its own runner. Nothing here is about what
a prompt *says* — `PROMPTS` points at the package's real `prompts/` and
`test_prompts.py` is what reads them — nor about how one is rendered, which is
`Prompts` and lives in `tests/runtime/test_agents.py`.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import (
    AgentError,
    AgentQuestion,
    AgentResult,
    AgentRunner,
    AgentSpec,
    Model,
)
from agl.runtime.agents import PromptError, Prompts
from agl.runtime.context import RunContext
from agl.workflows.tickets import agents
from agl.workflows.tickets.agents import (
    GIT_WRITES,
    decompose,
    implement,
    interview,
    review,
    triage,
)
from agl.workflows.tickets.documents.store_keys import review_key
from agl.workflows.tickets.errors import CoverageError, RoleIncompleteError
from agl.workflows.tickets.findings import Finding, Severity
from agl.workflows.tickets.models import Status, Ticket
from tests.fakes import FakeAgentRunner, ScriptedRun
from tests.runtime.conftest import context as run_context

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


def context(repo: Path, runner: AgentRunner) -> RunContext:
    """One run's context over `repo`, with `runner` standing in for the model."""
    return run_context(repo, agent=runner)


def findings_result(*findings: dict[str, Any]) -> ScriptedRun:
    """A run that reports through `save_findings`, the way a real one now must."""
    return ScriptedRun(calls=(("save_findings", {"findings": list(findings)}),))


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


def groups_result(*groups: dict[str, Any]) -> ScriptedRun:
    """A run that reports through `save_triage`, the way a real one now must."""
    return ScriptedRun(calls=(("save_triage", {"groups": list(groups)}),))


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


async def test_interview_spec(repo: Path) -> None:
    runner = FakeAgentRunner({"interview": "ok"})
    ctx = context(repo, runner)

    await interview(ctx, "Add password login", None)

    spec = runner.specs[0]
    assert spec.role == "interview"
    assert spec.cwd == ctx.project.repo
    assert [tool.name for tool in spec.tools] == ["save_spec"]
    assert spec.permission_mode == "plan"
    assert "Add password login" in spec.prompt


async def test_decompose_spec(repo: Path) -> None:
    runner = FakeAgentRunner({"decompose": "ok"})
    ctx = context(repo, runner)

    await decompose(ctx, None)

    spec = runner.specs[0]
    assert spec.role == "decompose"
    assert spec.cwd == ctx.project.repo
    assert [tool.name for tool in spec.tools] == ["read_spec", "save_tickets"]


async def test_implement_spec(repo: Path) -> None:
    runner = FakeAgentRunner({"implement": "ok"})
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    await implement(ctx, feature_ticket(), tree, None)

    spec = runner.specs[0]
    assert spec.role == "implement"
    assert spec.cwd == tree
    assert [tool.name for tool in spec.tools] == ["get_ticket", "read_spec", "read_standards"]
    assert spec.disallowed_tools == GIT_WRITES


async def test_review_specs(repo: Path) -> None:
    runner = FakeAgentRunner(
        {"review-quality": findings_result(), "review-spec": findings_result()}
    )
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    by_role = {spec.role: spec for spec in runner.specs}
    assert by_role["review-quality"].cwd == tree
    assert [tool.name for tool in by_role["review-quality"].tools] == [
        "get_ticket",
        "read_standards",
        "save_findings",
    ]
    assert by_role["review-quality"].disallowed_tools == GIT_WRITES
    assert by_role["review-quality"].output_schema is None

    assert by_role["review-spec"].cwd == tree
    assert [tool.name for tool in by_role["review-spec"].tools] == [
        "get_ticket",
        "read_spec",
        "save_findings",
    ]
    assert by_role["review-spec"].disallowed_tools == GIT_WRITES
    assert by_role["review-spec"].output_schema is None


async def test_triage_spec(repo: Path) -> None:
    runner = FakeAgentRunner({"triage": groups_result(group())})
    ctx = context(repo, runner)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    await triage(ctx, feature_ticket(), findings, None)

    spec = runner.specs[0]
    assert spec.role == "triage"
    assert [tool.name for tool in spec.tools] == ["save_triage"]
    assert spec.output_schema is None


# -- GIT_WRITES ---------------------------------------------------------------


async def test_git_writes_denied_on_every_worktree_role(repo: Path) -> None:
    runner = FakeAgentRunner(
        {
            "implement": "ok",
            "review-quality": findings_result(),
            "review-spec": findings_result(),
        }
    )
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

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


async def test_ask_user_question_never_appears_as_a_tool(repo: Path) -> None:
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
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    await interview(ctx, "hello", None)
    await decompose(ctx, None)
    await implement(ctx, feature_ticket(), tree, None)
    await review(ctx, feature_ticket(review_round=1), tree, "main", None)
    await triage(ctx, feature_ticket(), (a_finding(id="Q-1"), a_finding(id="Q-2")), None)

    for spec in runner.specs:
        assert "AskUserQuestion" not in [tool.name for tool in spec.tools]


# -- implement: bug vs feature prompt -----------------------------------------


async def test_implement_uses_a_different_prompt_for_a_bug_ticket(repo: Path) -> None:
    runner = FakeAgentRunner(["ok", "ok"])
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    await implement(ctx, feature_ticket(), tree, None)
    await implement(ctx, bug_ticket(), tree, None)

    feature_prompt, bug_prompt = runner.specs[0].prompt, runner.specs[1].prompt
    assert feature_prompt != bug_prompt
    assert feature_prompt == agents.PROMPTS.render("implement")
    assert bug_prompt == agents.PROMPTS.render("implement_bug")


# -- review: two calls, concatenated, persisted -------------------------------


async def test_review_issues_two_calls_and_returns_both_lists_concatenated(
    repo: Path,
) -> None:
    runner = FakeAgentRunner(
        {
            "review-quality": findings_result(finding(id="Q-1")),
            "review-spec": findings_result(finding(id="S-1", severity="medium")),
        }
    )
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    findings = await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    assert {f.id for f in findings} == {"Q-1", "S-1"}


async def test_review_persists_both_under_the_right_review_key(repo: Path) -> None:
    runner = FakeAgentRunner(
        {
            "review-quality": findings_result(finding(id="Q-1")),
            "review-spec": findings_result(finding(id="S-1")),
        }
    )
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    await review(ctx, feature_ticket(review_round=2), tree, "main", None)

    assert ctx.store.read_json(review_key("T-03", 2, "quality")) == {
        "findings": [finding(id="Q-1")]
    }
    assert ctx.store.read_json(review_key("T-03", 2, "spec")) == {"findings": [finding(id="S-1")]}


async def test_a_reviewer_that_never_calls_save_findings_raises_naming_role_and_ticket(
    repo: Path,
) -> None:
    # Prose instead of the tool call — the exact failure that crashed a real
    # run on a clean review: the model summarized instead of reporting.
    runner = FakeAgentRunner(
        {
            "review-quality": ScriptedRun("Everything checks out cleanly."),
            "review-spec": findings_result(),
        }
    )
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    with pytest.raises(RoleIncompleteError, match="review-quality"):
        await review(ctx, feature_ticket(review_round=1), tree, "main", None)


async def test_a_clean_review_that_saves_empty_lists_returns_no_findings(repo: Path) -> None:
    # The case that crashed the real run: nothing to report is not an error.
    runner = FakeAgentRunner(
        {"review-quality": findings_result(), "review-spec": findings_result()}
    )
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    findings = await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    assert findings == ()


async def test_review_propagates_a_failure_rather_than_returning_half_a_review(
    repo: Path,
) -> None:
    # `review-spec` still saves its own findings through its own tool call —
    # that write is real and stands. What `review` guarantees is that a
    # failure on the other reviewer is not swallowed into a partial result:
    # the caller sees `AgentError` and does not treat this as a finished
    # review.
    ok = FakeAgentRunner({"review-spec": findings_result(finding(id="S-1"))})
    runner = _FailingRunner(ok, fails_role="review-quality")
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    with pytest.raises(AgentError):
        await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    assert ctx.store.list() == (review_key("T-03", 1, "spec"),)
    assert ctx.store.exists(review_key("T-03", 1, "quality")) is False


# -- review: a role whose report is already on disk is not run again -----------


async def test_review_runs_only_the_reviewer_whose_findings_are_missing(repo: Path) -> None:
    runner = FakeAgentRunner({"review-spec": findings_result(finding(id="S-1"))})
    ctx = context(repo, runner)
    ctx.store.write_json(review_key("T-03", 1, "quality"), {"findings": [finding(id="Q-1")]})
    tree = repo.parent / "tree"

    findings = await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    assert [spec.role for spec in runner.specs] == ["review-spec"]
    assert {f.id for f in findings} == {"Q-1", "S-1"}


async def test_review_with_both_findings_documents_present_calls_neither(repo: Path) -> None:
    runner = FakeAgentRunner([])
    ctx = context(repo, runner)
    ctx.store.write_json(review_key("T-03", 1, "quality"), {"findings": [finding(id="Q-1")]})
    ctx.store.write_json(review_key("T-03", 1, "spec"), {"findings": [finding(id="S-1")]})
    tree = repo.parent / "tree"

    findings = await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    assert runner.specs == []
    assert {f.id for f in findings} == {"Q-1", "S-1"}


# -- activity prefixes ---------------------------------------------------------


async def test_review_activity_is_prefixed_and_implement_is_not(repo: Path) -> None:
    runner = FakeAgentRunner(
        {
            "implement": ScriptedRun("done", activity=("wrote a.py",)),
            "review-quality": ScriptedRun(
                calls=(("save_findings", {"findings": []}),), activity=("read a.py",)
            ),
            "review-spec": ScriptedRun(
                calls=(("save_findings", {"findings": []}),), activity=("read spec",)
            ),
        }
    )
    ctx = context(repo, runner)
    tree = repo.parent / "tree"
    seen: list[str] = []

    await implement(ctx, feature_ticket(), tree, seen.append)
    await review(ctx, feature_ticket(review_round=1), tree, "main", seen.append)

    assert "wrote a.py" in seen
    assert "quality · read a.py" in seen
    assert "spec · read spec" in seen


async def test_triage_activity_is_prefixed(repo: Path) -> None:
    runner = FakeAgentRunner(
        {
            "triage": ScriptedRun(
                calls=(("save_triage", {"groups": [group()]}),), activity=("thinking",)
            )
        }
    )
    ctx = context(repo, runner)
    seen: list[str] = []

    await triage(ctx, feature_ticket(), (a_finding(id="Q-1"), a_finding(id="Q-2")), seen.append)

    assert seen == ["triage · thinking"]


async def test_interview_reports_activity_unprefixed(repo: Path) -> None:
    runner = FakeAgentRunner({"interview": ScriptedRun("noted", activity=("read spec.md",))})
    ctx = context(repo, runner)
    seen: list[str] = []

    await interview(ctx, "Add password login", seen.append)

    assert seen == ["read spec.md"]


async def test_decompose_reports_activity_unprefixed(repo: Path) -> None:
    runner = FakeAgentRunner({"decompose": ScriptedRun("planned", activity=("read spec.md",))})
    ctx = context(repo, runner)
    seen: list[str] = []

    await decompose(ctx, seen.append)

    assert seen == ["read spec.md"]


# -- triage: skip when there is nothing to decide -----------------------------


async def test_triage_skips_the_call_for_zero_high_findings(repo: Path) -> None:
    runner = FakeAgentRunner([])  # no script needed: nothing should be called
    ctx = context(repo, runner)
    findings = (
        a_finding(id="Q-1", severity=Severity.MEDIUM),
        a_finding(id="Q-2", severity=Severity.LOW),
    )

    groups = await triage(ctx, feature_ticket(), findings, None)

    assert groups == ()
    assert runner.specs == []


async def test_triage_skips_the_call_for_exactly_one_high_finding(repo: Path) -> None:
    runner = FakeAgentRunner([])
    ctx = context(repo, runner)
    only = a_finding(id="Q-1", title="Missing null check", detail="auth() can NPE")
    findings = (only, a_finding(id="Q-2", severity=Severity.LOW))

    groups = await triage(ctx, feature_ticket(), findings, None)

    assert runner.specs == []
    assert len(groups) == 1
    assert groups[0].findings == ("Q-1",)


async def test_triage_with_no_high_findings_still_writes_its_document(repo: Path) -> None:
    runner = FakeAgentRunner([])  # no script needed: nothing should be called
    ctx = context(repo, runner)
    findings = (
        a_finding(id="Q-1", severity=Severity.MEDIUM),
        a_finding(id="Q-2", severity=Severity.LOW),
    )

    groups = await triage(ctx, feature_ticket(), findings, None)

    assert groups == ()
    assert runner.specs == []
    assert ctx.store.read_json(review_key("T-03", 0, "triage")) == {"groups": []}


async def test_triage_with_one_high_finding_writes_the_group_it_built(repo: Path) -> None:
    runner = FakeAgentRunner([])
    ctx = context(repo, runner)
    only = a_finding(id="Q-1", title="Missing null check", detail="auth() can NPE")
    findings = (only, a_finding(id="Q-2", severity=Severity.LOW))

    await triage(ctx, feature_ticket(), findings, None)

    assert runner.specs == []
    assert ctx.store.read_json(review_key("T-03", 0, "triage")) == {
        "groups": [
            {
                "title": "Missing null check",
                "deliverables": ["auth() can NPE"],
                "findings": ["Q-1"],
            }
        ]
    }


async def test_triage_over_an_existing_document_reads_it_back_without_calling(
    repo: Path,
) -> None:
    runner = FakeAgentRunner({"triage": groups_result(group())})
    ctx = context(repo, runner)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    first = await triage(ctx, feature_ticket(), findings, None)
    second = await triage(ctx, feature_ticket(), findings, None)

    assert len(runner.specs) == 1
    assert second == first


async def test_triage_reads_back_a_recorded_empty_document_without_calling(repo: Path) -> None:
    # An empty `groups` array is never a legitimate agent result, but it is a
    # legitimate recorded one: it is how "this round found nothing to fix" is
    # written down.
    runner = FakeAgentRunner([])
    ctx = context(repo, runner)
    ctx.store.write_json(review_key("T-03", 0, "triage"), {"groups": []})

    findings = (a_finding(id="Q-1", severity=Severity.LOW),)

    groups = await triage(ctx, feature_ticket(), findings, None)

    assert groups == ()
    assert runner.specs == []


async def test_triage_calls_the_agent_for_two_or_more_high_findings(repo: Path) -> None:
    runner = FakeAgentRunner({"triage": groups_result(group(findings=["Q-1", "Q-2"]))})
    ctx = context(repo, runner)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    groups = await triage(ctx, feature_ticket(), findings, None)

    assert len(runner.specs) == 1
    assert [g.findings for g in groups] == [("Q-1", "Q-2")]


# -- triage: no file or shell tools -------------------------------------------


async def test_triage_has_no_file_or_shell_tools(repo: Path) -> None:
    runner = FakeAgentRunner({"triage": groups_result(group())})
    ctx = context(repo, runner)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    await triage(ctx, feature_ticket(), findings, None)

    spec = runner.specs[0]
    assert [tool.name for tool in spec.tools] == ["save_triage"]
    assert "Bash" in spec.disallowed_tools
    assert "Read" in spec.disallowed_tools
    assert "Write" in spec.disallowed_tools
    assert "Edit" in spec.disallowed_tools


# -- triage: never called its tool ---------------------------------------


async def test_triage_that_never_calls_save_triage_raises_naming_role_and_ticket(
    repo: Path,
) -> None:
    # The scripted run ends without calling its tool at all — prose instead of
    # the call, the exact failure this workflow moved off `output_schema` to
    # make loud instead of silently mistaken for "no groups".
    runner = FakeAgentRunner({"triage": ScriptedRun("Here is my analysis...")})
    ctx = context(repo, runner)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    with pytest.raises(RoleIncompleteError, match="triage"):
        await triage(ctx, feature_ticket(), findings, None)


# -- triage: check_coverage runs again as a backstop after reading back -------


async def test_triage_backstop_raises_if_the_store_holds_uncovered_groups(
    repo: Path,
) -> None:
    # `save_triage` already refuses groups that fail coverage, so the only way
    # this branch is reached is data landing at the key some other way — which
    # is exactly why `agents.triage` does not just trust that the tool ran.
    runner = FakeAgentRunner({"triage": ScriptedRun()})
    ctx = context(repo, runner)
    ctx.store.write_json(review_key("T-03", 0, "triage"), {"groups": [group(findings=["Q-1"])]})
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2"))

    with pytest.raises(CoverageError):
        await triage(ctx, feature_ticket(), findings, None)


# -- each role's model ---------------------------------------------------------


async def test_each_role_runs_on_the_model_it_chose(repo: Path) -> None:
    """The per-role split, pinned: judgement on opus, execution on sonnet.

    Every role is driven once and the assertion is keyed by `spec.role`, so
    changing any single role's model fails this and names which one.
    """
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
    ctx = context(repo, runner)
    tree = repo.parent / "tree"

    await interview(ctx, "hello", None)
    await decompose(ctx, None)
    await implement(ctx, feature_ticket(), tree, None)
    await review(ctx, feature_ticket(review_round=1), tree, "main", None)
    await triage(ctx, feature_ticket(), (a_finding(id="Q-1"), a_finding(id="Q-2")), None)

    assert {spec.role: spec.model for spec in runner.specs} == {
        "interview": Model.OPUS,
        "decompose": Model.OPUS,
        "review-quality": Model.OPUS,
        "review-spec": Model.OPUS,
        "implement": Model.SONNET,
        "triage": Model.SONNET,
    }


# -- a prompt that will not render ---------------------------------------------


async def test_a_prompt_with_an_unsubstituted_placeholder_fails_the_role(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`PromptError` reaches the caller instead of a `$placeholder` reaching a model.

    The only test here that swaps `PROMPTS`: the real files render, which is
    the point of `test_prompts.py`, so the failure has to be built. What is
    under test is that the role does not catch it — the agent is never called.
    """
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "decompose.md").write_text("Break the spec into $unfilled tickets.")
    monkeypatch.setattr(agents, "PROMPTS", Prompts(broken))
    runner = FakeAgentRunner({"decompose": "ok"})
    ctx = context(repo, runner)

    with pytest.raises(PromptError):
        await decompose(ctx, None)

    assert runner.specs == []
