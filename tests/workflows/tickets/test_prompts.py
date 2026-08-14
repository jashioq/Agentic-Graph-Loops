"""The real prompt files under `prompts/`: they load, substitute cleanly, and
carry the constraints Step 0 depends on.

`FakeAgentRunner` throughout — no test here calls a real model. Every role is
run through `agents.py` with `ctx.prompts` pointed at the real package
directory, so what is under test is the actual files a run would read.
"""

from pathlib import Path
from string import Template

import pytest

import agl.workflows.tickets as tickets_pkg
from agl.workflows.tickets.agents import (
    AgentContext,
    Limits,
    decompose,
    implement,
    interview,
    review,
    triage,
)
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.reviews import Finding, Severity
from tests.fakes import FakeAgentRunner, MemoryStore, ScriptedRun

PROMPTS_DIR = Path(tickets_pkg.__file__).parent / "prompts"

_ROLE_FILES = (
    "interview",
    "decompose",
    "implement",
    "implement_bug",
    "review_quality",
    "review_spec",
    "triage",
)


def context(runner: FakeAgentRunner, tmp_path: Path) -> AgentContext:
    return AgentContext(
        runner=runner,
        store=MemoryStore(),
        repo=tmp_path / "repo",
        prompts=PROMPTS_DIR,
        limits=Limits(),
    )


def feature_ticket(**overrides: object) -> Ticket:
    fields: dict[str, object] = {
        "id": "T-03",
        "title": "Add auth",
        "status": Status.IN_PROGRESS,
        "deliverables": ("TokenStore in data/auth/",),
    }
    fields.update(overrides)
    return Ticket(**fields)  # type: ignore[arg-type]


def bug_ticket(**overrides: object) -> Ticket:
    fields: dict[str, object] = {
        "id": "T-03-bug-1",
        "title": "Fix null check",
        "status": Status.IN_PROGRESS,
        "deliverables": ("Guard against a None token",),
        "parent": "T-03",
    }
    fields.update(overrides)
    return Ticket(**fields)  # type: ignore[arg-type]


def a_finding(**overrides: object) -> Finding:
    fields: dict[str, object] = {
        "id": "Q-1",
        "severity": Severity.HIGH,
        "title": "Missing null check",
        "detail": "auth() does not check for a None token.",
        "files": ("src/auth.py",),
    }
    fields.update(overrides)
    return Finding(**fields)  # type: ignore[arg-type]


def findings_result() -> ScriptedRun:
    """A run that reports through `save_findings`, the way a real one now must."""
    return ScriptedRun(text="done", calls=(("save_findings", {"findings": []}),))


def groups_result(*groups: dict[str, object]) -> ScriptedRun:
    """A run that reports through `save_triage`, the way a real one now must."""
    return ScriptedRun(text="done", calls=(("save_triage", {"groups": list(groups)}),))


# -- every file exists and has no stray placeholders after substitution -----


@pytest.mark.parametrize("name", _ROLE_FILES)
def test_prompt_file_exists(name: str) -> None:
    assert (PROMPTS_DIR / f"{name}.md").is_file()


@pytest.mark.parametrize("name", ("decompose", "implement", "implement_bug"))
def test_prompts_with_no_placeholders_render_as_is(name: str) -> None:
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    rendered = Template(text).substitute()
    assert rendered == text
    assert "$" not in rendered


async def test_decompose_prompt_has_no_placeholders(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"decompose": "ok"})
    ctx = context(runner, tmp_path)

    await decompose(ctx, None)

    assert "$" not in runner.specs[0].prompt


# -- interview: $user_input --------------------------------------------------


async def test_interview_prompt_substitutes_user_input(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"interview": "ok"})
    ctx = context(runner, tmp_path)

    await interview(ctx, "Add password login", None)

    prompt = runner.specs[0].prompt
    assert "Add password login" in prompt
    assert "$" not in prompt


def test_interview_prompt_missing_substitution_raises() -> None:
    text = (PROMPTS_DIR / "interview.md").read_text()
    with pytest.raises(KeyError):
        Template(text).substitute()


def test_interview_prompt_has_the_fixed_sections_in_order() -> None:
    text = (PROMPTS_DIR / "interview.md").read_text()
    sections = (
        "Goal",
        "Out of scope",
        "Architecture",
        "Dependencies",
        "Constraints",
        "Decisions",
        "Verification",
    )
    positions = [text.index(f"## {section}") for section in sections]
    assert positions == sorted(positions)


# -- review: $base_branch, three-dot diff ------------------------------------


async def test_review_prompts_substitute_base_branch(tmp_path: Path) -> None:
    runner = FakeAgentRunner(
        {"review-quality": findings_result(), "review-spec": findings_result()}
    )
    ctx = context(runner, tmp_path)
    tree = tmp_path / "tree"

    await review(ctx, feature_ticket(review_round=1), tree, "main", None)

    by_role = {spec.role: spec for spec in runner.specs}
    assert "$base_branch" not in by_role["review-quality"].prompt
    assert "$base_branch" not in by_role["review-spec"].prompt
    assert "$" not in by_role["review-quality"].prompt
    assert "$" not in by_role["review-spec"].prompt


async def test_review_prompts_use_the_three_dot_diff_form(tmp_path: Path) -> None:
    runner = FakeAgentRunner(
        {"review-quality": findings_result(), "review-spec": findings_result()}
    )
    ctx = context(runner, tmp_path)
    tree = tmp_path / "tree"

    await review(ctx, feature_ticket(review_round=1), tree, "release/9", None)

    by_role = {spec.role: spec for spec in runner.specs}
    for prompt in (by_role["review-quality"].prompt, by_role["review-spec"].prompt):
        assert "git diff release/9...HEAD" in prompt
        assert "git diff release/9..HEAD" not in prompt


def test_review_prompt_files_use_three_dots_not_two() -> None:
    for name in ("review_quality", "review_spec"):
        text = (PROMPTS_DIR / f"{name}.md").read_text()
        assert "diff $base_branch...HEAD" in text
        assert "diff $base_branch..HEAD" not in text


def test_review_prompts_missing_base_branch_raises() -> None:
    for name in ("review_quality", "review_spec"):
        text = (PROMPTS_DIR / f"{name}.md").read_text()
        with pytest.raises(KeyError):
            Template(text).substitute()


# -- review + triage: severity rubric identical wording ----------------------


def test_severity_rubric_is_worded_identically_in_both_reviewers() -> None:
    quality = (PROMPTS_DIR / "review_quality.md").read_text()
    spec = (PROMPTS_DIR / "review_spec.md").read_text()
    rubric_lines = (
        "- **HIGH** — the deliverable is not met, or the change introduces a",
        "- **MEDIUM** — a real improvement that is not required.",
        "- **LOW** — taste and preference.",
    )
    for line in rubric_lines:
        assert line in quality
        assert line in spec


def test_reviewer_prompts_carry_the_calibration_line() -> None:
    for name in ("review_quality", "review_spec"):
        text = (PROMPTS_DIR / f"{name}.md").read_text()
        assert "you have misunderstood the rubric" in text


def test_reviewer_prompts_require_a_fix_in_every_detail() -> None:
    for name in ("review_quality", "review_spec"):
        text = (PROMPTS_DIR / f"{name}.md").read_text()
        assert "must say both what is wrong and what would satisfy it" in text
        assert "leaks Retrofit types upward" in text  # the worked example


# -- every role reports through its tool, never its final message ------------


def test_reviewer_prompts_name_save_findings_and_the_empty_case() -> None:
    for name in ("review_quality", "review_spec"):
        text = (PROMPTS_DIR / f"{name}.md").read_text()
        assert "save_findings" in text
        assert "empty list" in text


def test_triage_prompt_names_save_triage() -> None:
    text = (PROMPTS_DIR / "triage.md").read_text()
    assert "save_triage" in text


# -- implement: bug vs feature use different files ---------------------------


async def test_implement_uses_a_different_prompt_for_a_bug_ticket(tmp_path: Path) -> None:
    runner = FakeAgentRunner(["ok", "ok"])
    ctx = context(runner, tmp_path)
    tree = tmp_path / "tree"

    await implement(ctx, feature_ticket(), tree, None)
    await implement(ctx, bug_ticket(), tree, None)

    feature_prompt, bug_prompt = runner.specs[0].prompt, runner.specs[1].prompt
    assert feature_prompt != bug_prompt
    assert "$" not in feature_prompt
    assert "$" not in bug_prompt


# -- triage: $findings, $deliverables ----------------------------------------


async def test_triage_prompt_substitutes_findings_and_deliverables(tmp_path: Path) -> None:
    runner = FakeAgentRunner(
        {
            "triage": groups_result(
                {"title": "Fix nulls", "deliverables": ["Guard tokens"], "findings": ["Q-1", "Q-2"]}
            )
        }
    )
    ctx = context(runner, tmp_path)
    findings = (a_finding(id="Q-1"), a_finding(id="Q-2", title="Leaks a Retrofit type"))

    await triage(ctx, feature_ticket(), findings, None)

    prompt = runner.specs[0].prompt
    assert "Q-1" in prompt
    assert "Missing null check" in prompt
    assert "Leaks a Retrofit type" in prompt
    assert "TokenStore in data/auth/" in prompt
    assert "$" not in prompt


def test_triage_prompt_missing_substitutions_raises() -> None:
    text = (PROMPTS_DIR / "triage.md").read_text()
    with pytest.raises(KeyError):
        Template(text).substitute()
    with pytest.raises(KeyError):
        Template(text).substitute(findings="x")
    with pytest.raises(KeyError):
        Template(text).substitute(deliverables="y")


def test_triage_prompt_never_asks_it_to_look_at_code() -> None:
    text = (PROMPTS_DIR / "triage.md").read_text()
    for word in ("Read", "Grep", "Glob", "the diff", "git diff"):
        assert word not in text
