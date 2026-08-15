"""Prompt rendering and the one call that assembles an `AgentSpec`.

`FakeAgentRunner` throughout — nothing here calls a model. What is under test
is the spec `call` builds and what `Prompts` does to a template, so most
assertions read `runner.specs`.

Prompts are written into `tmp_path` rather than read from any workflow's
`prompts/` directory: this module has never heard of a role, and a test that
borrowed real prompt files would be testing their content instead.
"""

from collections.abc import Awaitable
from pathlib import Path

import pytest

from agl.core.agent import NO_PARAMS, AgentOption, AgentQuestion, AgentResult, Model, Tool
from agl.runtime.agents import Limits, PromptError, Prompts, call
from tests.fakes import FakeAgentRunner, ScriptedRun


def prompts_dir(tmp_path: Path, **files: str) -> Path:
    directory = tmp_path / "prompts"
    directory.mkdir()
    for name, text in files.items():
        (directory / f"{name}.md").write_text(text)
    return directory


def a_tool(name: str = "save_spec") -> Tool:
    async def handler(arguments: dict[str, object]) -> str:
        return "saved"

    return Tool(name=name, description="save it", schema=NO_PARAMS, handler=handler)


# -- Prompts ------------------------------------------------------------------


def test_render_reads_the_named_file_from_the_directory(tmp_path: Path) -> None:
    prompts = Prompts(prompts_dir(tmp_path, decompose="Break the spec into tickets."))

    assert prompts.render("decompose") == "Break the spec into tickets."


def test_render_substitutes_every_placeholder(tmp_path: Path) -> None:
    prompts = Prompts(prompts_dir(tmp_path, triage="Findings:\n$findings\nFor:\n$deliverables"))

    rendered = prompts.render("triage", findings="Q-1", deliverables="A token store")

    assert rendered == "Findings:\nQ-1\nFor:\nA token store"
    assert "$" not in rendered


def test_a_placeholder_with_nothing_to_fill_it_raises(tmp_path: Path) -> None:
    prompts = Prompts(prompts_dir(tmp_path, interview="Talk about: $user_input"))

    with pytest.raises(PromptError):
        prompts.render("interview")


def test_the_error_names_the_file_and_the_placeholder(tmp_path: Path) -> None:
    """A missing substitution is a bug in the caller, so it says which one."""
    directory = prompts_dir(tmp_path, interview="Talk about: $user_input")
    prompts = Prompts(directory)

    with pytest.raises(PromptError) as raised:
        prompts.render("interview")

    message = str(raised.value)
    assert str(directory / "interview.md") in message
    assert "user_input" in message


def test_a_substitution_the_template_does_not_use_is_ignored(tmp_path: Path) -> None:
    prompts = Prompts(prompts_dir(tmp_path, decompose="No placeholders here."))

    assert prompts.render("decompose", unused="x") == "No placeholders here."


def test_a_missing_prompt_file_raises_rather_than_rendering_empty(tmp_path: Path) -> None:
    prompts = Prompts(prompts_dir(tmp_path))

    with pytest.raises(FileNotFoundError):
        prompts.render("nowhere")


# -- call ----------------------------------------------------------------------


async def test_call_runs_the_spec_it_was_asked_for(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"implement": "done"})
    tool = a_tool()

    result = await call(
        runner,
        role="implement",
        prompt="Build it.",
        cwd=tmp_path,
        tools=(tool,),
        model=Model.SONNET,
        limits=Limits(),
    )

    assert isinstance(result, AgentResult)
    assert result.text == "done"
    spec = runner.specs[0]
    assert spec.role == "implement"
    assert spec.prompt == "Build it."
    assert spec.cwd == tmp_path
    assert spec.tools == (tool,)


async def test_the_model_the_caller_named_reaches_the_spec(tmp_path: Path) -> None:
    """The model is the caller's per-call choice, not something `Limits` decides."""
    runner = FakeAgentRunner({"implement": "done"})

    await call(
        runner,
        role="implement",
        prompt="Build it.",
        cwd=tmp_path,
        model=Model.OPUS,
        limits=Limits(),
    )

    assert runner.specs[0].model is Model.OPUS


async def test_two_calls_can_name_different_models(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"implement": "done", "decompose": "done"})

    await call(
        runner,
        role="decompose",
        prompt="Split it.",
        cwd=tmp_path,
        model=Model.OPUS,
        limits=Limits(),
    )
    await call(
        runner,
        role="implement",
        prompt="Build it.",
        cwd=tmp_path,
        model=Model.SONNET,
        limits=Limits(),
    )

    assert [spec.model for spec in runner.specs] == [Model.OPUS, Model.SONNET]


async def test_limits_reach_the_spec(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"implement": "done"})

    await call(
        runner,
        role="implement",
        prompt="Build it.",
        cwd=tmp_path,
        model=Model.SONNET,
        limits=Limits(max_turns=12, max_budget_usd=3.5),
    )

    spec = runner.specs[0]
    assert spec.max_turns == 12
    assert spec.max_budget_usd == 3.5


async def test_a_call_with_no_scoping_asked_for_gets_none(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"decompose": "done"})

    await call(
        runner,
        role="decompose",
        prompt="Split it.",
        cwd=tmp_path,
        model=Model.SONNET,
        limits=Limits(),
    )

    spec = runner.specs[0]
    assert spec.tools == ()
    assert spec.disallowed_tools == ()
    assert spec.permission_mode == "default"


async def test_denials_and_the_permission_mode_reach_the_spec(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"interview": "done"})

    await call(
        runner,
        role="interview",
        prompt="Ask.",
        cwd=tmp_path,
        disallowed=("Bash(git commit:*)",),
        permission_mode="plan",
        model=Model.SONNET,
        limits=Limits(),
    )

    spec = runner.specs[0]
    assert spec.disallowed_tools == ("Bash(git commit:*)",)
    assert spec.permission_mode == "plan"


async def test_activity_reaches_the_caller_s_callback(tmp_path: Path) -> None:
    runner = FakeAgentRunner({"implement": ScriptedRun(activity=("editing api.py",))})
    seen: list[str] = []

    await call(
        runner,
        role="implement",
        prompt="Build it.",
        cwd=tmp_path,
        model=Model.SONNET,
        limits=Limits(),
        on_activity=seen.append,
    )

    assert seen == ["editing api.py"]


async def test_a_question_reaches_the_ask_the_caller_supplied(tmp_path: Path) -> None:
    question = AgentQuestion(title="Which store?", options=(AgentOption("sqlite", "a file"),))
    runner = FakeAgentRunner({"implement": ScriptedRun(question=question)})
    asked: list[AgentQuestion] = []

    def ask(q: AgentQuestion) -> Awaitable[str]:
        asked.append(q)

        async def answer() -> str:
            return "sqlite"

        return answer()

    await call(
        runner,
        role="implement",
        prompt="Build it.",
        cwd=tmp_path,
        model=Model.SONNET,
        limits=Limits(),
        ask=ask,
    )

    assert asked == [question]
    assert runner.answers == ["sqlite"]
