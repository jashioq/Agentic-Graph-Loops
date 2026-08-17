"""What a run was given, the checks before it starts, and the build gate.

Real git in `tmp_path` throughout: `preflight` is entirely about what a
repository is in the middle of, and a fake would only ever agree with whatever
this module already believes. The build gate runs real subprocesses for the same
reason — a timeout that does not actually kill a child is the failure worth
catching, and `sys.executable` is a build command every machine has.

The context itself is data, so its tests are about exactly that: frozen, and
carrying the connectors it was handed rather than constructing any.
"""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agl.core.agent import Model
from agl.core.store.impl.file_store import FileStore
from agl.core.terminal import Row, Rows, Screen, Text
from agl.core.vcs.impl.git import Git
from agl.runtime import paths
from agl.runtime.agents import call
from agl.runtime.context import (
    PreflightError,
    ProjectSettings,
    RunContext,
    build_gate,
    preflight,
    resume_preflight,
)
from agl.runtime.display import Board, live
from tests.conftest import git, make_diverged
from tests.fakes import FakeAgentRunner, HeadlessTerminal, MemoryStore
from tests.runtime.conftest import LABEL, PROJECT, context, feature, settings

# -- the context is data ----------------------------------------------------


def test_a_context_carries_the_connectors_it_was_handed(repo: Path, tmp_path: Path) -> None:
    agent, terminal = FakeAgentRunner(), HeadlessTerminal()
    feature(repo)

    ctx = context(repo, agent=agent, terminal=terminal, home=tmp_path / "elsewhere")

    assert ctx.agent is agent
    assert ctx.terminal is terminal
    assert ctx.vcs.root() == repo.resolve()
    assert isinstance(ctx.store, FileStore)
    assert ctx.store.root == paths.run_dir(tmp_path / "elsewhere", LABEL).resolve()


def test_a_context_cannot_be_mutated(ctx: RunContext) -> None:
    """No lifecycle and no behaviour: a run is handed this and changes nothing."""
    with pytest.raises(FrozenInstanceError):
        ctx.label = "other"  # type: ignore[misc]


def test_project_settings_cannot_be_mutated(repo: Path) -> None:
    with pytest.raises(FrozenInstanceError):
        settings(repo).build = ()  # type: ignore[misc]


def test_a_context_records_this_run_s_request(repo: Path) -> None:
    feature(repo, "auth")

    ctx = context(repo, request="Add auth", max_concurrent=3)

    assert ctx.request == "Add auth"
    assert ctx.base_branch == "auth"
    assert ctx.max_concurrent == 3


def test_a_context_names_the_workflow_it_belongs_to(repo: Path) -> None:
    """The one field a run cannot work out for itself: which workflow it is a
    run of, which is what a resume needs before it has anything else."""
    feature(repo)

    assert context(repo).workflow == "tickets"
    assert context(repo, workflow="review").workflow == "review"


def test_project_settings_hold_what_runtime_needs_of_a_project(repo: Path) -> None:
    project = ProjectSettings(
        name=PROJECT,
        repo=repo,
        trees_root=repo.parent / "trees",
        build=("just", "build"),
        build_timeout=12.0,
    )

    assert project.name == PROJECT
    assert project.repo == repo
    assert project.trees_root == repo.parent / "trees"
    assert project.build == ("just", "build")
    assert project.build_timeout == 12.0


# -- preflight ----------------------------------------------------------------


def test_preflight_passes_on_a_clean_feature_branch(repo: Path) -> None:
    feature(repo)

    preflight(Git(repo), MemoryStore(), LABEL)


def test_preflight_refuses_a_dirty_repo(repo: Path) -> None:
    feature(repo)
    (repo / "dirty.txt").write_text("oops\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="uncommitted"):
        preflight(Git(repo), MemoryStore(), LABEL)


@pytest.mark.parametrize("trunk", ["main", "master"])
def test_preflight_refuses_the_trunk(repo: Path, trunk: str) -> None:
    if trunk != "main":
        git(repo, "branch", "-m", trunk)

    with pytest.raises(PreflightError, match=trunk):
        preflight(Git(repo), MemoryStore(), LABEL)


def test_preflight_refuses_a_label_whose_branches_are_still_around(repo: Path) -> None:
    feature(repo)
    git(repo, "branch", paths.branch(LABEL, "T-01"))

    with pytest.raises(PreflightError, match="already in use"):
        preflight(Git(repo), MemoryStore(), LABEL)


def test_preflight_refuses_a_label_whose_run_state_is_still_around(repo: Path) -> None:
    feature(repo)
    store = MemoryStore()
    store.write("spec.md", "already started\n")

    with pytest.raises(PreflightError, match="already in use"):
        preflight(Git(repo), store, LABEL)


def test_preflight_allows_a_label_that_matches_the_current_branch(repo: Path) -> None:
    """A person may reasonably pass `--name` matching the branch they are
    standing on, so the label names a real ref: their own branch. That must not
    read as "already in use" — only leftover `agl/<label>/*` branches or run
    state should.
    """
    feature(repo, name=LABEL)

    preflight(Git(repo), MemoryStore(), LABEL)


def test_preflight_refuses_a_label_that_could_never_name_a_branch(repo: Path) -> None:
    feature(repo)

    with pytest.raises(paths.InvalidNameError):
        preflight(Git(repo), MemoryStore(), "Add Auth")


def test_preflight_offers_both_ways_out_of_a_label_already_in_use(repo: Path) -> None:
    """A run left behind is two different situations — one to pick up and one to
    throw away — so the refusal names the command for each."""
    feature(repo)
    git(repo, "branch", paths.branch(LABEL, "T-01"))

    with pytest.raises(PreflightError) as refusal:
        preflight(Git(repo), MemoryStore(), LABEL)

    assert f"agl resume {LABEL}" in str(refusal.value)
    assert f"agl clean {LABEL}" in str(refusal.value)


# -- resume_preflight ---------------------------------------------------------


def test_resume_preflight_passes_on_the_branch_the_run_was_started_from(repo: Path) -> None:
    branch = feature(repo)

    resume_preflight(Git(repo), branch)


def test_resume_preflight_allows_the_leftovers_a_start_refuses(repo: Path) -> None:
    """Branches and run state under the label are what there is to resume, so
    none of `preflight`'s other checks are made here."""
    branch = feature(repo)
    git(repo, "branch", paths.branch(LABEL, "T-01"))

    resume_preflight(Git(repo), branch)


def test_resume_preflight_allows_a_merge_this_run_left_in_progress(repo: Path) -> None:
    """The one kind of mess a resume knows how to settle: a conflicted merge
    into the base branch, sitting in the repository exactly as the queue left
    it."""
    diverged = make_diverged(repo, "app.py", "base\n", "ours\n", "theirs\n")
    vcs = Git(repo)
    assert not vcs.merge(repo, diverged.theirs).clean

    resume_preflight(vcs, diverged.ours)


def test_resume_preflight_refuses_another_branch(repo: Path) -> None:
    feature(repo, "other")

    with pytest.raises(PreflightError, match="feature"):
        resume_preflight(Git(repo), "feature")


def test_resume_preflight_refuses_a_tree_dirty_for_any_other_reason(repo: Path) -> None:
    branch = feature(repo)
    (repo / "dirty.txt").write_text("oops\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="uncommitted"):
        resume_preflight(Git(repo), branch)


# -- the build gate -----------------------------------------------------------


async def test_the_gate_runs_the_project_s_build_and_reports_it_passing(repo: Path) -> None:
    gate = build_gate(settings(repo))

    result = await gate()

    assert result.ok
    assert not result.timed_out


async def test_a_failing_build_comes_back_as_a_result_not_an_exception(repo: Path) -> None:
    """A failed build is the answer to the question the gate asked, not an error."""
    gate = build_gate(settings(repo, build=(sys.executable, "-c", "raise SystemExit(3)")))

    result = await gate()

    assert not result.ok
    assert result.code == 3


async def test_the_build_runs_in_the_project_s_repository(repo: Path) -> None:
    gate = build_gate(settings(repo, build=(sys.executable, "-c", "import os; print(os.getcwd())")))

    result = await gate()

    assert Path(result.stdout.strip()).resolve() == repo.resolve()


async def test_a_build_that_hangs_is_killed_and_reported_as_timed_out(repo: Path) -> None:
    gate = build_gate(
        settings(
            repo,
            build=(sys.executable, "-c", "import time; time.sleep(30)"),
            build_timeout=0.2,
        )
    )

    result = await gate()

    assert result.timed_out
    assert not result.ok


async def test_the_gate_can_be_called_again_for_the_next_merge(repo: Path) -> None:
    """One gate serves a whole run: the queue holds it and calls it per request."""
    gate = build_gate(settings(repo))

    assert (await gate()).ok
    assert (await gate()).ok


# -- the harness --------------------------------------------------------------
#
# `context()` is the deliverable here: a workflow's whole input is a
# `RunContext`, so a test that builds one is a test that can drive a run. These
# two say it produces a context a run would actually get through — the checks
# pass, the gate is open, and the fakes are reached by the runtime calls a
# workflow makes rather than only by assertions about the context itself.


async def test_the_harness_produces_a_context_a_run_could_start_from(ctx: RunContext) -> None:
    preflight(ctx.vcs, ctx.store, ctx.label)

    assert ctx.base_branch == "feature"
    assert (await build_gate(ctx.project)()).ok
    assert ctx.store.list() == ()


async def test_the_harness_s_fakes_are_reached_by_the_calls_a_workflow_makes(
    repo: Path,
) -> None:
    feature(repo)
    agent = FakeAgentRunner({"interview": "the specification"})
    terminal = HeadlessTerminal()
    ctx = context(repo, agent=agent, terminal=terminal)
    board = Board(started_at=0.0)

    async with live(ctx.terminal, board) as display:
        display.show(lambda: Screen(Rows(Row(Text(ctx.label)))))
        result = await call(
            ctx.agent,
            role="interview",
            prompt="what should this do?",
            cwd=ctx.project.repo,
            model=Model.SONNET,
            on_activity=display.activity(ctx.label),
        )

    assert result.text == "the specification"
    assert agent.specs[0].model is Model.SONNET
    assert any(LABEL in frame for frame in terminal.frames)
