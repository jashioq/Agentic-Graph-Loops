"""`agl run`, `agl resume`, `agl clean`, and `agl init`: argument handling and
composition-root wiring.

A full run is already covered by `test_workflow.py`, so `run` here never
drives a real ticket workflow to completion. Wiring tests replace the ticket
workflow's `run` — or its `resume` — with a stub that records the `RunContext`
it was handed and returns immediately; only the refusal tests let the real
entry point start, because every refusal they check happens before any agent or
terminal call.

The `RunContext` those stubs capture is the whole contract between the cli and a
workflow, so asserting over it is how this file checks the composition root: the
label, the description, `--max-concurrent`, and the `ProjectSettings` that
`config.toml` was translated into.

Real git throughout — `repo` and `git` come from `tests/conftest.py` — and a
real `config.toml`. Nothing here ever constructs `FakeAgentRunner` or
`HeadlessTerminal` because nothing here ever reaches an agent call: the run
either fails before one (preflight) or is stubbed out before one (wiring).
"""

import shutil
import sys
from pathlib import Path

import pytest

from agl.cli import main
from agl.config import load_project
from agl.core.store.impl.file_store import FileStore
from agl.core.vcs.impl.git import Git
from agl.runtime import paths
from agl.runtime.context import RunContext
from agl.runtime.record import RunRecord, StateFile, write_record
from agl.workflows.tickets import workflow as tickets_workflow
from agl.workflows.tickets.documents.state_document import StateDocument
from agl.workflows.tickets.errors import Halt, HaltedError
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import Run
from tests.conftest import git

PROJECT = "demo"


# -- setup helpers ------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def trees(tmp_path: Path) -> Path:
    return tmp_path / "trees"


def write_config(home: Path, repo: Path, trees: Path, name: str = PROJECT) -> Path:
    config_path = home / "projects" / name / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f'name = "{name}"\n'
        f'repo = "{repo}"\n'
        f'trees_root = "{trees}"\n'
        f'build = ["true"]\n'
        f"build_timeout = 30\n",
        encoding="utf-8",
    )
    return config_path


def setup(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path, name: str = PROJECT
) -> None:
    """Point `AGL_HOME` at `home`, `cwd` at `repo`, and write a matching config."""
    write_config(home, repo, trees, name)
    monkeypatch.setenv("AGL_HOME", str(home))
    monkeypatch.chdir(repo)


def stub_run(monkeypatch: pytest.MonkeyPatch) -> list[RunContext]:
    """Replace the ticket workflow's `run` with one that only records its context.

    It returns before touching `ctx.agent` or `ctx.terminal`, which is what
    keeps these tests from ever reaching a real model call.
    """
    contexts: list[RunContext] = []

    async def recording(ctx: RunContext) -> None:
        contexts.append(ctx)

    monkeypatch.setattr(tickets_workflow, "run", recording)
    return contexts


def failing_run(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """Replace the ticket workflow's `run` with one that raises `error`."""

    async def raising(ctx: RunContext) -> None:
        raise error

    monkeypatch.setattr(tickets_workflow, "run", raising)


def stub_resume(monkeypatch: pytest.MonkeyPatch) -> list[RunContext]:
    """Replace the ticket workflow's `resume` with one that only records its context."""
    contexts: list[RunContext] = []

    async def recording(ctx: RunContext) -> None:
        contexts.append(ctx)

    monkeypatch.setattr(tickets_workflow, "resume", recording)
    return contexts


# -- config resolution --------------------------------------------------------


def test_resolves_the_config_from_a_repo_inside_a_known_project(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert contexts[0].project.name == PROJECT
    assert contexts[0].project.repo == repo.resolve()


def test_the_config_file_arrives_as_project_settings(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    """Every field of `config.toml` a run may use, and nothing of the file format.

    This is the one translation in the codebase, so it is the one place the two
    shapes can drift apart without anything else noticing.
    """
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    settings = contexts[0].project
    config = load_project(home, repo)
    assert settings.name == config.name
    assert settings.repo == config.repo
    assert settings.trees_root == config.trees_root
    assert settings.build == config.build
    assert settings.build_timeout == config.build_timeout


def test_the_context_carries_the_connectors_and_the_current_branch(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    """What the composition root is for: real connectors, and the base to merge into."""
    git(repo, "checkout", "-b", "feature", "main")
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    ctx = contexts[0]
    assert ctx.base_branch == "feature"
    assert ctx.vcs.root() == repo.resolve()
    assert paths.run_dir(home, "add-auth").is_dir()


def test_fails_clearly_outside_a_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, home: Path
) -> None:
    monkeypatch.setenv("AGL_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code != 0


def test_a_missing_agl_home_exits_nonzero_with_a_message(
    monkeypatch: pytest.MonkeyPatch, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("AGL_HOME", raising=False)
    monkeypatch.chdir(repo)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code != 0
    assert "AGL_HOME" in capsys.readouterr().err
    assert "Traceback" not in capsys.readouterr().err


# -- label ---------------------------------------------------------------


def test_name_is_required(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    stub_run(monkeypatch)

    code = main(["run", "tickets", "Add auth"])

    assert code != 0
    assert "--name" in capsys.readouterr().err


def test_name_sets_the_label(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert contexts[0].label == "add-auth"


def test_the_short_name_flag_sets_the_label(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)

    code = main(["run", "tickets", "-n", "add-auth", "Add auth"])

    assert code == 0
    assert contexts[0].label == "add-auth"


def test_an_invalid_label_is_rejected_before_anything_is_created(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "Bad-Label", "Add auth"])

    assert code != 0
    assert not (home / "runs").exists()
    vcs = Git(repo)
    assert vcs.branches("agl/") == ()


# -- workflow discovery --------------------------------------------------


def test_an_unknown_workflow_lists_the_available_ones(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)

    code = main(["run", "no-such-workflow", "--name", "add-auth", "Add auth"])

    assert code != 0
    assert "tickets" in capsys.readouterr().err


def test_a_workflow_is_a_package_exposing_only_run(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    """`run(ctx)` is the whole entry point a workflow has to provide.

    Stripping every other name off the module and still getting a run is what
    says so — the cli constructs nothing of the workflow's and reaches into
    nothing of it afterwards.
    """
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)
    for name in ("Deps", "Run", "Loop", "Job"):
        monkeypatch.delattr(tickets_workflow, name, raising=False)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert len(contexts) == 1


# -- description and max-concurrent ---------------------------------------


def test_a_multi_word_description_arrives_intact(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add", "a", "login", "page"])

    assert code == 0
    assert contexts[0].request == "Add a login page"


def test_a_double_dash_lets_the_description_start_with_a_dash(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "--", "--fix", "things"])

    assert code == 0
    assert contexts[0].request == "--fix things"


def test_max_concurrent_defaults_to_three(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert contexts[0].max_concurrent == 3


def test_max_concurrent_reaches_run(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "--max-concurrent", "7", "Add auth"])

    assert code == 0
    assert contexts[0].max_concurrent == 7


# -- description from stdin ------------------------------------------------


class _FakeStdin:
    """Stands in for `sys.stdin`: a fixed `isatty()`, and a `read()` that
    records whether it was ever called."""

    def __init__(self, text: str, isatty: bool) -> None:
        self._text = text
        self._isatty = isatty
        self.read_called = False

    def isatty(self) -> bool:
        return self._isatty

    def read(self) -> str:
        self.read_called = True
        return self._text


def test_piped_stdin_description_arrives_intact_including_newlines(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)
    stdin = _FakeStdin("Add auth\nwith OAuth\n", isatty=False)
    monkeypatch.setattr(sys, "stdin", stdin)

    code = main(["run", "tickets", "--name", "add-auth"])

    assert code == 0
    assert contexts[0].request == "Add auth\nwith OAuth\n"


def test_a_positional_description_wins_and_stdin_is_untouched(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    contexts = stub_run(monkeypatch)
    stdin = _FakeStdin("should not be read", isatty=False)
    monkeypatch.setattr(sys, "stdin", stdin)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert contexts[0].request == "Add auth"
    assert stdin.read_called is False


def test_no_description_with_a_terminal_stdin_exits_nonzero_with_a_message(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    stub_run(monkeypatch)
    stdin = _FakeStdin("", isatty=True)
    monkeypatch.setattr(sys, "stdin", stdin)

    code = main(["run", "tickets", "--name", "add-auth"])

    assert code != 0
    assert capsys.readouterr().err.strip()
    assert stdin.read_called is False


def test_an_empty_positional_description_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "   "])

    assert code != 0


def test_a_whitespace_only_piped_stdin_description_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    stub_run(monkeypatch)
    stdin = _FakeStdin("   \n  ", isatty=False)
    monkeypatch.setattr(sys, "stdin", stdin)

    code = main(["run", "tickets", "--name", "add-auth"])

    assert code != 0


# -- preflight -----------------------------------------------------------


def test_a_preflight_refusal_exits_nonzero_with_the_reason_printed(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    git(repo, "checkout", "-b", "feature", "main")
    (repo / "dirty.txt").write_text("oops\n", encoding="utf-8")
    setup(monkeypatch, repo, home, trees)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code != 0
    err = capsys.readouterr().err
    assert "uncommitted" in err
    assert "Traceback" not in err


# -- exit status -----------------------------------------------------------
#
# The exception `run` raises *is* the exit status. Nothing here reaches into a
# workflow to ask how it went, so a workflow that returns is a run that
# succeeded and any exception out of it is a run that failed, with its message
# on stderr and no traceback.


def test_a_workflow_that_returns_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert capsys.readouterr().err == ""


def test_a_run_that_ends_halted_exits_nonzero_with_the_halts_reason(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    halt = Halt("T-01 hit a conflict", "auth.py", resumable=False)
    failing_run(monkeypatch, HaltedError(halt))

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code != 0
    err = capsys.readouterr().err
    assert halt.reason in err
    assert "Traceback" not in err


def test_any_other_failure_out_of_a_workflow_exits_nonzero_with_its_message(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    failing_run(monkeypatch, RuntimeError("the agent went missing"))

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code != 0
    err = capsys.readouterr().err
    assert "the agent went missing" in err
    assert "Traceback" not in err


def test_an_interrupted_run_names_the_worktrees_it_left(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    label = "add-auth"
    _, worktree = _ticket_branch_and_worktree(repo, home, trees, label)
    failing_run(monkeypatch, KeyboardInterrupt())

    code = main(["run", "tickets", "--name", label, "Add auth"])

    assert code != 0
    err = capsys.readouterr().err
    assert str(worktree) in err
    assert f"agl resume {label}" in err
    assert f"agl clean {label}" in err


# -- resume ------------------------------------------------------------------
#
# `resume` takes a label and nothing else: everything the original invocation
# settled is read back out of the run's record, so these tests are about that
# record reaching the workflow intact — and about the four ways a label can name
# something that cannot be picked up.

LABEL = "add-auth"


def write_record_for(
    home: Path,
    label: str = LABEL,
    *,
    workflow: str = "tickets",
    request: str = "Add auth",
    base_branch: str = "feature",
    project: str = PROJECT,
    max_concurrent: int = 3,
) -> RunRecord:
    """The record a started run left behind, where `agl resume` looks for it."""
    record = RunRecord(
        workflow=workflow,
        label=label,
        request=request,
        base_branch=base_branch,
        project=project,
        max_concurrent=max_concurrent,
    )
    write_record(FileStore(paths.run_dir(home, label)), record)
    return record


def write_state(home: Path, label: str = LABEL) -> None:
    """A state document with one unfinished ticket, so the run has work left."""
    StateDocument(StateFile(FileStore(paths.run_dir(home, label)))).write(
        Run(
            tickets=(
                Ticket(
                    id="T-01",
                    title="Add auth",
                    status=Status.PENDING,
                    deliverables=("auth.py",),
                ),
            )
        )
    )


def test_resume_hands_the_workflow_the_context_the_record_describes(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    """The record is the authority, not the repository `resume` was typed in.

    The checkout here is deliberately not the branch the run was started from
    and the command line carries neither a request nor a concurrency, so every
    field asserted below can only have come out of `run.json`.
    """
    git(repo, "checkout", "-b", "somewhere-else", "main")
    setup(monkeypatch, repo, home, trees)
    write_record_for(home, base_branch="feature", request="Add auth", max_concurrent=5)
    contexts = stub_resume(monkeypatch)

    code = main(["resume", LABEL])

    assert code == 0
    ctx = contexts[0]
    assert ctx.workflow == "tickets"
    assert ctx.label == LABEL
    assert ctx.base_branch == "feature"
    assert ctx.request == "Add auth"
    assert ctx.max_concurrent == 5
    assert ctx.project.name == PROJECT
    assert ctx.vcs.root() == repo.resolve()
    assert ctx.store.exists("run.json")


def test_resume_max_concurrent_overrides_the_recorded_one(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    write_record_for(home, max_concurrent=3)
    contexts = stub_resume(monkeypatch)

    code = main(["resume", LABEL, "--max-concurrent", "7"])

    assert code == 0
    assert contexts[0].max_concurrent == 7


def test_resume_on_an_unknown_label_lists_what_exists_in_runs(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    (home / "runs" / "existing-run").mkdir(parents=True)

    code = main(["resume", "never-started"])

    assert code != 0
    assert "existing-run" in capsys.readouterr().err
    assert not (home / "runs" / "never-started").exists()


def test_resume_refuses_a_run_belonging_to_another_project(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same label, different project: the branches and trees are not this repo's."""
    setup(monkeypatch, repo, home, trees)
    write_record_for(home, project="other-project")
    stub_resume(monkeypatch)

    code = main(["resume", LABEL])

    assert code != 0
    err = capsys.readouterr().err
    assert "other-project" in err
    assert PROJECT in err


def test_resume_refuses_a_workflow_that_cannot_be_resumed(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`resume` is optional: a workflow that does not expose one is told apart
    from one that does by nothing but `getattr`."""
    setup(monkeypatch, repo, home, trees)
    write_record_for(home)
    monkeypatch.delattr(tickets_workflow, "resume")

    code = main(["resume", LABEL])

    assert code != 0
    err = capsys.readouterr().err
    assert "tickets" in err
    assert "Traceback" not in err


def test_resume_reports_an_unknown_recorded_workflow(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    write_record_for(home, workflow="no-such-workflow")

    code = main(["resume", LABEL])

    assert code != 0
    err = capsys.readouterr().err
    assert "no-such-workflow" in err
    assert "tickets" in err
    assert "Traceback" not in err


def test_resume_reports_a_state_with_nothing_to_continue(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A record but no specification: the real `resume` refuses, and its message
    is the exit status — nothing here reaches an agent."""
    setup(monkeypatch, repo, home, trees)
    write_record_for(home)

    code = main(["resume", LABEL])

    assert code != 0
    err = capsys.readouterr().err
    assert "start a new run" in err
    assert "Traceback" not in err


def test_resume_from_the_wrong_branch_names_the_one_to_check_out(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    write_record_for(home, base_branch="feature")
    write_state(home)

    code = main(["resume", LABEL])

    assert code != 0
    err = capsys.readouterr().err
    assert "feature" in err
    assert "main" in err
    assert "Traceback" not in err


def test_resume_rejects_an_invalid_label(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)

    code = main(["resume", "Bad-Label"])

    assert code != 0
    assert not (home / "runs").exists()


# -- help ------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["run", "--help"],
        ["resume", "--help"],
        ["clean", "--help"],
        ["init", "--help"],
    ],
)
def test_help_exits_cleanly(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    code = main(argv)

    assert code == 0
    assert capsys.readouterr().out


# -- clean -----------------------------------------------------------------


def _ticket_branch_and_worktree(
    repo: Path, home: Path, trees: Path, label: str, node_id: str = "T-01"
) -> tuple[str, Path]:
    vcs = Git(repo)
    branch = paths.branch(label, node_id)
    worktree = paths.worktree_dir(trees, PROJECT, label, node_id)
    vcs.add_worktree(worktree, branch, "main")
    return branch, worktree


def test_clean_with_no_label_refuses_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)

    code = main(["clean"])

    assert code != 0
    assert not home.exists() or not (home / "runs").exists()


def test_clean_removes_worktrees_branches_and_run_directory(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    label = "add-auth"
    branch, worktree = _ticket_branch_and_worktree(repo, home, trees, label)
    run_dir = home / "runs" / label
    run_dir.mkdir(parents=True)
    (run_dir / "spec.md").write_text("# spec\n", encoding="utf-8")

    code = main(["clean", label])

    assert code == 0
    vcs = Git(repo)
    assert branch not in vcs.branches()
    assert worktree not in [w.path for w in vcs.list_worktrees()]
    assert not worktree.exists()
    assert not run_dir.exists()


def test_clean_leaves_the_base_branch_and_branches_outside_the_namespace_alone(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    label = "add-auth"
    vcs = Git(repo)
    vcs.create_branch("feature-other", "main")
    _ticket_branch_and_worktree(repo, home, trees, label)

    code = main(["clean", label])

    assert code == 0
    assert vcs.branch_exists("feature-other")
    assert vcs.branch_exists("main")


def test_clean_on_an_unknown_label_lists_what_exists_in_runs(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(monkeypatch, repo, home, trees)
    (home / "runs" / "existing-run").mkdir(parents=True)

    code = main(["clean", "never-started"])

    assert code != 0
    assert "existing-run" in capsys.readouterr().err


def test_clean_succeeds_when_a_worktree_directory_was_deleted_by_hand(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    label = "add-auth"
    branch, worktree = _ticket_branch_and_worktree(repo, home, trees, label)

    shutil.rmtree(worktree)

    code = main(["clean", label])

    assert code == 0
    vcs = Git(repo)
    assert branch not in vcs.branches()
    assert worktree not in [w.path for w in vcs.list_worktrees()]


def test_clean_succeeds_when_the_run_directory_does_not_exist(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    label = "add-auth"
    branch, worktree = _ticket_branch_and_worktree(repo, home, trees, label)

    code = main(["clean", label])

    assert code == 0
    vcs = Git(repo)
    assert branch not in vcs.branches()
    assert not worktree.exists()
    assert not (home / "runs" / label).exists()


# -- init --------------------------------------------------------------------


def _chdir_repo(monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path) -> None:
    monkeypatch.setenv("AGL_HOME", str(home))
    monkeypatch.chdir(repo)


def test_init_creates_config_standards_and_runs_dir(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path
) -> None:
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init"])

    assert code == 0
    project_dir = home / "projects" / "repo"
    assert (project_dir / "config.toml").is_file()
    assert (project_dir / "standards.md").is_file()
    assert (home / "runs").is_dir()


def test_init_config_round_trips_through_load_project(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path
) -> None:
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init"])

    assert code == 0
    config = load_project(home, repo)
    assert config.name == "repo"
    assert config.repo == repo.resolve()
    assert config.trees_root.is_absolute()
    assert config.build
    assert config.build_timeout > 0


def test_init_standards_has_headings_and_no_content(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path
) -> None:
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init"])

    assert code == 0
    text = (home / "projects" / "repo" / "standards.md").read_text(encoding="utf-8")
    assert "Architecture" in text
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    assert lines == []


def test_init_name_defaults_to_the_repo_directory_name(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path
) -> None:
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init"])

    assert code == 0
    assert (home / "projects" / "repo").is_dir()


def test_init_name_is_overridden_by_the_flag(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path
) -> None:
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init", "--name", "myproject"])

    assert code == 0
    assert (home / "projects" / "myproject").is_dir()
    assert not (home / "projects" / "repo").exists()
    assert load_project(home, repo).name == "myproject"


def test_init_trees_root_is_absolute_and_a_sibling_of_the_repo(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path
) -> None:
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init"])

    assert code == 0
    config = load_project(home, repo)
    assert config.trees_root.is_absolute()
    assert config.trees_root.parent == repo.resolve().parent
    assert config.trees_root != repo.resolve()


@pytest.mark.parametrize(
    ("marker", "expected_build"),
    [
        ("gradlew", ("./gradlew", "compileDebugKotlin")),
        ("Cargo.toml", ("cargo", "check")),
        ("package.json", ("npm", "run", "build")),
    ],
)
def test_init_guesses_the_build_command_from_a_marker_file(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    marker: str,
    expected_build: tuple[str, ...],
) -> None:
    (repo / marker).write_text("", encoding="utf-8")
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init"])

    assert code == 0
    config = load_project(home, repo)
    assert config.build == expected_build


def test_init_falls_back_to_a_placeholder_build_when_nothing_is_recognised(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path
) -> None:
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init"])

    assert code == 0
    config = load_project(home, repo)
    assert config.build not in (
        ("./gradlew", "compileDebugKotlin"),
        ("cargo", "check"),
        ("npm", "run", "build"),
    )
    assert "config.toml" in " ".join(config.build)


def test_init_refuses_when_the_project_directory_already_exists(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path
) -> None:
    project_dir = home / "projects" / "repo"
    project_dir.mkdir(parents=True)
    marker = project_dir / "keep.txt"
    marker.write_text("mine\n", encoding="utf-8")
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init"])

    assert code != 0
    assert marker.read_text(encoding="utf-8") == "mine\n"
    assert not (project_dir / "config.toml").exists()


def test_init_refuses_a_duplicate_repo_naming_the_existing_config(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    home: Path,
    trees: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    existing = write_config(home, repo, trees, name="other")
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init"])

    assert code != 0
    assert str(existing) in capsys.readouterr().err
    assert not (home / "projects" / "repo").exists()


def test_init_refuses_outside_a_git_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, home: Path
) -> None:
    monkeypatch.setenv("AGL_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    code = main(["init"])

    assert code != 0
    assert not (home / "projects").exists()


def test_init_refuses_an_invalid_name(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path
) -> None:
    _chdir_repo(monkeypatch, repo, home)

    code = main(["init", "--name", "Bad Name"])

    assert code != 0
    assert not (home / "projects").exists()
