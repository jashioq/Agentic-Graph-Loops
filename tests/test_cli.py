"""`agl run` and `agl clean`: argument handling and composition-root wiring.

A full run is already covered by `test_workflow.py`, so `run` here never
drives a real ticket workflow to completion. Wiring tests replace the ticket
workflow's `Run` with a stub that records what it was constructed with and
returns immediately; only the preflight-refusal test lets the real `Run`
start, because a preflight refusal happens before any agent or terminal call.

Real git throughout — `repo` and `git` come from `tests/conftest.py` — and a
real `config.toml`. Nothing here ever constructs `FakeAgentRunner` or
`HeadlessTerminal` because nothing here ever reaches an agent call: the run
either fails before one (preflight) or is stubbed out before one (wiring).
"""

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agl.cli import main
from agl.core import paths
from agl.core.vcs.impl.git import Git
from agl.workflows.tickets import workflow as tickets_workflow
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


class StubRun:
    """Stands in for the ticket workflow's `Run`: records its args, does nothing.

    `go()` returns immediately without touching `deps.agent` or `deps.terminal`,
    which is what keeps these tests from ever reaching a real model call.
    """

    def __init__(
        self, deps: Any, label: str, description: str, max_concurrent: int
    ) -> None:
        self.deps = deps
        self.label = label
        self.description = description
        self.max_concurrent = max_concurrent
        self.state = SimpleNamespace(halt=None)


def stub_run(monkeypatch: pytest.MonkeyPatch) -> list[StubRun]:
    """Patch the real `Run` with `StubRun`; returns the list it appends to."""
    created: list[StubRun] = []

    class _Recording(StubRun):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

        async def go(self) -> None:
            return None

    monkeypatch.setattr(tickets_workflow, "Run", _Recording)
    return created


# -- config resolution --------------------------------------------------------


def test_resolves_the_config_from_a_repo_inside_a_known_project(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "Add auth"])

    assert code == 0
    assert created[0].deps.config.name == PROJECT
    assert created[0].deps.config.repo == repo.resolve()


def test_fails_clearly_outside_a_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, home: Path
) -> None:
    monkeypatch.setenv("AGL_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    code = main(["run", "tickets", "Add auth"])

    assert code != 0


def test_a_missing_agl_home_exits_nonzero_with_a_message(
    monkeypatch: pytest.MonkeyPatch, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("AGL_HOME", raising=False)
    monkeypatch.chdir(repo)

    code = main(["run", "tickets", "Add auth"])

    assert code != 0
    assert "AGL_HOME" in capsys.readouterr().err
    assert "Traceback" not in capsys.readouterr().err


# -- label ---------------------------------------------------------------


def test_label_defaults_to_the_current_branch_name(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "Add auth"])

    assert code == 0
    assert created[0].label == "main"


def test_name_overrides_the_default_label(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert created[0].label == "add-auth"


def test_an_invalid_label_is_rejected_before_anything_is_created(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    git(repo, "checkout", "-b", "Feature-X", "main")
    setup(monkeypatch, repo, home, trees)
    stub_run(monkeypatch)

    code = main(["run", "tickets", "Add auth"])

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

    code = main(["run", "no-such-workflow", "Add auth"])

    assert code != 0
    assert "tickets" in capsys.readouterr().err


# -- description and max-concurrent ---------------------------------------


def test_a_multi_word_description_arrives_intact(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "Add", "a", "login", "page"])

    assert code == 0
    assert created[0].description == "Add a login page"


def test_a_double_dash_lets_the_description_start_with_a_dash(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "--", "--fix", "things"])

    assert code == 0
    assert created[0].description == "--fix things"


def test_max_concurrent_defaults_to_three(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "Add auth"])

    assert code == 0
    assert created[0].max_concurrent == 3


def test_max_concurrent_reaches_run(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "--max-concurrent", "7", "Add auth"])

    assert code == 0
    assert created[0].max_concurrent == 7


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

    code = main(["run", "tickets", "Add auth"])

    assert code != 0
    err = capsys.readouterr().err
    assert "uncommitted" in err
    assert "Traceback" not in err


# -- help ------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["--help"], ["run", "--help"], ["clean", "--help"]])
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
