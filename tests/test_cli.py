"""`agl run`, `agl clean`, and `agl init`: argument handling and
composition-root wiring.

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
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agl.cli import main
from agl.config import load_project
from agl.core.vcs.impl.git import Git
from agl.runtime import paths
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

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert created[0].deps.config.name == PROJECT
    assert created[0].deps.config.repo == repo.resolve()


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
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert created[0].label == "add-auth"


def test_the_short_name_flag_sets_the_label(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "-n", "add-auth", "Add auth"])

    assert code == 0
    assert created[0].label == "add-auth"


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


# -- description and max-concurrent ---------------------------------------


def test_a_multi_word_description_arrives_intact(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add", "a", "login", "page"])

    assert code == 0
    assert created[0].description == "Add a login page"


def test_a_double_dash_lets_the_description_start_with_a_dash(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "--", "--fix", "things"])

    assert code == 0
    assert created[0].description == "--fix things"


def test_max_concurrent_defaults_to_three(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert created[0].max_concurrent == 3


def test_max_concurrent_reaches_run(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)

    code = main(["run", "tickets", "--name", "add-auth", "--max-concurrent", "7", "Add auth"])

    assert code == 0
    assert created[0].max_concurrent == 7


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
    created = stub_run(monkeypatch)
    stdin = _FakeStdin("Add auth\nwith OAuth\n", isatty=False)
    monkeypatch.setattr(sys, "stdin", stdin)

    code = main(["run", "tickets", "--name", "add-auth"])

    assert code == 0
    assert created[0].description == "Add auth\nwith OAuth\n"


def test_a_positional_description_wins_and_stdin_is_untouched(
    monkeypatch: pytest.MonkeyPatch, repo: Path, home: Path, trees: Path
) -> None:
    setup(monkeypatch, repo, home, trees)
    created = stub_run(monkeypatch)
    stdin = _FakeStdin("should not be read", isatty=False)
    monkeypatch.setattr(sys, "stdin", stdin)

    code = main(["run", "tickets", "--name", "add-auth", "Add auth"])

    assert code == 0
    assert created[0].description == "Add auth"
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


# -- help ------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv", [["--help"], ["run", "--help"], ["clean", "--help"], ["init", "--help"]]
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
