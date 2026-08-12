"""Real git repositories in `tmp_path`, isolated from the developer's config.

Faking git would hide exactly the behaviour `vcs` exists to get right, so its
tests run against real repositories. That only works if the repositories are
deterministic: a global `init.defaultBranch`, a `commit.gpgsign`, or a
`merge.conflictStyle` picked up from the machine running the suite would make
tests pass here and fail there.

`_isolated_git_config` is autouse and points `GIT_CONFIG_GLOBAL` and
`GIT_CONFIG_SYSTEM` at a path that does not exist, so every git process started
by a test — including the ones started by the implementation under test — sees
no configuration but the repository's own. Everything that matters is then set
locally, in the template every `repo` is copied from.

The repo is built once per session and copied per test rather than initialised
each time. Six process spawns per test is most of the suite's runtime, and a
copy of a two-file repository is not: `.git` in a plain repository refers to
nothing outside itself, so copying it is a faithful clone of the fixture.
"""

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

__all__ = ["Diverged", "commit_file", "git", "make_diverged"]


@pytest.fixture(scope="session", autouse=True)
def _isolated_git_config(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Cut every git process in the suite off from global and system config."""
    absent = tmp_path_factory.mktemp("git-isolation") / "no-such-gitconfig"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GIT_CONFIG_GLOBAL", str(absent))
        patch.setenv("GIT_CONFIG_SYSTEM", str(absent))
        yield absent


@pytest.fixture(scope="session")
def _template_repo(
    tmp_path_factory: pytest.TempPathFactory, _isolated_git_config: Path
) -> Path:
    """The repository `repo` hands out a fresh copy of."""
    root = tmp_path_factory.mktemp("template") / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "AGL Test")
    git(root, "config", "user.email", "test@agl.invalid")
    git(root, "config", "commit.gpgsign", "false")
    # The parser handles diff3 too; pinning it keeps the default-style tests
    # honest about which style they are asserting on.
    git(root, "config", "merge.conflictStyle", "merge")
    commit_file(root, "README.md", "# repo\n", "initial commit")
    return root


@pytest.fixture
def repo(tmp_path: Path, _template_repo: Path) -> Path:
    """A git repo on `main` with one commit, at `tmp_path/repo`."""
    root = tmp_path / "repo"
    shutil.copytree(_template_repo, root, symlinks=True)
    return root


# -- helpers --------------------------------------------------------------


def git(cwd: Path, *args: str) -> str:
    """Run a git command in `cwd` and return its stdout, raising on failure.

    Deliberately not `agl.core._exec.run`: a fixture that shares the code under
    test can only ever agree with it.
    """
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def commit_file(repo: Path, relpath: str, content: str, message: str) -> str:
    """Write `content` to `relpath`, commit it, and return the new sha."""
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    git(repo, "add", "--", relpath)
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


@dataclass(frozen=True)
class Diverged:
    """Two branches off a common commit, both having edited the same file."""

    base: str
    ours: str
    theirs: str
    ours_sha: str
    theirs_sha: str


def make_diverged(
    repo: Path,
    path: str,
    base_content: str,
    ours: str,
    theirs: str,
    ours_branch: str = "ours",
    theirs_branch: str = "theirs",
) -> Diverged:
    """Commit `base_content`, then branch twice and rewrite the file each way.

    Leaves the repo checked out on `ours_branch`. The starting branch stays at
    the base commit, so it is also a merge target that has not moved.
    """
    start = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    base = commit_file(repo, path, base_content, f"add {path}")

    git(repo, "checkout", "-b", theirs_branch, base)
    theirs_sha = commit_file(repo, path, theirs, f"{theirs_branch}: rewrite {path}")

    git(repo, "checkout", start)
    git(repo, "checkout", "-b", ours_branch, base)
    ours_sha = commit_file(repo, path, ours, f"{ours_branch}: rewrite {path}")

    return Diverged(
        base=base,
        ours=ours_branch,
        theirs=theirs_branch,
        ours_sha=ours_sha,
        theirs_sha=theirs_sha,
    )
