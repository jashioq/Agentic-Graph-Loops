"""Setup every integration scenario needs: a repo, a home, and a trees root.

The `repo` fixture and the git helpers come from `tests/conftest.py` — these
tests use real git, real files, and a real `Dag`, and fake only the agent and
the terminal. Everything here is layout: where a run's documents live and where
its worktrees go, both handed out as bare directories so each test can name them
through `paths` itself rather than being given pre-baked names.
"""

import shutil
from pathlib import Path

import pytest

from agl.core.vcs.impl.git import Git

__all__ = ["PROJECT", "copy_repo"]

PROJECT = "demo"


def copy_repo(root: Path, template: Path) -> Path:
    """A fresh repository at `root/repo`, copied from the session template.

    For a scenario that drives its whole flow once for the module and then
    asserts over what it left: `repo` is function-scoped, and this is the same
    copy it makes.
    """
    repo = root / "repo"
    shutil.copytree(template, repo, symlinks=True)
    return repo


@pytest.fixture
def vcs(repo: Path) -> Git:
    """`Vcs` over the real repository the test will branch and merge in."""
    return Git(repo)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """The orchestrator's home — `paths` lays out runs and projects under it."""
    return tmp_path / "home"


@pytest.fixture
def trees(tmp_path: Path) -> Path:
    """The trees root, outside both the repository and the home."""
    return tmp_path / "trees"
