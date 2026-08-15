"""The worktree lifecycle: checkout, keep-alive across review rounds, teardown.

Real git in `tmp_path`, the same as `vcs`'s own tests — a fake here would hide
exactly the worktree behaviour this module exists to get right.
"""

from pathlib import Path

import pytest

from agl.config import ProjectConfig
from agl.core.vcs.impl.git import Git
from agl.runtime import paths
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.worktrees import Worktrees
from tests.conftest import git
from tests.integration.conftest import PROJECT

LABEL = "add-auth"


def config(repo: Path, trees: Path) -> ProjectConfig:
    config_dir = repo.parent / "agl-config"
    return ProjectConfig(
        name=PROJECT,
        repo=repo,
        trees_root=trees,
        build=(),
        build_timeout=30.0,
        standards=config_dir / "standards.md",
        config_dir=config_dir,
    )


def start(repo: Path) -> None:
    git(repo, "checkout", "-b", "feature", "main")


def ticket(id_: str, parent: str | None = None) -> Ticket:
    return Ticket(id=id_, title=id_, status=Status.PENDING, deliverables=("x.py",), parent=parent)


@pytest.fixture
def worktrees(tmp_path: Path, repo: Path) -> Worktrees:
    start(repo)
    return Worktrees(Git(repo), config(repo, tmp_path / "trees"), LABEL, "feature")


def test_acquire_checks_out_a_fresh_worktree_on_the_run_s_base_branch(
    worktrees: Worktrees, repo: Path
) -> None:
    t = ticket("T-01")

    w = worktrees.acquire(t)

    assert w.ticket is t
    assert w.branch == paths.branch(LABEL, "T-01")
    assert w.tree == paths.worktree_dir(repo.parent / "trees", PROJECT, LABEL, "T-01").resolve()
    assert git(w.tree, "rev-parse", "--abbrev-ref", "HEAD").strip() == w.branch


def test_a_bug_s_worktree_is_cut_from_its_parent_s_branch(
    worktrees: Worktrees, repo: Path
) -> None:
    parent = ticket("T-01")
    worktrees.acquire(parent)
    bug = ticket("T-01-bug-1", parent="T-01")

    assert worktrees.base_for(bug) == paths.branch(LABEL, "T-01")
    w = worktrees.acquire(bug)
    assert git(w.tree, "rev-parse", "--abbrev-ref", "HEAD").strip() == w.branch


def test_base_for_a_ticket_with_no_parent_is_the_run_s_base_branch(worktrees: Worktrees) -> None:
    assert worktrees.base_for(ticket("T-01")) == "feature"


def test_keep_makes_the_same_worktree_come_back_from_a_second_acquire(
    worktrees: Worktrees,
) -> None:
    t = ticket("T-01")
    first = worktrees.acquire(t)

    worktrees.keep(first)
    second = worktrees.acquire(t)

    assert second.tree == first.tree
    assert second.branch == first.branch


def test_release_removes_the_worktree_from_disk(worktrees: Worktrees, repo: Path) -> None:
    w = worktrees.acquire(ticket("T-01"))
    assert w.tree.exists()

    worktrees.release(w)

    assert not w.tree.exists()


def test_tree_of_returns_a_kept_worktree_s_path(worktrees: Worktrees) -> None:
    w = worktrees.acquire(ticket("T-01"))
    worktrees.keep(w)

    assert worktrees.tree_of("T-01") == w.tree
