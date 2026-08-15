"""The worktree pool: checkout, keep-alive across rounds, teardown.

Real git in `tmp_path`, the same as `vcs`'s own tests — a fake here would hide
exactly the worktree behaviour this module exists to get right.

Nothing here knows what a key means. A key is a node id, the branch and the
base are whatever the caller asked for, and "a bug branches off its parent" is
expressed the only way runtime can express it: by passing the parent's branch
as the base.
"""

from pathlib import Path

import pytest

from agl.core.vcs.impl.git import Git
from agl.runtime import paths
from agl.runtime.worktrees import Worktrees
from tests.conftest import git
from tests.integration.conftest import PROJECT

LABEL = "add-auth"
BASE = "feature"


@pytest.fixture
def worktrees(tmp_path: Path, repo: Path) -> Worktrees:
    git(repo, "checkout", "-b", BASE, "main")
    return Worktrees(
        Git(repo), trees_root=tmp_path / "trees", project=PROJECT, label=LABEL
    )


def test_branch_for_is_the_run_s_namespace_over_the_key(worktrees: Worktrees) -> None:
    assert worktrees.branch_for("T-01") == paths.branch(LABEL, "T-01")


def test_acquire_checks_out_a_fresh_worktree_on_the_base_it_was_given(
    worktrees: Worktrees, repo: Path
) -> None:
    branch = worktrees.branch_for("T-01")

    w = worktrees.acquire("T-01", branch, BASE)

    assert w.key == "T-01"
    assert w.branch == branch
    assert w.tree == paths.worktree_dir(repo.parent / "trees", PROJECT, LABEL, "T-01").resolve()
    assert git(w.tree, "rev-parse", "--abbrev-ref", "HEAD").strip() == branch


def test_a_worktree_cut_from_another_key_s_branch_starts_there(
    worktrees: Worktrees, repo: Path
) -> None:
    parent = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)
    worktrees.keep(parent)

    branch = worktrees.branch_for("T-01-bug-1")
    w = worktrees.acquire("T-01-bug-1", branch, parent.branch)

    assert git(w.tree, "rev-parse", "--abbrev-ref", "HEAD").strip() == branch
    assert (
        git(repo, "rev-parse", branch).strip() == git(repo, "rev-parse", parent.branch).strip()
    )


def test_keep_makes_the_same_worktree_come_back_from_a_second_acquire(
    worktrees: Worktrees,
) -> None:
    first = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)

    worktrees.keep(first)
    second = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)

    assert second.tree == first.tree
    assert second.branch == first.branch


def test_a_kept_worktree_is_handed_out_once_and_then_gone(worktrees: Worktrees) -> None:
    # `acquire` pops: a kept tree comes back exactly once, and the pool is
    # empty again afterwards, so `tree_of` has nothing to answer with.
    w = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)
    worktrees.keep(w)
    worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)

    with pytest.raises(KeyError):
        worktrees.tree_of("T-01")


def test_release_removes_the_worktree_from_disk(worktrees: Worktrees) -> None:
    w = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)
    assert w.tree.exists()

    worktrees.release(w)

    assert not w.tree.exists()


def test_tree_of_returns_a_kept_worktree_s_path(worktrees: Worktrees) -> None:
    w = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)
    worktrees.keep(w)

    assert worktrees.tree_of("T-01") == w.tree
