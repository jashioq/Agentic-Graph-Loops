"""The worktree pool: checkout, keep-alive across rounds, teardown.

Real git in `tmp_path`, the same as `vcs`'s own tests — a fake here would hide
exactly the worktree behaviour this module exists to get right.

Nothing here knows what a key means. A key is a node id, the branch and the
base are whatever the caller asked for, and "a bug branches off its parent" is
expressed the only way runtime can express it: by passing the parent's branch
as the base.

`reopen` and `adopt` are the resume half, and they are tested by building a
pool, walking away from it, and building a second one over the same repository
— which is what a second process is, minus the process.
"""

import shutil
from pathlib import Path

import pytest

from agl.core.vcs import VcsError
from agl.core.vcs.impl.git import Git
from agl.runtime import paths
from agl.runtime.worktrees import Work, Worktrees
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


def pool(tmp_path: Path, repo: Path, label: str = LABEL) -> Worktrees:
    """A second pool over the same repository — a later process, in one line."""
    return Worktrees(Git(repo), trees_root=tmp_path / "trees", project=PROJECT, label=label)


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


# -- taking over what a dead process left ----------------------------------


def test_reopen_finds_this_run_s_trees_and_takes_them_as_open(
    worktrees: Worktrees, tmp_path: Path, repo: Path
) -> None:
    first = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)
    second = worktrees.acquire("T-02", worktrees.branch_for("T-02"), BASE)

    reopened = pool(tmp_path, repo).reopen()

    assert reopened == (first, second)


def test_a_reopened_worktree_is_open_and_comes_back_from_acquire(
    worktrees: Worktrees, tmp_path: Path, repo: Path
) -> None:
    """Reopened is kept, not merely listed: the run goes on using the tree it
    finds instead of checking the branch out a second time."""
    w = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)

    later = pool(tmp_path, repo)
    later.reopen()

    assert later.tree_of("T-01") == w.tree
    assert later.acquire("T-01", w.branch, BASE) == w


def test_reopen_ignores_another_label_s_trees_and_the_main_worktree(
    worktrees: Worktrees, tmp_path: Path, repo: Path
) -> None:
    mine = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)
    other = pool(tmp_path, repo, label="other-run")
    other.acquire("T-01", other.branch_for("T-01"), BASE)

    assert pool(tmp_path, repo).reopen() == (mine,)


def test_reopen_does_not_resurrect_a_tree_whose_directory_is_gone(
    worktrees: Worktrees, tmp_path: Path, repo: Path
) -> None:
    """A directory deleted under git leaves a stale registry entry. `reopen`
    prunes first, so what comes back is trees that are actually there."""
    w = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)
    shutil.rmtree(w.tree)

    later = pool(tmp_path, repo)

    assert later.reopen() == ()
    assert not w.tree.exists()
    assert all(entry.path != w.tree for entry in Git(repo).list_worktrees())


def test_adopt_checks_an_existing_branch_out_again(
    worktrees: Worktrees, tmp_path: Path, repo: Path
) -> None:
    """The branch survived and its tree did not — what a run that was killed
    after `release` but before the ticket was done leaves behind."""
    w = worktrees.acquire("T-01", worktrees.branch_for("T-01"), BASE)
    worktrees.release(w)

    adopted = pool(tmp_path, repo).adopt("T-01", w.branch)

    assert adopted == Work(key="T-01", tree=w.tree, branch=w.branch)
    assert git(adopted.tree, "rev-parse", "--abbrev-ref", "HEAD").strip() == w.branch


def test_adopt_refuses_a_branch_that_is_not_there(worktrees: Worktrees) -> None:
    with pytest.raises(VcsError):
        worktrees.adopt("T-01", worktrees.branch_for("T-01"))
