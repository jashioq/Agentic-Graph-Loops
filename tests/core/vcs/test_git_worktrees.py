"""Worktrees: adding, listing, removing, pruning, and doing it concurrently."""

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agl.core.vcs import (
    BranchExistsError,
    DirtyWorktreeError,
    UnknownRefError,
    VcsError,
    Worktree,
)
from agl.core.vcs.impl.git import Git
from tests.conftest import commit_file, git


@pytest.fixture
def vcs(repo: Path) -> Git:
    return Git(repo)


@pytest.fixture
def trees(tmp_path: Path) -> Path:
    """Worktrees live outside the repository, as `paths` lays them out."""
    root = tmp_path / "trees"
    root.mkdir()
    return root


# -- adding ---------------------------------------------------------------


def test_add_worktree_creates_the_directory(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "agl/add-auth/T-03", "main")
    assert tree.is_dir()
    assert (tree / "README.md").read_text(encoding="utf-8") == "# repo\n"


def test_add_worktree_returns_the_worktree(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    worktree = vcs.add_worktree(tree, "agl/add-auth/T-03", "main")
    assert worktree == Worktree(path=tree.resolve(), branch="agl/add-auth/T-03")


def test_add_worktree_creates_the_branch_at_the_base(repo: Path, vcs: Git, trees: Path) -> None:
    base = commit_file(repo, "a.txt", "a\n", "add a")
    vcs.add_worktree(trees / "T-03", "agl/add-auth/T-03", "main")
    assert vcs.rev_parse("agl/add-auth/T-03") == base


def test_the_worktree_has_the_new_branch_checked_out(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "agl/add-auth/T-03", "main")
    assert vcs.current_branch(cwd=tree) == "agl/add-auth/T-03"


def test_the_main_repo_stays_on_its_branch(vcs: Git, trees: Path) -> None:
    vcs.add_worktree(trees / "T-03", "agl/add-auth/T-03", "main")
    assert vcs.current_branch() == "main"


def test_add_worktree_creates_missing_parent_directories(vcs: Git, tmp_path: Path) -> None:
    tree = tmp_path / "deep" / "nested" / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    assert tree.is_dir()


def test_two_worktrees_for_two_branches_coexist(vcs: Git, trees: Path) -> None:
    first = vcs.add_worktree(trees / "T-01", "agl/x/T-01", "main")
    second = vcs.add_worktree(trees / "T-02", "agl/x/T-02", "main")
    assert vcs.current_branch(cwd=first.path) == "agl/x/T-01"
    assert vcs.current_branch(cwd=second.path) == "agl/x/T-02"


def test_work_in_one_worktree_does_not_touch_the_other(vcs: Git, trees: Path) -> None:
    first = vcs.add_worktree(trees / "T-01", "agl/x/T-01", "main")
    second = vcs.add_worktree(trees / "T-02", "agl/x/T-02", "main")
    (first.path / "a.txt").write_text("a\n", encoding="utf-8")
    assert vcs.is_dirty(cwd=first.path) is True
    assert vcs.is_dirty(cwd=second.path) is False


def test_add_worktree_for_an_existing_branch_raises(vcs: Git, trees: Path) -> None:
    vcs.create_branch("feat", "main")
    with pytest.raises(BranchExistsError):
        vcs.add_worktree(trees / "T-03", "feat", "main")


def test_a_refused_add_leaves_no_directory_behind(vcs: Git, trees: Path) -> None:
    vcs.create_branch("feat", "main")
    tree = trees / "T-03"
    with pytest.raises(BranchExistsError):
        vcs.add_worktree(tree, "feat", "main")
    assert not tree.exists()
    assert len(vcs.list_worktrees()) == 1


def test_add_worktree_off_an_unknown_base_raises(vcs: Git, trees: Path) -> None:
    with pytest.raises(VcsError):
        vcs.add_worktree(trees / "T-03", "feat", "no-such-base")


def test_add_worktree_onto_an_occupied_path_raises(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "agl/x/T-03", "main")
    with pytest.raises(VcsError):
        vcs.add_worktree(tree, "agl/x/other", "main")


# -- attaching ------------------------------------------------------------


def test_attach_worktree_checks_the_branch_out(vcs: Git, trees: Path) -> None:
    vcs.create_branch("feat", "main")
    tree = trees / "T-03"
    vcs.attach_worktree(tree, "feat")
    assert tree.is_dir()
    assert vcs.current_branch(cwd=tree) == "feat"


def test_attach_worktree_returns_the_worktree(vcs: Git, trees: Path) -> None:
    vcs.create_branch("feat", "main")
    tree = trees / "T-03"
    assert vcs.attach_worktree(tree, "feat") == Worktree(path=tree.resolve(), branch="feat")


def test_attach_worktree_registers_it(vcs: Git, trees: Path) -> None:
    vcs.create_branch("feat", "main")
    tree = trees / "T-03"
    vcs.attach_worktree(tree, "feat")
    assert Worktree(tree.resolve(), "feat") in vcs.list_worktrees()


def test_an_attached_tree_holds_the_branch_content(repo: Path, vcs: Git, trees: Path) -> None:
    # The point of attaching: the work that branch already holds comes back.
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "work on feat")
    git(repo, "checkout", "-q", "main")
    tree = trees / "T-03"
    vcs.attach_worktree(tree, "feat")
    assert (tree / "a.txt").read_text(encoding="utf-8") == "a\n"


def test_attach_worktree_creates_missing_parent_directories(vcs: Git, tmp_path: Path) -> None:
    vcs.create_branch("feat", "main")
    tree = tmp_path / "deep" / "nested" / "T-03"
    vcs.attach_worktree(tree, "feat")
    assert tree.is_dir()


def test_attaching_a_branch_that_does_not_exist_raises(vcs: Git, trees: Path) -> None:
    with pytest.raises(UnknownRefError):
        vcs.attach_worktree(trees / "T-03", "no-such-branch")


def test_a_refused_attach_leaves_no_directory_behind(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    with pytest.raises(UnknownRefError):
        vcs.attach_worktree(tree, "no-such-branch")
    assert not tree.exists()
    assert len(vcs.list_worktrees()) == 1


def test_attaching_a_branch_checked_out_elsewhere_raises(vcs: Git, trees: Path) -> None:
    vcs.add_worktree(trees / "T-03", "feat", "main")
    with pytest.raises(VcsError):
        vcs.attach_worktree(trees / "T-04", "feat")


def test_attaching_the_branch_the_main_repo_holds_raises(vcs: Git, trees: Path) -> None:
    with pytest.raises(VcsError):
        vcs.attach_worktree(trees / "T-03", "main")


def test_attach_worktree_onto_an_occupied_path_raises(vcs: Git, trees: Path) -> None:
    vcs.create_branch("feat", "main")
    vcs.create_branch("other", "main")
    tree = trees / "T-03"
    vcs.attach_worktree(tree, "feat")
    with pytest.raises(VcsError):
        vcs.attach_worktree(tree, "other")


def test_a_removed_worktree_can_be_attached_again(vcs: Git, trees: Path) -> None:
    # What resume does: the branch outlives the tree, so the tree comes back.
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    sha = commit_file(tree, "a.txt", "a\n", "work")
    vcs.remove_worktree(tree)
    vcs.attach_worktree(tree, "feat")
    assert vcs.rev_parse("feat") == sha
    assert (tree / "a.txt").read_text(encoding="utf-8") == "a\n"


# -- listing --------------------------------------------------------------


def test_list_worktrees_on_a_fresh_repo_is_the_main_one(repo: Path, vcs: Git) -> None:
    assert vcs.list_worktrees() == (Worktree(path=repo.resolve(), branch="main"),)


def test_list_worktrees_includes_the_main_one_and_every_added_one(
    repo: Path, vcs: Git, trees: Path
) -> None:
    vcs.add_worktree(trees / "T-01", "agl/x/T-01", "main")
    vcs.add_worktree(trees / "T-02", "agl/x/T-02", "main")
    assert set(vcs.list_worktrees()) == {
        Worktree(repo.resolve(), "main"),
        Worktree((trees / "T-01").resolve(), "agl/x/T-01"),
        Worktree((trees / "T-02").resolve(), "agl/x/T-02"),
    }


def test_the_main_worktree_comes_first(repo: Path, vcs: Git, trees: Path) -> None:
    vcs.add_worktree(trees / "T-01", "agl/x/T-01", "main")
    assert vcs.list_worktrees()[0].path == repo.resolve()


def test_list_worktrees_handles_a_path_containing_a_space(vcs: Git, trees: Path) -> None:
    tree = trees / "my tree"
    vcs.add_worktree(tree, "feat", "main")
    assert Worktree(tree.resolve(), "feat") in vcs.list_worktrees()


# -- removing -------------------------------------------------------------


def test_remove_worktree_deletes_the_directory(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    vcs.remove_worktree(tree)
    assert not tree.exists()


def test_remove_worktree_deregisters_it(repo: Path, vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    vcs.remove_worktree(tree)
    assert vcs.list_worktrees() == (Worktree(repo.resolve(), "main"),)


def test_remove_worktree_leaves_the_branch_alone(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    vcs.remove_worktree(tree)
    assert vcs.branch_exists("feat")


def test_removing_a_dirty_worktree_raises(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    (tree / "README.md").write_text("edited\n", encoding="utf-8")
    with pytest.raises(DirtyWorktreeError):
        vcs.remove_worktree(tree)
    assert tree.is_dir()


def test_removing_a_worktree_with_untracked_files_raises(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    (tree / "scratch.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(DirtyWorktreeError):
        vcs.remove_worktree(tree)


def test_removing_a_dirty_worktree_with_force_succeeds(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    (tree / "README.md").write_text("edited\n", encoding="utf-8")
    (tree / "scratch.txt").write_text("x\n", encoding="utf-8")
    vcs.remove_worktree(tree, force=True)
    assert not tree.exists()


def test_removing_a_worktree_that_is_not_registered_raises(vcs: Git, trees: Path) -> None:
    with pytest.raises(VcsError):
        vcs.remove_worktree(trees / "never-added")


def test_committed_work_in_a_worktree_survives_its_removal(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    sha = commit_file(tree, "a.txt", "a\n", "work")
    vcs.remove_worktree(tree)
    assert vcs.rev_parse("feat") == sha


# -- pruning --------------------------------------------------------------


def test_prune_clears_the_registry_after_a_directory_is_deleted_by_hand(
    repo: Path, vcs: Git, trees: Path
) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    shutil.rmtree(tree)
    assert len(vcs.list_worktrees()) == 2  # still registered, just broken
    vcs.prune_worktrees()
    assert vcs.list_worktrees() == (Worktree(repo.resolve(), "main"),)


def test_prune_keeps_live_worktrees(vcs: Git, trees: Path) -> None:
    kept = vcs.add_worktree(trees / "T-01", "agl/x/T-01", "main")
    gone = vcs.add_worktree(trees / "T-02", "agl/x/T-02", "main")
    shutil.rmtree(gone.path)
    vcs.prune_worktrees()
    assert kept in vcs.list_worktrees()
    assert gone not in vcs.list_worktrees()


def test_prune_on_a_repo_with_nothing_to_prune_is_a_no_op(vcs: Git, trees: Path) -> None:
    vcs.add_worktree(trees / "T-01", "agl/x/T-01", "main")
    before = vcs.list_worktrees()
    vcs.prune_worktrees()
    assert vcs.list_worktrees() == before


def test_a_pruned_path_can_be_added_again(vcs: Git, trees: Path) -> None:
    tree = trees / "T-03"
    vcs.add_worktree(tree, "feat", "main")
    shutil.rmtree(tree)
    vcs.prune_worktrees()
    vcs.add_worktree(tree, "feat-2", "main")
    assert vcs.current_branch(cwd=tree) == "feat-2"


# -- concurrency ----------------------------------------------------------


def test_worktrees_added_from_many_threads_all_register(vcs: Git, trees: Path) -> None:
    # Git's own index.lock does not cover worktree registration; this is what
    # the lock inside the implementation is for.
    names = [f"T-{index:02d}" for index in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        added = list(pool.map(lambda n: vcs.add_worktree(trees / n, f"agl/x/{n}", "main"), names))

    registered = vcs.list_worktrees()
    assert len(registered) == len(names) + 1
    for worktree in added:
        assert worktree in registered
        assert worktree.path.is_dir()
    assert {w.branch for w in registered} == {"main"} | {f"agl/x/{n}" for n in names}


def test_concurrent_adds_and_removes_leave_a_consistent_registry(vcs: Git, trees: Path) -> None:
    names = [f"T-{index:02d}" for index in range(6)]
    for name in names:
        vcs.add_worktree(trees / name, f"agl/x/{name}", "main")

    doomed, kept = names[:3], names[3:]
    fresh = [f"N-{index:02d}" for index in range(3)]

    def work(name: str) -> None:
        if name in doomed:
            vcs.remove_worktree(trees / name)
        else:
            vcs.add_worktree(trees / name, f"agl/x/{name}", "main")

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(work, doomed + fresh))

    paths = {worktree.path for worktree in vcs.list_worktrees()}
    assert paths == {vcs.root()} | {(trees / name).resolve() for name in kept + fresh}


def test_branches_created_from_many_threads_all_exist(vcs: Git) -> None:
    names = [f"agl/x/T-{index:02d}" for index in range(16)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda name: vcs.create_branch(name, "main"), names))
    assert vcs.branches("agl/") == tuple(sorted(names))
