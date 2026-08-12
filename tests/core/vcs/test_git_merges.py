"""Merging: clean, conflicted, aborted, and resolved by hand."""

from pathlib import Path

import pytest

from agl.core.vcs import FileStatus, UnknownRefError, VcsError
from agl.core.vcs.impl.git import Git
from tests.conftest import Diverged, commit_file, git, make_diverged


@pytest.fixture
def vcs(repo: Path) -> Git:
    return Git(repo)


@pytest.fixture
def diverged(repo: Path) -> Diverged:
    """Two branches that will conflict on `a.txt`, checked out on `ours`."""
    return make_diverged(repo, "a.txt", "base\n", "ours\n", "theirs\n")


# -- clean merges ---------------------------------------------------------


def test_a_clean_merge_reports_clean(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "add a")
    git(repo, "checkout", "-q", "main")
    result = vcs.merge(repo, "feat")
    assert result.clean is True
    assert result.conflicts == ()


def test_a_clean_merge_returns_the_new_sha(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "add a")
    git(repo, "checkout", "-q", "main")
    result = vcs.merge(repo, "feat")
    assert result.sha == vcs.rev_parse("HEAD")


def test_nothing_is_in_progress_after_a_clean_merge(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "add a")
    git(repo, "checkout", "-q", "main")
    vcs.merge(repo, "feat")
    assert vcs.merge_in_progress(repo) is False
    assert vcs.conflicts(repo) == ()
    assert vcs.has_changes(repo) is False


def test_a_clean_merge_brings_the_work_across(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "from the branch\n", "add a")
    git(repo, "checkout", "-q", "main")
    vcs.merge(repo, "feat")
    assert (repo / "a.txt").read_text(encoding="utf-8") == "from the branch\n"


def test_merging_two_branches_that_touched_different_files_is_clean(
    repo: Path, vcs: Git
) -> None:
    # Genuinely divergent history, but no overlapping file: git merges it.
    git(repo, "checkout", "-q", "-b", "theirs")
    commit_file(repo, "theirs-only.txt", "theirs\n", "theirs work")
    git(repo, "checkout", "-q", "main")
    commit_file(repo, "ours-only.txt", "ours\n", "ours work")

    assert vcs.merge(repo, "theirs").clean is True
    assert (repo / "theirs-only.txt").is_file()
    assert (repo / "ours-only.txt").is_file()


def test_no_ff_makes_a_merge_commit_even_when_fast_forwardable(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "add a")
    git(repo, "checkout", "-q", "main")
    result = vcs.merge(repo, "feat", no_ff=True)
    parents = git(repo, "rev-list", "--parents", "-1", "HEAD").split()
    assert len(parents) == 3  # the commit itself plus two parents
    assert result.sha != vcs.rev_parse("feat")


def test_without_no_ff_a_fast_forwardable_merge_moves_the_pointer(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "add a")
    git(repo, "checkout", "-q", "main")
    result = vcs.merge(repo, "feat", no_ff=False)
    assert result.clean is True
    assert result.sha == vcs.rev_parse("feat")


def test_merging_an_already_merged_branch_is_clean_and_changes_nothing(
    repo: Path, vcs: Git
) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "add a")
    git(repo, "checkout", "-q", "main")
    vcs.merge(repo, "feat")
    after_first = vcs.rev_parse("HEAD")
    second = vcs.merge(repo, "feat")
    assert second.clean is True
    assert second.sha == after_first
    assert vcs.rev_parse("HEAD") == after_first


def test_a_merge_lands_where_the_workflow_checks_for_it(repo: Path, vcs: Git) -> None:
    # `is_ancestor(source, target)` is how the workflow confirms a merge landed.
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "add a")
    git(repo, "checkout", "-q", "main")
    assert vcs.is_ancestor("feat", "main") is False
    vcs.merge(repo, "feat")
    assert vcs.is_ancestor("feat", "main") is True


def test_merging_in_a_worktree_leaves_the_main_repo_alone(
    repo: Path, vcs: Git, tmp_path: Path
) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "add a")
    git(repo, "checkout", "-q", "main")
    before = vcs.rev_parse("main")
    tree = vcs.add_worktree(tmp_path / "merge-tree", "integration", "main")
    result = vcs.merge(tree.path, "feat")
    assert result.clean is True
    assert vcs.rev_parse("integration") == result.sha
    assert vcs.rev_parse("main") == before


def test_merging_an_unknown_ref_raises(repo: Path, vcs: Git) -> None:
    with pytest.raises(UnknownRefError):
        vcs.merge(repo, "no-such-branch")


# -- conflicted merges ----------------------------------------------------


def test_a_conflicting_merge_reports_not_clean(repo: Path, vcs: Git, diverged: Diverged) -> None:
    result = vcs.merge(repo, diverged.theirs)
    assert result.clean is False
    assert result.sha is None


def test_a_conflicting_merge_stays_in_progress(repo: Path, vcs: Git, diverged: Diverged) -> None:
    vcs.merge(repo, diverged.theirs)
    assert vcs.merge_in_progress(repo) is True


def test_a_conflicting_merge_lists_the_unmerged_path(
    repo: Path, vcs: Git, diverged: Diverged
) -> None:
    vcs.merge(repo, diverged.theirs)
    assert vcs.unmerged_paths(repo) == ("a.txt",)


def test_a_conflicting_merge_returns_the_parsed_conflict(
    repo: Path, vcs: Git, diverged: Diverged
) -> None:
    result = vcs.merge(repo, diverged.theirs)
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.path == "a.txt"
    assert len(conflict.hunks) == 1
    assert conflict.hunks[0].ours == ("ours",)
    assert conflict.hunks[0].theirs == ("theirs",)


def test_conflicts_can_be_asked_for_again_afterwards(
    repo: Path, vcs: Git, diverged: Diverged
) -> None:
    result = vcs.merge(repo, diverged.theirs)
    assert vcs.conflicts(repo) == result.conflicts


def test_conflicts_outside_a_merge_are_empty(repo: Path, vcs: Git) -> None:
    assert vcs.conflicts(repo) == ()
    assert vcs.unmerged_paths(repo) == ()
    assert vcs.merge_in_progress(repo) is False


def test_two_conflicting_files_produce_two_conflicts(repo: Path, vcs: Git) -> None:
    commit_file(repo, "a.txt", "base a\n", "add a")
    commit_file(repo, "b.txt", "base b\n", "add b")
    git(repo, "checkout", "-q", "-b", "theirs")
    commit_file(repo, "a.txt", "theirs a\n", "their a")
    commit_file(repo, "b.txt", "theirs b\n", "their b")
    git(repo, "checkout", "-q", "main")
    commit_file(repo, "a.txt", "ours a\n", "our a")
    commit_file(repo, "b.txt", "ours b\n", "our b")

    result = vcs.merge(repo, "theirs")
    assert [conflict.path for conflict in result.conflicts] == ["a.txt", "b.txt"]
    assert vcs.unmerged_paths(repo) == ("a.txt", "b.txt")


def test_two_regions_in_one_file_produce_two_hunks(repo: Path, vcs: Git) -> None:
    base = "one\n" + "filler\n" * 20 + "two\n"
    ours = "ONE\n" + "filler\n" * 20 + "TWO\n"
    theirs = "1\n" + "filler\n" * 20 + "2\n"
    diverged = make_diverged(repo, "a.txt", base, ours, theirs)

    result = vcs.merge(repo, diverged.theirs)
    assert len(result.conflicts) == 1
    hunks = result.conflicts[0].hunks
    assert len(hunks) == 2
    assert hunks[0].ours == ("ONE",)
    assert hunks[1].ours == ("TWO",)


def test_a_modify_delete_conflict_is_reported_without_hunks(repo: Path, vcs: Git) -> None:
    # Git leaves no markers in the file, but the path is still unresolved.
    commit_file(repo, "a.txt", "base\n", "add a")
    git(repo, "checkout", "-q", "-b", "theirs")
    git(repo, "rm", "-q", "a.txt")
    git(repo, "commit", "-q", "-m", "delete a")
    git(repo, "checkout", "-q", "main")
    commit_file(repo, "a.txt", "edited\n", "edit a")

    result = vcs.merge(repo, "theirs")
    assert result.clean is False
    assert [conflict.path for conflict in result.conflicts] == ["a.txt"]
    assert result.conflicts[0].hunks == ()


def test_a_conflict_shows_up_in_status_as_unmerged(
    repo: Path, vcs: Git, diverged: Diverged
) -> None:
    vcs.merge(repo, diverged.theirs)
    assert vcs.status(repo) == (FileStatus(path="a.txt", code="UU"),)


def test_the_conflicted_file_holds_both_sides_on_disk(
    repo: Path, vcs: Git, diverged: Diverged
) -> None:
    vcs.merge(repo, diverged.theirs)
    content = (repo / "a.txt").read_text(encoding="utf-8")
    assert "<<<<<<<" in content and "ours" in content and "theirs" in content


def test_a_three_way_conflict_style_is_parsed_with_its_base(
    repo: Path, vcs: Git, diverged: Diverged
) -> None:
    # A developer's `merge.conflictStyle = diff3` must not change the shape of
    # the answer, only fill in the base.
    git(repo, "config", "merge.conflictStyle", "diff3")
    result = vcs.merge(repo, diverged.theirs)
    assert result.conflicts[0].hunks[0].base == ("base",)


# -- aborting -------------------------------------------------------------


def test_abort_clears_the_merge(repo: Path, vcs: Git, diverged: Diverged) -> None:
    vcs.merge(repo, diverged.theirs)
    vcs.abort_merge(repo)
    assert vcs.merge_in_progress(repo) is False
    assert vcs.unmerged_paths(repo) == ()


def test_abort_restores_the_pre_merge_state(repo: Path, vcs: Git, diverged: Diverged) -> None:
    before = vcs.rev_parse("HEAD")
    vcs.merge(repo, diverged.theirs)
    vcs.abort_merge(repo)
    assert vcs.rev_parse("HEAD") == before
    assert (repo / "a.txt").read_text(encoding="utf-8") == "ours\n"
    assert vcs.has_changes(repo) is False


def test_aborting_when_no_merge_is_running_raises(repo: Path, vcs: Git) -> None:
    with pytest.raises(VcsError):
        vcs.abort_merge(repo)


def test_a_branch_can_be_merged_again_after_an_abort(
    repo: Path, vcs: Git, diverged: Diverged
) -> None:
    vcs.merge(repo, diverged.theirs)
    vcs.abort_merge(repo)
    assert vcs.merge(repo, diverged.theirs).clean is False


# -- resolving ------------------------------------------------------------


def test_a_hand_resolved_merge_can_be_committed(repo: Path, vcs: Git, diverged: Diverged) -> None:
    vcs.merge(repo, diverged.theirs)
    (repo / "a.txt").write_text("resolved\n", encoding="utf-8")
    git(repo, "add", "a.txt")

    sha = vcs.commit_merge(repo, "merge theirs into ours")
    assert sha == vcs.rev_parse("HEAD")
    assert vcs.merge_in_progress(repo) is False
    assert vcs.has_changes(repo) is False
    assert (repo / "a.txt").read_text(encoding="utf-8") == "resolved\n"


def test_a_resolved_merge_has_both_parents(repo: Path, vcs: Git, diverged: Diverged) -> None:
    vcs.merge(repo, diverged.theirs)
    (repo / "a.txt").write_text("resolved\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    vcs.commit_merge(repo, "merge")
    assert len(git(repo, "rev-list", "--parents", "-1", "HEAD").split()) == 3


def test_a_resolved_merge_makes_the_source_an_ancestor(
    repo: Path, vcs: Git, diverged: Diverged
) -> None:
    # The check the workflow uses to confirm a merge actually landed.
    vcs.merge(repo, diverged.theirs)
    (repo / "a.txt").write_text("resolved\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    vcs.commit_merge(repo, "merge")
    assert vcs.is_ancestor(diverged.theirs, diverged.ours) is True


def test_committing_with_paths_still_unresolved_raises(
    repo: Path, vcs: Git, diverged: Diverged
) -> None:
    vcs.merge(repo, diverged.theirs)
    with pytest.raises(VcsError):
        vcs.commit_merge(repo, "premature")
    assert vcs.merge_in_progress(repo) is True


def test_committing_a_merge_when_none_is_running_raises(repo: Path, vcs: Git) -> None:
    with pytest.raises(VcsError):
        vcs.commit_merge(repo, "nothing to merge")


def test_the_merge_message_round_trips(repo: Path, vcs: Git, diverged: Diverged) -> None:
    vcs.merge(repo, diverged.theirs)
    (repo / "a.txt").write_text("resolved\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    message = 'merge "theirs"\n\nResolved a.txt by hand.\n'
    vcs.commit_merge(repo, message)
    assert git(repo, "log", "-1", "--pretty=%B").strip() == message.strip()
