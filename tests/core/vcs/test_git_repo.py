"""Repository queries and ref plumbing, against real repositories."""

from pathlib import Path

import pytest

from agl.core.vcs import BranchExistsError, FileStatus, UnknownRefError, VcsError
from agl.core.vcs.impl.git import Git
from tests.conftest import commit_file, git, make_diverged


@pytest.fixture
def vcs(repo: Path) -> Git:
    return Git(repo)


# -- construction ---------------------------------------------------------


def test_root_is_the_repository_root(repo: Path, vcs: Git) -> None:
    assert vcs.root() == repo.resolve()


def test_root_from_a_subdirectory_is_still_the_repository_root(repo: Path) -> None:
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    assert Git(sub).root() == repo.resolve()


def test_root_from_a_worktree_is_the_main_repository_root(repo: Path, tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    git(repo, "worktree", "add", "-b", "feat", str(tree), "main")
    assert Git(tree).root() == repo.resolve()


def test_constructing_on_a_non_repository_raises(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(VcsError):
        Git(plain)


def test_constructing_on_a_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(VcsError):
        Git(tmp_path / "nope")


# -- the current branch ---------------------------------------------------


def test_a_fresh_repo_is_on_main(vcs: Git) -> None:
    assert vcs.current_branch() == "main"


def test_current_branch_follows_a_checkout(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "agl/add-auth/T-03")
    assert vcs.current_branch() == "agl/add-auth/T-03"


def test_current_branch_on_a_detached_head_raises(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "--detach", "HEAD")
    with pytest.raises(VcsError):
        vcs.current_branch()


# -- dirtiness ------------------------------------------------------------


def test_a_clean_repo_is_not_dirty(vcs: Git) -> None:
    assert vcs.is_dirty() is False


def test_an_unstaged_edit_is_dirty(repo: Path, vcs: Git) -> None:
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    assert vcs.is_dirty() is True


def test_a_staged_but_uncommitted_change_is_dirty(repo: Path, vcs: Git) -> None:
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    git(repo, "add", "README.md")
    assert vcs.is_dirty() is True


def test_an_untracked_file_is_dirty(repo: Path, vcs: Git) -> None:
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    assert vcs.is_dirty() is True


def test_dirtiness_clears_after_a_commit(repo: Path, vcs: Git) -> None:
    commit_file(repo, "new.txt", "new\n", "add new")
    assert vcs.is_dirty() is False


# -- status ---------------------------------------------------------------


def test_status_on_a_clean_repo_is_empty(vcs: Git) -> None:
    assert vcs.status() == ()


def test_status_reports_a_modified_file(repo: Path, vcs: Git) -> None:
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    assert vcs.status() == (FileStatus(path="README.md", code="M"),)


def test_status_reports_an_added_file(repo: Path, vcs: Git) -> None:
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    assert vcs.status() == (FileStatus(path="a.txt", code="A"),)


def test_status_reports_a_deleted_file(repo: Path, vcs: Git) -> None:
    (repo / "README.md").unlink()
    assert vcs.status() == (FileStatus(path="README.md", code="D"),)


def test_status_reports_an_untracked_file(repo: Path, vcs: Git) -> None:
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    assert vcs.status() == (FileStatus(path="a.txt", code="??"),)


def test_status_reports_a_rename_under_its_new_path(repo: Path, vcs: Git) -> None:
    git(repo, "mv", "README.md", "DOCS.md")
    assert vcs.status() == (FileStatus(path="DOCS.md", code="R"),)


def test_status_handles_a_filename_containing_a_space(repo: Path, vcs: Git) -> None:
    # Splitting on NUL rather than whitespace is the whole reason for -z.
    (repo / "my notes.md").write_text("notes\n", encoding="utf-8")
    assert vcs.status() == (FileStatus(path="my notes.md", code="??"),)


def test_status_handles_a_tracked_filename_containing_a_space(repo: Path, vcs: Git) -> None:
    commit_file(repo, "my notes.md", "notes\n", "add notes")
    (repo / "my notes.md").write_text("edited\n", encoding="utf-8")
    assert vcs.status() == (FileStatus(path="my notes.md", code="M"),)


def test_status_lists_several_files_sorted_by_path(repo: Path, vcs: Git) -> None:
    (repo / "z.txt").write_text("z\n", encoding="utf-8")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    assert [entry.path for entry in vcs.status()] == ["README.md", "a.txt", "z.txt"]


def test_status_is_scoped_to_the_given_worktree(repo: Path, tmp_path: Path, vcs: Git) -> None:
    tree = tmp_path / "tree"
    vcs.add_worktree(tree, "feat", "main")
    (tree / "only-here.txt").write_text("x\n", encoding="utf-8")
    assert vcs.status(cwd=tree) == (FileStatus(path="only-here.txt", code="??"),)
    assert vcs.status() == ()
    assert vcs.is_dirty(cwd=tree) is True
    assert vcs.is_dirty() is False


# -- refs -----------------------------------------------------------------


def test_rev_parse_resolves_head_to_a_full_sha(repo: Path, vcs: Git) -> None:
    assert vcs.rev_parse("HEAD") == git(repo, "rev-parse", "HEAD").strip()
    assert len(vcs.rev_parse("HEAD")) == 40


def test_rev_parse_resolves_a_branch(repo: Path, vcs: Git) -> None:
    assert vcs.rev_parse("main") == vcs.rev_parse("HEAD")


def test_rev_parse_on_an_unknown_ref_raises(vcs: Git) -> None:
    with pytest.raises(UnknownRefError):
        vcs.rev_parse("no-such-branch")


def test_ref_exists_is_true_for_a_present_ref(vcs: Git) -> None:
    assert vcs.ref_exists("main") is True
    assert vcs.ref_exists("HEAD") is True


def test_ref_exists_is_false_for_an_absent_ref(vcs: Git) -> None:
    assert vcs.ref_exists("no-such-branch") is False


def test_branch_exists_tracks_creation_and_deletion(vcs: Git) -> None:
    assert vcs.branch_exists("feat") is False
    vcs.create_branch("feat", "main")
    assert vcs.branch_exists("feat") is True
    vcs.delete_branch("feat")
    assert vcs.branch_exists("feat") is False


def test_branch_exists_is_false_for_a_tag_of_that_name(repo: Path, vcs: Git) -> None:
    # A tag resolves as a ref but is not a branch.
    git(repo, "tag", "v1")
    assert vcs.ref_exists("v1") is True
    assert vcs.branch_exists("v1") is False


# -- branches -------------------------------------------------------------


def test_branches_lists_short_names(vcs: Git) -> None:
    assert vcs.branches() == ("main",)


def test_branches_filters_by_prefix(vcs: Git) -> None:
    for name in ("agl/add-auth/T-01", "agl/add-auth/T-02", "other"):
        vcs.create_branch(name, "main")
    assert vcs.branches("agl/") == ("agl/add-auth/T-01", "agl/add-auth/T-02")


def test_branches_is_sorted(vcs: Git) -> None:
    for name in ("zeta", "alpha", "mid"):
        vcs.create_branch(name, "main")
    assert vcs.branches() == ("alpha", "main", "mid", "zeta")


def test_branches_with_an_unmatched_prefix_is_empty(vcs: Git) -> None:
    assert vcs.branches("nothing/") == ()


# -- creating and deleting branches ---------------------------------------


def test_create_branch_points_at_the_base(repo: Path, vcs: Git) -> None:
    base = commit_file(repo, "a.txt", "a\n", "add a")
    vcs.create_branch("feat", "main")
    assert vcs.rev_parse("feat") == base


def test_create_branch_does_not_check_it_out(vcs: Git) -> None:
    vcs.create_branch("feat", "main")
    assert vcs.current_branch() == "main"


def test_create_branch_off_an_older_commit(repo: Path, vcs: Git) -> None:
    first = vcs.rev_parse("HEAD")
    commit_file(repo, "a.txt", "a\n", "add a")
    vcs.create_branch("feat", first)
    assert vcs.rev_parse("feat") == first


def test_creating_an_existing_branch_raises(vcs: Git) -> None:
    vcs.create_branch("feat", "main")
    with pytest.raises(BranchExistsError):
        vcs.create_branch("feat", "main")


def test_creating_a_branch_off_an_unknown_base_raises(vcs: Git) -> None:
    with pytest.raises(UnknownRefError):
        vcs.create_branch("feat", "no-such-base")


def test_delete_branch_removes_it(vcs: Git) -> None:
    vcs.create_branch("feat", "main")
    vcs.delete_branch("feat")
    assert "feat" not in vcs.branches()


def test_deleting_an_unmerged_branch_raises(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "only on feat")
    git(repo, "checkout", "-q", "main")
    with pytest.raises(VcsError):
        vcs.delete_branch("feat")
    assert vcs.branch_exists("feat")


def test_deleting_an_unmerged_branch_with_force_succeeds(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "a\n", "only on feat")
    git(repo, "checkout", "-q", "main")
    vcs.delete_branch("feat", force=True)
    assert not vcs.branch_exists("feat")


def test_deleting_a_missing_branch_raises(vcs: Git) -> None:
    with pytest.raises(UnknownRefError):
        vcs.delete_branch("no-such-branch")


# -- ancestry -------------------------------------------------------------


def test_merge_base_of_two_diverged_branches_is_the_common_ancestor(repo: Path, vcs: Git) -> None:
    diverged = make_diverged(repo, "a.txt", "base\n", "ours\n", "theirs\n")
    assert vcs.merge_base(diverged.ours, diverged.theirs) == diverged.base


def test_merge_base_of_a_ref_with_itself_is_itself(vcs: Git) -> None:
    assert vcs.merge_base("main", "main") == vcs.rev_parse("main")


def test_merge_base_of_an_unknown_ref_raises(vcs: Git) -> None:
    with pytest.raises(UnknownRefError):
        vcs.merge_base("main", "no-such-branch")


def test_is_ancestor_is_true_for_a_real_ancestor(repo: Path, vcs: Git) -> None:
    first = vcs.rev_parse("HEAD")
    second = commit_file(repo, "a.txt", "a\n", "add a")
    assert vcs.is_ancestor(first, second) is True


def test_is_ancestor_is_false_the_other_way_round(repo: Path, vcs: Git) -> None:
    first = vcs.rev_parse("HEAD")
    second = commit_file(repo, "a.txt", "a\n", "add a")
    assert vcs.is_ancestor(second, first) is False


def test_is_ancestor_is_false_for_siblings(repo: Path, vcs: Git) -> None:
    diverged = make_diverged(repo, "a.txt", "base\n", "ours\n", "theirs\n")
    assert vcs.is_ancestor(diverged.ours, diverged.theirs) is False
    assert vcs.is_ancestor(diverged.theirs, diverged.ours) is False


def test_a_ref_is_its_own_ancestor(vcs: Git) -> None:
    assert vcs.is_ancestor("main", "main") is True


def test_is_ancestor_on_an_unknown_ref_raises(vcs: Git) -> None:
    with pytest.raises(UnknownRefError):
        vcs.is_ancestor("main", "no-such-branch")
