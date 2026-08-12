"""Committing an agent's work, and diffing it against where it started."""

from pathlib import Path

import pytest

from agl.core.vcs import UnknownRefError
from agl.core.vcs.impl.git import Git
from tests.conftest import commit_file, git


@pytest.fixture
def vcs(repo: Path) -> Git:
    return Git(repo)


# -- has_changes ----------------------------------------------------------


def test_a_clean_tree_has_no_changes(repo: Path, vcs: Git) -> None:
    assert vcs.has_changes(repo) is False


def test_an_edit_is_a_change(repo: Path, vcs: Git) -> None:
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    assert vcs.has_changes(repo) is True


def test_an_untracked_file_is_a_change(repo: Path, vcs: Git) -> None:
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    assert vcs.has_changes(repo) is True


def test_a_staged_change_is_a_change(repo: Path, vcs: Git) -> None:
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    git(repo, "add", "new.txt")
    assert vcs.has_changes(repo) is True


def test_has_changes_is_scoped_to_its_worktree(repo: Path, vcs: Git, tmp_path: Path) -> None:
    tree = vcs.add_worktree(tmp_path / "tree", "feat", "main")
    (tree.path / "new.txt").write_text("new\n", encoding="utf-8")
    assert vcs.has_changes(tree.path) is True
    assert vcs.has_changes(repo) is False


# -- commit_all -----------------------------------------------------------


def test_committing_a_clean_tree_returns_none(repo: Path, vcs: Git) -> None:
    # An agent that made no changes is a normal outcome, not an error.
    assert vcs.commit_all(repo, "nothing to do") is None


def test_committing_a_clean_tree_creates_no_commit(repo: Path, vcs: Git) -> None:
    before = vcs.rev_parse("HEAD")
    vcs.commit_all(repo, "nothing to do")
    assert vcs.rev_parse("HEAD") == before


def test_commit_all_returns_the_new_sha(repo: Path, vcs: Git) -> None:
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    sha = vcs.commit_all(repo, "edit the readme")
    assert sha == vcs.rev_parse("HEAD")
    assert len(sha or "") == 40


def test_the_tree_is_clean_after_committing(repo: Path, vcs: Git) -> None:
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    vcs.commit_all(repo, "edit the readme")
    assert vcs.has_changes(repo) is False


def test_commit_all_picks_up_untracked_files(repo: Path, vcs: Git) -> None:
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    (repo / "nested" / "deep").mkdir(parents=True)
    (repo / "nested" / "deep" / "other.txt").write_text("other\n", encoding="utf-8")
    vcs.commit_all(repo, "add files")
    assert vcs.changed_files(repo, "HEAD~1", "HEAD") == ("nested/deep/other.txt", "new.txt")


def test_commit_all_records_a_deletion(repo: Path, vcs: Git) -> None:
    (repo / "README.md").unlink()
    vcs.commit_all(repo, "drop the readme")
    assert vcs.has_changes(repo) is False
    assert not (repo / "README.md").exists()


def test_commit_all_uses_the_message(repo: Path, vcs: Git) -> None:
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    vcs.commit_all(repo, "a short subject")
    assert git(repo, "log", "-1", "--pretty=%B").strip() == "a short subject"


def test_a_message_with_a_newline_round_trips(repo: Path, vcs: Git) -> None:
    message = "subject line\n\nA body paragraph explaining the change.\n"
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    vcs.commit_all(repo, message)
    assert git(repo, "log", "-1", "--pretty=%B").strip() == message.strip()


def test_a_message_with_quotes_round_trips(repo: Path, vcs: Git) -> None:
    message = """fix the "quoted" thing and the 'other' one; $HOME `date` \\n"""
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    vcs.commit_all(repo, message)
    assert git(repo, "log", "-1", "--pretty=%B").strip() == message


def test_commit_all_commits_in_the_right_worktree(repo: Path, vcs: Git, tmp_path: Path) -> None:
    tree = vcs.add_worktree(tmp_path / "tree", "feat", "main")
    (tree.path / "a.txt").write_text("a\n", encoding="utf-8")
    sha = vcs.commit_all(tree.path, "work on the ticket")
    assert vcs.rev_parse("feat") == sha
    assert vcs.rev_parse("main") != sha


# -- diff -----------------------------------------------------------------


def test_diff_shows_the_branch_changes(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "hello\n", "add a")
    output = vcs.diff(repo, "main", "feat")
    assert "a.txt" in output
    assert "+hello" in output


def test_diff_of_a_ref_against_itself_is_empty(repo: Path, vcs: Git) -> None:
    assert vcs.diff(repo, "main", "main") == ""


def test_diff_is_taken_from_the_merge_base(repo: Path, vcs: Git) -> None:
    # `base` moving ahead after the divergence must not show up in the diff:
    # the review step reads this as "what this ticket did".
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "mine.txt", "mine\n", "ticket work")
    git(repo, "checkout", "-q", "main")
    commit_file(repo, "theirs.txt", "someone else\n", "unrelated work on main")

    output = vcs.diff(repo, "main", "feat")
    assert "mine.txt" in output
    assert "theirs.txt" not in output
    assert "someone else" not in output


def test_diff_shows_a_deletion(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    (repo / "README.md").unlink()
    vcs.commit_all(repo, "drop the readme")
    assert "-# repo" in vcs.diff(repo, "main", "feat")


def test_diff_on_an_unknown_ref_raises(repo: Path, vcs: Git) -> None:
    with pytest.raises(UnknownRefError):
        vcs.diff(repo, "main", "no-such-branch")


# -- changed_files --------------------------------------------------------


def test_changed_files_lists_what_the_branch_touched(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "src/a.py", "a\n", "add a")
    commit_file(repo, "src/b.py", "b\n", "add b")
    assert vcs.changed_files(repo, "main", "feat") == ("src/a.py", "src/b.py")


def test_changed_files_excludes_work_done_on_the_base(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "mine.txt", "mine\n", "ticket work")
    git(repo, "checkout", "-q", "main")
    commit_file(repo, "theirs.txt", "theirs\n", "unrelated")
    assert vcs.changed_files(repo, "main", "feat") == ("mine.txt",)


def test_changed_files_of_a_ref_against_itself_is_empty(repo: Path, vcs: Git) -> None:
    assert vcs.changed_files(repo, "main", "main") == ()


def test_changed_files_counts_a_file_touched_twice_once(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "a.txt", "one\n", "first")
    commit_file(repo, "a.txt", "two\n", "second")
    assert vcs.changed_files(repo, "main", "feat") == ("a.txt",)


def test_changed_files_includes_deletions(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    (repo / "README.md").unlink()
    vcs.commit_all(repo, "drop it")
    assert vcs.changed_files(repo, "main", "feat") == ("README.md",)


def test_changed_files_handles_a_path_containing_a_space(repo: Path, vcs: Git) -> None:
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "my notes.md", "notes\n", "add notes")
    assert vcs.changed_files(repo, "main", "feat") == ("my notes.md",)


def test_changed_files_handles_a_non_ascii_path(repo: Path, vcs: Git) -> None:
    # Git quotes such paths in its human-readable output; -z is why this works.
    git(repo, "checkout", "-q", "-b", "feat")
    commit_file(repo, "café.md", "x\n", "add café")
    assert vcs.changed_files(repo, "main", "feat") == ("café.md",)


def test_changed_files_on_an_unknown_ref_raises(repo: Path, vcs: Git) -> None:
    with pytest.raises(UnknownRefError):
        vcs.changed_files(repo, "no-such-branch", "main")
