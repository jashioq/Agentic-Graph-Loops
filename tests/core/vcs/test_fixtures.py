"""The fixtures themselves: every git test below rests on these being right."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import commit_file, git, make_diverged

# -- the repo -------------------------------------------------------------


def test_the_repo_is_a_git_repository(repo: Path) -> None:
    assert (repo / ".git").is_dir()


def test_the_repo_is_on_main(repo: Path) -> None:
    # `git init -b main`, never the machine's `init.defaultBranch`.
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"


def test_the_repo_has_exactly_one_commit(repo: Path) -> None:
    assert git(repo, "rev-list", "--count", "HEAD").strip() == "1"


def test_the_repo_is_clean(repo: Path) -> None:
    assert git(repo, "status", "--porcelain") == ""


def test_the_identity_is_set_locally(repo: Path) -> None:
    assert git(repo, "config", "user.name").strip() == "AGL Test"
    assert git(repo, "config", "user.email").strip() == "test@agl.invalid"


def test_the_conflict_style_is_the_two_way_default(repo: Path) -> None:
    assert git(repo, "config", "merge.conflictStyle").strip() == "merge"


# -- isolation ------------------------------------------------------------


def test_global_and_system_config_point_at_nothing() -> None:
    for name in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        assert not Path(os.environ[name]).exists()


def test_a_global_setting_cannot_leak_into_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real global config, in the place git is told to look for one — and
    # still not seen. Pointed at this test's own file so the setting dies here.
    hostile = tmp_path / "hostile-gitconfig"
    hostile.write_text(
        "[init]\n\tdefaultBranch = trunk\n[core]\n\thooksPath = /nope\n", encoding="utf-8"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile))
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-b", "main")
    # `symbolic-ref`, because HEAD is unborn until the first commit.
    assert git(other, "symbolic-ref", "--short", "HEAD").strip() == "main"


# -- commit_file ----------------------------------------------------------


def test_commit_file_returns_the_new_head_sha(repo: Path) -> None:
    sha = commit_file(repo, "a.txt", "a\n", "add a")
    assert sha == git(repo, "rev-parse", "HEAD").strip()
    assert len(sha) == 40


def test_commit_file_leaves_the_tree_clean(repo: Path) -> None:
    commit_file(repo, "a.txt", "a\n", "add a")
    assert git(repo, "status", "--porcelain") == ""


def test_commit_file_writes_the_content(repo: Path) -> None:
    commit_file(repo, "a.txt", "content\n", "add a")
    assert (repo / "a.txt").read_text(encoding="utf-8") == "content\n"


def test_commit_file_creates_intermediate_directories(repo: Path) -> None:
    commit_file(repo, "src/deep/a.txt", "a\n", "add a")
    assert (repo / "src" / "deep" / "a.txt").is_file()


def test_commit_file_can_amend_an_existing_path(repo: Path) -> None:
    first = commit_file(repo, "a.txt", "one\n", "add a")
    second = commit_file(repo, "a.txt", "two\n", "edit a")
    assert first != second
    assert (repo / "a.txt").read_text(encoding="utf-8") == "two\n"


# -- make_diverged --------------------------------------------------------


def test_make_diverged_creates_both_branches(repo: Path) -> None:
    make_diverged(repo, "a.txt", "base\n", "ours\n", "theirs\n")
    heads = git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/").split()
    assert set(heads) == {"main", "ours", "theirs"}


def test_make_diverged_leaves_the_repo_on_the_ours_branch(repo: Path) -> None:
    diverged = make_diverged(repo, "a.txt", "base\n", "ours\n", "theirs\n")
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == diverged.ours
    assert (repo / "a.txt").read_text(encoding="utf-8") == "ours\n"


def test_the_branches_share_the_base_commit(repo: Path) -> None:
    diverged = make_diverged(repo, "a.txt", "base\n", "ours\n", "theirs\n")
    merge_base = git(repo, "merge-base", diverged.ours, diverged.theirs).strip()
    assert merge_base == diverged.base


def test_neither_branch_is_an_ancestor_of_the_other(repo: Path) -> None:
    diverged = make_diverged(repo, "a.txt", "base\n", "ours\n", "theirs\n")
    assert diverged.ours_sha != diverged.theirs_sha
    for a, b in ((diverged.ours, diverged.theirs), (diverged.theirs, diverged.ours)):
        assert git(repo, "rev-list", f"{a}..{b}").strip() != ""


def test_the_starting_branch_stays_at_the_base(repo: Path) -> None:
    diverged = make_diverged(repo, "a.txt", "base\n", "ours\n", "theirs\n")
    assert git(repo, "rev-parse", "main").strip() == diverged.base


def test_the_two_branches_actually_conflict(repo: Path) -> None:
    # The whole point of the helper: a merge that stops with markers in the file.
    diverged = make_diverged(repo, "a.txt", "base\n", "ours\n", "theirs\n")
    merge = subprocess.run(
        ["git", "merge", "--no-edit", diverged.theirs],
        cwd=repo,
        capture_output=True,
        encoding="utf-8",
    )
    assert merge.returncode != 0
    assert "<<<<<<<" in (repo / "a.txt").read_text(encoding="utf-8")


def test_branch_names_can_be_chosen(repo: Path) -> None:
    diverged = make_diverged(
        repo, "a.txt", "base\n", "x\n", "y\n", ours_branch="agl/x", theirs_branch="agl/y"
    )
    assert (diverged.ours, diverged.theirs) == ("agl/x", "agl/y")
    heads = git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/").split()
    assert {"agl/x", "agl/y"} <= set(heads)
