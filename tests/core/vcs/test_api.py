"""The API: data types, the error hierarchy, and what an implementation owes."""

import dataclasses
import inspect
from pathlib import Path

import pytest

from agl.core.vcs import (
    BranchExistsError,
    Conflict,
    ConflictHunk,
    DirtyWorktreeError,
    FileStatus,
    MergeResult,
    UnknownRefError,
    Vcs,
    VcsError,
    Worktree,
)

# -- the abstract base ----------------------------------------------------

REPOSITORY = {"root", "current_branch", "is_dirty", "status"}
REFS = {
    "rev_parse",
    "ref_exists",
    "branch_exists",
    "branches",
    "create_branch",
    "delete_branch",
    "merge_base",
    "is_ancestor",
}
WORKTREES = {"add_worktree", "remove_worktree", "list_worktrees", "prune_worktrees"}
COMMITS = {"has_changes", "commit_all"}
DIFFS = {"diff", "changed_files"}
MERGES = {
    "merge",
    "merge_in_progress",
    "conflicts",
    "unmerged_paths",
    "abort_merge",
    "commit_merge",
}
EVERYTHING = REPOSITORY | REFS | WORKTREES | COMMITS | DIFFS | MERGES


def test_vcs_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Vcs()  # type: ignore[abstract]


def test_an_incomplete_implementation_fails_at_instantiation() -> None:
    class Partial(Vcs):
        def root(self) -> Path:
            return Path()

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_every_declared_method_is_abstract() -> None:
    assert Vcs.__abstractmethods__ == EVERYTHING


def test_no_method_takes_a_reporting_callback() -> None:
    # Core modules report by returning values. A callback parameter here would
    # be telemetry sneaking in through the back door.
    for name in EVERYTHING:
        parameters = inspect.signature(getattr(Vcs, name)).parameters
        assert not [p for p in parameters if p.startswith("on_")], name


def test_queries_default_to_the_main_repository() -> None:
    for method in ("current_branch", "is_dirty", "status"):
        parameters = inspect.signature(getattr(Vcs, method)).parameters
        assert parameters["cwd"].default is None, method


# -- errors ---------------------------------------------------------------


def test_every_error_is_a_vcs_error() -> None:
    for error in (BranchExistsError, UnknownRefError, DirtyWorktreeError):
        assert issubclass(error, VcsError)


def test_vcs_error_is_an_exception() -> None:
    assert issubclass(VcsError, Exception)


# -- data types -----------------------------------------------------------


def test_worktree_carries_a_path_and_a_branch() -> None:
    worktree = Worktree(path=Path("/trees/T-03"), branch="agl/add-auth/T-03")
    assert (worktree.path, worktree.branch) == (Path("/trees/T-03"), "agl/add-auth/T-03")


def test_file_status_carries_a_path_and_a_porcelain_code() -> None:
    status = FileStatus(path="src/a.py", code="M")
    assert (status.path, status.code) == ("src/a.py", "M")


@pytest.mark.parametrize(
    "kind", [Worktree, FileStatus, ConflictHunk, Conflict, MergeResult]
)
def test_the_data_types_are_frozen_dataclasses(kind: type) -> None:
    assert dataclasses.is_dataclass(kind)
    assert kind.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_worktrees_compare_by_value() -> None:
    assert Worktree(Path("/a"), "main") == Worktree(Path("/a"), "main")


def test_a_two_way_hunk_has_no_base() -> None:
    hunk = ConflictHunk(ours=("ours",), theirs=("theirs",), base=None)
    assert hunk.base is None


def test_a_three_way_hunk_carries_the_base() -> None:
    hunk = ConflictHunk(ours=("a",), theirs=("b",), base=("original",))
    assert hunk.base == ("original",)


def test_a_conflict_carries_a_path_and_its_hunks() -> None:
    hunk = ConflictHunk(ours=(), theirs=(), base=None)
    conflict = Conflict(path="src/a.py", hunks=(hunk,))
    assert (conflict.path, conflict.hunks) == ("src/a.py", (hunk,))


def test_a_clean_merge_result_has_a_sha_and_no_conflicts() -> None:
    result = MergeResult(clean=True, conflicts=(), sha="abc123")
    assert (result.clean, result.conflicts, result.sha) == (True, (), "abc123")


def test_a_conflicted_merge_result_has_no_sha() -> None:
    result = MergeResult(clean=False, conflicts=(Conflict("a.py", ()),), sha=None)
    assert result.sha is None
    assert result.conflicts[0].path == "a.py"
