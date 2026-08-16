"""Vcs API: git, as structured data.

Layer: core. Worktrees, branches, commits, diffs and merges, in and out as
dataclasses. It has never heard of a work item, an agent or a run.

A conflicted merge reports the conflicting paths and nothing more — no
classifier, so a caller resolves in the worktree or halts for a person.
Synchronous throughout: git calls are milliseconds, and one made from an async
task serializes on the event loop rather than racing.

`cwd` chooses which worktree a question is about; where optional, `None` is the
main repository root.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BranchExistsError",
    "DirtyWorktreeError",
    "FileStatus",
    "MergeResult",
    "UnknownRefError",
    "Vcs",
    "VcsError",
    "Worktree",
]


@dataclass(frozen=True)
class Worktree:
    """A checked-out tree and the branch it has checked out."""

    path: Path
    branch: str


@dataclass(frozen=True)
class FileStatus:
    """One entry of a porcelain status: `M`, `A`, `D`, `R`, `??`, `UU`."""

    path: str
    code: str


@dataclass(frozen=True)
class MergeResult:
    """What a merge did: committed, or stopped with files to look at.

    `conflicted` is what `unmerged_paths` said when the merge stopped, carried
    along so a caller does not have to go back and ask.
    """

    clean: bool
    conflicted: tuple[str, ...]  # paths, empty when clean
    sha: str | None


class VcsError(Exception):
    """Anything git refused to do that a caller might reasonably handle."""


class BranchExistsError(VcsError):
    """A branch was asked for by a name that is already taken."""


class UnknownRefError(VcsError):
    """A ref does not resolve in this repository."""


class DirtyWorktreeError(VcsError):
    """An operation would have discarded uncommitted work."""


class Vcs(ABC):
    """Git operations a workflow needs, over one repository and its worktrees."""

    # -- repository -------------------------------------------------------

    @abstractmethod
    def root(self) -> Path:
        """The main repository root, even when constructed inside a worktree."""

    @abstractmethod
    def current_branch(self, cwd: Path | None = None) -> str:
        """The checked-out branch. Raises `VcsError` on a detached HEAD."""

    @abstractmethod
    def is_dirty(self, cwd: Path | None = None) -> bool:
        """Whether there is anything uncommitted — staged, unstaged, or untracked."""

    @abstractmethod
    def status(self, cwd: Path | None = None) -> tuple[FileStatus, ...]:
        """Every changed path with its porcelain code, sorted by path."""

    @abstractmethod
    def discard_changes(self, cwd: Path) -> None:
        """Throws away every uncommitted change in this tree, tracked or not.

        Ignored files are left alone. `cwd` is required, never defaulted.
        """

    # -- refs -------------------------------------------------------------

    @abstractmethod
    def rev_parse(self, ref: str) -> str:
        """The full sha `ref` resolves to. Raises `UnknownRefError`."""

    @abstractmethod
    def ref_exists(self, ref: str) -> bool:
        """Whether `ref` resolves to anything at all."""

    @abstractmethod
    def branch_exists(self, name: str) -> bool:
        """Whether a local branch of this name exists."""

    @abstractmethod
    def branches(self, prefix: str = "") -> tuple[str, ...]:
        """Short names of local branches starting with `prefix`, sorted."""

    @abstractmethod
    def create_branch(self, name: str, base: str) -> None:
        """Creates `name` at `base` without checking it out.

        Raises `BranchExistsError` or `UnknownRefError`.
        """

    @abstractmethod
    def delete_branch(self, name: str, force: bool = False) -> None:
        """Deletes a local branch.

        param: force - delete even when it holds commits merged nowhere
        """

    @abstractmethod
    def merge_base(self, a: str, b: str) -> str:
        """The sha where the two refs diverged. Raises `UnknownRefError`."""

    @abstractmethod
    def is_ancestor(self, maybe_ancestor: str, descendant: str) -> bool:
        """Whether the first ref is reachable from the second. A ref is its own ancestor."""

    # -- worktrees --------------------------------------------------------

    @abstractmethod
    def add_worktree(self, path: Path, branch: str, base: str) -> Worktree:
        """Creates `branch` off `base` and checks it out at `path`, in one step.

        Raises `BranchExistsError`, `UnknownRefError`, or `VcsError` on a taken path.
        """

    @abstractmethod
    def attach_worktree(self, path: Path, branch: str) -> Worktree:
        """Checks an existing branch out at `path` — the counterpart to `add_worktree`.

        Raises `UnknownRefError`, or `VcsError` if the path or branch is taken.
        """

    @abstractmethod
    def remove_worktree(self, path: Path, force: bool = False) -> None:
        """Deregisters the worktree and deletes its directory, leaving its branch alone.

        param: force - remove even when it holds uncommitted work
        """

    @abstractmethod
    def list_worktrees(self) -> tuple[Worktree, ...]:
        """Every registered worktree, the main one first."""

    @abstractmethod
    def prune_worktrees(self) -> None:
        """Drop registry entries whose directories are gone."""

    # -- commits ----------------------------------------------------------

    @abstractmethod
    def has_changes(self, cwd: Path) -> bool:
        """Whether `commit_all` in this tree would have anything to commit."""

    @abstractmethod
    def commit_all(self, cwd: Path, message: str) -> str | None:
        """Stages everything including untracked files, commits, returns the sha.

        return: str | None - `None` when there was nothing to commit, which is not an error
        """

    # -- diffs ------------------------------------------------------------

    @abstractmethod
    def diff(self, cwd: Path, base: str, head: str) -> str:
        """Unified diff of what `head` added since it diverged from `base`.

        The merge-base diff, `base...head`, so later commits on `base` do not appear.
        """

    @abstractmethod
    def changed_files(self, cwd: Path, base: str, head: str) -> tuple[str, ...]:
        """Paths `head` touched since diverging from `base`, sorted. Same comparison as `diff`."""

    # -- merges -----------------------------------------------------------

    @abstractmethod
    def merge(self, cwd: Path, source: str, no_ff: bool = True) -> MergeResult:
        """Merges `source` into whatever `cwd` has checked out.

        return: MergeResult - clean, with the new sha; or conflicted, with the paths
            and the merge left *in progress* for `commit_merge` or `abort_merge`
        """

    @abstractmethod
    def merge_in_progress(self, cwd: Path) -> bool:
        """Whether this tree is sitting in an unfinished merge."""

    @abstractmethod
    def unmerged_paths(self, cwd: Path) -> tuple[str, ...]:
        """Paths git considers unresolved, sorted. Empty outside a merge in progress."""

    @abstractmethod
    def abort_merge(self, cwd: Path) -> None:
        """Undo the merge and restore the tree. Raises `VcsError` if none is running."""

    @abstractmethod
    def commit_merge(self, cwd: Path, message: str) -> str:
        """Commits a merge whose conflicts have been resolved in the worktree.

        Stages exactly the paths the merge left unmerged, so unrelated work in
        the tree stays out. Leftover conflict markers are not checked for:
        verifying a resolution is the caller's job.
        """
