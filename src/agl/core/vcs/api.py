"""Vcs API: git, as structured data.

Layer: core. Worktrees, branches, commits, diffs, and merges. It takes paths and
ref names and hands back dataclasses; it has never heard of a work item, an
agent, or a run.

A conflicted merge is reported as the paths that conflicted, and nothing more.
There is no classifier here deciding which conflicts are trivial, and the shape
of what a classifier would need — three-way output, the base text — is
deliberately absent rather than half-provided: a caller resolves in the worktree
or halts for a person.

The API is synchronous. Git operations are milliseconds, a sync call is far
easier to test and reason about, and a sync call made from an async task
serializes on the event loop rather than racing. A caller that has a genuinely
slow operation — a merge on a large repository — wraps it in
`asyncio.to_thread` itself.

`cwd` is how a caller chooses which worktree a question is about. Where it is
optional, `None` means the main repository root; where a method must act inside
one particular tree, it is required.
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

    Self-describing on purpose. `conflicted` is what `unmerged_paths` would say
    at the moment the merge stopped, carried along so a caller writing a halt
    banner does not have to go back and ask.
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
        """The checked-out branch, in the main repo or in the worktree at `cwd`.

        Raises `VcsError` on a detached HEAD, which no worktree this module
        creates is ever in.
        """

    @abstractmethod
    def is_dirty(self, cwd: Path | None = None) -> bool:
        """Whether there is anything uncommitted — staged, unstaged, or untracked."""

    @abstractmethod
    def status(self, cwd: Path | None = None) -> tuple[FileStatus, ...]:
        """Every changed path with its porcelain code, sorted by path."""

    @abstractmethod
    def discard_changes(self, cwd: Path) -> None:
        """Throw away everything uncommitted in this tree.

        Tracked edits and untracked files alike. Ignored files are left alone —
        build output is not uncommitted work, and it is expensive to make
        again. `cwd` is required: discarding is never something to do to the
        main repository by accident.
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
        """Create `name` at `base` without checking it out.

        Raises `BranchExistsError` if the name is taken — the caller decides
        whether that is recoverable — or `UnknownRefError` if `base` does not
        resolve.
        """

    @abstractmethod
    def delete_branch(self, name: str, force: bool = False) -> None:
        """Delete a local branch.

        Raises `VcsError` if it holds commits that are not merged anywhere,
        unless `force`, and `UnknownRefError` if there is no such branch.
        """

    @abstractmethod
    def merge_base(self, a: str, b: str) -> str:
        """The sha where the two refs diverged. Raises `UnknownRefError`."""

    @abstractmethod
    def is_ancestor(self, maybe_ancestor: str, descendant: str) -> bool:
        """Whether the first ref is reachable from the second.

        A ref is its own ancestor. Raises `UnknownRefError` if either side does
        not resolve.
        """

    # -- worktrees --------------------------------------------------------

    @abstractmethod
    def add_worktree(self, path: Path, branch: str, base: str) -> Worktree:
        """Create `branch` off `base` and check it out at `path`, in one step.

        Raises `BranchExistsError` if the branch is already there,
        `UnknownRefError` if `base` does not resolve, and `VcsError` if the
        path is taken.
        """

    @abstractmethod
    def attach_worktree(self, path: Path, branch: str) -> Worktree:
        """Check an existing branch out at `path`.

        The counterpart to `add_worktree`, which creates one. Raises
        `UnknownRefError` if the branch does not exist and `VcsError` if the
        path is taken or the branch is checked out in another tree.
        """

    @abstractmethod
    def remove_worktree(self, path: Path, force: bool = False) -> None:
        """Deregister the worktree and delete its directory.

        Raises `DirtyWorktreeError` if it holds uncommitted work and `force` is
        not set, or `VcsError` if no worktree is registered at `path`. The
        branch it had checked out is left alone.
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
        """Stage everything, including untracked files, commit, return the sha.

        Returns `None` when there was nothing to commit, rather than raising:
        a tree that ended up unchanged is an ordinary outcome for the caller to
        handle, not an error. Raises `VcsError` if the commit fails.
        """

    # -- diffs ------------------------------------------------------------

    @abstractmethod
    def diff(self, cwd: Path, base: str, head: str) -> str:
        """Unified diff of what `head` added since it diverged from `base`.

        This is the merge-base diff — `base...head` — so commits made on `base`
        after the divergence do not appear: what `head` did is `head`'s, and
        nothing that landed elsewhere meanwhile belongs in it. Raises
        `UnknownRefError`.
        """

    @abstractmethod
    def changed_files(self, cwd: Path, base: str, head: str) -> tuple[str, ...]:
        """Paths `head` touched since diverging from `base`, sorted.

        The same merge-base comparison `diff` makes. Raises `UnknownRefError`.
        """

    # -- merges -----------------------------------------------------------

    @abstractmethod
    def merge(self, cwd: Path, source: str, no_ff: bool = True) -> MergeResult:
        """Merge `source` into whatever `cwd` has checked out.

        A clean merge commits and comes back with `clean=True` and the new sha.
        A conflicted one comes back with `clean=False`, the conflicting paths,
        and no sha, leaving the merge *in progress* on purpose: the caller
        inspects, resolves, and then calls `commit_merge` — or `abort_merge`.
        There is no third, silent state.

        Raises `UnknownRefError` if `source` does not resolve, and `VcsError`
        if git refused for any reason other than conflicts.
        """

    @abstractmethod
    def merge_in_progress(self, cwd: Path) -> bool:
        """Whether this tree is sitting in an unfinished merge."""

    @abstractmethod
    def unmerged_paths(self, cwd: Path) -> tuple[str, ...]:
        """Paths git considers unresolved, sorted.

        Only meaningful during a merge in progress; otherwise empty. This is
        what tells a conflicted merge from a hard failure, and what a halt
        banner names.
        """

    @abstractmethod
    def abort_merge(self, cwd: Path) -> None:
        """Undo the merge and restore the tree. Raises `VcsError` if none is running."""

    @abstractmethod
    def commit_merge(self, cwd: Path, message: str) -> str:
        """Commit a merge whose conflicts have been resolved in the worktree.

        Resolving means leaving the files as they should be — edited, or
        deleted — and nothing else: the paths the merge left unmerged are
        staged here, since `Vcs` offers no way to stage and the documented
        path has to be walkable through this interface alone. Only those paths
        are staged, so unrelated work in the tree stays out of the merge commit.

        Conflict markers still in a file are not checked for. This commits what
        it was given; verifying a resolution is the caller's job.

        Raises `VcsError` if no merge is in progress.
        """
