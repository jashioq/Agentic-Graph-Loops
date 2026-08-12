"""Commits, diffs, and merges — the half of `Git` that moves history.

Layer: core. `MergeOps` is not usable on its own; it is the group of operations
`Git` inherits, split out because one file holding all of git was over the size
this project allows. Conflict marker parsing lives in `conflicts.py`, so this
file reads files and runs commands and never looks at a `<<<<<<<` itself.

A conflicted merge is deliberately left in progress. The caller inspects the
conflicts, resolves them in the worktree, and calls `commit_merge` — which
stages what the merge had left unmerged — or `abort_merge`. Nothing here
silently unwinds a merge on the caller's behalf.
"""

from pathlib import Path

from agl.core._exec import ExecResult
from agl.core.vcs.api import Conflict, MergeResult, VcsError
from agl.core.vcs.impl._runner import GitRunner
from agl.core.vcs.impl.conflicts import parse_conflicts

__all__ = ["MergeOps"]


class MergeOps(GitRunner):
    """Committing an agent's work, diffing it, and merging it back."""

    # -- commits ----------------------------------------------------------

    def has_changes(self, cwd: Path) -> bool:
        return bool(self._porcelain(cwd))

    def commit_all(self, cwd: Path, message: str) -> str | None:
        if not self.has_changes(cwd):
            # Asked first rather than letting `git commit` fail on an empty
            # commit: "the agent changed nothing" is an answer, not a failure.
            return None
        self._run(["add", "-A"], cwd)
        result = self._run(["commit", "-m", message], cwd, check=False)
        if result.code != 0:
            raise VcsError(f"cannot commit in {cwd}: {self._reason(result)}")
        return self._head(cwd)

    # -- diffs ------------------------------------------------------------

    def diff(self, cwd: Path, base: str, head: str) -> str:
        return self._diff(["diff", f"{base}...{head}"], cwd, base, head).stdout

    def changed_files(self, cwd: Path, base: str, head: str) -> tuple[str, ...]:
        argv = ["diff", "--name-only", "-z", f"{base}...{head}"]
        result = self._diff(argv, cwd, base, head)
        return tuple(sorted(path for path in result.stdout.split("\0") if path))

    # -- merges -----------------------------------------------------------

    def merge(self, cwd: Path, source: str, no_ff: bool = True) -> MergeResult:
        argv = ["merge", "--no-edit", "--no-ff" if no_ff else "--ff", "--", source]
        result = self._run(argv, cwd, check=False)
        if result.code == 0:
            return MergeResult(clean=True, conflicts=(), sha=self._head(cwd))
        if self.unmerged_paths(cwd):
            # Left in progress on purpose: the caller resolves and commits, or
            # aborts. There is no third state where the merge quietly vanished.
            return MergeResult(clean=False, conflicts=self.conflicts(cwd), sha=None)
        self._require_refs(source)
        raise VcsError(f"cannot merge {source} into {cwd}: {self._reason(result)}")

    def merge_in_progress(self, cwd: Path) -> bool:
        result = self._run(["rev-parse", "-q", "--verify", "MERGE_HEAD"], cwd, check=False)
        return result.code == 0

    def conflicts(self, cwd: Path) -> tuple[Conflict, ...]:
        return tuple(
            Conflict(path=path, hunks=parse_conflicts(self._read(cwd / path)))
            for path in self.unmerged_paths(cwd)
        )

    def unmerged_paths(self, cwd: Path) -> tuple[str, ...]:
        result = self._run(["diff", "--name-only", "--diff-filter=U", "-z"], cwd)
        return tuple(sorted(path for path in result.stdout.split("\0") if path))

    def abort_merge(self, cwd: Path) -> None:
        result = self._run(["merge", "--abort"], cwd, check=False)
        if result.code != 0:
            raise VcsError(f"cannot abort the merge in {cwd}: {self._reason(result)}")

    def commit_merge(self, cwd: Path, message: str) -> str:
        if not self.merge_in_progress(cwd):
            # Without this, a stray call would quietly make an ordinary commit
            # out of whatever happened to be staged.
            raise VcsError(f"no merge in progress in {cwd}")
        self._stage_resolution(cwd)
        result = self._run(["commit", "-m", message], cwd, check=False)
        if result.code != 0:
            raise VcsError(f"cannot commit the merge in {cwd}: {self._reason(result)}")
        return self._head(cwd)

    # -- internals --------------------------------------------------------

    def _stage_resolution(self, cwd: Path) -> None:
        """Stage the paths this merge left unmerged, and nothing else.

        Named paths rather than `add -A`, because a merge commit is the worst
        place for a surprise: whatever else the tree happens to hold — an
        unrelated edit, an untracked scratch file — is none of this merge's
        business. `-A` under that pathspec is what makes a conflicted file the
        resolver *deleted* stage as a deletion instead of failing on a path
        that is no longer there; deleting is a legitimate resolution.

        Whether the content still holds conflict markers is not asked. This
        stages and commits what it was given; verification is the caller's.
        """
        unmerged = self.unmerged_paths(cwd)
        if not unmerged:
            return
        result = self._run(["add", "-A", "--", *unmerged], cwd, check=False)
        if result.code != 0:
            raise VcsError(f"cannot stage the resolution in {cwd}: {self._reason(result)}")

    def _diff(self, argv: list[str], cwd: Path, base: str, head: str) -> ExecResult:
        """Run a `base...head` diff, naming whichever ref failed to resolve.

        Three dots, always: the comparison is against the merge base, so work
        done on `base` after the divergence is not attributed to `head`.
        """
        result = self._run(argv, cwd, check=False)
        if result.code != 0:
            self._require_refs(base, head)
            raise VcsError(f"cannot diff {base}...{head}: {self._reason(result)}")
        return result
