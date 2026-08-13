"""Subprocess plumbing shared by the operation groups that make up `Git`.

Layer: core. Not an operation group itself: it holds the repository root, the
lock, and the handful of things every group needs — running git somewhere,
reading a porcelain status, naming a failure. `merges.py` and `git.py` build on
it, and nothing outside `impl/` touches it.

Every operation is a subprocess call through `agl.core._exec`, with refs and
paths passed as arguments so nothing is ever interpolated into a shell string,
and machine-readable output only — `--porcelain`, `-z`, `for-each-ref
--format` — because the human-readable forms are explicitly not an interface.

Failures are classified after the fact rather than guarded against in advance.
Asking `ref_exists` before every operation would double or triple the number of
processes on the path where nothing is wrong, and it would still be a guess:
git checks again anyway. So the command runs, and only when it refuses does the
implementation ask what was wrong.
"""

import threading
from pathlib import Path

from agl.core._exec import ExecError, ExecResult, run
from agl.core.vcs.api import FileStatus, UnknownRefError, VcsError

__all__ = ["GitRunner"]


class GitRunner:
    """One repository, the lock over it, and the way commands reach it."""

    def __init__(self, path: Path) -> None:
        """Resolve the main repository root from `path`.

        Raises `VcsError` if `path` is not inside a git repository. The root is
        the *main* one even when `path` is a linked worktree, so every later
        call has one fixed place to run repository-wide commands from.

        The lock guards operations that mutate shared repository state:
        worktree add and remove, branch create and delete. Git's own
        `index.lock` covers the index of one worktree and says nothing about
        the worktree registry or `refs/heads`, so two threads adding worktrees
        at once can otherwise lose an entry. A plain lock works whether the
        caller is on the event loop or in a thread, and everything it guards is
        milliseconds long.
        """
        try:
            common = self._git(["rev-parse", "--path-format=absolute", "--git-common-dir"], path)
        except FileNotFoundError as error:
            raise VcsError("git is not installed") from error
        except (ExecError, OSError) as error:
            raise VcsError(f"not a git repository: {path}") from error
        self._root = Path(common.stdout.strip()).parent.resolve()
        self._lock = threading.Lock()

    # -- running git ------------------------------------------------------

    def _run(self, argv: list[str], cwd: Path | None, check: bool = True) -> ExecResult:
        """Run a git command in a worktree, or in the main root when `cwd` is None."""
        return self._git(argv, cwd if cwd is not None else self._root, check)

    @staticmethod
    def _git(argv: list[str], cwd: Path, check: bool = True) -> ExecResult:
        """Run git in `cwd`. PATH resolution is left to the exec itself."""
        return run(["git", *argv], cwd=cwd, check=check)

    @staticmethod
    def _reason(result: ExecResult) -> str:
        """The most useful thing git said about why it refused."""
        return result.stderr.strip() or result.stdout.strip() or f"exit {result.code}"

    # -- questions every group asks ---------------------------------------

    def _head(self, cwd: Path | None = None) -> str:
        """The sha `cwd` has checked out — not necessarily the main repo's HEAD."""
        return self._run(["rev-parse", "HEAD"], cwd).stdout.strip()

    def _ref_exists(self, ref: str) -> bool:
        """Whether `ref` resolves. Exit-code based, so `--quiet` and no check."""
        result = self._run(["rev-parse", "--verify", "--quiet", ref], None, check=False)
        return result.code == 0 and bool(result.stdout.strip())

    def _require_refs(self, *refs: str) -> None:
        """Raise `UnknownRefError` for the first ref that does not resolve."""
        for ref in refs:
            if not self._ref_exists(ref):
                raise UnknownRefError(ref)

    def _porcelain(self, cwd: Path | None) -> tuple[FileStatus, ...]:
        """Parse `status --porcelain=v1 -z`.

        Each field is `XY<space><path>`: the code is exactly two columns wide,
        so it cannot be split off on whitespace — ` M` would lose its meaning.
        NUL separation is what keeps a filename with a space in it one field. A
        rename or copy is followed by an extra field holding the original path,
        which is consumed and dropped: the entry is reported under its new path.

        The two columns are reported with padding stripped, so an unstaged edit
        and a staged one are both `M`, while `??` and `UU` stay as they are.
        """
        result = self._run(["status", "--porcelain=v1", "-z"], cwd)
        fields = iter(result.stdout.split("\0"))
        entries: list[FileStatus] = []
        for field in fields:
            if not field:
                continue
            if len(field) < 4 or field[2] != " ":
                raise VcsError(f"cannot parse status entry: {field!r}")
            code, path = field[:2], field[3:]
            if code[0] in ("R", "C"):
                next(fields, None)
            entries.append(FileStatus(path=path, code=code.strip()))
        return tuple(entries)
