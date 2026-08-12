"""Vcs core module. Re-exports the API only — never anything from `impl`.

Workflows import from here. Only `cli.py` may reach into `impl`.
"""

from agl.core.vcs.api import (
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

__all__ = [
    "BranchExistsError",
    "Conflict",
    "ConflictHunk",
    "DirtyWorktreeError",
    "FileStatus",
    "MergeResult",
    "UnknownRefError",
    "Vcs",
    "VcsError",
    "Worktree",
]
