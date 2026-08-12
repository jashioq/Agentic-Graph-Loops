"""Where things live: run directories, worktree paths, and branch names.

Layer: core. Pure functions over `Path` and `str` — nothing here creates,
checks, or touches anything on disk, and the paths it returns need not exist.
The modules that do the I/O receive concrete paths in their constructors, which
is what keeps `store` and `vcs` free of any config knowledge.

The layout::

    <home>/projects/<project>/                    project config and standards
    <home>/runs/<label>/                          run artifacts
    <trees_root>/<project>/<label>/<ticket_id>/   one worktree per ticket

Worktrees sit outside both the orchestrator repo and the target repo, and the
trees root is supplied by the caller rather than derived from the home, so it
stays configurable.

There is no merge worktree here, and that is a decision rather than an omission:
merges happen in the main repository root. Git refuses to check out a branch
that another worktree already holds, and the base branch is checked out in the
main repository, so a merge worktree could never hold the branch being merged
into. The root is already on the base branch, ticket worktrees are untouched
either way, and a conflict halt leaves the markers where someone looking to
resolve them would go.

Branches are `agl/<label>/<ticket_id>`, with bug branches as hyphenated
siblings: `agl/<label>/<ticket_id>-bug-<n>`. Git stores refs as files, so
`agl/add-auth/T-03` being a file rules out `agl/add-auth/T-03/bug-1` — one path
cannot be both a file and a directory. Filesystem paths carry no such
constraint and nest freely. The `agl/` prefix keeps a user's own branch named
exactly the label clear of everything this tool creates and deletes.
"""

import re
from pathlib import Path

__all__ = [
    "InvalidLabelError",
    "branch_namespace",
    "bug_branch",
    "project_config",
    "project_dir",
    "project_standards",
    "run_dir",
    "ticket_branch",
    "trees_dir",
    "validate_label",
    "worktree_dir",
]

BRANCH_PREFIX = "agl"

_LABEL = re.compile(r"\A[a-z0-9][a-z0-9-]*\Z")


class InvalidLabelError(Exception):
    """Raised when a run label would be illegal as a path segment or a git ref."""


# -- the home layout ------------------------------------------------------


def project_dir(home: Path, project: str) -> Path:
    """The directory holding one project's configuration and standards."""
    return home / "projects" / project


def project_config(home: Path, project: str) -> Path:
    """The project's `config.toml`."""
    return project_dir(home, project) / "config.toml"


def project_standards(home: Path, project: str) -> Path:
    """The project's `standards.md`, handed to agents as context."""
    return project_dir(home, project) / "standards.md"


def run_dir(home: Path, label: str) -> Path:
    """The directory holding one run's artifacts — `spec.md`, `tickets.json`."""
    return home / "runs" / label


# -- worktrees ------------------------------------------------------------


def trees_dir(trees_root: Path, project: str, label: str) -> Path:
    """The directory holding every worktree for one run of one project."""
    return trees_root / project / label


def worktree_dir(trees_root: Path, project: str, label: str, ticket_id: str) -> Path:
    """The worktree a single ticket is worked in."""
    return trees_dir(trees_root, project, label) / ticket_id


# -- branches -------------------------------------------------------------


def branch_namespace(label: str) -> str:
    """`agl/<label>` — the prefix a clean deletes, and nothing outside it."""
    return f"{BRANCH_PREFIX}/{label}"


def ticket_branch(label: str, ticket_id: str) -> str:
    """The branch a ticket's work lands on: `agl/<label>/<ticket_id>`."""
    return f"{branch_namespace(label)}/{ticket_id}"


def bug_branch(label: str, parent_ticket_id: str, n: int) -> str:
    """The nth bug branch off a ticket: `agl/<label>/<ticket_id>-bug-<n>`.

    A sibling of its parent branch rather than a child of it, because a git ref
    that exists as a file cannot also be a directory.
    """
    return f"{branch_namespace(label)}/{parent_ticket_id}-bug-{n}"


# -- validation -----------------------------------------------------------


def validate_label(label: str) -> None:
    """Raise `InvalidLabelError` unless `label` matches `[a-z0-9][a-z0-9-]*`.

    A label becomes both a path segment and part of a git ref, so it is checked
    once up front instead of being discovered as a git error several steps in.
    """
    if not _LABEL.fullmatch(label):
        raise InvalidLabelError(
            f"invalid run label {label!r}: expected lowercase letters, digits, and hyphens, "
            "starting with a letter or digit"
        )
