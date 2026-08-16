"""Where things live: run directories, worktree paths, and branch names.

Layer: runtime. Pure functions over `Path` and `str` — nothing touches disk and
the paths returned need not exist.

    <home>/projects/<project>/                  AGL's own settings for a project
    <home>/runs/<label>/                        one run's artifacts
    <trees_root>/<project>/<label>/<node_id>/   one worktree per node

Branches are `agl/<label>/<node_id>`, exactly one level below the namespace:
git stores refs as files, so nothing can nest under one. `label`, `project` and
`node_id` reach both a path and a ref, so every function that takes one
validates it here rather than letting git discover it later.
"""

import re
from pathlib import Path

__all__ = [
    "BRANCH_PREFIX",
    "InvalidNameError",
    "branch",
    "branch_namespace",
    "project_config",
    "project_dir",
    "run_dir",
    "trees_dir",
    "validate_label",
    "validate_node_id",
    "validate_project",
    "worktree_dir",
]

BRANCH_PREFIX = "agl"

# One shape, two character sets. They differ only on case: a label and a project
# are typed by a person, and on a case-insensitive filesystem `Add-Auth` and
# `add-auth` would be one directory shared by two runs. A node id is composed by
# a workflow from a single scheme, so it keeps its capitals.
_SUPPLIED = re.compile(r"\A[a-z0-9][a-z0-9-]*\Z")
_COMPOSED = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9-]*\Z")

_LOWER_ONLY = "lowercase letters, digits, and hyphens"
_ANY_CASE = "letters, digits, and hyphens"


class InvalidNameError(Exception):
    """Raised when a name would be illegal as a path segment or a git ref."""


# -- the home layout ------------------------------------------------------


def project_dir(home: Path, project: str) -> Path:
    """The directory holding AGL's own configuration for one project."""
    validate_project(project)
    return home / "projects" / project


def project_config(home: Path, project: str) -> Path:
    """The project's `config.toml`."""
    return project_dir(home, project) / "config.toml"


def run_dir(home: Path, label: str) -> Path:
    """The directory holding one run's artifacts, whatever a workflow puts there."""
    validate_label(label)
    return home / "runs" / label


# -- worktrees ------------------------------------------------------------


def trees_dir(trees_root: Path, project: str, label: str) -> Path:
    """The directory holding every worktree for one run of one project."""
    validate_project(project)
    validate_label(label)
    return trees_root / project / label


def worktree_dir(trees_root: Path, project: str, label: str, node_id: str) -> Path:
    """The worktree one node's work is done in."""
    validate_node_id(node_id)
    return trees_dir(trees_root, project, label) / node_id


# -- branches -------------------------------------------------------------


def branch_namespace(label: str) -> str:
    """`agl/<label>` — the prefix a clean deletes, and nothing outside it."""
    validate_label(label)
    return f"{BRANCH_PREFIX}/{label}"


def branch(label: str, node_id: str) -> str:
    """The branch one node's work lands on: `agl/<label>/<node_id>`."""
    validate_node_id(node_id)
    return f"{branch_namespace(label)}/{node_id}"


# -- validation -----------------------------------------------------------


def validate_label(label: str) -> None:
    """Raise `InvalidNameError` unless `label` matches `[a-z0-9][a-z0-9-]*`."""
    _validate("run label", label, _SUPPLIED, _LOWER_ONLY)


def validate_project(project: str) -> None:
    """Raise `InvalidNameError` unless `project` matches `[a-z0-9][a-z0-9-]*`."""
    _validate("project", project, _SUPPLIED, _LOWER_ONLY)


def validate_node_id(node_id: str) -> None:
    """Raise `InvalidNameError` unless `node_id` matches `[A-Za-z0-9][A-Za-z0-9-]*`.

    Refusing `/` is load-bearing: it keeps a derived id a sibling branch.
    """
    _validate("node id", node_id, _COMPOSED, _ANY_CASE)


def _validate(kind: str, value: str, pattern: re.Pattern[str], allowed: str) -> None:
    """One policy, one message. `kind` is what the caller called the value."""
    if not pattern.fullmatch(value):
        raise InvalidNameError(
            f"invalid {kind} {value!r}: expected {allowed}, "
            "starting with a letter or digit"
        )
