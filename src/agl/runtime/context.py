"""What a run was given: its settings, its connectors, and the checks before it starts.

Layer: runtime. `RunContext` is data and nothing else — no lifecycle, no
`start()`: every resource is opened by the workflow, visibly, in its own `run`.
The two preflights are functions a workflow calls rather than checks imposed on
it, so each entry point picks the set it wants. `build_gate` is the one place a
project's build command becomes the callable the merge queue awaits.
"""

from dataclasses import dataclass
from pathlib import Path

from agl.core.agent import AgentRunner
from agl.core.command import ExecResult, run_async
from agl.core.store import Store
from agl.core.terminal import Terminal
from agl.core.vcs import Vcs
from agl.runtime import paths
from agl.runtime.merge import Build

__all__ = [
    "PreflightError",
    "ProjectSettings",
    "RunContext",
    "build_gate",
    "preflight",
    "resume_preflight",
]

_TRUNK = ("main", "master")
"""Branch names a run refuses to start on. Not configuration: a run branches from
and merges into wherever it started, which must never be a shared trunk."""


@dataclass(frozen=True)
class ProjectSettings:
    """What runtime needs to know about a project, as data.

    A narrow translation of the cli's `config.toml`, carrying none of the
    file-format knowledge that produced it.
    """

    name: str
    repo: Path
    trees_root: Path
    build: tuple[str, ...]
    build_timeout: float


@dataclass(frozen=True)
class RunContext:
    """One run's inputs: what it was asked to do, and what it may use to do it.

    Frozen, and holding no run state — nothing here changes between a run's
    first line and its last. `workflow` is the one field a run cannot work out
    from the inside, and the record has to name it for `agl resume`.
    """

    workflow: str
    label: str
    request: str
    base_branch: str
    max_concurrent: int
    project: ProjectSettings
    agent: AgentRunner
    vcs: Vcs
    store: Store
    terminal: Terminal


class PreflightError(Exception):
    """Raised when the repository or the label is not in a state to start a run."""

# TODO preflight checks should be defined by each workflow specifically. To achieve that lets add some PreflightConfig object as a param to preflight functions. We can specify there a set of rules like if a workflow can run on main branch, can we allow for uncommited changes and stuff like that. There will be more and some of these requirements might be reused by other workflows but each will have a subset of them.
def preflight(vcs: Vcs, store: Store, label: str) -> None:
    """Refuses to start unless the repository and the label are clear.

    Raises `PreflightError` on uncommitted changes, a trunk branch, or a label
    already in use — every case being work that would silently mix into this run.
    """
    if vcs.is_dirty():
        raise PreflightError("the repository has uncommitted changes")
    branch = vcs.current_branch()
    if branch in _TRUNK:
        raise PreflightError(f"cannot run on {branch!r}; check out a feature branch first")
    namespace = paths.branch_namespace(label)
    if vcs.branches(namespace) or store.list():
        raise PreflightError(
            f"{label!r} is already in use; run `agl resume {label}` to continue it, "
            f"or `agl clean {label}` to discard it"
        )


def resume_preflight(vcs: Vcs, base_branch: str) -> None:
    """Refuses to resume unless the repository is where the run left it.

    The base branch must be checked out and the tree clean, except for a merge
    left in progress, which is the one mess a resume knows how to settle. None
    of `preflight`'s other checks: leftovers are the point of resuming.
    """
    branch = vcs.current_branch()
    if branch != base_branch:
        raise PreflightError(
            f"the run was started from {base_branch!r} and the repository is on "
            f"{branch!r}; check {base_branch!r} out first"
        )
    if vcs.is_dirty() and not vcs.merge_in_progress(vcs.root()):
        raise PreflightError("the repository has uncommitted changes")


def build_gate(project: ProjectSettings) -> Build:
    """The project's build, as the callable a merge queue awaits.

    return: Build - `check=False`, so a failing build is an answer, not an error;
        a timeout kills the child, ending the merge rather than the run
    """

    async def build() -> ExecResult:
        return await run_async(
            list(project.build),
            project.repo,
            check=False,
            timeout=project.build_timeout,
        )

    return build
