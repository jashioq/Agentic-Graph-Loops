"""What a run was given: its settings, its connectors, and the checks before it starts.

Layer: runtime. `RunContext` is data and nothing else — the connectors a run
was handed, the project it is against, and the four things that make this run
this run. It has no lifecycle and no `start()`: every resource a run needs is
opened by the workflow, visibly, in its own `run()`, so reading that function
tells you what is open and when it closes. A context that opened things would
move half of that story in here, where it is nobody's local variable.

`preflight` is a function rather than something imposed on a workflow. It is the
first line of `run` because a workflow decides its own first line — one that
resumes an interrupted run would want different checks, and it should not have
to inherit these to get a context.

`build_gate` is the one place a project's build command becomes the callable the
merge queue holds. The queue never learns what a build is: it gets something to
await that answers with an `ExecResult`, which is why a run with no build and a
run behind Gradle are the same code path there.
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
]

_TRUNK = ("main", "master")
"""Branch names a run refuses to start on. Not configuration: the point is that
a run creates and deletes branches under a namespace and merges into the branch
it was started from, and doing that to a shared trunk is a mistake in every
project, whatever the trunk is called."""


@dataclass(frozen=True)
class ProjectSettings:
    """What runtime needs to know about a project, as data.

    A translation of the cli's `config.toml`, kept deliberately narrow: five
    fields a workflow uses, and none of the file-format knowledge that produced
    them. `agl.config` stays a cli concern, and a workflow can be handed
    settings assembled any other way — by a test, or by a caller with no config
    file at all.
    """

    name: str
    repo: Path
    trees_root: Path
    build: tuple[str, ...]
    build_timeout: float


@dataclass(frozen=True)
class RunContext:
    """One run's inputs: what it was asked to do, and what it may use to do it.

    Frozen, and holding no run state — nothing here changes between the first
    line of a run and the last. What a run has *done* lives in the workflow's
    own state and on the display's board, both of which the workflow creates and
    owns; this is only the part that was decided before it started.
    """

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


def preflight(vcs: Vcs, store: Store, label: str) -> None:
    """Refuse to start unless the repository and the label are clear.

    Every check is about work that already exists and would be silently mixed
    into this run's: uncommitted edits that a worktree's commits would land on
    top of, a trunk that must not be branched from and merged into, and a label
    whose branches or documents are still around from a run that did not finish.
    The last one names `agl clean` because that is the whole remedy.
    """
    if vcs.is_dirty():
        raise PreflightError("the repository has uncommitted changes")
    branch = vcs.current_branch()
    if branch in _TRUNK:
        raise PreflightError(f"cannot run on {branch!r}; check out a feature branch first")
    namespace = paths.branch_namespace(label)
    if vcs.branches(namespace) or store.list():
        raise PreflightError(f"{label!r} is already in use; run `agl clean {label}` first")


def build_gate(project: ProjectSettings) -> Build:
    """The project's build, as the callable a merge queue awaits.

    `check=False`: a failing build is the answer to the question the gate asked,
    not an error to raise past the queue that asked it. The timeout is the
    project's, and `run_async` kills the child when it expires, so a hung build
    ends the merge rather than the run.
    """

    async def build() -> ExecResult:
        return await run_async(
            list(project.build),
            project.repo,
            check=False,
            timeout=project.build_timeout,
        )

    return build
