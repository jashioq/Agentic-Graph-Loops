"""The harness a run is assembled on: one `RunContext` over fakes and real git.

This is what makes a workflow cheap to test. A workflow's whole input is a
`RunContext`, so `context(repo)` is the entire setup for a test that drives one:
real git in `tmp_path`, a real `FileStore` under a home beside it, a
`FakeAgentRunner` that never calls a model, and a `HeadlessTerminal` that
records every frame it would have painted. A test that cares about one of those
passes its own and takes the defaults for the rest.

Real where it is cheap, faked where it is not — the rule in CLAUDE.md. Git and
the store are the things a run's correctness is actually about, so both are
real; the agent and the terminal are the slow, costly, interactive ones, so both
are fakes from `tests/fakes.py`.

`feature` is separate from `context` because moving a repository onto a branch
is a change to the world and building a context is not. A run refuses to start
on `main`, so a test that drives one calls `feature(repo)` first — and a test
about that refusal is the one that does not.
"""

import sys
from pathlib import Path

import pytest

from agl.core.agent import AgentRunner
from agl.core.store.impl.file_store import FileStore
from agl.core.terminal import Terminal
from agl.core.vcs.impl.git import Git
from agl.runtime import paths
from agl.runtime.context import ProjectSettings, RunContext
from tests.conftest import git
from tests.fakes import FakeAgentRunner, HeadlessTerminal

__all__ = [
    "LABEL",
    "NO_OP_BUILD",
    "PROJECT",
    "REQUEST",
    "context",
    "feature",
    "settings",
]

LABEL = "add-auth"
PROJECT = "demo"
REQUEST = "Add authentication"

NO_OP_BUILD = (sys.executable, "-c", "pass")
"""A build that always passes, so the merge gate is open unless a test shuts it."""

def feature(repo: Path, name: str = "feature") -> str:
    """Move `repo` off `main` onto a feature branch, as a real run requires."""
    git(repo, "checkout", "-b", name, "main")
    return name


def settings(
    repo: Path,
    trees: Path | None = None,
    build: tuple[str, ...] = NO_OP_BUILD,
    build_timeout: float = 30.0,
) -> ProjectSettings:
    """The project a test run is against: the repo, and where its trees go."""
    return ProjectSettings(
        name=PROJECT,
        repo=repo,
        trees_root=repo.parent / "trees" if trees is None else trees,
        build=build,
        build_timeout=build_timeout,
    )


def context(
    repo: Path,
    *,
    agent: AgentRunner | None = None,
    terminal: Terminal | None = None,
    label: str = LABEL,
    request: str = REQUEST,
    base_branch: str | None = None,
    max_concurrent: int = 1,
    home: Path | None = None,
    trees: Path | None = None,
    build: tuple[str, ...] = NO_OP_BUILD,
    build_timeout: float = 30.0,
) -> RunContext:
    """Everything a run is given, over `repo`.

    `home` and `trees` default to siblings of the repository — the `repo`
    fixture puts it at `tmp_path/repo`, so both land in `tmp_path` and no test
    has to name them to get a layout that is already isolated. `base_branch`
    defaults to the branch the repository is standing on, which is what the cli
    passes and what `feature` has just created.
    """
    vcs = Git(repo)
    run_home = repo.parent / "home" if home is None else home
    return RunContext(
        label=label,
        request=request,
        base_branch=vcs.current_branch() if base_branch is None else base_branch,
        max_concurrent=max_concurrent,
        project=settings(repo, trees, build, build_timeout),
        agent=FakeAgentRunner() if agent is None else agent,
        vcs=vcs,
        store=FileStore(paths.run_dir(run_home, label)),
        terminal=HeadlessTerminal() if terminal is None else terminal,
    )


@pytest.fixture
def ctx(repo: Path) -> RunContext:
    """A context a run could start from: a feature branch and every default."""
    feature(repo)
    return context(repo)
