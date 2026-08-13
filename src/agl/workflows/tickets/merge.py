"""The merge queue: one merge at a time into the base branch, behind a build.

Layer: workflows. Composes `vcs` and `command` with this workflow's `state`; it
reports through callbacks and never touches a `RunState` itself.

**Why a queue.** Concurrent merges into one branch are concurrent writers to one
ref, with real failure modes — `index.lock` contention, a lost update, a
half-applied merge when two processes race. A single consumer makes those
impossible rather than rare. The second reason matters more: serialized merges
each integrate against the *current* tip, so a build can run before the ref is
trusted. That catches the failure git cannot see — one ticket renames a method,
another adds a caller, both merge cleanly, nothing conflicts, the build breaks.

**Where merges happen.** In the main repository root, which is checked out on the
base branch. Not in a dedicated merge worktree: git refuses to check out a branch
another tree already holds, so a `_merge` tree could never hold the base branch.
The root is already there, the ticket worktrees are untouched either way, and a
conflict leaves the markers exactly where a person would go to resolve them.

**A halt stops at the head.** Requests are still accepted; none are processed.
Everything behind a stuck merge would otherwise integrate against a base that is
about to change.

**What runs where.** Git calls stay on the event loop: they are milliseconds, and
leaving them there means no other task can interleave a git command between this
one's merge and the questions it asks about the result. The build is the one
thing here that runs for minutes, so it goes to a thread — a build that blocked
the loop would freeze the dashboard for its whole duration.
"""

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agl.core.command import ExecResult
from agl.core.vcs import Vcs, VcsError
from agl.workflows.tickets.state import Halt

__all__ = ["MergeQueue", "MergeRequest"]

TAIL_LINES = 20
"""How many lines of a failed build's output a halt carries.

A display choice, not a fact about builds. Which slice of the output matters is
language-specific — a Kotlin error sits early under a stack trace, a Rust one is
structured and late, a bundler dumps module paths — which is why `command`
returns the output whole and the truncation lives up here, next to the banner it
is being truncated for.
"""


@dataclass(frozen=True)
class MergeRequest:
    """One ticket's branch, asking to land."""

    ticket_id: str
    branch: str


class MergeQueue:
    """The single consumer that merges branches into the base branch, in order."""

    def __init__(
        self,
        vcs: Vcs,
        repo: Path,
        base_branch: str,
        build: Callable[[], ExecResult] | None,
        on_merged: Callable[[str], None],
        on_halt: Callable[[Halt], None],
        on_abandoned: Callable[[str], None],
    ) -> None:
        """Wire the queue to git, to the build gate, and to whoever is listening.

        `build` is a callable rather than a command line so the queue never has
        to know what a build looks like; `None` means no build gate. In
        production it closes over the project's command and calls
        `command.run(..., check=False, timeout=...)` — `check=False` because a
        failing build is the answer to the question, not an error.

        `on_abandoned` is the third outcome the other two cannot say: a person
        who resolves a halt by aborting the merge has ended that ticket's
        attempt without merging it and without leaving anything to halt on.
        """
        self._vcs = vcs
        self._repo = repo
        self._base_branch = base_branch
        self._build = build
        self._on_merged = on_merged
        self._on_halt = on_halt
        self._on_abandoned = on_abandoned

        self._pending: deque[MergeRequest] = deque()
        self._current: MergeRequest | None = None  # the one a halt is about
        self._halt: Halt | None = None
        self._resume_asked = False
        self._stopping = False
        self._running = False
        # One event for every reason the consumer might have to look again:
        # new work, a resume, or a stop. Set by the sync methods, cleared only
        # by the consumer, so a wake-up cannot be lost between the two.
        self._wake = asyncio.Event()
        self._stopped = asyncio.Event()

    # -- what the workflow calls ------------------------------------------

    def put(self, request: MergeRequest) -> None:
        """Queue a branch to be merged. Accepted while halted, like any other."""
        self._pending.append(request)
        self._wake.set()

    def resume(self) -> None:
        """Tell the consumer a person has dealt with the halt.

        Says nothing about *how*: the consumer goes and looks at git rather
        than trusting a protocol. Does no work itself, because the look may end
        in a build, and this is called from the event loop. A resume with
        nothing halted is ignored.
        """
        self._resume_asked = True
        self._wake.set()

    async def run(self) -> None:
        """The single consumer. Runs until `stop`.

        One merge is in flight at a time, and nothing behind a halt is started.
        """
        self._running = True
        self._stopped.clear()
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                if self._stopping:
                    return
                if self._resume_asked:
                    self._resume_asked = False
                    if self._halt is not None:
                        await self._resume_current()
                while self._halt is None and self._pending and not self._stopping:
                    await self._attempt(self._pending.popleft())
        finally:
            self._running = False
            self._stopped.set()

    async def stop(self) -> None:
        """End the consumer once whatever is in flight has finished.

        Whatever is still queued stays queued and is never started. Returns as
        soon as the consumer has left `run`, or immediately if none is going —
        a stopped queue stays stopped, so `run` returns at once if called
        after this.
        """
        self._stopping = True
        self._wake.set()
        if self._running:
            await self._stopped.wait()

    # -- one request ------------------------------------------------------

    async def _attempt(self, request: MergeRequest) -> None:
        """Merge one branch, and either gate it on the build or halt."""
        self._current = request
        try:
            result = self._vcs.merge(self._repo, request.branch)
        except VcsError as error:
            # Not a conflict — a branch that does not resolve, or a git that
            # refused for its own reasons. It halts rather than escaping,
            # because an exception out of `run` would end the consumer and take
            # every queued merge with it.
            self._halt_with(Halt(f"{request.ticket_id} cannot be merged", str(error)))
            return
        if not result.clean:
            # Left in progress on purpose: the markers are in the root, which is
            # where a person resolving them would go.
            self._halt_with(_conflict(request, result.conflicted))
            return
        await self._gate(request)

    async def _gate(self, request: MergeRequest) -> None:
        """The build, if there is one, standing between a merge and `on_merged`."""
        if self._build is None:
            self._finish(request)
            return
        result = await asyncio.to_thread(self._build)
        if result.ok:
            self._finish(request)
            return
        self._halt_with(_build_failed(request, result))

    # -- after a person has been at it ------------------------------------

    async def _resume_current(self) -> None:
        """Look at what the repository is in now, and carry on from there.

        Four states, and git is asked which one rather than the caller being
        asked what they did. The order is the order that distinguishes them:
        unmerged paths survive an unfinished resolution, `MERGE_HEAD` survives a
        finished one that was never committed, and once both are gone the only
        question left is whether the branch is in the base branch or not.

        A build-failure halt falls through the same table without needing to be
        told: its merge was committed by git, so nothing is unmerged, no merge
        is in progress, and the branch is an ancestor — which is the row that
        goes straight back to the build gate.
        """
        request = self._current
        if request is None:  # nothing was ever attempted; nothing to resume
            self._halt = None
            return
        try:
            unmerged = self._vcs.unmerged_paths(self._repo)
            if unmerged:
                self._halt_with(_conflict(request, unmerged))
                return
            if self._vcs.merge_in_progress(self._repo):
                self._vcs.commit_merge(self._repo, _merge_message(request))
                await self._gate(request)
                return
            if self._landed(request.branch):
                await self._gate(request)
                return
        except VcsError as error:
            self._halt_with(Halt(f"{request.ticket_id} cannot be merged", str(error)))
            return
        self._abandon(request)

    def _landed(self, branch: str) -> bool:
        """Whether `branch` is in the base branch now.

        A branch that no longer resolves — deleted while the halt was being
        dealt with — reads as gone rather than landed, which puts it on the
        abandoned row where it belongs.
        """
        try:
            return self._vcs.is_ancestor(branch, self._base_branch)
        except VcsError:
            return False

    # -- reporting --------------------------------------------------------

    def _finish(self, request: MergeRequest) -> None:
        """The branch is in, and the build agreed."""
        self._halt = None
        self._current = None
        self._on_merged(request.ticket_id)

    def _abandon(self, request: MergeRequest) -> None:
        """The merge was thrown away by whoever dealt with the halt."""
        self._halt = None
        self._current = None
        self._on_abandoned(request.ticket_id)

    def _halt_with(self, halt: Halt) -> None:
        """Stop at the head and say why. `_current` stays, to resume from."""
        self._halt = halt
        self._on_halt(halt)


# -- pure ------------------------------------------------------------------


def _conflict(request: MergeRequest, paths: tuple[str, ...]) -> Halt:
    """The halt a person opens the repository root to deal with."""
    return Halt(
        reason=f"{request.ticket_id} conflicts with the base branch",
        detail=f"resolve in the repository root: {', '.join(paths)}",
    )


def _build_failed(request: MergeRequest, result: ExecResult) -> Halt:
    """The halt for the failure git cannot see: it merged, and the build broke."""
    what = "timed out" if result.timed_out else f"failed with exit {result.code}"
    return Halt(
        reason=f"{request.ticket_id} merged but the build {what}",
        detail=_tail(_output(result), TAIL_LINES),
    )


def _merge_message(request: MergeRequest) -> str:
    """The message on a merge commit this queue finishes on someone's behalf."""
    return f"Merge {request.branch} ({request.ticket_id})"


def _output(result: ExecResult) -> str:
    """Both streams as one text, since which of them carries the diagnosis is
    the build tool's choice and not something this queue can know."""
    return "\n".join(stream.strip("\n") for stream in (result.stdout, result.stderr) if stream)


def _tail(text: str, lines: int) -> str:
    """The last `lines` lines of `text`, and nothing about which ones matter."""
    kept = text.strip("\n").split("\n")[-lines:]
    return "\n".join(kept)
