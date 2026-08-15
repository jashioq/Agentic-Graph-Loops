"""The merge queue: one merge at a time into a target branch, behind a build.

Layer: runtime. Composes `vcs` and `command` and knows nothing about the
workflow above it — not what a ticket is, not what a halt is, not what any of
its outcomes ought to mean. It does the merge and *reports*: `submit` returns
the `MergeOutcome`, and anything that did not land is put back to the workflow
through `MergeConfig.resolve`, which answers with a `MergeDecision`.

**Why a queue.** Concurrent merges into one branch are concurrent writers to one
ref, with real failure modes — `index.lock` contention, a lost update, a
half-applied merge when two processes race. A single consumer makes those
impossible rather than rare. The second reason matters more: serialized merges
each integrate against the *current* tip, so a build can run before the ref is
trusted. That catches the failure git cannot see — one ticket renames a method,
another adds a caller, both merge cleanly, nothing conflicts, the build breaks.

**Where merges happen.** Wherever the request's `cwd` says: the repository root
for work merging into the run's base branch, or another kept-alive worktree for
work merging into the branch that tree holds. Never a dedicated merge worktree:
git refuses to check out a branch another tree already holds, so a `_merge` tree
could never hold a branch that is checked out somewhere else. `cwd` is always a
tree that already has `target` checked out, untouched either way by the source
worktrees, and a conflict leaves the markers exactly where a person would go to
resolve them.

**A bad outcome stops at the head.** The consumer holds the line inside
`resolve` — requests behind it are still accepted, none are processed, and
everything behind a stuck merge would otherwise integrate against a base that is
about to change. `RETRY` re-reads git and carries on from whatever is there,
`ABANDON` drops the request and drains on, `STOP` ends the consumer. The default
`resolve` says `STOP`, so a caller that wires nothing up never hangs.

**Nothing is left waiting.** Every `submit` is answered exactly once. A consumer
that ends — through `stop`, through a `STOP` decision, or because `resolve`
itself raised — settles the request in flight and everything still queued with
`STOPPED` on its way out, and a `submit` to an already-stopped queue answers
`STOPPED` without queueing anything.

**What runs where.** Git calls stay on the event loop: they are milliseconds, and
leaving them there means no other task can interleave a git command between this
one's merge and the questions it asks about the result. The build is the one
thing here that runs for minutes, so it is a coroutine built on
`command.run_async` — a real subprocess handle, awaited directly rather than
run in a thread, so `stop` can cancel it and kill the child instead of waiting
out however long is left.
"""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from agl.core.command import ExecResult
from agl.core.vcs import Vcs, VcsError

__all__ = [
    "Build",
    "MergeConfig",
    "MergeDecision",
    "MergeOutcome",
    "MergeQueue",
    "MergeRequest",
    "MergeStatus",
    "Resolve",
]


class MergeStatus(Enum):
    """How one request ended. Facts about git and the build, not verdicts."""

    MERGED = "merged"
    CONFLICT = "conflict"
    BUILD_FAILED = "build_failed"
    VCS_ERROR = "vcs_error"
    ERROR = "error"
    ABANDONED = "abandoned"
    STOPPED = "stopped"


class MergeDecision(Enum):
    """What the workflow wants done about an outcome that did not land.

    `RETRY` re-reads the repository and carries on from whatever state it is
    in now — the queue asks git what happened rather than being told. `ABANDON`
    gives up on this request and drains what is behind it. `STOP` ends the
    consumer, and every request still outstanding comes back `STOPPED`.
    """

    RETRY = "retry"
    ABANDON = "abandon"
    STOP = "stop"


@dataclass(frozen=True)
class MergeRequest:
    """One branch asking to land somewhere.

    `key` is the caller's own name for the work; the queue only ever hands it
    back on the outcome.
    """

    key: str
    branch: str  # source
    target: str  # branch to merge into
    cwd: Path  # working tree holding `target`


@dataclass(frozen=True)
class MergeOutcome:
    """What happened to one request.

    `build` is the whole `ExecResult`, never a slice of it: which lines of a
    failed build matter is language-specific — a Kotlin error sits early under a
    stack trace, a Rust one is structured and late, a bundler dumps module paths
    — so the caller that knows what it is running decides what a person reads.
    """

    key: str
    status: MergeStatus
    conflicted: tuple[str, ...] = ()
    build: ExecResult | None = None
    error: str = ""


type Build = Callable[[], Awaitable[ExecResult]]
type Resolve = Callable[[MergeOutcome], Awaitable[MergeDecision]]


async def _stop(outcome: MergeOutcome) -> MergeDecision:
    """The default `resolve`: any outcome that did not land ends the queue.

    Safe rather than clever. A caller that has not said what a conflict means
    to it has nobody to ask, and a queue that guessed `RETRY` would re-read the
    same unresolved repository forever.
    """
    return MergeDecision.STOP


@dataclass(frozen=True)
class MergeConfig:
    """The two things a queue is given about the run it is serving.

    `build` is a callable rather than a command line so the queue never has to
    know what a build looks like; `None` means no build gate. In production it
    closes over the project's command and calls `command.run_async(...,
    check=False, timeout=...)` — `check=False` because a failing build is the
    answer to the question, not an error.
    """

    build: Build | None = None
    resolve: Resolve = _stop


@dataclass(frozen=True)
class _Job:
    """A queued request and the `submit` waiting on its outcome."""

    request: MergeRequest
    future: asyncio.Future[MergeOutcome]


_SETTLED = (MergeStatus.MERGED, MergeStatus.ABANDONED)
"""The outcomes nobody is asked about: the work is in, or it is given up on."""

_DEFAULT = MergeConfig()
"""The config a queue gets when it is not given one. A module-level singleton
rather than a call in the signature, which `MergeConfig` being frozen allows."""


class MergeQueue:
    """The single consumer that merges branches into their targets, in order."""

    def __init__(self, vcs: Vcs, config: MergeConfig = _DEFAULT) -> None:
        """Wire the queue to git and to this run's build gate and policy.

        There is no fixed `repo` or `base_branch` here: each `MergeRequest`
        carries the `cwd` and `target` it merges into, so one queue serializes
        merges into any number of trees.
        """
        self._vcs = vcs
        self._config = config

        self._pending: deque[_Job] = deque()
        self._current: _Job | None = None  # the one `resolve` is being asked about
        self._stopping = False
        self._running = False
        # The task running the current request, so `stop` has something to
        # cancel instead of only something to wait out.
        self._inflight: asyncio.Task[None] | None = None
        # One event for every reason the consumer might have to look again: new
        # work or a stop. Set by the sync callers, cleared only by the consumer,
        # so a wake-up cannot be lost between the two.
        self._wake = asyncio.Event()
        self._stopped = asyncio.Event()

    # -- what the workflow calls ------------------------------------------

    async def submit(self, request: MergeRequest) -> MergeOutcome:
        """Queue a branch to be merged and wait for what became of it.

        Accepted while the head of the queue is stuck, like any other request;
        it simply does not start until the one in front of it is done with. A
        queue that has already stopped answers `STOPPED` at once rather than
        taking work it will never do.
        """
        if self._stopping:
            return MergeOutcome(key=request.key, status=MergeStatus.STOPPED)
        job = _Job(request, asyncio.get_running_loop().create_future())
        self._pending.append(job)
        self._wake.set()
        return await job.future

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """Run the consumer for the duration of the block, stopping and joining
        it on the way out whichever way the block leaves — so an exception out
        of the body leaves nothing merging behind it, and one out of the
        consumer (a `resolve` that raised) reaches the caller here.
        """
        consumer = asyncio.create_task(self._consume())
        try:
            yield
        finally:
            await self.stop()
            await consumer

    async def stop(self) -> None:
        """End the consumer, cancelling whatever is in flight rather than
        waiting it out, and answer every outstanding `submit` with `STOPPED`.

        Nothing still queued is started. A merge already committed by git is not
        unwound; a build still running is cancelled, which is the whole reason
        `_gate` holds a real process handle instead of a thread nothing can
        reach — killing the child leaves the repository exactly as git left it,
        the same state a person would find after a conflict. Returns once the
        consumer has left `_consume`, or immediately if none is going; a stopped
        queue stays stopped, so a later `running()` returns at once.
        """
        self._stopping = True
        self._wake.set()
        if self._inflight is not None:
            self._inflight.cancel()
        if self._running:
            await self._stopped.wait()
        else:
            self._settle_all()

    # -- the consumer ------------------------------------------------------

    async def _consume(self) -> None:
        """The single consumer: one merge in flight at a time, until stopped.

        Whatever ends it — a stop, a `STOP` decision, or a `resolve` that
        raised — the `finally` answers everything outstanding on the way out.
        """
        self._running = True
        self._stopped.clear()
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                if self._stopping:
                    return
                while self._pending and not self._stopping:
                    if not await self._perform(self._process(self._pending.popleft())):
                        return
        finally:
            self._running = False
            self._settle_all()
            self._stopped.set()

    async def _perform(self, attempt: Coroutine[Any, Any, None]) -> bool:
        """Run one request as its own task, so `stop` can reach in and cancel it
        instead of only being able to wait it out.

        Returns `False` when `stop` cancelled it out from under the loop —
        `_consume` returns immediately rather than looping back to look at
        state a cancelled attempt never finished updating.
        """
        task = asyncio.ensure_future(attempt)
        self._inflight = task
        try:
            await task
            return True
        except asyncio.CancelledError:
            return False
        finally:
            self._inflight = None

    # -- one request ------------------------------------------------------

    async def _process(self, job: _Job) -> None:
        """Take one request as far as it goes, asking about anything but a
        clean landing.

        The loop is the hold: while `resolve` is awaited the consumer is in
        here, so nothing behind this request moves. A `STOP` leaves the job on
        `_current` for `_settle_all` to answer, rather than answering it twice.
        """
        self._current = job
        outcome = await self._attempt(job.request)
        while outcome.status not in _SETTLED:
            decision = await self._config.resolve(outcome)
            if decision is MergeDecision.STOP:
                self._stopping = True
                self._wake.set()
                return
            if decision is MergeDecision.ABANDON:
                outcome = _abandoned(job.request)
                break
            outcome = await self._reinspect(job.request)
        self._current = None
        _settle(job, outcome)

    async def _attempt(self, request: MergeRequest) -> MergeOutcome:
        """Merge one branch, and either gate it on the build or report why not."""
        try:
            try:
                result = self._vcs.merge(request.cwd, request.branch)
            except VcsError as error:
                # Not a conflict — a branch that does not resolve, or a git
                # that refused for its own reasons.
                return _vcs_error(request, error)
            if not result.clean:
                # Left in progress on purpose: the markers are in `cwd`, which
                # is where a person resolving them would go.
                return _conflict(request, result.conflicted)
            return await self._gate(request)
        except Exception as error:  # noqa: BLE001 - reported, not re-raised
            # The backstop under the `VcsError` handling above: a build gate
            # that raises something else entirely — a missing executable, a bad
            # config. An exception escaping here would end the single consumer
            # and take every queued merge with it, so it is reported as an
            # outcome like any other.
            return _error(request, error)

    async def _gate(self, request: MergeRequest) -> MergeOutcome:
        """The build, if there is one, standing between a merge and `MERGED`.

        Awaited directly rather than run in a thread: `build` closes over
        `command.run_async`, which holds a real subprocess handle, so `stop`
        cancelling this await kills the child instead of leaving a thread that
        cannot be reached running behind it.
        """
        if self._config.build is None:
            return _merged(request)
        result = await self._config.build()
        if result.ok:
            return _merged(request)
        return _build_failed(request, result)

    # -- after a person has been at it ------------------------------------

    async def _reinspect(self, request: MergeRequest) -> MergeOutcome:
        """Look at what the repository is in now, and carry on from there.

        Four states, and git is asked which one rather than the caller being
        asked what they did. The order is the order that distinguishes them:
        unmerged paths survive an unfinished resolution, `MERGE_HEAD` survives a
        finished one that was never committed, and once both are gone the only
        question left is whether the branch is in the target or not.

        A build failure falls through the same table without needing to be
        told: its merge was committed by git, so nothing is unmerged, no merge
        is in progress, and the branch is an ancestor — which is the row that
        goes straight back to the build gate.
        """
        try:
            try:
                unmerged = self._vcs.unmerged_paths(request.cwd)
                if unmerged:
                    return _conflict(request, unmerged)
                if self._vcs.merge_in_progress(request.cwd):
                    self._vcs.commit_merge(request.cwd, _merge_message(request))
                    return await self._gate(request)
                if self._landed(request.branch, request.target):
                    return await self._gate(request)
            except VcsError as error:
                return _vcs_error(request, error)
            return _abandoned(request)
        except Exception as error:  # noqa: BLE001 - reported, not re-raised
            return _error(request, error)  # same backstop as `_attempt`

    def _landed(self, branch: str, target: str) -> bool:
        """Whether `branch` is in `target` now.

        A branch that no longer resolves — deleted while the outcome was being
        dealt with — reads as gone rather than landed, which puts it on the
        abandoned row where it belongs.
        """
        try:
            return self._vcs.is_ancestor(branch, target)
        except VcsError:
            return False

    # -- reporting --------------------------------------------------------

    def _settle_all(self) -> None:
        """Answer everything still outstanding with `STOPPED`.

        The one thing standing between a consumer that ends and a caller parked
        in `submit` forever. Called from the consumer's `finally` and from a
        `stop` on a queue that was never consuming.
        """
        jobs = list(self._pending)
        self._pending.clear()
        if self._current is not None:
            jobs.append(self._current)
            self._current = None
        for job in jobs:
            _settle(job, MergeOutcome(key=job.request.key, status=MergeStatus.STOPPED))


# -- pure ------------------------------------------------------------------


def _settle(job: _Job, outcome: MergeOutcome) -> None:
    """Hand one outcome back, unless the waiter is already gone.

    A `submit` whose task was cancelled leaves a cancelled future behind, and
    resolving it would raise on a path that has nowhere to report.
    """
    if not job.future.done():
        job.future.set_result(outcome)


def _merged(request: MergeRequest) -> MergeOutcome:
    """The branch is in, and the build agreed."""
    return MergeOutcome(key=request.key, status=MergeStatus.MERGED)


def _abandoned(request: MergeRequest) -> MergeOutcome:
    """The merge was given up on — aborted in the repository, or by decision."""
    return MergeOutcome(key=request.key, status=MergeStatus.ABANDONED)


def _conflict(request: MergeRequest, paths: tuple[str, ...]) -> MergeOutcome:
    """A merge a person opens `cwd` to deal with, naming what is unmerged."""
    return MergeOutcome(key=request.key, status=MergeStatus.CONFLICT, conflicted=paths)


def _build_failed(request: MergeRequest, result: ExecResult) -> MergeOutcome:
    """The failure git cannot see: it merged, and the build broke."""
    return MergeOutcome(key=request.key, status=MergeStatus.BUILD_FAILED, build=result)


def _vcs_error(request: MergeRequest, error: VcsError) -> MergeOutcome:
    """Git refused outright: a branch that does not resolve, or worse."""
    return MergeOutcome(key=request.key, status=MergeStatus.VCS_ERROR, error=str(error))


def _error(request: MergeRequest, error: Exception) -> MergeOutcome:
    """Anything that escaped as an exception, reported rather than raised."""
    return MergeOutcome(
        key=request.key,
        status=MergeStatus.ERROR,
        error=f"{type(error).__name__}: {error}",
    )


def _merge_message(request: MergeRequest) -> str:
    """The message on a merge commit this queue finishes on someone's behalf."""
    return f"Merge {request.branch} ({request.key})"
