"""The merge queue: one merge at a time into a target branch, behind a build.

Layer: runtime. Composes `vcs` and `command` and knows nothing of the workflow
above — it merges and *reports*: `submit` returns the `MergeOutcome`, anything
that did not land goes back through `MergeConfig.resolve` for a `MergeDecision`.

Serializing merges is what lets a build run against the current tip before the
ref is trusted. A bad outcome holds the head of the queue inside `resolve`;
nothing behind it moves. Every `submit` is answered exactly once, `STOPPED` at
worst. Git calls stay on the event loop; only the build is awaited as a real
subprocess, so `stop` can kill the child.
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

    `RETRY` re-reads the repository and carries on from what is there now;
    `ABANDON` drops this request and drains the rest; `STOP` ends the consumer.
    """

    RETRY = "retry"
    ABANDON = "abandon"
    STOP = "stop"


@dataclass(frozen=True)
class MergeRequest:
    """One branch asking to land somewhere. `key` is the caller's own name for it."""

    key: str
    branch: str  # source
    target: str  # branch to merge into
    cwd: Path  # working tree holding `target`


@dataclass(frozen=True)
class MergeOutcome:
    """What happened to one request.

    `build` is the whole `ExecResult`, never a slice: which lines of a failed
    build matter is language-specific, so the caller decides what a person reads.
    """

    key: str
    status: MergeStatus
    conflicted: tuple[str, ...] = ()
    build: ExecResult | None = None
    error: str = ""


type Build = Callable[[], Awaitable[ExecResult]]
type Resolve = Callable[[MergeOutcome], Awaitable[MergeDecision]]


async def _stop(outcome: MergeOutcome) -> MergeDecision:
    """The default `resolve`: any outcome that did not land ends the queue."""
    return MergeDecision.STOP


@dataclass(frozen=True)
class MergeConfig:
    """The two things a queue is given about the run it is serving.

    param: build - a callable, so the queue need not know what a build is; `None` = no gate
    param: resolve - what an outcome that did not land means; defaults to ending the queue
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
"""The config a queue gets when it is not given one."""


class MergeQueue:
    """The single consumer that merges branches into their targets, in order."""

    def __init__(self, vcs: Vcs, config: MergeConfig = _DEFAULT) -> None:
        """Wire the queue to git and to this run's build gate and policy.

        No fixed repo or base branch: each request carries its own `cwd` and
        `target`, so one queue serializes merges into any number of trees.
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
        """Queues a branch to be merged and waits for what became of it.

        return: MergeOutcome - `STOPPED` at once if the queue has already stopped
        """
        if self._stopping:
            return MergeOutcome(key=request.key, status=MergeStatus.STOPPED)
        job = _Job(request, asyncio.get_running_loop().create_future())
        self._pending.append(job)
        self._wake.set()
        return await job.future

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """Runs the consumer for the block, stopping and joining it on the way out.

        An exception out of the consumer — a `resolve` that raised — surfaces here.
        """
        consumer = asyncio.create_task(self._consume())
        try:
            yield
        finally:
            await self.stop()
            await consumer

    async def stop(self) -> None:
        """Ends the consumer, cancelling what is in flight, answering the rest `STOPPED`.

        A committed merge is not unwound; a running build is cancelled, killing
        the child. Idempotent — a stopped queue stays stopped.
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

        Whatever ends it, the `finally` answers everything outstanding.
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
        """Runs one request as its own task, so `stop` can cancel it.

        return: bool - `False` when `stop` cancelled it, so `_consume` returns at once
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
        """Takes one request as far as it goes, asking about anything but a clean landing.

        The loop is the hold: while `resolve` is awaited, nothing behind moves.
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
                # Not a conflict — git refused outright.
                return _vcs_error(request, error)
            if not result.clean:
                # Left in progress on purpose: the markers are in `cwd`.
                return _conflict(request, result.conflicted)
            return await self._gate(request)
        except Exception as error:  # noqa: BLE001 - reported, not re-raised
            # Backstop: an exception escaping here would end the single consumer
            # and take every queued merge with it.
            return _error(request, error)

    async def _gate(self, request: MergeRequest) -> MergeOutcome:
        """The build, if there is one, standing between a merge and `MERGED`.

        Awaited directly, not threaded, so cancelling this await kills the child.
        """
        if self._config.build is None:
            return _merged(request)
        result = await self._config.build()
        if result.ok:
            return _merged(request)
        return _build_failed(request, result)

    # -- after a person has been at it ------------------------------------

    async def _reinspect(self, request: MergeRequest) -> MergeOutcome:
        """Asks git what state the repository is in now, and carries on from there.

        Four rows, in the order that distinguishes them: unmerged paths, then a
        merge in progress, then whether the branch already landed, then abandoned.
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
        """Whether `branch` is in `target` now. A branch that no longer resolves is not."""
        try:
            return self._vcs.is_ancestor(branch, target)
        except VcsError:
            return False

    # -- reporting --------------------------------------------------------

    def _settle_all(self) -> None:
        """Answers everything still outstanding with `STOPPED`, so no `submit` hangs."""
        jobs = list(self._pending)
        self._pending.clear()
        if self._current is not None:
            jobs.append(self._current)
            self._current = None
        for job in jobs:
            _settle(job, MergeOutcome(key=job.request.key, status=MergeStatus.STOPPED))


# -- pure ------------------------------------------------------------------


def _settle(job: _Job, outcome: MergeOutcome) -> None:
    """Hands one outcome back, unless the waiter is already gone."""
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
