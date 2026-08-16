"""The concurrency loop that turns claimable work into running work.

Layer: runtime. Imports `agl.runtime.dag` for `NodeId` and nothing else. It is
handed a `NodeId` and never learns what the caller keeps under it.

Two hazards, both easy to reintroduce. A slot is acquired *before* the graph is
asked what is ready, never after: claiming first would leave a node `CLAIMED`
while nothing runs, and deadlock a graph deeper than the cap. And every point
the loop resumes after `acquire` re-checks completion, admission and halt — the
task that woke it may have been the last one running.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agl.runtime.dag import NodeId

__all__ = ["Claims", "StalledGraphError", "drive", "run"]


@dataclass(frozen=True)
class Claims:
    """Where a scheduler gets its work, and where it hands it back.

    Four callables rather than a graph, so the caller may derive one fresh on
    every question and react to a state document that changed underneath it.
    """

    next: Callable[[], NodeId | None]
    """Claim one node, atomically, or `None` when nothing is claimable."""

    release: Callable[[NodeId], None]
    """A body raised; give the node back so it is not left claimed forever."""

    complete: Callable[[], bool]
    """Whether there is nothing left to do."""

    stalled: Callable[[], tuple[NodeId, ...] | None]
    """The still-pending ids when the work cannot advance, `None` otherwise."""


class StalledGraphError(Exception):
    """The graph cannot advance: nothing is ready and nothing is claimed.

    A workflow that sees this has a bug upstream; a legally-built `Dag` cannot
    reach it on its own.
    """

    def __init__(self, pending: tuple[NodeId, ...]) -> None:
        super().__init__(f"stalled with {len(pending)} node(s) pending: {', '.join(pending)}")
        self.pending = pending


def _never() -> bool:
    """The default halt predicate: a caller who never halts."""
    return False


async def run(
    claims: Claims,
    body: Callable[[NodeId], Awaitable[None]],
    max_concurrent: int,
    on_error: Callable[[NodeId | None, BaseException], None],
    halted: Callable[[], bool] = _never,
) -> None:
    """Runs every node's `body` once, at most `max_concurrent` at a time.

    Stops admitting the moment `body` raises or `halted()` is true, then waits
    for what is running. Cancellation and a raising `on_error` both cancel and
    await every in-flight task before propagating, so a caller never sees orphans.

    param: body - does the actual work for one node; the scheduler learns nothing of it
    param: on_error - told about a failing body, or a stalled graph with `None` for the node
    param: halted - consulted before admitting more work; the default never halts
    """
    slots = asyncio.Semaphore(max_concurrent)
    progress = asyncio.Event()
    tasks: set[asyncio.Task[None]] = set()
    admitting = True

    async def run_one(node_id: NodeId) -> None:
        nonlocal admitting
        try:
            await body(node_id)
        except Exception as error:  # noqa: BLE001 - reported, not re-raised
            claims.release(node_id)
            admitting = False
            on_error(node_id, error)
        finally:
            slots.release()
            progress.set()

    def should_admit() -> bool:
        return admitting and not halted() and not claims.complete()

    try:
        while should_admit():
            # The slot comes first and goes straight back when nothing is
            # ready: holding one while waiting on a blocker is how a graph
            # deeper than the cap deadlocks.
            await slots.acquire()
            if not should_admit():
                slots.release()
                break
            node = claims.next()
            if node is not None:
                task = asyncio.create_task(run_one(node))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                continue
            slots.release()
            if (pending := claims.stalled()) is not None:
                on_error(None, StalledGraphError(pending))
                break
            # A slot is free but nothing is claimable, so only something in
            # flight can move the graph. `progress` is set from the same
            # `finally` that releases a slot, so this wait cannot be orphaned.
            progress.clear()
            await progress.wait()
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, Exception):
        # Cancellation and a raising `on_error` land here alike: the failure
        # stays loud, but every in-flight task is closed out first.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def drive(
    claims: Claims,
    body: Callable[[NodeId], Awaitable[None]],
    max_concurrent: int,
    on_error: Callable[[NodeId | None, BaseException], None],
    halted: Callable[[], bool] = _never,
) -> None:
    """Re-enters `run` until the graph is complete or a halt outlives a pass.

    A halt still set when a pass returns is one nothing resolved, so `drive`
    returns with the graph incomplete and the caller reads its own halt.
    """
    while not claims.complete():
        await run(claims, body, max_concurrent, on_error, halted)
        if halted():
            return
