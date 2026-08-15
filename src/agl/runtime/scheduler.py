"""The concurrency loop that turns claimable work into running work.

Layer: runtime. Imports `agl.runtime.dag` for its `NodeId` alias and nothing
else — not a workflow, not a core connector. `body` is what actually does
anything — worktrees, agents, merges — and this module never learns any of that
exists; it is handed a `NodeId` and the caller looks up whatever it keeps under
that key.

`halted` is the one other thing a workflow supplies: a predicate the loop
consults before admitting more work. What a halt *is* — a merge conflict, a
failed build, a person to ask — is the workflow's knowledge, and none of it
reaches here. The default never halts, so a caller with no notion of halting
gets a loop that stops only on completion, a failing body, or a stall.

Two hazards, both easy to reintroduce.

**A slot is acquired before the graph is asked what is ready, never after.**
Claiming a node and only then waiting for a slot would leave it `CLAIMED` while
nothing runs — a lie to any dashboard, and a deadlock the moment the graph is
deeper than the cap, because the held slot would belong to a node that cannot
start. A slot acquired with nothing to claim is handed straight back before the
loop waits again.

**Waiting on the semaphore is the one place the loop suspends without looking
at the graph.** The task that wakes it by releasing a slot can be the last one
running, so every point the loop resumes after `acquire` re-checks completion,
admission and halt before trusting what it saw when it went to sleep. The
other wait — a slot is free but nothing is claimable — is `progress`, set from
the same `finally` that releases a slot, so a task finishing is what wakes the
loop back up on both waits, and neither can be left asleep by the last node
completing underneath it.

`claims.next()` is one call rather than "what is ready" followed by "claim this
one": reading the ready set and claiming out of it has to be a single
synchronous step, which is the only thing that stops two passes from claiming
the same node. The caller guarantees that; this module only has to call it once
and trust the answer.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agl.runtime.dag import NodeId

__all__ = ["Claims", "StalledGraphError", "drive", "run"]


@dataclass(frozen=True)
class Claims:
    """Where a scheduler gets its work, and where it hands it back.

    Four callables rather than a graph, because the caller may derive its graph
    fresh on every question — which is what lets a run react to a state document
    that changed underneath it.
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

    A workflow that sees this has a bug upstream — the graph only reports the
    condition, since a legally-built `Dag` cannot reach it on its own: a node
    with a real cycle of unsatisfied blockers is what `add_edge` refuses to
    create in the first place.
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
    """Run every node's `body` once, at most `max_concurrent` at a time.

    Stops admitting new work the moment `body` raises or `halted()` becomes
    true, and waits for whatever is already running before returning. A stalled
    graph — nothing ready, nothing claimed, not complete — is reported through
    `on_error` with `None` in place of a node, rather than spinning forever.

    Cancelling the run cancels every in-flight `body` and propagates, the way
    Ctrl-C now reaches a running build. A raising `on_error` is treated the
    same way: it is the error-reporting channel, so a broken one has nowhere
    quieter to go than the exception it raised, but every in-flight `body` is
    still cancelled and awaited before that exception leaves `run` — a
    caller sees a clean stop, never a set of orphaned tasks.

    The scheduler never writes anything a workflow keeps about a node: that is
    `body`'s job, through whatever single writer the workflow has. The one
    write here is `claims.release` on a node whose `body` raised, so it is not
    left claimed forever; what that means for the workflow is its call.
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
            # Nothing is claimable but a slot is free, so the graph can only
            # move once something in flight finishes. `progress` is set from
            # the same `finally` that releases a slot, so that completion is
            # what wakes this wait too — never left asleep with nothing left
            # to signal it.
            progress.clear()
            await progress.wait()
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, Exception):
        # Cancellation and a raising `on_error` land here alike: both call
        # sites let their exception through unguarded, so a broken handler
        # takes this path exactly as Ctrl-C does. Swallowing it would let a
        # broken `on_error` masquerade as a run with no errors; this keeps
        # the failure loud while still closing out every in-flight task
        # first, so what a caller sees is a clean stop, never an orphan.
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
    """Re-enter `run` until the graph is complete or a halt outlives a pass.

    A node whose `body` is parked waiting on a person does not return until
    somebody deals with it, so `run` cannot return on its own while one is
    stuck there — whatever the workflow arranged to unstick it is what lets
    the pass finish. This loop only has to notice work a pass returned early
    from: a resolved halt that left something newly ready to claim.

    A halt still set when a pass returns is one nothing resolved, so going
    round again would admit nothing and spin. That is where `drive` returns
    with the graph incomplete, and the caller reads its own halt to find out
    why.
    """
    while not claims.complete():
        await run(claims, body, max_concurrent, on_error, halted)
        if halted():
            return
