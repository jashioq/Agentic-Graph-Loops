"""The concurrency loop that turns a ticket graph into running work.

Layer: workflows. Imports `agl.runtime.dag` and this workflow's `state` and
`models`; nothing else from `agl.core`. `body` is what actually does anything —
worktrees, agents, merges — and this module never learns any of that exists.

Two hazards, both easy to reintroduce.

**A slot is acquired before the graph is asked what is ready, never after.**
Claiming a node and only then waiting for a slot would leave it `CLAIMED` while
nothing runs — a lie to the dashboard, and a deadlock the moment the graph is
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

`claim_next` is used rather than `ready()` followed by `claim()`: reading the
ready set and claiming out of it is one synchronous step, which is the only
thing that stops two passes from claiming the same node. `Dag` guarantees
that; this module only has to call it correctly.
"""

import asyncio
from collections.abc import Awaitable, Callable

from agl.runtime.dag import Dag, NodeId, NodeState
from agl.workflows.tickets.models import Ticket
from agl.workflows.tickets.state import RunState

__all__ = ["StalledGraphError", "bugs_first", "run"]


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


async def run(
    state: RunState,
    body: Callable[[Ticket], Awaitable[None]],
    max_concurrent: int,
    on_error: Callable[[Ticket | None, BaseException], None],
) -> None:
    """Run every ticket's `body` once, at most `max_concurrent` at a time.

    Stops admitting new work the moment `body` raises or `state.halt` is set,
    and waits for whatever is already running before returning. A stalled
    graph — nothing ready, nothing claimed, not complete — is reported through
    `on_error` with `None` in place of a ticket, rather than spinning forever.

    Cancelling the run cancels every in-flight `body` and propagates, the way
    Ctrl-C now reaches a running build. A raising `on_error` is treated the
    same way: it is the error-reporting channel, so a broken one has nowhere
    quieter to go than the exception it raised, but every in-flight `body` is
    still cancelled and awaited before that exception leaves `run` — a
    caller sees a clean stop, never a set of orphaned tasks.

    The scheduler never mutates a ticket's status: `body` does that through
    `state.set_status`, the single writer. The one graph write here is
    `dag.release` on a ticket whose `body` raised, so it is not left `CLAIMED`
    forever; what that means for the ticket's status is the workflow's call.
    """
    dag = state.dag
    slots = asyncio.Semaphore(max_concurrent)
    progress = asyncio.Event()
    tasks: set[asyncio.Task[None]] = set()
    admitting = True

    async def run_one(ticket_id: NodeId) -> None:
        nonlocal admitting
        ticket = state.tickets[ticket_id]
        try:
            await body(ticket)
        except Exception as error:  # noqa: BLE001 - reported, not re-raised
            dag.release(ticket_id)
            admitting = False
            on_error(ticket, error)
        finally:
            slots.release()
            progress.set()

    def should_admit() -> bool:
        return admitting and state.halt is None and not dag.is_complete()

    try:
        while should_admit():
            # The slot comes first and goes straight back when nothing is
            # ready: holding one while waiting on a blocker is how a graph
            # deeper than the cap deadlocks.
            await slots.acquire()
            if not should_admit():
                slots.release()
                break
            node = dag.claim_next()
            if node is not None:
                task = asyncio.create_task(run_one(node))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                continue
            slots.release()
            if dag.is_stalled():
                on_error(None, StalledGraphError(_pending(dag)))
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


def _pending(dag: Dag) -> tuple[NodeId, ...]:
    """Nodes still `PENDING` when the graph stalled, for the error message."""
    return tuple(node for node in dag.nodes() if dag.state(node) is NodeState.PENDING)


def bugs_first(state: RunState) -> Callable[[NodeId], bool]:
    """A `Dag` priority key that puts every ready bug ahead of every ready feature.

    `Dag.ready()` sorts with this key using a stable sort, so ties — bug vs
    bug, feature vs feature — keep insertion order for free; the key only has
    to say which of the two groups a node belongs to.

    A run that keeps generating bugs can leave feature tickets waiting a long
    time even though they became ready first. That is intended: finishing
    what is already open takes priority over opening more.
    """

    def priority(node_id: NodeId) -> bool:
        return not state.tickets[node_id].is_bug

    return priority
