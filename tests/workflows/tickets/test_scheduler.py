"""The scheduler: a semaphore loop over a ticket graph, driven by hand-held gates.

Nothing here sleeps. Every body waits on an `asyncio.Event` the test owns, so
the test decides exactly when each finishes and can inspect the whole system
while some are held. Every run is wrapped in a timeout so a deadlock fails the
test instead of hanging the suite.

The graph shapes mirror `tests/integration/test_concurrency.py::run_graph`,
the reference this scheduler was built from: `wide_state` is five independent
roots, two joins over them, and one join over those — wider than any cap used
against it, so the cap is what holds work back rather than the graph.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import pytest

from agl.runtime.dag import Dag, NodeId, NodeState
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.scheduler import StalledGraphError, bugs_first, run
from agl.workflows.tickets.state import Halt, RunState, add_tickets, file_bugs

TIMEOUT = 10.0

ROOTS = ("T1", "T2", "T3", "T4", "T5")
BLOCKED: dict[str, tuple[str, ...]] = {
    "T6": ("T1", "T2"),
    "T7": ("T3", "T4"),
    "T8": ("T6", "T7"),
}
CAP = 3


# -- building runs ----------------------------------------------------------


def feature(ticket_id: str, *blocked_by: str) -> Ticket:
    return Ticket(
        id=ticket_id,
        title=f"Do {ticket_id}",
        status=Status.PENDING,
        deliverables=(f"{ticket_id}.py",),
        blocked_by=blocked_by,
    )


def bug(ticket_id: str, parent: str, *blocked_by: str) -> Ticket:
    return Ticket(
        id=ticket_id,
        title=f"Fix {ticket_id}",
        status=Status.PENDING,
        deliverables=("the finding",),
        blocked_by=blocked_by,
        parent=parent,
    )


def build_state(
    tickets: Sequence[Ticket],
    *,
    priority: Callable[[RunState], Callable[[NodeId], object]] | None = None,
) -> RunState:
    """A run holding `tickets`, with their `blocked_by` edges already built."""
    state = RunState(label="wide", base_branch="main", dag=Dag(), tickets={})
    if priority is not None:
        state.dag = Dag(priority=priority(state))
    add_tickets(state, None, tickets)
    return state


def wide_state() -> RunState:
    """Five independent roots, two joins over them, and one join over those."""
    tickets = [feature(node_id) for node_id in ROOTS]
    tickets += [feature(node_id, *blockers) for node_id, blockers in BLOCKED.items()]
    return build_state(tickets)


# -- watching a run, without sleeping ----------------------------------------


class Observer:
    """What ran, in what order, and the only way anything here waits."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[str] = []
        self.running: set[str] = set()
        self.peak = 0
        self._changed = asyncio.Event()

    def enter(self, ticket_id: str) -> None:
        self.started.append(ticket_id)
        self.running.add(ticket_id)
        self.peak = max(self.peak, len(self.running))
        self._changed.set()

    def leave(self, ticket_id: str) -> None:
        self.running.discard(ticket_id)
        self.finished.append(ticket_id)
        self._changed.set()

    async def until(
        self, predicate: Callable[[], bool], guard: "asyncio.Task[None] | None" = None
    ) -> None:
        """Wait for `predicate`. `guard` is the run, so its failure surfaces here."""
        async with asyncio.timeout(TIMEOUT):
            while not predicate():
                if guard is not None and guard.done():
                    guard.result()
                    raise AssertionError("the run finished before the wait was satisfied")
                self._changed.clear()
                await self._changed.wait()


class Gates:
    """One event per ticket. A ticket's body does not finish until its gate opens."""

    def __init__(self, ticket_ids: Sequence[str]) -> None:
        self._gates = {ticket_id: asyncio.Event() for ticket_id in ticket_ids}

    def __getitem__(self, ticket_id: str) -> asyncio.Event:
        return self._gates[ticket_id]

    def open(self, *ticket_ids: str) -> None:
        for ticket_id in ticket_ids or tuple(self._gates):
            self._gates[ticket_id].set()


class Errors:
    """What `on_error` was called with, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[Ticket | None, BaseException]] = []

    def __call__(self, ticket: Ticket | None, error: BaseException) -> None:
        self.calls.append((ticket, error))


def gated_body(
    state: RunState, observer: Observer, gates: Gates
) -> Callable[[Ticket], Awaitable[None]]:
    """A body that waits for its ticket's gate, then completes the graph node.

    Stands in for the real body's whole pipeline down to the merge that calls
    `dag.complete` on the workflow's behalf; this scheduler test only needs
    that the node moves, not how.
    """

    async def body(ticket: Ticket) -> None:
        observer.enter(ticket.id)
        try:
            await gates[ticket.id].wait()
            state.dag.complete(ticket.id)
        finally:
            observer.leave(ticket.id)

    return body


async def run_to_completion(state: RunState, cap: int) -> Observer:
    """Run every gate open from the start, and hand back what happened."""
    observer = Observer()
    gates = Gates(state.dag.nodes())
    gates.open()
    errors = Errors()
    async with asyncio.timeout(TIMEOUT):
        await run(state, gated_body(state, observer, gates), cap, errors)
    assert errors.calls == []
    return observer


# -- the cap ------------------------------------------------------------


@dataclass(frozen=True)
class Ran:
    state: RunState
    observer: Observer
    errors: Errors
    held_running: tuple[str, ...]
    held_ready: tuple[str, ...]
    after_one_release_running: tuple[str, ...]
    after_one_release_started: tuple[str, ...]


async def drive_wide(cap: int) -> Ran:
    """Hold `cap` nodes running, look around, release one, look again, finish."""
    state = wide_state()
    observer, errors = Observer(), Errors()
    gates = Gates(state.dag.nodes())
    task = asyncio.create_task(run(state, gated_body(state, observer, gates), cap, errors))
    await observer.until(lambda: len(observer.running) == cap, task)

    held_running = tuple(sorted(observer.running))
    held_ready = state.dag.ready()

    gates.open(held_running[0])
    await observer.until(lambda: len(observer.started) == cap + 1, task)
    after_one_release_running = tuple(sorted(observer.running))
    after_one_release_started = tuple(observer.started)

    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await task

    return Ran(
        state=state,
        observer=observer,
        errors=errors,
        held_running=held_running,
        held_ready=held_ready,
        after_one_release_running=after_one_release_running,
        after_one_release_started=after_one_release_started,
    )


async def test_the_cap_is_reached() -> None:
    ran = await drive_wide(CAP)
    assert ran.observer.peak == CAP


async def test_the_cap_is_never_exceeded() -> None:
    ran = await drive_wide(CAP)
    assert ran.observer.peak <= CAP
    assert len(ran.held_running) == CAP


async def test_the_cap_is_what_holds_them_back() -> None:
    ran = await drive_wide(CAP)
    # Nodes were ready and unstarted while the cap was saturated: the graph
    # was not what was holding them back.
    assert ran.held_ready
    assert set(ran.held_ready).isdisjoint(ran.held_running)
    assert set(ran.held_ready) <= set(ROOTS)


async def test_releasing_one_admits_exactly_one_more() -> None:
    ran = await drive_wide(CAP)
    assert len(ran.after_one_release_started) == CAP + 1
    assert len(ran.after_one_release_running) == CAP
    assert ran.held_running[0] not in ran.after_one_release_running


async def test_every_node_ran_exactly_once() -> None:
    ran = await drive_wide(CAP)
    assert sorted(ran.observer.started) == sorted(ran.state.dag.nodes())
    assert sorted(ran.observer.finished) == sorted(ran.state.dag.nodes())


async def test_the_graph_completes_and_the_run_returns() -> None:
    ran = await drive_wide(CAP)
    assert ran.state.dag.is_complete() is True
    assert ran.observer.running == set()
    assert ran.errors.calls == []


async def test_every_node_runs_exactly_once_even_with_simultaneous_completions() -> None:
    state = wide_state()
    observer, errors = Observer(), Errors()
    gates = Gates(state.dag.nodes())
    task = asyncio.create_task(run(state, gated_body(state, observer, gates), CAP, errors))
    await observer.until(lambda: len(observer.running) == CAP, task)

    # Releasing every held gate at once: several bodies finish in the same
    # synchronous burst, rather than one at a time.
    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await task

    assert errors.calls == []
    for node in state.dag.nodes():
        assert observer.started.count(node) == 1
    assert sorted(observer.finished) == sorted(state.dag.nodes())
    assert state.dag.is_complete() is True


# -- the graph, under contention -----------------------------------------


async def test_no_node_starts_before_its_blockers_have_finished() -> None:
    state = wide_state()
    observer = await run_to_completion(state, CAP)

    for node, blockers in BLOCKED.items():
        for blocker in blockers:
            assert observer.finished.index(blocker) < observer.started.index(node)


async def test_a_cap_above_the_ready_set_runs_everything_at_once() -> None:
    state = build_state([feature(node_id) for node_id in ROOTS])
    observer, errors = Observer(), Errors()
    gates = Gates(state.dag.nodes())
    task = asyncio.create_task(run(state, gated_body(state, observer, gates), 8, errors))

    await observer.until(lambda: len(observer.running) == len(ROOTS), task)
    assert observer.peak == len(ROOTS)

    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await task
    assert state.dag.is_complete() is True
    assert errors.calls == []


async def test_a_cap_of_one_is_fully_serial_and_topologically_ordered() -> None:
    state = wide_state()
    observer = await run_to_completion(state, 1)

    assert observer.peak == 1
    order = observer.finished
    assert sorted(order) == sorted(state.dag.nodes())
    for node, blockers in BLOCKED.items():
        for blocker in blockers:
            assert order.index(blocker) < order.index(node)


async def test_a_blocked_node_never_occupies_a_slot() -> None:
    # A cap of two with a join deeper than that: a scheduler that claimed a
    # slot before checking readiness would hold both slots on the roots and
    # then deadlock waiting for the join, since nothing would ever free one.
    tickets = [feature("A"), feature("B"), feature("C", "A", "B"), feature("D", "C")]
    state = build_state(tickets)
    observer = await run_to_completion(state, 2)

    assert state.dag.is_complete() is True
    assert observer.peak <= 2
    assert sorted(observer.finished) == ["A", "B", "C", "D"]


# -- priority: bugs first --------------------------------------------------


def test_bugs_first_puts_ready_bugs_ahead_of_ready_features() -> None:
    tickets = [feature("F1"), feature("F2"), bug("B1", "F1"), feature("F3"), bug("B2", "F1")]
    state = build_state(tickets, priority=bugs_first)
    assert state.dag.ready() == ("B1", "B2", "F1", "F2", "F3")


def test_bugs_first_preserves_insertion_order_within_each_group() -> None:
    tickets = [feature("F2"), bug("B2", "F2"), feature("F1"), bug("B1", "F1")]
    state = build_state(tickets, priority=bugs_first)
    assert state.dag.ready() == ("B2", "B1", "F2", "F1")


# -- the runtime mutation: a bug filed mid-run -----------------------------


async def test_a_bug_filed_mid_run_reorders_the_bugs_before_the_released_parent() -> None:
    state = build_state([feature("P")])
    observer, errors = Observer(), Errors()
    gates = Gates(("P", "B1", "B2"))
    filed = asyncio.Event()

    async def body(ticket: Ticket) -> None:
        observer.enter(ticket.id)
        try:
            if ticket.id == "P" and not filed.is_set():
                # The parent's first pass: findings come back, so it files
                # bugs against itself and returns without completing. The
                # node is already `PENDING` again by the time this returns —
                # `file_bugs` releases it, following `state`'s ordering:
                # nodes, then edges, then the release.
                file_bugs(state, None, "P", [bug("B1", "P"), bug("B2", "P")])
                filed.set()
                return
            await gates[ticket.id].wait()
            state.dag.complete(ticket.id)
        finally:
            observer.leave(ticket.id)

    task = asyncio.create_task(run(state, body, 2, errors))
    await observer.until(lambda: filed.is_set(), task)

    # The parent is blocked by the bugs it just filed; only the bugs are ready.
    assert state.dag.state("P") is NodeState.PENDING
    assert "P" not in state.dag.ready()
    await observer.until(lambda: observer.running == {"B1", "B2"}, task)

    gates.open("B1", "B2")
    # Both bugs run to completion before the parent becomes ready again. `P`
    # is already in `finished` from its first, incomplete pass.
    await observer.until(lambda: observer.started.count("P") == 2, task)
    assert sorted(observer.finished) == ["B1", "B2", "P"]
    assert observer.started == ["P", "B1", "B2", "P"] or observer.started == ["P", "B2", "B1", "P"]

    gates.open("P")
    async with asyncio.timeout(TIMEOUT):
        await task

    assert errors.calls == []
    assert state.dag.is_complete() is True


# -- halting ----------------------------------------------------------------


async def test_halt_set_mid_run_stops_admission_and_lets_inflight_finish() -> None:
    state = build_state([feature("T1"), feature("T2"), feature("T3")])
    observer, errors = Observer(), Errors()
    gates = Gates(state.dag.nodes())
    task = asyncio.create_task(run(state, gated_body(state, observer, gates), 2, errors))

    await observer.until(lambda: len(observer.running) == 2, task)
    assert "T3" not in observer.started  # ready, but the cap held it back

    state.halt = Halt(reason="stopped for the test")
    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await task

    assert "T3" not in observer.started
    assert errors.calls == []
    assert state.dag.is_complete() is False


# -- a stalled graph ----------------------------------------------------


class _AlwaysStalled(Dag):
    """A `Dag` that reports every incomplete state as stalled.

    A real graph cannot reach `is_stalled() is True` while something is
    `CLAIMED` — `Dag`'s own docstring is explicit that this is a
    can't-happen — so this override forces the condition the scheduler has to
    react to, to prove it reports through `on_error` and returns instead of
    waiting on a `progress` that will now never come.
    """

    def is_stalled(self) -> bool:
        return not self.is_complete()


async def test_a_stalled_graph_reports_through_on_error_and_returns() -> None:
    dag = _AlwaysStalled()
    dag.add_node("T1")
    dag.claim("T1")  # claimed by nobody the scheduler knows about
    state = RunState(label="wide", base_branch="main", dag=dag, tickets={"T1": feature("T1")})
    errors = Errors()

    async def body(ticket: Ticket) -> None:
        raise AssertionError("nothing was ever claimable")

    async with asyncio.timeout(TIMEOUT):
        await run(state, body, 2, errors)

    assert len(errors.calls) == 1
    ticket, error = errors.calls[0]
    assert ticket is None
    assert isinstance(error, StalledGraphError)


# -- body raising -------------------------------------------------------


async def test_body_raising_reports_on_error_and_does_not_leave_the_node_claimed() -> None:
    state = build_state([feature("T1")])
    errors = Errors()
    boom = RuntimeError("boom")

    async def body(ticket: Ticket) -> None:
        raise boom

    async with asyncio.timeout(TIMEOUT):
        await run(state, body, 2, errors)

    assert len(errors.calls) == 1
    ticket, error = errors.calls[0]
    assert ticket is not None
    assert ticket.id == "T1"
    assert error is boom
    assert state.dag.state("T1") is NodeState.PENDING


async def test_body_raising_stops_admission_and_lets_inflight_work_finish() -> None:
    state = build_state([feature("T1"), feature("T2")])
    observer, errors = Observer(), Errors()
    gates = Gates(state.dag.nodes())
    boom = RuntimeError("boom")

    async def body(ticket: Ticket) -> None:
        observer.enter(ticket.id)
        try:
            if ticket.id == "T1":
                raise boom
            await gates[ticket.id].wait()
            state.dag.complete(ticket.id)
        finally:
            observer.leave(ticket.id)

    task = asyncio.create_task(run(state, body, 2, errors))
    await observer.until(lambda: len(errors.calls) == 1, task)

    assert observer.running == {"T2"}

    gates.open("T2")
    async with asyncio.timeout(TIMEOUT):
        await task

    assert sorted(observer.finished) == ["T1", "T2"]
    assert len(errors.calls) == 1


async def test_two_bodies_raising_at_once_are_both_reported() -> None:
    state = build_state([feature("T1"), feature("T2")])
    observer, errors = Observer(), Errors()
    release = asyncio.Event()
    booms = {"T1": RuntimeError("one"), "T2": RuntimeError("two")}

    async def body(ticket: Ticket) -> None:
        observer.enter(ticket.id)
        try:
            await release.wait()
            raise booms[ticket.id]
        finally:
            observer.leave(ticket.id)

    task = asyncio.create_task(run(state, body, 2, errors))
    await observer.until(lambda: len(observer.running) == 2, task)
    release.set()

    async with asyncio.timeout(TIMEOUT):
        await task

    assert sorted(t.id for t, _ in errors.calls if t is not None) == ["T1", "T2"]
    assert {e for _, e in errors.calls} == {booms["T1"], booms["T2"]}
    assert state.dag.state("T1") is NodeState.PENDING
    assert state.dag.state("T2") is NodeState.PENDING


# -- cancellation -----------------------------------------------------------


class RaisingErrors:
    """An `on_error` that is itself broken: it records the call, then raises."""

    def __init__(self, to_raise: BaseException) -> None:
        self._to_raise = to_raise
        self.calls: list[tuple[Ticket | None, BaseException]] = []

    def __call__(self, ticket: Ticket | None, error: BaseException) -> None:
        self.calls.append((ticket, error))
        raise self._to_raise


async def test_a_raising_on_error_on_a_body_failure_still_cancels_and_awaits_inflight() -> None:
    # T1 fails immediately and its `on_error` call is what's broken; T2 has
    # nothing gating it, so it is still in flight when that happens.
    state = build_state([feature("T1"), feature("T2")])
    observer = Observer()
    cancelled: set[str] = set()
    boom = RuntimeError("boom")
    handler_boom = RuntimeError("on_error is broken")
    errors = RaisingErrors(handler_boom)

    async def body(ticket: Ticket) -> None:
        observer.enter(ticket.id)
        try:
            if ticket.id == "T1":
                raise boom
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            cancelled.add(ticket.id)
            raise
        finally:
            observer.leave(ticket.id)

    task = asyncio.create_task(run(state, body, 2, errors))
    with pytest.raises(RuntimeError) as excinfo:
        async with asyncio.timeout(TIMEOUT):
            await task

    # The run stays loud: the broken handler's own exception is what a
    # caller sees, not the error it failed to report or a swallowed no-op.
    assert excinfo.value is handler_boom
    assert errors.calls == [(state.tickets["T1"], boom)]
    # But it does not leave T2 behind: it is cancelled and awaited before
    # `run` raises, not abandoned as an orphan nothing will ever collect.
    assert cancelled == {"T2"}
    assert observer.running == set()


async def test_a_raising_on_error_on_a_stalled_graph_still_cancels_and_awaits_inflight() -> None:
    # T1 is claimed by nobody the scheduler knows about, forcing a stall once
    # nothing else is left ready. T3 is admitted alongside T2 so the loop
    # takes a genuine suspension with both in flight; opening T3's gate is
    # what frees the slot that lets the loop notice the stall, by which
    # point T2 is still running and never gets released on its own.
    dag = _AlwaysStalled()
    for node_id in ("T1", "T2", "T3"):
        dag.add_node(node_id)
    dag.claim("T1")
    state = RunState(
        label="wide",
        base_branch="main",
        dag=dag,
        tickets={node_id: feature(node_id) for node_id in ("T1", "T2", "T3")},
    )
    observer = Observer()
    gates = Gates(("T2", "T3"))
    cancelled: set[str] = set()
    handler_boom = RuntimeError("on_error is broken")
    errors = RaisingErrors(handler_boom)

    async def body(ticket: Ticket) -> None:
        observer.enter(ticket.id)
        try:
            if ticket.id == "T3":
                await gates["T3"].wait()
                state.dag.complete("T3")
                return
            try:
                await asyncio.Event().wait()  # T2 never completes on its own
            except asyncio.CancelledError:
                cancelled.add(ticket.id)
                raise
        finally:
            observer.leave(ticket.id)

    task = asyncio.create_task(run(state, body, 2, errors))
    await observer.until(lambda: observer.running == {"T2", "T3"}, task)

    gates.open("T3")
    with pytest.raises(RuntimeError) as excinfo:
        async with asyncio.timeout(TIMEOUT):
            await task

    assert excinfo.value is handler_boom
    assert len(errors.calls) == 1
    ticket, error = errors.calls[0]
    assert ticket is None
    assert isinstance(error, StalledGraphError)
    assert cancelled == {"T2"}
    assert observer.running == set()


async def test_cancelling_the_run_cancels_inflight_bodies_and_propagates() -> None:
    state = build_state([feature("T1"), feature("T2")])
    observer, errors = Observer(), Errors()
    cancelled: set[str] = set()

    async def body(ticket: Ticket) -> None:
        observer.enter(ticket.id)
        try:
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            cancelled.add(ticket.id)
            raise
        finally:
            observer.leave(ticket.id)

    task = asyncio.create_task(run(state, body, 2, errors))
    await observer.until(lambda: len(observer.running) == 2, task)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(TIMEOUT):
            await task

    assert cancelled == {"T1", "T2"}
    assert observer.running == set()
