"""The scheduler: a semaphore loop over a graph, driven by hand-held gates.

Nothing here sleeps. Every body waits on an `asyncio.Event` the test owns, so
the test decides exactly when each finishes and can inspect the whole system
while some are held. Every run is wrapped in a timeout so a deadlock fails the
test instead of hanging the suite.

Nothing here knows what a node is, either — the scheduler takes a `Dag` and a
body over node ids, so these tests are graphs and gates and nothing else.

The graph shapes mirror `tests/integration/test_concurrency.py::run_graph`,
the reference this scheduler was built from: `wide_dag` is five independent
roots, two joins over them, and one join over those — wider than any cap used
against it, so the cap is what holds work back rather than the graph.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import pytest

from agl.runtime.dag import Dag, NodeId, NodeState
from agl.runtime.scheduler import StalledGraphError, drive, run

TIMEOUT = 10.0

ROOTS = ("T1", "T2", "T3", "T4", "T5")
BLOCKED: dict[str, tuple[str, ...]] = {
    "T6": ("T1", "T2"),
    "T7": ("T3", "T4"),
    "T8": ("T6", "T7"),
}
CAP = 3


# -- building graphs --------------------------------------------------------


def build_dag(nodes: Sequence[tuple[str, tuple[str, ...]]]) -> Dag:
    """A graph holding `nodes`, with their blocker edges already built."""
    dag = Dag()
    for node_id, _ in nodes:
        dag.add_node(node_id)
    for node_id, blockers in nodes:
        for blocker in blockers:
            dag.add_edge(node_id, blocker)
    return dag


def plain(*node_ids: str) -> list[tuple[str, tuple[str, ...]]]:
    """Independent nodes, nothing blocking anything."""
    return [(node_id, ()) for node_id in node_ids]


def wide_dag() -> Dag:
    """Five independent roots, two joins over them, and one join over those."""
    nodes = plain(*ROOTS)
    nodes += [(node_id, blockers) for node_id, blockers in BLOCKED.items()]
    return build_dag(nodes)


class Halted:
    """The halt predicate a workflow supplies, as a flag a test can flip."""

    def __init__(self) -> None:
        self.on = False

    def __call__(self) -> bool:
        return self.on


# -- watching a run, without sleeping ----------------------------------------


class Observer:
    """What ran, in what order, and the only way anything here waits."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[str] = []
        self.running: set[str] = set()
        self.peak = 0
        self._changed = asyncio.Event()

    def enter(self, node_id: str) -> None:
        self.started.append(node_id)
        self.running.add(node_id)
        self.peak = max(self.peak, len(self.running))
        self._changed.set()

    def leave(self, node_id: str) -> None:
        self.running.discard(node_id)
        self.finished.append(node_id)
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
    """One event per node. A node's body does not finish until its gate opens."""

    def __init__(self, node_ids: Sequence[str]) -> None:
        self._gates = {node_id: asyncio.Event() for node_id in node_ids}

    def __getitem__(self, node_id: str) -> asyncio.Event:
        return self._gates[node_id]

    def open(self, *node_ids: str) -> None:
        for node_id in node_ids or tuple(self._gates):
            self._gates[node_id].set()


class Errors:
    """What `on_error` was called with, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[NodeId | None, BaseException]] = []

    def __call__(self, node_id: NodeId | None, error: BaseException) -> None:
        self.calls.append((node_id, error))


def gated_body(
    dag: Dag, observer: Observer, gates: Gates
) -> Callable[[NodeId], Awaitable[None]]:
    """A body that waits for its node's gate, then completes the graph node.

    Stands in for a real body's whole pipeline down to the merge that calls
    `dag.complete` on the workflow's behalf; this scheduler test only needs
    that the node moves, not how.
    """

    async def body(node_id: NodeId) -> None:
        observer.enter(node_id)
        try:
            await gates[node_id].wait()
            dag.complete(node_id)
        finally:
            observer.leave(node_id)

    return body


async def run_to_completion(dag: Dag, cap: int) -> Observer:
    """Run every gate open from the start, and hand back what happened."""
    observer = Observer()
    gates = Gates(dag.nodes())
    gates.open()
    errors = Errors()
    async with asyncio.timeout(TIMEOUT):
        await run(dag, gated_body(dag, observer, gates), cap, errors)
    assert errors.calls == []
    return observer


# -- the cap ------------------------------------------------------------


@dataclass(frozen=True)
class Ran:
    dag: Dag
    observer: Observer
    errors: Errors
    held_running: tuple[str, ...]
    held_ready: tuple[str, ...]
    after_one_release_running: tuple[str, ...]
    after_one_release_started: tuple[str, ...]


async def drive_wide(cap: int) -> Ran:
    """Hold `cap` nodes running, look around, release one, look again, finish."""
    dag = wide_dag()
    observer, errors = Observer(), Errors()
    gates = Gates(dag.nodes())
    task = asyncio.create_task(run(dag, gated_body(dag, observer, gates), cap, errors))
    await observer.until(lambda: len(observer.running) == cap, task)

    held_running = tuple(sorted(observer.running))
    held_ready = dag.ready()

    gates.open(held_running[0])
    await observer.until(lambda: len(observer.started) == cap + 1, task)
    after_one_release_running = tuple(sorted(observer.running))
    after_one_release_started = tuple(observer.started)

    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await task

    return Ran(
        dag=dag,
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
    assert sorted(ran.observer.started) == sorted(ran.dag.nodes())
    assert sorted(ran.observer.finished) == sorted(ran.dag.nodes())


async def test_the_graph_completes_and_the_run_returns() -> None:
    ran = await drive_wide(CAP)
    assert ran.dag.is_complete() is True
    assert ran.observer.running == set()
    assert ran.errors.calls == []


async def test_every_node_runs_exactly_once_even_with_simultaneous_completions() -> None:
    dag = wide_dag()
    observer, errors = Observer(), Errors()
    gates = Gates(dag.nodes())
    task = asyncio.create_task(run(dag, gated_body(dag, observer, gates), CAP, errors))
    await observer.until(lambda: len(observer.running) == CAP, task)

    # Releasing every held gate at once: several bodies finish in the same
    # synchronous burst, rather than one at a time.
    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await task

    assert errors.calls == []
    for node in dag.nodes():
        assert observer.started.count(node) == 1
    assert sorted(observer.finished) == sorted(dag.nodes())
    assert dag.is_complete() is True


# -- the graph, under contention -----------------------------------------


async def test_no_node_starts_before_its_blockers_have_finished() -> None:
    dag = wide_dag()
    observer = await run_to_completion(dag, CAP)

    for node, blockers in BLOCKED.items():
        for blocker in blockers:
            assert observer.finished.index(blocker) < observer.started.index(node)


async def test_a_cap_above_the_ready_set_runs_everything_at_once() -> None:
    dag = build_dag(plain(*ROOTS))
    observer, errors = Observer(), Errors()
    gates = Gates(dag.nodes())
    task = asyncio.create_task(run(dag, gated_body(dag, observer, gates), 8, errors))

    await observer.until(lambda: len(observer.running) == len(ROOTS), task)
    assert observer.peak == len(ROOTS)

    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await task
    assert dag.is_complete() is True
    assert errors.calls == []


async def test_a_cap_of_one_is_fully_serial_and_topologically_ordered() -> None:
    dag = wide_dag()
    observer = await run_to_completion(dag, 1)

    assert observer.peak == 1
    order = observer.finished
    assert sorted(order) == sorted(dag.nodes())
    for node, blockers in BLOCKED.items():
        for blocker in blockers:
            assert order.index(blocker) < order.index(node)


async def test_a_blocked_node_never_occupies_a_slot() -> None:
    # A cap of two with a join deeper than that: a scheduler that claimed a
    # slot before checking readiness would hold both slots on the roots and
    # then deadlock waiting for the join, since nothing would ever free one.
    dag = build_dag([*plain("A", "B"), ("C", ("A", "B")), ("D", ("C",))])
    observer = await run_to_completion(dag, 2)

    assert dag.is_complete() is True
    assert observer.peak <= 2
    assert sorted(observer.finished) == ["A", "B", "C", "D"]


# -- the runtime mutation: work discovered mid-run -------------------------


async def test_work_filed_mid_run_runs_before_the_released_parent() -> None:
    dag = build_dag(plain("P"))
    observer, errors = Observer(), Errors()
    gates = Gates(("P", "B1", "B2"))
    filed = asyncio.Event()

    async def body(node_id: NodeId) -> None:
        observer.enter(node_id)
        try:
            if node_id == "P" and not filed.is_set():
                # The parent's first pass: it discovers work that has to
                # happen before it can finish, and returns without completing.
                # The node is already `PENDING` again by the time this
                # returns — nodes, then edges, then the release, which is the
                # ordering `Dag` documents.
                for child in ("B1", "B2"):
                    dag.add_node(child)
                for child in ("B1", "B2"):
                    dag.add_edge("P", child)
                dag.release("P")
                filed.set()
                return
            await gates[node_id].wait()
            dag.complete(node_id)
        finally:
            observer.leave(node_id)

    task = asyncio.create_task(run(dag, body, 2, errors))
    await observer.until(lambda: filed.is_set(), task)

    # The parent is blocked by the work it just filed; only that work is ready.
    assert dag.state("P") is NodeState.PENDING
    assert "P" not in dag.ready()
    await observer.until(lambda: observer.running == {"B1", "B2"}, task)

    gates.open("B1", "B2")
    # Both children run to completion before the parent becomes ready again.
    # `P` is already in `finished` from its first, incomplete pass.
    await observer.until(lambda: observer.started.count("P") == 2, task)
    assert sorted(observer.finished) == ["B1", "B2", "P"]
    assert observer.started == ["P", "B1", "B2", "P"] or observer.started == ["P", "B2", "B1", "P"]

    gates.open("P")
    async with asyncio.timeout(TIMEOUT):
        await task

    assert errors.calls == []
    assert dag.is_complete() is True


# -- halting ----------------------------------------------------------------


async def test_halt_set_mid_run_stops_admission_and_lets_inflight_finish() -> None:
    dag = build_dag(plain("T1", "T2", "T3"))
    observer, errors = Observer(), Errors()
    gates = Gates(dag.nodes())
    halted = Halted()
    task = asyncio.create_task(run(dag, gated_body(dag, observer, gates), 2, errors, halted))

    await observer.until(lambda: len(observer.running) == 2, task)
    assert "T3" not in observer.started  # ready, but the cap held it back

    halted.on = True
    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await task

    assert "T3" not in observer.started
    assert errors.calls == []
    assert dag.is_complete() is False


async def test_a_run_with_no_halt_predicate_never_stops_admitting() -> None:
    # The default is the safe one: a caller with no notion of halting gets a
    # run that only stops on completion, a failing body, or a stall.
    dag = build_dag(plain("T1", "T2"))
    observer = await run_to_completion(dag, 2)

    assert sorted(observer.finished) == ["T1", "T2"]
    assert dag.is_complete() is True


async def test_a_halt_set_before_the_run_admits_nothing() -> None:
    dag = build_dag(plain("T1", "T2"))
    errors = Errors()
    halted = Halted()
    halted.on = True

    async def body(node_id: NodeId) -> None:
        raise AssertionError("nothing should have been admitted")

    async with asyncio.timeout(TIMEOUT):
        await run(dag, body, 2, errors, halted)

    assert errors.calls == []
    assert dag.is_complete() is False


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
    errors = Errors()

    async def body(node_id: NodeId) -> None:
        raise AssertionError("nothing was ever claimable")

    async with asyncio.timeout(TIMEOUT):
        await run(dag, body, 2, errors)

    assert len(errors.calls) == 1
    node_id, error = errors.calls[0]
    assert node_id is None
    assert isinstance(error, StalledGraphError)


# -- body raising -------------------------------------------------------


async def test_body_raising_reports_on_error_and_does_not_leave_the_node_claimed() -> None:
    dag = build_dag(plain("T1"))
    errors = Errors()
    boom = RuntimeError("boom")

    async def body(node_id: NodeId) -> None:
        raise boom

    async with asyncio.timeout(TIMEOUT):
        await run(dag, body, 2, errors)

    assert errors.calls == [("T1", boom)]
    assert dag.state("T1") is NodeState.PENDING


async def test_body_raising_stops_admission_and_lets_inflight_work_finish() -> None:
    dag = build_dag(plain("T1", "T2"))
    observer, errors = Observer(), Errors()
    gates = Gates(dag.nodes())
    boom = RuntimeError("boom")

    async def body(node_id: NodeId) -> None:
        observer.enter(node_id)
        try:
            if node_id == "T1":
                raise boom
            await gates[node_id].wait()
            dag.complete(node_id)
        finally:
            observer.leave(node_id)

    task = asyncio.create_task(run(dag, body, 2, errors))
    await observer.until(lambda: len(errors.calls) == 1, task)

    assert observer.running == {"T2"}

    gates.open("T2")
    async with asyncio.timeout(TIMEOUT):
        await task

    assert sorted(observer.finished) == ["T1", "T2"]
    assert len(errors.calls) == 1


async def test_two_bodies_raising_at_once_are_both_reported() -> None:
    dag = build_dag(plain("T1", "T2"))
    observer, errors = Observer(), Errors()
    release = asyncio.Event()
    booms = {"T1": RuntimeError("one"), "T2": RuntimeError("two")}

    async def body(node_id: NodeId) -> None:
        observer.enter(node_id)
        try:
            await release.wait()
            raise booms[node_id]
        finally:
            observer.leave(node_id)

    task = asyncio.create_task(run(dag, body, 2, errors))
    await observer.until(lambda: len(observer.running) == 2, task)
    release.set()

    async with asyncio.timeout(TIMEOUT):
        await task

    assert sorted(node for node, _ in errors.calls if node is not None) == ["T1", "T2"]
    assert {e for _, e in errors.calls} == {booms["T1"], booms["T2"]}
    assert dag.state("T1") is NodeState.PENDING
    assert dag.state("T2") is NodeState.PENDING


# -- cancellation -----------------------------------------------------------


class RaisingErrors:
    """An `on_error` that is itself broken: it records the call, then raises."""

    def __init__(self, to_raise: BaseException) -> None:
        self._to_raise = to_raise
        self.calls: list[tuple[NodeId | None, BaseException]] = []

    def __call__(self, node_id: NodeId | None, error: BaseException) -> None:
        self.calls.append((node_id, error))
        raise self._to_raise


async def test_a_raising_on_error_on_a_body_failure_still_cancels_and_awaits_inflight() -> None:
    # T1 fails immediately and its `on_error` call is what's broken; T2 has
    # nothing gating it, so it is still in flight when that happens.
    dag = build_dag(plain("T1", "T2"))
    observer = Observer()
    cancelled: set[str] = set()
    boom = RuntimeError("boom")
    handler_boom = RuntimeError("on_error is broken")
    errors = RaisingErrors(handler_boom)

    async def body(node_id: NodeId) -> None:
        observer.enter(node_id)
        try:
            if node_id == "T1":
                raise boom
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            cancelled.add(node_id)
            raise
        finally:
            observer.leave(node_id)

    task = asyncio.create_task(run(dag, body, 2, errors))
    with pytest.raises(RuntimeError) as excinfo:
        async with asyncio.timeout(TIMEOUT):
            await task

    # The run stays loud: the broken handler's own exception is what a
    # caller sees, not the error it failed to report or a swallowed no-op.
    assert excinfo.value is handler_boom
    assert errors.calls == [("T1", boom)]
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
    observer = Observer()
    gates = Gates(("T2", "T3"))
    cancelled: set[str] = set()
    handler_boom = RuntimeError("on_error is broken")
    errors = RaisingErrors(handler_boom)

    async def body(node_id: NodeId) -> None:
        observer.enter(node_id)
        try:
            if node_id == "T3":
                await gates["T3"].wait()
                dag.complete("T3")
                return
            try:
                await asyncio.Event().wait()  # T2 never completes on its own
            except asyncio.CancelledError:
                cancelled.add(node_id)
                raise
        finally:
            observer.leave(node_id)

    task = asyncio.create_task(run(dag, body, 2, errors))
    await observer.until(lambda: observer.running == {"T2", "T3"}, task)

    gates.open("T3")
    with pytest.raises(RuntimeError) as excinfo:
        async with asyncio.timeout(TIMEOUT):
            await task

    assert excinfo.value is handler_boom
    assert len(errors.calls) == 1
    node_id, error = errors.calls[0]
    assert node_id is None
    assert isinstance(error, StalledGraphError)
    assert cancelled == {"T2"}
    assert observer.running == set()


async def test_cancelling_the_run_cancels_inflight_bodies_and_propagates() -> None:
    dag = build_dag(plain("T1", "T2"))
    observer, errors = Observer(), Errors()
    cancelled: set[str] = set()

    async def body(node_id: NodeId) -> None:
        observer.enter(node_id)
        try:
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            cancelled.add(node_id)
            raise
        finally:
            observer.leave(node_id)

    task = asyncio.create_task(run(dag, body, 2, errors))
    await observer.until(lambda: len(observer.running) == 2, task)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(TIMEOUT):
            await task

    assert cancelled == {"T1", "T2"}
    assert observer.running == set()


# -- drive: passes, until complete or halted --------------------------------


async def test_drive_returns_once_the_graph_is_complete() -> None:
    dag = wide_dag()
    observer, errors = Observer(), Errors()
    gates = Gates(dag.nodes())
    gates.open()

    async with asyncio.timeout(TIMEOUT):
        await drive(dag, gated_body(dag, observer, gates), CAP, errors)

    assert dag.is_complete() is True
    assert errors.calls == []
    assert sorted(observer.finished) == sorted(dag.nodes())


async def test_drive_re_enters_run_when_a_pass_left_the_graph_incomplete() -> None:
    # A pass that returns with a resolved halt and newly-ready work is the
    # case `drive` exists for: `run` returned early, the graph is not
    # complete, and nothing else would go back for what is left.
    dag = build_dag(plain("T1", "T2"))
    observer, errors = Observer(), Errors()
    gates = Gates(dag.nodes())
    halted = Halted()
    passes = 0

    async def body(node_id: NodeId) -> None:
        nonlocal passes
        observer.enter(node_id)
        try:
            if passes == 0:
                # The first admitted node halts the run behind it, then
                # clears the halt on its way out — the shape a merge conflict
                # a person resolves takes.
                passes = 1
                halted.on = True
                await gates[node_id].wait()
                dag.complete(node_id)
                halted.on = False
                return
            await gates[node_id].wait()
            dag.complete(node_id)
        finally:
            observer.leave(node_id)

    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await drive(dag, body, 1, errors, halted)

    assert errors.calls == []
    assert dag.is_complete() is True
    assert sorted(observer.finished) == ["T1", "T2"]


async def test_drive_returns_with_the_graph_incomplete_when_the_halt_is_still_set() -> None:
    dag = build_dag(plain("T1", "T2", "T3"))
    observer, errors = Observer(), Errors()
    gates = Gates(dag.nodes())
    gates.open()
    halted = Halted()

    async def body(node_id: NodeId) -> None:
        observer.enter(node_id)
        try:
            halted.on = True  # set by the first node through, never cleared
            dag.complete(node_id)
        finally:
            observer.leave(node_id)

    async with asyncio.timeout(TIMEOUT):
        await drive(dag, body, 1, errors, halted)

    assert dag.is_complete() is False
    assert len(observer.finished) == 1
    assert errors.calls == []


async def test_drive_over_a_complete_graph_never_calls_the_body() -> None:
    dag = build_dag(plain("T1"))
    dag.claim("T1")
    dag.complete("T1")
    errors = Errors()

    async def body(node_id: NodeId) -> None:
        raise AssertionError("the graph was already complete")

    async with asyncio.timeout(TIMEOUT):
        await drive(dag, body, 2, errors)

    assert errors.calls == []


async def test_drive_returns_when_a_failing_body_leaves_the_graph_incomplete() -> None:
    # `on_error` here does what a workflow's does: it records the failure and
    # halts, which is what stops `drive` going round again on a graph that
    # cannot advance.
    dag = build_dag(plain("T1", "T2"))
    errors = Errors()
    halted = Halted()
    boom = RuntimeError("boom")

    def on_error(node_id: NodeId | None, error: BaseException) -> None:
        errors(node_id, error)
        halted.on = True

    async def body(node_id: NodeId) -> None:
        raise boom

    async with asyncio.timeout(TIMEOUT):
        await drive(dag, body, 1, on_error, halted)

    assert len(errors.calls) == 1
    assert dag.is_complete() is False
