"""A graph wider than the cap, driven by gates the test opens by hand.

The point of the shape below is that the cap is what holds work back, not the
graph: five nodes are ready at the start and only three may run, so a scheduler
that had lost its semaphore would be caught, and one that was accidentally
serial would fail to reach the cap at all.

Nothing sleeps. Each node's work waits inside the agent call on an
`asyncio.Event` the test owns, so the test decides exactly when a node finishes
and can inspect the whole system while three are held. `Observer.until` is the
only way anything here waits, and it is bounded by a timeout so a deadlock fails
the test instead of hanging the suite.

The scheduler is written inline rather than imported: `workflows/tickets/` does
not exist yet, and what this file is probing is which shape the real one has to
have. The two properties it was written to keep are that a slot is taken before
the graph is asked what is ready — a node still waiting on its blockers must
never occupy one — and that reading `ready()` and claiming what it names is a
single synchronous step, so two passes cannot pick up the same node. The second
one is `Dag.claim_next`'s to keep, not the scheduler's.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agl.core import paths
from agl.core.agent import NO_PARAMS, AgentSpec, Tool
from agl.core.dag import Dag, NodeId
from agl.core.vcs.impl.git import Git
from tests.fakes import FakeAgentRunner, ScriptedRun
from tests.integration.conftest import PROJECT, copy_repo

LABEL = "wide"
CAP = 3
TIMEOUT = 10.0

ROOTS = ("T1", "T2", "T3", "T4", "T5")
BLOCKED: dict[NodeId, tuple[NodeId, ...]] = {
    "T6": ("T1", "T2"),
    "T7": ("T3", "T4"),
    "T8": ("T6", "T7"),
}


def build_graph() -> Dag:
    """Five independent roots, two joins over them, and one join over those."""
    dag = Dag()
    for node in (*ROOTS, *BLOCKED):
        dag.add_node(node)
    for blocked, blockers in BLOCKED.items():
        for blocker in blockers:
            dag.add_edge(blocked, blocker)
    return dag


# -- the scheduler under probe --------------------------------------------

type Work = Callable[[NodeId], Awaitable[None]]


async def run_graph(dag: Dag, cap: int, work: Work) -> None:
    """Run every node, at most `cap` at once, until the graph is complete."""
    slots = asyncio.Semaphore(cap)
    progress = asyncio.Event()
    tasks: set[asyncio.Task[None]] = set()
    failures: list[Exception] = []

    async def run_one(node: NodeId) -> None:
        try:
            await work(node)
            dag.complete(node)
        except Exception as error:  # noqa: BLE001 - re-raised once the loop stops
            failures.append(error)
        finally:
            slots.release()
            progress.set()

    while not dag.is_complete() and not failures:
        # The slot comes first and goes straight back when nothing is ready:
        # holding one while waiting on a blocker is how a graph deeper than the
        # cap deadlocks.
        await slots.acquire()
        if dag.is_complete() or failures:
            # Waiting for a slot is the one place the loop suspends without
            # looking at the graph, so the last node can finish underneath it.
            slots.release()
            break
        # `claim_next` rather than `ready()` and `claim()`: reading the ready
        # set and claiming out of it must not be separated by an await, or two
        # passes reach the same node between the two steps. The graph is what
        # guarantees that, which is why the method exists.
        node = dag.claim_next()
        if node is not None:
            task = asyncio.create_task(run_one(node))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            continue
        slots.release()
        if dag.is_stalled():
            raise AssertionError(f"stalled with {dag.ready()} ready")
        progress.clear()
        await progress.wait()

    await asyncio.gather(*tasks)
    if failures:
        raise failures[0]


# -- watching it, without sleeping ----------------------------------------


class Observer:
    """What the scheduler did, and the only way anything here waits."""

    def __init__(self) -> None:
        self.started: list[NodeId] = []
        self.finished: list[NodeId] = []
        self.running: set[NodeId] = set()
        self.peak = 0
        self.worktree_peak = 0
        self._changed = asyncio.Event()

    def enter(self, node: NodeId) -> None:
        self.started.append(node)
        self.running.add(node)
        self.peak = max(self.peak, len(self.running))
        self._changed.set()

    def leave(self, node: NodeId) -> None:
        self.running.discard(node)
        self.finished.append(node)
        self._changed.set()

    def note_worktrees(self, count: int) -> None:
        self.worktree_peak = max(self.worktree_peak, count)

    async def until(
        self, predicate: Callable[[], bool], guard: asyncio.Task[None] | None = None
    ) -> None:
        """Wait for `predicate`. `guard` is the run, so its failure surfaces here."""
        async with asyncio.timeout(TIMEOUT):
            while not predicate():
                if guard is not None and guard.done():
                    guard.result()
                    raise AssertionError("the run finished before the wait was satisfied")
                # Nothing can change between the check above and the wait below:
                # every mutation happens synchronously alongside `_changed.set()`.
                self._changed.clear()
                await self._changed.wait()


class Gates:
    """One event per node. A node's work does not finish until its gate opens."""

    def __init__(self, nodes: tuple[NodeId, ...]) -> None:
        self._gates = {node: asyncio.Event() for node in nodes}

    def __getitem__(self, node: NodeId) -> asyncio.Event:
        return self._gates[node]

    def open(self, *nodes: NodeId) -> None:
        for node in nodes or tuple(self._gates):
            self._gates[node].set()


def gate_tool(gate: asyncio.Event, tree: Path, node: NodeId) -> Tool:
    """The node's whole job: wait to be let go, then write its one file."""

    async def handler(arguments: dict[str, Any]) -> str:
        await gate.wait()
        (tree / f"{node}.txt").write_text(f"{node}\n", encoding="utf-8")
        return f"wrote {node}.txt"

    return Tool(name="work", description="Do the ticket.", schema=NO_PARAMS, handler=handler)


# -- the run the assertions are made against ------------------------------


@dataclass(frozen=True)
class Ran:
    """One gated run of the wide graph, and what was true at each held moment."""

    dag: Dag
    observer: Observer
    held_running: tuple[NodeId, ...]
    held_ready: tuple[NodeId, ...]
    held_worktree_branches: tuple[str, ...]
    after_one_release_running: tuple[NodeId, ...]
    after_one_release_started: tuple[NodeId, ...]
    files: tuple[str, ...]
    levels: tuple[tuple[NodeId, ...], ...]


async def drive(repo: Path, trees: Path) -> Ran:
    """Hold three nodes, look around, let one go, look again, then finish."""
    vcs = Git(repo)
    dag = build_graph()
    nodes = dag.nodes()
    gates, observer = Gates(nodes), Observer()
    runner = FakeAgentRunner({"implement": ScriptedRun("done", calls=(("work", {}),))})

    async def work(node: NodeId) -> None:
        observer.enter(node)
        branch = paths.branch(LABEL, node)
        tree = vcs.add_worktree(paths.worktree_dir(trees, PROJECT, LABEL, node), branch, "main")
        observer.note_worktrees(len(vcs.list_worktrees()) - 1)
        await runner.run(
            AgentSpec(
                prompt=f"Do {node}.",
                cwd=tree.path,
                role="implement",
                tools=(gate_tool(gates[node], tree.path, node),),
            )
        )
        vcs.commit_all(tree.path, f"{node}: work")
        vcs.merge(repo, branch)
        vcs.remove_worktree(tree.path)
        observer.leave(node)

    run = asyncio.create_task(run_graph(dag, CAP, work))
    await observer.until(lambda: len(observer.running) == CAP, run)

    held_running = tuple(sorted(observer.running))
    held_ready = dag.ready()
    held_worktree_branches = tuple(sorted(tree.branch for tree in vcs.list_worktrees()[1:]))

    gates.open(held_running[0])
    await observer.until(lambda: len(observer.started) == CAP + 1, run)
    after_one_release_running = tuple(sorted(observer.running))
    after_one_release_started = tuple(observer.started)

    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await run

    return Ran(
        dag=dag,
        observer=observer,
        held_running=held_running,
        held_ready=held_ready,
        held_worktree_branches=held_worktree_branches,
        after_one_release_running=after_one_release_running,
        after_one_release_started=after_one_release_started,
        files=tuple(sorted(path.name for path in repo.glob("T*.txt"))),
        levels=dag.levels(),
    )


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory, _template_repo: Path) -> Ran:
    """Drive the wide graph once; every test below reads what it recorded."""
    root = tmp_path_factory.mktemp("concurrency")
    return asyncio.run(drive(copy_repo(root, _template_repo), root / "trees"))


# -- the cap ---------------------------------------------------------------


def test_the_cap_is_reached(ran: Ran) -> None:
    # Without this, a scheduler that ran everything one at a time would satisfy
    # every other assertion in the file.
    assert ran.observer.peak == CAP


def test_the_cap_is_never_exceeded(ran: Ran) -> None:
    assert ran.observer.peak <= CAP
    assert len(ran.held_running) == CAP


def test_the_cap_is_what_holds_them(ran: Ran) -> None:
    # Nodes were ready and unstarted while three ran: the graph was not the
    # thing keeping them back.
    assert ran.held_ready
    assert set(ran.held_ready).isdisjoint(ran.held_running)
    assert set(ran.held_ready) <= set(ROOTS)


def test_releasing_one_admits_exactly_one_more(ran: Ran) -> None:
    assert len(ran.after_one_release_started) == CAP + 1
    assert len(ran.after_one_release_running) == CAP
    assert ran.held_running[0] not in ran.after_one_release_running


# -- the graph, under contention ------------------------------------------


def test_no_node_starts_before_its_blockers_have_completed(ran: Ran) -> None:
    started, finished = ran.observer.started, ran.observer.finished
    for node, blockers in BLOCKED.items():
        for blocker in blockers:
            assert finished.index(blocker) < started.index(node), (
                f"{node} started before {blocker} finished"
            )


def test_every_node_ran_exactly_once(ran: Ran) -> None:
    assert sorted(ran.observer.started) == sorted(ran.dag.nodes())
    assert sorted(ran.observer.finished) == sorted(ran.dag.nodes())


def test_the_graph_completes(ran: Ran) -> None:
    assert ran.dag.is_complete() is True
    assert ran.observer.running == set()


def test_levels_match_the_shape_that_was_built(ran: Ran) -> None:
    assert ran.levels == (ROOTS, ("T6", "T7"), ("T8",))


# -- the worktrees ---------------------------------------------------------


def test_worktrees_track_the_cap(ran: Ran) -> None:
    assert ran.observer.worktree_peak == CAP
    assert len(ran.held_worktree_branches) == CAP


def test_each_running_node_is_on_its_own_branch(ran: Ran) -> None:
    assert ran.held_worktree_branches == tuple(
        paths.branch(LABEL, node) for node in ran.held_running
    )


def test_every_node_merged_into_the_base(ran: Ran) -> None:
    assert ran.files == tuple(f"{node}.txt" for node in sorted(ran.dag.nodes()))


# -- variations ------------------------------------------------------------


def gated_work(observer: Observer, gates: Gates) -> Work:
    """Work with no git in it: enter, wait for the gate, leave.

    The variations are about the scheduler alone, so they skip the worktrees the
    run above already proved out.
    """

    async def work(node: NodeId) -> None:
        observer.enter(node)
        await gates[node].wait()
        observer.leave(node)

    return work


async def test_a_cap_above_the_ready_set_does_not_restrict() -> None:
    dag = Dag()
    for node in ROOTS:
        dag.add_node(node)
    gates, observer = Gates(ROOTS), Observer()

    run = asyncio.create_task(run_graph(dag, 8, gated_work(observer, gates)))
    await observer.until(lambda: len(observer.running) == len(ROOTS), run)
    assert observer.peak == len(ROOTS)

    gates.open()
    async with asyncio.timeout(TIMEOUT):
        await run
    assert dag.is_complete() is True


async def test_a_cap_of_one_is_serial_and_topologically_ordered() -> None:
    dag = build_graph()
    gates, observer = Gates(dag.nodes()), Observer()
    gates.open()

    async with asyncio.timeout(TIMEOUT):
        await run_graph(dag, 1, gated_work(observer, gates))

    assert observer.peak == 1
    order = observer.finished
    assert sorted(order) == sorted(dag.nodes())
    for node, blockers in BLOCKED.items():
        for blocker in blockers:
            assert order.index(blocker) < order.index(node)


async def test_two_scheduler_passes_never_start_the_same_node_twice() -> None:
    # Claiming has to be atomic with respect to reading `ready()`: a pass that
    # read the ready set, awaited, and only then claimed would hand the same
    # node to both. `Dag.claim_next` is the graph making that impossible.
    dag = Dag()
    for node in ROOTS:
        dag.add_node(node)
    observer = Observer()
    resume = asyncio.Event()
    by_pass: dict[str, list[NodeId]] = {"a": [], "b": []}

    async def scheduler_pass(name: str) -> None:
        while (node := dag.claim_next()) is not None:
            by_pass[name].append(node)
            observer.enter(node)
            await resume.wait()
            observer.leave(node)
            dag.complete(node)

    passes = asyncio.gather(scheduler_pass("a"), scheduler_pass("b"))
    await observer.until(lambda: len(observer.running) == 2)
    assert len(set(observer.running)) == 2

    resume.set()
    async with asyncio.timeout(TIMEOUT):
        await passes

    assert sorted(observer.started) == sorted(dag.nodes())
    assert sorted(by_pass["a"] + by_pass["b"]) == sorted(dag.nodes())
    assert by_pass["a"] and by_pass["b"]
    assert dag.is_complete() is True


async def test_a_node_waiting_on_its_blockers_never_occupies_a_slot() -> None:
    # Two roots, a join over both, and a node after that, with a cap of two: a
    # loop that claimed a slot before checking readiness would hold both slots
    # on blocked nodes and never let the roots finish.
    dag = Dag()
    for node in ("A", "B", "C", "D"):
        dag.add_node(node)
    dag.add_edge("C", "A")
    dag.add_edge("C", "B")
    dag.add_edge("D", "C")

    gates, observer = Gates(dag.nodes()), Observer()
    gates.open()

    async with asyncio.timeout(TIMEOUT):
        await run_graph(dag, 2, gated_work(observer, gates))

    assert dag.is_complete() is True
    assert observer.peak <= 2
    assert sorted(observer.finished) == ["A", "B", "C", "D"]
