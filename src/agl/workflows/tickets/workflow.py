"""The ticket workflow's graph: interview, decompose, then drive every ticket to merged.

Layer: workflows. This file holds the shape of the loop and nothing else —
every method delegates to `agents`, `state`, `merge`, `reviews`, `render`,
`approval`, `wiring`, `tools`, or the runtime's `scheduler` and `worktrees`. A
method that grows past the shape belongs in one of those modules, not here.

The two things the runtime cannot know are here: `_base_for`, which is the
ticket rule that a bug branches off its parent, and `_is_halted`, the predicate
that turns this run's `Halt` into the yes-or-no the scheduler asks for.

`Run` owns one run end to end: the `RunState` that is execution truth, the
`Live` that is display-only, the merge queue, and the live terminal session.
`go` is the whole story — interview, decompose, then every ticket to merged —
and it has no local variables, because everything it needs flows through
`self.state` and the store.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from agl.config import ProjectConfig
from agl.core.agent import AgentRunner
from agl.core.command import ExecResult, run_async
from agl.core.store import Store
from agl.core.terminal import LiveSession, Option, Question, Screen, Terminal
from agl.core.vcs import Vcs
from agl.runtime import paths
from agl.runtime.dag import Dag, NodeId
from agl.runtime.scheduler import drive
from agl.runtime.worktrees import Worktrees
from agl.workflows.tickets import agents, state
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.approval import Approval, DecomposeAbortedError, session_screen
from agl.workflows.tickets.merge import MergeQueue, MergeRequest
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.render import render
from agl.workflows.tickets.reviews import next_bug_start, to_bug_tickets
from agl.workflows.tickets.state import Halt, Live, RunState, bugs_first
from agl.workflows.tickets.wiring import Wiring

__all__ = [
    "DecomposeAbortedError",
    "Deps",
    "InterviewIncompleteError",
    "PreflightError",
    "Run",
    "Work",
]


@dataclass(frozen=True)
class Work:
    """One ticket, bound to the worktree its work happens in.

    The runtime pool is keyed by node id and knows nothing of tickets, so this
    is where the two are put back together — once, in `_run_one`, rather than
    threaded through every step below it.
    """

    ticket: Ticket
    tree: Path
    branch: str


@dataclass(frozen=True)
class Deps:
    """What every run needs, independent of any one run's label or request."""

    agent: AgentRunner
    vcs: Vcs
    store: Store
    terminal: Terminal
    config: ProjectConfig


class PreflightError(Exception):
    """Raised when the repository or the label is not in a state to start a run."""


class InterviewIncompleteError(Exception):
    """Raised when the interview ended without saving a specification."""


class Run:
    """The graph. Edit this file to change the shape of the loop."""

    def __init__(self, deps: Deps, label: str, description: str, max_concurrent: int) -> None:
        self.deps = deps
        self.label = label
        self.description = description
        self.max_concurrent = max_concurrent
        base_branch = deps.vcs.current_branch()
        self.state = RunState(label=label, base_branch=base_branch, dag=Dag(), tickets={})
        self.state.dag = Dag(priority=bugs_first(self.state))
        # Created here, not after approval: `started_at` has to cover the
        # whole session for the interview and decompose headers to have a
        # timer, and `activity` has to have somewhere to go the moment either
        # one starts. `approved_at` — the dashboard footer's clock — is set
        # later, in `decompose`, once tickets exist to approve.
        self.live: Live | None = Live(started_at=time.monotonic())
        self.session: LiveSession | None = None
        self.merge_queue: MergeQueue | None = None
        self.worktrees = Worktrees(
            deps.vcs,
            trees_root=deps.config.trees_root,
            project=deps.config.name,
            label=label,
        )
        self.wiring = Wiring(
            deps.agent, deps.store, deps.config.repo, self.state, label, lambda: self.live
        )
        self._pending_merges: dict[str, asyncio.Event] = {}
        self._halted = asyncio.Event()

    # -- the loop -----------------------------------------------------------

    async def go(self) -> None:
        self.preflight()
        await self.interview()
        await self.decompose()
        async with self.dashboard():
            await self.implement_all()

    async def ticket(self, w: Work) -> None:
        state.set_status(self.state, self.live, w.ticket.id, Status.IN_PROGRESS)
        if w.ticket.first_pass:
            await self.implement(w)
        if findings := await self.review(w):
            self.file_bugs(w, findings)
        else:
            await self.enqueue_merge(w)

    # -- preflight ------------------------------------------------------------

    def preflight(self) -> None:
        if self.deps.vcs.is_dirty():
            raise PreflightError("the repository has uncommitted changes")
        branch = self.deps.vcs.current_branch()
        if branch in ("main", "master"):
            raise PreflightError(f"cannot run on {branch!r}; check out a feature branch first")
        namespace = paths.branch_namespace(self.label)
        if self.deps.vcs.branches(namespace) or self.deps.store.list():
            raise PreflightError(
                f"{self.label!r} is already in use; run `agl clean {self.label}` first"
            )

    # -- interview ------------------------------------------------------------

    async def interview(self) -> None:
        async with self.deps.terminal.live(self._session_screen) as session:
            ctx = self.wiring.ctx(self.wiring.ask(session, None))
            await agents.interview(ctx, self.description, self.wiring.activity(self.label))
        if not self.deps.store.exists(ticket_tools.SPEC_KEY):
            raise InterviewIncompleteError("the interview ended without saving a specification")

    def _session_screen(self) -> Screen:
        assert self.live is not None
        return session_screen(self.label, self.live)

    # -- decompose --------------------------------------------------------------

    async def decompose(self) -> None:
        approval = Approval(self.deps.terminal, self.deps.store, self.label, self.wiring)
        tickets = await approval.run()
        assert self.live is not None
        self.live.approved_at = time.monotonic()
        state.add_tickets(self.state, self.live, tickets)

    # -- implement_all ----------------------------------------------------------

    @asynccontextmanager
    async def dashboard(self) -> AsyncIterator[None]:
        async with self.deps.terminal.live(self._screen) as session:
            self.session = session
            try:
                yield
            finally:
                self.session = None

    def _screen(self) -> Screen:
        assert self.live is not None
        return render(self.state, self.live, time.monotonic())

    async def implement_all(self) -> None:
        self.merge_queue = self._merge_queue()
        consumer = asyncio.create_task(self.merge_queue.run())
        resumer = asyncio.create_task(self._resume_loop())
        try:
            await self._drive()
        finally:
            resumer.cancel()
            await self.merge_queue.stop()
            await consumer

    async def _drive(self) -> None:
        # A halted ticket's own task does not return until it is resumed, so
        # a pass cannot return on its own while one is stuck waiting —
        # `_resume_loop`, running alongside it, is what unsticks it. A halt
        # that is still set when a pass returns is therefore one nothing
        # resolved, which is exactly `drive`'s stopping condition.
        await drive(
            self.state.dag,
            self._run_one,
            self.max_concurrent,
            self._on_error,
            self._is_halted,
        )

    def _is_halted(self) -> bool:
        return self.state.halt is not None

    async def _resume_loop(self) -> None:
        while True:
            await self._halted.wait()
            self._halted.clear()
            halt = self.state.halt
            if halt is not None and halt.resumable:
                await self._await_resume()

    async def _await_resume(self) -> None:
        assert self.session is not None
        assert self.merge_queue is not None
        question = Question(
            header=self.label,
            title="press enter to continue",
            options=(Option("continue", "resume the run"),),
        )
        await self.session.ask(question)
        self.merge_queue.resume()
        self.state.halt = None

    def _merge_queue(self) -> MergeQueue:
        return MergeQueue(
            self.deps.vcs,
            self._build,
            self._on_merged,
            self._on_halt,
            self._on_abandoned,
        )

    async def _build(self) -> ExecResult:
        return await run_async(
            list(self.deps.config.build),
            self.deps.config.repo,
            check=False,
            timeout=self.deps.config.build_timeout,
        )

    async def _run_one(self, ticket_id: NodeId) -> None:
        t = self.state.tickets[ticket_id]
        held = self.worktrees.acquire(
            t.id, self.worktrees.branch_for(t.id), self._base_for(t)
        )
        await self.ticket(Work(ticket=t, tree=held.tree, branch=held.branch))
        if t.status is Status.PENDING:
            self.worktrees.keep(held)
        else:
            self.worktrees.release(held)

    def _base_for(self, ticket: Ticket) -> str:
        """The branch a ticket's own branch is cut from: the run's base, or its parent's."""
        if ticket.parent is None:
            return self.state.base_branch
        return self.worktrees.branch_for(ticket.parent)

    # -- one ticket's body --------------------------------------------------

    async def implement(self, w: Work) -> None:
        assert self.session is not None
        ctx = self.wiring.ctx(self.wiring.ticket_ask(self.session, w.ticket.id))
        await agents.implement(ctx, w.ticket, w.tree, self.wiring.activity(w.ticket.id))
        self.deps.vcs.commit_all(w.tree, f"{w.ticket.id}: {w.ticket.title}")

    async def review(self, w: Work) -> tuple[Ticket, ...]:
        assert self.session is not None
        state.set_status(self.state, self.live, w.ticket.id, Status.IN_REVIEW)
        ctx = self.wiring.ctx(self.wiring.ticket_ask(self.session, w.ticket.id))
        activity = self.wiring.activity(w.ticket.id)
        base = self._base_for(w.ticket)
        findings = await agents.review(ctx, w.ticket, w.tree, base, activity)
        groups = await agents.triage(ctx, w.ticket, findings, activity)
        return to_bug_tickets(w.ticket, groups, next_bug_start(self.state.tickets, w.ticket.id))

    def file_bugs(self, w: Work, bugs: Sequence[Ticket]) -> None:
        state.file_bugs(self.state, self.live, w.ticket.id, bugs)
        w.ticket.review_round += 1

    async def enqueue_merge(self, w: Work) -> None:
        state.set_status(self.state, self.live, w.ticket.id, Status.MERGING)
        target = self._base_for(w.ticket)
        if w.ticket.parent is None:
            cwd = self.deps.config.repo
        else:
            cwd = self.worktrees.tree_of(w.ticket.parent)
        resolved = asyncio.Event()
        self._pending_merges[w.ticket.id] = resolved
        assert self.merge_queue is not None
        self.merge_queue.put(MergeRequest(w.ticket.id, w.branch, target, cwd))
        await resolved.wait()

    # -- merge queue callbacks ----------------------------------------------

    def _on_merged(self, ticket_id: str) -> None:
        state.set_status(self.state, self.live, ticket_id, Status.MERGED)
        self._pending_merges.pop(ticket_id).set()

    def _on_halt(self, halt: Halt) -> None:
        self._halt(halt)

    def _on_abandoned(self, ticket_id: str) -> None:
        if self.live is not None:
            self.live.activity[ticket_id] = "merge abandoned"
        self._pending_merges.pop(ticket_id).set()

    def _on_error(self, ticket_id: NodeId | None, error: BaseException) -> None:
        who = ticket_id if ticket_id is not None else "the run"
        self._halt(Halt(f"{who} failed: {error}", str(error), resumable=False))

    def _halt(self, halt: Halt) -> None:
        self.state.halt = halt
        self._halted.set()
