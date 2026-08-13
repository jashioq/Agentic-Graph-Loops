"""The ticket workflow's graph: interview, decompose, then drive every ticket to merged.

Layer: workflows. This file holds the shape of the loop and nothing else —
every method delegates to `agents`, `state`, `scheduler`, `merge`, `reviews`,
`render`, or `tools`. A method that grows past the shape belongs in one of
those modules, not here.

`Run` owns one run end to end: the `RunState` that is execution truth, the
`Live` that is display-only, the merge queue, and the live terminal session.
`go` is the whole story — interview, decompose, then every ticket to merged —
and it has no local variables, because everything it needs flows through
`self.state` and the store.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from agl.config import ProjectConfig
from agl.core import paths
from agl.core.agent import AgentQuestion, AgentRunner
from agl.core.command import ExecResult, run_async
from agl.core.dag import Dag
from agl.core.store import Store
from agl.core.terminal import LiveSession, Option, Question, Row, Rows, Screen, Terminal, Text
from agl.core.vcs import Vcs
from agl.workflows.tickets import agents, scheduler, state
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.agents import AgentContext, Limits
from agl.workflows.tickets.merge import MergeQueue, MergeRequest
from agl.workflows.tickets.models import Status, Ticket, tickets_from_json
from agl.workflows.tickets.render import render
from agl.workflows.tickets.reviews import to_bug_tickets
from agl.workflows.tickets.state import Halt, Live, RunState

__all__ = [
    "DecomposeAbortedError",
    "Deps",
    "InterviewIncompleteError",
    "PreflightError",
    "Run",
    "Work",
]

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Deps:
    """What every run needs, independent of any one run's label or request."""

    agent: AgentRunner
    vcs: Vcs
    store: Store
    terminal: Terminal
    config: ProjectConfig


@dataclass(frozen=True)
class Work:
    """One ticket, bound to the worktree its work happens in."""

    ticket: Ticket
    tree: Path
    branch: str


class PreflightError(Exception):
    """Raised when the repository or the label is not in a state to start a run."""


class InterviewIncompleteError(Exception):
    """Raised when the interview ended without saving a specification."""


class DecomposeAbortedError(Exception):
    """Raised when the user aborted decomposition before approving any tickets."""


class Run:
    """The graph. Edit this file to change the shape of the loop."""

    def __init__(self, deps: Deps, label: str, description: str, max_concurrent: int) -> None:
        self.deps = deps
        self.label = label
        self.description = description
        self.max_concurrent = max_concurrent
        base_branch = deps.vcs.current_branch()
        self.state = RunState(label=label, base_branch=base_branch, dag=Dag(), tickets={})
        self.state.dag = Dag(priority=scheduler.bugs_first(self.state))
        self.live: Live | None = None
        self.session: LiveSession | None = None
        self.merge_queue: MergeQueue | None = None
        self._trees: dict[str, Work] = {}
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
        if self.deps.vcs.ref_exists(self.label) or self.deps.store.list():
            raise PreflightError(
                f"{self.label!r} is already in use; run `agl clean {self.label}` first"
            )

    # -- interview ------------------------------------------------------------

    async def interview(self) -> None:
        async with self.deps.terminal.live(self._plain_screen) as session:
            ctx = self._ctx(self._ask(session, None))
            await agents.interview(ctx, self.description)
        if not self.deps.store.exists(ticket_tools.SPEC_KEY):
            raise InterviewIncompleteError("the interview ended without saving a specification")

    def _plain_screen(self) -> Screen:
        return Screen(content=Rows(Row(Text(self.label))))

    # -- decompose --------------------------------------------------------------

    async def decompose(self) -> None:
        tickets: tuple[Ticket, ...] = ()

        def screen() -> Screen:
            return _decompose_screen(self.label, tickets)

        async with self.deps.terminal.live(screen) as session:
            revision = ""
            while True:
                tickets = await self._propose(session, revision)
                answer = await self._ask_approval(session, tickets)
                if answer is None:
                    break
                revision = answer
        self.live = Live(started_at=time.monotonic())
        state.add_tickets(self.state, self.live, tickets)

    async def _propose(self, session: LiveSession, revision: str) -> tuple[Ticket, ...]:
        if revision:
            self._append_spec(revision)
        ctx = self._ctx(self._ask(session, None))
        await agents.decompose(ctx)
        payload = self.deps.store.read_json(ticket_tools.TICKETS_KEY)
        return tickets_from_json(payload)

    async def _ask_approval(self, session: LiveSession, tickets: tuple[Ticket, ...]) -> str | None:
        question = Question(
            header=self.label,
            title=f"Approve these {len(tickets)} tickets?",
            options=(
                Option("approve", "Start the run with these tickets"),
                Option("abort", "Cancel without creating any tickets"),
            ),
        )
        answer = await session.ask(question)
        if answer.was_free_text:
            return answer.text
        if answer.text == "approve":
            return None
        raise DecomposeAbortedError("the user aborted decomposition")

    def _append_spec(self, revision: str) -> None:
        spec = self.deps.store.read(ticket_tools.SPEC_KEY)
        self.deps.store.write(
            ticket_tools.SPEC_KEY, f"{spec}\n\n## Decomposition feedback\n\n{revision}\n"
        )

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
        # `scheduler.run` cannot return on its own while one is stuck waiting
        # — `_resume_loop`, running alongside it, is what unsticks it. This
        # loop only has to notice work `scheduler.run` returned early from: a
        # resolved halt that left something newly ready to claim.
        while not self.state.dag.is_complete():
            await scheduler.run(self.state, self._run_one, self.max_concurrent, self._on_error)
            halt = self.state.halt
            if halt is not None and not halt.resumable:
                return

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
            self.deps.config.repo,
            self.state.base_branch,
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

    async def _run_one(self, t: Ticket) -> None:
        w = self._trees.pop(t.id, None) or self._checkout(t)
        await self.ticket(w)
        if w.ticket.status is Status.PENDING:
            self._trees[t.id] = w
        else:
            self.deps.vcs.remove_worktree(w.tree)

    def _checkout(self, ticket: Ticket) -> Work:
        branch = paths.branch(self.label, ticket.id)
        base = self._base_for(ticket)
        tree = self.deps.vcs.add_worktree(self._worktree_dir(ticket.id), branch, base)
        return Work(ticket=ticket, tree=tree.path, branch=branch)

    def _worktree_dir(self, ticket_id: str) -> Path:
        return paths.worktree_dir(
            self.deps.config.trees_root, self.deps.config.name, self.label, ticket_id
        )

    def _base_for(self, ticket: Ticket) -> str:
        if ticket.parent is None:
            return self.state.base_branch
        return paths.branch(self.label, ticket.parent)

    # -- one ticket's body --------------------------------------------------

    async def implement(self, w: Work) -> None:
        ctx = self._ctx(self._ticket_ask(w.ticket.id))
        await agents.implement(ctx, w.ticket, w.tree, self._activity(w.ticket.id))
        self.deps.vcs.commit_all(w.tree, f"{w.ticket.id}: {w.ticket.title}")

    async def review(self, w: Work) -> tuple[Ticket, ...]:
        state.set_status(self.state, self.live, w.ticket.id, Status.IN_REVIEW)
        ctx = self._ctx(self._ticket_ask(w.ticket.id))
        activity = self._activity(w.ticket.id)
        findings = await agents.review(ctx, w.ticket, w.tree, self._base_for(w.ticket), activity)
        groups = await agents.triage(ctx, w.ticket, findings, activity)
        return to_bug_tickets(w.ticket, groups, self._next_bug_start(w.ticket.id))

    def file_bugs(self, w: Work, bugs: Sequence[Ticket]) -> None:
        state.file_bugs(self.state, self.live, w.ticket.id, bugs)
        w.ticket.review_round += 1

    async def enqueue_merge(self, w: Work) -> None:
        state.set_status(self.state, self.live, w.ticket.id, Status.MERGING)
        if w.ticket.parent is not None:
            self._merge_bug(w)
            return
        resolved = asyncio.Event()
        self._pending_merges[w.ticket.id] = resolved
        assert self.merge_queue is not None
        self.merge_queue.put(MergeRequest(w.ticket.id, w.branch))
        await resolved.wait()

    def _merge_bug(self, w: Work) -> None:
        assert w.ticket.parent is not None
        parent_tree = self._trees[w.ticket.parent].tree
        result = self.deps.vcs.merge(parent_tree, w.branch)
        if not result.clean:
            reason = f"{w.ticket.id} conflicts with {w.ticket.parent}"
            self._halt(Halt(reason, resumable=False))
            return
        state.set_status(self.state, self.live, w.ticket.id, Status.MERGED)

    def _next_bug_start(self, parent_id: str) -> int:
        prefix = f"{parent_id}-bug-"
        used = [
            int(ticket_id[len(prefix) :])
            for ticket_id in self.state.tickets
            if ticket_id.startswith(prefix) and ticket_id[len(prefix) :].isdigit()
        ]
        return max(used, default=0) + 1

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

    def _on_error(self, ticket: Ticket | None, error: BaseException) -> None:
        who = ticket.id if ticket is not None else "the run"
        self._halt(Halt(f"{who} failed: {error}", str(error), resumable=False))

    def _halt(self, halt: Halt) -> None:
        self.state.halt = halt
        self._halted.set()

    # -- agent context and activity ------------------------------------------

    def _ctx(self, ask: Callable[[AgentQuestion], Awaitable[str]]) -> AgentContext:
        return AgentContext(
            runner=self.deps.agent,
            store=self.deps.store,
            repo=self.deps.config.repo,
            prompts=PROMPTS_DIR,
            limits=Limits(),
            ask=ask,
        )

    def _activity(self, ticket_id: str) -> Callable[[str], None]:
        def on_activity(text: str) -> None:
            assert self.live is not None
            self.live.activity[ticket_id] = text

        return on_activity

    # -- questions ------------------------------------------------------------

    def _ticket_ask(self, ticket_id: str) -> Callable[[AgentQuestion], Awaitable[str]]:
        assert self.session is not None
        return self._ask(self.session, ticket_id)

    def _ask(
        self, session: LiveSession, ticket_id: str | None
    ) -> Callable[[AgentQuestion], Awaitable[str]]:
        async def ask(question: AgentQuestion) -> str:
            frm = self._suspend(ticket_id)
            answer = await session.ask(_to_question(ticket_id or self.label, question))
            self._resume(ticket_id, frm)
            return answer.text

        return ask

    def _suspend(self, ticket_id: str | None) -> Status | None:
        if ticket_id is None:
            return None
        frm = self.state.tickets[ticket_id].status
        state.set_status(self.state, self.live, ticket_id, Status.AWAITING_INPUT)
        return frm

    def _resume(self, ticket_id: str | None, frm: Status | None) -> None:
        if ticket_id is not None and frm is not None:
            state.set_status(self.state, self.live, ticket_id, frm)


# -- pure -----------------------------------------------------------------


def _to_question(header: str, question: AgentQuestion) -> Question:
    return Question(
        header=header,
        title=question.title,
        options=tuple(Option(o.label, o.description) for o in question.options),
    )


def _decompose_screen(label: str, tickets: tuple[Ticket, ...]) -> Screen:
    if not tickets:
        return Screen(content=Rows(Row(Text(label))))
    dag = Dag()
    for ticket in tickets:
        dag.add_node(ticket.id)
    for ticket in tickets:
        for blocker in ticket.blocked_by:
            dag.add_edge(ticket.id, blocker)
    by_id = {t.id: t for t in tickets}
    rows = [Row(Text(label))]
    for level in dag.levels():
        for ticket_id in level:
            ticket = by_id[ticket_id]
            blocked = ", ".join(ticket.blocked_by) if ticket.blocked_by else "—"
            rows.append(Row(Text(f"{ticket.id}: {ticket.title} (blocked by: {blocked})")))
    return Screen(content=Rows(*rows))
