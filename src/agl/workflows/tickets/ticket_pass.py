"""One ticket's pass: implement it, review it, then merge it or file bugs.

Layer: workflows. The body the scheduler calls once per admitted ticket, and
the steps it is made of. Every step asks `step_for` whether it is still owed and
skips itself when it is not, which is what makes a resumed pass and a fresh one
the same code. It imports nothing from `ticket_claims`, `halting` or `workflow`.
"""

from dataclasses import dataclass
from pathlib import Path

from agl.core.agent import AgentQuestion
from agl.runtime.agents import Activity, Ask
from agl.runtime.context import RunContext
from agl.runtime.dag import NodeId
from agl.runtime.display import Board, Display
from agl.runtime.merge import MergeQueue, MergeRequest, MergeStatus
from agl.runtime.worktrees import Work, Worktrees
from agl.workflows.tickets import agents
from agl.workflows.tickets.documents.state_document import StateDocument
from agl.workflows.tickets.errors import NothingImplementedError
from agl.workflows.tickets.findings import next_bug_start, to_bug_tickets
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import Run, with_base_sha, with_bugs, with_status
from agl.workflows.tickets.steps import Step, look, step_for

__all__ = [
    "Job",
    "Loop",
    "asker",
    "base_for",
    "one_ticket",
]


@dataclass(frozen=True)
class Loop:
    """This run's collaborators, assembled by the workflow. Assembly, no behaviour."""

    ctx: RunContext
    display: Display
    state: StateDocument
    board: Board
    trees: Worktrees
    merges: MergeQueue


@dataclass(frozen=True)
class Job:
    """One ticket, bound to the worktree its work happens in and its two callbacks.

    Both callbacks close over the ticket id: concurrent tickets sharing either
    would report into each other's rows. `ticket` is the snapshot this pass
    began from, so anything asking about its status asks the state instead.
    """

    ticket: Ticket
    tree: Path
    branch: str
    on_activity: Activity
    ask: Ask


def base_for(loop: Loop, ticket: Ticket) -> str:
    """The branch a ticket's own branch is cut from: the run's base, or its parent's."""
    if ticket.parent is None:
        return loop.ctx.base_branch
    return loop.trees.branch_for(ticket.parent)


def step_of(loop: Loop, ticket: Ticket) -> Step:
    """What this ticket is owed right now, asked of git and the store.

    Called before each of the three steps below rather than once at the top: a
    step that has just run changes the answer.
    """
    return step_for(
        look(
            loop.ctx.vcs,
            loop.ctx.store,
            ticket,
            loop.trees.branch_for(ticket.id),
            base_for(loop, ticket),
        )
    )


# -- one pass -------------------------------------------------------------


async def one_ticket(loop: Loop, node_id: NodeId) -> None:
    """One pass over one ticket, each step skipping itself when it is not owed."""
    ticket = loop.state.load().ticket(node_id)
    work = loop.trees.acquire(ticket.id, loop.trees.branch_for(ticket.id), base_for(loop, ticket))
    ticket = _with_base(loop, ticket, work)
    task = job(loop, ticket, work)

    if step_of(loop, ticket) is Step.IMPLEMENT:
        await implement(loop, task)
    if step_of(loop, ticket) is Step.REVIEW:
        if bugs := await review(loop, task):
            loop.state.update(lambda run: with_bugs(run, node_id, bugs))
            loop.trees.keep(work)
            return
    if step_of(loop, ticket) is Step.MERGE:
        await merge_it(loop, ticket, work.branch)

    # Where the ticket ended up is a question only the state can answer: every
    # step above has moved the run's copy since. Anything but a merged ticket
    # keeps its worktree, so a halted run leaves what a person may want to see.
    if loop.state.load().ticket(node_id).status is Status.MERGED:
        loop.trees.release(work)
    else:
        loop.trees.keep(work)


def _with_base(loop: Loop, ticket: Ticket, work: Work) -> Ticket:
    """Record where this ticket's branch stood before any of its work, once.

    A second pass finds the mark already there and leaves it: re-taking it
    after the ticket's commit landed would erase the difference it measures.
    """
    # `base_sha` is taken once, in the same synchronous step that opens the worktree.
    if ticket.base_sha is not None:
        return ticket
    sha = loop.ctx.vcs.rev_parse(work.branch)
    return loop.state.update(lambda run: with_base_sha(run, ticket.id, sha)).ticket(ticket.id)


def job(loop: Loop, ticket: Ticket, work: Work) -> Job:
    """One ticket bound to its worktree and to the two callbacks its calls need."""
    return Job(
        ticket=ticket,
        tree=work.tree,
        branch=work.branch,
        on_activity=loop.display.activity(ticket.id),
        ask=asker(loop.state, loop.display, ticket.id),
    )


def asker(state: StateDocument, display: Display, ticket_id: str) -> Ask:
    """The `ask` one ticket's calls are given, closed over the ticket it is for.

    The ticket is suspended into `AWAITING_INPUT` for as long as the question is
    open, so the dashboard says which row is waiting. The resume is in a
    `finally` because a question that failed still leaves a ticket that is no
    longer waiting on anybody.
    """

    async def ask(question: AgentQuestion) -> str:
        state.update(lambda run: with_status(run, ticket_id, Status.AWAITING_INPUT))
        try:
            return await display.ask_agent(ticket_id, question)
        finally:
            state.update(lambda run: _resumed(run, ticket_id))

    return ask


def _resumed(run: Run, ticket_id: str) -> Run:
    """A waiting ticket put back where it was suspended from.

    A ticket that is no longer waiting has already been resumed by something
    else and is left alone, so the `finally` above is safe to run twice.
    """
    waiting = run.ticket(ticket_id)
    if waiting.resume_to is None:
        return run
    return with_status(run, ticket_id, waiting.resume_to)


# -- the three steps ------------------------------------------------------


async def implement(loop: Loop, job: Job) -> None:
    """The implementation agent, and the one commit its work becomes."""
    loop.state.update(lambda run: with_status(run, job.ticket.id, Status.IN_PROGRESS))
    await agents.implement(loop.ctx, job.ticket, job.tree, job.on_activity, job.ask)
    if loop.ctx.vcs.commit_all(job.tree, f"{job.ticket.id}: {job.ticket.title}") is None:
        raise NothingImplementedError(
            f"{job.ticket.id}: the implementation agent left the worktree unchanged"
        )


async def review(loop: Loop, job: Job) -> tuple[Ticket, ...]:
    """Both reviewers and triage, as the bug tickets they come to."""
    loop.state.update(lambda run: with_status(run, job.ticket.id, Status.IN_REVIEW))
    base = base_for(loop, job.ticket)
    # `agents.review` runs both reviewers as top-level calls, never subagents.
    findings = await agents.review(loop.ctx, job.ticket, job.tree, base, job.on_activity, job.ask)
    groups = await agents.triage(loop.ctx, job.ticket, findings, job.on_activity, job.ask)
    start = next_bug_start((t.id for t in loop.state.load().tickets), job.ticket.id)
    return to_bug_tickets(job.ticket, groups, start)


async def merge_it(loop: Loop, ticket: Ticket, branch: str) -> None:
    """Hand the ticket's branch to the queue and record what became of it.

    A bug merges into its parent's branch, in the parent's still-open worktree;
    everything else into the run's base branch, in the repository root.
    `STOPPED` leaves the ticket `MERGING`: the queue is saying nobody will ever
    deal with it, because the run is already ending on somebody else's halt.
    """
    loop.state.update(lambda run: with_status(run, ticket.id, Status.MERGING))
    cwd = loop.ctx.project.repo if ticket.parent is None else loop.trees.tree_of(ticket.parent)
    outcome = await loop.merges.submit(
        MergeRequest(ticket.id, branch, base_for(loop, ticket), cwd)
    )
    if outcome.status is MergeStatus.MERGED:
        loop.state.update(lambda run: with_status(run, ticket.id, Status.MERGED))
    elif outcome.status is MergeStatus.ABANDONED:
        loop.board.activity[ticket.id] = "merge abandoned"
