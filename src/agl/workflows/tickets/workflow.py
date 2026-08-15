"""The ticket workflow: interview, decompose, then drive every ticket to merged.

Layer: workflows. The shape of the loop and this run's policy, and nothing else
— every step delegates to `agents`, `state`, `screens`, `reviews`, `tools`, or
the runtime's `scheduler`, `worktrees`, `merge` and `display`. A function that
grows past the shape belongs in one of those, not here.

Three things the runtime cannot know are here. `base_for` is the ticket rule
that a bug branches off its parent. `resolve` with `halt_for` is the whole halt
policy: what a merge that did not land means, and whether a person pressing
enter can help. And `failed` is what an exception out of a ticket means.

Everything is a function over `Loop`, the run's collaborators assembled once.
No object holds the run: the state is a local of `run`, the display is the one
session `live` yields, and both are passed to the steps that need them. The
terminal is entered exactly once and each stage swaps the screen on it.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from agl.core.agent import AgentQuestion
from agl.core.terminal import Option, Question
from agl.runtime.context import RunContext, build_gate, preflight
from agl.runtime.dag import NodeId
from agl.runtime.display import Board, Display, live
from agl.runtime.merge import (
    MergeConfig,
    MergeDecision,
    MergeOutcome,
    MergeQueue,
    MergeRequest,
    MergeStatus,
)
from agl.runtime.scheduler import drive
from agl.runtime.worktrees import Work, Worktrees
from agl.workflows.tickets import agents, screens
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.agents import Activity, Ask
from agl.workflows.tickets.models import Status, Ticket, tickets_from_json
from agl.workflows.tickets.reviews import next_bug_start, to_bug_tickets
from agl.workflows.tickets.state import Halt, RunState, halt_for

__all__ = [
    "DecomposeAbortedError",
    "HaltedError",
    "InterviewIncompleteError",
    "Job",
    "Loop",
    "run",
]


class InterviewIncompleteError(Exception):
    """Raised when the interview ended without saving a specification."""


class DecomposeAbortedError(Exception):
    """Raised when the user aborted decomposition before approving any tickets."""


class HaltedError(Exception):
    """Raised when a run ended with a halt nobody resolved.

    Carries the halt, because the run's exit status *is* this exception:
    nothing outside reaches into a run's state to see how it went.
    """

    def __init__(self, halt: Halt) -> None:
        super().__init__(halt.reason)
        self.halt = halt


@dataclass(frozen=True)
class Loop:
    """This run's collaborators, assembled by the workflow. Assembly, no behaviour."""

    ctx: RunContext
    display: Display
    state: RunState
    trees: Worktrees
    merges: MergeQueue


@dataclass(frozen=True)
class Job:
    """One ticket, bound to the worktree its work happens in and its two callbacks.

    Built once in `one_ticket` and handed to each step below it, so neither
    callback has to be threaded through every signature. Both close over the
    ticket id: concurrent tickets sharing either would report into each other's
    rows and answer each other's questions.
    """

    ticket: Ticket
    tree: Path
    branch: str
    on_activity: Activity
    ask: Ask


# -- the run --------------------------------------------------------------


async def run(ctx: RunContext) -> None:
    """One run, end to end. Edit this function to change the shape of the loop."""
    preflight(ctx.vcs, ctx.store, ctx.label)
    board = Board(started_at=time.monotonic())
    async with live(ctx.terminal, board) as display:
        display.show(lambda: screens.session(ctx.label, board))
        await interview(ctx, display)
        state = RunState(ctx.label, ctx.base_branch, board)
        state.add(await decompose(ctx, display, board))
        board.mark(screens.APPROVED)
        display.show(lambda: screens.dashboard(state, time.monotonic()))
        await implement_all(ctx, display, state)
    if state.halt is not None:
        raise HaltedError(state.halt)


async def interview(ctx: RunContext, display: Display) -> None:
    """Interrogate the user until the run has a specification to work from."""
    await agents.interview(
        ctx, ctx.request, display.activity(ctx.label), partial(display.ask_agent, ctx.label)
    )
    if not ctx.store.exists(ticket_tools.SPEC_KEY):
        raise InterviewIncompleteError("the interview ended without saving a specification")


async def decompose(ctx: RunContext, display: Display, board: Board) -> tuple[Ticket, ...]:
    """Propose tickets, ask for approval, and loop on a revision until settled.

    A revision is appended to the spec rather than passed to the next call, so
    the agent re-reads one document that says everything and the feedback is
    still there when a later role reads it.
    """
    tickets: tuple[Ticket, ...] = ()
    display.show(lambda: screens.decompose(ctx.label, board, tickets))
    while True:
        await agents.decompose(
            ctx, display.activity(ctx.label), partial(display.ask_agent, ctx.label)
        )
        tickets = tickets_from_json(ctx.store.read_json(ticket_tools.TICKETS_KEY))
        answer = await display.ask(
            Question(
                header=ctx.label,
                title=f"Approve these {len(tickets)} tickets?",
                options=(
                    Option("approve", "Start the run with these tickets"),
                    Option("abort", "Cancel without creating any tickets"),
                ),
            )
        )
        if not answer.was_free_text:
            if answer.text == "approve":
                return tickets
            raise DecomposeAbortedError("the user aborted decomposition")
        spec = ctx.store.read(ticket_tools.SPEC_KEY)
        ctx.store.write(
            ticket_tools.SPEC_KEY,
            f"{spec}\n\n## Decomposition feedback\n\n{answer.text}\n",
        )


async def implement_all(ctx: RunContext, display: Display, state: RunState) -> None:
    """Every approved ticket to merged, at most `max_concurrent` at a time."""
    merges = MergeQueue(
        ctx.vcs,
        MergeConfig(
            build=build_gate(ctx.project),
            resolve=partial(resolve, display, state, ctx.label),
        ),
    )
    loop = Loop(ctx, display, state, _trees(ctx), merges)
    async with merges.running():
        # A ticket whose merge did not land is parked in `submit` until
        # `resolve` deals with it, so a halt still set when a pass returns is
        # one nothing resolved — exactly `drive`'s stopping condition.
        await drive(
            state.dag,
            partial(one_ticket, loop),
            ctx.max_concurrent,
            partial(failed, state),
            state.is_halted,
        )


# -- this run's policy ----------------------------------------------------


async def resolve(
    display: Display, state: RunState, label: str, outcome: MergeOutcome
) -> MergeDecision:
    """What a merge that did not land means to this run: the halt policy.

    The queue reports; this decides. A halt a person can act on holds the run
    at the dashboard until they press enter, then looks at git again. One they
    cannot ends the queue, which answers every ticket still waiting on a merge
    so the run can come back with the halt still set.
    """
    state.halt = halt_for(outcome)  # the dashboard shows it on the next frame
    if not state.halt.resumable:
        return MergeDecision.STOP
    await display.confirm(label, "press enter to continue")
    state.halt = None
    return MergeDecision.RETRY


def failed(state: RunState, node_id: NodeId | None, error: BaseException) -> None:
    """An exception out of a ticket, or out of the loop itself: a halt to restart from."""
    who = node_id if node_id is not None else "the run"
    state.halt = Halt(f"{who} failed: {error}", str(error), resumable=False)


def base_for(loop: Loop, ticket: Ticket) -> str:
    """The branch a ticket's own branch is cut from: the run's base, or its parent's."""
    if ticket.parent is None:
        return loop.state.base_branch
    return loop.trees.branch_for(ticket.parent)


# -- one ticket -----------------------------------------------------------


async def one_ticket(loop: Loop, node_id: NodeId) -> None:
    """One pass over one ticket: implement it, review it, then merge or file bugs.

    Anything but a merged ticket keeps its worktree — work still to do, or a
    merge that did not land — so a halted run leaves what a person may want to
    look at where it is.
    """
    ticket = loop.state.tickets[node_id]
    work = loop.trees.acquire(ticket.id, loop.trees.branch_for(ticket.id), base_for(loop, ticket))
    task = job(loop, ticket, work)
    loop.state.set_status(ticket.id, Status.IN_PROGRESS)
    if ticket.first_pass:
        await implement(loop, task)
    if bugs := await review(loop, task):
        file_bugs(loop, task, bugs)
    else:
        await merge_it(loop, task)
    if ticket.status is Status.MERGED:
        loop.trees.release(work)
    else:
        loop.trees.keep(work)


def job(loop: Loop, ticket: Ticket, work: Work) -> Job:
    """One ticket bound to its worktree and to the two callbacks its calls need."""
    return Job(
        ticket=ticket,
        tree=work.tree,
        branch=work.branch,
        on_activity=loop.display.activity(ticket.id),
        ask=asker(loop.state, loop.display, ticket.id),
    )


def asker(state: RunState, display: Display, ticket_id: str) -> Ask:
    """The `ask` one ticket's calls are given, closed over the ticket it is for.

    The ticket is suspended into `AWAITING_INPUT` for as long as the question is
    open, so the dashboard says which row is waiting and the question says who
    is asking. A shared `ask` would land answers on the wrong agent in a log
    that looks perfectly well-formed, so every ticket gets its own.
    """

    async def ask(question: AgentQuestion) -> str:
        with state.awaiting(ticket_id):
            return await display.ask_agent(ticket_id, question)

    return ask


async def implement(loop: Loop, job: Job) -> None:
    """The implementation agent, and the one commit its work becomes."""
    await agents.implement(loop.ctx, job.ticket, job.tree, job.on_activity, job.ask)
    loop.ctx.vcs.commit_all(job.tree, f"{job.ticket.id}: {job.ticket.title}")


async def review(loop: Loop, job: Job) -> tuple[Ticket, ...]:
    """Both reviewers and triage, as the bug tickets they come to."""
    loop.state.set_status(job.ticket.id, Status.IN_REVIEW)
    base = base_for(loop, job.ticket)
    findings = await agents.review(
        loop.ctx, job.ticket, job.tree, base, job.on_activity, job.ask
    )
    groups = await agents.triage(loop.ctx, job.ticket, findings, job.on_activity, job.ask)
    start = next_bug_start(loop.state.tickets, job.ticket.id)
    return to_bug_tickets(job.ticket, groups, start)


def file_bugs(loop: Loop, job: Job, bugs: Sequence[Ticket]) -> None:
    """Put a review's findings into the graph as work the ticket now waits on."""
    loop.state.file_bugs(job.ticket.id, bugs)
    job.ticket.review_round += 1


async def merge_it(loop: Loop, job: Job) -> None:
    """Hand the ticket's branch to the queue and record what became of it.

    A bug merges into its parent's branch, in the parent's still-open worktree;
    everything else into the run's base branch, in the repository root.
    `STOPPED` is the queue saying nobody will ever deal with this one — the run
    is already ending on somebody else's halt — so the ticket is left `MERGING`,
    which is what the last frame shows.
    """
    loop.state.set_status(job.ticket.id, Status.MERGING)
    parent = job.ticket.parent
    cwd = loop.ctx.project.repo if parent is None else loop.trees.tree_of(parent)
    outcome = await loop.merges.submit(
        MergeRequest(job.ticket.id, job.branch, base_for(loop, job.ticket), cwd)
    )
    if outcome.status is MergeStatus.MERGED:
        loop.state.set_status(job.ticket.id, Status.MERGED)
    elif outcome.status is MergeStatus.ABANDONED:
        loop.state.board.activity[job.ticket.id] = "merge abandoned"


def _trees(ctx: RunContext) -> Worktrees:
    """This run's worktree pool, under the project's trees root."""
    return Worktrees(
        ctx.vcs,
        trees_root=ctx.project.trees_root,
        project=ctx.project.name,
        label=ctx.label,
    )
