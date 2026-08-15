"""One ticket's pass: implement it, review it, then merge it or file bugs.

Layer: workflows. Split out of `workflow.py`, which keeps the shape of the run
and this run's policy; what is here is the body the scheduler calls once per
admitted ticket, and the four steps it is made of.

**Every step re-derives what is owed before it runs.** A pass is one straight
line through implement, review and merge, and each of the three asks `step_for`
whether it is the step still owed and skips itself when it is not. That is the
whole of what makes a resumed ticket and a fresh one the same code: a ticket
whose commit already exists skips implement and reviews, one whose review
settled skips both and merges, and nothing anywhere carries a note about which
pass this is.

Anything but a merged ticket keeps its worktree — work still to do, or a merge
that did not land — so a halted run leaves what a person may want to look at
where it is.
"""

from dataclasses import dataclass
from pathlib import Path

from agl.core.agent import AgentQuestion
from agl.runtime.context import RunContext
from agl.runtime.dag import NodeId
from agl.runtime.display import Board, Display
from agl.runtime.merge import MergeQueue, MergeRequest, MergeStatus
from agl.runtime.worktrees import Work, Worktrees
from agl.workflows.tickets import agents
from agl.workflows.tickets.agents import Activity, Ask
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.reviews import next_bug_start, to_bug_tickets
from agl.workflows.tickets.snapshot import RunFile
from agl.workflows.tickets.state import Run, with_base_sha, with_bugs, with_status
from agl.workflows.tickets.steps import Step, look, step_for

__all__ = [
    "Job",
    "Loop",
    "NothingImplementedError",
    "asker",
    "base_for",
    "one_ticket",
    "trees_for",
]


class NothingImplementedError(Exception):
    """Raised when an implementation agent finished and the tree was unchanged.

    A ticket with no commit of its own is one `step_for` will answer `IMPLEMENT`
    about forever, so a run that let this pass would loop rather than fail. It
    is also worth surfacing on its own account: an agent that produced nothing
    is a failure, not a diff for reviewers to find nothing wrong with.
    """


@dataclass(frozen=True)
class Loop:
    """This run's collaborators, assembled by the workflow. Assembly, no behaviour."""

    ctx: RunContext
    display: Display
    state: RunFile
    board: Board
    trees: Worktrees
    merges: MergeQueue


@dataclass(frozen=True)
class Job:
    """One ticket, bound to the worktree its work happens in and its two callbacks.

    Built once in `one_ticket` and handed to each step below it, so neither
    callback has to be threaded through every signature. Both close over the
    ticket id: concurrent tickets sharing either would report into each other's
    rows and answer each other's questions.

    `ticket` is the snapshot this pass began from — id, title, deliverables and
    the review round the steps below are running, none of which move while it
    runs. Where the ticket *is* moves constantly, so anything asking about its
    status asks the state instead.
    """

    ticket: Ticket
    tree: Path
    branch: str
    on_activity: Activity
    ask: Ask


def trees_for(ctx: RunContext) -> Worktrees:
    """This run's worktree pool, under the project's trees root.

    Here rather than beside the loop that uses it because a resume builds one
    too, before any pass has run, and two processes over the same run have to
    name the same trees. A pool is addressed by project and label alone, so
    they do: the same three fields, in one place, whoever is asking.

    The pool comes back owning nothing. Taking over what is already on disk is
    `reopen`, which the caller asks for when it wants the answer.
    """
    return Worktrees(
        ctx.vcs,
        trees_root=ctx.project.trees_root,
        project=ctx.project.name,
        label=ctx.label,
    )


def base_for(loop: Loop, ticket: Ticket) -> str:
    """The branch a ticket's own branch is cut from: the run's base, or its parent's."""
    if ticket.parent is None:
        return loop.ctx.base_branch
    return loop.trees.branch_for(ticket.parent)


def step_of(loop: Loop, ticket: Ticket) -> Step:
    """What this ticket is owed right now, asked of git and the store.

    Called before each of the three steps below rather than once at the top: a
    step that has just run changes the answer, and re-deriving is what keeps the
    pass from having to remember what it did.
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

    # Where the ticket ended up is a question only the state can answer:
    # `ticket` is the snapshot this pass started from, and every step above has
    # moved the run's copy since.
    if loop.state.load().ticket(node_id).status is Status.MERGED:
        loop.trees.release(work)
    else:
        loop.trees.keep(work)


def _with_base(loop: Loop, ticket: Ticket, work: Work) -> Ticket:
    """Record where this ticket's branch stood before any of its work, once.

    In the same synchronous step that opened the worktree, because that is the
    only moment the branch is guaranteed to be where the base put it. A second
    pass finds the mark already there and leaves it: re-taking it after the
    ticket's commit landed would erase the difference it exists to measure.
    """
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


def asker(state: RunFile, display: Display, ticket_id: str) -> Ask:
    """The `ask` one ticket's calls are given, closed over the ticket it is for.

    The ticket is suspended into `AWAITING_INPUT` for as long as the question is
    open, so the dashboard says which row is waiting and the question says who is
    asking. A shared `ask` would land answers on the wrong agent in a log that
    looks perfectly well-formed, so every ticket gets its own.

    Symmetric by construction: `transition` recorded where the ticket was
    suspended from, so the way back is read out of the ticket rather than
    remembered here. The resume is in a `finally` because a question that failed
    still leaves a ticket that is no longer waiting on anybody.
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
    findings = await agents.review(loop.ctx, job.ticket, job.tree, base, job.on_activity, job.ask)
    groups = await agents.triage(loop.ctx, job.ticket, findings, job.on_activity, job.ask)
    start = next_bug_start((t.id for t in loop.state.load().tickets), job.ticket.id)
    return to_bug_tickets(job.ticket, groups, start)


async def merge_it(loop: Loop, ticket: Ticket, branch: str) -> None:
    """Hand the ticket's branch to the queue and record what became of it.

    A bug merges into its parent's branch, in the parent's still-open worktree;
    everything else into the run's base branch, in the repository root.
    `STOPPED` is the queue saying nobody will ever deal with this one — the run
    is already ending on somebody else's halt — so the ticket is left `MERGING`,
    which is what the last frame shows.

    Re-submitting a branch that is already merged is "Already up to date": a
    clean no-op that still runs the build gate, which is exactly what a resumed
    merge wants.
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
