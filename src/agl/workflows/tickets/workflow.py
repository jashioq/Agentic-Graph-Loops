"""The ticket workflow: interview, decompose, then drive every ticket to merged.

Layer: workflows. The shape of the loop and this run's policy, and nothing else
— every step delegates to `agents`, `state`, `snapshot`, `steps`, `screens`,
`ticket`, or the runtime's `scheduler`, `worktrees`, `merge` and `display`. A
function that grows past the shape belongs in one of those, not here.

**The loop is a reactor over a document.** It does not run the stages in order;
it asks `stage_of` what the state supports, does that stage, and asks again. The
difference shows the moment anything is picked back up: a process that died
between two stages restarts into the one it was owed, because nothing about
"where we got to" was ever held in a local variable. It is also why deleting a
document walks the run backwards rather than confusing it.

Two of the three things the runtime cannot know are here. `resolve` with
`halt_for` is the whole halt policy: what a merge that did not land means, and
whether a person pressing enter can help. `failed` is what an exception out of a
ticket means. The third — that a bug branches off its parent — is `base_for`, in
`ticket.py`, beside the pass that uses it.

Everything is a function over `Loop`, the run's collaborators assembled once. No
object holds the run: the state is a document, the display is the one session
`live` yields, and both are passed to the steps that need them. The terminal is
entered exactly once and each stage swaps the screen on it.
"""

import time
from functools import partial

from agl.core.terminal import Option, Question
from agl.runtime.context import RunContext, build_gate, preflight
from agl.runtime.dag import NodeId, NodeState
from agl.runtime.display import Board, Display, live
from agl.runtime.merge import MergeConfig, MergeDecision, MergeOutcome, MergeQueue
from agl.runtime.record import RunRecord, StateFile, write_record
from agl.runtime.scheduler import Claims, drive
from agl.runtime.worktrees import Worktrees
from agl.workflows.tickets import agents, screens
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.models import Status, Ticket, tickets_from_json
from agl.workflows.tickets.snapshot import RunFile
from agl.workflows.tickets.state import Halt, dag_of, halt_for, with_halt, with_status, with_tickets
from agl.workflows.tickets.steps import Stage, Step, look, stage_of, step_for
from agl.workflows.tickets.ticket import Loop, base_for, one_ticket

__all__ = [
    "DecomposeAbortedError",
    "HaltedError",
    "InterviewIncompleteError",
    "loop",
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


# The status a ticket is claimed into, per step it is owed. What the dashboard
# says about a row the instant it is admitted is the truth about that row, on a
# resumed run as much as on a fresh one — a ticket whose work is already done
# and only needs merging is claimed as `MERGING`, not started over as
# `IN_PROGRESS`.
_STATUS_FOR: dict[Step, Status] = {
    Step.IMPLEMENT: Status.IN_PROGRESS,
    Step.REVIEW: Status.IN_REVIEW,
    Step.MERGE: Status.MERGING,
    Step.DONE: Status.MERGED,
}


# -- the run --------------------------------------------------------------


async def run(ctx: RunContext) -> None:
    """One run, end to end. Edit this function to change the shape of the loop."""
    preflight(ctx.vcs, ctx.store, ctx.label)
    write_record(
        ctx.store,
        RunRecord(
            workflow=ctx.workflow,
            label=ctx.label,
            request=ctx.request,
            base_branch=ctx.base_branch,
            project=ctx.project.name,
            max_concurrent=ctx.max_concurrent,
        ),
    )
    await loop(ctx)


async def loop(ctx: RunContext) -> None:
    """React to the state until it says the run is done, or a halt outlives a stage.

    The whole run is this: ask what stage the state supports, do it, ask again.
    No stage knows what comes after it, and none of them returns anything the
    next one needs — what one stage produced is a document the next one reads.
    """
    board = Board(started_at=time.monotonic())
    state = RunFile(StateFile(ctx.store))
    async with live(ctx.terminal, board) as display:
        while True:
            match stage_of(state.load(), ctx.store):
                case Stage.INTERVIEW:
                    await interview(ctx, display, board)
                case Stage.DECOMPOSE:
                    await decompose(ctx, display, state, board)
                case Stage.IMPLEMENT:
                    await implement_all(ctx, display, state, board)
                case Stage.DONE:
                    break
            if state.load().halt is not None:
                break
    if (halt := state.load().halt) is not None:
        raise HaltedError(halt)


async def interview(ctx: RunContext, display: Display, board: Board) -> None:
    """Interrogate the user until the run has a specification to work from."""
    display.show(lambda: screens.session(ctx.label, board))
    await agents.interview(
        ctx, ctx.request, display.activity(ctx.label), partial(display.ask_agent, ctx.label)
    )
    if not ctx.store.exists(ticket_tools.SPEC_KEY):
        raise InterviewIncompleteError("the interview ended without saving a specification")


async def decompose(ctx: RunContext, display: Display, state: RunFile, board: Board) -> None:
    """Propose tickets, ask for approval, and loop on a revision until settled.

    A revision is appended to the spec rather than passed to the next call, so
    the agent re-reads one document that says everything and the feedback is
    still there when a later role reads it. Approval is the write that ends the
    stage: the tickets go into the state, and the state is what says so.
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
                state.update(partial(with_tickets, tickets=tickets))
                board.mark(screens.APPROVED)
                return
            raise DecomposeAbortedError("the user aborted decomposition")
        spec = ctx.store.read(ticket_tools.SPEC_KEY)
        ctx.store.write(
            ticket_tools.SPEC_KEY,
            f"{spec}\n\n## Decomposition feedback\n\n{answer.text}\n",
        )


async def implement_all(
    ctx: RunContext, display: Display, state: RunFile, board: Board
) -> None:
    """Every approved ticket to merged, at most `max_concurrent` at a time."""
    display.show(
        lambda: screens.dashboard(state.latest(), board, ctx.label, time.monotonic())
    )
    merges = MergeQueue(
        ctx.vcs,
        MergeConfig(
            build=build_gate(ctx.project),
            resolve=partial(resolve, display, state, ctx.label),
        ),
    )
    run_loop = Loop(ctx, display, state, board, _trees(ctx), merges)
    async with merges.running():
        # A ticket whose merge did not land is parked in `submit` until
        # `resolve` deals with it, so a halt still set when a pass returns is
        # one nothing resolved — exactly `drive`'s stopping condition.
        await drive(
            claims(run_loop),
            partial(one_ticket, run_loop),
            ctx.max_concurrent,
            partial(failed, state),
            lambda: state.load().halt is not None,
        )


def claims(run_loop: Loop) -> Claims:
    """The scheduler's four questions, answered off the state document every time.

    Nothing is held between them. Each derives a graph, asks it one thing, and
    throws it away, so a state a person edited mid-run — or one another pass
    just wrote — is what the next admission is decided from.
    """
    state = run_loop.state

    def next_() -> NodeId | None:
        """Claim one ticket, atomically: load, decide, write, no `await`."""
        current = state.load()
        node = dag_of(current).claim_next()
        if node is None:
            return None
        ticket = current.ticket(node)
        step = step_for(
            look(
                run_loop.ctx.vcs,
                run_loop.ctx.store,
                ticket,
                run_loop.trees.branch_for(node),
                base_for(run_loop, ticket),
            )
        )
        state.write(with_status(current, node, _STATUS_FOR[step]))
        return node

    def release(node: NodeId) -> None:
        """Hand a ticket whose pass raised back to the queue.

        A merged ticket is left alone: it is terminal, and a resumed run can
        admit one — a ticket whose branch was already on its base is claimed as
        `MERGED` — so a pass that raised over one would otherwise ask for a move
        the life cycle forbids. This runs inside the scheduler's own error
        handler, where an exception would leave the loop waiting on a slot that
        nothing is ever going to free.
        """
        state.update(
            lambda run: run
            if run.ticket(node).status is Status.MERGED
            else with_status(run, node, Status.PENDING)
        )

    def stalled() -> tuple[NodeId, ...] | None:
        dag = dag_of(state.load())
        if not dag.is_stalled():
            return None
        return tuple(n for n in dag.nodes() if dag.state(n) is NodeState.PENDING)

    return Claims(
        next=next_,
        release=release,
        complete=lambda: dag_of(state.load()).is_complete(),
        stalled=stalled,
    )


# -- this run's policy ----------------------------------------------------


async def resolve(
    display: Display, state: RunFile, label: str, outcome: MergeOutcome
) -> MergeDecision:
    """What a merge that did not land means to this run: the halt policy.

    The queue reports; this decides. A halt a person can act on holds the run
    at the dashboard until they press enter, then looks at git again. One they
    cannot ends the queue, which answers every ticket still waiting on a merge
    so the run can come back with the halt still set.
    """
    halt = halt_for(outcome)
    state.update(lambda run: with_halt(run, halt))  # the dashboard shows it next frame
    if not halt.resumable:
        return MergeDecision.STOP
    await display.confirm(label, "press enter to continue")
    state.update(lambda run: with_halt(run, None))
    return MergeDecision.RETRY


def failed(state: RunFile, node_id: NodeId | None, error: BaseException) -> None:
    """An exception out of a ticket, or out of the loop itself: a halt to restart from."""
    who = node_id if node_id is not None else "the run"
    halt = Halt(f"{who} failed: {error}", str(error), resumable=False)
    state.update(lambda run: with_halt(run, halt))


def _trees(ctx: RunContext) -> Worktrees:
    """This run's worktree pool, under the project's trees root."""
    return Worktrees(
        ctx.vcs,
        trees_root=ctx.project.trees_root,
        project=ctx.project.name,
        label=ctx.label,
    )
