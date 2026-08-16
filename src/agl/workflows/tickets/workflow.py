"""The ticket workflow: interview, decompose, then drive every ticket to merged.

Layer: workflows. The shape of the run and nothing else — a reactor that asks
`stage_of` what the state supports, does that stage, and asks again, so a run
picked back up re-enters the stage it was owed. `run` and `resume` differ only
in what they do before `loop`, and nothing below is told which one it was.
"""

import time
from functools import partial

from agl.core.terminal import Option, Question
from agl.runtime import worktrees
from agl.runtime.context import RunContext, build_gate, preflight, resume_preflight
from agl.runtime.display import Board, Display, live
from agl.runtime.merge import MergeConfig, MergeQueue
from agl.runtime.record import RunRecord, StateFile, write_record
from agl.runtime.scheduler import drive
from agl.workflows.tickets import agents, screens
from agl.workflows.tickets.documents.state_document import StateDocument
from agl.workflows.tickets.documents.store_keys import SPEC_KEY, TICKETS_KEY
from agl.workflows.tickets.documents.tickets_document import tickets_from_json
from agl.workflows.tickets.errors import (
    DecomposeAbortedError,
    HaltedError,
    InterviewIncompleteError,
    NothingToResumeError,
)
from agl.workflows.tickets.halting import failed, resolve
from agl.workflows.tickets.models import Ticket
from agl.workflows.tickets.reconcile_on_resume import reconcile
from agl.workflows.tickets.run_state import with_tickets
from agl.workflows.tickets.steps import Stage, stage_of
from agl.workflows.tickets.ticket_claims import claims
from agl.workflows.tickets.ticket_pass import Loop, one_ticket

__all__ = [
    "loop",
    "resume",
    "run",
]


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


async def resume(ctx: RunContext) -> None:
    """The same run, picked back up. Edit this to change what resuming means.

    The two states with nothing to continue are refused before the repository is
    so much as looked at. `reconcile` stands where `write_record` does in `run`,
    because the record was written by the process that started this.
    """
    stage = stage_of(StateDocument(StateFile(ctx.store)).load(), ctx.store)
    if stage is Stage.INTERVIEW:
        raise NothingToResumeError("nothing was agreed yet — start a new run")
    if stage is Stage.DONE:
        raise NothingToResumeError("this run already finished")
    resume_preflight(ctx.vcs, ctx.base_branch)
    reconcile(ctx)
    await loop(ctx)


async def loop(ctx: RunContext) -> None:
    """React to the state until it says the run is done, or a halt outlives a stage.

    No stage returns anything the next one needs: what one stage produced is a
    document the next one reads.
    """
    board = Board(started_at=time.monotonic())
    state = StateDocument(StateFile(ctx.store))
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
    if not ctx.store.exists(SPEC_KEY):
        raise InterviewIncompleteError("the interview ended without saving a specification")


async def decompose(ctx: RunContext, display: Display, state: StateDocument, board: Board) -> None:
    """Propose tickets, ask for approval, and loop on a revision until settled.

    A revision is appended to the spec rather than passed to the next call, so
    the agent re-reads one document that says everything.
    """
    tickets: tuple[Ticket, ...] = ()
    display.show(lambda: screens.decompose(ctx.label, board, tickets))
    while True:
        await agents.decompose(
            ctx, display.activity(ctx.label), partial(display.ask_agent, ctx.label)
        )
        tickets = tickets_from_json(ctx.store.read_json(TICKETS_KEY))
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
        spec = ctx.store.read(SPEC_KEY)
        ctx.store.write(
            SPEC_KEY,
            f"{spec}\n\n## Decomposition feedback\n\n{answer.text}\n",
        )


async def implement_all(
    ctx: RunContext, display: Display, state: StateDocument, board: Board
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
    pool = worktrees.for_run(ctx)
    # `reopen` on every run, not only a resumed one: a fresh run finds an empty
    # trees directory, and a picked-up one takes over the trees a dead process
    # left rather than checking the same branches out a second time.
    pool.reopen()
    run_loop = Loop(ctx, display, state, board, pool, merges)
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
