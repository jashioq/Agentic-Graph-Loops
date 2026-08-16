"""Everything git and the state can disagree about, settled before anything runs.

Layer: workflows. One function, called once between `resume_preflight` and the
loop, so that by the time the first agent starts a resumed run and a fresh one
are looking at the same kind of world. It must not import `workflow.py`, which
imports this for its `resume` entry point.
"""

from pathlib import Path

from agl.core.vcs import Vcs
from agl.runtime import worktrees
from agl.runtime.context import RunContext
from agl.runtime.record import StateFile
from agl.runtime.worktrees import Work, Worktrees
from agl.workflows.tickets.documents.state_document import StateDocument
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import Run, with_base_sha, with_halt, with_status

__all__ = ["MERGE_MESSAGE", "reconcile"]

_CLAIMED = (Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING, Status.AWAITING_INPUT)
"""Every status that says somebody is working a ticket. Nobody is: the process
that said so is gone, which is the whole of what a resume knows about them."""

MERGE_MESSAGE = "Merge resolved before the run was resumed"
"""The message on a merge this commits on a person's behalf, saying so."""


def reconcile(ctx: RunContext) -> None:
    """Reconcile the repository and the state, before a single agent runs.

    Synchronous from the first line to the last, so nothing here can be
    interleaved with a task that reads the state halfway through the fixing.
    """
    trees = worktrees.for_run(ctx)
    open_trees = trees.reopen()
    _finish_merge(ctx.vcs, ctx.vcs.root())
    for work in open_trees:
        _finish_merge(ctx.vcs, work.tree)
        ctx.vcs.discard_changes(work.tree)
    _reconcile_state(ctx, trees, {work.key: work for work in open_trees})


def _finish_merge(vcs: Vcs, cwd: Path) -> None:
    """Take a tree out of the merge it was left standing in, whichever way is right.

    No unmerged paths is the only evidence there is that somebody finished a
    resolution, and it is enough to commit their work. A half-finished one is
    aborted, after which the ticket re-merges and halts the ordinary way.
    """
    if not vcs.merge_in_progress(cwd):
        return
    if vcs.unmerged_paths(cwd):
        vcs.abort_merge(cwd)
    else:
        vcs.commit_merge(cwd, MERGE_MESSAGE)


def _reconcile_state(ctx: RunContext, trees: Worktrees, open_trees: dict[str, Work]) -> None:
    """One pass over the tickets, written once at the end.

    One write because it is one change: a reader catching the document halfway
    through the pass would see a run that was never true.
    """
    state = StateDocument(StateFile(ctx.store))
    run = with_halt(state.load(), None)
    for ticket in run.tickets:
        if ticket.status is Status.MERGED:
            _tear_down(trees, open_trees.get(ticket.id))
        elif ticket.status in _CLAIMED:
            run = _released(run, ticket)
        run = _marked(ctx.vcs, run, ticket, trees.branch_for(ticket.id))
    state.write(run)


def _tear_down(trees: Worktrees, work: Work | None) -> None:
    """Remove a merged ticket's worktree, if it still has one.

    The teardown a process that died between the merge and the release owed.
    """
    if work is not None:
        trees.release(work)


def _released(run: Run, ticket: Ticket) -> Run:
    """A claim handed back to the queue, through the life cycle's own moves.

    A waiting ticket goes back where it was suspended from first and is
    released from there: `AWAITING_INPUT` is a suspension of a status, and the
    way out of it is the status it suspended.
    """
    if ticket.status is Status.AWAITING_INPUT and ticket.resume_to is not None:
        run = with_status(run, ticket.id, ticket.resume_to)
    return with_status(run, ticket.id, Status.PENDING)


def _marked(vcs: Vcs, run: Run, ticket: Ticket, branch: str) -> Run:
    """A branch with no `base_sha` recorded, marked where it stands.

    Only ever a branch created moments before the process died, so it is still
    where it was cut.
    """
    # `base_sha` is taken once, in the same synchronous step that opens the worktree.
    if ticket.base_sha is not None or not vcs.branch_exists(branch):
        return run
    return with_base_sha(run, ticket.id, vcs.rev_parse(branch))
