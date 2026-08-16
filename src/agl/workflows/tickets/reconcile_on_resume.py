"""Everything git and the state can disagree about, settled before anything runs.

Layer: workflows. Called once between `resume_preflight` and the loop, so a
resumed run and a fresh one look at the same kind of world. Must not import
`workflow.py`, which imports this for its `resume` entry point.
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
"""Every status saying somebody is working a ticket. Nobody is: that process is gone."""

MERGE_MESSAGE = "Merge resolved before the run was resumed"
"""The message on a merge this commits on a person's behalf, saying so."""


def reconcile(ctx: RunContext) -> None:
    """Reconciles the repository and the state, before a single agent runs.

    Synchronous throughout, so no task can read the state halfway through.
    """
    trees = worktrees.for_run(ctx)
    open_trees = trees.reopen()
    _finish_merge(ctx.vcs, ctx.vcs.root())
    for work in open_trees:
        _finish_merge(ctx.vcs, work.tree)
        ctx.vcs.discard_changes(work.tree)
    _reconcile_state(ctx, trees, {work.key: work for work in open_trees})


def _finish_merge(vcs: Vcs, cwd: Path) -> None:
    """Takes a tree out of the merge it was left standing in.

    No unmerged paths means somebody finished the resolution, so it is committed;
    a half-finished one is aborted and the ticket re-merges the ordinary way.
    """
    if not vcs.merge_in_progress(cwd):
        return
    if vcs.unmerged_paths(cwd):
        vcs.abort_merge(cwd)
    else:
        vcs.commit_merge(cwd, MERGE_MESSAGE)


def _reconcile_state(ctx: RunContext, trees: Worktrees, open_trees: dict[str, Work]) -> None:
    """One pass over the tickets, written once at the end because it is one change."""
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
    """Removes a merged ticket's worktree, if a dead process left it one."""
    if work is not None:
        trees.release(work)


def _released(run: Run, ticket: Ticket) -> Run:
    """A claim handed back to the queue, through the life cycle's own moves.

    A waiting ticket returns to the status it suspended first, then is released.
    """
    if ticket.status is Status.AWAITING_INPUT and ticket.resume_to is not None:
        run = with_status(run, ticket.id, ticket.resume_to)
    return with_status(run, ticket.id, Status.PENDING)


def _marked(vcs: Vcs, run: Run, ticket: Ticket, branch: str) -> Run:
    """A branch with no `base_sha` recorded, marked where it stands."""
    # `base_sha` is taken once, in the same synchronous step that opens the worktree.
    if ticket.base_sha is not None or not vcs.branch_exists(branch):
        return run
    return with_base_sha(run, ticket.id, vcs.rev_parse(branch))
