"""Everything git and the state can disagree about, settled before anything runs.

Layer: workflows. One function, called once, between `resume_preflight` and the
loop — so that by the time the first agent starts, a resumed run and a fresh one
are looking at the same kind of world.

**It only deals with what a killed process is remembered as having claimed.** A
branch already merged, a commit that landed a moment before the status naming it
was written, a review that half finished — none of those are cases here, because
none of them are written down anywhere to be wrong about. `step_for` asks git and
the store about each of them every time it matters, so there is nothing to
reconcile. What *is* written down is who was working what, which worktrees are
open, and where a branch was cut from, and those are exactly the three things
below.

The repository is the other half. A merge left in progress is the one mess a
resume knows how to clear, and it is cleared by asking git what state it is in
rather than by guessing what a person did: unmerged paths mean nobody finished
it, so it is aborted; none mean somebody did, so it is committed and their work
is kept. Only then is anything discarded, and only ever in a ticket tree —
`resume_preflight` has already refused a repository root that was dirty for any
other reason.

It must not import `workflow.py`: that module imports this one for its `resume`
entry point, and the other direction would close the circle.
"""

from pathlib import Path

from agl.core.vcs import Vcs
from agl.runtime.context import RunContext
from agl.runtime.record import StateFile
from agl.runtime.worktrees import Work, Worktrees
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.snapshot import RunFile
from agl.workflows.tickets.state import Run, with_base_sha, with_halt, with_status
from agl.workflows.tickets.ticket import trees_for

__all__ = ["MERGE_MESSAGE", "settle"]

_CLAIMED = (Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING, Status.AWAITING_INPUT)
"""Every status that says somebody is working a ticket. Nobody is: the process
that said so is gone, which is the whole of what a resume knows about them."""

MERGE_MESSAGE = "Merge resolved before the run was resumed"
"""The message on a merge this settles on a person's behalf. Deliberately says
who committed it, because they did not."""


def settle(ctx: RunContext) -> None:
    """Reconcile the repository and the state, before a single agent runs.

    Synchronous from the first line to the last, which is what makes it safe to
    do at all: nothing else in the run has started, and nothing in here can be
    interleaved with a task that reads the state halfway through the fixing.
    """
    trees = trees_for(ctx)
    open_trees = trees.reopen()
    _finish_merge(ctx.vcs, ctx.vcs.root())
    for work in open_trees:
        _finish_merge(ctx.vcs, work.tree)
        ctx.vcs.discard_changes(work.tree)
    _settle_state(ctx, trees, {work.key: work for work in open_trees})


def _finish_merge(vcs: Vcs, cwd: Path) -> None:
    """Take a tree out of the merge it was left standing in, whichever way is right.

    Unmerged paths are what a resolution removes, so their absence is the only
    evidence there is that somebody got to the end of one — and it is enough:
    committing their work costs nothing if it was going to be re-merged anyway,
    and throwing it away would cost them the afternoon. A half-finished
    resolution has no such evidence and is aborted, after which the ticket
    re-merges and halts the ordinary way if it conflicts again.
    """
    if not vcs.merge_in_progress(cwd):
        return
    if vcs.unmerged_paths(cwd):
        vcs.abort_merge(cwd)
    else:
        vcs.commit_merge(cwd, MERGE_MESSAGE)


def _settle_state(ctx: RunContext, trees: Worktrees, open_trees: dict[str, Work]) -> None:
    """One pass over the tickets, written once at the end.

    One write because it is one change — a document a reader could catch
    halfway through the pass would show a run that was never true, with half
    its claims released. The worktree removals happen inside the same pass and
    are not undone by anything below them, which is fine: a tree that is gone
    and a ticket that is `MERGED` say the same thing, and `MERGED` was already
    in the document before this started.
    """
    state = RunFile(StateFile(ctx.store))
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
    Nothing is branched off a merged ticket that has not already been cut, so
    keeping the tree alive would only leave a checkout nobody is going to use.
    """
    if work is not None:
        trees.release(work)


def _released(run: Run, ticket: Ticket) -> Run:
    """A claim handed back to the queue, through the life cycle's own moves.

    A waiting ticket goes back where it was suspended from first and is
    released from there, rather than being written straight to `PENDING`:
    `AWAITING_INPUT` is a suspension of a status, and the way out of it is the
    status it suspended. `check` has already refused a document that says a
    ticket is waiting with nowhere to return to.
    """
    if ticket.status is Status.AWAITING_INPUT and ticket.resume_to is not None:
        run = with_status(run, ticket.id, ticket.resume_to)
    return with_status(run, ticket.id, Status.PENDING)


def _marked(vcs: Vcs, run: Run, ticket: Ticket, branch: str) -> Run:
    """A branch with no `base_sha` recorded, marked where it stands.

    Only ever a branch created moments before the process died: the mark is
    written in the same synchronous step that creates the worktree, so the one
    way to have one without the other is to have been killed between them —
    and nothing had a tree to commit in yet, so the branch is still where it
    was cut. Without the mark git cannot be asked whether the ticket has been
    worked on at all.
    """
    if ticket.base_sha is not None or not vcs.branch_exists(branch):
        return run
    return with_base_sha(run, ticket.id, vcs.rev_parse(branch))
