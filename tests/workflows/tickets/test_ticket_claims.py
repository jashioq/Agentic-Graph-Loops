"""How the scheduler claims a ticket off the state document, and hands it back.

The four questions `drive` asks, answered directly rather than through a whole
run. Real git and a real store throughout: which status a claim lands in is a
question about what the repository already contains, so a fake would only ever
agree with the state document it was built from.
"""

from pathlib import Path

from agl.runtime import worktrees
from agl.runtime.context import RunContext
from agl.runtime.display import Board, Display, live
from agl.runtime.merge import MergeConfig, MergeQueue
from agl.runtime.record import StateFile
from agl.workflows.tickets.documents.state_document import StateDocument
from agl.workflows.tickets.documents.store_keys import REVIEWERS, review_key
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import Run, with_base_sha, with_status, with_tickets
from agl.workflows.tickets.ticket_claims import claims
from agl.workflows.tickets.ticket_pass import Loop
from tests.conftest import commit_file
from tests.fakes import HeadlessTerminal
from tests.runtime.conftest import context, feature
from tests.workflows.tickets.conftest import NOW

# -- a run to claim off --------------------------------------------------------


def ticket(ticket_id: str, *blocked_by: str) -> Ticket:
    """A pending feature ticket, optionally waiting on others."""
    return Ticket(
        id=ticket_id,
        title=f"Do {ticket_id}",
        status=Status.PENDING,
        deliverables=(f"{ticket_id}.py",),
        blocked_by=blocked_by,
    )


def started(repo: Path, *tickets: Ticket) -> tuple[RunContext, StateDocument]:
    """A run on a feature branch whose state document holds `tickets`."""
    feature(repo)
    ctx = context(repo)
    state = StateDocument(StateFile(ctx.store))
    state.write(with_tickets(Run(), tickets))
    return ctx, state


def a_loop(ctx: RunContext, display: Display, state: StateDocument) -> Loop:
    """This run's collaborators; `claims` reaches the context, the state and the pool."""
    return Loop(
        ctx,
        display,
        state,
        Board(started_at=NOW),
        worktrees.for_run(ctx),
        MergeQueue(ctx.vcs, MergeConfig()),
    )


def implemented(ctx: RunContext, state: StateDocument, ticket_id: str) -> None:
    """A ticket's own commit on its own branch, marked from where the branch started."""
    pool = worktrees.for_run(ctx)
    state.write(with_base_sha(state.load(), ticket_id, ctx.vcs.rev_parse(ctx.base_branch)))
    work = pool.acquire(ticket_id, pool.branch_for(ticket_id), ctx.base_branch)
    commit_file(work.tree, f"{ticket_id}.py", "done\n", f"{ticket_id}: done")


def reviewed(ctx: RunContext, ticket_id: str, round_: int = 0) -> None:
    """The documents a finished review round that found nothing leaves behind."""
    for source in REVIEWERS:
        ctx.store.write_json(review_key(ticket_id, round_, source), {"findings": []})
    ctx.store.write_json(review_key(ticket_id, round_, "triage"), {"groups": []})


# -- next ----------------------------------------------------------------------


async def test_next_claims_the_ready_ticket_and_writes_the_claim_down(repo: Path) -> None:
    """A ticket nothing has been done to is claimed to be implemented."""
    ctx, state = started(repo, ticket("T-01"), ticket("T-02", "T-01"))

    async with live(HeadlessTerminal(), Board(started_at=NOW)) as display:
        node = claims(a_loop(ctx, display, state)).next()

    assert node == "T-01"
    assert state.load().ticket("T-01").status is Status.IN_PROGRESS
    assert state.load().ticket("T-02").status is Status.PENDING


async def test_next_answers_none_when_nothing_is_ready(repo: Path) -> None:
    """T-02 waits on T-01, and T-01 is claimed: there is nothing to admit."""
    ctx, state = started(repo, ticket("T-01"), ticket("T-02", "T-01"))

    async with live(HeadlessTerminal(), Board(started_at=NOW)) as display:
        held = claims(a_loop(ctx, display, state))
        held.next()

        assert held.next() is None


async def test_a_ticket_whose_work_is_on_its_branch_is_claimed_into_review(repo: Path) -> None:
    """The resumed case: git says the implementation happened, so it is not redone."""
    ctx, state = started(repo, ticket("T-01"))
    implemented(ctx, state, "T-01")

    async with live(HeadlessTerminal(), Board(started_at=NOW)) as display:
        node = claims(a_loop(ctx, display, state)).next()

    assert node == "T-01"
    assert state.load().ticket("T-01").status is Status.IN_REVIEW


async def test_a_ticket_owed_only_its_merge_is_claimed_as_merging(repo: Path) -> None:
    """Implemented, reviewed, nothing found: claimed to be merged, not started over."""
    ctx, state = started(repo, ticket("T-01"))
    implemented(ctx, state, "T-01")
    reviewed(ctx, "T-01")

    async with live(HeadlessTerminal(), Board(started_at=NOW)) as display:
        claims(a_loop(ctx, display, state)).next()

    assert state.load().ticket("T-01").status is Status.MERGING


# -- release -------------------------------------------------------------------


async def test_release_hands_a_claimed_ticket_back_to_the_queue(repo: Path) -> None:
    ctx, state = started(repo, ticket("T-01"))

    async with live(HeadlessTerminal(), Board(started_at=NOW)) as display:
        held = claims(a_loop(ctx, display, state))
        held.next()
        held.release("T-01")

    assert state.load().ticket("T-01").status is Status.PENDING


async def test_release_leaves_a_merged_ticket_where_it_is(repo: Path) -> None:
    """Merged is terminal, and a resumed run may admit one."""
    ctx, state = started(repo, ticket("T-01"))
    state.write(with_status(state.load(), "T-01", Status.MERGED))

    async with live(HeadlessTerminal(), Board(started_at=NOW)) as display:
        claims(a_loop(ctx, display, state)).release("T-01")

    assert state.load().ticket("T-01").status is Status.MERGED


# -- complete and stalled -------------------------------------------------------


async def test_complete_only_once_every_ticket_is_merged(repo: Path) -> None:
    ctx, state = started(repo, ticket("T-01"), ticket("T-02", "T-01"))

    async with live(HeadlessTerminal(), Board(started_at=NOW)) as display:
        held = claims(a_loop(ctx, display, state))

        assert held.complete() is False
        for ticket_id in ("T-01", "T-02"):
            state.write(with_status(state.load(), ticket_id, Status.MERGED))

        assert held.complete() is True


async def test_nothing_is_stalled_while_a_claim_is_still_in_flight(repo: Path) -> None:
    """A graph that cannot advance needs a cycle, and `check` refuses to load one."""
    ctx, state = started(repo, ticket("T-01"), ticket("T-02", "T-01"))

    async with live(HeadlessTerminal(), Board(started_at=NOW)) as display:
        held = claims(a_loop(ctx, display, state))

        assert held.stalled() is None
        held.next()

        assert held.stalled() is None
