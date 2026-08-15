"""Picking a run back up: what `settle` reconciles, and what `resume` refuses.

Real git in `tmp_path` throughout, because everything here is about the two
halves disagreeing — a branch that moved after the status was written, a tree
left standing in a merge, a claim held by a process that is gone. A fake git
would only ever agree with the state, which is the one thing these tests are
looking for.

A test builds the mess a killed process leaves — a state document written by
hand, worktrees opened and walked away from — and then calls `settle` over it,
which is what a second process is, minus the process.

The last test drives a whole resumed run. It needs no agent at all: a ticket
whose commit is on its branch and whose review documents are on disk is owed
nothing but its merge, and that is the point — a resumed ticket takes the step
git and the store say it is owed, not the one it was in the middle of.
"""

from pathlib import Path

import pytest

from agl.core.store import Store
from agl.runtime.context import RunContext
from agl.runtime.record import StateFile
from agl.runtime.worktrees import Work
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.resume import settle
from agl.workflows.tickets.reviews import REVIEWERS, review_key
from agl.workflows.tickets.snapshot import RunFile
from agl.workflows.tickets.state import Halt, Run
from agl.workflows.tickets.ticket import trees_for
from agl.workflows.tickets.workflow import NothingToResumeError, resume
from tests.conftest import commit_file, git
from tests.runtime.conftest import context, feature

SPEC_KEY = "spec.md"


# -- the mess a killed process leaves -----------------------------------------


def ticket(
    id_: str = "T-01",
    status: Status = Status.PENDING,
    base_sha: str | None = None,
    resume_to: Status | None = None,
) -> Ticket:
    return Ticket(
        id=id_,
        title="Add auth",
        status=status,
        deliverables=("auth.py",),
        base_sha=base_sha,
        resume_to=resume_to,
    )


def started(
    repo: Path, *tickets: Ticket, halt: Halt | None = None
) -> tuple[RunContext, RunFile]:
    """A context over a run that has already written a state document."""
    feature(repo)
    ctx = context(repo)
    state = RunFile(StateFile(ctx.store))
    state.write(Run(tickets=tickets, halt=halt))
    return ctx, state


def open_tree(ctx: RunContext, ticket_id: str = "T-01") -> Work:
    """A worktree checked out for one ticket and then walked away from."""
    trees = trees_for(ctx)
    return trees.acquire(ticket_id, trees.branch_for(ticket_id), ctx.base_branch)


def reviewed(store: Store, ticket_id: str, round_: int = 0) -> None:
    """Everything a review round that found nothing leaves behind."""
    for source in REVIEWERS:
        store.write_json(review_key(ticket_id, round_, source), {"findings": []})
    store.write_json(review_key(ticket_id, round_, "triage"), {"groups": []})


def mid_merge(ctx: RunContext, repo: Path) -> Work:
    """A ticket tree stopped in a merge, its one conflict unresolved."""
    commit_file(repo, "shared.txt", "base\n", "add shared")
    work = open_tree(ctx)
    commit_file(work.tree, "shared.txt", "theirs\n", "T-01: rewrite shared")
    commit_file(repo, "shared.txt", "ours\n", "feature: rewrite shared")
    assert not ctx.vcs.merge(work.tree, ctx.base_branch).clean
    return work


# -- claims -------------------------------------------------------------------


def test_every_claimed_ticket_goes_back_to_the_queue(repo: Path) -> None:
    """Nobody is running any of them: the process that claimed them is gone."""
    ctx, state = started(
        repo,
        ticket("T-01", Status.IN_PROGRESS),
        ticket("T-02", Status.IN_REVIEW),
        ticket("T-03", Status.MERGING),
        ticket("T-04", Status.AWAITING_INPUT, resume_to=Status.IN_PROGRESS),
    )

    settle(ctx)

    settled = state.load()
    assert [t.status for t in settled.tickets] == [Status.PENDING] * 4
    assert all(t.resume_to is None for t in settled.tickets)


def test_a_pending_ticket_and_a_merged_one_are_left_where_they_are(repo: Path) -> None:
    ctx, state = started(repo, ticket("T-01"), ticket("T-02", Status.MERGED))

    settle(ctx)

    assert [t.status for t in state.load().tickets] == [Status.PENDING, Status.MERGED]


def test_the_halt_the_last_process_stopped_on_does_not_stop_this_one(repo: Path) -> None:
    """A halt is why a run ended, not a fact about the repository now.

    Whatever it named has either been dealt with or is about to be reported
    again by the step that meets it, and a run that resumed into its own last
    halt would admit nothing and stop where it stood.
    """
    ctx, state = started(repo, ticket(), halt=Halt("T-01 conflicts with the base branch"))

    settle(ctx)

    assert state.load().halt is None


# -- worktrees ----------------------------------------------------------------


def test_a_merged_ticket_s_worktree_is_removed(repo: Path) -> None:
    """The teardown a process that died between the merge and the release owed."""
    ctx, _ = started(repo, ticket(status=Status.MERGED))
    work = open_tree(ctx)

    settle(ctx)

    assert not work.tree.exists()
    assert [w.path for w in ctx.vcs.list_worktrees()] == [repo.resolve()]


def test_an_unfinished_ticket_keeps_its_worktree(repo: Path) -> None:
    """Anything branched off it points at the branch that tree is sitting on."""
    ctx, _ = started(repo, ticket(status=Status.IN_PROGRESS))
    work = open_tree(ctx)

    settle(ctx)

    assert work.tree.exists()
    assert trees_for(ctx).reopen() == (work,)


def test_uncommitted_work_in_a_ticket_tree_is_discarded(repo: Path) -> None:
    """Half of an agent's edit is not work, and no step below re-derives it."""
    ctx, _ = started(repo, ticket(status=Status.IN_PROGRESS))
    work = open_tree(ctx)
    (work.tree / "scratch.py").write_text("half a thought\n", encoding="utf-8")
    (work.tree / "README.md").write_text("rewritten\n", encoding="utf-8")

    settle(ctx)

    assert not (work.tree / "scratch.py").exists()
    assert not ctx.vcs.is_dirty(work.tree)


# -- merges nobody finished ---------------------------------------------------


def test_a_merge_nobody_resolved_is_aborted(repo: Path) -> None:
    ctx, _ = started(repo, ticket(status=Status.MERGING))
    work = mid_merge(ctx, repo)

    settle(ctx)

    assert not ctx.vcs.merge_in_progress(work.tree)
    assert not ctx.vcs.is_ancestor(ctx.base_branch, work.branch)
    assert (work.tree / "shared.txt").read_text(encoding="utf-8") == "theirs\n"


def test_a_merge_somebody_resolved_is_committed(repo: Path) -> None:
    """The resolution is work, so it is committed before anything is discarded."""
    ctx, _ = started(repo, ticket(status=Status.MERGING))
    work = mid_merge(ctx, repo)
    (work.tree / "shared.txt").write_text("resolved\n", encoding="utf-8")
    git(work.tree, "add", "--", "shared.txt")

    settle(ctx)

    assert not ctx.vcs.merge_in_progress(work.tree)
    assert ctx.vcs.is_ancestor(ctx.base_branch, work.branch)
    assert (work.tree / "shared.txt").read_text(encoding="utf-8") == "resolved\n"


def test_a_merge_the_repository_root_was_left_in_is_settled_too(repo: Path) -> None:
    """The root is where a ticket merges into the base, so it is where a
    conflict is left — and the one tree nothing is ever discarded in."""
    ctx, _ = started(repo, ticket(status=Status.MERGING))
    commit_file(repo, "shared.txt", "base\n", "add shared")
    git(repo, "checkout", "-b", "other")
    commit_file(repo, "shared.txt", "theirs\n", "other: rewrite shared")
    git(repo, "checkout", ctx.base_branch)
    commit_file(repo, "shared.txt", "ours\n", "feature: rewrite shared")
    assert not ctx.vcs.merge(repo, "other").clean
    (repo / "notes.txt").write_text("mine\n", encoding="utf-8")

    settle(ctx)

    assert not ctx.vcs.merge_in_progress(repo)
    assert (repo / "notes.txt").read_text(encoding="utf-8") == "mine\n"


# -- base_sha -----------------------------------------------------------------


def test_a_branch_with_no_base_sha_recorded_gets_one(repo: Path) -> None:
    """The process died between creating the branch and writing the mark, so
    the branch still stands exactly where it was cut."""
    ctx, state = started(repo, ticket())
    trees = trees_for(ctx)
    ctx.vcs.create_branch(trees.branch_for("T-01"), ctx.base_branch)

    settle(ctx)

    assert state.load().ticket("T-01").base_sha == ctx.vcs.rev_parse(ctx.base_branch)


def test_a_base_sha_already_recorded_is_left_alone(repo: Path) -> None:
    """Re-taking it after a commit landed would erase what it measures."""
    ctx, state = started(repo, ticket(status=Status.IN_PROGRESS))
    work = open_tree(ctx)
    cut = ctx.vcs.rev_parse(work.branch)
    state.write(Run(tickets=(ticket(status=Status.IN_PROGRESS, base_sha=cut),)))
    commit_file(work.tree, "auth.py", "auth\n", "T-01: add auth")

    settle(ctx)

    assert state.load().ticket("T-01").base_sha == cut


def test_a_ticket_with_no_branch_yet_stays_unmarked(repo: Path) -> None:
    ctx, state = started(repo, ticket())

    settle(ctx)

    assert state.load().ticket("T-01").base_sha is None


# -- what cannot be resumed ---------------------------------------------------


async def test_a_run_that_agreed_nothing_cannot_be_resumed(repo: Path) -> None:
    feature(repo)
    ctx = context(repo)

    with pytest.raises(NothingToResumeError) as raised:
        await resume(ctx)

    assert "start a new run" in str(raised.value)


async def test_a_finished_run_cannot_be_resumed(repo: Path) -> None:
    ctx, _ = started(repo, ticket(status=Status.MERGED))

    with pytest.raises(NothingToResumeError) as raised:
        await resume(ctx)

    assert "already finished" in str(raised.value)


# -- the whole thing ----------------------------------------------------------


async def test_resume_takes_the_step_the_ticket_is_owed_and_finishes_the_run(
    repo: Path,
) -> None:
    """A ticket left claimed as `MERGING` with its work committed and its review
    on disk: no agent runs, and the run ends merged.

    Everything a resume is made of, in one pass — the claim released, the tree
    taken over rather than checked out again, the step re-derived from git and
    the store, and the state written to say so.
    """
    ctx, state = started(repo, ticket(status=Status.MERGING))
    ctx.store.write(SPEC_KEY, "# spec\n")
    work = open_tree(ctx)
    cut = ctx.vcs.rev_parse(work.branch)
    commit_file(work.tree, "T-01.txt", "T-01\n", "T-01: add auth")
    state.write(Run(tickets=(ticket(status=Status.MERGING, base_sha=cut),)))
    reviewed(ctx.store, "T-01")

    await resume(ctx)

    assert (repo / "T-01.txt").read_text(encoding="utf-8") == "T-01\n"
    assert state.load().ticket("T-01").status is Status.MERGED
    assert [w.path for w in ctx.vcs.list_worktrees()] == [repo.resolve()]
