"""What the state says to do next: the run's stage, and one ticket's step.

`step_for` is pure and is tested over every combination of the three facts —
eight rows, written out rather than derived, because a table that agreed with
itself would prove nothing.

`look` and `stage_of` read git and the store, and both are real: a branch that
has moved and a branch that has been merged are exactly the thing `base_sha`
exists to tell apart, and no fake would be evidence of that. The store is a
`FileStore` in `tmp_path`, so "the document is there" means the document is
there.
"""

import itertools
from pathlib import Path

import pytest

from agl.core.store import Store
from agl.core.store.impl.file_store import FileStore
from agl.core.vcs.impl.git import Git
from agl.workflows.tickets.documents.store_keys import REVIEWERS, review_key
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import Run, with_status, with_tickets
from agl.workflows.tickets.steps import Facts, Stage, Step, look, stage_of, step_for
from tests.conftest import commit_file, git

SPEC_KEY = "spec.md"


# -- step_for -------------------------------------------------------------


def expected(implemented: bool, merged: bool, settled: bool) -> Step:
    """The step those three facts leave owed, written out independently."""
    if merged:
        return Step.DONE
    if settled:
        return Step.MERGE
    if implemented:
        return Step.REVIEW
    return Step.IMPLEMENT


@pytest.mark.parametrize(
    ("implemented", "merged", "settled"), list(itertools.product([False, True], repeat=3))
)
def test_step_for_over_every_combination_of_facts(
    implemented: bool, merged: bool, settled: bool
) -> None:
    facts = Facts(implemented=implemented, merged=merged, settled=settled)

    assert step_for(facts) is expected(implemented, merged, settled)


def test_a_merged_ticket_is_done_whatever_else_is_true_of_it() -> None:
    assert step_for(Facts(implemented=True, merged=True, settled=False)) is Step.DONE


def test_a_settled_review_leaves_the_merge_owed() -> None:
    assert step_for(Facts(implemented=True, merged=False, settled=True)) is Step.MERGE


def test_a_ticket_nothing_has_happened_to_is_owed_its_implementation() -> None:
    assert step_for(Facts(implemented=False, merged=False, settled=False)) is Step.IMPLEMENT


# -- stage_of -------------------------------------------------------------


def ticket(ticket_id: str, status: Status = Status.PENDING) -> Ticket:
    return Ticket(
        id=ticket_id, title=f"Do {ticket_id}", status=status, deliverables=(f"{ticket_id}.py",)
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return FileStore(tmp_path / "run")


def test_a_run_with_nothing_written_is_owed_its_interview(store: Store) -> None:
    assert stage_of(Run(), store) is Stage.INTERVIEW


def test_a_spec_and_no_tickets_is_owed_a_decomposition(store: Store) -> None:
    store.write(SPEC_KEY, "# spec\n")

    assert stage_of(Run(), store) is Stage.DECOMPOSE


def test_tickets_mean_the_run_is_implementing_them(store: Store) -> None:
    store.write(SPEC_KEY, "# spec\n")
    run = with_tickets(Run(), (ticket("T-01"), ticket("T-02")))

    assert stage_of(run, store) is Stage.IMPLEMENT


def test_a_run_whose_every_ticket_merged_is_done(store: Store) -> None:
    store.write(SPEC_KEY, "# spec\n")
    run = with_tickets(Run(), (ticket("T-01"),))
    for status in (Status.IN_PROGRESS, Status.MERGING, Status.MERGED):
        run = with_status(run, "T-01", status)

    assert stage_of(run, store) is Stage.DONE


def test_one_unmerged_ticket_keeps_the_run_implementing(store: Store) -> None:
    store.write(SPEC_KEY, "# spec\n")
    run = with_tickets(Run(), (ticket("T-01"), ticket("T-02")))
    for status in (Status.IN_PROGRESS, Status.MERGING, Status.MERGED):
        run = with_status(run, "T-01", status)

    assert stage_of(run, store) is Stage.IMPLEMENT


def test_deleting_the_tickets_walks_the_run_back_to_decomposing(store: Store) -> None:
    """The state is the only thing that says how far a run got, so taking a
    document away is how a person sends it back for the step that makes one."""
    store.write(SPEC_KEY, "# spec\n")
    run = with_tickets(Run(), (ticket("T-01"),))
    assert stage_of(run, store) is Stage.IMPLEMENT

    assert stage_of(Run(), store) is Stage.DECOMPOSE


def test_deleting_the_spec_walks_the_run_back_to_interviewing(store: Store) -> None:
    store.write(SPEC_KEY, "# spec\n")
    assert stage_of(Run(), store) is Stage.DECOMPOSE

    store.delete(SPEC_KEY)

    assert stage_of(Run(), store) is Stage.INTERVIEW


# -- look -----------------------------------------------------------------


def settle(
    store: Store, ticket_id: str, round_: int = 0, *, groups: list[dict] | None = None
) -> None:
    """Write everything a finished review round leaves behind."""
    for source in REVIEWERS:
        store.write_json(review_key(ticket_id, round_, source), {"findings": []})
    store.write_json(review_key(ticket_id, round_, "triage"), {"groups": groups or []})


def branched(repo: Path, name: str) -> str:
    """A branch off `main` with no commits of its own yet."""
    git(repo, "branch", name, "main")
    return name


def test_a_branch_that_has_not_moved_has_not_been_implemented(
    repo: Path, store: Store
) -> None:
    vcs = Git(repo)
    branch = branched(repo, "agl/add-auth/T-01")
    marked = Ticket(
        id="T-01",
        title="Add auth",
        status=Status.PENDING,
        deliverables=("auth.py",),
        base_sha=vcs.rev_parse(branch),
    )

    facts = look(vcs, store, marked, branch, "main")

    assert facts == Facts(implemented=False, merged=False, settled=False)


def test_a_ticket_with_no_base_sha_is_never_asked_about_its_branch(
    repo: Path, store: Store
) -> None:
    """A branch that does not exist yet would raise, so the mark guards the question."""
    vcs = Git(repo)

    facts = look(vcs, store, ticket("T-01"), "agl/add-auth/T-01", "main")

    assert facts.implemented is False
    assert facts.merged is False


def test_a_commit_on_the_branch_is_what_implemented_means(repo: Path, store: Store) -> None:
    vcs = Git(repo)
    branch = branched(repo, "agl/add-auth/T-01")
    base = vcs.rev_parse(branch)
    git(repo, "checkout", branch)
    commit_file(repo, "auth.py", "TOKEN = 1\n", "T-01: add auth")
    git(repo, "checkout", "main")
    marked = Ticket(
        id="T-01",
        title="Add auth",
        status=Status.PENDING,
        deliverables=("auth.py",),
        base_sha=base,
    )

    facts = look(vcs, store, marked, branch, "main")

    assert facts.implemented is True
    assert facts.merged is False
    assert step_for(facts) is Step.REVIEW


def test_a_branch_reachable_from_its_target_is_merged(repo: Path, store: Store) -> None:
    """And `base_sha` is why: after the merge, the branch is an ancestor of main
    exactly as it was before it had any commits at all."""
    vcs = Git(repo)
    branch = branched(repo, "agl/add-auth/T-01")
    base = vcs.rev_parse(branch)
    git(repo, "checkout", branch)
    commit_file(repo, "auth.py", "TOKEN = 1\n", "T-01: add auth")
    git(repo, "checkout", "main")
    vcs.merge(repo, branch)
    marked = Ticket(
        id="T-01",
        title="Add auth",
        status=Status.PENDING,
        deliverables=("auth.py",),
        base_sha=base,
    )

    facts = look(vcs, store, marked, branch, "main")

    assert facts.merged is True
    assert step_for(facts) is Step.DONE


def test_an_empty_branch_is_an_ancestor_too_and_is_still_not_merged(
    repo: Path, store: Store
) -> None:
    vcs = Git(repo)
    branch = branched(repo, "agl/add-auth/T-01")
    marked = Ticket(
        id="T-01",
        title="Add auth",
        status=Status.PENDING,
        deliverables=("auth.py",),
        base_sha=vcs.rev_parse(branch),
    )

    assert vcs.is_ancestor(branch, "main") is True
    assert look(vcs, store, marked, branch, "main").merged is False


# -- settled --------------------------------------------------------------


def test_a_round_with_every_document_and_no_groups_is_settled(
    repo: Path, store: Store
) -> None:
    settle(store, "T-01")

    assert look(Git(repo), store, ticket("T-01"), "agl/add-auth/T-01", "main").settled is True


def test_a_round_that_produced_groups_is_not_settled(repo: Path, store: Store) -> None:
    settle(
        store,
        "T-01",
        groups=[{"title": "Fix it", "deliverables": ["Fix it."], "findings": ["Q-1"]}],
    )

    assert look(Git(repo), store, ticket("T-01"), "agl/add-auth/T-01", "main").settled is False


@pytest.mark.parametrize("missing", [*REVIEWERS, "triage"])
def test_one_missing_document_leaves_the_round_unsettled(
    repo: Path, store: Store, missing: str
) -> None:
    settle(store, "T-01")
    store.delete(review_key("T-01", 0, missing))

    assert look(Git(repo), store, ticket("T-01"), "agl/add-auth/T-01", "main").settled is False


def test_settled_is_asked_about_this_round_and_no_other(repo: Path, store: Store) -> None:
    """A second round starts unsettled however clean the first one was."""
    settle(store, "T-01", round_=0)
    second = Ticket(
        id="T-01",
        title="Add auth",
        status=Status.PENDING,
        deliverables=("auth.py",),
        review_round=1,
    )

    assert look(Git(repo), store, second, "agl/add-auth/T-01", "main").settled is False

    settle(store, "T-01", round_=1)

    assert look(Git(repo), store, second, "agl/add-auth/T-01", "main").settled is True
