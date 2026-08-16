"""One run's state as a value: the transitions, the derivations, and `check`.

Every function under test is pure, so every test here is "build a `Run`, apply
one thing, look at what came back" — and the run that went in is looked at too,
because a transition that mutated its argument would be a second writer nobody
asked for.
"""

from dataclasses import replace

import pytest

from agl.runtime.dag import NodeState
from agl.workflows.tickets.errors import (
    DuplicateTicketError,
    Halt,
    IllegalTransitionError,
    InvalidStateError,
    UnknownTicketError,
)
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import (
    Run,
    check,
    dag_of,
    display_order,
    with_base_sha,
    with_bugs,
    with_halt,
    with_status,
    with_tickets,
)

# -- helpers --------------------------------------------------------------


def feature(ticket_id: str, *blocked_by: str) -> Ticket:
    """A pending feature ticket, optionally waiting on others."""
    return Ticket(
        id=ticket_id,
        title=f"Do {ticket_id}",
        status=Status.PENDING,
        deliverables=(f"{ticket_id}.py",),
        blocked_by=blocked_by,
    )


def bug(ticket_id: str, parent: str, *blocked_by: str) -> Ticket:
    """A pending bug ticket against `parent`, carrying one finding."""
    return Ticket(
        id=ticket_id,
        title=f"Fix {ticket_id}",
        status=Status.PENDING,
        deliverables=("the finding",),
        blocked_by=blocked_by,
        parent=parent,
    )


def run_of(*tickets: Ticket) -> Run:
    """A run holding `tickets`, built the way an approval builds one."""
    return with_tickets(Run(), tickets)


def two_tickets() -> Run:
    """`T-01`, and `T-02` waiting on it."""
    return run_of(feature("T-01"), feature("T-02", "T-01"))


def in_review(run: Run, ticket_id: str) -> Run:
    """Walk a ticket from pending to under review, as the workflow would."""
    return with_status(with_status(run, ticket_id, Status.IN_PROGRESS), ticket_id, Status.IN_REVIEW)


def merged(run: Run, ticket_id: str) -> Run:
    """Walk a ticket all the way to merged."""
    for status in (Status.IN_PROGRESS, Status.MERGING, Status.MERGED):
        run = with_status(run, ticket_id, status)
    return run


def states(run: Run) -> dict[str, NodeState]:
    node = dag_of(run)
    return {n: node.state(n) for n in node.nodes()}


# -- with_tickets ---------------------------------------------------------


def test_with_tickets_keeps_insertion_order() -> None:
    run = run_of(feature("T-01"), feature("T-02", "T-01"), feature("T-03", "T-01"))

    assert [t.id for t in run.tickets] == ["T-01", "T-02", "T-03"]


def test_the_run_that_went_in_is_left_alone() -> None:
    empty = Run()

    with_tickets(empty, (feature("T-01"),))

    assert empty.tickets == ()


def test_blocked_by_becomes_the_graphs_edges() -> None:
    run = run_of(feature("T-01"), feature("T-02", "T-01"))
    graph = dag_of(run)

    assert graph.blockers("T-02") == ("T-01",)
    assert graph.blockers("T-01") == ()
    assert graph.ready() == ("T-01",)


def test_a_blocker_may_be_defined_later_in_the_same_batch() -> None:
    run = run_of(feature("T-01", "T-02"), feature("T-02"))

    assert dag_of(run).blockers("T-01") == ("T-02",)
    assert dag_of(run).ready() == ("T-02",)


def test_a_second_batch_may_name_a_blocker_already_in_the_run() -> None:
    run = with_tickets(run_of(feature("T-01")), (feature("T-02", "T-01"),))

    assert dag_of(run).blockers("T-02") == ("T-01",)


def test_an_id_the_run_already_holds_is_refused() -> None:
    run = two_tickets()

    with pytest.raises(DuplicateTicketError, match="T-01"):
        with_tickets(run, (feature("T-04"), feature("T-01")))

    assert run == two_tickets()


def test_a_batch_that_repeats_an_id_is_refused() -> None:
    with pytest.raises(DuplicateTicketError, match="T-01"):
        with_tickets(Run(), (feature("T-01"), feature("T-01")))


def test_a_blocker_no_ticket_answers_to_is_refused() -> None:
    with pytest.raises(UnknownTicketError, match="T-99"):
        with_tickets(Run(), (feature("T-01", "T-99"),))


def test_a_ticket_that_is_not_pending_is_refused() -> None:
    started = replace(feature("T-01"), status=Status.IN_PROGRESS)

    with pytest.raises(InvalidStateError, match="T-01"):
        with_tickets(Run(), (started,))


def test_a_cycle_within_the_batch_is_refused() -> None:
    run = two_tickets()

    with pytest.raises(InvalidStateError, match="cycle"):
        with_tickets(run, (feature("T-03", "T-04"), feature("T-04", "T-03")))

    assert run == two_tickets()


# -- with_status ----------------------------------------------------------


def test_with_status_moves_the_ticket_and_the_derived_graph_together() -> None:
    run = with_status(two_tickets(), "T-01", Status.IN_PROGRESS)

    assert run.ticket("T-01").status is Status.IN_PROGRESS
    assert states(run)["T-01"] is NodeState.CLAIMED


def test_with_status_refuses_an_illegal_move_and_builds_nothing() -> None:
    run = with_status(two_tickets(), "T-01", Status.IN_PROGRESS)

    with pytest.raises(IllegalTransitionError):
        with_status(run, "T-01", Status.MERGED)

    assert run.ticket("T-01").status is Status.IN_PROGRESS


def test_with_status_on_an_unknown_ticket_raises() -> None:
    with pytest.raises(UnknownTicketError, match="T-99"):
        with_status(two_tickets(), "T-99", Status.IN_PROGRESS)


def test_a_question_and_its_answer_leave_the_graph_where_it_was() -> None:
    run = in_review(two_tickets(), "T-01")

    waiting = with_status(run, "T-01", Status.AWAITING_INPUT)
    assert states(waiting)["T-01"] is NodeState.CLAIMED
    assert waiting.ticket("T-01").resume_to is Status.IN_REVIEW

    back = with_status(waiting, "T-01", Status.IN_REVIEW)
    assert states(back)["T-01"] is NodeState.CLAIMED
    assert back.ticket("T-01").resume_to is None


def test_a_merge_failure_hands_the_ticket_back_to_the_queue() -> None:
    run = with_status(in_review(two_tickets(), "T-01"), "T-01", Status.MERGING)

    given_back = with_status(run, "T-01", Status.PENDING)

    assert states(given_back)["T-01"] is NodeState.PENDING
    assert dag_of(given_back).ready() == ("T-01",)


def test_a_pending_ticket_may_be_claimed_straight_into_any_status() -> None:
    """A resumed run reads git and claims a ticket where the repository says it is."""
    run = two_tickets()

    for status in (Status.IN_REVIEW, Status.MERGING, Status.MERGED):
        assert with_status(run, "T-01", status).ticket("T-01").status is status


@pytest.mark.parametrize("claimed", [Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING])
def test_a_claim_can_be_given_back_from_wherever_it_got_to(claimed: Status) -> None:
    """What the scheduler does with a ticket whose pass raised: it goes back in
    the queue, and its branch is still where its work left it."""
    run = with_status(two_tickets(), "T-01", claimed)

    given_back = with_status(run, "T-01", Status.PENDING)

    assert states(given_back)["T-01"] is NodeState.PENDING


# -- with_bugs ------------------------------------------------------------


def filed(count: int = 2) -> Run:
    """`T-01` under review, sent back behind `count` bugs."""
    run = in_review(two_tickets(), "T-01")
    return with_bugs(run, "T-01", [bug(f"T-01-bug-{n}", "T-01") for n in range(1, count + 1)])


def test_with_bugs_puts_the_parent_behind_every_bug() -> None:
    run = filed()

    assert run.ticket("T-01").status is Status.PENDING
    assert run.ticket("T-01").blocked_by == ("T-01-bug-1", "T-01-bug-2")
    assert dag_of(run).unsatisfied_blockers("T-01") == ("T-01-bug-1", "T-01-bug-2")
    assert dag_of(run).ready() == ("T-01-bug-1", "T-01-bug-2")
    assert dag_of(run).is_stalled() is False


def test_blocked_by_is_the_graph_so_the_edges_survive_a_round_trip() -> None:
    """Nothing keeps the parent→bug edges but the parent's own `blocked_by`."""
    run = filed()

    rebuilt = Run(tickets=run.tickets)

    assert dag_of(rebuilt).unsatisfied_blockers("T-01") == ("T-01-bug-1", "T-01-bug-2")


def test_the_parent_becomes_ready_once_every_bug_is_merged() -> None:
    run = filed()

    for bug_id in ("T-01-bug-1", "T-01-bug-2"):
        run = merged(run, bug_id)

    assert dag_of(run).ready() == ("T-01",)


def test_with_bugs_advances_the_parents_review_round() -> None:
    run = filed(1)

    assert run.ticket("T-01").review_round == 1
    assert run.ticket("T-01-bug-1").review_round == 0


def test_a_second_round_advances_the_round_again() -> None:
    run = merged(filed(1), "T-01-bug-1")
    run = in_review(run, "T-01")

    run = with_bugs(run, "T-01", (bug("T-01-bug-2", "T-01"),))

    assert run.ticket("T-01").review_round == 2
    assert run.ticket("T-01").blocked_by == ("T-01-bug-1", "T-01-bug-2")


def test_a_refused_filing_leaves_the_run_exactly_as_it_was() -> None:
    run = in_review(two_tickets(), "T-01")

    with pytest.raises(DuplicateTicketError, match="T-02"):
        with_bugs(run, "T-01", (bug("T-02", "T-01"),))

    assert run.ticket("T-01").review_round == 0
    assert run == in_review(two_tickets(), "T-01")


def test_a_bug_may_not_be_blocked_by_the_ticket_it_fixes() -> None:
    run = in_review(two_tickets(), "T-01")

    with pytest.raises(ValueError, match="T-01"):
        with_bugs(run, "T-01", (bug("T-01-bug-1", "T-01", "T-01"),))


def test_a_bug_that_reaches_its_parent_the_long_way_round_is_refused() -> None:
    # T-02 already waits on T-01, so a bug on T-01 that waits on T-02 would
    # close the loop through a ticket rather than directly.
    run = in_review(two_tickets(), "T-01")

    with pytest.raises(InvalidStateError, match="cycle"):
        with_bugs(run, "T-01", (bug("T-01-bug-1", "T-01", "T-02"),))

    assert run == in_review(two_tickets(), "T-01")


def test_with_bugs_refuses_a_ticket_that_is_not_a_bug_against_this_parent() -> None:
    run = in_review(two_tickets(), "T-01")

    with pytest.raises(ValueError, match="T-02"):
        with_bugs(run, "T-01", (bug("T-01-bug-1", "T-02"),))

    with pytest.raises(ValueError):
        with_bugs(run, "T-01", (feature("T-01-bug-1"),))


def test_with_bugs_refuses_an_empty_set_of_findings() -> None:
    with pytest.raises(ValueError, match="no bugs"):
        with_bugs(in_review(two_tickets(), "T-01"), "T-01", ())


def test_with_bugs_on_an_unknown_parent_raises() -> None:
    with pytest.raises(UnknownTicketError, match="T-99"):
        with_bugs(two_tickets(), "T-99", (bug("T-99-bug-1", "T-99"),))


def test_bugs_may_be_blocked_by_each_other() -> None:
    run = in_review(two_tickets(), "T-01")

    run = with_bugs(
        run, "T-01", (bug("T-01-bug-1", "T-01"), bug("T-01-bug-2", "T-01", "T-01-bug-1"))
    )

    assert dag_of(run).ready() == ("T-01-bug-1",)


# -- with_halt and with_base_sha ------------------------------------------


def test_with_halt_sets_and_clears() -> None:
    halt = Halt(reason="merge conflict")
    stopped = with_halt(two_tickets(), halt)

    assert stopped.halt == halt
    assert with_halt(stopped, None).halt is None
    assert two_tickets().halt is None


def test_with_base_sha_records_the_mark_on_one_ticket_only() -> None:
    run = with_base_sha(two_tickets(), "T-01", "abc123")

    assert run.ticket("T-01").base_sha == "abc123"
    assert run.ticket("T-02").base_sha is None


def test_with_base_sha_on_an_unknown_ticket_raises() -> None:
    with pytest.raises(UnknownTicketError, match="T-99"):
        with_base_sha(two_tickets(), "T-99", "abc123")


# -- ticket lookup --------------------------------------------------------


def test_get_answers_none_rather_than_raising() -> None:
    run = two_tickets()

    assert run.get("T-01") is not None
    assert run.get("T-99") is None


def test_ticket_raises_for_an_id_the_run_does_not_hold() -> None:
    with pytest.raises(UnknownTicketError, match="T-99"):
        two_tickets().ticket("T-99")


# -- dag_of ---------------------------------------------------------------


def test_a_graph_is_derived_fresh_and_never_kept() -> None:
    run = two_tickets()

    first = dag_of(run)
    first.claim("T-01")

    assert dag_of(run).state("T-01") is NodeState.PENDING


@pytest.mark.parametrize(
    ("status", "node"),
    [
        (Status.PENDING, NodeState.PENDING),
        (Status.IN_PROGRESS, NodeState.CLAIMED),
        (Status.IN_REVIEW, NodeState.CLAIMED),
        (Status.MERGING, NodeState.CLAIMED),
        (Status.AWAITING_INPUT, NodeState.CLAIMED),
        (Status.MERGED, NodeState.DONE),
    ],
)
def test_every_status_states_its_node(status: Status, node: NodeState) -> None:
    run = Run(tickets=(replace(feature("T-01"), status=status, resume_to=Status.IN_PROGRESS),))

    assert dag_of(run).state("T-01") is node


def test_bugs_first_puts_ready_bugs_ahead_of_ready_features() -> None:
    run = run_of(feature("F1"), feature("F2"), bug("B1", "F1"), feature("F3"), bug("B2", "F1"))

    assert dag_of(run).ready() == ("B1", "B2", "F1", "F2", "F3")


def test_bugs_first_preserves_insertion_order_within_each_group() -> None:
    run = run_of(feature("F2"), bug("B2", "F2"), feature("F1"), bug("B1", "F1"))

    assert dag_of(run).ready() == ("B2", "B1", "F2", "F1")


# -- display_order --------------------------------------------------------


def test_display_order_is_insertion_order_when_there_are_no_bugs() -> None:
    run = run_of(feature("T-01"), feature("T-02"), feature("T-03"))

    assert display_order(run) == ("T-01", "T-02", "T-03")


def test_display_order_puts_bugs_immediately_after_their_parent() -> None:
    run = run_of(feature("T-01"), feature("T-02"), feature("T-03"))
    run = with_bugs(
        in_review(run, "T-02"), "T-02", (bug("T-02-bug-1", "T-02"), bug("T-02-bug-2", "T-02"))
    )

    assert display_order(run) == ("T-01", "T-02", "T-02-bug-1", "T-02-bug-2", "T-03")


def test_a_bug_filed_later_still_appears_under_its_parent() -> None:
    run = run_of(feature("T-01"), feature("T-02"))
    run = with_bugs(in_review(run, "T-01"), "T-01", (bug("T-01-bug-1", "T-01"),))
    run = with_bugs(in_review(run, "T-02"), "T-02", (bug("T-02-bug-1", "T-02"),))

    assert display_order(run) == ("T-01", "T-01-bug-1", "T-02", "T-02-bug-1")


def test_display_order_covers_every_ticket_exactly_once() -> None:
    run = filed()

    order = display_order(run)

    assert sorted(order) == sorted(t.id for t in run.tickets)
    assert len(set(order)) == len(order)


# -- check ----------------------------------------------------------------


def test_a_run_the_transitions_built_always_checks_out() -> None:
    check(Run())
    check(two_tickets())
    check(filed())


def test_check_catches_a_duplicate_id() -> None:
    run = Run(tickets=(feature("T-01"), feature("T-01")))

    with pytest.raises(InvalidStateError, match="T-01"):
        check(run)


def test_check_catches_a_blocker_that_is_not_in_the_run() -> None:
    run = Run(tickets=(feature("T-01", "T-99"),))

    with pytest.raises(InvalidStateError, match="T-99"):
        check(run)


def test_check_catches_a_parent_that_is_not_in_the_run() -> None:
    run = Run(tickets=(bug("T-01-bug-1", "T-01"),))

    with pytest.raises(InvalidStateError, match="T-01"):
        check(run)


def test_check_catches_a_waiting_ticket_with_nowhere_to_return_to() -> None:
    run = Run(tickets=(replace(feature("T-01"), status=Status.AWAITING_INPUT),))

    with pytest.raises(InvalidStateError, match="return to"):
        check(run)


def test_check_catches_a_cycle() -> None:
    run = Run(tickets=(feature("T-01", "T-02"), feature("T-02", "T-01")))

    with pytest.raises(InvalidStateError, match="cycle"):
        check(run)
