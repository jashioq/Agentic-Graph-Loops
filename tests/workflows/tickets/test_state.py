"""One run's state: the graph, the tickets, and keeping the two from disagreeing."""

from typing import Any

import pytest

from agl.core.dag import CycleError, Dag, NodeState
from agl.workflows.tickets.models import IllegalTransitionError, Status, Ticket
from agl.workflows.tickets.state import (
    DuplicateTicketError,
    Halt,
    InconsistentStateError,
    Live,
    RunState,
    UnknownTicketError,
    add_tickets,
    check_consistent,
    display_order,
    file_bugs,
    set_status,
)

# -- helpers --------------------------------------------------------------


def new_state() -> RunState:
    """An empty run, before anything has been decomposed."""
    return RunState(label="add-auth", base_branch="main", dag=Dag(), tickets={})


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


def snapshot(state: RunState) -> tuple[Any, ...]:
    """Everything in a `RunState` that a run's correctness depends on.

    `RunState` cannot be compared with `==` because a `Dag` is compared by
    identity, so the graph is flattened into something that can be.
    """
    return (
        state.label,
        state.base_branch,
        state.halt,
        dict(state.tickets),
        tuple(
            (node, state.dag.state(node), state.dag.blockers(node))
            for node in state.dag.nodes()
        ),
    )


def two_tickets() -> RunState:
    """`T-01`, and `T-02` waiting on it."""
    state = new_state()
    add_tickets(state, None, (feature("T-01"), feature("T-02", "T-01")))
    return state


def claimed_in_review(state: RunState, ticket_id: str) -> None:
    """Walk a ticket from pending to under review, as the workflow would."""
    set_status(state, None, ticket_id, Status.IN_PROGRESS)
    set_status(state, None, ticket_id, Status.IN_REVIEW)


# Every pairing of graph state and ticket status the invariant permits.
CONSISTENT = {
    (NodeState.PENDING, Status.PENDING),
    (NodeState.CLAIMED, Status.IN_PROGRESS),
    (NodeState.CLAIMED, Status.IN_REVIEW),
    (NodeState.CLAIMED, Status.MERGING),
    (NodeState.CLAIMED, Status.AWAITING_INPUT),
    (NodeState.DONE, Status.MERGED),
}

PAIRS = [(node, status) for node in NodeState for status in Status]


# -- add_tickets ----------------------------------------------------------


def test_add_tickets_builds_a_dag_whose_edges_match_blocked_by() -> None:
    state = new_state()

    add_tickets(state, None, (feature("T-01"), feature("T-02", "T-01"), feature("T-03", "T-01")))

    assert state.dag.nodes() == ("T-01", "T-02", "T-03")
    assert state.dag.blockers("T-02") == ("T-01",)
    assert state.dag.blockers("T-03") == ("T-01",)
    assert state.dag.blockers("T-01") == ()
    assert set(state.tickets) == {"T-01", "T-02", "T-03"}


def test_add_tickets_leaves_only_the_unblocked_tickets_ready() -> None:
    state = two_tickets()

    assert state.dag.ready() == ("T-01",)


def test_add_tickets_accepts_a_blocker_defined_later_in_the_same_batch() -> None:
    state = new_state()

    add_tickets(state, None, (feature("T-01", "T-02"), feature("T-02")))

    assert state.dag.blockers("T-01") == ("T-02",)
    assert state.dag.ready() == ("T-02",)


def test_add_tickets_stamps_every_new_ticket() -> None:
    state = new_state()
    live = Live(started_at=100.0)

    add_tickets(state, live, (feature("T-01"), feature("T-02", "T-01")), now=101.0)

    assert live.status_since == {"T-01": 101.0, "T-02": 101.0}


def test_add_tickets_refuses_an_id_the_run_already_holds() -> None:
    state = two_tickets()
    before = snapshot(state)

    with pytest.raises(DuplicateTicketError, match="T-01"):
        add_tickets(state, None, (feature("T-04"), feature("T-01")))

    assert snapshot(state) == before


def test_add_tickets_refuses_a_batch_that_repeats_an_id() -> None:
    state = new_state()

    with pytest.raises(DuplicateTicketError, match="T-01"):
        add_tickets(state, None, (feature("T-01"), feature("T-01")))

    assert snapshot(state) == snapshot(new_state())


def test_add_tickets_refuses_a_blocker_no_ticket_answers_to() -> None:
    state = new_state()

    with pytest.raises(UnknownTicketError, match="T-99"):
        add_tickets(state, None, (feature("T-01", "T-99"),))

    assert state.dag.nodes() == ()


def test_add_tickets_refuses_a_ticket_that_is_not_pending() -> None:
    state = new_state()
    started = feature("T-01")
    started.status = Status.IN_PROGRESS

    with pytest.raises(InconsistentStateError, match="T-01"):
        add_tickets(state, None, (started,))

    assert state.dag.nodes() == ()


def test_a_cycle_in_the_batch_leaves_the_run_untouched() -> None:
    state = two_tickets()
    before = snapshot(state)

    with pytest.raises(CycleError):
        add_tickets(state, None, (feature("T-03", "T-04"), feature("T-04", "T-03")))

    assert snapshot(state) == before


# -- the invariant --------------------------------------------------------


def test_a_fresh_state_is_consistent() -> None:
    check_consistent(new_state())
    check_consistent(two_tickets())


def test_check_consistent_passes_through_a_whole_life_cycle() -> None:
    state = two_tickets()
    check_consistent(state)

    for status in (Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING, Status.MERGED):
        set_status(state, None, "T-01", status)
        check_consistent(state)

    for status in (Status.IN_PROGRESS, Status.AWAITING_INPUT, Status.IN_PROGRESS):
        set_status(state, None, "T-02", status)
        check_consistent(state)


@pytest.mark.parametrize(("node", "status"), PAIRS)
def test_check_consistent_holds_the_table_exactly(node: NodeState, status: Status) -> None:
    state = new_state()
    add_tickets(state, None, (feature("T-01"),))
    if node is not NodeState.PENDING:
        state.dag.claim("T-01")
    if node is NodeState.DONE:
        state.dag.complete("T-01")
    # Straight past `set_status`, which is the only way to build the mismatch.
    state.tickets["T-01"].status = status

    if (node, status) in CONSISTENT:
        check_consistent(state)
    else:
        with pytest.raises(InconsistentStateError, match="T-01"):
            check_consistent(state)


def test_check_consistent_catches_a_ticket_with_no_node() -> None:
    state = two_tickets()
    state.tickets["T-99"] = feature("T-99")

    with pytest.raises(InconsistentStateError, match="T-99"):
        check_consistent(state)


def test_check_consistent_catches_a_node_with_no_ticket() -> None:
    state = two_tickets()
    state.dag.add_node("T-99")

    with pytest.raises(InconsistentStateError, match="T-99"):
        check_consistent(state)


# -- set_status -----------------------------------------------------------


def test_set_status_moves_the_ticket_and_the_graph_together() -> None:
    state = two_tickets()

    set_status(state, None, "T-01", Status.IN_PROGRESS)

    assert state.tickets["T-01"].status is Status.IN_PROGRESS
    assert state.dag.state("T-01") is NodeState.CLAIMED
    check_consistent(state)


def test_set_status_stamps_status_since() -> None:
    state = two_tickets()
    live = Live(started_at=100.0)

    set_status(state, live, "T-01", Status.IN_PROGRESS, now=140.0)

    assert live.status_since["T-01"] == 140.0


def test_re_entering_a_status_re_stamps_it() -> None:
    state = two_tickets()
    live = Live(started_at=100.0)
    set_status(state, live, "T-01", Status.IN_PROGRESS, now=140.0)

    set_status(state, live, "T-01", Status.IN_PROGRESS, now=200.0)

    assert live.status_since["T-01"] == 200.0
    assert state.tickets["T-01"].status is Status.IN_PROGRESS
    check_consistent(state)


def test_set_status_stamps_a_real_clock_when_it_is_not_given_one() -> None:
    state = two_tickets()
    live = Live(started_at=0.0)

    set_status(state, live, "T-01", Status.IN_PROGRESS)

    assert live.status_since["T-01"] > 0.0


def test_set_status_rejects_an_illegal_transition_and_changes_nothing() -> None:
    state = two_tickets()
    live = Live(started_at=100.0)
    set_status(state, live, "T-01", Status.IN_PROGRESS, now=140.0)
    before = snapshot(state)

    with pytest.raises(IllegalTransitionError):
        set_status(state, live, "T-01", Status.MERGED, now=200.0)

    assert snapshot(state) == before
    assert live.status_since["T-01"] == 140.0


def test_set_status_refuses_to_start_a_ticket_that_is_still_blocked() -> None:
    state = two_tickets()
    before = snapshot(state)

    with pytest.raises(ValueError, match="T-02"):
        set_status(state, None, "T-02", Status.IN_PROGRESS)

    assert snapshot(state) == before


def test_set_status_on_an_unknown_ticket_raises() -> None:
    state = two_tickets()

    with pytest.raises(UnknownTicketError, match="T-99"):
        set_status(state, None, "T-99", Status.IN_PROGRESS)


def test_a_question_and_its_answer_leave_the_graph_where_it_was() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    set_status(state, None, "T-01", Status.AWAITING_INPUT)
    assert state.dag.state("T-01") is NodeState.CLAIMED
    check_consistent(state)

    set_status(state, None, "T-01", Status.IN_REVIEW)
    assert state.dag.state("T-01") is NodeState.CLAIMED
    check_consistent(state)


def test_a_merge_failure_hands_the_ticket_back_to_the_queue() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")
    set_status(state, None, "T-01", Status.MERGING)

    set_status(state, None, "T-01", Status.PENDING)

    assert state.dag.state("T-01") is NodeState.PENDING
    assert state.dag.ready() == ("T-01",)
    check_consistent(state)


# -- file_bugs ------------------------------------------------------------


def test_file_bugs_puts_the_parent_behind_both_bugs() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    file_bugs(state, None, "T-01", (bug("T-01-bug-1", "T-01"), bug("T-01-bug-2", "T-01")))

    assert state.tickets["T-01"].status is Status.PENDING
    assert state.dag.unsatisfied_blockers("T-01") == ("T-01-bug-1", "T-01-bug-2")
    assert "T-01" not in state.dag.ready()
    assert state.dag.ready() == ("T-01-bug-1", "T-01-bug-2")
    assert state.dag.is_stalled() is False
    check_consistent(state)


def test_the_parent_becomes_ready_once_both_bugs_are_merged() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")
    file_bugs(state, None, "T-01", (bug("T-01-bug-1", "T-01"), bug("T-01-bug-2", "T-01")))

    for bug_id in ("T-01-bug-1", "T-01-bug-2"):
        # Bugs are not reviewed; the parent's next round covers them.
        for status in (Status.IN_PROGRESS, Status.MERGING, Status.MERGED):
            set_status(state, None, bug_id, status)
        check_consistent(state)

    assert state.dag.ready() == ("T-01",)


def test_file_bugs_stamps_the_bugs_and_the_parent() -> None:
    state = two_tickets()
    live = Live(started_at=100.0)
    claimed_in_review(state, "T-01")
    set_status(state, live, "T-01", Status.IN_REVIEW, now=140.0)

    file_bugs(state, live, "T-01", (bug("T-01-bug-1", "T-01"),), now=200.0)

    assert live.status_since["T-01"] == 200.0
    assert live.status_since["T-01-bug-1"] == 200.0


def test_file_bugs_with_a_colliding_id_raises_and_changes_nothing() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")
    before = snapshot(state)

    with pytest.raises(DuplicateTicketError, match="T-02"):
        file_bugs(state, None, "T-01", (bug("T-02", "T-01"),))

    assert snapshot(state) == before


def test_a_bug_may_not_be_blocked_by_the_ticket_it_fixes() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")
    before = snapshot(state)

    with pytest.raises(ValueError, match="T-01"):
        file_bugs(state, None, "T-01", (bug("T-01-bug-1", "T-01", "T-01"),))

    assert snapshot(state) == before


def test_file_bugs_refuses_a_ticket_that_is_not_a_bug_against_this_parent() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    with pytest.raises(ValueError, match="T-02"):
        file_bugs(state, None, "T-01", (bug("T-01-bug-1", "T-02"),))

    with pytest.raises(ValueError):
        file_bugs(state, None, "T-01", (feature("T-01-bug-1"),))


def test_file_bugs_refuses_an_empty_set_of_findings() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    with pytest.raises(ValueError, match="no bugs"):
        file_bugs(state, None, "T-01", ())


def test_a_bug_that_reaches_its_parent_the_long_way_round_is_refused() -> None:
    # T-02 already waits on T-01, so a bug on T-01 that waits on T-02 would
    # close the loop through a ticket rather than directly.
    state = two_tickets()
    live = Live(started_at=100.0)
    claimed_in_review(state, "T-01")
    before = snapshot(state)
    stamps = dict(live.status_since)

    with pytest.raises(CycleError):
        file_bugs(state, live, "T-01", (bug("T-01-bug-1", "T-01", "T-02"),), now=200.0)

    assert snapshot(state) == before
    assert live.status_since == stamps
    check_consistent(state)


def test_file_bugs_on_an_unknown_parent_raises() -> None:
    state = two_tickets()

    with pytest.raises(UnknownTicketError, match="T-99"):
        file_bugs(state, None, "T-99", (bug("T-99-bug-1", "T-99"),))


def test_bugs_may_be_blocked_by_each_other() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    file_bugs(
        state,
        None,
        "T-01",
        (bug("T-01-bug-1", "T-01"), bug("T-01-bug-2", "T-01", "T-01-bug-1")),
    )

    assert state.dag.ready() == ("T-01-bug-1",)
    check_consistent(state)


# -- display_order --------------------------------------------------------


def test_display_order_is_insertion_order_when_there_are_no_bugs() -> None:
    state = new_state()
    add_tickets(state, None, (feature("T-01"), feature("T-02"), feature("T-03")))

    assert display_order(state) == ("T-01", "T-02", "T-03")


def test_display_order_puts_bugs_immediately_after_their_parent() -> None:
    state = new_state()
    add_tickets(state, None, (feature("T-01"), feature("T-02"), feature("T-03")))
    claimed_in_review(state, "T-02")
    file_bugs(state, None, "T-02", (bug("T-02-bug-1", "T-02"), bug("T-02-bug-2", "T-02")))

    assert display_order(state) == ("T-01", "T-02", "T-02-bug-1", "T-02-bug-2", "T-03")


def test_a_bug_filed_later_still_appears_under_its_parent() -> None:
    state = new_state()
    add_tickets(state, None, (feature("T-01"), feature("T-02")))
    claimed_in_review(state, "T-01")
    file_bugs(state, None, "T-01", (bug("T-01-bug-1", "T-01"),))
    claimed_in_review(state, "T-02")
    file_bugs(state, None, "T-02", (bug("T-02-bug-1", "T-02"),))

    assert display_order(state) == ("T-01", "T-01-bug-1", "T-02", "T-02-bug-1")


def test_display_order_is_stable_across_calls() -> None:
    state = new_state()
    add_tickets(state, None, (feature("T-01"), feature("T-02")))
    claimed_in_review(state, "T-01")
    file_bugs(state, None, "T-01", (bug("T-01-bug-1", "T-01"),))

    assert display_order(state) == display_order(state) == display_order(state)


def test_display_order_covers_every_ticket_exactly_once() -> None:
    state = new_state()
    add_tickets(state, None, (feature("T-01"), feature("T-02")))
    claimed_in_review(state, "T-01")
    file_bugs(state, None, "T-01", (bug("T-01-bug-1", "T-01"),))

    order = display_order(state)

    assert sorted(order) == sorted(state.tickets)
    assert len(set(order)) == len(order)


# -- the lifetime split ---------------------------------------------------


def run_a_sequence(state: RunState, live: Live | None) -> None:
    """Everything a run does to its state, in the order a run would do it."""
    add_tickets(state, live, (feature("T-01"), feature("T-02", "T-01")))
    set_status(state, live, "T-01", Status.IN_PROGRESS)
    set_status(state, live, "T-01", Status.AWAITING_INPUT)
    set_status(state, live, "T-01", Status.IN_PROGRESS)
    set_status(state, live, "T-01", Status.IN_REVIEW)
    file_bugs(state, live, "T-01", (bug("T-01-bug-1", "T-01"),))
    for status in (Status.IN_PROGRESS, Status.MERGING, Status.MERGED):
        set_status(state, live, "T-01-bug-1", status)
    state.halt = Halt(reason="asked", detail="the user stopped the run")


def test_the_run_state_does_not_depend_on_live_existing() -> None:
    with_live = new_state()
    without_live = new_state()

    run_a_sequence(with_live, Live(started_at=100.0))
    run_a_sequence(without_live, None)

    assert snapshot(with_live) == snapshot(without_live)
    assert display_order(with_live) == display_order(without_live)
    check_consistent(without_live)


def test_live_holds_nothing_the_run_state_needs() -> None:
    live = Live(started_at=100.0)
    state = new_state()
    run_a_sequence(state, live)

    live.activity["T-01"] = "reading src/auth/store.py"

    assert set(live.status_since) == set(state.tickets)
    check_consistent(state)


def test_halt_carries_a_reason_and_a_detail() -> None:
    halt = Halt(reason="budget", detail="ran out at $4.10")

    assert halt.reason == "budget"
    assert halt.detail == "ran out at $4.10"
    assert new_state().halt is None


def test_halt_defaults_to_resumable() -> None:
    assert Halt(reason="budget").resumable is True


def test_halt_resumable_can_be_set_false() -> None:
    assert Halt(reason="budget", resumable=False).resumable is False
