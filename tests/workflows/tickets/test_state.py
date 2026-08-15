"""One run's state: the graph, the tickets, and keeping the two from disagreeing."""

from collections.abc import Sequence
from typing import Any

import pytest

from agl.core.command import ExecResult
from agl.runtime.dag import CycleError, NodeState
from agl.runtime.display import Board
from agl.runtime.merge import MergeOutcome, MergeStatus
from agl.workflows.tickets.models import IllegalTransitionError, Status, Ticket
from agl.workflows.tickets.state import (
    TAIL_LINES,
    DuplicateTicketError,
    Halt,
    InconsistentStateError,
    RunState,
    UnknownTicketError,
    halt_for,
)

# -- helpers --------------------------------------------------------------


def new_state() -> RunState:
    """An empty run, before anything has been decomposed."""
    return RunState("add-auth", "main", Board(started_at=100.0))


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
    state.add((feature("T-01"), feature("T-02", "T-01")))
    return state


def claimed_in_review(state: RunState, ticket_id: str) -> None:
    """Walk a ticket from pending to under review, as the workflow would."""
    state.set_status(ticket_id, Status.IN_PROGRESS)
    state.set_status(ticket_id, Status.IN_REVIEW)


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

    state.add((feature("T-01"), feature("T-02", "T-01"), feature("T-03", "T-01")))

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

    state.add((feature("T-01", "T-02"), feature("T-02")))

    assert state.dag.blockers("T-01") == ("T-02",)
    assert state.dag.ready() == ("T-02",)


def test_add_tickets_stamps_every_new_ticket() -> None:
    state = new_state()

    state.add((feature("T-01"), feature("T-02", "T-01")), now=101.0)

    assert state.board.status_since == {"T-01": 101.0, "T-02": 101.0}


def test_add_tickets_refuses_an_id_the_run_already_holds() -> None:
    state = two_tickets()
    before = snapshot(state)

    with pytest.raises(DuplicateTicketError, match="T-01"):
        state.add((feature("T-04"), feature("T-01")))

    assert snapshot(state) == before


def test_add_tickets_refuses_a_batch_that_repeats_an_id() -> None:
    state = new_state()

    with pytest.raises(DuplicateTicketError, match="T-01"):
        state.add((feature("T-01"), feature("T-01")))

    assert snapshot(state) == snapshot(new_state())


def test_add_tickets_refuses_a_blocker_no_ticket_answers_to() -> None:
    state = new_state()

    with pytest.raises(UnknownTicketError, match="T-99"):
        state.add((feature("T-01", "T-99"),))

    assert state.dag.nodes() == ()


def test_add_tickets_refuses_a_ticket_that_is_not_pending() -> None:
    state = new_state()
    started = feature("T-01")
    started.status = Status.IN_PROGRESS

    with pytest.raises(InconsistentStateError, match="T-01"):
        state.add((started,))

    assert state.dag.nodes() == ()


def test_a_cycle_in_the_batch_leaves_the_run_untouched() -> None:
    state = two_tickets()
    before = snapshot(state)

    with pytest.raises(CycleError):
        state.add((feature("T-03", "T-04"), feature("T-04", "T-03")))

    assert snapshot(state) == before


# -- the invariant --------------------------------------------------------


def test_a_fresh_state_is_consistent() -> None:
    new_state().check_consistent()
    two_tickets().check_consistent()


def test_check_consistent_passes_through_a_whole_life_cycle() -> None:
    state = two_tickets()
    state.check_consistent()

    for status in (Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING, Status.MERGED):
        state.set_status("T-01", status)
        state.check_consistent()

    for status in (Status.IN_PROGRESS, Status.AWAITING_INPUT, Status.IN_PROGRESS):
        state.set_status("T-02", status)
        state.check_consistent()


@pytest.mark.parametrize(("node", "status"), PAIRS)
def test_check_consistent_holds_the_table_exactly(node: NodeState, status: Status) -> None:
    state = new_state()
    state.add((feature("T-01"),))
    if node is not NodeState.PENDING:
        state.dag.claim("T-01")
    if node is NodeState.DONE:
        state.dag.complete("T-01")
    # Straight past `set_status`, which is the only way to build the mismatch.
    state.tickets["T-01"].status = status

    if (node, status) in CONSISTENT:
        state.check_consistent()
    else:
        with pytest.raises(InconsistentStateError, match="T-01"):
            state.check_consistent()


def test_check_consistent_catches_a_ticket_with_no_node() -> None:
    state = two_tickets()
    state.tickets["T-99"] = feature("T-99")

    with pytest.raises(InconsistentStateError, match="T-99"):
        state.check_consistent()


def test_check_consistent_catches_a_node_with_no_ticket() -> None:
    state = two_tickets()
    state.dag.add_node("T-99")

    with pytest.raises(InconsistentStateError, match="T-99"):
        state.check_consistent()


# -- set_status -----------------------------------------------------------


def test_set_status_moves_the_ticket_and_the_graph_together() -> None:
    state = two_tickets()

    state.set_status("T-01", Status.IN_PROGRESS)

    assert state.tickets["T-01"].status is Status.IN_PROGRESS
    assert state.dag.state("T-01") is NodeState.CLAIMED
    state.check_consistent()


def test_set_status_stamps_status_since() -> None:
    state = two_tickets()

    state.set_status("T-01", Status.IN_PROGRESS, now=140.0)

    assert state.board.status_since["T-01"] == 140.0


def test_re_entering_a_status_re_stamps_it() -> None:
    state = two_tickets()
    state.set_status("T-01", Status.IN_PROGRESS, now=140.0)

    state.set_status("T-01", Status.IN_PROGRESS, now=200.0)

    assert state.board.status_since["T-01"] == 200.0
    assert state.tickets["T-01"].status is Status.IN_PROGRESS
    state.check_consistent()


def test_set_status_stamps_a_real_clock_when_it_is_not_given_one() -> None:
    state = two_tickets()

    state.set_status("T-01", Status.IN_PROGRESS)

    assert state.board.status_since["T-01"] > 0.0


def test_set_status_rejects_an_illegal_transition_and_changes_nothing() -> None:
    state = two_tickets()
    state.set_status("T-01", Status.IN_PROGRESS, now=140.0)
    before = snapshot(state)

    with pytest.raises(IllegalTransitionError):
        state.set_status("T-01", Status.MERGED, now=200.0)

    assert snapshot(state) == before
    assert state.board.status_since["T-01"] == 140.0


def test_set_status_refuses_to_start_a_ticket_that_is_still_blocked() -> None:
    state = two_tickets()
    before = snapshot(state)

    with pytest.raises(ValueError, match="T-02"):
        state.set_status("T-02", Status.IN_PROGRESS)

    assert snapshot(state) == before


def test_set_status_on_an_unknown_ticket_raises() -> None:
    state = two_tickets()

    with pytest.raises(UnknownTicketError, match="T-99"):
        state.set_status("T-99", Status.IN_PROGRESS)


def test_a_question_and_its_answer_leave_the_graph_where_it_was() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    state.set_status("T-01", Status.AWAITING_INPUT)
    assert state.dag.state("T-01") is NodeState.CLAIMED
    state.check_consistent()

    state.set_status("T-01", Status.IN_REVIEW)
    assert state.dag.state("T-01") is NodeState.CLAIMED
    state.check_consistent()


def test_a_merge_failure_hands_the_ticket_back_to_the_queue() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")
    state.set_status("T-01", Status.MERGING)

    state.set_status("T-01", Status.PENDING)

    assert state.dag.state("T-01") is NodeState.PENDING
    assert state.dag.ready() == ("T-01",)
    state.check_consistent()


# -- file_bugs ------------------------------------------------------------


def test_file_bugs_puts_the_parent_behind_both_bugs() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    state.file_bugs("T-01", (bug("T-01-bug-1", "T-01"), bug("T-01-bug-2", "T-01")))

    assert state.tickets["T-01"].status is Status.PENDING
    assert state.dag.unsatisfied_blockers("T-01") == ("T-01-bug-1", "T-01-bug-2")
    assert "T-01" not in state.dag.ready()
    assert state.dag.ready() == ("T-01-bug-1", "T-01-bug-2")
    assert state.dag.is_stalled() is False
    state.check_consistent()


def test_the_parent_becomes_ready_once_both_bugs_are_merged() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")
    state.file_bugs("T-01", (bug("T-01-bug-1", "T-01"), bug("T-01-bug-2", "T-01")))

    for bug_id in ("T-01-bug-1", "T-01-bug-2"):
        # Bugs are not reviewed; the parent's next round covers them.
        for status in (Status.IN_PROGRESS, Status.MERGING, Status.MERGED):
            state.set_status(bug_id, status)
        state.check_consistent()

    assert state.dag.ready() == ("T-01",)


def test_file_bugs_stamps_the_bugs_and_the_parent() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")
    state.set_status("T-01", Status.IN_REVIEW, now=140.0)

    state.file_bugs("T-01", (bug("T-01-bug-1", "T-01"),), now=200.0)

    assert state.board.status_since["T-01"] == 200.0
    assert state.board.status_since["T-01-bug-1"] == 200.0


def test_file_bugs_with_a_colliding_id_raises_and_changes_nothing() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")
    before = snapshot(state)

    with pytest.raises(DuplicateTicketError, match="T-02"):
        state.file_bugs("T-01", (bug("T-02", "T-01"),))

    assert snapshot(state) == before


def test_a_bug_may_not_be_blocked_by_the_ticket_it_fixes() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")
    before = snapshot(state)

    with pytest.raises(ValueError, match="T-01"):
        state.file_bugs("T-01", (bug("T-01-bug-1", "T-01", "T-01"),))

    assert snapshot(state) == before


def test_file_bugs_refuses_a_ticket_that_is_not_a_bug_against_this_parent() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    with pytest.raises(ValueError, match="T-02"):
        state.file_bugs("T-01", (bug("T-01-bug-1", "T-02"),))

    with pytest.raises(ValueError):
        state.file_bugs("T-01", (feature("T-01-bug-1"),))


def test_file_bugs_refuses_an_empty_set_of_findings() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    with pytest.raises(ValueError, match="no bugs"):
        state.file_bugs("T-01", ())


def test_a_bug_that_reaches_its_parent_the_long_way_round_is_refused() -> None:
    # T-02 already waits on T-01, so a bug on T-01 that waits on T-02 would
    # close the loop through a ticket rather than directly.
    state = two_tickets()
    claimed_in_review(state, "T-01")
    before = snapshot(state)
    stamps = dict(state.board.status_since)

    with pytest.raises(CycleError):
        state.file_bugs("T-01", (bug("T-01-bug-1", "T-01", "T-02"),), now=200.0)

    assert snapshot(state) == before
    assert state.board.status_since == stamps
    state.check_consistent()


def test_file_bugs_on_an_unknown_parent_raises() -> None:
    state = two_tickets()

    with pytest.raises(UnknownTicketError, match="T-99"):
        state.file_bugs("T-99", (bug("T-99-bug-1", "T-99"),))


def test_bugs_may_be_blocked_by_each_other() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    state.file_bugs("T-01",
        (bug("T-01-bug-1", "T-01"), bug("T-01-bug-2", "T-01", "T-01-bug-1")),
    )

    assert state.dag.ready() == ("T-01-bug-1",)
    state.check_consistent()


# -- display_order --------------------------------------------------------


def test_display_order_is_insertion_order_when_there_are_no_bugs() -> None:
    state = new_state()
    state.add((feature("T-01"), feature("T-02"), feature("T-03")))

    assert state.display_order() == ("T-01", "T-02", "T-03")


def test_display_order_puts_bugs_immediately_after_their_parent() -> None:
    state = new_state()
    state.add((feature("T-01"), feature("T-02"), feature("T-03")))
    claimed_in_review(state, "T-02")
    state.file_bugs("T-02", (bug("T-02-bug-1", "T-02"), bug("T-02-bug-2", "T-02")))

    assert state.display_order() == ("T-01", "T-02", "T-02-bug-1", "T-02-bug-2", "T-03")


def test_a_bug_filed_later_still_appears_under_its_parent() -> None:
    state = new_state()
    state.add((feature("T-01"), feature("T-02")))
    claimed_in_review(state, "T-01")
    state.file_bugs("T-01", (bug("T-01-bug-1", "T-01"),))
    claimed_in_review(state, "T-02")
    state.file_bugs("T-02", (bug("T-02-bug-1", "T-02"),))

    assert state.display_order() == ("T-01", "T-01-bug-1", "T-02", "T-02-bug-1")


def test_display_order_is_stable_across_calls() -> None:
    state = new_state()
    state.add((feature("T-01"), feature("T-02")))
    claimed_in_review(state, "T-01")
    state.file_bugs("T-01", (bug("T-01-bug-1", "T-01"),))

    assert state.display_order() == state.display_order() == state.display_order()


def test_display_order_covers_every_ticket_exactly_once() -> None:
    state = new_state()
    state.add((feature("T-01"), feature("T-02")))
    claimed_in_review(state, "T-01")
    state.file_bugs("T-01", (bug("T-01-bug-1", "T-01"),))

    order = state.display_order()

    assert sorted(order) == sorted(state.tickets)
    assert len(set(order)) == len(order)


# -- bugs_first -----------------------------------------------------------


def prioritised(tickets: Sequence[Ticket]) -> RunState:
    """A run holding `tickets`. Every `RunState` sorts its ready set bugs-first."""
    state = new_state()
    state.add(tickets)
    return state


def test_bugs_first_puts_ready_bugs_ahead_of_ready_features() -> None:
    tickets = [feature("F1"), feature("F2"), bug("B1", "F1"), feature("F3"), bug("B2", "F1")]
    state = prioritised(tickets)

    assert state.dag.ready() == ("B1", "B2", "F1", "F2", "F3")


def test_bugs_first_preserves_insertion_order_within_each_group() -> None:
    tickets = [feature("F2"), bug("B2", "F2"), feature("F1"), bug("B1", "F1")]
    state = prioritised(tickets)

    assert state.dag.ready() == ("B2", "B1", "F2", "F1")


# -- the lifetime split ---------------------------------------------------


def run_a_sequence(state: RunState) -> None:
    """Everything a run does to its state, in the order a run would do it."""
    state.add((feature("T-01"), feature("T-02", "T-01")))
    state.set_status("T-01", Status.IN_PROGRESS)
    state.set_status("T-01", Status.AWAITING_INPUT)
    state.set_status("T-01", Status.IN_PROGRESS)
    state.set_status("T-01", Status.IN_REVIEW)
    state.file_bugs("T-01", (bug("T-01-bug-1", "T-01"),))
    for status in (Status.IN_PROGRESS, Status.MERGING, Status.MERGED):
        state.set_status("T-01-bug-1", status)
    state.halt = Halt(reason="asked", detail="the user stopped the run")


def test_the_run_state_does_not_depend_on_what_the_board_holds() -> None:
    """Nothing on the board is ever read back to decide anything.

    Two identical runs, one watched through a board somebody else has been
    scribbling on, and the state they arrive at is the same — which is what
    keeps the display-only rule real rather than aspirational.
    """
    watched = new_state()
    scribbled = RunState("add-auth", "main", Board(started_at=100.0))
    scribbled.board.mark("approved", 100.0)
    scribbled.board.activity["T-01"] = "reading src/auth/store.py"
    scribbled.board.status_since["T-99"] = 1.0

    run_a_sequence(watched)
    run_a_sequence(scribbled)

    assert snapshot(watched) == snapshot(scribbled)
    assert watched.display_order() == scribbled.display_order()
    scribbled.check_consistent()


def test_the_board_ends_up_holding_a_stamp_for_every_ticket() -> None:
    state = new_state()
    run_a_sequence(state)

    state.board.activity["T-01"] = "reading src/auth/store.py"

    assert set(state.board.status_since) == set(state.tickets)
    state.check_consistent()


def test_halt_carries_a_reason_and_a_detail() -> None:
    halt = Halt(reason="budget", detail="ran out at $4.10")

    assert halt.reason == "budget"
    assert halt.detail == "ran out at $4.10"
    assert new_state().halt is None


def test_halt_defaults_to_resumable() -> None:
    assert Halt(reason="budget").resumable is True


def test_halt_resumable_can_be_set_false() -> None:
    assert Halt(reason="budget", resumable=False).resumable is False


# -- awaiting -------------------------------------------------------------


def test_awaiting_suspends_and_puts_the_ticket_back_where_it_was() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    with state.awaiting("T-01"):
        assert state.tickets["T-01"].status is Status.AWAITING_INPUT
        assert state.dag.state("T-01") is NodeState.CLAIMED
        state.check_consistent()

    assert state.tickets["T-01"].status is Status.IN_REVIEW
    state.check_consistent()


def test_awaiting_puts_the_ticket_back_even_when_the_block_raises() -> None:
    state = two_tickets()
    claimed_in_review(state, "T-01")

    with pytest.raises(RuntimeError), state.awaiting("T-01"):
        raise RuntimeError("the question failed")

    assert state.tickets["T-01"].status is Status.IN_REVIEW
    state.check_consistent()


def test_awaiting_an_unknown_ticket_raises_before_anything_moves() -> None:
    state = two_tickets()

    with pytest.raises(UnknownTicketError, match="T-99"), state.awaiting("T-99"):
        pass  # pragma: no cover - the context manager never opens


# -- is_halted ------------------------------------------------------------


def test_is_halted_follows_the_halt() -> None:
    state = new_state()

    assert state.is_halted() is False

    state.halt = Halt(reason="merge conflict")

    assert state.is_halted() is True


# -- halt_for -------------------------------------------------------------


def outcome(status: MergeStatus, **overrides: Any) -> MergeOutcome:
    return MergeOutcome(key="T-01", status=status, **overrides)


def test_a_conflict_is_resumable_and_names_what_to_resolve() -> None:
    halt = halt_for(outcome(MergeStatus.CONFLICT, conflicted=("a.py", "b.py")))

    assert halt.resumable is True
    assert "T-01" in halt.reason
    assert "a.py, b.py" in halt.detail


def test_a_failed_build_is_resumable_and_carries_the_tail_of_its_output() -> None:
    lines = "\n".join(f"line {index}" for index in range(1, 41))
    build = ExecResult(argv=("build",), code=1, stdout=lines, stderr="", timed_out=False)

    halt = halt_for(outcome(MergeStatus.BUILD_FAILED, build=build))

    assert halt.resumable is True
    assert "exit 1" in halt.reason
    assert halt.detail.splitlines() == [f"line {index}" for index in range(41 - TAIL_LINES, 41)]


def test_a_failed_build_carries_both_streams() -> None:
    """Which stream carries the diagnosis is the build tool's choice, not this run's."""
    build = ExecResult(
        argv=("build",), code=1, stdout="compiling\n", stderr="error: boom\n", timed_out=False
    )

    halt = halt_for(outcome(MergeStatus.BUILD_FAILED, build=build))

    assert halt.detail.splitlines() == ["compiling", "error: boom"]


def test_a_failed_build_says_so_when_it_timed_out_instead() -> None:
    build = ExecResult(argv=("build",), code=-9, stdout="", stderr="killed", timed_out=True)

    halt = halt_for(outcome(MergeStatus.BUILD_FAILED, build=build))

    assert "timed out" in halt.reason
    assert "killed" in halt.detail


def test_a_vcs_error_cannot_be_resumed() -> None:
    halt = halt_for(outcome(MergeStatus.VCS_ERROR, error="no such branch"))

    assert halt.resumable is False
    assert halt.detail == "no such branch"


def test_anything_else_cannot_be_resumed_either() -> None:
    halt = halt_for(outcome(MergeStatus.ERROR, error="FileNotFoundError: gradlew"))

    assert halt.resumable is False
    assert "gradlew" in halt.reason
