"""Mutating a graph that is already running — what bug tickets do mid-flight."""

import pytest

from agl.core.dag import CycleError, Dag, NodeState


def test_a_new_unblocked_node_is_ready_immediately() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.claim("A")
    assert dag.ready() == ()

    dag.add_node("B")

    assert dag.ready() == ("B",)


def test_a_new_blocker_pulls_a_pending_node_out_of_ready_on_the_same_tick() -> None:
    dag = Dag()
    dag.add_node("A")
    assert dag.ready() == ("A",)

    dag.add_node("B")
    dag.add_edge("A", "B")

    assert dag.ready() == ("B",)
    assert dag.unsatisfied_blockers("A") == ("B",)


def test_bugs_found_in_review_block_their_parent_until_they_are_done() -> None:
    dag = Dag()
    for node_id in ("T-01", "T-02", "T-03"):
        dag.add_node(node_id)
    dag.claim("T-03")
    assert dag.ready() == ("T-01", "T-02")

    # Review of T-03 finds two bugs: they become work of their own, and T-03 goes
    # back in the queue behind them.
    dag.add_node("T-03-bug-1")
    dag.add_node("T-03-bug-2")
    dag.release("T-03")
    dag.add_edge("T-03", "T-03-bug-1")
    dag.add_edge("T-03", "T-03-bug-2")

    assert dag.state("T-03") is NodeState.PENDING
    assert "T-03" not in dag.ready()
    assert dag.ready() == ("T-01", "T-02", "T-03-bug-1", "T-03-bug-2")
    assert dag.unsatisfied_blockers("T-03") == ("T-03-bug-1", "T-03-bug-2")

    dag.claim("T-03-bug-1")
    dag.complete("T-03-bug-1")
    assert "T-03" not in dag.ready()
    assert dag.unsatisfied_blockers("T-03") == ("T-03-bug-2",)

    dag.claim("T-03-bug-2")
    dag.complete("T-03-bug-2")
    assert "T-03" in dag.ready()

    dag.claim("T-03")
    dag.complete("T-03")
    assert dag.unsatisfied_blockers("T-03") == ()


def test_the_documented_order_never_leaves_the_parent_claimable() -> None:
    # Nodes, then edges, then the release — the order `Dag`'s docstring gives.
    # Releasing first would leave the parent ready for a beat, and a scheduler
    # that looked in that window would claim it out from under its own bugs.
    dag = Dag()
    dag.add_node("T-03")
    dag.claim("T-03")

    dag.add_node("T-03-bug-1")
    dag.add_node("T-03-bug-2")
    dag.add_edge("T-03", "T-03-bug-1")
    dag.add_edge("T-03", "T-03-bug-2")
    dag.release("T-03")

    assert dag.ready() == ("T-03-bug-1", "T-03-bug-2")
    assert dag.claim_next() == "T-03-bug-1"
    assert dag.claim_next() == "T-03-bug-2"
    assert dag.claim_next() is None
    assert dag.state("T-03") is NodeState.PENDING


def test_releasing_before_the_edges_exist_exposes_the_parent() -> None:
    # The failure the order avoids, driven rather than argued: between the
    # release and the first edge the parent is ready, and a `claim_next` in that
    # window takes it.
    dag = Dag()
    dag.add_node("T-03")
    dag.claim("T-03")
    dag.add_node("T-03-bug-1")

    dag.release("T-03")

    assert dag.claim_next() == "T-03"


def test_a_bug_may_not_depend_on_the_parent_it_blocks() -> None:
    dag = Dag()
    dag.add_node("T-03")
    dag.add_node("T-03-bug-1")
    dag.add_edge("T-03", "T-03-bug-1")

    with pytest.raises(CycleError) as caught:
        dag.add_edge("T-03-bug-1", "T-03")

    assert caught.value.cycle == ("T-03-bug-1", "T-03", "T-03-bug-1")
    assert dag.blockers("T-03-bug-1") == ()
    assert dag.dependents("T-03") == ()
    assert dag.ready() == ("T-03-bug-1",)
    assert dag.levels() == (("T-03-bug-1",), ("T-03",))


def test_an_edge_onto_a_done_blocker_changes_nothing() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.add_node("B")
    dag.claim("B")
    dag.complete("B")

    dag.add_edge("A", "B")

    assert dag.blockers("A") == ("B",)
    assert dag.unsatisfied_blockers("A") == ()
    assert dag.ready() == ("A",)
    assert dag.state("A") is NodeState.PENDING


def test_a_done_node_stays_done_when_a_new_blocker_is_added_to_it() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.add_node("B")
    dag.claim("A")
    dag.complete("A")

    dag.add_edge("A", "B")

    assert dag.state("A") is NodeState.DONE
    assert dag.unsatisfied_blockers("A") == ("B",)
    assert dag.ready() == ("B",)
    assert dag.is_complete() is False


def test_a_claimed_node_keeps_running_when_a_blocker_is_added_to_it() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.add_node("B")
    dag.claim("A")

    dag.add_edge("A", "B")

    assert dag.state("A") is NodeState.CLAIMED
    assert dag.ready() == ("B",)
    assert dag.unsatisfied_blockers("A") == ("B",)

    # It can still finish; `ready()` only ever speaks about pending nodes.
    dag.complete("A")
    assert dag.state("A") is NodeState.DONE


def test_releasing_a_claimed_node_that_gained_a_blocker_leaves_it_blocked() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.add_node("B")
    dag.claim("A")
    dag.add_edge("A", "B")

    dag.release("A")

    assert dag.ready() == ("B",)
    with pytest.raises(ValueError):
        dag.claim("A")


def test_removing_a_node_mid_run_does_not_disturb_claimed_work() -> None:
    dag = Dag()
    for node_id in ("A", "B", "C"):
        dag.add_node(node_id)
    dag.add_edge("A", "B")
    dag.claim("C")

    dag.remove_node("B")

    assert dag.state("C") is NodeState.CLAIMED
    assert dag.ready() == ("A",)
    assert dag.nodes() == ("A", "C")


def test_a_node_added_after_completion_reopens_the_graph() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.claim("A")
    dag.complete("A")
    assert dag.is_complete() is True

    dag.add_node("A-bug-1")

    assert dag.is_complete() is False
    assert dag.ready() == ("A-bug-1",)
