"""Readiness, the claim/release/complete lifecycle, and the terminal states."""

import pytest

from agl.core.dag import Dag, NodeState, UnknownNodeError


def chain(*node_ids: str) -> Dag:
    """A graph of unconnected nodes, added in the order given."""
    dag = Dag()
    for node_id in node_ids:
        dag.add_node(node_id)
    return dag


def test_an_empty_graph_is_complete_and_has_nothing_ready() -> None:
    dag = Dag()
    assert dag.ready() == ()
    assert dag.is_complete() is True


def test_nodes_without_edges_are_all_ready() -> None:
    dag = chain("A", "B", "C")
    assert dag.ready() == ("A", "B", "C")


def test_a_new_node_starts_pending() -> None:
    dag = chain("A")
    assert dag.state("A") is NodeState.PENDING


def test_only_the_blocker_is_ready() -> None:
    dag = chain("A", "B")
    dag.add_edge("A", "B")
    assert dag.ready() == ("B",)
    assert dag.unsatisfied_blockers("A") == ("B",)


def test_claiming_the_blocker_empties_ready() -> None:
    dag = chain("A", "B")
    dag.add_edge("A", "B")
    dag.claim("B")
    assert dag.state("B") is NodeState.CLAIMED
    assert dag.ready() == ()


def test_completing_the_blocker_releases_its_dependent() -> None:
    dag = chain("A", "B")
    dag.add_edge("A", "B")
    dag.claim("B")
    dag.complete("B")
    assert dag.state("B") is NodeState.DONE
    assert dag.ready() == ("A",)
    assert dag.unsatisfied_blockers("A") == ()
    assert dag.blockers("A") == ("B",)


def test_a_node_with_two_blockers_waits_for_both() -> None:
    dag = chain("A", "B", "C")
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    assert dag.ready() == ("B", "C")

    dag.claim("B")
    dag.complete("B")
    assert dag.ready() == ("C",)
    assert dag.unsatisfied_blockers("A") == ("C",)

    dag.claim("C")
    dag.complete("C")
    assert dag.ready() == ("A",)


def test_a_diamond_advances_one_layer_at_a_time() -> None:
    dag = chain("A", "B", "C", "D")
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")

    assert dag.ready() == ("D",)
    finish(dag, "D")
    assert dag.ready() == ("B", "C")
    finish(dag, "B")
    assert dag.ready() == ("C",)
    finish(dag, "C")
    assert dag.ready() == ("A",)
    finish(dag, "A")
    assert dag.ready() == ()
    assert dag.is_complete() is True


def finish(dag: Dag, node_id: str) -> None:
    """Take a ready node all the way to `DONE`."""
    dag.claim(node_id)
    dag.complete(node_id)


def test_claiming_a_blocked_node_raises() -> None:
    dag = chain("A", "B")
    dag.add_edge("A", "B")
    with pytest.raises(ValueError):
        dag.claim("A")
    assert dag.state("A") is NodeState.PENDING


def test_claiming_an_already_claimed_node_raises() -> None:
    dag = chain("A")
    dag.claim("A")
    with pytest.raises(ValueError):
        dag.claim("A")
    assert dag.state("A") is NodeState.CLAIMED


def test_claiming_a_done_node_raises() -> None:
    dag = chain("A")
    finish(dag, "A")
    with pytest.raises(ValueError):
        dag.claim("A")
    assert dag.state("A") is NodeState.DONE


def test_completing_a_pending_node_raises() -> None:
    dag = chain("A")
    with pytest.raises(ValueError):
        dag.complete("A")
    assert dag.state("A") is NodeState.PENDING


def test_completing_a_done_node_raises() -> None:
    dag = chain("A")
    finish(dag, "A")
    with pytest.raises(ValueError):
        dag.complete("A")
    assert dag.state("A") is NodeState.DONE


def test_releasing_a_pending_node_raises() -> None:
    dag = chain("A")
    with pytest.raises(ValueError):
        dag.release("A")
    assert dag.state("A") is NodeState.PENDING


def test_releasing_a_done_node_raises() -> None:
    dag = chain("A")
    finish(dag, "A")
    with pytest.raises(ValueError):
        dag.release("A")
    assert dag.state("A") is NodeState.DONE


def test_release_returns_a_claimed_node_to_ready() -> None:
    dag = chain("A", "B")
    dag.claim("A")
    assert dag.ready() == ("B",)

    dag.release("A")

    assert dag.state("A") is NodeState.PENDING
    assert dag.ready() == ("A", "B")


def test_a_released_node_can_be_claimed_again() -> None:
    dag = chain("A")
    dag.claim("A")
    dag.release("A")
    finish(dag, "A")
    assert dag.is_complete() is True


def test_lifecycle_calls_on_an_unknown_node_raise() -> None:
    dag = Dag()
    for transition in (dag.claim, dag.release, dag.complete):
        with pytest.raises(UnknownNodeError):
            transition("ghost")


def test_a_graph_with_work_left_is_not_complete() -> None:
    dag = chain("A", "B")
    finish(dag, "A")
    assert dag.is_complete() is False


def test_removing_a_blocker_frees_its_dependent_rather_than_stalling() -> None:
    dag = chain("A", "B")
    dag.add_edge("A", "B")
    dag.claim("B")
    dag.remove_node("B")
    assert dag.ready() == ("A",)


def test_every_node_can_be_worked_through_to_completion() -> None:
    # The invariant behind that claim: with nothing claimed, the pending subgraph
    # is acyclic, so it has a node whose blockers are all done — a ready node.
    dag = chain("A", "B", "C", "D")
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    for _ in range(len(dag.nodes())):
        node_id = dag.ready()[0]
        dag.claim(node_id)
        dag.complete(node_id)
    assert dag.is_complete() is True
