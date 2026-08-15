"""Construction, edges, and the cycle rule."""

import pytest

from agl.runtime.dag import CycleError, Dag, NodeState, UnknownNodeError


def test_nodes_preserve_insertion_order() -> None:
    dag = Dag()
    for node_id in ("C", "A", "B"):
        dag.add_node(node_id)
    assert dag.nodes() == ("C", "A", "B")


def test_nodes_is_empty_for_a_new_graph() -> None:
    assert Dag().nodes() == ()


def test_duplicate_node_id_raises() -> None:
    dag = Dag()
    dag.add_node("A")
    with pytest.raises(ValueError):
        dag.add_node("A")
    assert dag.nodes() == ("A",)


def test_a_node_starts_pending_by_default() -> None:
    dag = Dag()
    dag.add_node("A")
    assert dag.state("A") is NodeState.PENDING


def test_a_node_can_be_added_in_the_state_a_snapshot_says_it_is_in() -> None:
    dag = Dag()
    dag.add_node("A", NodeState.DONE)
    dag.add_node("B", NodeState.CLAIMED)
    dag.add_node("C")
    assert (dag.state("A"), dag.state("B"), dag.state("C")) == (
        NodeState.DONE,
        NodeState.CLAIMED,
        NodeState.PENDING,
    )
    assert dag.ready() == ("C",)


def test_a_duplicate_id_raises_whatever_state_it_is_added_in() -> None:
    dag = Dag()
    dag.add_node("A", NodeState.DONE)
    with pytest.raises(ValueError):
        dag.add_node("A", NodeState.PENDING)
    assert dag.state("A") is NodeState.DONE


def test_a_graph_rebuilt_from_a_snapshot_states_edges_onto_finished_nodes() -> None:
    # The derive path: every node arrives in the state the snapshot recorded,
    # and the edges follow, so an edge onto an already-DONE blocker is normal.
    dag = Dag()
    dag.add_node("A", NodeState.DONE)
    dag.add_node("B", NodeState.CLAIMED)
    dag.add_node("C")
    dag.add_edge("B", "A")
    dag.add_edge("C", "A")

    assert dag.unsatisfied_blockers("C") == ()
    assert dag.ready() == ("C",)


def test_a_graph_rebuilt_wholly_done_is_complete() -> None:
    dag = Dag()
    for node_id in ("A", "B"):
        dag.add_node(node_id, NodeState.DONE)
    dag.add_edge("A", "B")
    assert dag.is_complete()


def test_a_rebuilt_node_can_still_be_moved() -> None:
    dag = Dag()
    dag.add_node("A", NodeState.CLAIMED)
    dag.complete("A")
    assert dag.state("A") is NodeState.DONE


def test_edge_from_unknown_node_raises() -> None:
    dag = Dag()
    dag.add_node("A")
    with pytest.raises(UnknownNodeError):
        dag.add_edge("ghost", "A")


def test_edge_to_unknown_node_raises() -> None:
    dag = Dag()
    dag.add_node("A")
    with pytest.raises(UnknownNodeError):
        dag.add_edge("A", "ghost")


def test_edge_records_both_directions() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.add_node("B")
    dag.add_edge("A", "B")
    assert dag.blockers("A") == ("B",)
    assert dag.dependents("B") == ("A",)
    assert dag.blockers("B") == ()
    assert dag.dependents("A") == ()


def test_adding_the_same_edge_twice_is_a_no_op() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.add_node("B")
    dag.add_edge("A", "B")
    dag.add_edge("A", "B")
    assert dag.blockers("A") == ("B",)
    assert dag.dependents("B") == ("A",)


def test_blockers_follow_insertion_order() -> None:
    dag = Dag()
    for node_id in ("A", "B", "C"):
        dag.add_node(node_id)
    dag.add_edge("A", "C")
    dag.add_edge("A", "B")
    assert dag.blockers("A") == ("B", "C")


def test_queries_on_an_unknown_node_raise() -> None:
    dag = Dag()
    for query in (dag.blockers, dag.unsatisfied_blockers, dag.dependents, dag.state):
        with pytest.raises(UnknownNodeError):
            query("ghost")


def test_self_edge_raises_a_cycle_error() -> None:
    dag = Dag()
    dag.add_node("A")
    with pytest.raises(CycleError):
        dag.add_edge("A", "A")
    assert dag.blockers("A") == ()


def test_two_node_cycle_raises() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.add_node("B")
    dag.add_edge("A", "B")
    with pytest.raises(CycleError):
        dag.add_edge("B", "A")


def test_longer_cycle_raises() -> None:
    dag = Dag()
    for node_id in ("A", "B", "C"):
        dag.add_node(node_id)
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    with pytest.raises(CycleError):
        dag.add_edge("C", "A")


def test_cycle_error_carries_the_self_edge_path() -> None:
    dag = Dag()
    dag.add_node("A")
    with pytest.raises(CycleError) as caught:
        dag.add_edge("A", "A")
    assert caught.value.cycle == ("A", "A")


def test_cycle_error_carries_the_two_node_path() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.add_node("B")
    dag.add_edge("A", "B")
    with pytest.raises(CycleError) as caught:
        dag.add_edge("B", "A")
    assert caught.value.cycle == ("B", "A", "B")


def test_cycle_error_carries_the_whole_path_for_a_longer_cycle() -> None:
    dag = Dag()
    for node_id in ("A", "B", "C"):
        dag.add_node(node_id)
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    with pytest.raises(CycleError) as caught:
        dag.add_edge("C", "A")
    assert caught.value.cycle == ("C", "A", "B", "C")


def test_cycle_error_path_skips_branches_that_do_not_close_the_loop() -> None:
    dag = Dag()
    for node_id in ("A", "B", "C", "D"):
        dag.add_node(node_id)
    dag.add_edge("A", "D")  # a dead end off the eventual cycle
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    with pytest.raises(CycleError) as caught:
        dag.add_edge("C", "A")
    assert caught.value.cycle == ("C", "A", "B", "C")


def test_a_rejected_edge_leaves_the_graph_exactly_as_it_was() -> None:
    dag = Dag()
    for node_id in ("A", "B", "C"):
        dag.add_node(node_id)
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    before = (
        dag.nodes(),
        dag.ready(),
        dag.levels(),
        {node: (dag.blockers(node), dag.dependents(node)) for node in dag.nodes()},
    )

    with pytest.raises(CycleError):
        dag.add_edge("C", "A")

    assert dag.blockers("C") == ()
    assert dag.dependents("A") == ()
    assert (
        dag.nodes(),
        dag.ready(),
        dag.levels(),
        {node: (dag.blockers(node), dag.dependents(node)) for node in dag.nodes()},
    ) == before


def test_a_diamond_is_not_a_cycle() -> None:
    dag = Dag()
    for node_id in ("A", "B", "C", "D"):
        dag.add_node(node_id)
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    assert dag.blockers("A") == ("B", "C")
    assert dag.dependents("D") == ("B", "C")


def test_remove_node_drops_edges_in_both_directions() -> None:
    dag = Dag()
    for node_id in ("A", "B", "C"):
        dag.add_node(node_id)
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")

    dag.remove_node("B")

    assert dag.nodes() == ("A", "C")
    assert dag.blockers("A") == ()
    assert dag.dependents("C") == ()


def test_remove_node_leaves_the_rest_of_the_graph_intact() -> None:
    dag = Dag()
    for node_id in ("A", "B", "C", "D"):
        dag.add_node(node_id)
    dag.add_edge("A", "B")
    dag.add_edge("C", "D")

    dag.remove_node("B")

    assert dag.nodes() == ("A", "C", "D")
    assert dag.blockers("C") == ("D",)
    assert dag.dependents("D") == ("C",)


def test_remove_node_frees_a_node_it_was_blocking() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.add_node("B")
    dag.add_edge("A", "B")
    assert dag.ready() == ("B",)

    dag.remove_node("B")

    assert dag.ready() == ("A",)


def test_remove_unknown_node_raises() -> None:
    dag = Dag()
    with pytest.raises(UnknownNodeError):
        dag.remove_node("ghost")


def test_a_removed_id_can_be_added_again() -> None:
    dag = Dag()
    dag.add_node("A")
    dag.remove_node("A")
    dag.add_node("A")
    assert dag.nodes() == ("A",)
