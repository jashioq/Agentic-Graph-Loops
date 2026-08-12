"""Depth grouping, used to render the graph during ticket approval."""

from agl.core.dag import Dag


def build(*node_ids: str) -> Dag:
    dag = Dag()
    for node_id in node_ids:
        dag.add_node(node_id)
    return dag


def test_an_empty_graph_has_no_levels() -> None:
    assert Dag().levels() == ()


def test_a_linear_chain_gives_one_node_per_level() -> None:
    dag = build("A", "B", "C")
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    assert dag.levels() == (("C",), ("B",), ("A",))


def test_independent_nodes_all_sit_at_level_zero() -> None:
    dag = build("A", "B", "C")
    assert dag.levels() == (("A", "B", "C"),)


def test_a_diamond_gives_three_levels_with_a_shared_middle() -> None:
    dag = build("A", "B", "C", "D")
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    assert dag.levels() == (("D",), ("B", "C"), ("A",))


def test_depth_is_the_longest_path_not_the_shortest() -> None:
    # A is blocked by D directly and by C -> B -> D, so it lands at depth 3.
    dag = build("A", "B", "C", "D")
    dag.add_edge("A", "D")
    dag.add_edge("A", "C")
    dag.add_edge("C", "B")
    dag.add_edge("B", "D")
    assert dag.levels() == (("D",), ("B",), ("C",), ("A",))


def test_order_within_a_level_follows_insertion_order() -> None:
    dag = build("C", "A", "B", "root")
    for node_id in ("C", "A", "B"):
        dag.add_edge(node_id, "root")
    assert dag.levels() == (("root",), ("C", "A", "B"))


def test_disconnected_subgraphs_share_the_same_levels() -> None:
    dag = build("A", "B", "X", "Y")
    dag.add_edge("A", "B")
    dag.add_edge("X", "Y")
    assert dag.levels() == (("B", "Y"), ("A", "X"))


def test_levels_ignore_node_state() -> None:
    dag = build("A", "B")
    dag.add_edge("A", "B")
    before = dag.levels()

    dag.claim("B")
    dag.complete("B")

    assert dag.levels() == before


def test_levels_follow_a_graph_that_grows() -> None:
    dag = build("A")
    assert dag.levels() == (("A",),)

    dag.add_node("A-bug-1")
    assert dag.levels() == (("A", "A-bug-1"),)

    dag.add_edge("A", "A-bug-1")
    assert dag.levels() == (("A-bug-1",), ("A",))


def test_removing_the_deepest_node_flattens_the_levels() -> None:
    dag = build("A", "B", "C")
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")

    dag.remove_node("C")

    assert dag.levels() == (("B",), ("A",))
