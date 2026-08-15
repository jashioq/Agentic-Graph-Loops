"""The sort key applied to `ready()`, and the insertion-order fallback."""

import subprocess
import sys

from agl.runtime.dag import Dag, NodeId


def test_ready_follows_insertion_order_by_default() -> None:
    dag = Dag()
    for node_id in ("C", "A", "B"):
        dag.add_node(node_id)
    assert dag.ready() == ("C", "A", "B")


def test_a_priority_function_reorders_ready() -> None:
    dag = Dag(priority=lambda node_id: node_id)
    for node_id in ("C", "A", "B"):
        dag.add_node(node_id)
    assert dag.ready() == ("A", "B", "C")


def test_priority_does_not_touch_nodes_or_levels() -> None:
    dag = Dag(priority=lambda node_id: node_id)
    for node_id in ("C", "A", "B"):
        dag.add_node(node_id)
    assert dag.nodes() == ("C", "A", "B")
    assert dag.levels() == (("C", "A", "B"),)


def test_priority_does_not_touch_blockers_or_dependents() -> None:
    dag = Dag(priority=lambda node_id: node_id)
    for node_id in ("C", "B", "A"):
        dag.add_node(node_id)
    dag.add_edge("C", "B")
    dag.add_edge("C", "A")
    assert dag.blockers("C") == ("B", "A")


def test_ties_fall_back_to_insertion_order() -> None:
    dag = Dag(priority=lambda node_id: 0)
    for node_id in ("C", "A", "B"):
        dag.add_node(node_id)
    assert dag.ready() == ("C", "A", "B")


def test_ties_within_a_priority_group_keep_insertion_order() -> None:
    dag = Dag(priority=lambda node_id: len(node_id))
    for node_id in ("ccc", "b", "aaa", "a"):
        dag.add_node(node_id)
    assert dag.ready() == ("b", "a", "ccc", "aaa")


def test_priority_only_ranks_nodes_that_are_actually_ready() -> None:
    dag = Dag(priority=lambda node_id: node_id)
    for node_id in ("A", "B", "C"):
        dag.add_node(node_id)
    dag.add_edge("A", "C")
    assert dag.ready() == ("B", "C")

    dag.claim("B")
    assert dag.ready() == ("C",)


def test_bugs_outrank_features_and_creation_order_breaks_the_tie() -> None:
    kinds = {"T-01": 1, "T-02": 1, "T-01-bug-1": 0, "T-02-bug-1": 0}
    order = {node_id: seq for seq, node_id in enumerate(kinds)}

    def priority(node_id: NodeId) -> tuple[int, int]:
        return (kinds[node_id], order[node_id])

    dag = Dag(priority=priority)
    for node_id in kinds:
        dag.add_node(node_id)

    assert dag.ready() == ("T-01-bug-1", "T-02-bug-1", "T-01", "T-02")


def test_a_reordering_priority_survives_completion() -> None:
    kinds = {"feat": 1, "bug": 0}

    def priority(node_id: NodeId) -> int:
        return kinds[node_id]

    dag = Dag(priority=priority)
    for node_id in kinds:
        dag.add_node(node_id)
    assert dag.ready() == ("bug", "feat")

    dag.claim("bug")
    dag.complete("bug")

    assert dag.ready() == ("feat",)


def test_the_module_imports_and_takes_a_priority_at_runtime() -> None:
    # The priority annotation names `_typeshed`, which exists for type checkers
    # and not at runtime. In a fresh interpreter, so an earlier import cannot
    # mask a failure here.
    source = (
        "from agl.runtime.dag import Dag; "
        "d = Dag(priority=lambda node_id: node_id); "
        "d.add_node('B'); d.add_node('A'); print(d.ready())"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "('A', 'B')"
