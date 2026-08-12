"""A directed acyclic graph of work items and the scheduling questions about it.

Layer: core. A pure data structure — no I/O, no async, no imports from the rest
of `agl`. Node ids are opaque strings; the graph knows nothing about what they
stand for. A workflow that wants to attach a payload keeps its own dict keyed by
the same ids.

Edges point from blocked to blocker: `add_edge(blocked, blocker)` means `blocked`
cannot start until `blocker` is `DONE`. Every mutation validates before it
touches anything, so a raising mutation leaves the graph exactly as it was.
"""

from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any, cast

__all__ = [
    "CycleError",
    "Dag",
    "NodeId",
    "NodeState",
    "UnknownNodeError",
]

type NodeId = str


class CycleError(Exception):
    """Raised when a mutation would introduce a cycle. The graph is unchanged.

    `cycle` runs from the edge's `blocked` node along blocker edges back around
    to it, so its first and last entries are the same node.
    """

    def __init__(self, cycle: tuple[NodeId, ...]) -> None:
        super().__init__(" -> ".join(cycle))
        self.cycle = cycle


class UnknownNodeError(Exception):
    """Raised when an operation names a node the graph does not hold."""

    def __init__(self, node_id: NodeId) -> None:
        super().__init__(node_id)
        self.node_id = node_id


class NodeState(Enum):
    """Where a node is in its life: not started, in flight, or finished."""

    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"


class Dag:
    """Work items and their dependencies, plus what a scheduler may start now."""

    def __init__(self, priority: Callable[[NodeId], object] | None = None) -> None:
        """Build an empty graph.

        `priority` is a sort key applied to `ready()` results; `None` keeps
        insertion order. Ties keep insertion order either way.
        """
        self._priority = priority
        self._states: dict[NodeId, NodeState] = {}
        self._blockers: dict[NodeId, set[NodeId]] = {}
        self._dependents: dict[NodeId, set[NodeId]] = {}

    # -- mutation ---------------------------------------------------------

    def add_node(self, node_id: NodeId) -> None:
        """Add a `PENDING` node. Raises `ValueError` if the id is already held."""
        if node_id in self._states:
            raise ValueError(f"duplicate node id: {node_id!r}")
        self._states[node_id] = NodeState.PENDING
        self._blockers[node_id] = set()
        self._dependents[node_id] = set()

    def add_edge(self, blocked: NodeId, blocker: NodeId) -> None:
        """Make `blocked` wait for `blocker` to be `DONE`.

        Idempotent for an edge that already exists. Raises `UnknownNodeError` if
        either id is absent and `CycleError` if the edge would close a loop,
        leaving the graph unchanged in both cases.

        Either node's state is irrelevant. An edge onto a `DONE` blocker is
        allowed and changes nothing — a finished blocker is a satisfied one — and
        a blocker added to a `CLAIMED` node is allowed too: that node keeps
        running, since `ready()` only ever speaks about `PENDING` nodes.
        """
        self._require(blocked)
        self._require(blocker)
        if blocker in self._blockers[blocked]:
            return
        cycle = self._path_to(blocker, blocked)
        if cycle is not None:
            raise CycleError((blocked, *cycle))
        self._blockers[blocked].add(blocker)
        self._dependents[blocker].add(blocked)

    def claim(self, node_id: NodeId) -> None:
        """Hand a ready node to a worker: `PENDING` -> `CLAIMED`.

        Raises `UnknownNodeError`, or `ValueError` if the node is not ready —
        already claimed, already done, or still blocked.
        """
        self._require(node_id)
        if self._states[node_id] is not NodeState.PENDING:
            raise ValueError(f"cannot claim {node_id!r} in state {self._states[node_id].value}")
        blocking = self.unsatisfied_blockers(node_id)
        if blocking:
            raise ValueError(f"cannot claim {node_id!r}, blocked by {', '.join(blocking)}")
        self._states[node_id] = NodeState.CLAIMED

    def release(self, node_id: NodeId) -> None:
        """Give a failed node back for a retry: `CLAIMED` -> `PENDING`.

        Raises `UnknownNodeError`, or `ValueError` from any other state.
        """
        self._transition(node_id, NodeState.CLAIMED, NodeState.PENDING)

    def complete(self, node_id: NodeId) -> None:
        """Finish a node: `CLAIMED` -> `DONE`, and it stops blocking dependents.

        Raises `UnknownNodeError`, or `ValueError` from any other state.
        """
        self._transition(node_id, NodeState.CLAIMED, NodeState.DONE)

    def remove_node(self, node_id: NodeId) -> None:
        """Drop a node and every edge touching it. Raises `UnknownNodeError`."""
        self._require(node_id)
        for blocker in self._blockers[node_id]:
            self._dependents[blocker].discard(node_id)
        for dependent in self._dependents[node_id]:
            self._blockers[dependent].discard(node_id)
        del self._blockers[node_id]
        del self._dependents[node_id]
        del self._states[node_id]

    # -- queries ----------------------------------------------------------

    def nodes(self) -> tuple[NodeId, ...]:
        """Every node in insertion order, for stable display."""
        return tuple(self._states)

    def state(self, node_id: NodeId) -> NodeState:
        """The node's state. Raises `UnknownNodeError`."""
        self._require(node_id)
        return self._states[node_id]

    def blockers(self, node_id: NodeId) -> tuple[NodeId, ...]:
        """Direct blockers in insertion order, not transitive ones."""
        self._require(node_id)
        return self._ordered(self._blockers[node_id])

    def unsatisfied_blockers(self, node_id: NodeId) -> tuple[NodeId, ...]:
        """Direct blockers that are not `DONE` yet, in insertion order."""
        self._require(node_id)
        return self._ordered(
            blocker
            for blocker in self._blockers[node_id]
            if self._states[blocker] is not NodeState.DONE
        )

    def dependents(self, node_id: NodeId) -> tuple[NodeId, ...]:
        """Direct dependents in insertion order, not transitive ones."""
        self._require(node_id)
        return self._ordered(self._dependents[node_id])

    def ready(self) -> tuple[NodeId, ...]:
        """Every `PENDING` node whose blockers are all `DONE`, sorted by priority.

        `CLAIMED` and `DONE` nodes are excluded, so this is exactly the set a
        scheduler may start right now.
        """
        pending = [
            node_id
            for node_id, state in self._states.items()
            if state is NodeState.PENDING and not self.unsatisfied_blockers(node_id)
        ]
        if self._priority is None:
            return tuple(pending)
        return tuple(sorted(pending, key=cast(Callable[[NodeId], Any], self._priority)))

    def levels(self) -> tuple[tuple[NodeId, ...], ...]:
        """Nodes grouped by depth, where depth is the longest path from any root.

        A node with no blockers sits at level 0; every other node sits one below
        its deepest blocker. Order within a level is insertion order.
        """
        depths: dict[NodeId, int] = {}
        for node_id in self._states:
            self._depth(node_id, depths)
        if not depths:
            return ()
        grouped: list[list[NodeId]] = [[] for _ in range(max(depths.values()) + 1)]
        for node_id in self._states:
            grouped[depths[node_id]].append(node_id)
        return tuple(tuple(level) for level in grouped)

    def is_complete(self) -> bool:
        """True when every node is `DONE`. An empty graph is complete."""
        return all(state is NodeState.DONE for state in self._states.values())

    # -- internals --------------------------------------------------------

    def _transition(self, node_id: NodeId, expected: NodeState, target: NodeState) -> None:
        """Move a node between states, refusing anything but `expected`."""
        self._require(node_id)
        current = self._states[node_id]
        if current is not expected:
            raise ValueError(
                f"cannot move {node_id!r} to {target.value} from {current.value}, "
                f"expected {expected.value}"
            )
        self._states[node_id] = target

    def _require(self, node_id: NodeId) -> None:
        """Raise `UnknownNodeError` unless the graph holds `node_id`."""
        if node_id not in self._states:
            raise UnknownNodeError(node_id)

    def _ordered(self, node_ids: Iterable[NodeId]) -> tuple[NodeId, ...]:
        """Put an unordered set of ids back into insertion order."""
        wanted = set(node_ids)
        return tuple(node_id for node_id in self._states if node_id in wanted)

    def _path_to(self, start: NodeId, target: NodeId) -> tuple[NodeId, ...] | None:
        """A blocker-edge path from `start` to `target`, or `None` if there is none.

        Used before committing an edge: if the new blocker already depends on the
        blocked node, the edge would close a loop.
        """
        if start == target:
            return (start,)
        path: list[NodeId] = []
        seen: set[NodeId] = set()

        def walk(node_id: NodeId) -> bool:
            path.append(node_id)
            seen.add(node_id)
            for blocker in self._ordered(self._blockers[node_id]):
                if blocker == target or (blocker not in seen and walk(blocker)):
                    return True
            path.pop()
            return False

        if not walk(start):
            return None
        return (*path, target)

    def _depth(self, node_id: NodeId, depths: dict[NodeId, int]) -> int:
        """Longest-path depth of `node_id`, memoised into `depths`."""
        if node_id not in depths:
            blockers = self._blockers[node_id]
            deepest = max((self._depth(blocker, depths) for blocker in blockers), default=-1)
            depths[node_id] = deepest + 1
        return depths[node_id]
