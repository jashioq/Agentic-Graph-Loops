"""A directed acyclic graph of work items and the scheduling questions about it.

Layer: runtime. A pure data structure — no I/O, no async, nothing imported from
`agl`. Node ids are opaque strings. Edges point from blocked to blocker:
`add_edge(blocked, blocker)` means `blocked` waits for `blocker` to be `DONE`.
Every mutation validates first, so a raising one leaves the graph unchanged.

Mutating a running graph has one load-bearing ordering: add the new nodes, add
the edges, *then* release the claimed one — releasing first leaves a window a
scheduler can claim through. Edges onto `CLAIMED` and `DONE` nodes are legal,
which is also what lets a graph be rebuilt from a snapshot in a single pass.
"""

from collections.abc import Callable, Iterable
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only exists for type checkers, so the annotation below stays a string.
    from _typeshed import SupportsRichComparison

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

    `cycle` starts and ends on the same node.
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

    def __init__(
        self, priority: "Callable[[NodeId], SupportsRichComparison] | None" = None
    ) -> None:
        """Builds an empty graph.

        param: priority - sort key over `ready()`; `None` keeps insertion order, as do ties
        """
        self._priority = priority
        self._states: dict[NodeId, NodeState] = {}
        self._blockers: dict[NodeId, set[NodeId]] = {}
        self._dependents: dict[NodeId, set[NodeId]] = {}

    # -- mutation ---------------------------------------------------------

    def add_node(self, node_id: NodeId, state: NodeState = NodeState.PENDING) -> None:
        """Adds a node, raising `ValueError` on a duplicate id.

        param: state - omit on the live path; pass it when rebuilding from a snapshot
        """
        if node_id in self._states:
            raise ValueError(f"duplicate node id: {node_id!r}")
        self._states[node_id] = state
        self._blockers[node_id] = set()
        self._dependents[node_id] = set()

    def add_edge(self, blocked: NodeId, blocker: NodeId) -> None:
        """Makes `blocked` wait for `blocker` to be `DONE`.

        Idempotent, and indifferent to either node's state. Raises
        `UnknownNodeError` or `CycleError`, leaving the graph unchanged.
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
        """Hands a ready node to a worker: `PENDING` -> `CLAIMED`.

        Raises `UnknownNodeError`, or `ValueError` if the node is not ready.
        """
        self._require(node_id)
        if self._states[node_id] is not NodeState.PENDING:
            raise ValueError(f"cannot claim {node_id!r} in state {self._states[node_id].value}")
        blocking = self.unsatisfied_blockers(node_id)
        if blocking:
            raise ValueError(f"cannot claim {node_id!r}, blocked by {', '.join(blocking)}")
        self._states[node_id] = NodeState.CLAIMED

    def claim_next(self) -> NodeId | None:
        """The highest-priority ready node, claimed, or `None` if nothing is ready.

        Synchronous on purpose: reading the ready set and claiming out of it is
        one step no caller can split, so no node reaches two workers.
        """
        ready = self.ready()
        if not ready:
            return None
        self.claim(ready[0])
        return ready[0]

    def release(self, node_id: NodeId) -> None:
        """Gives a failed node back for a retry: `CLAIMED` -> `PENDING`.

        Raises `UnknownNodeError`, or `ValueError` from any other state.
        """
        self._transition(node_id, NodeState.CLAIMED, NodeState.PENDING)

    def complete(self, node_id: NodeId) -> None:
        """Finishes a node: `CLAIMED` -> `DONE`, and it stops blocking dependents.

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

        Exactly the set a scheduler may start right now.
        """
        pending = [
            node_id
            for node_id, state in self._states.items()
            if state is NodeState.PENDING and not self._is_blocked(node_id)
        ]
        if self._priority is None:
            return tuple(pending)
        return tuple(sorted(pending, key=self._priority))

    def levels(self) -> tuple[tuple[NodeId, ...], ...]:
        """Nodes grouped by depth, longest path from any root, insertion order within."""
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

    def is_stalled(self) -> bool:
        """Not complete, nothing claimed, nothing ready — the graph cannot advance.

        Separates a scheduler waiting on work in flight from one waiting forever.
        """
        if self.is_complete() or self.ready():
            return False
        return not any(state is NodeState.CLAIMED for state in self._states.values())

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

    def _is_blocked(self, node_id: NodeId) -> bool:
        """Whether anything still holds `node_id` back.

        `unsatisfied_blockers` answers the same question but builds the list;
        `ready()` asks this of every node, which is linear against quadratic.
        """
        return any(
            self._states[b] is not NodeState.DONE for b in self._blockers[node_id]
        )

    def _require(self, node_id: NodeId) -> None:
        """Raise `UnknownNodeError` unless the graph holds `node_id`."""
        if node_id not in self._states:
            raise UnknownNodeError(node_id)

    def _ordered(self, node_ids: Iterable[NodeId]) -> tuple[NodeId, ...]:
        """Put an unordered set of ids back into insertion order."""
        wanted = set(node_ids)
        return tuple(node_id for node_id in self._states if node_id in wanted)

    def _path_to(self, start: NodeId, target: NodeId) -> tuple[NodeId, ...] | None:
        """A blocker-edge path from `start` to `target`, or `None`. Checked before an edge."""
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
