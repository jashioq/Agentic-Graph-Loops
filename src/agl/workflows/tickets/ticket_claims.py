"""How the scheduler claims a ticket off the state document, and hands it back.

Layer: workflows. The four questions `drive` asks, answered from the state every
time and holding nothing between them. Imports `ticket_pass`; nothing in
`ticket_pass` imports this.
"""

from agl.runtime.dag import NodeId, NodeState
from agl.runtime.scheduler import Claims
from agl.workflows.tickets.models import Status
from agl.workflows.tickets.run_state import dag_of, with_status
from agl.workflows.tickets.steps import Step, look, step_for
from agl.workflows.tickets.ticket_pass import Loop, base_for

__all__ = ["claims"]


# The status a ticket is claimed into, per step it is owed: work already done
# and only needing a merge is claimed `MERGING`, not restarted `IN_PROGRESS`.
_STATUS_FOR: dict[Step, Status] = {
    Step.IMPLEMENT: Status.IN_PROGRESS,
    Step.REVIEW: Status.IN_REVIEW,
    Step.MERGE: Status.MERGING,
    Step.DONE: Status.MERGED,
}


def claims(loop: Loop) -> Claims:
    """The scheduler's four questions, answered off the state document every time."""
    state = loop.state

    def next_() -> NodeId | None:
        """Claim one ticket, atomically: load, decide, write, no `await`."""
        current = state.load()
        node = dag_of(current).claim_next()
        if node is None:
            return None
        ticket = current.ticket(node)
        step = step_for(
            look(
                loop.ctx.vcs,
                loop.ctx.store,
                ticket,
                loop.trees.branch_for(node),
                base_for(loop, ticket),
            )
        )
        state.write(with_status(current, node, _STATUS_FOR[step]))
        return node

    def release(node: NodeId) -> None:
        """Hand a ticket whose pass raised back to the queue, leaving a merged one alone."""
        state.update(
            lambda run: run
            if run.ticket(node).status is Status.MERGED
            else with_status(run, node, Status.PENDING)
        )

    def stalled() -> tuple[NodeId, ...] | None:
        """Every ticket still pending, once nothing can make progress."""
        dag = dag_of(state.load())
        if not dag.is_stalled():
            return None
        return tuple(n for n in dag.nodes() if dag.state(n) is NodeState.PENDING)

    return Claims(
        next=next_,
        release=release,
        complete=lambda: dag_of(state.load()).is_complete(),
        stalled=stalled,
    )
