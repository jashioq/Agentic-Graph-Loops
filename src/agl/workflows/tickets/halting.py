"""Everything about a run stopping: what a halt says, and what puts one there.

Layer: workflows. This run's halt policy — the runtime reports an outcome, these
decide what it means. Nothing below `workflows` may import it.
"""

from agl.runtime.dag import NodeId
from agl.runtime.display import Display
from agl.runtime.merge import MergeDecision, MergeOutcome, MergeStatus
from agl.workflows.tickets.documents.state_document import StateDocument
from agl.workflows.tickets.errors import Halt
from agl.workflows.tickets.run_state import with_halt

__all__ = [
    "TAIL_LINES",
    "failed",
    "halt_for",
    "resolve",
]

TAIL_LINES = 20
"""How many lines of a failed build's output a halt carries."""


def halt_for(outcome: MergeOutcome) -> Halt:
    """Takes a merge outcome that did not land, returns the halt this run shows for it.

    return: Halt - resumable only when editing the repository changes the answer
    """
    if outcome.status is MergeStatus.CONFLICT:
        return Halt(
            reason=f"{outcome.key} conflicts with the base branch",
            detail=f"resolve in the repository root: {', '.join(outcome.conflicted)}",
        )
    if outcome.status is MergeStatus.BUILD_FAILED and outcome.build is not None:
        build = outcome.build
        what = "timed out" if build.timed_out else f"failed with exit {build.code}"
        return Halt(
            reason=f"{outcome.key} merged but the build {what}",
            detail=_tail(build.output, TAIL_LINES),
        )
    if outcome.status is MergeStatus.VCS_ERROR:
        return Halt(f"{outcome.key} cannot be merged", outcome.error, resumable=False)
    return Halt(
        reason=f"{outcome.key} could not be processed: {outcome.error}",
        detail=outcome.error,
        resumable=False,
    )


def _tail(text: str, lines: int) -> str:
    """The last `lines` lines of `text`."""
    kept = text.strip("\n").split("\n")[-lines:]
    return "\n".join(kept)


async def resolve(
    display: Display, state: StateDocument, label: str, outcome: MergeOutcome
) -> MergeDecision:
    """Decides what a merge that did not land means to this run.

    return: MergeDecision - `RETRY` after a person cleared an actionable halt, else `STOP`
    """
    halt = halt_for(outcome)
    state.update(lambda run: with_halt(run, halt))  # the dashboard shows it next frame
    if not halt.resumable:
        return MergeDecision.STOP
    await display.confirm(label, "press enter to continue")
    state.update(lambda run: with_halt(run, None))
    return MergeDecision.RETRY


def failed(state: StateDocument, node_id: NodeId | None, error: BaseException) -> None:
    """Records an exception out of a ticket, or the loop itself, as a halt to restart from.

    param: node_id - the ticket that raised, or `None` for the loop
    """
    who = node_id if node_id is not None else "the run"
    halt = Halt(f"{who} failed: {error}", str(error), resumable=False)
    state.update(lambda run: with_halt(run, halt))
