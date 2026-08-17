"""What the state says to do next — for the run as a whole, and for one ticket.

Layer: workflows. Reads git and the store and reports; the workflow turns the
answer into a call. Nothing is stored: a stage and a step are re-derived every
time they matter, which is what makes a resumed ticket and a fresh one the same
code, and deleting a document walk the run back to the step that produces it.
"""

from dataclasses import dataclass
from enum import Enum

from agl.core.store import Store
from agl.core.vcs import Vcs
from agl.workflows.tickets.documents.review_documents import bug_groups_from_json
from agl.workflows.tickets.documents.store_keys import REVIEWERS, SPEC_KEY, review_key
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import Run

__all__ = ["Facts", "Stage", "Step", "look", "stage_of", "step_for"]


class Stage(Enum):
    """Which part of a run is owed work."""

    INTERVIEW = "interview"
    DECOMPOSE = "decompose"
    IMPLEMENT = "implement"
    DONE = "done"


class Step(Enum):
    """What one ticket is owed next."""

    IMPLEMENT = "implement"
    REVIEW = "review"
    MERGE = "merge"
    DONE = "done"


def stage_of(run: Run, store: Store) -> Stage:
    """The furthest stage the state supports.

    All tickets merged is `DONE`; any tickets, `IMPLEMENT`; a spec alone,
    `DECOMPOSE`; nothing at all, `INTERVIEW`.
    """
    if run.tickets:
        if all(ticket.status is Status.MERGED for ticket in run.tickets):
            return Stage.DONE
        return Stage.IMPLEMENT
    if store.exists(SPEC_KEY):
        return Stage.DECOMPOSE
    return Stage.INTERVIEW


@dataclass(frozen=True)
class Facts:
    """What git and the store say has happened to one ticket."""

    implemented: bool  # the branch has moved off `base_sha`, so its own commit exists
    merged: bool  # implemented, and the branch is reachable from its target
    settled: bool  # this round's documents are all on disk and triage found nothing


def look(vcs: Vcs, store: Store, ticket: Ticket, branch: str, target: str) -> Facts:
    """Asks git and the store where `ticket` actually got to.

    param: branch - the ticket's own branch
    param: target - what it merges into: the run's base, or its parent's branch
    return: Facts - `implemented` guards the rest, so a branchless ticket is never queried
    """
    implemented = ticket.base_sha is not None and vcs.rev_parse(branch) != ticket.base_sha
    return Facts(
        implemented=implemented,
        merged=implemented and vcs.is_ancestor(branch, target),
        settled=_settled(store, ticket),
    )


def step_for(facts: Facts) -> Step:
    """The one step those facts leave owed, ordered by how far the work got."""
    if facts.merged:
        return Step.DONE
    if facts.settled:
        return Step.MERGE
    if facts.implemented:
        return Step.REVIEW
    return Step.IMPLEMENT


def _settled(store: Store, ticket: Ticket) -> bool:
    """Whether this round's review finished and found nothing left to fix."""
    triage = review_key(ticket.id, ticket.review_round, "triage")
    keys = [review_key(ticket.id, ticket.review_round, source) for source in REVIEWERS]
    if not all(store.exists(key) for key in (*keys, triage)):
        return False
    return not bug_groups_from_json(store.read_json(triage), allow_empty=True)
