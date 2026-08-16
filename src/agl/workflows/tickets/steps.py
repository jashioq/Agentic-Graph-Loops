"""What the state says to do next — for the run as a whole, and for one ticket.

Layer: workflows. Reads git and the store and reports; the workflow above is
what turns the answer into a call. Nothing here is stored: a stage and a step
are asked again every time they matter, of the documents and branches that are
already the truth. That is what makes a resumed ticket and a fresh one the same
code, and what makes deleting a document walk the run back to the step that
produces it.
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

    Tickets in the state and all of them merged is a finished run; tickets at
    all means implementing them; a spec alone means it is owed a decomposition,
    and nothing at all means it is owed the interview.
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
    """Ask git and the store where `ticket` actually got to.

    Cheap enough to ask before every step of every pass. `implemented` guards
    the rest, so a ticket that has never had a worktree opened is never asked
    about a branch that does not exist yet.
    """
    implemented = ticket.base_sha is not None and vcs.rev_parse(branch) != ticket.base_sha
    return Facts(
        implemented=implemented,
        merged=implemented and vcs.is_ancestor(branch, target),
        settled=_settled(store, ticket),
    )


def step_for(facts: Facts) -> Step:
    """The one step those facts leave owed.

    Ordered by how far the work got, not by the order the steps run in: a merged
    ticket is done whatever else is true of it.
    """
    if facts.merged:
        return Step.DONE
    if facts.settled:
        return Step.MERGE
    if facts.implemented:
        return Step.REVIEW
    return Step.IMPLEMENT


def _settled(store: Store, ticket: Ticket) -> bool:
    """Whether this round's review finished and found nothing left to fix.

    An empty `groups` is a recorded outcome rather than a missing one, which is
    why the triage document is read with `allow_empty=True`.
    """
    triage = review_key(ticket.id, ticket.review_round, "triage")
    keys = [review_key(ticket.id, ticket.review_round, source) for source in REVIEWERS]
    if not all(store.exists(key) for key in (*keys, triage)):
        return False
    return not bug_groups_from_json(store.read_json(triage), allow_empty=True)
