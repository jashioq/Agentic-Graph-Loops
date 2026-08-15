"""What the state says to do next — for the run as a whole, and for one ticket.

Layer: workflows. Reads git and the store and decides nothing about what to do
with the answer: it reports which stage a run has reached and which step a
ticket is owed, and the workflow above is what turns that into a call.

**Nothing here is stored.** A stage and a step are questions asked again every
time they matter, of the documents and the branches that are already the truth.
That is what makes a resumed ticket and a fresh one the same code: neither
carries a note about where it got to, both ask.

It also means deleting something walks a run backwards to exactly the step that
produces it. Throw away `spec.md` and the run interviews again; throw away the
tickets and it decomposes again; throw away a round's findings and that round is
reviewed again. There is no "already did that" flag to contradict the absence.

`step_for` is a pure function of three facts and is the whole point of the file.
`look` is where the facts come from, and each is chosen to be one a killed
process cannot have lied about:

- `implemented` — the branch has moved off `base_sha`, so this ticket's own
  commit exists. `base_sha` is why the question is answerable at all: a branch
  with no commits yet and a branch whose commits have been merged away are both
  ancestors of their base, and git cannot tell them apart without a mark.
- `merged` — implemented, and the branch is reachable from its target.
- `settled` — both findings documents and the triage document for this round are
  on disk and triage produced no groups: the review is done and left nothing to
  fix.
"""

from dataclasses import dataclass
from enum import Enum

from agl.core.store import Store
from agl.core.vcs import Vcs
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.reviews import REVIEWERS, bug_groups_from_json, review_key
from agl.workflows.tickets.state import Run

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
    all means implementing them. Otherwise a spec means the run has been
    interviewed and is owed a decomposition, and nothing at all means it is owed
    the interview.

    Furthest rather than earliest, because each stage's output is what carries
    the run past it: a run is at the last stage whose input exists and whose
    output does not, so deleting a document walks the run back to exactly the
    step that produces it.
    """
    if run.tickets:
        if all(ticket.status is Status.MERGED for ticket in run.tickets):
            return Stage.DONE
        return Stage.IMPLEMENT
    if store.exists(ticket_tools.SPEC_KEY):
        return Stage.DECOMPOSE
    return Stage.INTERVIEW


@dataclass(frozen=True)
class Facts:
    """What git and the store say has happened to one ticket."""

    implemented: bool
    merged: bool
    settled: bool


def look(vcs: Vcs, store: Store, ticket: Ticket, branch: str, target: str) -> Facts:
    """Ask git and the store where `ticket` actually got to.

    Cheap enough to ask before every step of every pass: two `git rev-parse`-class
    calls and a handful of `exists`. `implemented` is checked first and guards
    the rest, so a ticket that has never had a worktree opened costs nothing and
    — importantly — is never asked about a branch that does not exist yet.
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
    ticket is done whatever else is true of it, and a settled review means the
    merge is what is left even though the reviewers ran long ago.
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

    Every role's document is its report, so their presence is what says the
    round ran. Triage's contents are what says it came to nothing: an empty
    `groups` is a recorded outcome rather than a missing one, which is why it is
    read with `allow_empty=True`.
    """
    triage = review_key(ticket.id, ticket.review_round, "triage")
    keys = [review_key(ticket.id, ticket.review_round, source) for source in REVIEWERS]
    if not all(store.exists(key) for key in (*keys, triage)):
        return False
    return not bug_groups_from_json(store.read_json(triage), allow_empty=True)
