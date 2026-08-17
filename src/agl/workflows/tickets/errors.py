"""Everything this workflow raises, and the halt a run stops on.

Layer: workflows. Imports nothing from the workflow, so any module in it can
raise from here without closing a cycle. `Halt` lives here for the same reason:
`HaltedError` carries one.
"""

from dataclasses import dataclass

__all__ = [
    "CoverageError",
    "DecomposeAbortedError",
    "DuplicateTicketError",
    "Halt",
    "HaltedError",
    "IllegalTransitionError",
    "InterviewIncompleteError",
    "InvalidFindingsError",
    "InvalidGroupsError",
    "InvalidStateError",
    "InvalidTicketsError",
    "NothingImplementedError",
    "NothingToResumeError",
    "RoleIncompleteError",
    "UnknownTicketError",
]


# -- the run ----------------------------------------------------------------


@dataclass(frozen=True)
class Halt:
    """Why a run stopped early, in the words the user is going to read.

    `resumable` is whether pressing enter can plausibly help: true of a
    conflict or a failing build, which a person editing the repository changes
    the answer to, and false when only restarting the process would.
    """

    reason: str
    detail: str = ""
    resumable: bool = True


class InterviewIncompleteError(Exception):
    """Raised when the interview ended without saving a specification."""


class DecomposeAbortedError(Exception):
    """Raised when the user aborted decomposition before approving any tickets."""


class NothingToResumeError(Exception):
    """Raised when a run is asked to continue from a state that has no next step."""


class HaltedError(Exception):
    """Raised when a run ended with a halt nobody resolved. Carries the halt."""

    def __init__(self, halt: Halt) -> None:
        super().__init__(halt.reason)
        self.halt = halt


# -- the state --------------------------------------------------------------


class InvalidStateError(Exception):
    """Raised when a `Run` does not describe a state this workflow could reach."""


class UnknownTicketError(Exception):
    """Raised when an operation names a ticket the run does not hold."""

    def __init__(self, ticket_id: str) -> None:
        super().__init__(ticket_id)
        self.ticket_id = ticket_id


class DuplicateTicketError(Exception):
    """Raised when a ticket would take an id the run already uses."""

    def __init__(self, ticket_id: str) -> None:
        super().__init__(f"ticket {ticket_id!r} is already in the run")
        self.ticket_id = ticket_id


class IllegalTransitionError(Exception):
    """Raised when a ticket is asked to move somewhere it cannot go."""


# -- what an agent produced -------------------------------------------------


class InvalidTicketsError(Exception):
    """Raised when agent output does not describe a usable set of tickets."""


class InvalidFindingsError(Exception):
    """Raised when reviewer output does not describe a usable set of findings."""


class InvalidGroupsError(Exception):
    """Raised when triage-agent output does not describe usable bug groups."""


class CoverageError(Exception):
    """Raised when a set of groups does not account for every `HIGH` finding."""


class RoleIncompleteError(Exception):
    """Raised when a role ended without calling the tool that reports its result."""


class NothingImplementedError(Exception):
    """Raised when an implementation agent finished and the tree was unchanged."""
