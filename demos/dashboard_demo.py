"""A fake ticket run, for looking at.

Ten tickets move through a scripted timeline while bug tickets appear underneath
the ones that spawned them. Nothing pushes updates: the live loop rebuilds this
screen several times a second, which is why the timers tick.

    uv run python demos/dashboard_demo.py

Ctrl-C to stop. The timeline loops.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

from agl.core.terminal import Color, Row, Rows, Screen, Spacer, Text, Timer
from agl.core.terminal.impl.rich_terminal import RichTerminal

WORKFLOW = "add-auth-ticket-18732"

ID_WIDTH = 18
TITLE_WIDTH = 38
STATUS_WIDTH = 18
INDENT = 6


class Status(Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    IN_PROGRESS = "in progress"
    IN_REVIEW = "in review"
    MERGING = "merging"
    MERGED = "merged"
    AWAITING_INPUT = "awaiting input"


COLORS = {
    Status.PENDING: Color.WHITE,
    Status.BLOCKED: Color.RED,
    Status.IN_PROGRESS: Color.BLUE,
    Status.IN_REVIEW: Color.MAGENTA,
    Status.MERGING: Color.YELLOW,
    Status.MERGED: Color.DIM_GREEN,
    Status.AWAITING_INPUT: Color.BOLD_YELLOW,
}

ACTIVE = {Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING, Status.AWAITING_INPUT}

TITLES = {
    "T-01": "Migrate login screen to Compose",
    "T-02": "Add biometric unlock prompt",
    "T-03": "Refresh token on 401 responses",
    "T-04": "Cache session in DataStore",
    "T-05": "Extract auth interceptor to :network",
    "T-06": "Handle account switch in nav graph",
    "T-07": "Add sign-out confirmation dialog",
    "T-08": "Instrument auth events for analytics",
    "T-09": "Update Gradle to AGP 8.5",
    "T-10": "Backfill auth unit tests",
    "T-03-bug-1": "Refresh loops on an expired token",
    "T-03-bug-2": "Race with in-flight requests",
    "T-06-bug-1": "Back stack lost after switching",
}

AGENT_MESSAGES = (
    "Exploring module structure",
    "Scrutinizing architectural choices",
    "Running Gradle build",
    "Reviewing against spec",
    "Rebasing onto main",
    "Waiting on review feedback",
)

# (seconds from the start of the cycle, ticket id, new status). A bug id showing
# up here for the first time is what inserts its row under its parent.
TIMELINE: tuple[tuple[float, str, Status], ...] = (
    (0, "T-01", Status.IN_PROGRESS),
    (0, "T-02", Status.IN_PROGRESS),
    (0, "T-03", Status.IN_PROGRESS),
    (8, "T-01", Status.IN_REVIEW),
    (12, "T-02", Status.AWAITING_INPUT),
    (14, "T-03-bug-1", Status.PENDING),
    (14, "T-03", Status.BLOCKED),
    (15, "T-01", Status.MERGING),
    (18, "T-01", Status.MERGED),
    (18, "T-04", Status.IN_PROGRESS),
    (20, "T-03-bug-1", Status.IN_PROGRESS),
    (22, "T-02", Status.IN_PROGRESS),
    (24, "T-03-bug-2", Status.PENDING),
    (26, "T-03-bug-1", Status.IN_REVIEW),
    (28, "T-02", Status.IN_REVIEW),
    (30, "T-03-bug-1", Status.MERGING),
    (32, "T-03-bug-1", Status.MERGED),
    (33, "T-03-bug-2", Status.IN_PROGRESS),
    (34, "T-02", Status.MERGING),
    (36, "T-02", Status.MERGED),
    (36, "T-05", Status.IN_PROGRESS),
    (40, "T-04", Status.IN_REVIEW),
    (42, "T-03-bug-2", Status.IN_REVIEW),
    (44, "T-03-bug-2", Status.MERGING),
    (46, "T-03-bug-2", Status.MERGED),
    (47, "T-03", Status.IN_PROGRESS),
    (48, "T-04", Status.MERGING),
    (50, "T-04", Status.MERGED),
    (52, "T-05", Status.AWAITING_INPUT),
    (54, "T-06", Status.IN_PROGRESS),
    (56, "T-03", Status.IN_REVIEW),
    (58, "T-05", Status.IN_PROGRESS),
    (60, "T-06-bug-1", Status.PENDING),
    (60, "T-06", Status.BLOCKED),
    (62, "T-03", Status.MERGING),
    (64, "T-03", Status.MERGED),
    (64, "T-07", Status.IN_PROGRESS),
    (66, "T-06-bug-1", Status.IN_PROGRESS),
    (68, "T-05", Status.IN_REVIEW),
    (70, "T-06-bug-1", Status.IN_REVIEW),
    (72, "T-05", Status.MERGING),
    (73, "T-06-bug-1", Status.MERGED),
    (74, "T-05", Status.MERGED),
    (74, "T-08", Status.IN_PROGRESS),
    (75, "T-06", Status.IN_PROGRESS),
    (76, "T-07", Status.IN_REVIEW),
    (78, "T-06", Status.IN_REVIEW),
    (80, "T-07", Status.MERGING),
    (81, "T-06", Status.MERGING),
    (82, "T-07", Status.MERGED),
    (83, "T-06", Status.MERGED),
    (83, "T-09", Status.IN_PROGRESS),
    (84, "T-08", Status.IN_REVIEW),
    (86, "T-09", Status.IN_REVIEW),
    (86, "T-10", Status.IN_PROGRESS),
    (87, "T-08", Status.MERGING),
    (88, "T-08", Status.MERGED),
    (89, "T-09", Status.MERGING),
    (90, "T-09", Status.MERGED),
    (91, "T-10", Status.IN_REVIEW),
    (93, "T-10", Status.MERGING),
    (95, "T-10", Status.MERGED),
)

CYCLE_SECONDS = 102.0
TICKET_IDS = tuple(f"T-{number:02d}" for number in range(1, 11))


@dataclass
class Ticket:
    """One row. `since` restarts on every status change, so the timer resets."""

    id: str
    status: Status
    since: float
    is_bug: bool


@dataclass
class Dashboard:
    """The workflow's own state. `build` only reads it."""

    started: float = field(default_factory=time.monotonic)
    note: str = ""
    _order: list[str] = field(default_factory=list)
    _tickets: dict[str, Ticket] = field(default_factory=dict)
    _cycle: int = -1
    _next_event: int = 0

    def build(self) -> Screen:
        """One frame. Called by the live loop; never blocks, never mutates I/O."""
        now = time.monotonic()
        self._advance(now)
        return Screen(
            header=Rows(Row(Text(WORKFLOW, Color.CYAN)), Row()),
            content=Rows(*(self._row(self._tickets[id], now) for id in self._order)),
            footer=self._footer(now),
        )

    def _row(self, ticket: Ticket, now: float) -> Row:
        status = Text(f"{self._status_text(ticket):<{STATUS_WIDTH}}", COLORS[ticket.status])
        title = Text(f"{TITLES[ticket.id]:<{TITLE_WIDTH}}", Color.GREY)
        if ticket.is_bug:
            label = Row(Spacer(INDENT), Text(f"{ticket.id:<{ID_WIDTH - INDENT}}", Color.DIM_GREY))
        else:
            label = Row(Text(f"{ticket.id:<{ID_WIDTH}}"))
        if ticket.status not in ACTIVE:
            return Row(label, title, status)
        return Row(label, title, status, Timer(ticket.since))

    def _status_text(self, ticket: Ticket) -> str:
        if ticket.status is not Status.BLOCKED:
            return ticket.status.value
        bugs = self._bug_count(ticket.id)
        return f"pending ({bugs} bug{'s' if bugs != 1 else ''})"

    def _footer(self, now: float) -> Rows:
        message = AGENT_MESSAGES[int((now - self.started) / 4) % len(AGENT_MESSAGES)]
        line = Row(
            Text("elapsed", Color.DIM_GREY),
            Spacer(1),
            Timer(self.started),
            Spacer(4),
            Text(f"{message}…", Color.DIM_GREY),
        )
        if not self.note:
            return Rows(Row(), line)
        return Rows(Row(), line, Row(Text(self.note, Color.GREEN)))

    def _bug_count(self, parent: str) -> int:
        return sum(1 for id in self._order if id.startswith(f"{parent}-bug-"))

    def _advance(self, now: float) -> None:
        """Apply every event the clock has passed, restarting on each loop.

        A status is stamped with the time the event was due rather than the time
        it was noticed, so a timer stays right even if several events come due
        in one frame.
        """
        elapsed = now - self.started
        cycle = int(elapsed // CYCLE_SECONDS)
        cycle_start = self.started + cycle * CYCLE_SECONDS
        if cycle != self._cycle:
            self._reset(cycle_start)
            self._cycle = cycle
        position = elapsed % CYCLE_SECONDS
        while self._next_event < len(TIMELINE) and TIMELINE[self._next_event][0] <= position:
            at, ticket_id, status = TIMELINE[self._next_event]
            self._apply(ticket_id, status, cycle_start + at)
            self._next_event += 1

    def _apply(self, ticket_id: str, status: Status, at: float) -> None:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            self._insert_bug(ticket_id, status, at)
            return
        ticket.status = status
        ticket.since = at

    def _insert_bug(self, ticket_id: str, status: Status, at: float) -> None:
        """A bug row lands directly under its parent's existing children."""
        parent = ticket_id.rsplit("-bug-", 1)[0]
        self._tickets[ticket_id] = Ticket(id=ticket_id, status=status, since=at, is_bug=True)
        after = self._order.index(parent) + self._bug_count(parent) + 1
        self._order.insert(after, ticket_id)

    def _reset(self, at: float) -> None:
        """Row order is fixed here at the start and never re-sorted."""
        self._order = list(TICKET_IDS)
        self._tickets = {
            ticket_id: Ticket(id=ticket_id, status=Status.PENDING, since=at, is_bug=False)
            for ticket_id in TICKET_IDS
        }
        self._next_event = 0


async def main() -> None:
    async with RichTerminal().live(Dashboard().build, fps=8):
        while True:
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped")
