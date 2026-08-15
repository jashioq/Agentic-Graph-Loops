"""The ticket dashboard, driven by a scripted run instead of real agents.

Ten Android tickets move through ninety seconds of work: three at a time, one
stretch waiting on the user, and two tickets sent back by review to wait on bugs
filed underneath them. Everything goes through the real `set_status` and
`file_bugs`, so the run this draws is one the workflow could actually produce —
the graph refuses an illegal move here exactly as it would in a live run.

Nothing pushes updates. The live loop rebuilds the frame several times a second
by calling `render`, which is why the timers tick and why each row's activity
line can change on its own schedule.

    uv run python demos/tickets_dashboard_demo.py

Ctrl-C to stop. The timeline loops.
"""

import asyncio
import time
from dataclasses import dataclass, field

from agl.core.terminal import Screen
from agl.core.terminal.impl.rich_terminal import RichTerminal
from agl.runtime.dag import Dag
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.render import render
from agl.workflows.tickets.state import Live, RunState, add_tickets, file_bugs, set_status

LABEL = "add-auth-ticket-18732"

# The features, in the order they were decomposed. A ticket that waits on
# another is the ordinary case, not the interesting one: the timeline below
# never starts it early, and the graph would refuse if it tried.
FEATURES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("T-01", "Version catalog for auth deps", ()),
    ("T-02", "TokenStore and silent refresh", ()),
    ("T-03", "Auth interceptor in :network", ()),
    ("T-04", "Session cache in DataStore", ()),
    ("T-05", "Login screen in Compose", ()),
    ("T-06", "Account switch in the nav graph", ("T-04",)),
    ("T-07", "Sign-out confirmation dialog", ()),
    ("T-08", "Auth events for analytics", ()),
    ("T-09", "Biometric unlock prompt", ()),
    ("T-10", "Backfill auth unit tests", ("T-08",)),
)

# What review found, filed against the ticket that failed it. A bug's row
# appears the moment it is filed, indented under its parent.
BUGS: dict[str, str] = {
    "T-03-bug-1": "401 mapped wrong",
    "T-03-bug-2": "Refresh loops",
    "T-06-bug-1": "Back stack lost",
}

# (seconds into the cycle, ticket id, status). A bug id appearing here for the
# first time is what files it — `file_bugs` puts the row on the screen and sends
# the parent back to pending behind it.
TIMELINE: tuple[tuple[float, str, Status], ...] = (
    (0, "T-01", Status.IN_PROGRESS),
    (0, "T-02", Status.IN_PROGRESS),
    (0, "T-03", Status.IN_PROGRESS),
    (8, "T-01", Status.IN_REVIEW),
    (12, "T-02", Status.AWAITING_INPUT),
    (14, "T-01", Status.MERGING),
    (16, "T-01", Status.MERGED),
    (16, "T-04", Status.IN_PROGRESS),
    (18, "T-03", Status.IN_REVIEW),
    (20, "T-02", Status.IN_PROGRESS),
    (22, "T-03-bug-1", Status.PENDING),
    (23, "T-03-bug-2", Status.PENDING),
    (26, "T-02", Status.IN_REVIEW),
    (28, "T-03-bug-1", Status.IN_PROGRESS),
    (32, "T-02", Status.MERGING),
    (34, "T-02", Status.MERGED),
    (34, "T-05", Status.IN_PROGRESS),
    (36, "T-03-bug-1", Status.IN_REVIEW),
    (40, "T-04", Status.IN_REVIEW),
    (42, "T-03-bug-1", Status.MERGING),
    (44, "T-03-bug-1", Status.MERGED),
    (45, "T-03-bug-2", Status.IN_PROGRESS),
    (48, "T-04", Status.MERGING),
    (50, "T-04", Status.MERGED),
    (50, "T-06", Status.IN_PROGRESS),
    (52, "T-05", Status.AWAITING_INPUT),
    (54, "T-03-bug-2", Status.IN_REVIEW),
    (56, "T-03-bug-2", Status.MERGING),
    (58, "T-03-bug-2", Status.MERGED),
    (59, "T-05", Status.IN_PROGRESS),
    (60, "T-03", Status.IN_PROGRESS),
    (62, "T-06", Status.IN_REVIEW),
    (64, "T-06-bug-1", Status.PENDING),
    (66, "T-05", Status.IN_REVIEW),
    (68, "T-06-bug-1", Status.IN_PROGRESS),
    (70, "T-03", Status.IN_REVIEW),
    (72, "T-05", Status.MERGING),
    (73, "T-06-bug-1", Status.IN_REVIEW),
    (74, "T-05", Status.MERGED),
    (74, "T-07", Status.IN_PROGRESS),
    (76, "T-06-bug-1", Status.MERGING),
    (77, "T-06-bug-1", Status.MERGED),
    (78, "T-03", Status.MERGING),
    (79, "T-06", Status.IN_PROGRESS),
    (80, "T-03", Status.MERGED),
    (80, "T-08", Status.IN_PROGRESS),
    (82, "T-07", Status.IN_REVIEW),
    (83, "T-06", Status.IN_REVIEW),
    (84, "T-07", Status.MERGING),
    (85, "T-07", Status.MERGED),
    (85, "T-09", Status.IN_PROGRESS),
    (86, "T-06", Status.MERGING),
    (87, "T-06", Status.MERGED),
    (87, "T-08", Status.IN_REVIEW),
    (88, "T-08", Status.MERGING),
    (89, "T-08", Status.MERGED),
    (89, "T-10", Status.IN_PROGRESS),
    (90, "T-09", Status.IN_REVIEW),
    (92, "T-09", Status.MERGING),
    (93, "T-09", Status.MERGED),
    (94, "T-10", Status.IN_REVIEW),
    (96, "T-10", Status.MERGING),
    (98, "T-10", Status.MERGED),
)

# Long enough to leave the finished run on the screen for a beat before it
# starts over.
CYCLE_SECONDS = 106.0

# What an agent reports while it works. Implementation and review are the two
# places an agent is wired to `on_activity`; merging is a Gradle build driven by
# Python, so no line ever arrives for it.
IMPLEMENTING: tuple[str, ...] = (
    "Reading TokenStore.kt",
    'Grep "fun refresh"',
    "Edit LoginScreen.kt",
    "Bash ./gradlew assembleDebug",
    "Edit AuthInterceptor.kt",
    'Grep "suspend fun authenticate"',
    "Write SessionCache.kt",
    "Bash ./gradlew :app:lintDebug",
)

REVIEWING: tuple[str, ...] = (
    "Reading the diff",
    'Grep "runBlocking"',
    "Reading AuthRepository.kt",
    "Bash ./gradlew testDebugUnitTest",
    "Checking against the deliverables",
    'Grep "TODO"',
    "Reading AndroidManifest.xml",
)

# How long one message stays up, per ticket. Deliberately coprime-ish and never
# shared, so three rows updating at once never fall into step.
BASE_PERIOD = 2.3
PERIOD_STEP = 0.37


@dataclass
class Demo:
    """The scripted run. `build` advances the clock and draws; nothing else does."""

    started: float = field(default_factory=time.monotonic)
    state: RunState = field(init=False)
    live: Live = field(init=False)
    _cycle: int = -1
    _next_event: int = 0

    def __post_init__(self) -> None:
        self._reset(self.started)

    def build(self) -> Screen:
        """One frame: apply everything the clock has passed, then render it."""
        now = time.monotonic()
        self._advance(now)
        self._report_activity(now)
        return render(self.state, self.live, now)

    # -- driving the run --------------------------------------------------

    def _advance(self, now: float) -> None:
        """Apply every event now due, restarting the run on each new cycle.

        A status is stamped with the time its event was due rather than the time
        the frame noticed it, so a row's timer stays honest even when several
        events come due between two frames.
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
        """Move a ticket, or file it as a bug the first time it is named."""
        if ticket_id not in self.state.tickets:
            self._file(ticket_id, at)
            return
        set_status(self.state, self.live, ticket_id, status, now=at)

    def _file(self, bug_id: str, at: float) -> None:
        """File one finding against the ticket its id names.

        One call per bug rather than one per review, so the two findings against
        `T-03` land a second apart and the count in its status cell can be
        watched going from one bug to two.
        """
        parent = bug_id.rsplit("-bug-", 1)[0]
        bug = Ticket(
            id=bug_id,
            title=BUGS[bug_id],
            status=Status.PENDING,
            deliverables=(BUGS[bug_id],),
            parent=parent,
        )
        file_bugs(self.state, self.live, parent, (bug,), now=at)

    def _reset(self, at: float) -> None:
        """Start the run over: a fresh graph, fresh tickets, fresh stamps."""
        self.state = RunState(label=LABEL, base_branch="main", dag=Dag(), tickets={})
        self.live = Live(started_at=at)
        add_tickets(
            self.state,
            self.live,
            tuple(
                Ticket(
                    id=ticket_id,
                    title=title,
                    status=Status.PENDING,
                    deliverables=(f"{ticket_id} deliverable",),
                    blocked_by=blocked_by,
                )
                for ticket_id, title, blocked_by in FEATURES
            ),
            now=at,
        )
        self._next_event = 0

    # -- what the agents are saying ---------------------------------------

    def _report_activity(self, now: float) -> None:
        """Stand in for `on_activity`: one line per working ticket, its own line.

        Recomputed rather than accumulated, so a ticket that stops working stops
        talking. Each ticket walks its own pool at its own period and from its
        own starting point, which is the thing worth looking at here: three rows
        changing independently rather than in lockstep.
        """
        for index, ticket_id in enumerate(self.state.tickets):
            ticket = self.state.tickets[ticket_id]
            pool = _POOLS.get(ticket.status)
            if pool is None:
                self.live.activity.pop(ticket_id, None)
                continue
            since = self.live.status_since.get(ticket_id, self.live.started_at)
            step = int(max(0.0, now - since) / (BASE_PERIOD + PERIOD_STEP * index))
            self.live.activity[ticket_id] = pool[(index * 3 + step) % len(pool)]


_POOLS: dict[Status, tuple[str, ...]] = {
    Status.IN_PROGRESS: IMPLEMENTING,
    Status.IN_REVIEW: REVIEWING,
}


async def main() -> None:
    async with RichTerminal().live(Demo().build, fps=8):
        while True:
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped")
