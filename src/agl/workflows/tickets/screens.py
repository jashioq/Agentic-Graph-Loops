"""Every screen this workflow draws: run state and a clock reading in, a `Screen` out.

Layer: workflows. Pure — no I/O, no async, and no clock read anywhere, so every
frame is a function of its arguments and a test can demand a timer reading
exactly `0:38`. The live loop calls these several times a second beside a
scheduler mutating the same objects: they read, and write nothing. Three
screens, one per stage, swapped on the one session the run opens.
"""

from collections.abc import Sequence

from agl.core.terminal import Color, Component, Row, Rows, Screen, Spacer, Text, Timer
from agl.runtime.dag import Dag
from agl.runtime.display import Board
from agl.workflows.tickets.errors import Halt
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import Run, display_order

__all__ = [
    "APPROVED",
    "SESSION_ACTIVITY_WIDTH",
    "dashboard",
    "decompose",
    "session",
    "session_header",
]

APPROVED = "approved"
"""The board mark the dashboard's footer counts from: when tickets were approved."""

# The columns. `LABEL_WIDTH` covers the id and the title together, indent
# included, so every status cell on the screen lines up.
INDENT = 6
LABEL_WIDTH = 40
STATUS_WIDTH = 17
GAP = 2
# What a timer occupies, so that a row without one still puts its activity in
# the activity column.
TIMER_WIDTH = 4

# The session headers are sticky one-liners drawn on top of the screen, so a
# wrapped line could not be undrawn: their activity is elided rather than left
# whole the way a ticket row's is.
SESSION_ACTIVITY_WIDTH = 40

# The statuses an agent — or the merge queue — is actually working. A ticket
# waiting on a person is not making progress, and a clock ticking beside it
# would read as though it were.
_TICKING = frozenset({Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING})

_STATUS_TEXT: dict[Status, str] = {
    Status.PENDING: "pending",
    Status.IN_PROGRESS: "in progress",
    Status.IN_REVIEW: "in review",
    Status.MERGING: "merging",
    Status.MERGED: "merged",
    Status.AWAITING_INPUT: "waiting for you",
}

_STATUS_COLOR: dict[Status, Color] = {
    Status.PENDING: Color.WHITE,
    Status.IN_PROGRESS: Color.BLUE,
    Status.IN_REVIEW: Color.MAGENTA,
    Status.MERGING: Color.YELLOW,
    Status.MERGED: Color.DIM_GREEN,
    Status.AWAITING_INPUT: Color.BOLD_YELLOW,
}


# -- the three screens ----------------------------------------------------


def session(label: str, board: Board) -> Screen:
    """The interview's screen: the session header and nothing under it."""
    return Screen(header=session_header(label, board), content=Rows())


def decompose(label: str, board: Board, tickets: Sequence[Ticket]) -> Screen:
    """The proposed tickets, in dependency order, under the session header.

    Falls back to the bare header while there are none yet. These tickets are
    not in the run until they are approved, so the graph is built here rather
    than read off one.
    """
    if not tickets:
        return session(label, board)
    dag = Dag()
    for ticket in tickets:
        dag.add_node(ticket.id)
    for ticket in tickets:
        for blocker in ticket.blocked_by:
            dag.add_edge(ticket.id, blocker)
    by_id = {ticket.id: ticket for ticket in tickets}
    rows = []
    for level in dag.levels():
        for ticket_id in level:
            ticket = by_id[ticket_id]
            blocked = ", ".join(ticket.blocked_by) if ticket.blocked_by else "—"
            rows.append(Row(Text(f"{ticket.id}: {ticket.title} (blocked by: {blocked})")))
    return Screen(header=session_header(label, board), content=Rows(*rows))


def dashboard(run: Run, board: Board, label: str, now: float) -> Screen:
    """The whole dashboard as of `now`: label and markers, one row per ticket, elapsed.

    `run` is the state document, `board` is display-only and may be empty — a
    missing stamp costs that row its timer, not the frame. `now` reaches no row:
    every elapsed reading belongs to a `Timer`, which is handed a stamp and
    works the rest out when it is drawn.
    """
    return Screen(
        header=_header(run, label),
        content=Rows(*(_row(run, run.ticket(t), board) for t in display_order(run))),
        footer=_footer(board),
    )


def session_header(label: str, board: Board) -> Row:
    """The header for the interview and decompose screens: label, timer, activity.

    These sessions have no ticket, so the board's activity is keyed by `label`
    itself. Unpadded, and with no gap held open for an activity that has not
    arrived yet.
    """
    cells: list[Component] = [Text(label, Color.CYAN), Spacer(GAP), Timer(board.started_at)]
    activity = board.activity.get(label)
    if activity is not None:
        cells.extend((Spacer(GAP), Text(_fit(activity, SESSION_ACTIVITY_WIDTH), Color.DIM_GREY)))
    return Row(*cells)


# -- the header -----------------------------------------------------------


def _header(run: Run, label: str) -> Rows:
    """The run label, whatever a person must not miss, and a blank line under it."""
    padded = Text(f"{label:<{LABEL_WIDTH + STATUS_WIDTH}}", Color.CYAN)
    waiting = _waiting_marker(run)
    title = Row(padded, waiting) if waiting is not None else Row(Text(label, Color.CYAN))
    if run.halt is None:
        return Rows(title, Row())
    return Rows(title, *_halt_banner(run.halt), Row())


def _waiting_marker(run: Run) -> Text | None:
    """`⏸ T-04 needs input`, or a count once one row cannot say it.

    The one state a person has to act on, said twice: on the row, and again here
    where it is visible without reading the list. Rarely seen, because a pending
    question takes the whole screen — it becomes load-bearing the day one can be
    left waiting while the run carries on.
    """
    waiting = [t.id for t in run.tickets if t.status is Status.AWAITING_INPUT]
    if not waiting:
        return None
    if len(waiting) == 1:
        return Text(f"⏸ {waiting[0]} needs input", Color.BOLD_YELLOW)
    return Text(f"⏸ {len(waiting)} tickets need input", Color.BOLD_YELLOW)


_RESUME_HINT = "fix it, then press enter to continue"
_RESTART_HINT = "this cannot be resumed — stop the run and restart it"


def _halt_banner(halt: Halt) -> tuple[Row, Row]:
    """Why the run stopped, and what a person can do about it.

    Two rows: the reason and its detail, then the hint that distinguishes a halt
    a person can work through from one that means the process has to restart.
    """
    banner = Text(f"■ halted: {halt.reason}", Color.BOLD_RED)
    if not halt.detail:
        reason_row = Row(banner)
    else:
        reason_row = Row(banner, Spacer(GAP), Text(halt.detail, Color.GREY))
    hint = _RESUME_HINT if halt.resumable else _RESTART_HINT
    return reason_row, Row(Text(hint, Color.BOLD_RED))


# -- one ticket -----------------------------------------------------------


def _row(run: Run, ticket: Ticket, board: Board) -> Row:
    """A ticket's line: id, title, status, and what is happening to it right now."""
    cells: list[Component] = [*_label(ticket), _status_cell(run, ticket)]
    stamp = board.status_since.get(ticket.id) if ticket.status in _TICKING else None
    activity = board.activity.get(ticket.id)
    if stamp is not None:
        cells.append(Timer(stamp))
    elif activity is not None:
        # Nothing to tick, but the activity column stays where the eye expects it.
        cells.append(Spacer(TIMER_WIDTH))
    if activity is not None:
        cells.extend((Spacer(GAP), Text(activity, Color.DIM_GREY)))
    return Row(*cells)


def _label(ticket: Ticket) -> tuple[Component, ...]:
    """The id and title block. Bug rows are indented under the ticket they fix.

    A bug row is red as well as indented; merged rows are dimmed rather than
    dropped, so the record of what was done stays on the screen. The title is
    cut to whatever the id leaves of the block; the id is never cut, because it
    is the handle a person types.
    """
    dim = ticket.status is Status.MERGED
    indent = INDENT if ticket.is_bug else 0
    title_width = max(1, LABEL_WIDTH - indent - len(ticket.id) - GAP)
    fitted = _fit(ticket.title, title_width)
    if not ticket.is_bug:
        label = Text(ticket.id, Color.DIM_GREY if dim else Color.WHITE)
        title = Text(fitted, Color.DIM_GREY if dim else Color.GREY)
        return (label, Spacer(GAP), title)
    label = Text(ticket.id, Color.DIM_RED if dim else Color.RED)
    title = Text(fitted, Color.DIM_GREY if dim else Color.DIM_RED)
    return (Spacer(INDENT), label, Spacer(GAP), title)


def _fit(content: str, width: int) -> str:
    """`content` in exactly `width` columns: padded if short, elided if long."""
    if len(content) <= width:
        return f"{content:<{width}}"
    return content[: width - 1] + "…" if width > 1 else content[:width]


def _status_cell(run: Run, ticket: Ticket) -> Text:
    """What a row says it is doing, which is not always what its status is called.

    A ticket whose review filed bugs sits at `PENDING` — there is no `fixing`
    status — so the count of open bugs is what tells it apart from work nobody
    has started.
    """
    if ticket.status is Status.PENDING:
        bugs = _open_bugs(run, ticket.id)
        if bugs:
            plural = "s" if bugs != 1 else ""
            return Text(f"{f'pending ({bugs} bug{plural})':<{STATUS_WIDTH}}", Color.RED)
    shown = _STATUS_TEXT[ticket.status]
    return Text(f"{shown:<{STATUS_WIDTH}}", _STATUS_COLOR[ticket.status])


def _open_bugs(run: Run, parent_id: str) -> int:
    """How many bugs filed against a ticket have yet to merge.

    Counted from the tickets rather than the graph edges: a parent's blockers
    include feature tickets it merely waits its turn behind.
    """
    return sum(
        1 for t in run.tickets if t.parent == parent_id and t.status is not Status.MERGED
    )


# -- the footer -----------------------------------------------------------


def _footer(board: Board) -> Rows:
    """How long the implementation loop has been going, set off by a blank line.

    Counts from the `approved` mark, not from `started_at`: time a person spent
    answering interview questions is not the run taking that long. A board with
    no mark yet keeps the blank line and shows no clock.
    """
    approved = board.since(APPROVED)
    if approved is None:
        return Rows(Row(), Row())
    return Rows(Row(), Row(Timer(approved)))
