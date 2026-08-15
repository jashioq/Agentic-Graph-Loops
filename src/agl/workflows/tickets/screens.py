"""Every screen this workflow draws: run state and a clock reading in, a `Screen` out.

Layer: workflows. Pure — no I/O, no async, and no clock read anywhere. `now` is
a parameter, which is what makes every frame assertable: a test can demand a
timer reading exactly `0:38` because nothing here can disagree about when now
is. The live loop calls these several times a second beside a scheduler mutating
the same objects, so they read them and write nothing.

Three screens, one per stage, swapped on the one session the run opens.
`session` is the bare header the interview shows, `decompose` is that header
over the tickets an agent has proposed, and `dashboard` is the run itself.

The status a row shows is not always `Ticket.status`. A ticket whose review
filed bugs sits at `PENDING` with edges to them — there is no `fixing` status,
by design — so a plain `pending` would render the most interesting row on the
screen identically to work nobody has started. `_status_cell` recovers the
difference by counting the bug children that have not merged yet.

Activity is per row rather than one shared line at the foot of the screen. Three
agents run at once, and a single line could only ever show one of them, flicking
between whichever wrote last. Each row reports its own work, so all three are
legible at a glance.
"""

from collections.abc import Sequence

from agl.core.terminal import Color, Component, Row, Rows, Screen, Spacer, Text, Timer
from agl.runtime.dag import Dag
from agl.runtime.display import Board
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.state import Halt, Run, display_order

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
# included, so an id sits right against its title and a bug row's title may
# start further right, while every status cell on the screen still lines up.
INDENT = 6
LABEL_WIDTH = 40
STATUS_WIDTH = 17
GAP = 2
# What a timer occupies, so that a row without one still puts its activity in
# the activity column.
TIMER_WIDTH = 4

# The interview and decompose headers are sticky — one line, drawn on top of
# whatever else the screen has — so a wrapped line cannot be undrawn. Unlike a
# ticket row's activity, which is left whole because the terminal can wrap the
# row below it, this is elided to a fixed width instead.
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

    Falls back to the bare header while there are none yet, which is what the
    first proposal is drawn against. The graph is built here rather than kept:
    these tickets are not in the run until they are approved, so there is
    nothing to read a level ordering off but themselves.
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

    The three arguments are the three separate things a frame is made of, and
    keeping them apart is the point: `run` is the state document, `board` is
    ephemeral and display-only, and `label` is what the run was called, which
    lives in the record and never changes. Nothing here can write to any of
    them.

    Reads the board for what it holds and tolerates what it does not. The board
    may be empty — a fresh one, or one belonging to a process that did not start
    this run — and a missing stamp costs that row its timer rather than the
    frame.

    `now` reaches no row: every elapsed reading on the screen belongs to a
    `Timer`, which is handed a stamp and works the rest out when it is drawn.
    What `now` is for is the guarantee that a frame is a function of its
    arguments, so a test can demand a row reading exactly `0:38`.
    """
    return Screen(
        header=_header(run, label),
        content=Rows(*(_row(run, run.ticket(t), board) for t in display_order(run))),
        footer=_footer(board),
    )


def session_header(label: str, board: Board) -> Row:
    """The header for the interview and decompose screens: label, timer, activity.

    These sessions have no ticket, so the board's activity is keyed by `label`
    itself — the same fallback the run uses for a question's header, for the
    same reason. Unpadded, unlike the dashboard header: there is no status
    column or waiting marker to hold a place for here, only a label that is
    whatever length it is. Renders before any activity has arrived: with nothing
    recorded for `label`, the row is the label and the timer alone, with no gap
    held open for a string that has not shown up yet.
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

    The one state a person has to act on, so it is said twice: on the row and
    again up here, where it is visible without reading the list.

    You will hardly ever see it, and that is expected rather than a bug. Asking
    stops the live loop and takes the whole screen, and a second question queues
    on a lock and takes the screen the moment the first is answered — so the
    dashboard is visible exactly when no question is pending, which is exactly
    when there is no marker to draw. It is kept because it is correct in the
    moment it does show, it costs nothing, and it becomes load-bearing the day a
    question can be skipped and left waiting while the run carries on.
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

    Two rows: the reason and its detail in the words the halt was written in,
    then an instruction row that is the one thing distinguishing a halt a
    person can work through from one that means the process has to restart.
    Both stay `BOLD_RED` — this is the line on the screen most worth a person
    not missing, whichever kind it is.
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
    """A ticket's line: id, title, status, and what is happening to it right now.

    Takes no `now`, and wants none: a `Timer` works out its own elapsed at draw
    time, so a row hands over the stamp and stays a description of what to draw
    rather than a reading of how long it has been.
    """
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

    A bug row is red as well as indented. Indentation alone made one read as a
    feature ticket a column over, and a bug is the row on the screen most worth
    noticing: something the run found wrong with work it had already done.

    Merged rows are dimmed rather than dropped: a list that shrinks as work
    finishes takes the record of what was done off the screen, and every row
    below it jumps. A merged bug keeps a dimmer red on its id, so the finished
    row still says what kind of row it was.

    The title takes whatever the id left of the block and is cut to fit it.
    Titles are written by an agent and one will eventually be long enough to
    reach the status column; a dashboard whose columns move is harder to read
    than one that abbreviates. The id is never cut — it is the handle a person
    types — so an id longer than the block costs its own row's alignment and
    nobody else's.
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
    """What a row says it is doing, which is not always what its status is called."""
    if ticket.status is Status.PENDING:
        bugs = _open_bugs(run, ticket.id)
        if bugs:
            plural = "s" if bugs != 1 else ""
            return Text(f"{f'pending ({bugs} bug{plural})':<{STATUS_WIDTH}}", Color.RED)
    shown = _STATUS_TEXT[ticket.status]
    return Text(f"{shown:<{STATUS_WIDTH}}", _STATUS_COLOR[ticket.status])


def _open_bugs(run: Run, parent_id: str) -> int:
    """How many bugs filed against a ticket have yet to merge.

    Counted from the tickets rather than the graph edges: the parent's blockers
    include feature tickets it merely waits its turn behind, and those are not
    what makes a row worth looking at.
    """
    return sum(
        1 for t in run.tickets if t.parent == parent_id and t.status is not Status.MERGED
    )


# -- the footer -----------------------------------------------------------


def _footer(board: Board) -> Rows:
    """How long the implementation loop has been going, set off by a blank line.

    Counts from the `approved` mark, not from `started_at`: the footer answers
    "how long has this run taken", and a person answering interview questions is
    not the run taking that long. The dashboard is only ever shown after tickets
    are approved; a board with no mark yet keeps the blank line and shows no
    clock, rather than showing one that would be counting the wrong thing.
    """
    approved = board.since(APPROVED)
    if approved is None:
        return Rows(Row(), Row())
    return Rows(Row(), Row(Timer(approved)))
