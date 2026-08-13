"""One frame of the ticket dashboard: run state and a clock reading in, a `Screen` out.

Layer: workflows. Pure — no I/O, no async, and no clock read anywhere. `now` is
a parameter, which is what makes every frame assertable: a test can demand a
timer reading exactly `0:38` because nothing here can disagree about when now
is. The live loop calls this several times a second beside a scheduler mutating
the same objects, so it reads them and writes nothing.

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

from agl.core.terminal import Color, Component, Row, Rows, Screen, Spacer, Text, Timer
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.state import Live, RunState, display_order

__all__ = ["render"]

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


def render(state: RunState, live: Live, now: float) -> Screen:
    """The whole dashboard as of `now`: label and markers, one row per ticket, elapsed.

    Reads `live` for what it holds and tolerates what it does not. `Live` is
    ephemeral and may be empty — a fresh one, or one that lost a ticket to an
    unwound mutation — and a missing stamp costs that row its timer rather than
    the frame.

    `now` reaches no row: every elapsed reading on the screen belongs to a
    `Timer`, which is handed a stamp and works the rest out when it is drawn.
    What `now` is for is the guarantee that a frame is a function of its
    arguments, so a test can demand a row reading exactly `0:38`.
    """
    return Screen(
        header=_header(state),
        content=Rows(*(_row(state, live, state.tickets[t]) for t in display_order(state))),
        footer=_footer(live),
    )


# -- the header -----------------------------------------------------------


def _header(state: RunState) -> Rows:
    """The run label, whatever a person must not miss, and a blank line under it."""
    label = Text(f"{state.label:<{LABEL_WIDTH + STATUS_WIDTH}}", Color.CYAN)
    waiting = _waiting_marker(state)
    title = Row(label, waiting) if waiting is not None else Row(Text(state.label, Color.CYAN))
    if state.halt is None:
        return Rows(title, Row())
    return Rows(title, _halt_banner(state.halt.reason, state.halt.detail), Row())


def _waiting_marker(state: RunState) -> Text | None:
    """`⏸ T-04 needs input`, or a count once one row cannot say it.

    The one state a person has to act on, so it is said twice: on the row and
    again up here, where it is visible without reading the list.
    """
    waiting = [t.id for t in state.tickets.values() if t.status is Status.AWAITING_INPUT]
    if not waiting:
        return None
    if len(waiting) == 1:
        return Text(f"⏸ {waiting[0]} needs input", Color.BOLD_YELLOW)
    return Text(f"⏸ {len(waiting)} tickets need input", Color.BOLD_YELLOW)


def _halt_banner(reason: str, detail: str) -> Row:
    """Why the run stopped, in the words the halt was written in."""
    banner = Text(f"■ halted: {reason}", Color.BOLD_RED)
    if not detail:
        return Row(banner)
    return Row(banner, Spacer(GAP), Text(detail, Color.GREY))


# -- one ticket -----------------------------------------------------------


def _row(state: RunState, live: Live, ticket: Ticket) -> Row:
    """A ticket's line: id, title, status, and what is happening to it right now.

    Takes no `now`, and wants none: a `Timer` works out its own elapsed at draw
    time, so a row hands over the stamp and stays a description of what to draw
    rather than a reading of how long it has been.
    """
    cells: list[Component] = [*_label(ticket), _status_cell(state, ticket)]
    stamp = live.status_since.get(ticket.id) if ticket.status in _TICKING else None
    activity = live.activity.get(ticket.id)
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

    Merged rows are dimmed rather than dropped: a list that shrinks as work
    finishes takes the record of what was done off the screen, and every row
    below it jumps.

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
    title = Text(_fit(ticket.title, title_width), Color.DIM_GREY if dim else Color.GREY)
    if not ticket.is_bug:
        label = Text(ticket.id, Color.DIM_GREY if dim else Color.WHITE)
        return (label, Spacer(GAP), title)
    return (Spacer(INDENT), Text(ticket.id, Color.DIM_GREY), Spacer(GAP), title)


def _fit(content: str, width: int) -> str:
    """`content` in exactly `width` columns: padded if short, elided if long."""
    if len(content) <= width:
        return f"{content:<{width}}"
    return content[: width - 1] + "…" if width > 1 else content[:width]


def _status_cell(state: RunState, ticket: Ticket) -> Text:
    """What a row says it is doing, which is not always what its status is called."""
    if ticket.status is Status.PENDING:
        bugs = _open_bugs(state, ticket.id)
        if bugs:
            plural = "s" if bugs != 1 else ""
            return Text(f"{f'pending ({bugs} bug{plural})':<{STATUS_WIDTH}}", Color.RED)
    shown = _STATUS_TEXT[ticket.status]
    return Text(f"{shown:<{STATUS_WIDTH}}", _STATUS_COLOR[ticket.status])


def _open_bugs(state: RunState, parent_id: str) -> int:
    """How many bugs filed against a ticket have yet to merge.

    Counted from the tickets rather than the graph edges: the parent's blockers
    include feature tickets it merely waits its turn behind, and those are not
    what makes a row worth looking at.
    """
    return sum(
        1
        for t in state.tickets.values()
        if t.parent == parent_id and t.status is not Status.MERGED
    )


# -- the footer -----------------------------------------------------------


def _footer(live: Live) -> Rows:
    """How long the whole run has been going, set off by a blank line."""
    return Rows(Row(), Row(Timer(live.started_at)))
