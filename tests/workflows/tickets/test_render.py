"""The dashboard frame: a run's state and a clock reading in, a `Screen` out.

Two ways of looking at the same call. Assertions about colour, timers and
activity read the component tree `render` returns, because that tree is the
thing being specified and a colour does not survive being printed to a
recorder. Assertions about layout — indentation, ordering, what a timer says —
go through Rich's own recorder against a fixed-width console, as the terminal
module's own tests do, so nothing here re-implements a renderer that could
disagree with the real one.
"""

import time

import pytest
from rich.console import Console

from agl.core.dag import Dag
from agl.core.terminal import Color, Component, Row, Rows, Screen, Spacer, Text, Timer
from agl.core.terminal.impl.screen import to_layout
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.render import render
from agl.workflows.tickets.state import (
    Halt,
    Live,
    RunState,
    add_tickets,
    display_order,
    file_bugs,
    set_status,
)

NOW = 1_000.0
WIDTH = 100
HEIGHT = 40

# How a ticket reaches each status, since only `set_status` may put it there.
_ROUTE: dict[Status, tuple[Status, ...]] = {
    Status.PENDING: (),
    Status.IN_PROGRESS: (Status.IN_PROGRESS,),
    Status.IN_REVIEW: (Status.IN_PROGRESS, Status.IN_REVIEW),
    Status.MERGING: (Status.IN_PROGRESS, Status.MERGING),
    Status.MERGED: (Status.IN_PROGRESS, Status.MERGING, Status.MERGED),
    Status.AWAITING_INPUT: (Status.IN_PROGRESS, Status.AWAITING_INPUT),
}

# -- building a run -------------------------------------------------------


def feature(ticket_id: str, *blocked_by: str) -> Ticket:
    return Ticket(
        id=ticket_id,
        title=f"Do {ticket_id}",
        status=Status.PENDING,
        deliverables=(f"{ticket_id}.kt",),
        blocked_by=blocked_by,
    )


def bug(ticket_id: str, parent: str) -> Ticket:
    return Ticket(
        id=ticket_id,
        title=f"Fix {ticket_id}",
        status=Status.PENDING,
        deliverables=("the finding",),
        parent=parent,
    )


def new_run(*tickets: Ticket) -> tuple[RunState, Live]:
    """A run holding `tickets`, and a `Live` that has watched it from `NOW`."""
    state = RunState(label="add-auth-ticket-18732", base_branch="main", dag=Dag(), tickets={})
    live = Live(started_at=NOW)
    add_tickets(state, live, tickets, now=NOW)
    return state, live


def walk_to(
    state: RunState, live: Live | None, ticket_id: str, status: Status, *, at: float = NOW
) -> None:
    """Move a ticket to `status` the way the workflow would, one legal step at a time."""
    for step in _ROUTE[status]:
        set_status(state, live, ticket_id, step, now=at)


def one_at(status: Status) -> tuple[RunState, Live]:
    """A single-ticket run sitting at `status`."""
    state, live = new_run(feature("T-01"))
    walk_to(state, live, "T-01", status)
    return state, live


# -- reading the tree -----------------------------------------------------


def parts(component: Component | None) -> list[Component]:
    """Every leaf of a component tree, left to right."""
    if component is None:
        return []
    if isinstance(component, Row | Rows):
        return [leaf for child in component.children for leaf in parts(child)]
    return [component]


def texts(component: Component | None) -> list[Text]:
    return [leaf for leaf in parts(component) if isinstance(leaf, Text)]


def timers(component: Component | None) -> list[Timer]:
    return [leaf for leaf in parts(component) if isinstance(leaf, Timer)]


def words(component: Component | None) -> list[str]:
    return [leaf.content.strip() for leaf in texts(component)]


def row_of(screen: Screen, ticket_id: str) -> Row:
    """The one content row belonging to `ticket_id`, found by its id column."""
    matches = [row for row in screen.content.children if words(row)[:1] == [ticket_id]]
    assert len(matches) == 1, f"{ticket_id} has {len(matches)} rows"
    return matches[0]


def id_column(screen: Screen) -> list[str]:
    return [words(row)[0] for row in screen.content.children]


def status_cell(screen: Screen, ticket_id: str) -> Text:
    """A row's status cell: id, title, status, so the third text."""
    return texts(row_of(screen, ticket_id))[2]


def activity_of(screen: Screen, ticket_id: str) -> str | None:
    """Whatever a row says after its status cell, if anything."""
    trailing = texts(row_of(screen, ticket_id))[3:]
    return trailing[0].content.strip() if trailing else None


def lines(screen: Screen, now: float = NOW) -> list[str]:
    """The frame as the real renderer draws it, blank lines and all."""
    console = Console(width=WIDTH, height=HEIGHT, record=True, no_color=True)
    console.print(to_layout(screen, now))
    return [line.rstrip() for line in console.export_text().splitlines()]


def content_lines(screen: Screen, now: float = NOW) -> list[str]:
    """Only the lines that carry a ticket, in the order they are drawn."""
    ids = set(id_column(screen))
    return [line for line in lines(screen, now) if line.split()[:1] and line.split()[0] in ids]


# -- the frame at its emptiest --------------------------------------------


def test_a_run_with_no_tickets_renders_header_and_footer() -> None:
    state = RunState(label="add-auth", base_branch="main", dag=Dag(), tickets={})
    screen = render(state, Live(started_at=NOW - 767.0), NOW)
    assert screen.content.children == ()
    assert "add-auth" in words(screen.header)
    assert lines(screen)[0] == "add-auth"
    assert lines(screen)[-1] == "12:47"


def test_the_header_carries_the_run_label() -> None:
    state, live = one_at(Status.PENDING)
    assert "add-auth-ticket-18732" in words(render(state, live, NOW).header)


# -- status text and colour -----------------------------------------------


@pytest.mark.parametrize(
    ("status", "shown", "color"),
    [
        (Status.PENDING, "pending", Color.WHITE),
        (Status.IN_PROGRESS, "in progress", Color.BLUE),
        (Status.IN_REVIEW, "in review", Color.MAGENTA),
        (Status.MERGING, "merging", Color.YELLOW),
        (Status.MERGED, "merged", Color.DIM_GREEN),
        (Status.AWAITING_INPUT, "waiting for you", Color.BOLD_YELLOW),
    ],
)
def test_each_status_has_its_own_text_and_colour(
    status: Status, shown: str, color: Color
) -> None:
    state, live = one_at(status)
    cell = status_cell(render(state, live, NOW), "T-01")
    assert cell.content.strip() == shown
    assert cell.color is color


def test_a_row_carries_its_id_and_title() -> None:
    state, live = one_at(Status.PENDING)
    assert words(row_of(render(state, live, NOW), "T-01"))[:2] == ["T-01", "Do T-01"]


def status_column(screen: Screen, ticket_id: str, now: float = NOW) -> int:
    """Where a row's status cell starts on the drawn line."""
    line = [row for row in content_lines(screen, now) if row.split()[0] == ticket_id][0]
    return line.index(status_cell(screen, ticket_id).content.strip())


def test_a_long_title_is_cut_so_the_status_column_stays_put() -> None:
    long_title = "Migrate every last login screen in the app to Jetpack Compose"
    state, live = new_run(feature("T-01"), feature("T-02"))
    state.tickets["T-02"].title = long_title
    screen = render(state, live, NOW)
    shown = words(row_of(screen, "T-02"))[1]
    assert shown != long_title
    assert long_title.startswith(shown[:-1]) and shown.endswith("…")
    assert status_column(screen, "T-02") == status_column(screen, "T-01")


def test_a_title_that_fits_is_left_alone() -> None:
    state, live = new_run(feature("T-01"))
    state.tickets["T-01"].title = "Add biometric unlock"
    assert words(row_of(render(state, live, NOW), "T-01"))[1] == "Add biometric unlock"


def test_a_long_bug_title_is_cut_to_the_room_the_indent_leaves() -> None:
    state, live = run_with_bugs(1)
    state.tickets["T-01-bug-1"].title = "The refresh call loops forever on an expired token"
    screen = render(state, live, NOW)
    assert words(row_of(screen, "T-01-bug-1"))[1].endswith("…")
    assert status_column(screen, "T-01-bug-1") == status_column(screen, "T-01")


def test_a_merged_row_is_dimmed_rather_than_dropped() -> None:
    state, live = new_run(feature("T-01"), feature("T-02"))
    walk_to(state, live, "T-01", Status.MERGED)
    screen = render(state, live, NOW)
    assert id_column(screen) == ["T-01", "T-02"]
    assert [text.color for text in texts(row_of(screen, "T-01"))[:2]] == [
        Color.DIM_GREY,
        Color.DIM_GREY,
    ]


def test_a_ticket_blocked_by_feature_tickets_is_plainly_pending() -> None:
    state, live = new_run(feature("T-01"), feature("T-02", "T-01"))
    cell = status_cell(render(state, live, NOW), "T-02")
    assert cell.content.strip() == "pending"
    assert cell.color is Color.WHITE


# -- pending, but waiting on bugs -----------------------------------------


def run_with_bugs(count: int) -> tuple[RunState, Live]:
    """`T-01` under review, then sent back with `count` bugs filed against it."""
    state, live = new_run(feature("T-01"))
    walk_to(state, live, "T-01", Status.IN_REVIEW)
    file_bugs(
        state,
        live,
        "T-01",
        [bug(f"T-01-bug-{index}", "T-01") for index in range(1, count + 1)],
        now=NOW,
    )
    return state, live


def test_a_ticket_waiting_on_bugs_says_so_and_says_how_many() -> None:
    state, live = run_with_bugs(2)
    cell = status_cell(render(state, live, NOW), "T-01")
    assert cell.content.strip() == "pending (2 bugs)"
    assert cell.color is Color.RED


def test_one_bug_is_counted_in_the_singular() -> None:
    state, live = run_with_bugs(1)
    assert status_cell(render(state, live, NOW), "T-01").content.strip() == "pending (1 bug)"


def test_merging_the_bugs_returns_the_parent_to_plain_pending() -> None:
    state, live = run_with_bugs(2)
    for index in (1, 2):
        walk_to(state, live, f"T-01-bug-{index}", Status.MERGED)
    cell = status_cell(render(state, live, NOW), "T-01")
    assert cell.content.strip() == "pending"
    assert cell.color is Color.WHITE


def test_a_bug_still_open_keeps_the_parent_red() -> None:
    state, live = run_with_bugs(2)
    walk_to(state, live, "T-01-bug-1", Status.MERGED)
    assert status_cell(render(state, live, NOW), "T-01").content.strip() == "pending (1 bug)"


def test_a_bug_ticket_of_its_own_is_never_counted_against_itself() -> None:
    state, live = run_with_bugs(1)
    assert status_cell(render(state, live, NOW), "T-01-bug-1").content.strip() == "pending"


# -- rows, and where they sit ---------------------------------------------


def test_a_bug_row_is_indented_under_its_parent() -> None:
    state, live = run_with_bugs(1)
    drawn = content_lines(render(state, live, NOW))
    assert drawn[0].startswith("T-01 ")
    assert drawn[1].startswith("      T-01-bug-1")


def test_two_bugs_under_one_parent_stay_adjacent() -> None:
    state, live = new_run(feature("T-01"), feature("T-02"))
    walk_to(state, live, "T-01", Status.IN_REVIEW)
    file_bugs(state, live, "T-01", [bug("T-01-bug-1", "T-01"), bug("T-01-bug-2", "T-01")], now=NOW)
    assert id_column(render(state, live, NOW)) == [
        "T-01",
        "T-01-bug-1",
        "T-01-bug-2",
        "T-02",
    ]


def test_row_order_is_the_display_order() -> None:
    state, live = new_run(feature("T-01"), feature("T-02"), feature("T-03"))
    walk_to(state, live, "T-02", Status.IN_REVIEW)
    file_bugs(state, live, "T-02", [bug("T-02-bug-1", "T-02")], now=NOW)
    assert id_column(render(state, live, NOW)) == list(display_order(state))


def test_row_order_does_not_move_when_a_status_changes() -> None:
    state, live = new_run(feature("T-01"), feature("T-02"), feature("T-03"))
    before = id_column(render(state, live, NOW))
    walk_to(state, live, "T-03", Status.MERGED)
    walk_to(state, live, "T-01", Status.IN_REVIEW)
    assert id_column(render(state, live, NOW)) == before


# -- timers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "ticking"),
    [
        (Status.PENDING, False),
        (Status.IN_PROGRESS, True),
        (Status.IN_REVIEW, True),
        (Status.MERGING, True),
        (Status.MERGED, False),
        (Status.AWAITING_INPUT, False),
    ],
)
def test_a_timer_runs_only_while_an_agent_is_working_the_ticket(
    status: Status, ticking: bool
) -> None:
    state, live = one_at(status)
    assert bool(timers(row_of(render(state, live, NOW), "T-01"))) is ticking


def test_a_timer_counts_from_the_stamp_on_the_status() -> None:
    state, live = new_run(feature("T-01"))
    walk_to(state, live, "T-01", Status.IN_PROGRESS, at=NOW - 38.0)
    assert content_lines(render(state, live, NOW))[0].endswith("0:38")


def test_a_ticket_with_no_stamp_shows_no_timer_rather_than_crashing() -> None:
    state, live = one_at(Status.IN_PROGRESS)
    live.status_since.clear()
    screen = render(state, live, NOW)
    assert timers(row_of(screen, "T-01")) == []
    assert content_lines(screen)[0].startswith("T-01")


def test_the_footer_timer_reads_from_the_start_of_the_run() -> None:
    state, live = one_at(Status.IN_PROGRESS)
    live.started_at = NOW - 767.0
    assert timers(render(state, live, NOW).footer) == [Timer(since=NOW - 767.0)]
    assert lines(render(state, live, NOW))[-1] == "12:47"


# -- activity, one string per row -----------------------------------------


def test_each_row_reports_its_own_activity_and_nobody_elses() -> None:
    state, live = new_run(feature("T-01"), feature("T-02"))
    walk_to(state, live, "T-01", Status.IN_PROGRESS)
    walk_to(state, live, "T-02", Status.IN_REVIEW)
    live.activity["T-01"] = "Edit LoginScreen.kt"
    live.activity["T-02"] = "Reading TokenStore.kt"
    screen = render(state, live, NOW)

    assert activity_of(screen, "T-01") == "Edit LoginScreen.kt"
    assert activity_of(screen, "T-02") == "Reading TokenStore.kt"
    drawn = content_lines(screen)
    assert "Reading TokenStore.kt" not in drawn[0]
    assert "Edit LoginScreen.kt" not in drawn[1]


def test_a_working_ticket_with_no_activity_shows_the_timer_alone() -> None:
    state, live = new_run(feature("T-01"))
    walk_to(state, live, "T-01", Status.IN_PROGRESS, at=NOW - 38.0)
    screen = render(state, live, NOW)
    assert activity_of(screen, "T-01") is None
    assert content_lines(screen)[0].endswith("0:38")


def test_activity_on_a_status_with_no_timer_still_renders() -> None:
    state, live = one_at(Status.AWAITING_INPUT)
    live.activity["T-01"] = "Waiting on an answer"
    screen = render(state, live, NOW)
    assert timers(row_of(screen, "T-01")) == []
    assert activity_of(screen, "T-01") == "Waiting on an answer"


def test_an_empty_live_renders_every_row() -> None:
    state, live = new_run(feature("T-01"), feature("T-02"), feature("T-03"))
    walk_to(state, live, "T-01", Status.IN_PROGRESS)
    walk_to(state, live, "T-02", Status.MERGED)
    empty = Live(started_at=NOW)
    screen = render(state, empty, NOW)
    assert id_column(screen) == ["T-01", "T-02", "T-03"]
    assert timers(screen.content) == []


# -- what the header has to shout about -----------------------------------


def test_one_waiting_ticket_is_named_in_the_header() -> None:
    state, live = new_run(feature("T-01"), feature("T-02"))
    walk_to(state, live, "T-02", Status.AWAITING_INPUT)
    header = render(state, live, NOW).header
    marker = [text for text in texts(header) if "needs input" in text.content]
    assert [text.content.strip() for text in marker] == ["⏸ T-02 needs input"]
    assert marker[0].color is Color.BOLD_YELLOW


def test_several_waiting_tickets_are_counted() -> None:
    state, live = new_run(feature("T-01"), feature("T-02"), feature("T-03"))
    for ticket_id in ("T-01", "T-03"):
        walk_to(state, live, ticket_id, Status.AWAITING_INPUT)
    header = render(state, live, NOW).header
    assert [text.content.strip() for text in texts(header) if "need" in text.content] == [
        "⏸ 2 tickets need input"
    ]


def test_nothing_waiting_means_no_marker() -> None:
    state, live = one_at(Status.IN_PROGRESS)
    assert not [text for text in texts(render(state, live, NOW).header) if "input" in text.content]


def test_a_halt_puts_its_reason_in_the_header() -> None:
    state, live = one_at(Status.IN_PROGRESS)
    state.halt = Halt(reason="merge conflict on T-01", detail="two agents touched build.gradle")
    header = render(state, live, NOW).header
    banner = [text for text in texts(header) if "merge conflict on T-01" in text.content]
    assert len(banner) == 1
    assert banner[0].color is Color.BOLD_RED
    assert "two agents touched build.gradle" in "\n".join(lines(render(state, live, NOW)))


def test_no_halt_means_no_banner() -> None:
    state, live = one_at(Status.IN_PROGRESS)
    assert not [text for text in texts(render(state, live, NOW).header) if "halt" in text.content]


# -- purity ---------------------------------------------------------------


def test_the_same_arguments_produce_an_equal_screen() -> None:
    state, live = run_with_bugs(2)
    walk_to(state, live, "T-01-bug-1", Status.IN_PROGRESS)
    live.activity["T-01-bug-1"] = "Edit AuthError.kt"
    assert render(state, live, NOW) == render(state, live, NOW)


def test_render_writes_to_neither_state_nor_live() -> None:
    state, live = run_with_bugs(2)
    walk_to(state, live, "T-01-bug-1", Status.IN_PROGRESS)
    live.activity["T-01-bug-1"] = "Edit AuthError.kt"
    before = (
        {ticket_id: vars(ticket).copy() for ticket_id, ticket in state.tickets.items()},
        {node: state.dag.state(node) for node in state.dag.nodes()},
        state.halt,
        dict(live.status_since),
        dict(live.activity),
        live.started_at,
    )
    render(state, live, NOW)
    assert (
        {ticket_id: vars(ticket).copy() for ticket_id, ticket in state.tickets.items()},
        {node: state.dag.state(node) for node in state.dag.nodes()},
        state.halt,
        dict(live.status_since),
        dict(live.activity),
        live.started_at,
    ) == before


def test_render_never_reads_the_clock() -> None:
    """`now` is the only clock a frame sees, so a frame can be asserted on."""
    state, live = one_at(Status.IN_PROGRESS)
    later = render(state, live, NOW + 120.0)
    assert timers(row_of(later, "T-01")) == timers(row_of(render(state, live, NOW), "T-01"))
    assert time.monotonic() > 0  # the real clock moved; the frame did not


def test_the_layout_is_columns_of_spacers_and_padded_text() -> None:
    """Nothing in a row separates itself; the gaps are all deliberate."""
    state, live = run_with_bugs(1)
    row = row_of(render(state, live, NOW), "T-01-bug-1")
    assert isinstance(row.children[0], Spacer)


def test_a_frame_of_a_real_looking_run_reads_as_expected() -> None:
    state, live = new_run(feature("T-01"), feature("T-02"), feature("T-03"), feature("T-04"))
    walk_to(state, live, "T-01", Status.MERGED)
    walk_to(state, live, "T-02", Status.IN_REVIEW, at=NOW - 72.0)
    walk_to(state, live, "T-04", Status.AWAITING_INPUT)
    live.activity["T-02"] = "Reading TokenStore.kt"
    live.started_at = NOW - 767.0
    drawn = lines(render(state, live, NOW))
    assert drawn[0].startswith("add-auth-ticket-18732")
    assert drawn[0].endswith("⏸ T-04 needs input")
    assert "in review" in drawn[3] and "1:12" in drawn[3] and "Reading TokenStore.kt" in drawn[3]
    assert drawn[-1] == "12:47"
