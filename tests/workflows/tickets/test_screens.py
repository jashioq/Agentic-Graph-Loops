"""Every screen this workflow draws: run state and a clock reading in, a `Screen` out.

Two ways of looking at the same call. Assertions about colour, timers and
activity read the component tree `render` returns, because that tree is the
thing being specified and a colour does not survive being printed to a
recorder. Assertions about layout — indentation, ordering, what a timer says —
go through Rich's own recorder against a fixed-width console, as the terminal
module's own tests do, so nothing here re-implements a renderer that could
disagree with the real one.
"""

import time
from dataclasses import dataclass, replace

import pytest
from rich.console import Console

from agl.core.terminal import Color, Component, Row, Rows, Screen, Spacer, Text, Timer
from agl.core.terminal.impl.screen import to_layout
from agl.runtime.display import Board
from agl.workflows.tickets.errors import Halt
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import (
    Run,
    display_order,
    with_bugs,
    with_halt,
    with_status,
    with_tickets,
)
from agl.workflows.tickets.screens import (
    APPROVED,
    SESSION_ACTIVITY_WIDTH,
    dashboard,
    decompose,
    session,
    session_header,
)

NOW = 1_000.0
WIDTH = 100
HEIGHT = 40
LABEL = "add-auth-ticket-18732"

# How a ticket reaches each status, one legal move at a time. A run does claim a
# pending ticket straight into a later status when git says the work is already
# there, but that is a resume, and what these frames are about is the ordinary
# life cycle.
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


def watched(started_at: float = NOW) -> Board:
    """A board that has been watching since `started_at`, tickets approved at `NOW`."""
    board = Board(started_at=started_at)
    board.mark(APPROVED, NOW)
    return board


@dataclass
class Scene:
    """The three things a dashboard frame is made of, held together for a test.

    They are separate arguments in the real call for a reason — the run is a
    document, the board is ephemeral, the label is neither — and this only puts
    them in one place so a test can say `draw(scene)`.
    """

    run: Run
    board: Board
    label: str = LABEL


def new_run(*tickets: Ticket) -> Scene:
    """A run holding `tickets`, watched since `NOW` and approved at `NOW`."""
    scene = Scene(with_tickets(Run(), tickets), watched())
    for ticket in tickets:
        scene.board.stamp(ticket.id, NOW)
    return scene


def walk_to(scene: Scene, ticket_id: str, status: Status, *, at: float = NOW) -> None:
    """Move a ticket to `status` the way the workflow would, one legal step at a time."""
    for step in _ROUTE[status]:
        scene.run = with_status(scene.run, ticket_id, step)
        scene.board.stamp(ticket_id, at)


def file_bugs(scene: Scene, parent_id: str, bugs: list[Ticket], *, at: float = NOW) -> None:
    """Send a ticket back behind its bugs, stamping every row that moved."""
    scene.run = with_bugs(scene.run, parent_id, bugs)
    for ticket_id in (parent_id, *(b.id for b in bugs)):
        scene.board.stamp(ticket_id, at)


def retitle(scene: Scene, ticket_id: str, title: str) -> None:
    """Give one ticket a different title, without going through a transition."""
    scene.run = replace(
        scene.run,
        tickets=tuple(
            replace(t, title=title) if t.id == ticket_id else t for t in scene.run.tickets
        ),
    )


def halted(scene: Scene, halt: Halt) -> None:
    scene.run = with_halt(scene.run, halt)


def draw(scene: Scene, now: float = NOW) -> Screen:
    """The dashboard this scene renders to."""
    return dashboard(scene.run, scene.board, scene.label, now)


def one_at(status: Status) -> Scene:
    """A single-ticket run sitting at `status`."""
    scene = new_run(feature("T-01"))
    walk_to(scene, "T-01", status)
    return scene


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
    board = Board(started_at=NOW)
    board.mark(APPROVED, NOW - 767.0)
    screen = dashboard(Run(), board, "add-auth", NOW)
    assert screen.content.children == ()
    assert "add-auth" in words(screen.header)
    assert lines(screen)[0] == "add-auth"
    assert lines(screen)[-1] == "12:47"


def test_the_header_carries_the_run_label() -> None:
    scene = one_at(Status.PENDING)
    assert "add-auth-ticket-18732" in words(draw(scene).header)


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
    scene = one_at(status)
    cell = status_cell(draw(scene), "T-01")
    assert cell.content.strip() == shown
    assert cell.color is color


def test_a_row_carries_its_id_and_title() -> None:
    scene = one_at(Status.PENDING)
    assert words(row_of(draw(scene), "T-01"))[:2] == ["T-01", "Do T-01"]


def status_column(screen: Screen, ticket_id: str, now: float = NOW) -> int:
    """Where a row's status cell starts on the drawn line."""
    line = [row for row in content_lines(screen, now) if row.split()[0] == ticket_id][0]
    return line.index(status_cell(screen, ticket_id).content.strip())


def test_a_long_title_is_cut_so_the_status_column_stays_put() -> None:
    long_title = "Migrate every last login screen in the app to Jetpack Compose"
    scene = new_run(feature("T-01"), feature("T-02"))
    retitle(scene, "T-02", long_title)
    screen = draw(scene)
    shown = words(row_of(screen, "T-02"))[1]
    assert shown != long_title
    assert long_title.startswith(shown[:-1]) and shown.endswith("…")
    assert status_column(screen, "T-02") == status_column(screen, "T-01")


def test_a_title_that_fits_is_left_alone() -> None:
    scene = new_run(feature("T-01"))
    retitle(scene, "T-01", "Add biometric unlock")
    assert words(row_of(draw(scene), "T-01"))[1] == "Add biometric unlock"


def test_a_long_bug_title_is_cut_to_the_room_the_indent_leaves() -> None:
    scene = run_with_bugs(1)
    retitle(scene, "T-01-bug-1", "The refresh call loops forever on an expired token")
    screen = draw(scene)
    assert words(row_of(screen, "T-01-bug-1"))[1].endswith("…")
    assert status_column(screen, "T-01-bug-1") == status_column(screen, "T-01")


def test_a_bug_row_is_red_so_it_does_not_read_as_a_feature_row() -> None:
    # Indentation alone made a bug look like a feature ticket one column over.
    scene = run_with_bugs(1)
    screen = draw(scene)
    assert [text.color for text in texts(row_of(screen, "T-01-bug-1"))[:2]] == [
        Color.RED,
        Color.DIM_RED,
    ]


def test_a_feature_row_keeps_the_colours_it_had() -> None:
    scene = run_with_bugs(1)
    screen = draw(scene)
    assert [text.color for text in texts(row_of(screen, "T-01"))[:2]] == [
        Color.WHITE,
        Color.GREY,
    ]


def test_a_merged_bug_row_dims_like_any_other_merged_row() -> None:
    scene = run_with_bugs(1)
    walk_to(scene, "T-01-bug-1", Status.MERGED)
    screen = draw(scene)
    assert [text.color for text in texts(row_of(screen, "T-01-bug-1"))[:2]] == [
        Color.DIM_RED,
        Color.DIM_GREY,
    ]


def test_a_merged_row_is_dimmed_rather_than_dropped() -> None:
    scene = new_run(feature("T-01"), feature("T-02"))
    walk_to(scene, "T-01", Status.MERGED)
    screen = draw(scene)
    assert id_column(screen) == ["T-01", "T-02"]
    assert [text.color for text in texts(row_of(screen, "T-01"))[:2]] == [
        Color.DIM_GREY,
        Color.DIM_GREY,
    ]


def test_a_ticket_blocked_by_feature_tickets_is_plainly_pending() -> None:
    scene = new_run(feature("T-01"), feature("T-02", "T-01"))
    cell = status_cell(draw(scene), "T-02")
    assert cell.content.strip() == "pending"
    assert cell.color is Color.WHITE


# -- pending, but waiting on bugs -----------------------------------------


def run_with_bugs(count: int) -> Scene:
    """`T-01` under review, then sent back with `count` bugs filed against it."""
    scene = new_run(feature("T-01"))
    walk_to(scene, "T-01", Status.IN_REVIEW)
    file_bugs(scene, "T-01", [bug(f"T-01-bug-{index}", "T-01") for index in range(1, count + 1)])
    return scene


def test_a_ticket_waiting_on_bugs_says_so_and_says_how_many() -> None:
    scene = run_with_bugs(2)
    cell = status_cell(draw(scene), "T-01")
    assert cell.content.strip() == "pending (2 bugs)"
    assert cell.color is Color.RED


def test_one_bug_is_counted_in_the_singular() -> None:
    scene = run_with_bugs(1)
    assert status_cell(draw(scene), "T-01").content.strip() == "pending (1 bug)"


def test_merging_the_bugs_returns_the_parent_to_plain_pending() -> None:
    scene = run_with_bugs(2)
    for index in (1, 2):
        walk_to(scene, f"T-01-bug-{index}", Status.MERGED)
    cell = status_cell(draw(scene), "T-01")
    assert cell.content.strip() == "pending"
    assert cell.color is Color.WHITE


def test_a_bug_still_open_keeps_the_parent_red() -> None:
    scene = run_with_bugs(2)
    walk_to(scene, "T-01-bug-1", Status.MERGED)
    assert status_cell(draw(scene), "T-01").content.strip() == "pending (1 bug)"


def test_a_bug_ticket_of_its_own_is_never_counted_against_itself() -> None:
    scene = run_with_bugs(1)
    assert status_cell(draw(scene), "T-01-bug-1").content.strip() == "pending"


# -- rows, and where they sit ---------------------------------------------


def test_a_bug_row_is_indented_under_its_parent() -> None:
    scene = run_with_bugs(1)
    drawn = content_lines(draw(scene, NOW))
    assert drawn[0].startswith("T-01 ")
    assert drawn[1].startswith("      T-01-bug-1")


def test_two_bugs_under_one_parent_stay_adjacent() -> None:
    scene = new_run(feature("T-01"), feature("T-02"))
    walk_to(scene, "T-01", Status.IN_REVIEW)
    file_bugs(scene, "T-01", [bug("T-01-bug-1", "T-01"), bug("T-01-bug-2", "T-01")])
    assert id_column(draw(scene)) == [
        "T-01",
        "T-01-bug-1",
        "T-01-bug-2",
        "T-02",
    ]


def test_row_order_is_the_display_order() -> None:
    scene = new_run(feature("T-01"), feature("T-02"), feature("T-03"))
    walk_to(scene, "T-02", Status.IN_REVIEW)
    file_bugs(scene, "T-02", [bug("T-02-bug-1", "T-02")])
    assert id_column(draw(scene)) == list(display_order(scene.run))


def test_row_order_does_not_move_when_a_status_changes() -> None:
    scene = new_run(feature("T-01"), feature("T-02"), feature("T-03"))
    before = id_column(draw(scene))
    walk_to(scene, "T-03", Status.MERGED)
    walk_to(scene, "T-01", Status.IN_REVIEW)
    assert id_column(draw(scene)) == before


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
    scene = one_at(status)
    assert bool(timers(row_of(draw(scene), "T-01"))) is ticking


def test_a_timer_counts_from_the_stamp_on_the_status() -> None:
    scene = new_run(feature("T-01"))
    walk_to(scene, "T-01", Status.IN_PROGRESS, at=NOW - 38.0)
    assert content_lines(draw(scene, NOW))[0].endswith("0:38")


def test_a_ticket_with_no_stamp_shows_no_timer_rather_than_crashing() -> None:
    scene = one_at(Status.IN_PROGRESS)
    scene.board.status_since.clear()
    screen = draw(scene)
    assert timers(row_of(screen, "T-01")) == []
    assert content_lines(screen)[0].startswith("T-01")


def test_the_footer_timer_reads_from_ticket_approval_not_session_start() -> None:
    scene = one_at(Status.IN_PROGRESS)
    scene.board.mark(APPROVED, NOW - 767.0)
    assert timers(draw(scene).footer) == [Timer(since=NOW - 767.0)]
    assert lines(draw(scene))[-1] == "12:47"


def test_the_footer_timer_does_not_move_when_the_session_start_does() -> None:
    """A long interview must not read as a long run: the two stamps are independent.

    A session that ran for 300 seconds before approval reads `0:00` on the
    dashboard's first frame — the footer is a function of the `approved` mark
    alone, never of `started_at`.
    """
    scene = one_at(Status.IN_PROGRESS)
    scene.board.started_at = NOW - 300.0
    scene.board.mark(APPROVED, NOW)
    assert timers(draw(scene).footer) == [Timer(since=NOW)]
    assert lines(draw(scene))[-1] == "0:00"


# -- activity, one string per row -----------------------------------------


def test_each_row_reports_its_own_activity_and_nobody_elses() -> None:
    scene = new_run(feature("T-01"), feature("T-02"))
    walk_to(scene, "T-01", Status.IN_PROGRESS)
    walk_to(scene, "T-02", Status.IN_REVIEW)
    scene.board.activity["T-01"] = "Edit LoginScreen.kt"
    scene.board.activity["T-02"] = "Reading TokenStore.kt"
    screen = draw(scene)

    assert activity_of(screen, "T-01") == "Edit LoginScreen.kt"
    assert activity_of(screen, "T-02") == "Reading TokenStore.kt"
    drawn = content_lines(screen)
    assert "Reading TokenStore.kt" not in drawn[0]
    assert "Edit LoginScreen.kt" not in drawn[1]


def test_a_working_ticket_with_no_activity_shows_the_timer_alone() -> None:
    scene = new_run(feature("T-01"))
    walk_to(scene, "T-01", Status.IN_PROGRESS, at=NOW - 38.0)
    screen = draw(scene)
    assert activity_of(screen, "T-01") is None
    assert content_lines(screen)[0].endswith("0:38")


def test_activity_on_a_status_with_no_timer_still_renders() -> None:
    scene = one_at(Status.AWAITING_INPUT)
    scene.board.activity["T-01"] = "Waiting on an answer"
    screen = draw(scene)
    assert timers(row_of(screen, "T-01")) == []
    assert activity_of(screen, "T-01") == "Waiting on an answer"


# -- activity with a role label on it -------------------------------------
#
# Three reviewers write to one row's `activity` during review, so phase 4
# prefixes each one's messages with its role — `quality · Read Auth.kt` — and
# the label belongs to whoever writes the string, not to `render`. Nothing here
# had to change for that; these tests are what keeps it true, and they pin how
# much room a label has before the row stops fitting.

LABELLED = "quality · Read AuthRepository.kt"


def in_review_with(activity: str) -> Scene:
    """`T-01` under review reporting `activity`, with `T-02` on the row below."""
    scene = new_run(feature("T-01"), feature("T-02"))
    walk_to(scene, "T-01", Status.IN_REVIEW, at=NOW - 72.0)
    scene.board.activity["T-01"] = activity
    return scene


def drawn_row(screen: Screen, ticket_id: str, now: float = NOW) -> tuple[list[str], int]:
    """Every line of the frame, and the index of the one `ticket_id` is drawn on."""
    drawn = lines(screen, now)
    index = next(i for i, line in enumerate(drawn) if line.startswith(f"{ticket_id} "))
    return drawn, index


def test_a_labelled_activity_is_drawn_whole_and_moves_nothing() -> None:
    scene = in_review_with(LABELLED)
    screen = draw(scene)

    assert activity_of(screen, "T-01") == LABELLED
    drawn, index = drawn_row(screen, "T-01")
    assert drawn[index].endswith(LABELLED)  # in full: no eliding, no wrapping
    assert drawn[index + 1].startswith("T-02")  # nothing was pushed down a line


def test_a_label_does_not_move_the_activity_column() -> None:
    # The label is part of the string, so it starts where any activity starts:
    # a labelled row and an unlabelled one stay aligned down the screen.
    labelled = in_review_with(LABELLED)
    walk_to(labelled, "T-02", Status.IN_REVIEW, at=NOW - 72.0)
    labelled.board.activity["T-02"] = "Read AuthRepository.kt"
    drawn = lines(draw(labelled))
    rows = [line for line in drawn if line.startswith(("T-01 ", "T-02 "))]
    assert rows[0].index(LABELLED) == rows[1].index("Read AuthRepository.kt")


def test_the_room_a_labelled_activity_has_is_what_the_screen_leaves() -> None:
    """`render` never cuts an activity string, so the terminal width is the limit.

    Titles are elided to keep the status column still; activity is the last
    thing on the line and has nothing to hold in place, so it is passed through
    as it was written. What that costs is stated here rather than left to be
    discovered: one column too many and the row wraps onto a second line,
    pushing everything below it down.
    """
    scene = in_review_with("x")
    _, index = drawn_row(draw(scene), "T-01")
    column = len(lines(draw(scene))[index]) - 1
    room = WIDTH - column
    assert (column, room) == (63, 37)

    fits, over = "q" * room, "q" * (room + 1)
    scene = in_review_with(fits)
    drawn, index = drawn_row(draw(scene), "T-01")
    assert drawn[index].endswith(fits) and drawn[index + 1].startswith("T-02")

    scene = in_review_with(over)
    drawn, index = drawn_row(draw(scene), "T-01")
    # Rich wraps at a word boundary, so one column too many does not trail off
    # the end of the line — the whole string leaves the row it belongs to.
    assert over not in drawn[index]
    assert drawn[index + 1].strip() == over
    assert drawn[index + 2].startswith("T-02")


def test_an_empty_board_renders_every_row() -> None:
    scene = new_run(feature("T-01"), feature("T-02"), feature("T-03"))
    walk_to(scene, "T-01", Status.IN_PROGRESS)
    walk_to(scene, "T-02", Status.MERGED)
    scene.board = watched()
    screen = draw(scene)
    assert id_column(screen) == ["T-01", "T-02", "T-03"]
    assert timers(screen.content) == []


# -- what the header has to shout about -----------------------------------


def test_one_waiting_ticket_is_named_in_the_header() -> None:
    scene = new_run(feature("T-01"), feature("T-02"))
    walk_to(scene, "T-02", Status.AWAITING_INPUT)
    header = draw(scene).header
    marker = [text for text in texts(header) if "needs input" in text.content]
    assert [text.content.strip() for text in marker] == ["⏸ T-02 needs input"]
    assert marker[0].color is Color.BOLD_YELLOW


def test_several_waiting_tickets_are_counted() -> None:
    scene = new_run(feature("T-01"), feature("T-02"), feature("T-03"))
    for ticket_id in ("T-01", "T-03"):
        walk_to(scene, ticket_id, Status.AWAITING_INPUT)
    header = draw(scene).header
    assert [text.content.strip() for text in texts(header) if "need" in text.content] == [
        "⏸ 2 tickets need input"
    ]


def test_nothing_waiting_means_no_marker() -> None:
    scene = one_at(Status.IN_PROGRESS)
    assert not [text for text in texts(draw(scene).header) if "input" in text.content]


def test_a_halt_puts_its_reason_in_the_header() -> None:
    scene = one_at(Status.IN_PROGRESS)
    halted(scene, Halt(reason="merge conflict on T-01", detail="two agents touched build.gradle"))
    header = draw(scene).header
    banner = [text for text in texts(header) if "merge conflict on T-01" in text.content]
    assert len(banner) == 1
    assert banner[0].color is Color.BOLD_RED
    assert "two agents touched build.gradle" in "\n".join(lines(draw(scene)))


def test_no_halt_means_no_banner() -> None:
    scene = one_at(Status.IN_PROGRESS)
    assert not [text for text in texts(draw(scene).header) if "halt" in text.content]


def test_a_resumable_halt_tells_a_person_enter_continues() -> None:
    scene = one_at(Status.IN_PROGRESS)
    halted(scene, Halt(reason="merge conflict on T-01", resumable=True))
    header = draw(scene).header
    hint = [text for text in texts(header) if "enter" in text.content.lower()]
    assert len(hint) == 1
    assert hint[0].color is Color.BOLD_RED


def test_a_non_resumable_halt_says_stop_and_restart() -> None:
    scene = one_at(Status.IN_PROGRESS)
    halted(scene, Halt(reason="build command not found", resumable=False))
    header = draw(scene).header
    hint = [text for text in texts(header) if "restart" in text.content.lower()]
    assert len(hint) == 1
    assert hint[0].color is Color.BOLD_RED
    assert not [text for text in texts(header) if "enter" in text.content.lower()]


def test_the_two_halt_banners_are_distinguishable() -> None:
    resumable = one_at(Status.IN_PROGRESS)
    halted(resumable, Halt(reason="x", resumable=True))

    stuck = one_at(Status.IN_PROGRESS)
    halted(stuck, Halt(reason="x", resumable=False))

    assert words(draw(resumable).header) != words(draw(stuck).header)


# -- purity ---------------------------------------------------------------


def test_the_same_arguments_produce_an_equal_screen() -> None:
    scene = run_with_bugs(2)
    walk_to(scene, "T-01-bug-1", Status.IN_PROGRESS)
    scene.board.activity["T-01-bug-1"] = "Edit AuthError.kt"
    assert draw(scene) == draw(scene)


def test_a_frame_writes_to_neither_the_run_nor_the_board() -> None:
    scene = run_with_bugs(2)
    walk_to(scene, "T-01-bug-1", Status.IN_PROGRESS)
    scene.board.activity["T-01-bug-1"] = "Edit AuthError.kt"

    def snapshot() -> tuple[object, ...]:
        return (
            scene.run,
            dict(scene.board.status_since),
            dict(scene.board.activity),
            scene.board.started_at,
        )

    before = snapshot()
    draw(scene)
    assert snapshot() == before


def test_a_frame_never_reads_the_clock() -> None:
    """`now` is the only clock a frame sees, so a frame can be asserted on."""
    scene = one_at(Status.IN_PROGRESS)
    later = draw(scene, NOW + 120.0)
    assert timers(row_of(later, "T-01")) == timers(row_of(draw(scene), "T-01"))
    assert time.monotonic() > 0  # the real clock moved; the frame did not


def test_the_layout_is_columns_of_spacers_and_padded_text() -> None:
    """Nothing in a row separates itself; the gaps are all deliberate."""
    scene = run_with_bugs(1)
    row = row_of(draw(scene), "T-01-bug-1")
    assert isinstance(row.children[0], Spacer)


def test_a_frame_of_a_real_looking_run_reads_as_expected() -> None:
    scene = new_run(feature("T-01"), feature("T-02"), feature("T-03"), feature("T-04"))
    walk_to(scene, "T-01", Status.MERGED)
    walk_to(scene, "T-02", Status.IN_REVIEW, at=NOW - 72.0)
    walk_to(scene, "T-04", Status.AWAITING_INPUT)
    scene.board.activity["T-02"] = "Reading TokenStore.kt"
    scene.board.mark(APPROVED, NOW - 767.0)
    drawn = lines(draw(scene))
    assert drawn[0].startswith("add-auth-ticket-18732")
    assert drawn[0].endswith("⏸ T-04 needs input")
    assert "in review" in drawn[3] and "1:12" in drawn[3] and "Reading TokenStore.kt" in drawn[3]
    assert drawn[-1] == "12:47"


# -- the interview and decompose header ------------------------------------


def header_activity(header: Row) -> str | None:
    """Whatever the header says after its label, if anything."""
    trailing = texts(header)[1:]
    return trailing[0].content.strip() if trailing else None


def test_the_session_header_carries_the_label() -> None:
    header = session_header("add-auth-ticket-18732", Board(started_at=NOW))
    assert words(header)[0] == "add-auth-ticket-18732"


def test_the_session_timer_reads_elapsed_since_the_session_began() -> None:
    header = session_header("add-auth", Board(started_at=NOW - 42.0))
    assert timers(header) == [Timer(since=NOW - 42.0)]
    assert lines(Screen(content=Rows(header)), NOW)[0].endswith("0:42")


def test_with_no_activity_the_header_is_the_label_and_timer_with_no_gap() -> None:
    header = session_header("add-auth", Board(started_at=NOW - 42.0))
    assert texts(header) == [Text("add-auth", Color.CYAN)]
    assert lines(Screen(content=Rows(header)), NOW)[0] == "add-auth  0:42"


def test_an_activity_written_through_the_boards_lookup_reaches_the_header() -> None:
    board = Board(started_at=NOW)
    board.activity["add-auth"] = "Read app/build.gradle.kts"
    header = session_header("add-auth", board)
    assert header_activity(header) == "Read app/build.gradle.kts"


def test_both_screens_render_before_any_activity_has_arrived() -> None:
    """The scene during the first seconds of every run: no crash, no gap."""
    board = Board(started_at=NOW)
    header = session_header("add-auth", board)
    assert timers(header) == [Timer(since=NOW)]
    assert [text.content for text in texts(header)] == ["add-auth"]


def test_a_long_activity_string_is_truncated_rather_than_wrapped() -> None:
    # Elided to a fixed width rather than left whole: unlike a ticket row,
    # this header is sticky, and a wrapped header cannot be undrawn once
    # content appears beneath it.
    board = Board(started_at=NOW)
    long_activity = "x" * (SESSION_ACTIVITY_WIDTH + 20)
    board.activity["add-auth"] = long_activity
    header = session_header("add-auth", board)
    shown = header_activity(header)
    assert shown is not None
    assert shown != long_activity
    assert len(shown) == SESSION_ACTIVITY_WIDTH
    assert shown.endswith("…")
    assert long_activity.startswith(shown[:-1])


def test_the_session_timer_and_the_dashboard_timer_are_independent() -> None:
    """A session that ran 300s before approval: the dashboard starts at zero
    regardless of how long the session took, and the session header still
    remembers the session's own elapsed time. Neither stamp is derived from
    the other.
    """
    scene = one_at(Status.IN_PROGRESS)
    scene.board.started_at = NOW - 300.0
    scene.board.mark(APPROVED, NOW)

    assert lines(draw(scene))[-1] == "0:00"
    header = session_header(scene.label, scene.board)
    assert lines(Screen(content=Rows(header)), NOW)[0].endswith("5:00")


# -- the interview and decompose screens -----------------------------------


def test_the_interview_screen_is_the_header_and_nothing_under_it() -> None:
    screen = session("add-auth", Board(started_at=NOW))
    assert screen.content.children == ()
    assert words(screen.header)[0] == "add-auth"
    assert lines(screen)[0] == "add-auth  0:00"


def test_decompose_falls_back_to_the_bare_header_before_a_proposal_arrives() -> None:
    """What the first decompose call is drawn against: no tickets to show yet."""
    board = Board(started_at=NOW)
    assert decompose("add-auth", board, ()) == session("add-auth", board)


def test_decompose_lists_every_proposed_ticket_under_the_header() -> None:
    board = Board(started_at=NOW)
    board.activity["add-auth"] = "reading spec.md"
    screen = decompose("add-auth", board, (feature("T-01"), feature("T-02")))
    drawn = lines(screen)
    assert drawn[0].startswith("add-auth")
    assert "reading spec.md" in drawn[0]
    assert "T-01: Do T-01 (blocked by: —)" in drawn
    assert "T-02: Do T-02 (blocked by: —)" in drawn


def test_a_proposed_ticket_names_what_blocks_it_and_is_drawn_after_it() -> None:
    """Dependency order, off a graph built from the proposal alone — these
    tickets are not in the run until somebody approves them."""
    screen = decompose(
        "add-auth", Board(started_at=NOW), (feature("T-02", "T-01"), feature("T-01"))
    )
    drawn = [line for line in lines(screen) if line.startswith("T-0")]
    assert drawn == [
        "T-01: Do T-01 (blocked by: —)",
        "T-02: Do T-02 (blocked by: T-01)",
    ]


def test_a_proposal_screen_writes_to_neither_the_board_nor_the_tickets() -> None:
    board = Board(started_at=NOW)
    tickets = (feature("T-02", "T-01"), feature("T-01"))
    before = (dict(board.activity), dict(board.status_since), [vars(t).copy() for t in tickets])

    decompose("add-auth", board, tickets)

    assert (
        dict(board.activity),
        dict(board.status_since),
        [vars(t).copy() for t in tickets],
    ) == before
