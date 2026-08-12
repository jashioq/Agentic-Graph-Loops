"""Three sticky regions: header on top, footer on the bottom, content between."""

from rich.console import Console

from agl.core.terminal import Row, Rows, Screen, Spacer, Text, Timer
from agl.core.terminal.impl.screen import to_layout

HEIGHT = 20


def render_screen(screen: Screen, now: float = 100.0, height: int = HEIGHT) -> list[str]:
    console = Console(width=80, height=height, record=True, no_color=True)
    console.print(to_layout(screen, now))
    return [line.rstrip() for line in console.export_text().splitlines()]


def screen_with(
    header: Row | Rows | None = None,
    content: Rows | None = None,
    footer: Row | Rows | None = None,
) -> Screen:
    return Screen(header=header, content=content if content is not None else Rows(), footer=footer)


def test_layout_fills_the_console_height() -> None:
    lines = render_screen(screen_with(header=Row(Text("h")), footer=Row(Text("f"))))
    assert len(lines) == HEIGHT


def test_header_is_pinned_to_the_first_line() -> None:
    lines = render_screen(screen_with(header=Row(Text("add-auth-ticket-18732"))))
    assert lines[0] == "add-auth-ticket-18732"


def test_footer_is_pinned_to_the_last_line() -> None:
    lines = render_screen(screen_with(footer=Row(Text("running"), Spacer(2), Timer(since=100.0))))
    assert lines[-1] == "running  0:00"


def test_content_sits_between_header_and_footer() -> None:
    lines = render_screen(
        screen_with(
            header=Row(Text("HEADER")),
            content=Rows(Row(Text("first")), Row(Text("second"))),
            footer=Row(Text("FOOTER")),
        )
    )
    assert lines[0] == "HEADER"
    assert lines[1] == "first"
    assert lines[2] == "second"
    assert lines[-1] == "FOOTER"


def test_content_starts_at_the_top_when_there_is_no_header() -> None:
    lines = render_screen(screen_with(content=Rows(Row(Text("first")))))
    assert lines[0] == "first"


def test_multi_line_header_reserves_its_own_height() -> None:
    header = Rows(Row(Text("line one")), Row(Text("line two")))
    lines = render_screen(screen_with(header=header, content=Rows(Row(Text("content")))))
    assert lines[:3] == ["line one", "line two", "content"]


def test_multi_line_footer_reserves_its_own_height() -> None:
    footer = Rows(Row(Text("foot one")), Row(Text("foot two")))
    lines = render_screen(screen_with(footer=footer))
    assert lines[-2:] == ["foot one", "foot two"]


def test_content_fills_the_gap_between_the_regions() -> None:
    lines = render_screen(
        screen_with(header=Row(Text("H")), content=Rows(Row(Text("c"))), footer=Row(Text("F")))
    )
    assert lines[2:-1] == [""] * (HEIGHT - 3)


def test_content_taller_than_the_region_is_cropped_not_scrolled() -> None:
    content = Rows(*(Row(Text(f"row-{index:02d}")) for index in range(40)))
    lines = render_screen(
        screen_with(header=Row(Text("H")), content=content, footer=Row(Text("F")))
    )
    assert len(lines) == HEIGHT
    assert lines[1] == "row-00"
    assert lines[-2] == f"row-{HEIGHT - 3:02d}"
    assert lines[-1] == "F"


def test_empty_screen_renders_blank_lines() -> None:
    assert render_screen(screen_with()) == [""] * HEIGHT


def test_layout_is_a_pure_function_of_the_screen_and_time() -> None:
    screen = screen_with(content=Rows(Row(Timer(since=0.0))))
    assert render_screen(screen, now=61.0) == render_screen(screen, now=61.0)
    assert render_screen(screen, now=61.0)[0] == "1:01"
