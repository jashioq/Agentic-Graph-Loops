"""Rendering is pure: component tree plus a clock reading in, text out.

Snapshots go through Rich's own recorder — there is no second renderer here to
disagree with the real one.
"""

from rich.console import Console
from rich.text import Text as RichText

from agl.core.terminal import Color, Row, Rows, Spacer, Text, Timer
from agl.core.terminal.api import Component
from agl.core.terminal.impl.render import line_count, to_renderable


def render(component: Component, now: float = 100.0) -> str:
    console = Console(width=80, record=True, no_color=True)
    console.print(to_renderable(component, now))
    return console.export_text()


def test_text_renders_its_content() -> None:
    assert render(Text("hello")) == "hello\n"


def test_text_color_does_not_leak_into_plain_output() -> None:
    assert render(Text("hello", Color.RED)) == "hello\n"


def test_spacer_alone_leaves_no_trailing_whitespace() -> None:
    assert render(Spacer(4)) == "\n"


def test_spacer_separates_inside_a_row() -> None:
    assert render(Row(Text("a"), Spacer(3), Text("b"))) == "a   b\n"


def test_row_composes_without_an_implicit_separator() -> None:
    assert render(Row(Text("ab"), Text("cd"))) == "abcd\n"


def test_empty_row_renders_one_blank_line() -> None:
    assert render(Row()) == "\n"


def test_row_strips_a_trailing_spacer() -> None:
    assert render(Row(Text("a"), Spacer(6))) == "a\n"


def test_nested_rows_compose_horizontally() -> None:
    assert render(Row(Row(Text("a"), Spacer(1)), Row(Text("b")))) == "a b\n"


def test_rows_stack_vertically() -> None:
    rows = Rows(Row(Text("first")), Row(Text("second")))
    assert render(rows) == "first\nsecond\n"


def test_rows_strip_trailing_whitespace_on_every_line() -> None:
    rows = Rows(Row(Text("a"), Spacer(9)), Row(Text("bb"), Spacer(2)))
    assert render(rows) == "a\nbb\n"


def test_empty_rows_render_one_blank_line() -> None:
    assert render(Rows()) == "\n"


def test_indented_child_row_keeps_its_leading_spacer() -> None:
    rows = Rows(Row(Text("T-03")), Row(Spacer(6), Text("T-03-bug-1")))
    assert render(rows) == "T-03\n      T-03-bug-1\n"


def test_timer_at_zero() -> None:
    assert render(Timer(since=100.0), now=100.0) == "0:00\n"


def test_timer_under_a_minute() -> None:
    assert render(Timer(since=0.0), now=38.0) == "0:38\n"


def test_timer_pads_seconds_to_two_digits() -> None:
    assert render(Timer(since=0.0), now=242.0) == "4:02\n"


def test_timer_does_not_pad_minutes_under_an_hour() -> None:
    assert render(Timer(since=0.0), now=767.0) == "12:47\n"


def test_timer_switches_to_hours_at_one_hour() -> None:
    assert render(Timer(since=0.0), now=3600.0) == "1:00:00\n"


def test_timer_in_hours() -> None:
    assert render(Timer(since=0.0), now=3735.0) == "1:02:15\n"


def test_timer_truncates_fractional_seconds() -> None:
    assert render(Timer(since=0.0), now=38.9) == "0:38\n"


def test_timer_clamps_negative_elapsed_to_zero() -> None:
    assert render(Timer(since=200.0), now=100.0) == "0:00\n"


def test_every_color_maps_to_a_distinct_style() -> None:
    styles = {}
    for color in Color:
        rendered = to_renderable(Text("x", color), now=0.0)
        assert isinstance(rendered, RichText)
        assert rendered.style, f"{color.name} has no style"
        styles[color] = str(rendered.style)
    assert len(set(styles.values())) == len(Color)


def test_timer_carries_its_color_as_a_style() -> None:
    rendered = to_renderable(Timer(since=0.0, color=Color.BOLD_RED), now=0.0)
    assert isinstance(rendered, RichText)
    assert str(rendered.style) == str(to_renderable(Text("x", Color.BOLD_RED), now=0.0).style)


def test_rendering_is_pure_and_repeatable() -> None:
    row = Row(Text("a"), Spacer(2), Timer(since=0.0))
    assert render(row, now=61.0) == render(row, now=61.0) == "a  1:01\n"


def test_line_count_of_a_single_row_is_one() -> None:
    assert line_count(Row(Text("a"), Spacer(2), Text("b")), now=0.0) == 1


def test_line_count_of_rows_is_the_number_of_rows() -> None:
    rows = Rows(Row(Text("a")), Row(Text("b")), Row(Text("c")))
    assert line_count(rows, now=0.0) == 3


def test_line_count_of_empty_rows_is_one() -> None:
    assert line_count(Rows(), now=0.0) == 1
