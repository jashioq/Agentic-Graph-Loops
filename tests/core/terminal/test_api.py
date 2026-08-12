"""The API surface is immutable data plus two abstract classes."""

import dataclasses

import pytest

from agl.core.terminal import (
    Answer,
    Color,
    LiveSession,
    Option,
    Question,
    Row,
    Rows,
    Screen,
    Spacer,
    Text,
    Timer,
)


def test_colors_cover_the_documented_palette() -> None:
    names = {c.name for c in Color}
    assert names == {
        "WHITE",
        "GREY",
        "RED",
        "GREEN",
        "BLUE",
        "YELLOW",
        "MAGENTA",
        "CYAN",
        "BOLD_YELLOW",
        "BOLD_RED",
        "DIM_GREEN",
        "DIM_GREY",
    }


def test_text_defaults_to_white() -> None:
    assert Text("hello").color is Color.WHITE
    assert Text("hello", Color.RED).color is Color.RED


def test_timer_defaults_to_grey() -> None:
    assert Timer(since=1.0).color is Color.GREY


def test_row_takes_children_variadically() -> None:
    row = Row(Text("a"), Spacer(2), Text("b"))
    assert row.children == (Text("a"), Spacer(2), Text("b"))


def test_rows_takes_children_variadically() -> None:
    rows = Rows(Row(Text("a")), Row(Text("b")))
    assert rows.children == (Row(Text("a")), Row(Text("b")))


def test_empty_row_and_rows_are_allowed() -> None:
    assert Row().children == ()
    assert Rows().children == ()


@pytest.mark.parametrize(
    "value",
    [
        Text("a"),
        Timer(since=0.0),
        Spacer(1),
        Row(Text("a")),
        Rows(Row(Text("a"))),
        Screen(header=None, content=Rows(), footer=None),
        Option("label", "description"),
        Question("header", "title", ()),
        Answer("text", was_free_text=False),
    ],
)
def test_data_types_are_frozen(value: object) -> None:
    assert dataclasses.is_dataclass(value)
    field = next(iter(dataclasses.fields(value)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, field.name, None)


def test_components_compare_by_value() -> None:
    assert Row(Text("a"), Spacer(2)) == Row(Text("a"), Spacer(2))
    assert Row(Text("a")) != Row(Text("b"))


def test_screen_holds_the_three_regions() -> None:
    header = Row(Text("h"))
    content = Rows(Row(Text("c")))
    footer = Rows(Row(Text("f")))
    screen = Screen(header=header, content=content, footer=footer)
    assert (screen.header, screen.content, screen.footer) == (header, content, footer)


def test_question_carries_options() -> None:
    question = Question(
        header="T-04  login screen",
        title="Which storage layer?",
        options=(Option("DataStore", "Survives process death"),),
    )
    assert question.options[0].label == "DataStore"


def test_terminal_and_live_session_are_abstract() -> None:
    from agl.core.terminal import Terminal

    with pytest.raises(TypeError):
        Terminal()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        LiveSession()  # type: ignore[abstract]
