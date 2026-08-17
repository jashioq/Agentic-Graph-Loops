"""Reading typed fields out of JSON that may be wrong: no fixtures, nothing on disk.

Every function takes the callable to hand a failure to, so each test passes its
own `nope` and asserts that its `Nope` is what came out — a caller's error
reaching the caller is half of what this module promises. The other half is that
a value which reads cleanly never touches the callable at all.
"""

from typing import Any, NoReturn

import pytest

from agl.runtime.json_fields import (
    as_object,
    as_optional_text,
    as_text,
    as_text_list,
    as_whole_number,
    reject_unknown_fields,
    require_fields,
    type_name,
)


class Nope(Exception):
    """A caller's own error type, to prove it is the one raised."""


def nope(message: str) -> NoReturn:
    """A caller's own `on_error`, which does what every real caller does: raise."""
    raise Nope(message)


WHERE = "ticket T-01"


# -- objects --------------------------------------------------------------


def test_an_object_comes_back_as_it_was() -> None:
    assert as_object({"id": "T-01"}, WHERE, nope) == {"id": "T-01"}


@pytest.mark.parametrize("value", [[], "T-01", 3, None, True])
def test_anything_that_is_not_an_object_raises_the_caller_s_error(value: Any) -> None:
    with pytest.raises(Nope):
        as_object(value, WHERE, nope)


def test_the_object_message_names_where_and_what_it_got() -> None:
    with pytest.raises(Nope, match=r"ticket T-01 must be an object, got list"):
        as_object([], WHERE, nope)


# -- the field set --------------------------------------------------------


def test_every_required_field_present_is_silence() -> None:
    require_fields({"id": "T-01", "title": "t"}, ["id", "title"], WHERE, nope)


def test_a_missing_required_field_raises_naming_it() -> None:
    with pytest.raises(Nope, match=r"ticket T-01: missing required field 'title'"):
        require_fields({"id": "T-01"}, ["id", "title"], WHERE, nope)


def test_require_fields_ignores_fields_it_was_not_asked_about() -> None:
    require_fields({"id": "T-01", "extra": 1}, ["id"], WHERE, nope)


def test_only_allowed_fields_is_silence() -> None:
    reject_unknown_fields({"id": "T-01"}, ["id", "title"], WHERE, nope)


def test_an_unknown_field_raises_naming_it() -> None:
    with pytest.raises(Nope, match=r"ticket T-01: unknown field 'extra'"):
        reject_unknown_fields({"id": "T-01", "extra": 1}, ["id"], WHERE, nope)


def test_an_allowed_field_that_is_absent_is_not_rejected() -> None:
    reject_unknown_fields({}, ["id", "title"], WHERE, nope)


# -- text -----------------------------------------------------------------


def test_text_comes_back_as_it_was() -> None:
    assert as_text("Add auth", "title", WHERE, nope) == "Add auth"


@pytest.mark.parametrize("value", ["", "   ", "\n", None, 3, ["a"], True])
def test_anything_that_is_not_non_empty_text_raises(value: Any) -> None:
    with pytest.raises(Nope):
        as_text(value, "title", WHERE, nope)


def test_the_text_message_names_the_field_and_shows_the_value() -> None:
    with pytest.raises(Nope, match=r"ticket T-01: title must be non-empty text, got None"):
        as_text(None, "title", WHERE, nope)


def test_optional_text_passes_none_through() -> None:
    assert as_optional_text(None, "detail", WHERE, nope) is None


def test_optional_text_holds_present_text_to_the_same_rule() -> None:
    assert as_optional_text("why", "detail", WHERE, nope) == "why"
    with pytest.raises(Nope):
        as_optional_text("  ", "detail", WHERE, nope)


# -- lists of text --------------------------------------------------------


def test_a_list_of_text_comes_back_as_a_tuple() -> None:
    assert as_text_list(["a", "b"], "deliverables", WHERE, nope) == ("a", "b")


def test_something_that_is_not_an_array_raises_saying_what_it_was() -> None:
    with pytest.raises(Nope, match=r"ticket T-01: deliverables must be an array, got str"):
        as_text_list("a", "deliverables", WHERE, nope)


@pytest.mark.parametrize("entry", ["", "   ", 3, None])
def test_an_entry_that_is_not_non_empty_text_raises(entry: Any) -> None:
    with pytest.raises(Nope, match=r"every deliverables entry must be non-empty text"):
        as_text_list([entry], "deliverables", WHERE, nope)


def test_an_empty_list_is_allowed_by_default() -> None:
    assert as_text_list([], "blocked_by", WHERE, nope) == ()


def test_an_empty_list_raises_when_the_caller_forbids_it() -> None:
    with pytest.raises(Nope, match=r"ticket T-01: deliverables is empty"):
        as_text_list([], "deliverables", WHERE, nope, allow_empty=False)


# -- whole numbers --------------------------------------------------------


def test_a_whole_number_comes_back_as_it_was() -> None:
    assert as_whole_number(3, "review_round", WHERE, nope) == 3


def test_zero_is_a_whole_number() -> None:
    assert as_whole_number(0, "review_round", WHERE, nope) == 0


@pytest.mark.parametrize("value", [-1, 1.5, "3", None, [3]])
def test_anything_that_is_not_a_whole_number_raises(value: Any) -> None:
    with pytest.raises(Nope):
        as_whole_number(value, "review_round", WHERE, nope)


def test_a_bool_is_refused_even_though_it_is_an_int() -> None:
    with pytest.raises(Nope, match=r"review_round must be a whole number, got True"):
        as_whole_number(True, "review_round", WHERE, nope)


# -- what the callable is handed ------------------------------------------


def test_the_callable_is_handed_the_message_and_nothing_else() -> None:
    """What reaches `on_error` is the text, so a caller can wrap it as it likes."""
    seen: list[str] = []

    def collect(message: str) -> NoReturn:
        seen.append(message)
        raise Nope(message)

    with pytest.raises(Nope):
        as_text(None, "title", WHERE, collect)
    assert seen == ["ticket T-01: title must be non-empty text, got None"]


def test_a_value_that_reads_cleanly_never_touches_the_callable() -> None:
    """The other half of the promise: no failure, no call."""
    called = False

    def boom(message: str) -> NoReturn:
        nonlocal called
        called = True
        raise Nope(message)

    as_object({"id": "T-01"}, WHERE, boom)
    require_fields({"id": "T-01"}, ["id"], WHERE, boom)
    reject_unknown_fields({"id": "T-01"}, ["id"], WHERE, boom)
    as_text("Add auth", "title", WHERE, boom)
    as_optional_text(None, "detail", WHERE, boom)
    as_text_list(["a"], "deliverables", WHERE, boom)
    as_whole_number(0, "review_round", WHERE, boom)
    assert not called


# -- naming a type for a person ------------------------------------------


@pytest.mark.parametrize(
    ("value", "name"),
    [("a", "str"), (3, "int"), (None, "NoneType"), ([], "list"), ({}, "dict"), (True, "bool")],
)
def test_type_name_is_what_a_person_would_call_it(value: Any, name: str) -> None:
    assert type_name(value) == name
