"""Reading typed fields out of JSON that may be wrong: no fixtures, nothing on disk.

Every function raises `InvalidFieldError` and takes nothing to say so with, the
way `paths` raises `InvalidNameError`. What each test pins is the message, since
that is the half a caller passes on to whoever has to fix the document.
"""

from typing import Any

import pytest

from agl.runtime.json_fields import (
    InvalidFieldError,
    as_object,
    as_optional_text,
    as_text,
    as_text_list,
    as_whole_number,
    reject_unknown_fields,
    require_fields,
    type_name,
)

WHERE = "ticket T-01"


# -- objects --------------------------------------------------------------


def test_an_object_comes_back_as_it_was() -> None:
    assert as_object({"id": "T-01"}, WHERE) == {"id": "T-01"}


@pytest.mark.parametrize("value", [[], "T-01", 3, None, True])
def test_anything_that_is_not_an_object_raises(value: Any) -> None:
    with pytest.raises(InvalidFieldError):
        as_object(value, WHERE)


def test_the_object_message_names_where_and_what_it_got() -> None:
    with pytest.raises(InvalidFieldError, match=r"ticket T-01 must be an object, got list"):
        as_object([], WHERE)


# -- the field set --------------------------------------------------------


def test_every_required_field_present_is_silence() -> None:
    require_fields({"id": "T-01", "title": "t"}, ["id", "title"], WHERE)


def test_a_missing_required_field_raises_naming_it() -> None:
    with pytest.raises(InvalidFieldError, match=r"ticket T-01: missing required field 'title'"):
        require_fields({"id": "T-01"}, ["id", "title"], WHERE)


def test_require_fields_ignores_fields_it_was_not_asked_about() -> None:
    require_fields({"id": "T-01", "extra": 1}, ["id"], WHERE)


def test_only_allowed_fields_is_silence() -> None:
    reject_unknown_fields({"id": "T-01"}, ["id", "title"], WHERE)


def test_an_unknown_field_raises_naming_it() -> None:
    with pytest.raises(InvalidFieldError, match=r"ticket T-01: unknown field 'extra'"):
        reject_unknown_fields({"id": "T-01", "extra": 1}, ["id"], WHERE)


def test_an_allowed_field_that_is_absent_is_not_rejected() -> None:
    reject_unknown_fields({}, ["id", "title"], WHERE)


# -- text -----------------------------------------------------------------


def test_text_comes_back_as_it_was() -> None:
    assert as_text("Add auth", "title", WHERE) == "Add auth"


@pytest.mark.parametrize("value", ["", "   ", "\n", None, 3, ["a"], True])
def test_anything_that_is_not_non_empty_text_raises(value: Any) -> None:
    with pytest.raises(InvalidFieldError):
        as_text(value, "title", WHERE)


def test_the_text_message_names_the_field_and_shows_the_value() -> None:
    with pytest.raises(
        InvalidFieldError, match=r"ticket T-01: title must be non-empty text, got None"
    ):
        as_text(None, "title", WHERE)


def test_optional_text_passes_none_through() -> None:
    assert as_optional_text(None, "detail", WHERE) is None


def test_optional_text_holds_present_text_to_the_same_rule() -> None:
    assert as_optional_text("why", "detail", WHERE) == "why"
    with pytest.raises(InvalidFieldError):
        as_optional_text("  ", "detail", WHERE)


# -- lists of text --------------------------------------------------------


def test_a_list_of_text_comes_back_as_a_tuple() -> None:
    assert as_text_list(["a", "b"], "deliverables", WHERE) == ("a", "b")


def test_something_that_is_not_an_array_raises_saying_what_it_was() -> None:
    with pytest.raises(
        InvalidFieldError, match=r"ticket T-01: deliverables must be an array, got str"
    ):
        as_text_list("a", "deliverables", WHERE)


@pytest.mark.parametrize("entry", ["", "   ", 3, None])
def test_an_entry_that_is_not_non_empty_text_raises(entry: Any) -> None:
    with pytest.raises(InvalidFieldError, match=r"every deliverables entry must be non-empty text"):
        as_text_list([entry], "deliverables", WHERE)


def test_an_empty_list_is_allowed_by_default() -> None:
    assert as_text_list([], "blocked_by", WHERE) == ()


def test_an_empty_list_raises_when_the_caller_forbids_it() -> None:
    with pytest.raises(InvalidFieldError, match=r"ticket T-01: deliverables is empty"):
        as_text_list([], "deliverables", WHERE, allow_empty=False)


# -- whole numbers --------------------------------------------------------


def test_a_whole_number_comes_back_as_it_was() -> None:
    assert as_whole_number(3, "review_round", WHERE) == 3


def test_zero_is_a_whole_number() -> None:
    assert as_whole_number(0, "review_round", WHERE) == 0


@pytest.mark.parametrize("value", [-1, 1.5, "3", None, [3]])
def test_anything_that_is_not_a_whole_number_raises(value: Any) -> None:
    with pytest.raises(InvalidFieldError):
        as_whole_number(value, "review_round", WHERE)


def test_a_bool_is_refused_even_though_it_is_an_int() -> None:
    with pytest.raises(InvalidFieldError, match=r"review_round must be a whole number, got True"):
        as_whole_number(True, "review_round", WHERE)


# -- what a caller sees ---------------------------------------------------


def test_a_caller_renames_the_error_by_catching_it_once() -> None:
    """The pattern every document module uses: catch, re-raise as its own, keep the message."""

    class Nope(Exception):
        pass

    try:
        as_text(None, "title", WHERE)
    except InvalidFieldError as invalid:
        renamed = Nope(str(invalid))
    assert str(renamed) == "ticket T-01: title must be non-empty text, got None"


# -- naming a type for a person ------------------------------------------


@pytest.mark.parametrize(
    ("value", "name"),
    [("a", "str"), (3, "int"), (None, "NoneType"), ([], "list"), ({}, "dict"), (True, "bool")],
)
def test_type_name_is_what_a_person_would_call_it(value: Any, name: str) -> None:
    assert type_name(value) == name
