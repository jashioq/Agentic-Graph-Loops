"""Conflict marker parsing, as plain strings — no git, no filesystem."""

from agl.core.vcs.api import ConflictHunk
from agl.core.vcs.impl.conflicts import parse_conflicts

TWO_WAY = """\
before
<<<<<<< HEAD
ours
=======
theirs
>>>>>>> feature
after
"""

THREE_WAY = """\
before
<<<<<<< HEAD
ours
||||||| merged common ancestors
base
=======
theirs
>>>>>>> feature
after
"""


# -- two-way markers ------------------------------------------------------


def test_a_two_way_conflict_yields_one_hunk() -> None:
    assert parse_conflicts(TWO_WAY) == (
        ConflictHunk(ours=("ours",), theirs=("theirs",), base=None),
    )


def test_a_two_way_hunk_has_no_base() -> None:
    assert parse_conflicts(TWO_WAY)[0].base is None


def test_multi_line_sides_keep_their_order() -> None:
    content = "<<<<<<< HEAD\na\nb\n=======\nx\ny\nz\n>>>>>>> feature\n"
    hunk = parse_conflicts(content)[0]
    assert hunk.ours == ("a", "b")
    assert hunk.theirs == ("x", "y", "z")


def test_text_outside_the_markers_is_not_part_of_any_hunk() -> None:
    hunk = parse_conflicts(TWO_WAY)[0]
    assert "before" not in hunk.ours + hunk.theirs
    assert "after" not in hunk.ours + hunk.theirs


# -- three-way markers ----------------------------------------------------


def test_a_three_way_conflict_carries_the_base() -> None:
    assert parse_conflicts(THREE_WAY) == (
        ConflictHunk(ours=("ours",), theirs=("theirs",), base=("base",)),
    )


def test_a_zdiff3_style_conflict_parses_the_same_way() -> None:
    # zdiff3 differs only in how much common text it hoists out of the hunk.
    content = "<<<<<<< HEAD\nours\n||||||| abc1234\noriginal\n=======\ntheirs\n>>>>>>> x\n"
    assert parse_conflicts(content)[0].base == ("original",)


def test_an_empty_base_section_is_an_empty_tuple_not_none() -> None:
    # The base was empty, which is a different fact from there being no base.
    content = "<<<<<<< HEAD\nours\n||||||| abc1234\n=======\ntheirs\n>>>>>>> x\n"
    assert parse_conflicts(content)[0].base == ()


def test_a_multi_line_base_keeps_its_order() -> None:
    content = "<<<<<<< HEAD\no\n||||||| abc1234\nb1\nb2\n=======\nt\n>>>>>>> x\n"
    assert parse_conflicts(content)[0].base == ("b1", "b2")


# -- several hunks --------------------------------------------------------


def test_two_conflict_regions_yield_two_hunks() -> None:
    content = (
        "top\n"
        "<<<<<<< HEAD\nfirst ours\n=======\nfirst theirs\n>>>>>>> x\n"
        "middle\n"
        "<<<<<<< HEAD\nsecond ours\n=======\nsecond theirs\n>>>>>>> x\n"
        "bottom\n"
    )
    hunks = parse_conflicts(content)
    assert len(hunks) == 2
    assert hunks[0].ours == ("first ours",)
    assert hunks[1].ours == ("second ours",)


def test_hunks_come_back_in_file_order() -> None:
    content = "".join(
        f"<<<<<<< HEAD\n{n}\n=======\nt{n}\n>>>>>>> x\n" for n in ("a", "b", "c")
    )
    assert [hunk.ours[0] for hunk in parse_conflicts(content)] == ["a", "b", "c"]


def test_two_way_and_three_way_hunks_can_share_a_file() -> None:
    content = (
        "<<<<<<< HEAD\na\n=======\nb\n>>>>>>> x\n"
        "<<<<<<< HEAD\nc\n||||||| abc\nbase\n=======\nd\n>>>>>>> x\n"
    )
    hunks = parse_conflicts(content)
    assert hunks[0].base is None
    assert hunks[1].base == ("base",)


# -- empty sides ----------------------------------------------------------


def test_an_empty_ours_side_is_an_empty_tuple() -> None:
    content = "<<<<<<< HEAD\n=======\ntheirs\n>>>>>>> x\n"
    assert parse_conflicts(content) == (ConflictHunk(ours=(), theirs=("theirs",), base=None),)


def test_an_empty_theirs_side_is_an_empty_tuple() -> None:
    content = "<<<<<<< HEAD\nours\n=======\n>>>>>>> x\n"
    assert parse_conflicts(content) == (ConflictHunk(ours=("ours",), theirs=(), base=None),)


def test_both_sides_empty_is_still_a_hunk() -> None:
    content = "<<<<<<< HEAD\n=======\n>>>>>>> x\n"
    assert parse_conflicts(content) == (ConflictHunk(ours=(), theirs=(), base=None),)


def test_a_blank_line_inside_a_side_is_kept() -> None:
    content = "<<<<<<< HEAD\na\n\nb\n=======\nt\n>>>>>>> x\n"
    assert parse_conflicts(content)[0].ours == ("a", "", "b")


# -- content that is not a marker -----------------------------------------


def test_a_file_with_no_markers_has_no_hunks() -> None:
    assert parse_conflicts("just\nsome\nlines\n") == ()


def test_an_empty_file_has_no_hunks() -> None:
    assert parse_conflicts("") == ()


def test_a_line_merely_starting_with_an_angle_bracket_is_content() -> None:
    content = "<<<<<<< HEAD\n<div>\n<not a marker\n=======\n<p>\n>>>>>>> x\n"
    hunk = parse_conflicts(content)[0]
    assert hunk.ours == ("<div>", "<not a marker")
    assert hunk.theirs == ("<p>",)


def test_angle_brackets_outside_a_conflict_are_ignored() -> None:
    assert parse_conflicts("<html>\n<<< not a marker\n>>> nor this\n") == ()


def test_a_short_run_of_angle_brackets_is_not_a_marker() -> None:
    # Git's markers are exactly seven characters.
    assert parse_conflicts("<<<<<< HEAD\nours\n======\ntheirs\n>>>>>> x\n") == ()


def test_a_markdown_underline_outside_a_conflict_is_not_a_separator() -> None:
    assert parse_conflicts("Heading\n=======\ntext\n") == ()


# -- shapes the parser must survive ---------------------------------------


def test_an_unterminated_conflict_is_ignored() -> None:
    # A truncated file should not produce a half-parsed hunk.
    assert parse_conflicts("<<<<<<< HEAD\nours\n=======\ntheirs\n") == ()


def test_a_start_marker_with_no_separator_is_ignored() -> None:
    assert parse_conflicts("<<<<<<< HEAD\nours\n>>>>>>> x\n") == ()


def test_a_stray_end_marker_is_ignored() -> None:
    assert parse_conflicts("text\n>>>>>>> x\nmore\n") == ()


def test_a_complete_hunk_after_a_broken_one_is_still_found() -> None:
    content = "<<<<<<< HEAD\ndangling\n<<<<<<< HEAD\na\n=======\nb\n>>>>>>> x\n"
    assert parse_conflicts(content) == (ConflictHunk(ours=("a",), theirs=("b",), base=None),)


def test_a_file_not_ending_in_a_newline_still_parses() -> None:
    content = "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> x"
    assert parse_conflicts(content) == (
        ConflictHunk(ours=("ours",), theirs=("theirs",), base=None),
    )


def test_crlf_line_endings_are_recognised_as_markers() -> None:
    content = "<<<<<<< HEAD\r\nours\r\n=======\r\ntheirs\r\n>>>>>>> x\r\n"
    hunks = parse_conflicts(content)
    assert len(hunks) == 1
    assert [line.rstrip("\r") for line in hunks[0].ours] == ["ours"]


def test_markers_longer_than_seven_characters_are_still_markers() -> None:
    # Git pads markers out when the conflicting content contains marker-like lines.
    content = "<<<<<<<< HEAD\nours\n========\ntheirs\n>>>>>>>> x\n"
    assert parse_conflicts(content) == (
        ConflictHunk(ours=("ours",), theirs=("theirs",), base=None),
    )


def test_parsing_does_not_mutate_or_lose_leading_whitespace() -> None:
    content = "<<<<<<< HEAD\n    indented\n=======\n\tt\n>>>>>>> x\n"
    hunk = parse_conflicts(content)[0]
    assert hunk.ours == ("    indented",)
    assert hunk.theirs == ("\tt",)
