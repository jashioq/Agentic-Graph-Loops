"""Conflict markers, parsed. Pure functions over text — no git, no I/O.

Layer: core. `git.py` reads the conflicted file and passes its contents in, so
the fiddliest logic in the module can be exercised with plain strings.

Both marker styles are handled, because which one a repository writes is a
config setting (`merge.conflictStyle`) that a caller does not control::

    <<<<<<< HEAD          <<<<<<< HEAD
    ours                  ours
    =======               ||||||| base
    theirs                base content
    >>>>>>> branch        =======
                          theirs
                          >>>>>>> branch

Two-way leaves `base` as `None`; `diff3` and `zdiff3` populate it — and an
empty `base` tuple means the base section was there and empty, which is a
different fact from there being no base section at all.

Lines come back without their terminator, and otherwise exactly as they were:
indentation, blank lines, and any stray `\\r` are the caller's to deal with.
A region that never closes is dropped rather than half-reported, so a truncated
file cannot turn into a hunk that claims text it does not have.

**A marker is exactly seven characters**, and matched as such. Git writes seven
and puts any label after a single space, so an eighth character means the line
is content: a reStructuredText heading underline, an ASCII divider, a row of
`=` in a comment. Matching on a prefix, one line of `========` inside the
"ours" side flips the parser to "theirs" and mis-attributes the rest of the
hunk — and that parse is what a conflict-classifying caller acts on.
"""

from agl.core.vcs.api import ConflictHunk

__all__ = ["parse_conflicts"]

START = "<<<<<<<"
BASE = "|||||||"
SEPARATOR = "======="
END = ">>>>>>>"


def parse_conflicts(content: str) -> tuple[ConflictHunk, ...]:
    """Every complete conflict region in `content`, in the order they appear."""
    hunks: list[ConflictHunk] = []
    ours: list[str] = []
    theirs: list[str] = []
    base: list[str] | None = None
    section = ""  # "" outside a conflict, else "ours", "base", or "theirs"

    for line in content.split("\n"):
        marker = line.rstrip("\r")
        if _labeled(marker, START):
            # A second start inside a region means the first never closed.
            ours, theirs, base, section = [], [], None, "ours"
        elif not section:
            continue
        elif _labeled(marker, BASE):
            base, section = [], "base"
        elif marker == SEPARATOR:
            # The separator never carries a label under any conflict style, so
            # the whole line has to be the seven characters and nothing else.
            section = "theirs"
        elif _labeled(marker, END):
            # Only a region that reached its separator has two sides to report;
            # anything else is a marker left behind by hand-editing.
            if section == "theirs":
                hunks.append(
                    ConflictHunk(
                        ours=tuple(ours),
                        theirs=tuple(theirs),
                        base=None if base is None else tuple(base),
                    )
                )
            section = ""
        elif section == "ours":
            ours.append(line)
        elif section == "base" and base is not None:
            base.append(line)
        else:
            theirs.append(line)

    return tuple(hunks)


def _labeled(line: str, marker: str) -> bool:
    """Whether `line` is exactly `marker`, optionally followed by a label.

    Git separates a marker from its label with one space, so seven characters
    then end-of-line or a space is a marker and anything else is content — an
    eighth `<` included.
    """
    return line.startswith(marker) and line[len(marker) : len(marker) + 1] in ("", " ")
