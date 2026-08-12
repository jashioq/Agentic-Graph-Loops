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
        if marker.startswith(START):
            # A second start inside a region means the first never closed.
            ours, theirs, base, section = [], [], None, "ours"
        elif not section:
            continue
        elif marker.startswith(BASE):
            base, section = [], "base"
        elif marker.startswith(SEPARATOR):
            section = "theirs"
        elif marker.startswith(END):
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
