"""The failure path: two tickets that touched the same lines, and what comes back.

Both branches are cut from the same base and both rewrite the same region, so
the first merge lands and the second cannot. What matters here is that the
second one comes back as *data* — a `Conflict` naming the file with both sides
of every hunk — and that the merge is still in progress afterwards, because the
caller is the one who decides between resolving it and throwing it away.

Two endings are exercised, each on a repository of its own: resolve and commit,
or abort and leave the base exactly where it was. The tests that only read the
conflict share one repository for the module, since none of them touch it.

The last few tests are about whether a classifier could ever be written over
this data. `additive` is not a classifier — it is a two-line predicate over one
hunk, there to show that the parsed sides carry enough to separate two imports
landing in the same place from two contradictory rewrites of one line.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from agl.core import paths
from agl.core.vcs import Conflict, ConflictHunk, MergeResult
from agl.core.vcs.impl.git import Git
from tests.conftest import commit_file, git
from tests.integration.conftest import PROJECT, copy_repo

LABEL = "add-auth"
FILE = "auth.py"

BASE = 'MODE = "off"\n'
OURS = 'MODE = "strict"\n'
THEIRS = 'MODE = "lenient"\n'

TAIL = "\n\ndef run() -> None:\n    pass\n"
IMPORTS_BASE = f"import os{TAIL}"
IMPORTS_OURS = f"import os\nimport json{TAIL}"
IMPORTS_THEIRS = f"import os\nimport sys{TAIL}"


@dataclass(frozen=True)
class Collision:
    """Two ticket branches over one file: the one that landed and the one that did not."""

    repo: Path
    vcs: Git
    first: str
    second: str
    base_before_second: str
    clean: MergeResult
    conflicted: MergeResult


def collide(root: Path, template: Path, base: str, ours: str, theirs: str) -> Collision:
    """Two tickets rewrite one file from the same base; merge both into it.

    Each ticket works in its own worktree off `main`, which is the shape the
    workflow will have — not two checkouts in the one repository.
    """
    repo = copy_repo(root, template)
    vcs = Git(repo)
    commit_file(repo, FILE, base, f"add {FILE}")

    branches = []
    for ticket, content in (("T-01", ours), ("T-02", theirs)):
        branch = paths.ticket_branch(LABEL, ticket)
        tree = vcs.add_worktree(
            paths.worktree_dir(root / "trees", PROJECT, LABEL, ticket), branch, "main"
        )
        (tree.path / FILE).write_text(content, encoding="utf-8")
        vcs.commit_all(tree.path, f"{ticket}: rewrite {FILE}")
        vcs.remove_worktree(tree.path)
        branches.append(branch)

    first, second = branches
    clean = vcs.merge(repo, first)
    base_before_second = vcs.rev_parse("main")
    return Collision(
        repo=repo,
        vcs=vcs,
        first=first,
        second=second,
        base_before_second=base_before_second,
        clean=clean,
        conflicted=vcs.merge(repo, second),
    )


@pytest.fixture(scope="module")
def collision(tmp_path_factory: pytest.TempPathFactory, _template_repo: Path) -> Collision:
    """Two contradictory rewrites of the same assignment. Read only."""
    return collide(tmp_path_factory.mktemp("collision"), _template_repo, BASE, OURS, THEIRS)


@pytest.fixture(scope="module")
def imports(tmp_path_factory: pytest.TempPathFactory, _template_repo: Path) -> Collision:
    """Two imports added in the same place — the kind a classifier would wave through."""
    return collide(
        tmp_path_factory.mktemp("imports"),
        _template_repo,
        IMPORTS_BASE,
        IMPORTS_OURS,
        IMPORTS_THEIRS,
    )


@pytest.fixture
def unresolved(tmp_path_factory: pytest.TempPathFactory, _template_repo: Path) -> Collision:
    """A conflict of the test's own, for the endings that finish it one way or the other."""
    return collide(tmp_path_factory.mktemp("unresolved"), _template_repo, BASE, OURS, THEIRS)


def only_hunk(conflicts: tuple[Conflict, ...]) -> ConflictHunk:
    """The single hunk of the single conflicted file."""
    assert len(conflicts) == 1, conflicts
    assert len(conflicts[0].hunks) == 1, conflicts[0].hunks
    return conflicts[0].hunks[0]


# -- what the two merges reported -----------------------------------------


def test_the_first_merge_landed(collision: Collision) -> None:
    assert collision.clean.clean is True
    assert collision.clean.conflicts == ()
    assert collision.vcs.is_ancestor(collision.first, "main") is True


def test_the_second_merge_came_back_conflicted(collision: Collision) -> None:
    assert collision.conflicted.clean is False
    assert collision.conflicted.sha is None
    assert collision.conflicted.conflicts != ()


def test_the_merge_was_left_in_progress(collision: Collision) -> None:
    # On purpose: there is no third state where the merge quietly vanished.
    assert collision.vcs.merge_in_progress(collision.repo) is True
    assert collision.vcs.unmerged_paths(collision.repo) == (FILE,)
    assert collision.vcs.is_dirty(collision.repo) is True


def test_the_conflict_names_the_file(collision: Collision) -> None:
    assert [conflict.path for conflict in collision.conflicted.conflicts] == [FILE]
    assert collision.vcs.conflicts(collision.repo) == collision.conflicted.conflicts


def test_the_hunk_carries_both_sides(collision: Collision) -> None:
    hunk = only_hunk(collision.conflicted.conflicts)
    assert hunk.ours == ('MODE = "strict"',)
    assert hunk.theirs == ('MODE = "lenient"',)


def test_the_base_side_is_absent_under_this_conflict_style(collision: Collision) -> None:
    # `merge.conflictStyle` is the repository's to set, so anything reading a
    # hunk has to cope with `base` being `None`.
    assert only_hunk(collision.conflicted.conflicts).base is None


# -- ending one: resolve it -------------------------------------------------


def test_a_hand_resolved_merge_lands(unresolved: Collision) -> None:
    vcs, repo = unresolved.vcs, unresolved.repo
    (repo / FILE).write_text('MODE = "strict"\n', encoding="utf-8")
    # `Vcs` has no way to stage a path, so the resolution is staged with git
    # itself; see the report.
    git(repo, "add", "--", FILE)

    sha = vcs.commit_merge(repo, f"merge {unresolved.second}")

    assert sha == vcs.rev_parse("main")
    assert vcs.merge_in_progress(repo) is False
    assert vcs.is_dirty(repo) is False
    assert (repo / FILE).read_text(encoding="utf-8") == 'MODE = "strict"\n'


def test_the_resolved_merge_has_both_branches_behind_it(unresolved: Collision) -> None:
    vcs, repo = unresolved.vcs, unresolved.repo
    (repo / FILE).write_text(OURS, encoding="utf-8")
    git(repo, "add", "--", FILE)
    vcs.commit_merge(repo, "merge")

    assert vcs.is_ancestor(unresolved.first, "main") is True
    assert vcs.is_ancestor(unresolved.second, "main") is True


# -- ending two: throw it away ---------------------------------------------


def test_an_aborted_merge_leaves_the_base_untouched(unresolved: Collision) -> None:
    vcs, repo = unresolved.vcs, unresolved.repo
    vcs.abort_merge(repo)

    assert vcs.rev_parse("main") == unresolved.base_before_second
    assert vcs.merge_in_progress(repo) is False
    assert vcs.unmerged_paths(repo) == ()
    assert vcs.is_dirty(repo) is False
    assert (repo / FILE).read_text(encoding="utf-8") == OURS
    assert vcs.is_ancestor(unresolved.second, "main") is False
    assert vcs.branch_exists(unresolved.second) is True


def test_an_aborted_merge_can_be_tried_again(unresolved: Collision) -> None:
    # Nothing was lost by aborting: the same merge comes back the same way.
    unresolved.vcs.abort_merge(unresolved.repo)
    assert unresolved.vcs.merge(unresolved.repo, unresolved.second).clean is False


# -- is the data enough for a classifier? ---------------------------------


def additive(hunk: ConflictHunk) -> bool:
    """Whether both sides only add imports, so their union is the resolution."""
    lines = [line.strip() for line in (*hunk.ours, *hunk.theirs)]
    return bool(lines) and all(line.startswith("import ") for line in lines)


def test_two_imports_in_the_same_place_still_conflict(imports: Collision) -> None:
    assert imports.conflicted.clean is False
    assert [conflict.path for conflict in imports.conflicted.conflicts] == [FILE]


def test_the_import_hunk_carries_each_side_s_added_line(imports: Collision) -> None:
    hunk = only_hunk(imports.conflicted.conflicts)
    assert hunk.ours == ("import json",)
    assert hunk.theirs == ("import sys",)


def test_the_import_conflict_reads_as_additive(imports: Collision) -> None:
    assert additive(only_hunk(imports.conflicted.conflicts)) is True


def test_the_rewrite_conflict_does_not(collision: Collision) -> None:
    # The same predicate over the same shape of data separates the two, which is
    # all a classifier would need from `vcs`.
    assert additive(only_hunk(collision.conflicted.conflicts)) is False
