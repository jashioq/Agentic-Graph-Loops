"""The failure path: two tickets that touched the same lines, and what comes back.

Both branches are cut from the same base and both rewrite the same region, so
the first merge lands and the second cannot. What matters here is that the
second one comes back as *data* — `clean=False` with the conflicting paths
named — and that the merge is still in progress afterwards, because the caller
is the one who decides between resolving it and throwing it away.

There is no conflict classifier and none is coming. Every conflict halts the run
for a person, including the ones that look easy: the additive case below is two
imports landing in the same place, and it conflicts and is reported exactly like
the contradictory rewrite beside it. Deciding "this one is trivial" from marker
text alone would need language-specific rules, and this tool drives Kotlin, Rust
and React alike.

Two endings are exercised, each on a repository of its own: resolve and commit,
or abort and leave the base exactly where it was. The tests that only read the
result share one repository for the module, since none of them touch it.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from agl.core import paths
from agl.core.vcs import MergeResult
from agl.core.vcs.impl.git import Git
from tests.conftest import commit_file
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
        branch = paths.branch(LABEL, ticket)
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
    """Two imports added in the same place — the easy-looking conflict. Read only."""
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


# -- what the two merges reported -----------------------------------------


def test_the_first_merge_landed(collision: Collision) -> None:
    assert collision.clean.clean is True
    assert collision.clean.conflicted == ()
    assert collision.vcs.is_ancestor(collision.first, "main") is True


def test_the_second_merge_came_back_conflicted(collision: Collision) -> None:
    assert collision.conflicted.clean is False
    assert collision.conflicted.sha is None


def test_the_result_names_the_file_a_person_has_to_open(collision: Collision) -> None:
    # A halt banner is written from this alone, so the paths travel with the
    # result rather than needing a second call to go and fetch.
    assert collision.conflicted.conflicted == (FILE,)


def test_the_merge_was_left_in_progress(collision: Collision) -> None:
    # On purpose: there is no third state where the merge quietly vanished.
    assert collision.vcs.merge_in_progress(collision.repo) is True
    assert collision.vcs.unmerged_paths(collision.repo) == (FILE,)
    assert collision.vcs.is_dirty(collision.repo) is True


def test_both_sides_are_left_in_the_file_for_whoever_resolves_it(
    collision: Collision,
) -> None:
    content = (collision.repo / FILE).read_text(encoding="utf-8")
    assert "<<<<<<<" in content
    assert OURS.strip() in content and THEIRS.strip() in content


# -- the easy-looking one halts too ---------------------------------------


def test_two_imports_in_the_same_place_conflict_like_anything_else(
    imports: Collision,
) -> None:
    # No classifier waves this through. It is reported exactly as the
    # contradictory rewrite is, and a person resolves it.
    assert imports.conflicted.clean is False
    assert imports.conflicted.sha is None
    assert imports.conflicted.conflicted == (FILE,)
    assert imports.vcs.merge_in_progress(imports.repo) is True


# -- ending one: resolve it -------------------------------------------------


def test_a_hand_resolved_merge_lands(unresolved: Collision) -> None:
    vcs, repo = unresolved.vcs, unresolved.repo
    # Writing the file is the whole resolution: `commit_merge` stages what the
    # merge left unmerged, so the scenario never leaves the `Vcs` interface.
    (repo / FILE).write_text('MODE = "strict"\n', encoding="utf-8")

    sha = vcs.commit_merge(repo, f"merge {unresolved.second}")

    assert sha == vcs.rev_parse("main")
    assert vcs.merge_in_progress(repo) is False
    assert vcs.is_dirty(repo) is False
    assert (repo / FILE).read_text(encoding="utf-8") == 'MODE = "strict"\n'


def test_the_resolved_merge_has_both_branches_behind_it(unresolved: Collision) -> None:
    vcs, repo = unresolved.vcs, unresolved.repo
    (repo / FILE).write_text(OURS, encoding="utf-8")
    vcs.commit_merge(repo, "merge")

    assert vcs.is_ancestor(unresolved.first, "main") is True
    assert vcs.is_ancestor(unresolved.second, "main") is True


def test_the_named_paths_are_the_ones_resolving_them_clears(unresolved: Collision) -> None:
    # What the result named is what has to be dealt with: deal with exactly
    # those and nothing is left unmerged.
    vcs, repo = unresolved.vcs, unresolved.repo
    for path in unresolved.conflicted.conflicted:
        (repo / path).write_text(OURS, encoding="utf-8")
    vcs.commit_merge(repo, "merge")

    assert vcs.unmerged_paths(repo) == ()


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
    retried = unresolved.vcs.merge(unresolved.repo, unresolved.second)
    assert retried.clean is False
    assert retried.conflicted == unresolved.conflicted.conflicted
