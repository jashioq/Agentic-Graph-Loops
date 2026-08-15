"""Work discovered mid-run that blocks the work which discovered it.

The trickiest thing in the design. `T-01` is claimed and running when its review
finds two problems; the two fixes become nodes in the live graph, `T-01` goes
back to pending, and edges make it wait for both. The graph has to take that
without stalling, and git has to take branches named as siblings of a branch
that already exists.

That last part is the one real git is here for. A ref is a file, so
`agl/add-auth/T-01` existing rules out `agl/add-auth/T-01/bug-1` ever existing —
a bug id is composed by hyphenating onto its parent for exactly this reason, and
`paths.validate_node_id` refusing the `/` that would nest it is what enforces
that. The test at the bottom of this file proves the rule holds against git
rather than against a docstring.

The whole flow runs once for the module; the assertions read what it recorded.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import NO_PARAMS, AgentResult, AgentSpec, Tool
from agl.core.store import Store
from agl.core.store.impl.file_store import FileStore
from agl.core.vcs import MergeResult, VcsError
from agl.core.vcs.impl.git import Git
from agl.runtime import paths
from agl.runtime.dag import Dag, NodeId, NodeState
from tests.fakes import FakeAgentRunner, ScriptedRun
from tests.integration.conftest import PROJECT, copy_repo

LABEL = "add-auth"
PARENT = "T-01"

BUGS = [
    {"id": f"{PARENT}-bug-1", "n": 1, "file": "fix-1.txt", "title": "Token never expires"},
    {"id": f"{PARENT}-bug-2", "n": 2, "file": "fix-2.txt", "title": "No test for the refusal"},
]

FOUND_BUGS = AgentResult(
    text="Two problems.",
    structured={"bugs": BUGS},
    session_id="fake-session",
    cost_usd=0.0,
    num_turns=3,
    duration_ms=0,
    terminal_reason="completed",
)


def write_tool(store: Store, tree: Path, key: str) -> Tool:
    """Writes one file into one worktree, with the content taken from the store."""

    async def handler(arguments: dict[str, Any]) -> str:
        (tree / key).write_text(store.read(f"work/{key}"), encoding="utf-8")
        return f"wrote {key}"

    return Tool(name="write", description="Write your file.", schema=NO_PARAMS, handler=handler)


@dataclass(frozen=True)
class Snapshot:
    """The graph the moment after the bugs were folded into it."""

    parent_state: NodeState
    ready: tuple[NodeId, ...]
    blockers: tuple[NodeId, ...]
    unsatisfied: tuple[NodeId, ...]
    stalled: bool


@dataclass(frozen=True)
class Ran:
    """One parent ticket, two bugs off it, and everything they left behind."""

    repo: Path
    trees: Path
    vcs: Git
    dag: Dag
    parent_branch: str
    parent_sha: str
    bug_branches: tuple[str, ...]
    bug_shas: tuple[str, ...]
    after_bugs: Snapshot
    bug_merges: tuple[MergeResult, ...]
    ready_again: tuple[NodeId, ...]
    base_before_merge: str
    merged: MergeResult


async def drive(repo: Path, home: Path, trees: Path) -> Ran:
    """Run the parent, find two bugs, fix them off its branch, then land it."""
    vcs = Git(repo)
    store = FileStore(paths.run_dir(home, LABEL))
    store.write("work/auth.py", "TOKEN = 'set'\n")
    for bug in BUGS:
        store.write(f"work/{bug['file']}", f"{bug['title']}\n")

    dag = Dag()
    dag.add_node(PARENT)
    dag.claim(PARENT)

    runner = FakeAgentRunner(
        {
            "implement": ScriptedRun("done", calls=(("write", {}),)),
            "review": ScriptedRun(result=FOUND_BUGS),
            "fix": ScriptedRun("fixed", calls=(("write", {}),)),
        }
    )

    parent_branch = paths.branch(LABEL, PARENT)
    parent = vcs.add_worktree(
        paths.worktree_dir(trees, PROJECT, LABEL, PARENT), parent_branch, "main"
    )
    await runner.run(
        AgentSpec(
            prompt=f"Implement {PARENT}.",
            cwd=parent.path,
            role="implement",
            tools=(write_tool(store, parent.path, "auth.py"),),
        )
    )
    parent_sha = vcs.commit_all(parent.path, f"{PARENT}: add auth")
    assert parent_sha is not None

    review = await runner.run(
        AgentSpec(prompt="Review it.", cwd=parent.path, role="review", output_schema={})
    )
    bugs = review.structured["bugs"]

    # The mutation, in the order `Dag`'s docstring gives: the new work goes in,
    # then the edges, then the parent goes back to pending. Adding a blocker to
    # a claimed node is legal, which is what makes edges-first possible;
    # releasing first would leave the parent ready for a beat and re-claimable.
    for bug in bugs:
        dag.add_node(bug["id"])
    for bug in bugs:
        dag.add_edge(PARENT, bug["id"])
    dag.release(PARENT)

    after_bugs = Snapshot(
        parent_state=dag.state(PARENT),
        ready=dag.ready(),
        blockers=dag.blockers(PARENT),
        unsatisfied=dag.unsatisfied_blockers(PARENT),
        stalled=dag.is_stalled(),
    )

    # Both bug worktrees are cut from the parent branch at the same commit, so
    # the second merge is a real merge and not a fast-forward of the first.
    branches = tuple(paths.branch(LABEL, str(bug["id"])) for bug in bugs)
    trees_for_bugs = [
        vcs.add_worktree(
            paths.worktree_dir(trees, PROJECT, LABEL, bug["id"]), branch, parent_branch
        )
        for bug, branch in zip(bugs, branches, strict=True)
    ]

    shas: list[str] = []
    merges: list[MergeResult] = []
    for bug, branch, tree in zip(bugs, branches, trees_for_bugs, strict=True):
        dag.claim(bug["id"])
        await runner.run(
            AgentSpec(
                prompt=f"Fix {bug['title']}.",
                cwd=tree.path,
                role="fix",
                tools=(write_tool(store, tree.path, bug["file"]),),
            )
        )
        sha = vcs.commit_all(tree.path, f"{bug['id']}: {bug['title']}")
        assert sha is not None
        shas.append(sha)
        merges.append(vcs.merge(parent.path, branch))
        vcs.remove_worktree(tree.path)
        dag.complete(bug["id"])

    ready_again = dag.ready()
    dag.claim(PARENT)
    base_before_merge = vcs.rev_parse("main")
    merged = vcs.merge(repo, parent_branch)
    dag.complete(PARENT)
    vcs.remove_worktree(parent.path)

    return Ran(
        repo=repo,
        trees=trees,
        vcs=vcs,
        dag=dag,
        parent_branch=parent_branch,
        parent_sha=parent_sha,
        bug_branches=branches,
        bug_shas=tuple(shas),
        after_bugs=after_bugs,
        bug_merges=tuple(merges),
        ready_again=ready_again,
        base_before_merge=base_before_merge,
        merged=merged,
    )


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory, _template_repo: Path) -> Ran:
    root = tmp_path_factory.mktemp("bug-tickets")
    return asyncio.run(drive(copy_repo(root, _template_repo), root / "home", root / "trees"))


# -- the graph took the mutation ------------------------------------------


def test_the_parent_went_back_to_pending(ran: Ran) -> None:
    assert ran.after_bugs.parent_state is NodeState.PENDING


def test_the_parent_is_not_ready_and_the_bugs_are(ran: Ran) -> None:
    assert ran.after_bugs.ready == tuple(bug["id"] for bug in BUGS)
    assert PARENT not in ran.after_bugs.ready


def test_the_parent_now_waits_on_both_bugs(ran: Ran) -> None:
    assert ran.after_bugs.blockers == tuple(bug["id"] for bug in BUGS)
    assert ran.after_bugs.unsatisfied == ran.after_bugs.blockers


def test_the_graph_is_not_stalled(ran: Ran) -> None:
    assert ran.after_bugs.stalled is False


def test_the_bugs_sit_a_level_above_the_parent(ran: Ran) -> None:
    assert ran.dag.levels() == (tuple(bug["id"] for bug in BUGS), (PARENT,))


def test_the_parent_became_ready_once_both_bugs_were_done(ran: Ran) -> None:
    assert ran.ready_again == (PARENT,)


def test_the_graph_completes(ran: Ran) -> None:
    assert ran.dag.is_complete() is True


# -- the branches ----------------------------------------------------------


def test_the_bug_branches_are_named_as_siblings(ran: Ran) -> None:
    assert ran.bug_branches == (
        f"agl/{LABEL}/{PARENT}-bug-1",
        f"agl/{LABEL}/{PARENT}-bug-2",
    )
    for branch in ran.bug_branches:
        assert branch.rsplit("/", 1)[0] == ran.parent_branch.rsplit("/", 1)[0]
        assert not branch.startswith(f"{ran.parent_branch}/")


def test_the_bug_branches_were_cut_from_the_parent(ran: Ran) -> None:
    for sha in ran.bug_shas:
        assert ran.vcs.is_ancestor(ran.parent_sha, sha) is True


def test_both_bugs_merged_back_into_the_parent_branch(ran: Ran) -> None:
    assert [merge.clean for merge in ran.bug_merges] == [True, True]
    for sha in ran.bug_shas:
        assert ran.vcs.is_ancestor(sha, ran.parent_branch) is True


def test_the_parent_carried_all_three_commits_into_the_base(ran: Ran) -> None:
    assert ran.merged.clean is True
    for sha in (ran.parent_sha, *ran.bug_shas):
        assert ran.vcs.is_ancestor(sha, "main") is True
    assert ran.vcs.changed_files(ran.repo, ran.base_before_merge, ran.parent_branch) == (
        "auth.py",
        "fix-1.txt",
        "fix-2.txt",
    )


def test_the_base_holds_every_file(ran: Ran) -> None:
    assert (ran.repo / "auth.py").read_text(encoding="utf-8") == "TOKEN = 'set'\n"
    for bug in BUGS:
        assert (ran.repo / bug["file"]).is_file()


def test_every_branch_of_the_run_is_under_one_namespace(ran: Ran) -> None:
    assert ran.vcs.branches(paths.branch_namespace(LABEL)) == tuple(
        sorted((ran.parent_branch, *ran.bug_branches))
    )


def test_all_the_worktrees_are_gone(ran: Ran) -> None:
    assert [tree.path for tree in ran.vcs.list_worktrees()] == [ran.repo.resolve()]
    for node in (PARENT, *(str(bug["id"]) for bug in BUGS)):
        assert not paths.worktree_dir(ran.trees, PROJECT, LABEL, node).exists()


# -- the rule the naming exists for ---------------------------------------


def test_the_parent_ref_is_a_file_on_disk(ran: Ran) -> None:
    # Which is the whole reason a bug id hyphenates onto its parent: this path
    # cannot also be a directory holding `bug-1`.
    assert (ran.repo / ".git" / "refs" / "heads" / ran.parent_branch).is_file()


def test_a_path_child_of_the_parent_branch_cannot_exist(ran: Ran) -> None:
    with pytest.raises(VcsError):
        ran.vcs.create_branch(f"{ran.parent_branch}/bug-3", "main")
    assert ran.vcs.branch_exists(f"{ran.parent_branch}/bug-3") is False


def test_the_sibling_name_is_the_one_git_accepts(ran: Ran) -> None:
    ran.vcs.create_branch(paths.branch(LABEL, f"{PARENT}-bug-3"), "main")
    assert ran.vcs.branch_exists(f"agl/{LABEL}/{PARENT}-bug-3") is True
    ran.vcs.delete_branch(f"agl/{LABEL}/{PARENT}-bug-3", force=True)
