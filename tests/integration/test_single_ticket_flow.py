"""One ticket, start to finish, with every module in the path but the terminal.

`store` holds the spec and the work file, `dag` says the work may start, `paths`
names the branch and the tree, `vcs` cuts the worktree and lands the commit, and
a scripted agent does the work inside it through tools that close over the
store. Nothing here is the ticket workflow: it is the smallest wiring that
proves the seams meet, and if it does not hold nothing after it will.

The flow runs once for the whole module and every test reads what it left
behind, because running eight git processes per assertion is most of the file's
runtime and none of its meaning. Nothing below mutates the flow.

The merge runs in the main repository root, which is where merges happen: the
root is already on the base branch, and git will not check that branch out a
second time in a worktree of its own.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import NO_PARAMS, AgentResult, AgentSpec, Tool
from agl.core.store import Store
from agl.core.store.impl.file_store import FileStore
from agl.core.vcs import MergeResult, Worktree
from agl.core.vcs.impl.git import Git
from agl.runtime import paths
from agl.runtime.dag import Dag, NodeState
from tests.fakes import FakeAgentRunner, ScriptedRun, ToolResult
from tests.integration.conftest import PROJECT, copy_repo

LABEL = "add-auth"
TICKET = "T-01"

SPEC = "# Add auth\n\nA token check, in the file the ticket names.\n"
SOURCE = f"# {TICKET}\nTOKEN = 'set'\n"


# -- the tools the run is given -------------------------------------------


def read_spec_tool(store: Store) -> Tool:
    """The run's spec, straight out of the store. No parameters to widen."""

    async def handler(arguments: dict[str, Any]) -> str:
        return store.read("spec.md")

    return Tool(
        name="read_spec",
        description="The specification for this run.",
        schema=NO_PARAMS,
        handler=handler,
    )


def implement_tool(store: Store, tree: Path) -> Tool:
    """Writes the file this ticket owns, into the worktree it was bound to.

    The ticket says which file that is, so the tool reads `tickets.json` back
    out of the store: a document one module wrote, reached by a tool handed to
    another, and resolved to a path inside the tree the agent was given.
    """

    async def handler(arguments: dict[str, Any]) -> str:
        ticket = next(t for t in store.read_json("tickets.json") if t["id"] == TICKET)
        (tree / ticket["file"]).write_text(SOURCE, encoding="utf-8")
        return f"wrote {ticket['file']}"

    return Tool(
        name="implement",
        description="Write this ticket's file.",
        schema=NO_PARAMS,
        handler=handler,
    )


# -- the flow -------------------------------------------------------------


@dataclass(frozen=True)
class Flow:
    """What one ticket left behind, for the assertions below to pick over."""

    repo: Path
    trees: Path
    vcs: Git
    store: FileStore
    dag: Dag
    runner: FakeAgentRunner
    worktree: Worktree
    branch: str
    result: AgentResult
    commit: str | None
    base_before_merge: str
    merged: MergeResult


async def drive(repo: Path, home: Path, trees: Path) -> Flow:
    """One ticket, the whole way: plan it, work it, commit it, land it, tidy up."""
    paths.validate_label(LABEL)
    vcs = Git(repo)

    store = FileStore(paths.run_dir(home, LABEL))
    store.write("spec.md", SPEC)
    store.write_json("tickets.json", [{"id": TICKET, "title": "Add auth", "file": "auth.py"}])

    dag = Dag()
    dag.add_node(TICKET)
    assert dag.ready() == (TICKET,)
    dag.claim(TICKET)

    branch = paths.branch(LABEL, TICKET)
    worktree = vcs.add_worktree(paths.worktree_dir(trees, PROJECT, LABEL, TICKET), branch, "main")

    runner = FakeAgentRunner(
        {"implement": ScriptedRun("done", calls=(("read_spec", {}), ("implement", {})))}
    )
    result = await runner.run(
        AgentSpec(
            prompt=f"Implement {TICKET}.",
            cwd=worktree.path,
            role="implement",
            tools=(read_spec_tool(store), implement_tool(store, worktree.path)),
        )
    )

    commit = vcs.commit_all(worktree.path, f"{TICKET}: add auth")
    base_before_merge = vcs.rev_parse("main")
    merged = vcs.merge(repo, branch)
    dag.complete(TICKET)
    vcs.remove_worktree(worktree.path)

    return Flow(
        repo=repo,
        trees=trees,
        vcs=vcs,
        store=store,
        dag=dag,
        runner=runner,
        worktree=worktree,
        branch=branch,
        result=result,
        commit=commit,
        base_before_merge=base_before_merge,
        merged=merged,
    )


@pytest.fixture(scope="module")
def flow(tmp_path_factory: pytest.TempPathFactory, _template_repo: Path) -> Flow:
    """Run the flow once. A sync fixture, so it may own its own event loop."""
    root = tmp_path_factory.mktemp("single-ticket")
    return asyncio.run(drive(copy_repo(root, _template_repo), root / "home", root / "trees"))


# -- what each seam was supposed to do ------------------------------------


def test_the_agent_was_run_in_the_worktree(flow: Flow) -> None:
    assert flow.runner.specs[0].cwd == flow.worktree.path
    assert flow.worktree.path == paths.worktree_dir(flow.trees, PROJECT, LABEL, TICKET).resolve()


def test_the_branch_is_the_one_paths_named(flow: Flow) -> None:
    assert flow.branch == f"agl/{LABEL}/{TICKET}"
    assert flow.worktree.branch == flow.branch


def test_a_tool_handed_to_the_agent_reached_the_store(flow: Flow) -> None:
    assert flow.runner.tool_results == [ToolResult(SPEC), ToolResult("wrote auth.py")]


def test_the_agent_returned_what_it_was_scripted_to(flow: Flow) -> None:
    assert flow.result.text == "done"


def test_the_commit_came_back_with_a_sha(flow: Flow) -> None:
    assert flow.commit is not None
    assert flow.vcs.rev_parse(flow.branch) == flow.commit


def test_nothing_was_left_uncommitted(flow: Flow) -> None:
    # Which is why `remove_worktree` at the end of the flow did not need forcing.
    assert flow.vcs.is_dirty(flow.repo) is False


def test_the_merge_was_clean(flow: Flow) -> None:
    assert flow.merged.clean is True
    assert flow.merged.conflicted == ()
    assert flow.merged.sha is not None


def test_the_work_landed_on_the_base(flow: Flow) -> None:
    assert flow.commit is not None
    assert flow.vcs.is_ancestor(flow.commit, "main") is True
    assert (flow.repo / "auth.py").read_text(encoding="utf-8") == SOURCE


def test_the_ticket_touched_only_its_own_file(flow: Flow) -> None:
    # Against the base as it stood before the merge: afterwards the branch is
    # reachable from `main`, so the merge-base diff between them is empty.
    assert flow.vcs.changed_files(flow.repo, flow.base_before_merge, flow.branch) == ("auth.py",)


def test_the_graph_is_complete(flow: Flow) -> None:
    assert flow.dag.state(TICKET) is NodeState.DONE
    assert flow.dag.is_complete() is True
    assert flow.dag.is_stalled() is False


def test_the_worktree_is_gone_and_the_registry_is_clean(flow: Flow) -> None:
    assert not paths.worktree_dir(flow.trees, PROJECT, LABEL, TICKET).exists()
    assert flow.vcs.list_worktrees() == (Worktree(flow.repo.resolve(), "main"),)


def test_the_branch_outlives_the_worktree(flow: Flow) -> None:
    # The record of what a ticket did is its branch, not the tree it was written
    # in; a clean at the end of a run is what deletes it.
    assert flow.vcs.branch_exists(flow.branch) is True
    assert flow.vcs.branches(paths.branch_namespace(LABEL)) == (flow.branch,)


def test_the_run_documents_are_still_readable_afterwards(flow: Flow) -> None:
    assert flow.store.list() == ("spec.md", "tickets.json")
    assert flow.store.read_json("tickets.json")[0]["file"] == "auth.py"
