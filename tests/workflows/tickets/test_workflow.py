"""End-to-end tests for the ticket workflow's graph.

`FakeAgentRunner` and `HeadlessTerminal` throughout — no test here calls a real
model or paints a real screen. `WritingAgentRunner` wraps the fake to give an
"implement" call the one thing the fake cannot do on its own: put a file in the
worktree, the way a real agent's file tools would. Everything else — git,
worktrees, the store — is real, in `tmp_path`.

These are the last safety net before a real run, so they drive `Run.go()`
whole, the way `agl.cli` will.
"""

import asyncio
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from agl.config import ProjectConfig
from agl.core import paths
from agl.core.agent import AgentOption, AgentQuestion, AgentResult, AgentRunner, AgentSpec
from agl.core.store.impl.file_store import FileStore
from agl.core.terminal import Answer, Question, Screen, Terminal
from agl.core.vcs.impl.git import Git
from agl.workflows.tickets import workflow as workflow_module
from agl.workflows.tickets.merge import MergeQueue, MergeRequest
from agl.workflows.tickets.models import Status
from agl.workflows.tickets.state import check_consistent
from agl.workflows.tickets.workflow import DecomposeAbortedError, Deps, PreflightError, Run
from tests.conftest import git
from tests.fakes import FakeAgentRunner, HeadlessTerminal, ScriptedRun, _HeadlessSession
from tests.integration.conftest import PROJECT

LABEL = "add-auth"


# -- wiring -----------------------------------------------------------------


def start(repo: Path, name: str = "feature") -> None:
    """Move `repo` off `main` onto a feature branch, as a real run requires."""
    git(repo, "checkout", "-b", name, "main")


def config(repo: Path, trees: Path) -> ProjectConfig:
    config_dir = repo.parent / "agl-config"
    return ProjectConfig(
        name=PROJECT,
        repo=repo,
        trees_root=trees,
        build=(sys.executable, "-c", "pass"),
        build_timeout=30.0,
        standards=config_dir / "standards.md",
        config_dir=config_dir,
    )


def deps(
    repo: Path, home: Path, trees: Path, agent: AgentRunner, terminal: Terminal
) -> Deps:
    return Deps(
        agent=agent,
        vcs=Git(repo),
        store=FileStore(paths.run_dir(home, LABEL)),
        terminal=terminal,
        config=config(repo, trees),
    )


class _GatedSession(_HeadlessSession):
    """Holds "press enter to continue" until the test says the fix is in.

    Every other question answers the instant it is asked, same as
    `HeadlessTerminal` — only this one title waits, which is what makes the
    scripted answer mean "a person looked and pressed enter" rather than
    "whatever answer happened to be queued".
    """

    def __init__(
        self, terminal: "GatedTerminal", build: Callable[[], Screen], ready: asyncio.Event
    ) -> None:
        super().__init__(terminal, build)
        self._ready = ready

    async def ask(self, question: Question) -> Answer:
        if question.title == "press enter to continue":
            await self._ready.wait()
        return await super().ask(question)


class GatedTerminal(HeadlessTerminal):
    """A `HeadlessTerminal` whose resume prompt is held by `_GatedSession`."""

    def __init__(self, ready: asyncio.Event, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ready = ready

    def queue(self, answer: Answer) -> None:
        self._answers.append(answer)

    @asynccontextmanager
    async def live(self, build: Callable[[], Screen], fps: int = 4) -> Any:
        session = _GatedSession(self, build, self._ready)
        session.frame()
        try:
            yield session
        finally:
            session.frame()


class RecordingQueue(MergeQueue):
    """A `MergeQueue` that remembers every request `put` to it.

    Stands in for the real queue via `monkeypatch`, so a test can assert the
    workflow routed a bug ticket's merge through the queue rather than only
    checking the end state, which a bypass could also reach.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seen: list[MergeRequest] = []

    def put(self, request: MergeRequest) -> None:
        self.seen.append(request)
        super().put(request)


type _Overrides = dict[str, tuple[str, str]]


class WritingAgentRunner(AgentRunner):
    """Wraps a `FakeAgentRunner`, and puts a file in the tree after "implement".

    The fake can only invoke tools from `spec.tools`, and `implement_tools`
    has none that write — a real agent's file-editing tools are the SDK's own,
    outside this module's `Tool` type entirely. This stands in for them:
    `overrides` names a different path/content for a ticket that must collide
    with another one, and every other ticket gets a file of its own.
    """

    def __init__(self, inner: AgentRunner, overrides: _Overrides | None = None) -> None:
        self._inner = inner
        self._overrides = overrides or {}
        self.implemented: list[str] = []

    async def run(
        self, spec: AgentSpec, on_activity: Any = None, on_question: Any = None
    ) -> AgentResult:
        result = await self._inner.run(spec, on_activity, on_question)
        if spec.role == "implement":
            ticket_id = spec.cwd.name
            self.implemented.append(ticket_id)
            default = (f"{ticket_id}.txt", f"{ticket_id}\n")
            relpath, content = self._overrides.get(ticket_id, default)
            (spec.cwd / relpath).write_text(content, encoding="utf-8")
        return result


class ScriptedByTicket(AgentRunner):
    """Routes one role to a per-ticket queue of results, popped in call order.

    Two tickets sharing a role — every review, across every ticket — never
    share a script this way, and a ticket reviewed more than once gets its
    results in the order it is reviewed. Everything else falls through to
    `inner`.
    """

    def __init__(self, inner: AgentRunner, role: str, queues: dict[str, list[AgentResult]]) -> None:
        self._inner = inner
        self._role = role
        self._queues = queues

    async def run(
        self, spec: AgentSpec, on_activity: Any = None, on_question: Any = None
    ) -> AgentResult:
        if spec.role == self._role:
            return self._queues[spec.cwd.name].pop(0)
        return await self._inner.run(spec, on_activity, on_question)


def ticket_json(
    id_: str, title: str, deliverables: tuple[str, ...], blocked_by: tuple[str, ...] = ()
) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": id_, "title": title, "deliverables": list(deliverables)}
    if blocked_by:
        payload["blocked_by"] = list(blocked_by)
    return payload


def findings_result(*findings: dict[str, Any]) -> AgentResult:
    return AgentResult(
        text="reviewed",
        structured={"findings": list(findings)},
        session_id="s-1",
        cost_usd=0.0,
        num_turns=1,
        duration_ms=0,
        terminal_reason="completed",
    )


def finding(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "Q-1",
        "severity": "high",
        "title": "Missing null check",
        "detail": "auth() does not check for a None token.",
        "files": ["src/auth.py"],
    }
    payload.update(overrides)
    return payload


def groups_result(*groups: dict[str, Any]) -> AgentResult:
    return AgentResult(
        text="triaged",
        structured={"groups": list(groups)},
        session_id="s-1",
        cost_usd=0.0,
        num_turns=1,
        duration_ms=0,
        terminal_reason="completed",
    )


def group(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": "Fix the finding",
        "deliverables": ["Fix it."],
        "findings": ["Q-1"],
    }
    payload.update(overrides)
    return payload


def clean_review() -> dict[str, ScriptedRun]:
    """Both reviewers scripted to find nothing."""
    return {
        "review-quality": ScriptedRun(result=findings_result()),
        "review-spec": ScriptedRun(result=findings_result()),
    }


# -- one ticket, start to finish ---------------------------------------------


async def test_one_ticket_end_to_end(tmp_path: Path, repo: Path) -> None:
    start(repo)
    home, trees = tmp_path / "home", tmp_path / "trees"
    terminal = HeadlessTerminal(answers=[Answer("approve", was_free_text=False)])
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    fake = FakeAgentRunner(
        {
            "interview": ScriptedRun("noted", calls=(("save_spec", {"content": "# Add auth\n"}),)),
            "decompose": ScriptedRun(
                "planned", calls=(("save_tickets", {"tickets": tickets}),)
            ),
            "implement": ScriptedRun("done", calls=(("get_ticket", {}), ("read_spec", {}))),
            **clean_review(),
        }
    )
    d = deps(repo, home, trees, WritingAgentRunner(fake), terminal)
    run = Run(d, LABEL, "Add auth please", max_concurrent=2)

    await run.go()

    assert run.state.tickets["T-01"].status is Status.MERGED
    assert run.state.dag.is_complete()
    check_consistent(run.state)
    assert (repo / "T-01.txt").read_text(encoding="utf-8") == "T-01\n"
    assert [w.path for w in d.vcs.list_worktrees()] == [repo.resolve()]
    assert d.vcs.current_branch() == "feature"
    assert d.vcs.is_ancestor(d.vcs.rev_parse("feature"), "feature")


def test_preflight_refuses_a_dirty_repo(tmp_path: Path, repo: Path) -> None:
    start(repo)
    (repo / "dirty.txt").write_text("oops\n", encoding="utf-8")
    home, trees = tmp_path / "home", tmp_path / "trees"
    d = deps(repo, home, trees, FakeAgentRunner(), HeadlessTerminal())
    run = Run(d, LABEL, "Add auth please", max_concurrent=1)

    with pytest.raises(PreflightError):
        run.preflight()


def test_preflight_refuses_main(tmp_path: Path, repo: Path) -> None:
    home, trees = tmp_path / "home", tmp_path / "trees"
    d = deps(repo, home, trees, FakeAgentRunner(), HeadlessTerminal())
    run = Run(d, "main-run", "Add auth please", max_concurrent=1)

    with pytest.raises(PreflightError):
        run.preflight()


def test_preflight_refuses_a_label_already_in_use(tmp_path: Path, repo: Path) -> None:
    start(repo)
    home, trees = tmp_path / "home", tmp_path / "trees"
    d = deps(repo, home, trees, FakeAgentRunner(), HeadlessTerminal())
    d.store.write("spec.md", "already started\n")
    run = Run(d, LABEL, "Add auth please", max_concurrent=1)

    with pytest.raises(PreflightError):
        run.preflight()


def test_preflight_allows_a_label_that_matches_the_current_branch(
    tmp_path: Path, repo: Path
) -> None:
    """A person may reasonably pass `--name` matching the branch they are
    standing on, so the label can name a real ref: their own branch. That
    must not read as "already in use" — only leftover `agl/<label>/*` branches
    or run state should.
    """
    start(repo, name=LABEL)
    home, trees = tmp_path / "home", tmp_path / "trees"
    d = deps(repo, home, trees, FakeAgentRunner(), HeadlessTerminal())
    run = Run(d, LABEL, "Add auth please", max_concurrent=1)

    run.preflight()


# -- four tickets, one dependency edge ---------------------------------------


async def test_four_tickets_with_a_dependency_edge(tmp_path: Path, repo: Path) -> None:
    start(repo)
    home, trees = tmp_path / "home", tmp_path / "trees"
    terminal = HeadlessTerminal(answers=[Answer("approve", was_free_text=False)])
    tickets = [
        ticket_json("T1", "One", ("t1.py",)),
        ticket_json("T2", "Two", ("t2.py",)),
        ticket_json("T3", "Three", ("t3.py",)),
        ticket_json("T4", "Four", ("t4.py",), blocked_by=("T3",)),
    ]
    fake = FakeAgentRunner(
        {
            "interview": ScriptedRun("noted", calls=(("save_spec", {"content": "# spec\n"}),)),
            "decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),)),
            "implement": ScriptedRun("done"),
            **clean_review(),
        }
    )
    writer = WritingAgentRunner(fake)
    d = deps(repo, home, trees, writer, terminal)
    run = Run(d, LABEL, "do four things", max_concurrent=2)

    await run.go()

    for ticket_id in ("T1", "T2", "T3", "T4"):
        assert run.state.tickets[ticket_id].status is Status.MERGED
        assert (repo / f"{ticket_id}.txt").read_text(encoding="utf-8") == f"{ticket_id}\n"
    assert run.state.dag.is_complete()
    check_consistent(run.state)
    assert writer.implemented.index("T3") < writer.implemented.index("T4")


# -- a review that finds bugs -------------------------------------------------


async def test_review_findings_file_bugs_that_run_before_the_parent_merges(
    tmp_path: Path, repo: Path
) -> None:
    start(repo)
    home, trees = tmp_path / "home", tmp_path / "trees"
    terminal = HeadlessTerminal(answers=[Answer("approve", was_free_text=False)])
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    fake = FakeAgentRunner(
        {
            "interview": ScriptedRun("noted", calls=(("save_spec", {"content": "# spec\n"}),)),
            "decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),)),
            "implement": ScriptedRun("done"),
            "review-quality": ScriptedRun(result=findings_result()),
            "triage": ScriptedRun(
                result=groups_result(
                    group(title="Fix S-1", deliverables=["Fix it."], findings=["S-1"]),
                    group(title="Fix S-2", deliverables=["Fix it too."], findings=["S-2"]),
                )
            ),
        }
    )
    spec_queues = {
        "T-01": [
            findings_result(
                finding(id="S-1", severity="high", files=["auth.py"]),
                finding(id="S-2", severity="high", files=["auth.py"]),
            ),
            findings_result(finding(id="S-3", severity="high", files=["auth.py"])),
            findings_result(),
        ],
        "T-01-bug-1": [findings_result()],
        "T-01-bug-2": [findings_result()],
        "T-01-bug-3": [findings_result()],
    }
    writer = WritingAgentRunner(fake)
    scripted = ScriptedByTicket(writer, "review-spec", spec_queues)
    d = deps(repo, home, trees, scripted, terminal)
    run = Run(d, LABEL, "Add auth please", max_concurrent=2)

    await run.go()

    assert run.state.tickets["T-01"].status is Status.MERGED
    for bug_id in ("T-01-bug-1", "T-01-bug-2", "T-01-bug-3"):
        assert run.state.tickets[bug_id].status is Status.MERGED
    assert run.state.dag.is_complete()
    check_consistent(run.state)

    # The parent's second and third passes ran no implementation agent.
    assert writer.implemented.count("T-01") == 1

    # The second round of bugs did not reuse the first round's ids.
    assert set(run.state.tickets) == {"T-01", "T-01-bug-1", "T-01-bug-2", "T-01-bug-3"}

    # Triage ran once — for the two-high round. One high skips it, zero skips it.
    triage_specs = [s for s in fake.specs if s.role == "triage"]
    assert len(triage_specs) == 1

    assert (repo / "T-01.txt").read_text(encoding="utf-8") == "T-01\n"
    for bug_id in ("T-01-bug-1", "T-01-bug-2", "T-01-bug-3"):
        assert (repo / f"{bug_id}.txt").read_text(encoding="utf-8") == f"{bug_id}\n"
    assert [w.path for w in d.vcs.list_worktrees()] == [repo.resolve()]


# -- a question mid-run -------------------------------------------------------


async def test_a_question_mid_run_reaches_the_terminal_and_shows_awaiting_input(
    tmp_path: Path, repo: Path
) -> None:
    start(repo)
    home, trees = tmp_path / "home", tmp_path / "trees"
    question = AgentQuestion(
        title="Which token store?", options=(AgentOption("memory", "In process"),)
    )
    terminal = HeadlessTerminal(
        answers=[Answer("approve", was_free_text=False), Answer("memory", was_free_text=False)]
    )
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    fake = FakeAgentRunner(
        {
            "interview": ScriptedRun("noted", calls=(("save_spec", {"content": "# spec\n"}),)),
            "decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),)),
            "implement": ScriptedRun("done", question=question),
            **clean_review(),
        }
    )
    writer = WritingAgentRunner(fake)
    d = deps(repo, home, trees, writer, terminal)
    run = Run(d, LABEL, "Add auth please", max_concurrent=1)

    await run.go()

    assert fake.answers == ["memory"]
    assert [q.title for q in terminal.questions] == [
        "Approve these 1 tickets?",
        "Which token store?",
    ]
    assert any("waiting for you" in frame for frame in terminal.frames)
    assert run.state.tickets["T-01"].status is Status.MERGED
    check_consistent(run.state)


# -- a merge conflict halts the run -------------------------------------------


async def test_a_merge_conflict_halts_the_run(tmp_path: Path, repo: Path) -> None:
    start(repo)
    (repo / "shared.py").write_text("MODE = 'off'\n", encoding="utf-8")
    git(repo, "add", "shared.py")
    git(repo, "commit", "-m", "add shared.py")
    home, trees = tmp_path / "home", tmp_path / "trees"
    ready = asyncio.Event()
    terminal = GatedTerminal(ready, answers=[Answer("approve", was_free_text=False)])
    tickets = [
        ticket_json("T1", "One", ("shared.py",)),
        ticket_json("T2", "Two", ("shared.py",)),
        ticket_json("T3", "Three", ("t3.py",), blocked_by=("T2",)),
    ]
    fake = FakeAgentRunner(
        {
            "interview": ScriptedRun("noted", calls=(("save_spec", {"content": "# spec\n"}),)),
            "decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),)),
            "implement": ScriptedRun("done"),
            **clean_review(),
        }
    )
    overrides = {"T1": ("shared.py", "MODE = 'ours'\n"), "T2": ("shared.py", "MODE = 'theirs'\n")}
    writer = WritingAgentRunner(fake, overrides)
    d = deps(repo, home, trees, writer, terminal)
    run = Run(d, LABEL, "conflict please", max_concurrent=2)

    task = asyncio.create_task(run.go())
    async with asyncio.timeout(10.0):
        while run.state.halt is None:
            await asyncio.sleep(0.01)

    # Nothing new started: T3 waits on T2, which is stuck mid-merge.
    assert run.state.tickets["T3"].status is Status.PENDING
    merged = {tid for tid in ("T1", "T2") if run.state.tickets[tid].status is Status.MERGED}
    assert len(merged) == 1
    conflicted = "T2" if "T1" in merged else "T1"

    # A person resolves it in the repository root, the way the halt says to,
    # then presses enter — modelled as answering only once that has happened.
    (repo / "shared.py").write_text("MODE = 'resolved'\n", encoding="utf-8")
    d.vcs.commit_merge(repo, f"merge {conflicted}")
    terminal.queue(Answer("continue", was_free_text=False))
    ready.set()

    async with asyncio.timeout(10.0):
        await task

    assert run.state.tickets["T1"].status is Status.MERGED
    assert run.state.tickets["T2"].status is Status.MERGED
    assert run.state.tickets["T3"].status is Status.MERGED
    assert run.state.dag.is_complete()
    check_consistent(run.state)
    assert [w.path for w in d.vcs.list_worktrees()] == [repo.resolve()]


# -- bug merges go through the queue too --------------------------------------


async def test_a_bug_tickets_merge_goes_through_the_queue(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow_module, "MergeQueue", RecordingQueue)

    start(repo)
    home, trees = tmp_path / "home", tmp_path / "trees"
    terminal = HeadlessTerminal(answers=[Answer("approve", was_free_text=False)])
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    fake = FakeAgentRunner(
        {
            "interview": ScriptedRun("noted", calls=(("save_spec", {"content": "# spec\n"}),)),
            "decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),)),
            "implement": ScriptedRun("done"),
            "review-quality": ScriptedRun(result=findings_result()),
            "triage": ScriptedRun(result=groups_result(group())),
        }
    )
    spec_queues = {
        "T-01": [findings_result(finding()), findings_result()],
        "T-01-bug-1": [findings_result()],
    }
    writer = WritingAgentRunner(fake)
    scripted = ScriptedByTicket(writer, "review-spec", spec_queues)
    d = deps(repo, home, trees, scripted, terminal)
    run = Run(d, LABEL, "Add auth please", max_concurrent=2)

    await run.go()

    assert run.state.tickets["T-01"].status is Status.MERGED
    assert run.state.tickets["T-01-bug-1"].status is Status.MERGED
    assert isinstance(run.merge_queue, RecordingQueue)
    ticket_ids = [r.ticket_id for r in run.merge_queue.seen]
    assert ticket_ids == ["T-01-bug-1", "T-01"]

    bug_request = run.merge_queue.seen[0]
    assert bug_request.target == paths.branch(LABEL, "T-01")
    assert bug_request.cwd != d.config.repo.resolve()

    feature_request = run.merge_queue.seen[1]
    assert feature_request.target == "feature"
    assert feature_request.cwd == d.config.repo


async def test_a_bug_merge_conflict_halts_resumably_and_resuming_lets_the_parent_merge(
    tmp_path: Path, repo: Path
) -> None:
    start(repo)
    (repo / "shared.py").write_text("MODE = 'off'\n", encoding="utf-8")
    git(repo, "add", "shared.py")
    git(repo, "commit", "-m", "add shared.py")
    home, trees = tmp_path / "home", tmp_path / "trees"
    ready = asyncio.Event()
    terminal = GatedTerminal(ready, answers=[Answer("approve", was_free_text=False)])
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    fake = FakeAgentRunner(
        {
            "interview": ScriptedRun("noted", calls=(("save_spec", {"content": "# spec\n"}),)),
            "decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),)),
            "implement": ScriptedRun("done"),
            "review-quality": ScriptedRun(result=findings_result()),
            "triage": ScriptedRun(
                result=groups_result(
                    group(title="Fix S-1", deliverables=["Fix it."], findings=["S-1"]),
                    group(title="Fix S-2", deliverables=["Fix it too."], findings=["S-2"]),
                )
            ),
        }
    )
    spec_queues = {
        "T-01": [
            findings_result(
                finding(id="S-1", severity="high", files=["shared.py"]),
                finding(id="S-2", severity="high", files=["shared.py"]),
            ),
            findings_result(),
        ],
        "T-01-bug-1": [findings_result()],
        "T-01-bug-2": [findings_result()],
    }
    overrides = {
        "T-01-bug-1": ("shared.py", "MODE = 'ours'\n"),
        "T-01-bug-2": ("shared.py", "MODE = 'theirs'\n"),
    }
    writer = WritingAgentRunner(fake, overrides)
    scripted = ScriptedByTicket(writer, "review-spec", spec_queues)
    d = deps(repo, home, trees, scripted, terminal)
    run = Run(d, LABEL, "conflict please", max_concurrent=2)

    task = asyncio.create_task(run.go())
    async with asyncio.timeout(10.0):
        while run.state.halt is None:
            await asyncio.sleep(0.01)

    bug_ids = ("T-01-bug-1", "T-01-bug-2")
    merged = {tid for tid in bug_ids if run.state.tickets[tid].status is Status.MERGED}
    assert len(merged) == 1
    conflicted = "T-01-bug-2" if "T-01-bug-1" in merged else "T-01-bug-1"
    assert run.state.halt is not None
    assert run.state.halt.resumable is True

    parent_branch = paths.branch(LABEL, "T-01")
    parent_tree = next(w.path for w in d.vcs.list_worktrees() if w.branch == parent_branch)

    # A person resolves it in the parent's worktree — not the repository root,
    # which was never touched by this merge — then presses enter.
    (parent_tree / "shared.py").write_text("MODE = 'resolved'\n", encoding="utf-8")
    d.vcs.commit_merge(parent_tree, f"merge {conflicted}")
    terminal.queue(Answer("continue", was_free_text=False))
    ready.set()

    async with asyncio.timeout(10.0):
        await task

    assert run.state.tickets["T-01-bug-1"].status is Status.MERGED
    assert run.state.tickets["T-01-bug-2"].status is Status.MERGED
    assert run.state.tickets["T-01"].status is Status.MERGED
    assert run.state.dag.is_complete()
    check_consistent(run.state)
    assert [w.path for w in d.vcs.list_worktrees()] == [repo.resolve()]


# -- decompose: revise and abort ----------------------------------------------


async def test_decompose_revision_sends_the_free_text_back_and_calls_the_agent_again(
    tmp_path: Path, repo: Path
) -> None:
    start(repo)
    home, trees = tmp_path / "home", tmp_path / "trees"
    terminal = HeadlessTerminal(
        answers=[
            Answer("split them into two tickets", was_free_text=True),
            Answer("approve", was_free_text=False),
        ]
    )
    first = [ticket_json("T-01", "Add auth", ("auth.py",))]
    second = [
        ticket_json("T-01", "Add token issuing", ("auth.py",)),
        ticket_json("T-02", "Add token checking", ("check.py",)),
    ]
    fake = FakeAgentRunner(
        [
            ScriptedRun("noted", calls=(("save_spec", {"content": "# spec\n"}),)),
            ScriptedRun("planned", calls=(("save_tickets", {"tickets": first}),)),
            ScriptedRun(
                "replanned",
                calls=(("read_spec", {}), ("save_tickets", {"tickets": second})),
            ),
            ScriptedRun("done"),
            ScriptedRun("done"),
            findings_result(),
            findings_result(),
            findings_result(),
            findings_result(),
        ]
    )
    writer = WritingAgentRunner(fake)
    d = deps(repo, home, trees, writer, terminal)
    run = Run(d, LABEL, "Add auth please", max_concurrent=2)

    await run.go()

    assert set(run.state.tickets) == {"T-01", "T-02"}
    revised_read = fake.tool_results[-2]
    assert "split them into two tickets" in revised_read.text
    check_consistent(run.state)


def test_decompose_abort_exits_without_creating_worktrees_or_branches(
    tmp_path: Path, repo: Path
) -> None:
    start(repo)
    home, trees = tmp_path / "home", tmp_path / "trees"
    terminal = HeadlessTerminal(answers=[Answer("abort", was_free_text=False)])
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    fake = FakeAgentRunner(
        {
            "interview": ScriptedRun("noted", calls=(("save_spec", {"content": "# spec\n"}),)),
            "decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),)),
        }
    )
    d = deps(repo, home, trees, WritingAgentRunner(fake), terminal)
    run = Run(d, LABEL, "Add auth please", max_concurrent=2)

    with pytest.raises(DecomposeAbortedError):
        asyncio.run(run.go())

    assert d.vcs.list_worktrees() == (d.vcs.list_worktrees()[0],)
    assert d.vcs.branches(paths.branch_namespace(LABEL)) == ()
