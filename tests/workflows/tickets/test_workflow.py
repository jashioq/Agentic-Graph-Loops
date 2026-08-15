"""The ticket workflow end to end: the loop, the policy, and the screens it swaps.

`FakeAgentRunner` and `HeadlessTerminal` throughout — no test here calls a real
model or paints a real screen. `WritingAgentRunner` wraps the fake to give an
"implement" call the one thing the fake cannot do on its own: put a file in the
worktree, the way a real agent's file tools would. Everything else — git,
worktrees, the store — is real, in `tmp_path`, and assembled by the harness in
`tests/runtime/conftest.py`.

Nothing here reaches into a running loop. What a run did is read off what it
left behind: the files that landed on the base branch, the worktrees it
released, the frames it painted — the dashboard is the window onto every
ticket's status — the documents in its store, and the exception it raised. That
is also what the cli sees, so these tests exercise the same surface it does.

The state document is now one of those things to read, and the one that says
most: `run.json` records what the run was asked for, and `state.json` is every
ticket at the moment the run stopped writing to it.

The policy the loop is built from is plain functions, tested as such at the foot
of the file: `resolve` (what a merge that did not land means), `failed` (what an
exception out of a ticket means), `base_for` (a bug branches off its parent) and
`asker` (a question suspends its own ticket and no other).
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import AgentOption, AgentQuestion, AgentResult, AgentRunner, AgentSpec
from agl.core.store.impl.file_store import FileStore
from agl.core.terminal import Answer, LiveSession, Question, Screen
from agl.runtime import paths
from agl.runtime.context import PreflightError, RunContext
from agl.runtime.display import Board, Display, live
from agl.runtime.merge import (
    MergeConfig,
    MergeDecision,
    MergeOutcome,
    MergeQueue,
    MergeRequest,
    MergeStatus,
)
from agl.runtime.record import StateFile, read_record
from agl.runtime.worktrees import Worktrees
from agl.workflows.tickets import workflow as workflow_module
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.snapshot import RunFile
from agl.workflows.tickets.state import Run, with_bugs, with_status, with_tickets
from agl.workflows.tickets.ticket import Loop, asker, base_for
from agl.workflows.tickets.workflow import (
    DecomposeAbortedError,
    HaltedError,
    InterviewIncompleteError,
    failed,
    resolve,
    run,
)
from tests.conftest import git
from tests.fakes import (
    FakeAgentRunner,
    HeadlessTerminal,
    ScriptedRun,
    _HeadlessSession,
)
from tests.runtime.conftest import LABEL, PROJECT, REQUEST, context, feature

NOW = 1_000.0
GATE = "press enter to continue"
"""The title `resolve` puts up while a person deals with a halt."""

APPROVE = Answer("approve", was_free_text=False)


# -- holding the run at a halt -------------------------------------------------


@dataclass
class Gate:
    """Two events around the resume prompt, so a test can stand where a person does.

    `asked` fires when the run puts the prompt up, which is the observable
    "this run has halted" from outside — the halt is in a document, but waiting
    on the prompt is what says the run has got as far as showing it. `ready` is
    the test saying the repository has been fixed, so the scripted answer means
    "somebody looked" rather than "an answer happened to be queued".
    """

    asked: asyncio.Event = field(default_factory=asyncio.Event)
    ready: asyncio.Event = field(default_factory=asyncio.Event)


class _GatedSession(_HeadlessSession):
    """Holds the resume prompt until the test says the fix is in."""

    def __init__(
        self, terminal: "GatedTerminal", build: Callable[[], Screen] | None, gate: Gate
    ) -> None:
        super().__init__(terminal, build)
        self._gate = gate

    async def ask(self, question: Question) -> Answer:
        if question.title == GATE:
            # The frame the halt is on, captured before the wait: a real
            # session is repainting throughout, so the banner is on screen for
            # as long as a person is being waited on.
            self.frame()
            self._gate.asked.set()
            await self._gate.ready.wait()
        return await super().ask(question)


class GatedTerminal(HeadlessTerminal):
    """A `HeadlessTerminal` whose resume prompt is held by `_GatedSession`."""

    def __init__(self, gate: Gate, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gate = gate

    def queue(self, answer: Answer) -> None:
        self._answers.append(answer)

    @asynccontextmanager
    async def live(
        self, build: Callable[[], Screen] | None = None, fps: int = 4
    ) -> AsyncIterator[LiveSession]:
        session = _GatedSession(self, build, self.gate)
        session.frame()
        try:
            yield session
        finally:
            session.frame()


# -- standing in for the agent's own file tools --------------------------------


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
    """Routes one role to a per-ticket queue of scripted runs, popped in call order.

    Two tickets sharing a role — every review, across every ticket — never
    share a script this way, and a ticket reviewed more than once gets its
    results in the order it is reviewed. Everything else falls through to
    `inner`.

    A scripted run's `calls` are invoked against the real spec's tools, the
    same as `FakeAgentRunner` does, so a review scripted here still reports
    through `save_findings` rather than a bare returned result.
    """

    def __init__(self, inner: AgentRunner, role: str, queues: dict[str, list[ScriptedRun]]) -> None:
        self._inner = inner
        self._role = role
        self._queues = queues

    async def run(
        self, spec: AgentSpec, on_activity: Any = None, on_question: Any = None
    ) -> AgentResult:
        if spec.role == self._role:
            scripted = self._queues[spec.cwd.name].pop(0)
            for name, arguments in scripted.calls:
                tool = next(tool for tool in spec.tools if tool.name == name)
                await tool.handler(arguments)
            return scripted.outcome()
        return await self._inner.run(spec, on_activity, on_question)


class _RecordingQueue(MergeQueue):
    """A `MergeQueue` that appends every request to a list the test holds."""

    def __init__(self, seen: list[MergeRequest], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._seen = seen

    async def submit(self, request: MergeRequest) -> MergeOutcome:
        self._seen.append(request)
        return await super().submit(request)


def recording_queue(seen: list[MergeRequest]) -> Callable[..., MergeQueue]:
    """A `MergeQueue` factory to monkeypatch in, so a test can see what was submitted.

    Asserting on the end state alone would pass for a workflow that bypassed
    the queue entirely; this is how a bug ticket's merge is shown to go through
    the same queue every other merge does.
    """

    def make(*args: Any, **kwargs: Any) -> MergeQueue:
        return _RecordingQueue(seen, *args, **kwargs)

    return make


# -- scripting a run -----------------------------------------------------------


def ticket_json(
    id_: str, title: str, deliverables: tuple[str, ...], blocked_by: tuple[str, ...] = ()
) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": id_, "title": title, "deliverables": list(deliverables)}
    if blocked_by:
        payload["blocked_by"] = list(blocked_by)
    return payload


def findings_result(*findings: dict[str, Any]) -> ScriptedRun:
    """A run that reports through `save_findings`, the way a real one now must."""
    return ScriptedRun(text="reviewed", calls=(("save_findings", {"findings": list(findings)}),))


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


def groups_result(*groups: dict[str, Any]) -> ScriptedRun:
    """A run that reports through `save_triage`, the way a real one now must."""
    return ScriptedRun(text="triaged", calls=(("save_triage", {"groups": list(groups)}),))


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
    return {"review-quality": findings_result(), "review-spec": findings_result()}


def opening(*tickets: dict[str, Any], **roles: ScriptedRun) -> dict[str, ScriptedRun]:
    """The interview and decompose calls every run below starts with."""
    return {
        "interview": ScriptedRun("noted", calls=(("save_spec", {"content": "# spec\n"}),)),
        "decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": list(tickets)}),)),
        **roles,
    }


def worktrees_of(ctx: RunContext) -> list[Path]:
    return [w.path for w in ctx.vcs.list_worktrees()]


def state_of(ctx: RunContext) -> RunFile:
    """The state document a finished run left in its store, read from outside."""
    return RunFile(StateFile(ctx.store))


def landed(repo: Path, *ticket_ids: str) -> bool:
    """Every named ticket's file is on the base branch — which is what merged means."""
    return all(
        (repo / f"{tid}.txt").read_text(encoding="utf-8") == f"{tid}\n" for tid in ticket_ids
    )


# -- one ticket, start to finish -----------------------------------------------


async def test_one_ticket_end_to_end(repo: Path) -> None:
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    runner = WritingAgentRunner(
        FakeAgentRunner(
            opening(
                ticket_json("T-01", "Add auth", ("auth.py",)),
                implement=ScriptedRun("done", calls=(("get_ticket", {}), ("read_spec", {}))),
                **clean_review(),
            )
        )
    )
    ctx = context(repo, agent=runner, terminal=terminal, max_concurrent=2)

    await run(ctx)

    assert landed(repo, "T-01")
    assert any("merged" in frame for frame in terminal.frames)
    assert worktrees_of(ctx) == [repo.resolve()]
    assert ctx.vcs.current_branch() == "feature"
    assert ctx.store.exists("spec.md")

    finished = state_of(ctx).load()
    assert [(t.id, t.status) for t in finished.tickets] == [("T-01", Status.MERGED)]
    assert finished.halt is None


async def test_the_run_writes_its_record_once_and_its_state_all_the_way_through(
    repo: Path,
) -> None:
    """The two documents a next process would have to work from.

    `run.json` is what the run was asked for and never moves. `state.json` is
    every ticket as the run last left it — including the `base_sha` each one was
    marked with, which is the only reason git can be asked whether a branch has
    been worked on at all.
    """
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    runner = WritingAgentRunner(
        FakeAgentRunner(
            opening(
                ticket_json("T-01", "Add auth", ("auth.py",)),
                ticket_json("T-02", "Add checks", ("check.py",), blocked_by=("T-01",)),
                implement=ScriptedRun("done"),
                **clean_review(),
            )
        )
    )
    ctx = context(repo, agent=runner, terminal=terminal, max_concurrent=2)

    await run(ctx)

    record = read_record(ctx.store)
    assert (record.workflow, record.label, record.base_branch) == ("tickets", LABEL, "feature")
    assert (record.request, record.project, record.max_concurrent) == (REQUEST, PROJECT, 2)

    finished = state_of(ctx).load()
    assert [t.id for t in finished.tickets] == ["T-01", "T-02"]
    assert all(t.status is Status.MERGED for t in finished.tickets)
    assert finished.ticket("T-02").blocked_by == ("T-01",)
    assert all(t.base_sha is not None for t in finished.tickets)


async def test_an_implementation_that_committed_nothing_stops_the_run(repo: Path) -> None:
    """An agent that changed nothing is a failure, not a diff to review.

    Nothing wraps the fake here, so "implement" leaves the worktree exactly as
    it found it. Without this the ticket would answer `IMPLEMENT` forever and
    the run would go round again on the same empty tree.
    """
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    ctx = context(
        repo,
        agent=FakeAgentRunner(
            opening(
                ticket_json("T-01", "Add auth", ("auth.py",)),
                implement=ScriptedRun("done"),
                **clean_review(),
            )
        ),
        terminal=terminal,
        max_concurrent=1,
    )

    with pytest.raises(HaltedError) as raised:
        await run(ctx)

    assert "T-01" in raised.value.halt.reason
    assert "unchanged" in raised.value.halt.reason
    assert raised.value.halt.resumable is False


async def test_a_halted_run_leaves_the_halt_in_the_state_document(repo: Path) -> None:
    """What `HaltedError` carries is also what a next process would read."""
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    fake = FakeAgentRunner(
        opening(
            ticket_json("T-01", "Add auth", ("auth.py",)),
            implement=ScriptedRun("done"),
            **clean_review(),
        )
    )
    ctx = context(
        repo,
        agent=WritingAgentRunner(fake),
        terminal=terminal,
        max_concurrent=1,
        build=(str(repo / "no-such-build"),),
    )

    with pytest.raises(HaltedError) as raised:
        await run(ctx)

    halt = state_of(ctx).load().halt
    assert halt == raised.value.halt
    assert halt is not None and halt.resumable is False


async def test_the_run_opens_one_session_and_swaps_the_screen_on_it(repo: Path) -> None:
    """The interview screen, then the dashboard, on the terminal entered once.

    The first frame is the blank one a session paints before any `show` — a run
    opens the display before it knows what its first stage looks like — and the
    label is on every frame after it.
    """
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    runner = WritingAgentRunner(
        FakeAgentRunner(
            opening(
                ticket_json("T-01", "Add auth", ("auth.py",)),
                implement=ScriptedRun("done"),
                **clean_review(),
            )
        )
    )
    ctx = context(repo, agent=runner, terminal=terminal, max_concurrent=1)

    await run(ctx)

    assert terminal.frames[0].strip() == ""
    assert LABEL in terminal.frames[1]
    assert "T-01" not in terminal.frames[1]
    assert any("T-01" in frame for frame in terminal.frames)


async def test_interview_activity_reaches_the_session_header(repo: Path) -> None:
    """The activity writer the interview is given lands on the header it draws."""
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    runner = WritingAgentRunner(
        FakeAgentRunner(
            opening(
                ticket_json("T-01", "Add auth", ("auth.py",)),
                implement=ScriptedRun("done"),
                **clean_review(),
            )
            | {
                "interview": ScriptedRun(
                    "noted",
                    activity=("read app/build.gradle.kts",),
                    calls=(("save_spec", {"content": "# Add auth\n"}),),
                )
            }
        )
    )
    ctx = context(repo, agent=runner, terminal=terminal, max_concurrent=1)

    await run(ctx)

    assert "read app/build.gradle.kts" not in terminal.frames[1]
    assert any(
        LABEL in frame and "read app/build.gradle.kts" in frame for frame in terminal.frames
    )


async def test_an_interview_that_saves_no_spec_stops_the_run(repo: Path) -> None:
    """A missing spec is a run that stopped short, not an empty one to carry on with."""
    feature(repo)
    terminal = HeadlessTerminal()
    runner = FakeAgentRunner({"interview": ScriptedRun("I have some thoughts.")})
    ctx = context(repo, agent=runner, terminal=terminal)

    with pytest.raises(InterviewIncompleteError):
        await run(ctx)

    assert ctx.vcs.branches(paths.branch_namespace(LABEL)) == ()


# -- preflight is a run's first line -------------------------------------------


async def test_a_run_refuses_to_start_before_it_touches_the_agent_or_the_terminal(
    repo: Path,
) -> None:
    """`preflight` is the first line of `run`, and its own cases live in
    `tests/runtime/test_context.py`. What is under test here is only that a
    workflow calls it before doing anything at all."""
    feature(repo)
    (repo / "dirty.txt").write_text("oops\n", encoding="utf-8")
    terminal = HeadlessTerminal()
    runner = FakeAgentRunner()
    ctx = context(repo, agent=runner, terminal=terminal)

    with pytest.raises(PreflightError):
        await run(ctx)

    assert runner.specs == []
    assert terminal.frames == []


# -- four tickets, one dependency edge -----------------------------------------


async def test_four_tickets_with_a_dependency_edge(repo: Path) -> None:
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    writer = WritingAgentRunner(
        FakeAgentRunner(
            opening(
                ticket_json("T1", "One", ("t1.py",)),
                ticket_json("T2", "Two", ("t2.py",)),
                ticket_json("T3", "Three", ("t3.py",)),
                ticket_json("T4", "Four", ("t4.py",), blocked_by=("T3",)),
                implement=ScriptedRun("done"),
                **clean_review(),
            )
        )
    )
    ctx = context(repo, agent=writer, terminal=terminal, max_concurrent=2)

    await run(ctx)

    assert landed(repo, "T1", "T2", "T3", "T4")
    assert worktrees_of(ctx) == [repo.resolve()]
    assert writer.implemented.index("T3") < writer.implemented.index("T4")


# -- a review that finds bugs --------------------------------------------------


async def test_review_findings_file_bugs_that_run_before_the_parent_merges(repo: Path) -> None:
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    fake = FakeAgentRunner(
        opening(
            ticket_json("T-01", "Add auth", ("auth.py",)),
            implement=ScriptedRun("done"),
            **{"review-quality": findings_result()},
            triage=groups_result(
                group(title="Fix S-1", deliverables=["Fix it."], findings=["S-1"]),
                group(title="Fix S-2", deliverables=["Fix it too."], findings=["S-2"]),
            ),
        )
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
    ctx = context(
        repo,
        agent=ScriptedByTicket(writer, "review-spec", spec_queues),
        terminal=terminal,
        max_concurrent=2,
    )

    await run(ctx)

    assert landed(repo, "T-01", "T-01-bug-1", "T-01-bug-2", "T-01-bug-3")
    assert worktrees_of(ctx) == [repo.resolve()]

    # The parent's second and third passes ran no implementation agent.
    assert writer.implemented.count("T-01") == 1

    # The second round of bugs did not reuse the first round's ids, and no
    # fourth round was ever opened.
    assert not (repo / "T-01-bug-4.txt").exists()

    # Triage ran once — for the two-high round. One high skips it, zero skips it.
    assert len([s for s in fake.specs if s.role == "triage"]) == 1


# -- a question mid-run --------------------------------------------------------


async def test_a_question_mid_run_reaches_the_terminal_and_shows_awaiting_input(
    repo: Path,
) -> None:
    feature(repo)
    question = AgentQuestion(
        title="Which token store?", options=(AgentOption("memory", "In process"),)
    )
    terminal = HeadlessTerminal(answers=[APPROVE, Answer("memory", was_free_text=False)])
    fake = FakeAgentRunner(
        opening(
            ticket_json("T-01", "Add auth", ("auth.py",)),
            implement=ScriptedRun("done", question=question),
            **clean_review(),
        )
    )
    ctx = context(repo, agent=WritingAgentRunner(fake), terminal=terminal, max_concurrent=1)

    await run(ctx)

    assert fake.answers == ["memory"]
    assert [q.title for q in terminal.questions] == [
        "Approve these 1 tickets?",
        "Which token store?",
    ]
    assert any("waiting for you" in frame for frame in terminal.frames)
    assert landed(repo, "T-01")


# -- a merge conflict halts the run --------------------------------------------


async def test_a_merge_conflict_halts_the_run_and_resuming_finishes_it(repo: Path) -> None:
    feature(repo)
    (repo / "shared.py").write_text("MODE = 'off'\n", encoding="utf-8")
    git(repo, "add", "shared.py")
    git(repo, "commit", "-m", "add shared.py")
    gate = Gate()
    terminal = GatedTerminal(gate, answers=[APPROVE])
    fake = FakeAgentRunner(
        opening(
            ticket_json("T1", "One", ("shared.py",)),
            ticket_json("T2", "Two", ("shared.py",)),
            ticket_json("T3", "Three", ("t3.py",), blocked_by=("T2",)),
            implement=ScriptedRun("done"),
            **clean_review(),
        )
    )
    overrides = {"T1": ("shared.py", "MODE = 'ours'\n"), "T2": ("shared.py", "MODE = 'theirs'\n")}
    ctx = context(
        repo,
        agent=WritingAgentRunner(fake, overrides),
        terminal=terminal,
        max_concurrent=2,
    )

    task = asyncio.create_task(run(ctx))
    async with asyncio.timeout(10.0):
        await gate.asked.wait()

    # Nothing new started: T3 waits on T2, which is stuck mid-merge.
    assert not (repo / "t3.txt").exists()
    assert not (repo / "T3.txt").exists()
    assert any("conflicts with the base branch" in frame for frame in terminal.frames)

    # A person resolves it in the repository root, the way the halt says to,
    # then presses enter — modelled as answering only once that has happened.
    (repo / "shared.py").write_text("MODE = 'resolved'\n", encoding="utf-8")
    ctx.vcs.commit_merge(repo, "merge the conflicted ticket")
    terminal.queue(Answer("continue", was_free_text=False))
    gate.ready.set()

    async with asyncio.timeout(10.0):
        await task

    assert landed(repo, "T3")
    assert (repo / "shared.py").read_text(encoding="utf-8") == "MODE = 'resolved'\n"
    assert worktrees_of(ctx) == [repo.resolve()]


async def test_a_non_resumable_merge_halt_ends_the_run_rather_than_hanging(repo: Path) -> None:
    """A halt nobody can press enter on still has to end the run.

    The build command does not exist, so the gate raises rather than returning
    a failed result — a halt this run marks non-resumable. Nothing will ever
    resume it, so the ticket waiting on that merge has to be told the answer is
    "never" instead of waiting for a resolution that cannot come. `run` reports
    it by raising, which is the whole of a run's exit status.
    """
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    fake = FakeAgentRunner(
        opening(
            ticket_json("T-01", "Add auth", ("auth.py",)),
            implement=ScriptedRun("done"),
            **clean_review(),
        )
    )
    ctx = context(
        repo,
        agent=WritingAgentRunner(fake),
        terminal=terminal,
        max_concurrent=1,
        build=(str(repo / "no-such-build"),),
    )

    async with asyncio.timeout(10.0):
        with pytest.raises(HaltedError) as raised:
            await run(ctx)

    assert raised.value.halt.resumable is False
    assert "T-01" in raised.value.halt.reason
    # The git merge itself landed; the gate behind it did not, so the ticket
    # never became `MERGED` and the last frame still shows it merging.
    assert not any("merged" in frame for frame in terminal.frames)
    assert "merging" in terminal.frames[-1]


async def test_a_halted_run_leaves_the_worktree_where_it_is(repo: Path) -> None:
    """Anything but a merged ticket keeps its tree, so a person can look at it."""
    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    fake = FakeAgentRunner(
        opening(
            ticket_json("T-01", "Add auth", ("auth.py",)),
            implement=ScriptedRun("done"),
            **clean_review(),
        )
    )
    ctx = context(
        repo,
        agent=WritingAgentRunner(fake),
        terminal=terminal,
        max_concurrent=1,
        build=(str(repo / "no-such-build"),),
    )

    with pytest.raises(HaltedError):
        await run(ctx)

    kept = [w.path for w in ctx.vcs.list_worktrees() if w.path != repo.resolve()]
    assert len(kept) == 1
    assert (kept[0] / "T-01.txt").exists()


# -- bug merges go through the queue too ---------------------------------------


async def test_a_bug_tickets_merge_goes_through_the_queue(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[MergeRequest] = []
    monkeypatch.setattr(workflow_module, "MergeQueue", recording_queue(seen))

    feature(repo)
    terminal = HeadlessTerminal(answers=[APPROVE])
    fake = FakeAgentRunner(
        opening(
            ticket_json("T-01", "Add auth", ("auth.py",)),
            implement=ScriptedRun("done"),
            **{"review-quality": findings_result()},
            triage=groups_result(group()),
        )
    )
    spec_queues = {
        "T-01": [findings_result(finding()), findings_result()],
        "T-01-bug-1": [findings_result()],
    }
    writer = WritingAgentRunner(fake)
    ctx = context(
        repo,
        agent=ScriptedByTicket(writer, "review-spec", spec_queues),
        terminal=terminal,
        max_concurrent=2,
    )

    await run(ctx)

    assert landed(repo, "T-01", "T-01-bug-1")
    assert [request.key for request in seen] == ["T-01-bug-1", "T-01"]

    bug_request = seen[0]
    assert bug_request.target == paths.branch(LABEL, "T-01")
    assert bug_request.cwd != ctx.project.repo.resolve()

    feature_request = seen[1]
    assert feature_request.target == "feature"
    assert feature_request.cwd == ctx.project.repo


async def test_a_bug_merge_conflict_halts_resumably_and_resuming_lets_the_parent_merge(
    repo: Path,
) -> None:
    feature(repo)
    (repo / "shared.py").write_text("MODE = 'off'\n", encoding="utf-8")
    git(repo, "add", "shared.py")
    git(repo, "commit", "-m", "add shared.py")
    gate = Gate()
    terminal = GatedTerminal(gate, answers=[APPROVE])
    fake = FakeAgentRunner(
        opening(
            ticket_json("T-01", "Add auth", ("auth.py",)),
            implement=ScriptedRun("done"),
            **{"review-quality": findings_result()},
            triage=groups_result(
                group(title="Fix S-1", deliverables=["Fix it."], findings=["S-1"]),
                group(title="Fix S-2", deliverables=["Fix it too."], findings=["S-2"]),
            ),
        )
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
    ctx = context(
        repo,
        agent=ScriptedByTicket(writer, "review-spec", spec_queues),
        terminal=terminal,
        max_concurrent=2,
    )

    task = asyncio.create_task(run(ctx))
    async with asyncio.timeout(10.0):
        await gate.asked.wait()

    # The conflict is in the parent's worktree — the repository root was never
    # touched by this merge — which is where the halt sends a person.
    parent_branch = paths.branch(LABEL, "T-01")
    parent_tree = next(w.path for w in ctx.vcs.list_worktrees() if w.branch == parent_branch)
    assert (repo / "shared.py").read_text(encoding="utf-8") == "MODE = 'off'\n"

    (parent_tree / "shared.py").write_text("MODE = 'resolved'\n", encoding="utf-8")
    ctx.vcs.commit_merge(parent_tree, "merge the conflicted bug")
    terminal.queue(Answer("continue", was_free_text=False))
    gate.ready.set()

    async with asyncio.timeout(10.0):
        await task

    assert landed(repo, "T-01")
    assert (repo / "shared.py").read_text(encoding="utf-8") == "MODE = 'resolved'\n"
    assert worktrees_of(ctx) == [repo.resolve()]


# -- decompose: propose, approve, revise, abort --------------------------------
#
# The loop's own shape, driven directly rather than through a whole run. What is
# under test is the loop, not the agent or the terminal.


async def decompose_with(ctx: RunContext, terminal: HeadlessTerminal) -> tuple[Ticket, ...]:
    """`decompose`, over a display opened the way `run` opens one.

    It returns nothing now — approval is a write to the state document, and the
    tickets it approved are read back out of it, which is the same thing the
    reactor does on its next turn round.
    """
    board = Board(started_at=NOW)
    state = RunFile(StateFile(ctx.store))
    async with live(terminal, board) as display:
        await workflow_module.decompose(ctx, display, state, board)
    return state.load().tickets


def a_decompose_run(*tickets: dict[str, Any], **extra: Any) -> ScriptedRun:
    return ScriptedRun("planned", calls=(("save_tickets", {"tickets": list(tickets)}),), **extra)


def spec_in(ctx: RunContext) -> RunContext:
    ctx.store.write("spec.md", "# spec\n")
    return ctx


async def test_approving_the_first_proposal_returns_it(repo: Path) -> None:
    terminal = HeadlessTerminal(answers=[APPROVE])
    ctx = spec_in(
        context(
            repo,
            agent=FakeAgentRunner(
                {"decompose": a_decompose_run(ticket_json("T-01", "Add auth", ("auth.py",)))}
            ),
            terminal=terminal,
        )
    )

    result = await decompose_with(ctx, terminal)

    assert [t.id for t in result] == ["T-01"]
    assert terminal.questions[0].title == "Approve these 1 tickets?"


async def test_a_revision_is_appended_to_the_spec_and_decompose_runs_again(repo: Path) -> None:
    terminal = HeadlessTerminal(
        answers=[
            Answer("split them into two tickets", was_free_text=True),
            APPROVE,
        ]
    )
    agent = FakeAgentRunner(
        [
            a_decompose_run(ticket_json("T-01", "Add auth", ("auth.py",))),
            ScriptedRun(
                "replanned",
                calls=(
                    ("read_spec", {}),
                    (
                        "save_tickets",
                        {
                            "tickets": [
                                ticket_json("T-01", "Add token issuing", ("auth.py",)),
                                ticket_json("T-02", "Add token checking", ("check.py",)),
                            ]
                        },
                    ),
                ),
            ),
        ]
    )
    ctx = spec_in(context(repo, agent=agent, terminal=terminal))

    result = await decompose_with(ctx, terminal)

    assert {t.id for t in result} == {"T-01", "T-02"}
    # The feedback went into the spec, and the second call read it back from
    # there rather than being handed it directly.
    assert "## Decomposition feedback" in ctx.store.read("spec.md")
    assert any("split them into two tickets" in result.text for result in agent.tool_results)


async def test_aborting_raises_without_returning_any_tickets(repo: Path) -> None:
    terminal = HeadlessTerminal(answers=[Answer("abort", was_free_text=False)])
    ctx = spec_in(
        context(
            repo,
            agent=FakeAgentRunner(
                {"decompose": a_decompose_run(ticket_json("T-01", "Add auth", ("auth.py",)))}
            ),
            terminal=terminal,
        )
    )

    with pytest.raises(DecomposeAbortedError):
        await decompose_with(ctx, terminal)


async def test_the_first_decompose_frame_has_no_tickets_and_no_activity_yet(repo: Path) -> None:
    """The state during the first seconds of the stage: a header, nothing else."""
    terminal = HeadlessTerminal(answers=[APPROVE], clock=lambda: NOW)
    ctx = spec_in(
        context(
            repo,
            agent=FakeAgentRunner(
                {"decompose": a_decompose_run(ticket_json("T-01", "Add auth", ("auth.py",)))}
            ),
            terminal=terminal,
        )
    )

    await decompose_with(ctx, terminal)

    first = next(frame for frame in terminal.frames if frame.strip())
    assert LABEL in first
    assert "T-01" not in first


async def test_decompose_activity_reaches_the_header_above_the_ticket_list(repo: Path) -> None:
    terminal = HeadlessTerminal(answers=[APPROVE], clock=lambda: NOW)
    ctx = spec_in(
        context(
            repo,
            agent=FakeAgentRunner(
                {
                    "decompose": a_decompose_run(
                        ticket_json("T-01", "Add auth", ("auth.py",)),
                        activity=("reading spec.md",),
                    )
                }
            ),
            terminal=terminal,
        )
    )

    await decompose_with(ctx, terminal)

    frame = next(f for f in terminal.frames if "T-01" in f)
    assert LABEL in frame
    assert "reading spec.md" in frame
    assert "T-01: Add auth" in frame


async def test_a_run_aborted_at_decompose_creates_no_worktrees_or_branches(repo: Path) -> None:
    feature(repo)
    terminal = HeadlessTerminal(answers=[Answer("abort", was_free_text=False)])
    ctx = context(
        repo,
        agent=FakeAgentRunner(opening(ticket_json("T-01", "Add auth", ("auth.py",)))),
        terminal=terminal,
        max_concurrent=2,
    )

    with pytest.raises(DecomposeAbortedError):
        await run(ctx)

    assert worktrees_of(ctx) == [repo.resolve()]
    assert ctx.vcs.branches(paths.branch_namespace(LABEL)) == ()


# -- this run's policy, as functions -------------------------------------------
#
# Every one of these takes the state document rather than an object, so each is
# exercised over a real store: what they write is what a next process would read.


def a_state(tmp_path: Path, *tickets: Ticket) -> RunFile:
    """A state document in a real store, holding `tickets`."""
    state = RunFile(StateFile(FileStore(tmp_path / "state")))
    state.write(with_tickets(Run(), tickets))
    return state


def a_ticket(ticket_id: str, parent: str | None = None) -> Ticket:
    return Ticket(
        id=ticket_id,
        title=f"Do {ticket_id}",
        status=Status.PENDING,
        deliverables=(f"{ticket_id}.py",),
        parent=parent,
    )


def a_loop(ctx: RunContext, display: Display, state: RunFile, board: Board) -> Loop:
    trees = Worktrees(
        ctx.vcs, trees_root=ctx.project.trees_root, project=ctx.project.name, label=ctx.label
    )
    return Loop(ctx, display, state, board, trees, MergeQueue(ctx.vcs, MergeConfig()))


async def test_resolve_holds_the_run_at_a_resumable_halt_and_then_retries(
    tmp_path: Path,
) -> None:
    """A conflict is shown, a person is asked, and the halt is cleared to retry."""
    terminal = HeadlessTerminal(answers=[Answer("continue", was_free_text=False)])
    state = a_state(tmp_path, a_ticket("T-01"))
    outcome = MergeOutcome("T-01", MergeStatus.CONFLICT, conflicted=("shared.py",))

    async with live(terminal, Board(started_at=NOW)) as display:
        decision = await resolve(display, state, LABEL, outcome)

    assert decision is MergeDecision.RETRY
    assert state.load().halt is None
    assert [q.title for q in terminal.questions] == [GATE]


async def test_resolve_stops_the_queue_on_a_halt_nobody_can_act_on(tmp_path: Path) -> None:
    """Nothing to press enter on, so the halt stays set and the queue ends."""
    terminal = HeadlessTerminal()
    state = a_state(tmp_path, a_ticket("T-01"))
    outcome = MergeOutcome("T-01", MergeStatus.VCS_ERROR, error="refusing to merge")

    async with live(terminal, Board(started_at=NOW)) as display:
        decision = await resolve(display, state, LABEL, outcome)

    assert decision is MergeDecision.STOP
    halt = state.load().halt
    assert halt is not None
    assert halt.resumable is False
    assert terminal.questions == []


def test_failed_names_the_node_and_is_not_resumable(tmp_path: Path) -> None:
    state = a_state(tmp_path, a_ticket("T-01"))

    failed(state, "T-01", RuntimeError("the agent blew up"))

    halt = state.load().halt
    assert halt is not None
    assert "T-01" in halt.reason
    assert "the agent blew up" in halt.reason
    assert halt.resumable is False


def test_failed_without_a_node_blames_the_run_itself(tmp_path: Path) -> None:
    state = a_state(tmp_path)

    failed(state, None, RuntimeError("the loop blew up"))

    halt = state.load().halt
    assert halt is not None
    assert "the run" in halt.reason
    assert halt.resumable is False


async def test_a_feature_branches_off_the_runs_base_and_a_bug_off_its_parent(
    repo: Path, tmp_path: Path
) -> None:
    """The one piece of ticket knowledge the worktree pool was not given."""
    feature(repo)
    terminal = HeadlessTerminal()
    ctx = context(repo, terminal=terminal)
    state = a_state(tmp_path, a_ticket("T-01"))
    state.update(lambda r: with_status(r, "T-01", Status.IN_PROGRESS))
    state.update(lambda r: with_status(r, "T-01", Status.IN_REVIEW))
    state.update(lambda r: with_bugs(r, "T-01", [a_ticket("T-01-bug-1", parent="T-01")]))
    run_state = state.load()

    async with live(terminal, Board(started_at=NOW)) as display:
        loop = a_loop(ctx, display, state, Board(started_at=NOW))
        assert base_for(loop, run_state.ticket("T-01")) == "feature"
        assert base_for(loop, run_state.ticket("T-01-bug-1")) == paths.branch(LABEL, "T-01")


class WatchingTerminal(HeadlessTerminal):
    """Reads the run's state at the moment a question is put to a person.

    What `asker` guarantees holds only while the question is open, which is
    gone by the time the call returns — so it is looked at from here. It is
    read off the document, because that is where a concurrent pass would read
    it too.
    """

    def __init__(self, watch: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._watch = watch

    def answer(self, question: Question) -> Answer:
        self._watch()
        return super().answer(question)


async def test_asker_suspends_only_its_own_ticket_and_puts_it_back(tmp_path: Path) -> None:
    """Two tickets in flight; a question moves the one it belongs to and no other."""
    seen: list[tuple[Status, Status]] = []
    state = a_state(tmp_path, a_ticket("T-01"), a_ticket("T-02"))

    def watch() -> None:
        run_state = state.load()
        seen.append((run_state.ticket("T-01").status, run_state.ticket("T-02").status))

    terminal = WatchingTerminal(watch, answers=[Answer("memory", was_free_text=False)])
    for ticket_id in ("T-01", "T-02"):
        state.update(partial(with_status, ticket_id=ticket_id, status=Status.IN_PROGRESS))

    async with live(terminal, Board(started_at=NOW)) as display:
        answer = await asker(state, display, "T-01")(
            AgentQuestion(title="Which store?", options=(AgentOption("memory", "In process"),))
        )

    assert answer == "memory"
    assert seen == [(Status.AWAITING_INPUT, Status.IN_PROGRESS)]
    assert state.load().ticket("T-01").status is Status.IN_PROGRESS
    assert state.load().ticket("T-01").resume_to is None
    assert terminal.questions[0].header == "T-01"


async def test_a_question_that_fails_still_gives_the_ticket_back(tmp_path: Path) -> None:
    """The resume is in a `finally`: a ticket is no longer waiting on anybody."""
    terminal = HeadlessTerminal()
    state = a_state(tmp_path, a_ticket("T-01"))
    state.update(lambda r: with_status(r, "T-01", Status.IN_PROGRESS))
    state.update(lambda r: with_status(r, "T-01", Status.IN_REVIEW))

    async with live(terminal, Board(started_at=NOW)) as display:
        ask = asker(state, display, "T-01")
        with pytest.raises(AssertionError):
            # `HeadlessTerminal` fails loudly with no scripted answer left.
            await ask(AgentQuestion(title="Which store?", options=()))

    assert state.load().ticket("T-01").status is Status.IN_REVIEW
