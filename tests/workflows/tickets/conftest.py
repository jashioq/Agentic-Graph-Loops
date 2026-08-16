"""What a tickets test is assembled from, above the harness in `tests/runtime`.

`context(repo)` is still where a `RunContext` comes from. What is here is
everything on top of it that more than one module in this package needs: the
runners that stand in for what a real agent does to a worktree, a terminal that
holds the resume prompt where a person would stand, the scripted payloads a run
is driven with, and the readers that say what a finished run left behind.

Nothing here reaches into a running loop. What a run did is read off what it
left behind: the files that landed on the base branch, the worktrees it
released, the frames it painted, and the documents in its store.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agl.core.agent import AgentResult, AgentRunner, AgentSpec
from agl.core.store.impl.file_store import FileStore
from agl.core.terminal import Answer, LiveSession, Question, Screen
from agl.runtime.context import RunContext
from agl.runtime.merge import MergeOutcome, MergeQueue, MergeRequest
from agl.runtime.record import StateFile
from agl.workflows.tickets.documents.state_document import StateDocument
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.run_state import Run, with_tickets
from tests.fakes import HeadlessTerminal, ScriptedRun, _HeadlessSession

__all__ = [
    "APPROVE",
    "GATE",
    "NOW",
    "Gate",
    "GatedTerminal",
    "ScriptedByTicket",
    "WritingAgentRunner",
    "a_state",
    "a_ticket",
    "clean_review",
    "finding",
    "findings_result",
    "group",
    "groups_result",
    "landed",
    "opening",
    "recording_queue",
    "state_of",
    "ticket_json",
    "worktrees_of",
]

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


# -- reading what a run left behind --------------------------------------------


def worktrees_of(ctx: RunContext) -> list[Path]:
    return [w.path for w in ctx.vcs.list_worktrees()]


def state_of(ctx: RunContext) -> StateDocument:
    """The state document a finished run left in its store, read from outside."""
    return StateDocument(StateFile(ctx.store))


def landed(repo: Path, *ticket_ids: str) -> bool:
    """Every named ticket's file is on the base branch — which is what merged means."""
    return all(
        (repo / f"{tid}.txt").read_text(encoding="utf-8") == f"{tid}\n" for tid in ticket_ids
    )


# -- a state document to drive a policy function over --------------------------


def a_state(tmp_path: Path, *tickets: Ticket) -> StateDocument:
    """A state document in a real store, holding `tickets`."""
    state = StateDocument(StateFile(FileStore(tmp_path / "state")))
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
