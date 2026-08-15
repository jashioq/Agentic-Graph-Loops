"""The decompose approval loop: propose, ask, revise, abort.

Previously only reachable through a full `Run.go()`. `FakeAgentRunner` and
`HeadlessTerminal` throughout — the loop's own shape is what is under test,
not the agent or the terminal.
"""

from pathlib import Path
from typing import Any

import pytest

from agl.core.terminal import Answer
from agl.runtime.dag import Dag
from agl.workflows.tickets.approval import Approval, DecomposeAbortedError
from agl.workflows.tickets.state import Live, RunState
from agl.workflows.tickets.wiring import Wiring
from tests.fakes import FakeAgentRunner, HeadlessTerminal, MemoryStore, ScriptedRun

LABEL = "add-auth"
NOW = 1_000.0


def wiring(agent: FakeAgentRunner, store: MemoryStore, live: Live | None = None) -> Wiring:
    state = RunState(label=LABEL, base_branch="feature", dag=Dag(), tickets={})
    watched = live if live is not None else Live(started_at=NOW)
    return Wiring(agent, store, Path("/repo"), state, LABEL, lambda: watched)


def ticket_json(id_: str, title: str, deliverables: tuple[str, ...]) -> dict[str, Any]:
    return {"id": id_, "title": title, "deliverables": list(deliverables)}


async def test_approving_the_first_proposal_returns_it() -> None:
    store = MemoryStore()
    store.write("spec.md", "# spec\n")
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    agent = FakeAgentRunner(
        {"decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),))}
    )
    terminal = HeadlessTerminal(answers=[Answer("approve", was_free_text=False)])
    approval = Approval(terminal, store, LABEL, wiring(agent, store))

    result = await approval.run()

    assert [t.id for t in result] == ["T-01"]
    assert terminal.questions[0].title == "Approve these 1 tickets?"


async def test_a_revision_is_appended_to_the_spec_and_decompose_runs_again() -> None:
    store = MemoryStore()
    store.write("spec.md", "# spec\n")
    first = [ticket_json("T-01", "Add auth", ("auth.py",))]
    second = [
        ticket_json("T-01", "Add token issuing", ("auth.py",)),
        ticket_json("T-02", "Add token checking", ("check.py",)),
    ]
    agent = FakeAgentRunner(
        [
            ScriptedRun("planned", calls=(("save_tickets", {"tickets": first}),)),
            ScriptedRun(
                "replanned",
                calls=(("read_spec", {}), ("save_tickets", {"tickets": second})),
            ),
        ]
    )
    terminal = HeadlessTerminal(
        answers=[
            Answer("split them into two tickets", was_free_text=True),
            Answer("approve", was_free_text=False),
        ]
    )
    approval = Approval(terminal, store, LABEL, wiring(agent, store))

    result = await approval.run()

    assert {t.id for t in result} == {"T-01", "T-02"}
    revised_read = agent.tool_results[-2]
    assert "split them into two tickets" in revised_read.text
    assert "## Decomposition feedback" in store.read("spec.md")


async def test_aborting_raises_without_returning_any_tickets() -> None:
    store = MemoryStore()
    store.write("spec.md", "# spec\n")
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    agent = FakeAgentRunner(
        {"decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),))}
    )
    terminal = HeadlessTerminal(answers=[Answer("abort", was_free_text=False)])
    approval = Approval(terminal, store, LABEL, wiring(agent, store))

    with pytest.raises(DecomposeAbortedError):
        await approval.run()


# -- the session header -----------------------------------------------------


def clocked_terminal(answers: list[Answer]) -> HeadlessTerminal:
    """A `HeadlessTerminal` whose timers all read from `NOW`, for repeatable frames."""
    return HeadlessTerminal(answers=answers, clock=lambda: NOW)


async def test_the_first_frame_has_no_tickets_and_no_activity_yet() -> None:
    """The state during the first seconds of every run: a header, nothing else."""
    store = MemoryStore()
    store.write("spec.md", "# spec\n")
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    agent = FakeAgentRunner(
        {"decompose": ScriptedRun("planned", calls=(("save_tickets", {"tickets": tickets}),))}
    )
    terminal = clocked_terminal([Answer("approve", was_free_text=False)])
    approval = Approval(terminal, store, LABEL, wiring(agent, store, Live(started_at=NOW)))

    await approval.run()

    first = terminal.frames[0]
    assert LABEL in first
    assert "T-01" not in first


async def test_decompose_activity_reaches_the_header_above_the_unchanged_ticket_list() -> None:
    store = MemoryStore()
    store.write("spec.md", "# spec\n")
    tickets = [ticket_json("T-01", "Add auth", ("auth.py",))]
    agent = FakeAgentRunner(
        {
            "decompose": ScriptedRun(
                "planned",
                activity=("reading spec.md",),
                calls=(("save_tickets", {"tickets": tickets}),),
            )
        }
    )
    terminal = clocked_terminal([Answer("approve", was_free_text=False)])
    approval = Approval(terminal, store, LABEL, wiring(agent, store, Live(started_at=NOW)))

    await approval.run()

    frame = next(f for f in terminal.frames if "T-01" in f)
    assert LABEL in frame
    assert "reading spec.md" in frame
    assert "T-01: Add auth" in frame
