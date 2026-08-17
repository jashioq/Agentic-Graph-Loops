"""Everything about a run stopping: the halt, what makes one, and who acts on it.

`Halt` and `halt_for` are values, tested as such. `resolve` and `failed` take
the state document rather than an object, so each is exercised over a real
store: what they write is what a next process would read.
"""

from pathlib import Path
from typing import Any

from agl.core.command import ExecResult
from agl.core.terminal import Answer
from agl.runtime.display import Board, live
from agl.runtime.merge import MergeDecision, MergeOutcome, MergeStatus
from agl.workflows.tickets.errors import Halt
from agl.workflows.tickets.halting import TAIL_LINES, failed, halt_for, resolve
from tests.fakes import HeadlessTerminal
from tests.runtime.conftest import LABEL
from tests.workflows.tickets.conftest import GATE, NOW, a_state, a_ticket

# -- the halt ---------------------------------------------------------------


def test_halt_carries_a_reason_and_a_detail() -> None:
    halt = Halt(reason="budget", detail="ran out at $4.10")

    assert halt.reason == "budget"
    assert halt.detail == "ran out at $4.10"


def test_halt_defaults_to_resumable() -> None:
    assert Halt(reason="budget").resumable is True
    assert Halt(reason="budget", resumable=False).resumable is False


# -- halt_for ---------------------------------------------------------------


def outcome(status: MergeStatus, **overrides: Any) -> MergeOutcome:
    return MergeOutcome(key="T-01", status=status, **overrides)


def test_a_conflict_is_resumable_and_names_what_to_resolve() -> None:
    halt = halt_for(outcome(MergeStatus.CONFLICT, conflicted=("a.py", "b.py")))

    assert halt.resumable is True
    assert "T-01" in halt.reason
    assert "a.py, b.py" in halt.detail


def test_a_failed_build_is_resumable_and_carries_the_tail_of_its_output() -> None:
    lines = "\n".join(f"line {index}" for index in range(1, 41))
    build = ExecResult(argv=("build",), code=1, stdout=lines, stderr="", timed_out=False)

    halt = halt_for(outcome(MergeStatus.BUILD_FAILED, build=build))

    assert halt.resumable is True
    assert "exit 1" in halt.reason
    assert halt.detail.splitlines() == [f"line {index}" for index in range(41 - TAIL_LINES, 41)]


def test_a_failed_build_carries_both_streams() -> None:
    """Which stream carries the diagnosis is the build tool's choice, not this run's."""
    build = ExecResult(
        argv=("build",), code=1, stdout="compiling\n", stderr="error: boom\n", timed_out=False
    )

    halt = halt_for(outcome(MergeStatus.BUILD_FAILED, build=build))

    assert halt.detail.splitlines() == ["compiling", "error: boom"]


def test_a_failed_build_says_so_when_it_timed_out_instead() -> None:
    build = ExecResult(argv=("build",), code=-9, stdout="", stderr="killed", timed_out=True)

    halt = halt_for(outcome(MergeStatus.BUILD_FAILED, build=build))

    assert "timed out" in halt.reason
    assert "killed" in halt.detail


def test_a_vcs_error_cannot_be_resumed() -> None:
    halt = halt_for(outcome(MergeStatus.VCS_ERROR, error="no such branch"))

    assert halt.resumable is False
    assert halt.detail == "no such branch"


def test_anything_else_cannot_be_resumed_either() -> None:
    halt = halt_for(outcome(MergeStatus.ERROR, error="FileNotFoundError: gradlew"))

    assert halt.resumable is False
    assert "gradlew" in halt.reason


# -- resolve ----------------------------------------------------------------


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


# -- failed -----------------------------------------------------------------


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
