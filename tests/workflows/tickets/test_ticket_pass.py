"""One ticket's pass, as the two decisions it makes that are not the steps.

`base_for` (a bug branches off its parent, a feature off the run's base) and
`asker` (a question suspends its own ticket and no other). Both take the state
document rather than an object, so each is exercised over a real store: what
they write is what a next process would read. The steps themselves are covered
end to end in `test_workflow.py`.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import AgentOption, AgentQuestion
from agl.core.terminal import Answer, Question
from agl.runtime import paths
from agl.runtime.context import RunContext
from agl.runtime.display import Board, Display, live
from agl.runtime.merge import MergeConfig, MergeQueue
from agl.runtime.worktrees import Worktrees
from agl.workflows.tickets.documents.state_document import StateDocument
from agl.workflows.tickets.models import Status
from agl.workflows.tickets.run_state import with_bugs, with_status
from agl.workflows.tickets.ticket_pass import Loop, asker, base_for
from tests.fakes import HeadlessTerminal
from tests.runtime.conftest import LABEL, context, feature
from tests.workflows.tickets.conftest import NOW, a_state, a_ticket

# -- base_for ---------------------------------------------------------------


def a_loop(ctx: RunContext, display: Display, state: StateDocument, board: Board) -> Loop:
    trees = Worktrees(
        ctx.vcs, trees_root=ctx.project.trees_root, project=ctx.project.name, label=ctx.label
    )
    return Loop(ctx, display, state, board, trees, MergeQueue(ctx.vcs, MergeConfig()))


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


# -- asker ------------------------------------------------------------------


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
