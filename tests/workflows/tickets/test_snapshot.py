"""The state document: the round trip, what it refuses, and what it tolerates.

The store is real — a `FileStore` in `tmp_path` — because the whole point of the
document is that it survives the process, and a dict standing in for the disk
would not exercise the one thing worth exercising. What a person with an editor
can do to the file is done here by writing the file.
"""

import json
from pathlib import Path

import pytest

from agl.core.store.impl.file_store import FileStore
from agl.runtime.record import STATE_KEY, VERSION, StateFile
from agl.workflows.tickets.models import Status, Ticket
from agl.workflows.tickets.snapshot import RunFile, from_json, to_json
from agl.workflows.tickets.state import (
    Halt,
    InvalidStateError,
    Run,
    with_bugs,
    with_halt,
    with_status,
    with_tickets,
)

# -- building a run -------------------------------------------------------


def feature(ticket_id: str, *blocked_by: str) -> Ticket:
    return Ticket(
        id=ticket_id,
        title=f"Do {ticket_id}",
        status=Status.PENDING,
        deliverables=(f"{ticket_id}.py",),
        blocked_by=blocked_by,
    )


def bug(ticket_id: str, parent: str) -> Ticket:
    return Ticket(
        id=ticket_id,
        title=f"Fix {ticket_id}",
        status=Status.PENDING,
        deliverables=("the finding",),
        parent=parent,
    )


def a_run() -> Run:
    """A run mid-flight: two features, a bug filed against one, and a halt."""
    run = with_tickets(Run(), (feature("T-01"), feature("T-02", "T-01")))
    run = with_status(with_status(run, "T-01", Status.IN_PROGRESS), "T-01", Status.IN_REVIEW)
    run = with_bugs(run, "T-01", (bug("T-01-bug-1", "T-01"),))
    return with_halt(run, Halt("T-01 conflicts", "resolve auth.py", resumable=True))


@pytest.fixture
def state(tmp_path: Path) -> RunFile:
    """A state document in a real store, with nothing written to it yet."""
    return RunFile(StateFile(FileStore(tmp_path / "run")))


def written(tmp_path: Path) -> dict[str, object]:
    return dict(json.loads((tmp_path / "run" / STATE_KEY).read_text(encoding="utf-8")))


def corrupt(tmp_path: Path, text: str) -> None:
    (tmp_path / "run" / STATE_KEY).write_text(text, encoding="utf-8")


# -- the round trip -------------------------------------------------------


def test_a_run_survives_the_round_trip_whole() -> None:
    run = a_run()

    assert from_json(to_json(run)) == run


def test_an_empty_run_round_trips() -> None:
    assert from_json(to_json(Run())) == Run()


def test_every_field_of_every_ticket_comes_back() -> None:
    run = Run(
        tickets=(
            Ticket(
                id="T-01",
                title="Add auth",
                status=Status.AWAITING_INPUT,
                deliverables=("auth.py", "tests"),
                blocked_by=(),
                parent=None,
                review_round=2,
                resume_to=Status.IN_REVIEW,
                base_sha="abc123",
            ),
        )
    )

    assert from_json(to_json(run)).tickets[0] == run.tickets[0]


def test_bug_parentage_and_its_edges_survive() -> None:
    """`blocked_by` is the graph, so the parent's edges are the only record of it."""
    run = from_json(to_json(a_run()))

    assert run.ticket("T-01-bug-1").parent == "T-01"
    assert run.ticket("T-01").blocked_by == ("T-01-bug-1",)
    assert run.ticket("T-01").review_round == 1


def test_a_halt_survives_with_all_three_of_its_fields() -> None:
    run = with_halt(Run(), Halt("build failed", "line 40", resumable=False))

    assert from_json(to_json(run)).halt == Halt("build failed", "line 40", resumable=False)


def test_no_halt_round_trips_as_no_halt() -> None:
    assert from_json(to_json(Run())).halt is None


def test_the_document_names_every_field_even_when_it_is_empty() -> None:
    """A field that appears only sometimes is a default a reader has to remember."""
    payload = to_json(with_tickets(Run(), (feature("T-01"),)))

    assert set(payload["tickets"][0]) == {
        "id",
        "title",
        "status",
        "deliverables",
        "blocked_by",
        "parent",
        "review_round",
        "resume_to",
        "base_sha",
    }


# -- what a hand-edited document is refused for ---------------------------


def test_an_unknown_status_is_named_in_the_error() -> None:
    payload = to_json(with_tickets(Run(), (feature("T-01"),)))
    payload["tickets"][0]["status"] = "nearly-done"

    with pytest.raises(InvalidStateError, match="nearly-done"):
        from_json(payload)


def test_an_unknown_field_is_refused_rather_than_ignored() -> None:
    payload = to_json(with_tickets(Run(), (feature("T-01"),)))
    payload["tickets"][0]["priority"] = "high"

    with pytest.raises(InvalidStateError, match="priority"):
        from_json(payload)


def test_a_missing_title_is_refused() -> None:
    payload = to_json(with_tickets(Run(), (feature("T-01"),)))
    del payload["tickets"][0]["title"]

    with pytest.raises(InvalidStateError, match="title"):
        from_json(payload)


def test_the_documents_own_rules_are_re_checked_on_the_way_in() -> None:
    """`check` has the last word: a parseable document can still be an impossible run."""
    payload = to_json(with_tickets(Run(), (feature("T-01"),)))
    payload["tickets"][0]["blocked_by"] = ["T-99"]

    with pytest.raises(InvalidStateError, match="T-99"):
        from_json(payload)


def test_tickets_must_be_an_array() -> None:
    with pytest.raises(InvalidStateError, match="array"):
        from_json({"tickets": {"T-01": {}}})


# -- the file -------------------------------------------------------------


def test_a_run_that_has_written_nothing_loads_as_an_empty_run(state: RunFile) -> None:
    assert state.load() == Run()


def test_write_then_load_comes_back_equal(state: RunFile, tmp_path: Path) -> None:
    run = a_run()

    state.write(run)

    assert state.load() == run
    assert written(tmp_path)["version"] == VERSION


def test_update_reads_decides_and_writes_in_one_call(state: RunFile) -> None:
    state.write(with_tickets(Run(), (feature("T-01"),)))

    returned = state.update(lambda run: with_status(run, "T-01", Status.IN_PROGRESS))

    assert returned.ticket("T-01").status is Status.IN_PROGRESS
    assert state.load() == returned


def test_a_document_from_another_version_is_refused(state: RunFile, tmp_path: Path) -> None:
    state.write(Run())
    corrupt(tmp_path, json.dumps({"version": VERSION + 1, "tickets": [], "halt": None}))

    with pytest.raises(InvalidStateError, match="version"):
        state.load()


def test_a_document_that_is_not_json_is_refused(state: RunFile, tmp_path: Path) -> None:
    state.write(Run())
    corrupt(tmp_path, "{not json at all")

    with pytest.raises(InvalidStateError):
        state.load()


def test_latest_keeps_drawing_the_last_good_run_over_a_broken_file(
    state: RunFile, tmp_path: Path
) -> None:
    """A frame is painted four times a second and must not raise over an edit
    somebody is halfway through."""
    run = a_run()
    state.write(run)
    assert state.latest() == run

    corrupt(tmp_path, '{"version": 1, "tickets": [{"id":')

    assert state.latest() == run
    with pytest.raises(InvalidStateError):
        state.load()


def test_latest_picks_the_new_run_up_again_once_the_file_parses(
    state: RunFile, tmp_path: Path
) -> None:
    state.write(a_run())
    corrupt(tmp_path, "{oops")
    assert state.latest() == a_run()

    state.write(Run())

    assert state.latest() == Run()


def test_latest_on_a_file_that_never_parsed_is_an_empty_run(
    state: RunFile, tmp_path: Path
) -> None:
    state.write(Run())
    corrupt(tmp_path, "{oops")

    assert state.latest() == Run()
