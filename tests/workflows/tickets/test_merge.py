"""The serialized merge queue: one merge at a time, and a gate on the build.

Real git in `tmp_path`, and a stubbed build — running a real build command would
test the project's build rather than the queue, and would be slow enough that
nobody ran these. The stub is also where the timing lives: `Gate` suspends
*inside* the build, awaiting an `asyncio.Event` on the test's own loop, which
turns "the second merge had not started yet" into an assertion about the queue
instead of a bet on how fast the machine is. Nothing here sleeps to wait for
the queue.

The four resume states are each built by hand, with plain git commands standing
in for the person who went to the repository root and dealt with the halt.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from agl.core.command import ExecResult
from agl.core.vcs.impl.git import Git
from agl.workflows.tickets.merge import MergeQueue, MergeRequest
from agl.workflows.tickets.state import Halt
from tests.conftest import commit_file, git

MAIN = "main"
FILE = "mode.py"
TICKET = "T-01"

# Long enough that a loaded machine never trips it, short enough that a genuine
# hang fails the suite instead of stalling it.
TIMEOUT = 10.0


# -- git states -----------------------------------------------------------


def branch_with(repo: Path, branch: str, path: str, content: str) -> str:
    """A branch off `main` holding one commit that writes `content` to `path`.

    Made by checking out and coming back, so the root ends where it started —
    on the base branch, which is where every merge happens.
    """
    git(repo, "checkout", "-b", branch, MAIN)
    sha = commit_file(repo, path, content, f"{branch}: write {path}")
    git(repo, "checkout", MAIN)
    return sha


def two_rewrites(repo: Path) -> tuple[str, str]:
    """Two branches that rewrite the same line of `FILE` from the same base.

    The one thing git will not merge on its own, and the reason the second of
    them cannot land once the first has.
    """
    commit_file(repo, FILE, "base\n", f"add {FILE}")
    for branch, content in (("t1", "one\n"), ("t2", "two\n")):
        git(repo, "checkout", "-b", branch, MAIN)
        commit_file(repo, FILE, content, f"{branch}: rewrite {FILE}")
        git(repo, "checkout", MAIN)
    return "t1", "t2"


def already_collided(repo: Path) -> str:
    """Land one rewrite of `FILE` on `main`, and hand back the branch that cannot."""
    first, second = two_rewrites(repo)
    git(repo, "merge", "--no-ff", "--no-edit", first)
    return second


# -- the build gate -------------------------------------------------------


def passed() -> ExecResult:
    return ExecResult(argv=("build",), code=0, stdout="ok\n", stderr="")


def failed(code: int = 2, stdout: str = "boom\n", stderr: str = "") -> ExecResult:
    return ExecResult(argv=("build",), code=code, stdout=stdout, stderr=stderr)


class StubBuild:
    """A build that answers from a script, and counts how often it was asked.

    The last result repeats, so a test that only cares about "it keeps failing"
    hands over one failure.
    """

    def __init__(self, *results: ExecResult) -> None:
        self.results = list(results)
        self.calls = 0
        self.probed: list[object] = []
        self.probe: Callable[[], object] | None = None

    async def __call__(self) -> ExecResult:
        if self.probe is not None:
            self.probed.append(self.probe())
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


class RaisingBuild:
    """A build that raises instead of returning, until told the fix landed.

    Stands in for a build callable closing over something wrong at startup —
    a stale `config.toml`, a typo'd wrapper path — the case a `VcsError` catch
    cannot see because it is not `vcs` raising at all.
    """

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.ok = False
        self.calls = 0

    async def __call__(self) -> ExecResult:
        self.calls += 1
        if not self.ok:
            raise self.error
        return passed()


class Gate:
    """A build that stops the queue mid-request, where a test can look at it.

    Suspends on an `asyncio.Event` the test holds the other end of, so the
    test decides exactly when the build finishes — and, for `stop`, whether it
    ever does: cancelling the await is what a real build's process-kill stands
    in for here.
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.cancelled = False

    async def __call__(self) -> ExecResult:
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return passed()


# -- the harness ----------------------------------------------------------


class Calls:
    """What the queue reported, and a way to wait for the next thing it says."""

    def __init__(self) -> None:
        self.merged: list[str] = []
        self.halts: list[Halt] = []
        self.abandoned: list[str] = []
        self._rang = asyncio.Event()

    def on_merged(self, ticket_id: str) -> None:
        self.merged.append(ticket_id)
        self._rang.set()

    def on_halt(self, halt: Halt) -> None:
        self.halts.append(halt)
        self._rang.set()

    def on_abandoned(self, ticket_id: str) -> None:
        self.abandoned.append(ticket_id)
        self._rang.set()

    async def until(self, predicate: Callable[[], bool]) -> None:
        """Wait until `predicate` holds, woken by the callbacks themselves."""
        while not predicate():
            await asyncio.wait_for(self._rang.wait(), TIMEOUT)
            self._rang.clear()


async def settle() -> None:
    """Give the consumer every chance to do the thing it must not do.

    Every step the queue takes is either immediate or awaited, so yielding a
    few times is enough for a wrongly-started second merge to have happened by
    the time the assertion below it runs.
    """
    for _ in range(20):
        await asyncio.sleep(0)


@dataclass
class Harness:
    queue: MergeQueue
    vcs: Git
    repo: Path
    calls: Calls


@asynccontextmanager
async def running(
    repo: Path, build: Callable[[], Awaitable[ExecResult]] | None = None
) -> AsyncIterator[Harness]:
    """A queue with its consumer going, stopped and joined on the way out."""
    calls = Calls()
    vcs = Git(repo)
    queue = MergeQueue(
        vcs=vcs,
        repo=repo,
        base_branch=MAIN,
        build=build,
        on_merged=calls.on_merged,
        on_halt=calls.on_halt,
        on_abandoned=calls.on_abandoned,
    )
    task = asyncio.create_task(queue.run())
    try:
        yield Harness(queue=queue, vcs=vcs, repo=repo, calls=calls)
    finally:
        await queue.stop()
        await asyncio.wait_for(task, TIMEOUT)


async def halt_on_conflict(harness: Harness, branch: str) -> None:
    """Queue the request that cannot merge, and wait for the halt it raises."""
    harness.queue.put(MergeRequest(ticket_id=TICKET, branch=branch))
    await harness.calls.until(lambda: len(harness.calls.halts) == 1)


# -- a clean merge --------------------------------------------------------


async def test_a_clean_merge_with_no_build_gate_is_reported_merged(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    async with running(repo) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: h.calls.merged == [TICKET])

        assert h.vcs.is_ancestor("t1", MAIN) is True
        assert h.calls.halts == []


async def test_the_merge_lands_in_the_repository_root(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    async with running(repo) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: h.calls.merged == [TICKET])

        # The root is on the base branch and stays on it; the file the ticket
        # added is there afterwards.
        assert h.vcs.current_branch() == MAIN
        assert (repo / "a.py").read_text(encoding="utf-8") == "a\n"


async def test_a_clean_merge_runs_the_build_before_reporting_merged(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    build = StubBuild(passed())
    async with running(repo, build) as h:
        build.probe = lambda: list(h.calls.merged)
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: h.calls.merged == [TICKET])

        assert build.calls == 1
        # Nothing had been reported merged at the moment the build ran.
        assert build.probed == [[]]


async def test_a_failing_build_halts_with_the_code_and_the_tail(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    output = "\n".join(f"line {index}" for index in range(200)) + "\n"
    build = StubBuild(failed(code=2, stdout=output))
    async with running(repo, build) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)

        halt = h.calls.halts[0]
        assert TICKET in halt.reason
        assert "2" in halt.reason
        assert "line 199" in halt.detail
        assert "line 0\n" not in halt.detail
        assert h.calls.merged == []
        assert halt.resumable is True


async def test_a_failing_build_leaves_the_merge_commit_alone(repo: Path) -> None:
    # The queue does not unwind a merge git already made. A person decides
    # between fixing the build and taking the merge back out.
    branch_with(repo, "t1", "a.py", "a\n")
    async with running(repo, StubBuild(failed())) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)

        assert h.vcs.is_ancestor("t1", MAIN) is True


# -- a conflicting merge --------------------------------------------------


async def test_a_conflicting_merge_halts_naming_the_paths(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await halt_on_conflict(h, branch)

        halt = h.calls.halts[0]
        assert TICKET in halt.reason
        assert FILE in halt.detail
        assert h.calls.merged == []
        assert halt.resumable is True


async def test_a_conflicting_merge_is_left_in_progress(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await halt_on_conflict(h, branch)

        assert h.vcs.merge_in_progress(repo) is True
        assert h.vcs.unmerged_paths(repo) == (FILE,)


async def test_a_conflict_does_not_run_the_build(repo: Path) -> None:
    branch = already_collided(repo)
    build = StubBuild(passed())
    async with running(repo, build) as h:
        await halt_on_conflict(h, branch)

        assert build.calls == 0


# -- one at a time --------------------------------------------------------


async def test_the_second_request_does_not_begin_until_the_first_is_finished(
    repo: Path,
) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    branch_with(repo, "t2", "b.py", "b\n")
    gate = Gate()
    async with running(repo, gate) as h:
        h.queue.put(MergeRequest(ticket_id="T-01", branch="t1"))
        h.queue.put(MergeRequest(ticket_id="T-02", branch="t2"))
        await asyncio.wait_for(gate.started.wait(), TIMEOUT)
        await settle()

        # The first is still inside its build gate: it has not been reported,
        # and the second has not reached a build of its own.
        assert gate.calls == 1
        assert h.calls.merged == []

        gate.release.set()
        await h.calls.until(lambda: h.calls.merged == ["T-01", "T-02"])
        assert gate.calls == 2


async def test_the_first_merge_is_what_makes_the_second_one_conflict(repo: Path) -> None:
    # The ordering that makes serializing worth the trouble: both branches are
    # clean against the base they were cut from, and only one of them can land.
    first, second = two_rewrites(repo)
    async with running(repo) as h:
        h.queue.put(MergeRequest(ticket_id="T-01", branch=first))
        h.queue.put(MergeRequest(ticket_id="T-02", branch=second))
        await h.calls.until(lambda: len(h.calls.halts) == 1)

        assert h.calls.merged == ["T-01"]
        assert "T-02" in h.calls.halts[0].reason
        assert FILE in h.calls.halts[0].detail


# -- a halt stops at the head ---------------------------------------------


async def test_a_halt_stops_everything_behind_it(repo: Path) -> None:
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch=branch))
        h.queue.put(MergeRequest(ticket_id="T-02", branch="t3"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)
        await settle()

        assert h.calls.merged == []
        assert h.vcs.is_ancestor("t3", MAIN) is False


async def test_put_is_still_accepted_while_halted(repo: Path) -> None:
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        await halt_on_conflict(h, branch)

        h.queue.put(MergeRequest(ticket_id="T-02", branch="t3"))
        await settle()

        assert h.calls.merged == []
        assert len(h.calls.halts) == 1


# -- resume: what the person left behind ----------------------------------


async def test_resume_halts_again_when_the_conflict_is_untouched(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await halt_on_conflict(h, branch)

        h.queue.resume()
        await h.calls.until(lambda: len(h.calls.halts) == 2)

        assert h.calls.halts[1].detail == h.calls.halts[0].detail
        assert h.calls.merged == []
        assert h.vcs.merge_in_progress(repo) is True


async def test_resume_commits_a_resolution_that_was_only_staged(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await halt_on_conflict(h, branch)

        # Resolved and staged, not committed.
        (repo / FILE).write_text("resolved\n", encoding="utf-8")
        git(repo, "add", "--", FILE)

        h.queue.resume()
        await h.calls.until(lambda: h.calls.merged == [TICKET])

        assert h.vcs.merge_in_progress(repo) is False
        assert h.vcs.is_ancestor(branch, MAIN) is True
        assert (repo / FILE).read_text(encoding="utf-8") == "resolved\n"


async def test_resume_accepts_a_merge_the_person_committed_themselves(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await halt_on_conflict(h, branch)

        (repo / FILE).write_text("resolved\n", encoding="utf-8")
        git(repo, "add", "--", FILE)
        git(repo, "commit", "--no-edit")

        h.queue.resume()
        await h.calls.until(lambda: h.calls.merged == [TICKET])

        assert h.vcs.is_ancestor(branch, MAIN) is True
        assert len(h.calls.halts) == 1


async def test_resume_reports_an_aborted_merge_as_abandoned(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await halt_on_conflict(h, branch)

        git(repo, "merge", "--abort")

        h.queue.resume()
        await h.calls.until(lambda: h.calls.abandoned == [TICKET])

        assert h.calls.merged == []
        assert h.vcs.is_ancestor(branch, MAIN) is False


async def test_an_abandoned_ticket_does_not_hold_up_the_queue(repo: Path) -> None:
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        await halt_on_conflict(h, branch)
        h.queue.put(MergeRequest(ticket_id="T-02", branch="t3"))

        git(repo, "merge", "--abort")
        h.queue.resume()
        await h.calls.until(lambda: h.calls.merged == ["T-02"])

        assert h.calls.abandoned == [TICKET]


async def test_a_resumed_resolution_still_has_to_pass_the_build(repo: Path) -> None:
    branch = already_collided(repo)
    build = StubBuild(failed())
    async with running(repo, build) as h:
        await halt_on_conflict(h, branch)
        assert build.calls == 0

        (repo / FILE).write_text("resolved\n", encoding="utf-8")
        git(repo, "add", "--", FILE)
        h.queue.resume()
        await h.calls.until(lambda: len(h.calls.halts) == 2)

        assert build.calls == 1
        assert h.calls.merged == []


# -- resume: after a build failure ----------------------------------------


async def test_resume_merges_when_the_build_now_passes(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    build = StubBuild(failed(), passed())
    async with running(repo, build) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)

        h.queue.resume()
        await h.calls.until(lambda: h.calls.merged == [TICKET])

        assert build.calls == 2
        assert len(h.calls.halts) == 1


async def test_resume_halts_again_when_the_build_still_fails(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    build = StubBuild(failed(code=3))
    async with running(repo, build) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)

        h.queue.resume()
        await h.calls.until(lambda: len(h.calls.halts) == 2)

        assert build.calls == 2
        assert h.calls.merged == []


async def test_a_build_failure_still_blocks_what_is_behind_it(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo, StubBuild(failed())) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        h.queue.put(MergeRequest(ticket_id="T-02", branch="t3"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)
        await settle()

        assert h.calls.merged == []
        assert h.vcs.is_ancestor("t3", MAIN) is False


# -- a branch git cannot find ---------------------------------------------


async def test_a_branch_that_does_not_resolve_halts_and_is_not_resumable(
    repo: Path,
) -> None:
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="no-such-branch"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)

        assert TICKET in h.calls.halts[0].reason
        assert h.calls.halts[0].resumable is False

        # The consumer is still going, but nothing about a branch that does
        # not exist is fixed by pressing enter, so resume does nothing and
        # what is behind it stays queued rather than landing.
        h.queue.resume()
        await settle()
        assert h.calls.abandoned == []
        assert len(h.calls.halts) == 1

        h.queue.put(MergeRequest(ticket_id="T-02", branch="t3"))
        await settle()
        assert h.calls.merged == []


# -- an exception escaping the build ---------------------------------------


async def test_a_raising_build_halts_rather_than_killing_the_consumer(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    build = RaisingBuild(FileNotFoundError(2, "No such file or directory", "./gradlew-x"))
    async with running(repo, build) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)

        assert TICKET in h.calls.halts[0].reason
        assert h.calls.merged == []

        # The consumer is still alive: `stop` is not left waiting on a dead
        # task. A resume is refused outright rather than retried, because the
        # build callable's brokenness is baked in until the process restarts.
        h.queue.resume()
        await settle()
        assert len(h.calls.halts) == 1
        assert h.calls.merged == []


async def test_a_request_behind_a_raising_build_stays_queued(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    branch_with(repo, "t2", "b.py", "b\n")
    build = RaisingBuild(FileNotFoundError(2, "No such file or directory", "./gradlew-x"))
    async with running(repo, build) as h:
        h.queue.put(MergeRequest(ticket_id="T-01", branch="t1"))
        h.queue.put(MergeRequest(ticket_id="T-02", branch="t2"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)
        await settle()

        assert build.calls == 1
        assert h.calls.merged == []
        assert h.vcs.is_ancestor("t2", MAIN) is False


async def test_a_raising_on_merged_halts_rather_than_killing_the_consumer(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    calls = Calls()

    def on_merged(ticket_id: str) -> None:
        raise RuntimeError("dashboard write failed")

    queue = MergeQueue(
        vcs=Git(repo),
        repo=repo,
        base_branch=MAIN,
        build=None,
        on_merged=on_merged,
        on_halt=calls.on_halt,
        on_abandoned=calls.on_abandoned,
    )
    task = asyncio.create_task(queue.run())
    try:
        queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await calls.until(lambda: len(calls.halts) == 1)

        assert TICKET in calls.halts[0].reason
    finally:
        await queue.stop()
        await asyncio.wait_for(task, TIMEOUT)


async def test_a_raising_build_halt_is_not_resumable(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    build = RaisingBuild(FileNotFoundError(2, "No such file or directory", "./gradlew-x"))
    async with running(repo, build) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)

        assert h.calls.halts[0].resumable is False


async def test_a_fixed_build_merges_the_ticket_after_the_process_restarts(repo: Path) -> None:
    # Flipping the closed-over build mid-run does not help, because a
    # non-resumable halt refuses to look again. What fixes it is what the halt
    # says: restart the process. A fresh queue is that restart, and the merge
    # `vcs.merge` already made when the build first raised is why the same
    # request lands clean the second time round.
    branch_with(repo, "t1", "a.py", "a\n")
    build = RaisingBuild(FileNotFoundError(2, "No such file or directory", "./gradlew-x"))
    async with running(repo, build) as h:
        h.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h.calls.until(lambda: len(h.calls.halts) == 1)

    async with running(repo, StubBuild(passed())) as h2:
        h2.queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))
        await h2.calls.until(lambda: h2.calls.merged == [TICKET])


# -- stopping -------------------------------------------------------------


async def test_stop_ends_the_consumer_with_work_still_queued(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    calls = Calls()
    queue = MergeQueue(
        vcs=Git(repo),
        repo=repo,
        base_branch=MAIN,
        build=None,
        on_merged=calls.on_merged,
        on_halt=calls.on_halt,
        on_abandoned=calls.on_abandoned,
    )
    queue.put(MergeRequest(ticket_id=TICKET, branch="t1"))

    await queue.stop()
    await asyncio.wait_for(asyncio.create_task(queue.run()), TIMEOUT)

    assert calls.merged == []


async def test_stop_cancels_an_in_flight_build_rather_than_waiting_for_it(
    repo: Path,
) -> None:
    # `stop` used to let whatever was in flight finish; now it cancels it, so
    # Ctrl-C during a slow build does not look hung for however long is left.
    branch_with(repo, "t1", "a.py", "a\n")
    branch_with(repo, "t2", "b.py", "b\n")
    gate = Gate()
    calls = Calls()
    queue = MergeQueue(
        vcs=Git(repo),
        repo=repo,
        base_branch=MAIN,
        build=gate,
        on_merged=calls.on_merged,
        on_halt=calls.on_halt,
        on_abandoned=calls.on_abandoned,
    )
    queue.put(MergeRequest(ticket_id="T-01", branch="t1"))
    queue.put(MergeRequest(ticket_id="T-02", branch="t2"))
    consumer = asyncio.create_task(queue.run())

    await asyncio.wait_for(gate.started.wait(), TIMEOUT)
    await asyncio.wait_for(queue.stop(), TIMEOUT)
    await asyncio.wait_for(consumer, TIMEOUT)

    assert gate.calls == 1
    assert gate.cancelled is True
    assert calls.merged == []
    assert calls.halts == []
    # `stop` does not unwind what git already did: `t1`'s merge commit stands,
    # only the build gate on top of it was cut short. `t2` was never reached.
    assert Git(repo).is_ancestor("t1", MAIN) is True
    assert Git(repo).is_ancestor("t2", MAIN) is False


async def test_stop_returns_even_though_nothing_is_consuming(repo: Path) -> None:
    calls = Calls()
    queue = MergeQueue(
        vcs=Git(repo),
        repo=repo,
        base_branch=MAIN,
        build=None,
        on_merged=calls.on_merged,
        on_halt=calls.on_halt,
        on_abandoned=calls.on_abandoned,
    )
    await asyncio.wait_for(queue.stop(), TIMEOUT)


# -- the request ----------------------------------------------------------


def test_a_request_is_frozen() -> None:
    request = MergeRequest(ticket_id=TICKET, branch="t1")
    with pytest.raises(AttributeError):
        request.branch = "other"  # type: ignore[misc]
