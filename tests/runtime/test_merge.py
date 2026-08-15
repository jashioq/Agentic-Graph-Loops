"""The serialized merge queue: one merge at a time, and a gate on the build.

Real git in `tmp_path`, and a stubbed build — running a real build command would
test the project's build rather than the queue, and would be slow enough that
nobody ran these. The stub is also where the timing lives: `Gate` suspends
*inside* the build, awaiting an `asyncio.Event` on the test's own loop, which
turns "the second merge had not started yet" into an assertion about the queue
instead of a bet on how fast the machine is. Nothing here sleeps to wait for
the queue.

`Policy` stands in for the workflow on the other end of `MergeConfig.resolve`:
it records every outcome the queue asked about and holds the consumer there
until the test answers, which is what a person at a dashboard is. The four
states a `RETRY` can find are each built by hand, with plain git commands
standing in for the person who went to the repository root and dealt with it.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agl.core.command import ExecResult
from agl.core.vcs.impl.git import Git
from agl.runtime.merge import (
    MergeConfig,
    MergeDecision,
    MergeOutcome,
    MergeQueue,
    MergeRequest,
    MergeStatus,
)
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


# -- a target other than the base ------------------------------------------


def worktree_on(vcs: Git, repo: Path, branch: str, base: str = MAIN) -> Path:
    """A second worktree, holding a new branch cut from `base`.

    Stands in for a kept-alive worktree: a tree, other than the repository
    root, that already has some branch checked out — the tree a request whose
    target is not the base branch merges into.
    """
    tree = repo.parent / f"{branch}-tree"
    return vcs.add_worktree(tree, branch, base).path


def branch_off(repo: Path, base: str, branch: str, path: str, content: str) -> str:
    """A branch off `base`, holding one commit that writes `content` to `path`.

    Like `branch_with`, but `base` need not be `main` — it may be a branch
    checked out in another worktree, which `checkout -b` can still cut a new
    branch from without touching that worktree.
    """
    git(repo, "checkout", "-b", branch, base)
    sha = commit_file(repo, path, content, f"{branch}: write {path}")
    git(repo, "checkout", MAIN)
    return sha


def collided_off(repo: Path, worktree: Path, base: str) -> str:
    """Land one rewrite of `FILE` on `base` (through `worktree`), and hand
    back the branch that cannot: `already_collided`, off a target that is not
    `main`.
    """
    commit_file(worktree, FILE, "base\n", f"add {FILE}")
    branch_off(repo, base, "b1", FILE, "one\n")
    branch_off(repo, base, "b2", FILE, "two\n")
    git(worktree, "merge", "--no-ff", "--no-edit", "b1")
    return "b2"


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


# -- the workflow on the other end ----------------------------------------


class Policy:
    """The `resolve` the queue asks, scripted and observable.

    `decisions` are answered in order and the last one repeats, so a test that
    only cares about "it keeps retrying" hands over one `RETRY`. With no script
    at all it says `STOP`, which is what the real default does.

    `hold` is what makes it a person rather than a rule: while it is clear the
    consumer is suspended inside `resolve`, which is exactly where a run sits
    while somebody deals with a conflict.
    """

    def __init__(self, *decisions: MergeDecision, hold: bool = False) -> None:
        self.decisions = list(decisions)
        self.asked: list[MergeOutcome] = []
        self.released = asyncio.Event()
        if not hold:
            self.released.set()
        self.rang = asyncio.Event()
        self.raises: Exception | None = None

    async def __call__(self, outcome: MergeOutcome) -> MergeDecision:
        self.asked.append(outcome)
        self.rang.set()
        if self.raises is not None:
            raise self.raises
        await self.released.wait()
        return self._decision()

    def answer(self, decision: MergeDecision) -> None:
        """Let the held ask through with `decision`, and hold the next one."""
        self.decisions = [decision]
        self.released.set()
        self.released = asyncio.Event()

    def _decision(self) -> MergeDecision:
        if not self.decisions:
            return MergeDecision.STOP
        return self.decisions[min(len(self.asked) - 1, len(self.decisions) - 1)]


# -- the harness ----------------------------------------------------------


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
    """A running queue, everything it was asked, and everything it answered."""

    queue: MergeQueue
    vcs: Git
    repo: Path
    policy: Policy
    rang: asyncio.Event
    done: list[MergeOutcome] = field(default_factory=list)
    tasks: list[asyncio.Task[None]] = field(default_factory=list)

    def submit(self, key: str, branch: str, target: str = MAIN, cwd: Path | None = None) -> None:
        """Submit a request and record its outcome whenever it arrives.

        A task, because `submit` is the caller's own await — in a run it is the
        ticket's body parked until its merge is dealt with.
        """
        request = MergeRequest(
            key=key, branch=branch, target=target, cwd=self.repo if cwd is None else cwd
        )

        async def wait() -> None:
            outcome = await self.queue.submit(request)
            self.done.append(outcome)
            self.rang.set()

        self.tasks.append(asyncio.create_task(wait()))

    @property
    def merged(self) -> list[str]:
        return [o.key for o in self.done if o.status is MergeStatus.MERGED]

    @property
    def abandoned(self) -> list[str]:
        return [o.key for o in self.done if o.status is MergeStatus.ABANDONED]

    @property
    def stopped(self) -> list[str]:
        return [o.key for o in self.done if o.status is MergeStatus.STOPPED]

    @property
    def asked(self) -> list[MergeOutcome]:
        return self.policy.asked

    async def until(self, predicate: Callable[[], bool]) -> None:
        """Wait until `predicate` holds, woken by the queue itself."""
        while not predicate():
            await asyncio.wait_for(self.rang.wait(), TIMEOUT)
            self.rang.clear()

    async def asked_about(self, count: int) -> None:
        """Wait until the queue has asked `count` times."""
        while len(self.policy.asked) < count:
            await asyncio.wait_for(self.policy.rang.wait(), TIMEOUT)
            self.policy.rang.clear()


@asynccontextmanager
async def running(
    repo: Path,
    build: Callable[[], Awaitable[ExecResult]] | None = None,
    policy: Policy | None = None,
) -> AsyncIterator[Harness]:
    """A queue with its consumer going, stopped and joined on the way out."""
    used = Policy(hold=True) if policy is None else policy
    vcs = Git(repo)
    queue = MergeQueue(vcs, MergeConfig(build=build, resolve=used))
    rang = asyncio.Event()
    # The policy's own ring wakes `until` too: an ask is a thing that happened,
    # and a test waiting for one must not have to poll for it.
    harness = Harness(queue=queue, vcs=vcs, repo=repo, policy=used, rang=rang)
    async with queue.running():
        try:
            yield harness
        finally:
            await queue.stop()
            await asyncio.wait_for(
                asyncio.gather(*harness.tasks, return_exceptions=True), TIMEOUT
            )


async def hold_on_conflict(harness: Harness, branch: str) -> None:
    """Submit the request that cannot merge, and wait for the queue to ask."""
    harness.submit(TICKET, branch)
    await harness.asked_about(1)


# -- a clean merge --------------------------------------------------------


async def test_a_clean_merge_with_no_build_gate_is_reported_merged(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    async with running(repo) as h:
        h.submit(TICKET, "t1")
        await h.until(lambda: h.merged == [TICKET])

        assert h.vcs.is_ancestor("t1", MAIN) is True
        assert h.asked == []


async def test_the_merge_lands_in_the_repository_root(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    async with running(repo) as h:
        h.submit(TICKET, "t1")
        await h.until(lambda: h.merged == [TICKET])

        # The root is on the base branch and stays on it; the file the ticket
        # added is there afterwards.
        assert h.vcs.current_branch() == MAIN
        assert (repo / "a.py").read_text(encoding="utf-8") == "a\n"


async def test_a_clean_merge_runs_the_build_before_reporting_merged(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    build = StubBuild(passed())
    async with running(repo, build) as h:
        build.probe = lambda: list(h.merged)
        h.submit(TICKET, "t1")
        await h.until(lambda: h.merged == [TICKET])

        assert build.calls == 1
        # Nothing had been reported merged at the moment the build ran.
        assert build.probed == [[]]


async def test_a_failing_build_asks_with_the_whole_result(repo: Path) -> None:
    # The whole `ExecResult`, not a slice of it: which lines of a failed build
    # a person should read is the workflow's call, so the outcome carries all
    # of them and the queue truncates nothing.
    branch_with(repo, "t1", "a.py", "a\n")
    output = "\n".join(f"line {index}" for index in range(200)) + "\n"
    build = StubBuild(failed(code=2, stdout=output))
    async with running(repo, build) as h:
        h.submit(TICKET, "t1")
        await h.asked_about(1)

        outcome = h.asked[0]
        assert outcome.key == TICKET
        assert outcome.status is MergeStatus.BUILD_FAILED
        assert outcome.build is not None
        assert outcome.build.code == 2
        assert "line 199" in outcome.build.stdout
        assert "line 0\n" in outcome.build.stdout
        assert h.merged == []


async def test_a_failing_build_leaves_the_merge_commit_alone(repo: Path) -> None:
    # The queue does not unwind a merge git already made. A person decides
    # between fixing the build and taking the merge back out.
    branch_with(repo, "t1", "a.py", "a\n")
    async with running(repo, StubBuild(failed())) as h:
        h.submit(TICKET, "t1")
        await h.asked_about(1)

        assert h.vcs.is_ancestor("t1", MAIN) is True


# -- a conflicting merge --------------------------------------------------


async def test_a_conflicting_merge_asks_naming_the_paths(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await hold_on_conflict(h, branch)

        outcome = h.asked[0]
        assert outcome.key == TICKET
        assert outcome.status is MergeStatus.CONFLICT
        assert outcome.conflicted == (FILE,)
        assert h.merged == []


async def test_a_conflicting_merge_is_left_in_progress(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await hold_on_conflict(h, branch)

        assert h.vcs.merge_in_progress(repo) is True
        assert h.vcs.unmerged_paths(repo) == (FILE,)


async def test_a_conflict_does_not_run_the_build(repo: Path) -> None:
    branch = already_collided(repo)
    build = StubBuild(passed())
    async with running(repo, build) as h:
        await hold_on_conflict(h, branch)

        assert build.calls == 0


# -- one at a time --------------------------------------------------------


async def test_the_second_request_does_not_begin_until_the_first_is_finished(
    repo: Path,
) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    branch_with(repo, "t2", "b.py", "b\n")
    gate = Gate()
    async with running(repo, gate) as h:
        h.submit("T-01", "t1")
        h.submit("T-02", "t2")
        await asyncio.wait_for(gate.started.wait(), TIMEOUT)
        await settle()

        # The first is still inside its build gate: it has not been reported,
        # and the second has not reached a build of its own.
        assert gate.calls == 1
        assert h.merged == []

        gate.release.set()
        await h.until(lambda: h.merged == ["T-01", "T-02"])
        assert gate.calls == 2


async def test_the_first_merge_is_what_makes_the_second_one_conflict(repo: Path) -> None:
    # The ordering that makes serializing worth the trouble: both branches are
    # clean against the base they were cut from, and only one of them can land.
    first, second = two_rewrites(repo)
    async with running(repo) as h:
        h.submit("T-01", first)
        h.submit("T-02", second)
        await h.asked_about(1)

        assert h.merged == ["T-01"]
        assert h.asked[0].key == "T-02"
        assert h.asked[0].conflicted == (FILE,)


# -- holding the line at the head -----------------------------------------


async def test_an_unanswered_outcome_stops_everything_behind_it(repo: Path) -> None:
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        h.submit(TICKET, branch)
        h.submit("T-02", "t3")
        await h.asked_about(1)
        await settle()

        assert h.merged == []
        assert h.vcs.is_ancestor("t3", MAIN) is False


async def test_a_submit_is_still_accepted_while_the_head_is_held(repo: Path) -> None:
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        await hold_on_conflict(h, branch)

        h.submit("T-02", "t3")
        await settle()

        assert h.merged == []
        assert len(h.asked) == 1


# -- RETRY: what the person left behind -----------------------------------


async def test_a_retry_asks_again_when_the_conflict_is_untouched(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await hold_on_conflict(h, branch)

        h.policy.answer(MergeDecision.RETRY)
        await h.asked_about(2)

        assert h.asked[1].status is MergeStatus.CONFLICT
        assert h.asked[1].conflicted == h.asked[0].conflicted
        assert h.merged == []
        assert h.vcs.merge_in_progress(repo) is True


async def test_a_retry_commits_a_resolution_that_was_only_staged(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await hold_on_conflict(h, branch)

        # Resolved and staged, not committed.
        (repo / FILE).write_text("resolved\n", encoding="utf-8")
        git(repo, "add", "--", FILE)

        h.policy.answer(MergeDecision.RETRY)
        await h.until(lambda: h.merged == [TICKET])

        assert h.vcs.merge_in_progress(repo) is False
        assert h.vcs.is_ancestor(branch, MAIN) is True
        assert (repo / FILE).read_text(encoding="utf-8") == "resolved\n"


async def test_a_retry_accepts_a_merge_the_person_committed_themselves(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await hold_on_conflict(h, branch)

        (repo / FILE).write_text("resolved\n", encoding="utf-8")
        git(repo, "add", "--", FILE)
        git(repo, "commit", "--no-edit")

        h.policy.answer(MergeDecision.RETRY)
        await h.until(lambda: h.merged == [TICKET])

        assert h.vcs.is_ancestor(branch, MAIN) is True
        assert len(h.asked) == 1


async def test_a_retry_reports_an_aborted_merge_as_abandoned(repo: Path) -> None:
    branch = already_collided(repo)
    async with running(repo) as h:
        await hold_on_conflict(h, branch)

        git(repo, "merge", "--abort")

        h.policy.answer(MergeDecision.RETRY)
        await h.until(lambda: h.abandoned == [TICKET])

        assert h.merged == []
        assert h.vcs.is_ancestor(branch, MAIN) is False


async def test_an_abandoned_request_does_not_hold_up_the_queue(repo: Path) -> None:
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        await hold_on_conflict(h, branch)
        h.submit("T-02", "t3")

        git(repo, "merge", "--abort")
        h.policy.answer(MergeDecision.RETRY)
        await h.until(lambda: h.merged == ["T-02"])

        assert h.abandoned == [TICKET]


async def test_a_retried_resolution_still_has_to_pass_the_build(repo: Path) -> None:
    branch = already_collided(repo)
    build = StubBuild(failed())
    async with running(repo, build) as h:
        await hold_on_conflict(h, branch)
        assert build.calls == 0

        (repo / FILE).write_text("resolved\n", encoding="utf-8")
        git(repo, "add", "--", FILE)
        h.policy.answer(MergeDecision.RETRY)
        await h.asked_about(2)

        assert build.calls == 1
        assert h.asked[1].status is MergeStatus.BUILD_FAILED
        assert h.merged == []


# -- RETRY: after a build failure -----------------------------------------


async def test_a_retry_merges_when_the_build_now_passes(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    build = StubBuild(failed(), passed())
    async with running(repo, build) as h:
        h.submit(TICKET, "t1")
        await h.asked_about(1)

        h.policy.answer(MergeDecision.RETRY)
        await h.until(lambda: h.merged == [TICKET])

        assert build.calls == 2
        assert len(h.asked) == 1


async def test_a_retry_asks_again_when_the_build_still_fails(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    build = StubBuild(failed(code=3))
    async with running(repo, build) as h:
        h.submit(TICKET, "t1")
        await h.asked_about(1)

        h.policy.answer(MergeDecision.RETRY)
        await h.asked_about(2)

        assert build.calls == 2
        assert h.merged == []


async def test_a_build_failure_still_blocks_what_is_behind_it(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo, StubBuild(failed())) as h:
        h.submit(TICKET, "t1")
        h.submit("T-02", "t3")
        await h.asked_about(1)
        await settle()

        assert h.merged == []
        assert h.vcs.is_ancestor("t3", MAIN) is False


# -- ABANDON --------------------------------------------------------------


async def test_abandon_answers_the_submitter_and_drains_on(repo: Path) -> None:
    # The realistic `ABANDON`: a person looked at the conflict, threw the merge
    # away, and said so. What is behind it merges into the base the abandoned
    # request never changed.
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        await hold_on_conflict(h, branch)
        h.submit("T-02", "t3")

        git(repo, "merge", "--abort")
        h.policy.answer(MergeDecision.ABANDON)
        await h.until(lambda: h.merged == ["T-02"])

        assert h.abandoned == [TICKET]
        assert len(h.asked) == 1
        assert h.vcs.is_ancestor(branch, MAIN) is False


async def test_abandon_leaves_the_repository_exactly_as_it_was_found(repo: Path) -> None:
    # The queue does not clean up after a decision it was handed: whoever said
    # `ABANDON` decides whether the half-done merge is aborted or looked at.
    branch = already_collided(repo)
    async with running(repo, policy=Policy(MergeDecision.ABANDON)) as h:
        h.submit(TICKET, branch)
        await h.until(lambda: h.abandoned == [TICKET])

        assert h.vcs.merge_in_progress(repo) is True


# -- STOP -----------------------------------------------------------------


async def test_stop_ends_the_queue_and_answers_the_request_it_asked_about(
    repo: Path,
) -> None:
    branch = already_collided(repo)
    async with running(repo, policy=Policy(MergeDecision.STOP)) as h:
        h.submit(TICKET, branch)
        await h.until(lambda: h.stopped == [TICKET])

        assert h.merged == []
        assert h.abandoned == []


async def test_stop_answers_everything_still_queued_behind_it(repo: Path) -> None:
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo, policy=Policy(MergeDecision.STOP)) as h:
        h.submit(TICKET, branch)
        h.submit("T-02", "t3")
        await h.until(lambda: set(h.stopped) == {TICKET, "T-02"})

        # Nothing behind the stop was merged on the way out.
        assert h.merged == []
        assert h.vcs.is_ancestor("t3", MAIN) is False


async def test_a_submit_to_a_stopped_queue_answers_stopped_at_once(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo, policy=Policy(MergeDecision.STOP)) as h:
        h.submit(TICKET, "no-such-branch")
        await h.until(lambda: h.stopped == [TICKET])

        h.submit("T-02", "t3")
        await h.until(lambda: h.stopped == [TICKET, "T-02"])

        assert h.merged == []
        assert h.vcs.is_ancestor("t3", MAIN) is False


# -- the default policy ---------------------------------------------------


async def test_the_default_resolve_stops_the_queue_on_anything_that_did_not_land(
    repo: Path,
) -> None:
    # A caller that wires nothing up gets a queue that ends rather than one
    # that guesses — and, above all, one that answers.
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    vcs = Git(repo)
    queue = MergeQueue(vcs)
    async with queue.running():
        first = await asyncio.wait_for(
            queue.submit(MergeRequest(key=TICKET, branch=branch, target=MAIN, cwd=repo)),
            TIMEOUT,
        )
        second = await asyncio.wait_for(
            queue.submit(MergeRequest(key="T-02", branch="t3", target=MAIN, cwd=repo)),
            TIMEOUT,
        )

    assert first.status is MergeStatus.STOPPED
    assert second.status is MergeStatus.STOPPED
    assert vcs.is_ancestor("t3", MAIN) is False


async def test_the_default_resolve_lets_a_clean_merge_through(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    vcs = Git(repo)
    queue = MergeQueue(vcs)
    async with queue.running():
        outcome = await asyncio.wait_for(
            queue.submit(MergeRequest(key=TICKET, branch="t1", target=MAIN, cwd=repo)), TIMEOUT
        )

    assert outcome.status is MergeStatus.MERGED
    assert vcs.is_ancestor("t1", MAIN) is True


# -- a branch git cannot find ---------------------------------------------


async def test_a_branch_that_does_not_resolve_is_reported_as_a_vcs_error(
    repo: Path,
) -> None:
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        h.submit(TICKET, "no-such-branch")
        await h.asked_about(1)

        outcome = h.asked[0]
        assert outcome.key == TICKET
        assert outcome.status is MergeStatus.VCS_ERROR
        assert outcome.error != ""

        # The queue does not decide what a branch git cannot find means — it
        # asks, and holds everything behind it until it is answered. Nothing
        # was merged and nothing was reported while it waits.
        h.submit("T-02", "t3")
        await settle()

        assert h.done == []
        assert h.vcs.is_ancestor("t3", MAIN) is False


async def test_a_retry_on_a_branch_git_cannot_find_reads_as_abandoned(repo: Path) -> None:
    # `RETRY` is always the same question — what is in the repository now? A
    # branch that never merged and no longer resolves is on the abandoned row,
    # the same as one whose merge was aborted.
    branch_with(repo, "t3", "other.py", "x\n")
    async with running(repo) as h:
        h.submit(TICKET, "no-such-branch")
        await h.asked_about(1)

        h.policy.answer(MergeDecision.RETRY)
        await h.until(lambda: h.abandoned == [TICKET])

        assert h.merged == []


# -- an exception escaping the build ---------------------------------------


async def test_a_raising_build_is_reported_rather_than_killing_the_consumer(
    repo: Path,
) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    build = RaisingBuild(FileNotFoundError(2, "No such file or directory", "./gradlew-x"))
    async with running(repo, build) as h:
        h.submit(TICKET, "t1")
        await h.asked_about(1)

        outcome = h.asked[0]
        assert outcome.key == TICKET
        assert outcome.status is MergeStatus.ERROR
        assert "FileNotFoundError" in outcome.error
        assert h.merged == []

        # The consumer is still alive: `stop` is not left waiting on a dead
        # task, and the queue is still able to be asked something else.
        h.policy.answer(MergeDecision.RETRY)
        await h.asked_about(2)
        assert h.merged == []


async def test_a_request_behind_a_raising_build_stays_queued(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    branch_with(repo, "t2", "b.py", "b\n")
    build = RaisingBuild(FileNotFoundError(2, "No such file or directory", "./gradlew-x"))
    async with running(repo, build) as h:
        h.submit("T-01", "t1")
        h.submit("T-02", "t2")
        await h.asked_about(1)
        await settle()

        assert build.calls == 1
        assert h.merged == []
        assert h.vcs.is_ancestor("t2", MAIN) is False


async def test_a_raising_resolve_ends_the_queue_without_stranding_anyone(
    repo: Path,
) -> None:
    # The one callback the queue awaits is the workflow's, and a broken one
    # must not leave a submitter waiting on a consumer that is already dead.
    branch = already_collided(repo)
    branch_with(repo, "t3", "other.py", "x\n")
    policy = Policy()
    policy.raises = RuntimeError("dashboard write failed")
    vcs = Git(repo)
    queue = MergeQueue(vcs, MergeConfig(resolve=policy))
    first = asyncio.create_task(
        queue.submit(MergeRequest(key=TICKET, branch=branch, target=MAIN, cwd=repo))
    )
    second = asyncio.create_task(
        queue.submit(MergeRequest(key="T-02", branch="t3", target=MAIN, cwd=repo))
    )

    with pytest.raises(RuntimeError, match="dashboard write failed"):
        async with queue.running():
            await asyncio.wait_for(asyncio.gather(first, second), TIMEOUT)

    assert first.result().status is MergeStatus.STOPPED
    assert second.result().status is MergeStatus.STOPPED


async def test_a_fixed_build_merges_the_request_after_the_process_restarts(repo: Path) -> None:
    # What the error says to do: restart. A fresh queue is that restart, and
    # the merge `vcs.merge` already made when the build first raised is why the
    # same request lands clean the second time round.
    branch_with(repo, "t1", "a.py", "a\n")
    build = RaisingBuild(FileNotFoundError(2, "No such file or directory", "./gradlew-x"))
    async with running(repo, build) as h:
        h.submit(TICKET, "t1")
        await h.asked_about(1)

    async with running(repo, StubBuild(passed())) as h2:
        h2.submit(TICKET, "t1")
        await h2.until(lambda: h2.merged == [TICKET])


# -- stopping -------------------------------------------------------------


async def test_stop_ends_the_consumer_with_work_still_queued(repo: Path) -> None:
    branch_with(repo, "t1", "a.py", "a\n")
    queue = MergeQueue(Git(repo))
    submitted = asyncio.create_task(
        queue.submit(MergeRequest(key=TICKET, branch="t1", target=MAIN, cwd=repo))
    )
    await settle()

    await asyncio.wait_for(queue.stop(), TIMEOUT)
    outcome = await asyncio.wait_for(submitted, TIMEOUT)

    assert outcome.status is MergeStatus.STOPPED
    assert Git(repo).is_ancestor("t1", MAIN) is False

    # A stopped queue stays stopped: running it again does nothing.
    async with queue.running():
        await settle()
    assert Git(repo).is_ancestor("t1", MAIN) is False


async def test_stop_cancels_an_in_flight_build_rather_than_waiting_for_it(
    repo: Path,
) -> None:
    # `stop` used to let whatever was in flight finish; now it cancels it, so
    # Ctrl-C during a slow build does not look hung for however long is left.
    branch_with(repo, "t1", "a.py", "a\n")
    branch_with(repo, "t2", "b.py", "b\n")
    gate = Gate()
    policy = Policy()
    async with running(repo, gate, policy) as h:
        h.submit("T-01", "t1")
        h.submit("T-02", "t2")
        await asyncio.wait_for(gate.started.wait(), TIMEOUT)
        await asyncio.wait_for(h.queue.stop(), TIMEOUT)

        assert gate.calls == 1
        assert gate.cancelled is True
        assert h.merged == []
        assert h.asked == []
        # `stop` does not unwind what git already did: `t1`'s merge commit
        # stands, only the build gate on top of it was cut short. `t2` was
        # never reached — and both submitters were answered.
        await h.until(lambda: set(h.stopped) == {"T-01", "T-02"})
        assert h.vcs.is_ancestor("t1", MAIN) is True
        assert h.vcs.is_ancestor("t2", MAIN) is False


async def test_stop_returns_even_though_nothing_is_consuming(repo: Path) -> None:
    queue = MergeQueue(Git(repo))
    await asyncio.wait_for(queue.stop(), TIMEOUT)


# -- a request's own target ------------------------------------------------


async def test_a_request_targeting_a_non_base_branch_merges_into_that_branch(
    repo: Path,
) -> None:
    vcs = Git(repo)
    parent = worktree_on(vcs, repo, "T-parent")
    branch_off(repo, "T-parent", "t1", "a.py", "a\n")
    async with running(repo) as h:
        h.submit(TICKET, "t1", target="T-parent", cwd=parent)
        await h.until(lambda: h.merged == [TICKET])

        assert h.vcs.is_ancestor("t1", "T-parent") is True
        assert h.vcs.is_ancestor("t1", MAIN) is False


async def test_two_requests_with_different_targets_both_land_in_their_own_tree(
    repo: Path,
) -> None:
    vcs = Git(repo)
    parent = worktree_on(vcs, repo, "T-parent")
    branch_with(repo, "t1", "a.py", "a\n")
    branch_off(repo, "T-parent", "t2", "b.py", "b\n")
    async with running(repo) as h:
        h.submit("T-01", "t1")
        h.submit("T-02", "t2", target="T-parent", cwd=parent)
        await h.until(lambda: h.merged == ["T-01", "T-02"])

        assert h.vcs.is_ancestor("t1", MAIN) is True
        assert h.vcs.is_ancestor("t2", "T-parent") is True
        assert h.vcs.is_ancestor("t2", MAIN) is False


async def test_a_conflict_on_a_non_base_target_is_left_in_that_tree(repo: Path) -> None:
    vcs = Git(repo)
    parent = worktree_on(vcs, repo, "T-parent")
    branch = collided_off(repo, parent, "T-parent")
    async with running(repo) as h:
        h.submit(TICKET, branch, target="T-parent", cwd=parent)
        await h.asked_about(1)

        outcome = h.asked[0]
        assert outcome.status is MergeStatus.CONFLICT
        assert outcome.conflicted == (FILE,)
        assert h.vcs.merge_in_progress(parent) is True
        assert h.vcs.unmerged_paths(parent) == (FILE,)
        # The root, on the base branch, was never touched by this merge.
        assert h.vcs.merge_in_progress(repo) is False


async def test_retrying_a_non_base_conflict_inspects_that_tree(repo: Path) -> None:
    vcs = Git(repo)
    parent = worktree_on(vcs, repo, "T-parent")
    branch = collided_off(repo, parent, "T-parent")
    async with running(repo) as h:
        h.submit(TICKET, branch, target="T-parent", cwd=parent)
        await h.asked_about(1)

        (parent / FILE).write_text("resolved\n", encoding="utf-8")
        git(parent, "add", "--", FILE)

        h.policy.answer(MergeDecision.RETRY)
        await h.until(lambda: h.merged == [TICKET])

        assert h.vcs.merge_in_progress(parent) is False
        assert h.vcs.is_ancestor(branch, "T-parent") is True
        assert (parent / FILE).read_text(encoding="utf-8") == "resolved\n"


async def test_the_build_gate_runs_for_a_non_base_target_too(repo: Path) -> None:
    vcs = Git(repo)
    parent = worktree_on(vcs, repo, "T-parent")
    branch_off(repo, "T-parent", "t1", "a.py", "a\n")
    build = StubBuild(passed())
    async with running(repo, build) as h:
        h.submit(TICKET, "t1", target="T-parent", cwd=parent)
        await h.until(lambda: h.merged == [TICKET])

        assert build.calls == 1


async def test_serialization_holds_across_requests_with_different_targets(
    repo: Path,
) -> None:
    vcs = Git(repo)
    parent = worktree_on(vcs, repo, "T-parent")
    branch_with(repo, "t1", "a.py", "a\n")
    branch_off(repo, "T-parent", "t2", "b.py", "b\n")
    gate = Gate()
    async with running(repo, gate) as h:
        h.submit("T-01", "t1")
        h.submit("T-02", "t2", target="T-parent", cwd=parent)
        await asyncio.wait_for(gate.started.wait(), TIMEOUT)
        await settle()

        # The first is still inside its build gate: the second, targeting a
        # different tree entirely, still has not been started.
        assert gate.calls == 1
        assert h.merged == []

        gate.release.set()
        await h.until(lambda: h.merged == ["T-01", "T-02"])
        assert gate.calls == 2


# -- the data --------------------------------------------------------------


def test_a_request_is_frozen(tmp_path: Path) -> None:
    request = MergeRequest(key=TICKET, branch="t1", target=MAIN, cwd=tmp_path)
    with pytest.raises(AttributeError):
        request.branch = "other"  # type: ignore[misc]


def test_an_outcome_is_frozen(tmp_path: Path) -> None:
    outcome = MergeOutcome(key=TICKET, status=MergeStatus.MERGED)
    with pytest.raises(AttributeError):
        outcome.status = MergeStatus.CONFLICT  # type: ignore[misc]
