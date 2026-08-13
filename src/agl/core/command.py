"""Running an external command and getting its output back.

Layer: core. A shared helper rather than a module in its own right: `vcs` runs
git through it and the merge gate runs a project's build through it. It is
outside the independence contract for that reason, and it stays outside by
knowing nothing about git, builds, agents, or anything else above it. Two
callers sharing one subprocess runner is what the helper is for; the contract
exists to stop `vcs` reaching into `store`.

Never `shell=True`: arguments are passed as a list so a ref or a filename can
never be read as syntax. Both streams are captured and decoded as UTF-8 with
`errors="replace"`, because a diff may carry bytes that are not text and a
mangled character is a better outcome than an exception from the middle of a
merge.

Output is never truncated. Which slice of a failed build matters is
language-specific, so the caller that knows what it is running decides.
"""

import asyncio
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["TIMEOUT_CODE", "ExecError", "ExecResult", "run", "run_async"]

TIMEOUT_CODE = -1
"""The `code` a timed-out command is reported with. Non-zero, so a caller
checking only for success still sees a failure.

It is **not** unique to a timeout, and nothing here pretends otherwise: a child
killed by a signal reports `-signum`, and SIGHUP is signal 1. `timed_out` is the
only thing that distinguishes a hang from a command that died some other way,
and it is what a caller reads. `code == TIMEOUT_CODE` proves nothing."""


@dataclass(frozen=True)
class ExecResult:
    """What a finished command left behind."""

    argv: tuple[str, ...]
    code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Whether the command succeeded. Reads better than `code == 0` at a
        call site asking a yes-or-no question of a build or a git plumbing
        command."""
        return self.code == 0


class ExecError(Exception):
    """A command failed under `check=True`. Carries the `ExecResult`."""

    def __init__(self, result: ExecResult, timeout: float | None = None) -> None:
        command = " ".join(result.argv)
        detail = result.stderr.strip() or result.stdout.strip()
        if result.timed_out:
            summary = f"timed out after {timeout}s"
        else:
            summary = f"exited {result.code}"
        super().__init__(f"`{command}` {summary}: {detail}")
        self.result = result


def run(
    argv: Sequence[str],
    cwd: Path,
    check: bool = True,
    timeout: float | None = None,
) -> ExecResult:
    """Run `argv` in `cwd` and return both streams.

    Raises `ExecError` on a non-zero exit when `check` is true. With
    `check=False` the failure comes back as a result instead, which is what
    callers of exit-code-signalling commands — `merge-base --is-ancestor`,
    `merge` — need in order to tell "no" from "broken".

    `timeout` is a wall-clock limit in seconds; `None` means wait forever. On
    expiry the child is killed and the call reports `timed_out=True` with
    `code=TIMEOUT_CODE` (-1) and whatever output was captured before the kill,
    raising `ExecError` under `check=True` like any other failure.

    `timed_out` is the only signal that a timeout happened, and the only one a
    caller should read for it. The code is not: a child killed by SIGHUP exits
    `-1` too, so `code == TIMEOUT_CODE` on a result whose `timed_out` is false
    means a signal, not a hang.

    The kill reaches the direct child only, not its descendants: a build tool
    that spawns a daemon can leave one running. Process-group handling is a
    later problem.
    """
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        result = ExecResult(
            argv=tuple(argv),
            code=TIMEOUT_CODE,
            stdout=_decoded(expired.stdout),
            stderr=_decoded(expired.stderr),
            timed_out=True,
        )
        if check:
            raise ExecError(result, timeout) from None
        return result
    result = ExecResult(
        argv=tuple(argv),
        code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.code != 0:
        raise ExecError(result)
    return result


async def run_async(
    argv: Sequence[str],
    cwd: Path,
    check: bool = True,
    timeout: float | None = None,
) -> ExecResult:
    """`run`, but with a real process handle a caller can kill.

    Same `ExecResult`, `check`, and `timeout` semantics as `run` — the
    difference is what happens when the *awaiting* coroutine is cancelled or
    the timeout expires. `run` wraps `subprocess.run`, which blocks a thread
    that cannot be cancelled: `asyncio.to_thread(run, ...)` leaves the thread
    running the command underneath a cancelled await, and since that thread is
    non-daemon, the interpreter will not exit until it finishes — a build that
    takes minutes makes Ctrl-C look hung for exactly that long. `run_async` is
    built on `asyncio.create_subprocess_exec` instead, so cancellation and a
    timeout both reach the child: it is killed and awaited before the
    cancellation or the timeout is allowed to propagate, so nothing is left
    running behind a call that has already returned.

    Killing `./gradlew` kills the client process, not the Gradle daemon it
    talks to over a socket. That is correct, not a shortcut: the daemon is
    meant to outlive any one build and be reused by the next one, and tearing
    it down on every interrupt would trade a fast cancel for a slow cold start
    on the next run.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.ensure_future(process.stdout.read())
    stderr_task = asyncio.ensure_future(process.stderr.read())
    try:
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except TimeoutError:
            await _kill(process)
            stdout, stderr = await _collect(stdout_task, stderr_task)
            result = ExecResult(
                argv=tuple(argv),
                code=TIMEOUT_CODE,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )
            if check:
                raise ExecError(result, timeout) from None
            return result
    except asyncio.CancelledError:
        await _kill(process)
        stdout_task.cancel()
        stderr_task.cancel()
        raise

    stdout, stderr = await _collect(stdout_task, stderr_task)
    assert process.returncode is not None  # `wait()` above returned
    result = ExecResult(argv=tuple(argv), code=process.returncode, stdout=stdout, stderr=stderr)
    if check and result.code != 0:
        raise ExecError(result)
    return result


async def _kill(process: asyncio.subprocess.Process) -> None:
    """Kill the child if it is still running, and reap it before returning."""
    if process.returncode is None:
        process.kill()
    await process.wait()


async def _collect(
    stdout_task: asyncio.Task[bytes], stderr_task: asyncio.Task[bytes]
) -> tuple[str, str]:
    """Both streams, decoded the same way `run` decodes them."""
    stdout_bytes, stderr_bytes = await asyncio.gather(stdout_task, stderr_task)
    return _decoded(stdout_bytes), _decoded(stderr_bytes)


def _decoded(partial: str | bytes | None) -> str:
    """Whatever `TimeoutExpired` captured, as text. It may be `None`, and it is
    bytes rather than `str` on some platforms despite the `encoding` above."""
    if partial is None:
        return ""
    if isinstance(partial, bytes):
        return partial.decode("utf-8", errors="replace")
    return partial
