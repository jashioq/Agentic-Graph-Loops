"""Running an external command and getting its output back.

Layer: core. The shared subprocess runner `vcs` and the merge build gate both
call. Outside the core independence contract deliberately, and it stays outside
by knowing nothing about git, builds or agents.

Never `shell=True`: argv is a list, so a ref or filename can never be read as
syntax. Both streams are decoded UTF-8 with `errors="replace"`, and output is
never truncated — which slice matters is the caller's call.
"""

import asyncio
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["TIMEOUT_CODE", "ExecError", "ExecResult", "run", "run_async"]

TIMEOUT_CODE = -1
"""The `code` a timed-out command reports. Not unique to a timeout — a child
killed by SIGHUP reports `-1` too — so read `timed_out`, never this."""


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
        """Whether the command succeeded."""
        return self.code == 0

    @property
    def output(self) -> str:
        """Both streams as one text: which one carries the diagnosis is the command's choice."""
        return "\n".join(stream.strip("\n") for stream in (self.stdout, self.stderr) if stream)


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
    """Runs `argv` in `cwd` and returns both streams.

    param: check - raise `ExecError` on a non-zero exit; `False` reports it as a
        result instead, which exit-code-signalling git commands need
    param: timeout - wall-clock seconds, `None` to wait forever; on expiry the
        direct child (not its descendants) is killed
    return: ExecResult - `timed_out=True` and `code=TIMEOUT_CODE` after a timeout
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

    Same `check` and `timeout` semantics. The difference: cancelling the await
    reaches the child, which is killed and reaped before the cancellation
    propagates, so nothing runs on behind a call that has returned. A daemon the
    child talks to over a socket — Gradle's — survives, deliberately.
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
    """Whatever `TimeoutExpired` captured, as text: it may be `None`, or bytes."""
    if partial is None:
        return ""
    if isinstance(partial, bytes):
        return partial.decode("utf-8", errors="replace")
    return partial
