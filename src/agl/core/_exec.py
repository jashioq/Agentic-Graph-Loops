"""Running an external command and getting its output back.

Layer: core, but not a core module — a private helper shared by the modules that
shell out. It is exempt from the independence
contract for that reason, and it stays exempt by knowing nothing about git,
agents, or anything else above it.

Never `shell=True`: arguments are passed as a list so a ref or a filename can
never be read as syntax. Both streams are captured and decoded as UTF-8 with
`errors="replace"`, because a diff may carry bytes that are not text and a
mangled character is a better outcome than an exception from the middle of a
merge.

Output is never truncated. Which slice of a failed build matters is
language-specific, so the caller that knows what it is running decides.
"""

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["TIMEOUT_CODE", "ExecError", "ExecResult", "run"]

TIMEOUT_CODE = -1
"""The `code` a timed-out command is reported with. Non-zero, and not an exit
status any process can produce, so `code == TIMEOUT_CODE` and `timed_out` agree."""


@dataclass(frozen=True)
class ExecResult:
    """What a finished command left behind."""

    argv: tuple[str, ...]
    code: int
    stdout: str
    stderr: str
    timed_out: bool = False


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
    raising `ExecError` under `check=True` like any other failure. Reading
    `timed_out` is how a caller tells a hang from a command that merely failed.

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


def _decoded(partial: str | bytes | None) -> str:
    """Whatever `TimeoutExpired` captured, as text. It may be `None`, and it is
    bytes rather than `str` on some platforms despite the `encoding` above."""
    if partial is None:
        return ""
    if isinstance(partial, bytes):
        return partial.decode("utf-8", errors="replace")
    return partial
