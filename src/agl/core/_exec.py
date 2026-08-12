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
"""

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ExecError", "ExecResult", "run"]


@dataclass(frozen=True)
class ExecResult:
    """What a finished command left behind."""

    argv: tuple[str, ...]
    code: int
    stdout: str
    stderr: str


class ExecError(Exception):
    """A command exited non-zero under `check=True`. Carries the `ExecResult`."""

    def __init__(self, result: ExecResult) -> None:
        command = " ".join(result.argv)
        detail = result.stderr.strip() or result.stdout.strip()
        super().__init__(f"`{command}` exited {result.code}: {detail}")
        self.result = result


def run(argv: Sequence[str], cwd: Path, check: bool = True) -> ExecResult:
    """Run `argv` in `cwd` and return both streams.

    Raises `ExecError` on a non-zero exit when `check` is true. With
    `check=False` the failure comes back as a result instead, which is what
    callers of exit-code-signalling commands — `merge-base --is-ancestor`,
    `merge` — need in order to tell "no" from "broken".
    """
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    result = ExecResult(
        argv=tuple(argv),
        code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.code != 0:
        raise ExecError(result)
    return result
