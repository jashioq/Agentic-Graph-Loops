"""The subprocess helper, against real trivial commands — no git yet.

Anything needing more than a bare coreutil goes through `sys.executable`, not
through `sh`. `/bin/sh` is bash on macOS and dash on Debian, and the two differ
on `printf` escapes among other things; the interpreter running the tests is the
one portable program we are guaranteed to have.
"""

import sys
import time
from pathlib import Path

import pytest

from agl.core._exec import TIMEOUT_CODE, ExecError, ExecResult, run


def python(source: str) -> list[str]:
    """An argv running `source` under the interpreter running the tests."""
    return [sys.executable, "-c", source]


# -- results --------------------------------------------------------------


def test_a_successful_command_reports_code_zero(tmp_path: Path) -> None:
    result = run(["true"], cwd=tmp_path)
    assert result.code == 0


def test_stdout_is_captured(tmp_path: Path) -> None:
    assert run(["echo", "hello"], cwd=tmp_path).stdout == "hello\n"


def test_stderr_is_captured(tmp_path: Path) -> None:
    result = run(python("import sys; print('oops', file=sys.stderr)"), cwd=tmp_path)
    assert result.stderr == "oops\n"
    assert result.stdout == ""


def test_the_result_carries_the_argv_it_ran(tmp_path: Path) -> None:
    result = run(["echo", "hello"], cwd=tmp_path)
    assert result.argv == ("echo", "hello")


def test_the_result_is_a_frozen_dataclass(tmp_path: Path) -> None:
    result = run(["true"], cwd=tmp_path)
    with pytest.raises(AttributeError):
        result.code = 1  # type: ignore[misc]


# -- failure --------------------------------------------------------------


def test_a_failing_command_raises_by_default(tmp_path: Path) -> None:
    with pytest.raises(ExecError):
        run(["false"], cwd=tmp_path)


FAIL_WITH_OOPS = "import sys; print('oops', file=sys.stderr); sys.exit(3)"


def test_the_error_carries_the_result(tmp_path: Path) -> None:
    with pytest.raises(ExecError) as caught:
        run(python(FAIL_WITH_OOPS), cwd=tmp_path)
    result = caught.value.result
    assert isinstance(result, ExecResult)
    assert result.code == 3
    assert result.stderr == "oops\n"
    assert result.argv == (sys.executable, "-c", FAIL_WITH_OOPS)


def test_the_error_message_names_the_command_and_the_stderr(tmp_path: Path) -> None:
    with pytest.raises(ExecError) as caught:
        run(python(FAIL_WITH_OOPS), cwd=tmp_path)
    message = str(caught.value)
    assert "oops" in message
    assert sys.executable in message


def test_check_false_returns_the_failure_instead_of_raising(tmp_path: Path) -> None:
    result = run(["false"], cwd=tmp_path, check=False)
    assert result.code != 0


def test_check_false_still_returns_output(tmp_path: Path) -> None:
    result = run(
        python("import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"),
        cwd=tmp_path,
        check=False,
    )
    assert (result.code, result.stdout, result.stderr) == (7, "out\n", "err\n")


# -- the working directory ------------------------------------------------


def test_the_command_runs_in_the_given_directory(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "marker.txt").write_text("here\n", encoding="utf-8")
    assert run(["cat", "marker.txt"], cwd=work).stdout == "here\n"


# -- no shell -------------------------------------------------------------


def test_arguments_are_not_interpreted_by_a_shell(tmp_path: Path) -> None:
    # `shell=True` would expand this; passing argv straight through must not.
    assert run(["echo", "$HOME"], cwd=tmp_path).stdout == "$HOME\n"


def test_a_glob_argument_is_passed_through_literally(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert run(["echo", "*.txt"], cwd=tmp_path).stdout == "*.txt\n"


def test_an_argument_with_a_space_stays_one_argument(tmp_path: Path) -> None:
    assert run(["echo", "two words"], cwd=tmp_path).stdout == "two words\n"


# -- timeouts -------------------------------------------------------------


def _sleep(seconds: float) -> list[str]:
    return python(f"import time; time.sleep({seconds})")


def test_a_fast_command_under_a_generous_timeout_is_unaffected(tmp_path: Path) -> None:
    result = run(["true"], cwd=tmp_path, timeout=30)
    assert (result.code, result.timed_out) == (0, False)


def test_no_timeout_means_no_timeout(tmp_path: Path) -> None:
    result = run(["echo", "hello"], cwd=tmp_path, timeout=None)
    assert (result.stdout, result.timed_out) == ("hello\n", False)


def test_a_hanging_command_comes_back_as_a_timed_out_result(tmp_path: Path) -> None:
    result = run(_sleep(30), cwd=tmp_path, check=False, timeout=0.2)
    assert result.timed_out is True
    assert result.code != 0


def test_a_timeout_under_check_raises(tmp_path: Path) -> None:
    with pytest.raises(ExecError) as caught:
        run(_sleep(30), cwd=tmp_path, timeout=0.2)
    assert caught.value.result.timed_out is True


def test_the_timeout_error_message_says_it_timed_out_and_after_how_long(tmp_path: Path) -> None:
    with pytest.raises(ExecError) as caught:
        run(_sleep(30), cwd=tmp_path, timeout=0.25)
    message = str(caught.value)
    assert "timed out" in message
    assert "0.25" in message


def test_a_plain_failure_is_not_reported_as_a_timeout(tmp_path: Path) -> None:
    result = run(["false"], cwd=tmp_path, check=False, timeout=30)
    assert result.timed_out is False


def test_partial_output_survives_a_timeout(tmp_path: Path) -> None:
    result = run(
        python("import sys, time; print('partial'); sys.stdout.flush(); time.sleep(30)"),
        cwd=tmp_path,
        check=False,
        timeout=1.0,
    )
    assert result.timed_out is True
    assert "partial" in result.stdout


def test_the_timeout_bounds_the_wall_clock(tmp_path: Path) -> None:
    started = time.monotonic()
    run(_sleep(30), cwd=tmp_path, check=False, timeout=0.2)
    assert time.monotonic() - started < 5


def test_a_signal_killed_child_can_report_the_timeout_code_without_timing_out(
    tmp_path: Path,
) -> None:
    # A child killed by a signal exits `-signum`, and SIGHUP is 1. The code a
    # timeout reports is therefore not unique to a timeout, and `timed_out` is
    # the only thing that tells the two apart — which is what the contract now
    # says, instead of claiming no real process can produce it.
    result = run(
        python("import os, signal; os.kill(os.getpid(), signal.SIGHUP)"),
        cwd=tmp_path,
        check=False,
        timeout=30,
    )

    assert result.code == TIMEOUT_CODE
    assert result.timed_out is False


# -- decoding -------------------------------------------------------------


def test_utf8_output_is_decoded(tmp_path: Path) -> None:
    assert run(["echo", "café ✅"], cwd=tmp_path).stdout == "café ✅\n"


def test_invalid_utf8_is_replaced_rather_than_raising(tmp_path: Path) -> None:
    result = run(
        python("import sys; sys.stdout.buffer.write(b'\\xff\\xfe')"),
        cwd=tmp_path,
    )
    assert result.stdout == "��"
