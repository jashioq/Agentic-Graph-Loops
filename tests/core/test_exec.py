"""The subprocess helper, against real trivial commands — no git yet."""

from pathlib import Path

import pytest

from agl.core._exec import ExecError, ExecResult, run

# -- results --------------------------------------------------------------


def test_a_successful_command_reports_code_zero(tmp_path: Path) -> None:
    result = run(["true"], cwd=tmp_path)
    assert result.code == 0


def test_stdout_is_captured(tmp_path: Path) -> None:
    assert run(["echo", "hello"], cwd=tmp_path).stdout == "hello\n"


def test_stderr_is_captured(tmp_path: Path) -> None:
    result = run(["sh", "-c", "echo oops >&2"], cwd=tmp_path)
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


def test_the_error_carries_the_result(tmp_path: Path) -> None:
    with pytest.raises(ExecError) as caught:
        run(["sh", "-c", "echo oops >&2; exit 3"], cwd=tmp_path)
    result = caught.value.result
    assert isinstance(result, ExecResult)
    assert result.code == 3
    assert result.stderr == "oops\n"
    assert result.argv == ("sh", "-c", "echo oops >&2; exit 3")


def test_the_error_message_names_the_command_and_the_stderr(tmp_path: Path) -> None:
    with pytest.raises(ExecError) as caught:
        run(["sh", "-c", "echo oops >&2; exit 3"], cwd=tmp_path)
    message = str(caught.value)
    assert "oops" in message
    assert "sh" in message


def test_check_false_returns_the_failure_instead_of_raising(tmp_path: Path) -> None:
    result = run(["false"], cwd=tmp_path, check=False)
    assert result.code != 0


def test_check_false_still_returns_output(tmp_path: Path) -> None:
    result = run(["sh", "-c", "echo out; echo err >&2; exit 7"], cwd=tmp_path, check=False)
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


# -- decoding -------------------------------------------------------------


def test_utf8_output_is_decoded(tmp_path: Path) -> None:
    assert run(["echo", "café ✅"], cwd=tmp_path).stdout == "café ✅\n"


def test_invalid_utf8_is_replaced_rather_than_raising(tmp_path: Path) -> None:
    result = run(["sh", "-c", r"printf '\xff\xfe'"], cwd=tmp_path)
    assert result.stdout == "��"
