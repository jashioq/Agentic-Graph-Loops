"""The package root is the API and nothing else."""

import subprocess
import sys

import agl.core.terminal as terminal


def test_root_exports_exactly_the_api() -> None:
    assert set(terminal.__all__) == {
        "Answer",
        "Color",
        "Component",
        "LiveSession",
        "Option",
        "Question",
        "Row",
        "Rows",
        "Screen",
        "Spacer",
        "Terminal",
        "Text",
        "Timer",
    }


def test_root_re_exports_nothing_from_impl() -> None:
    exported = [getattr(terminal, name) for name in terminal.__all__]
    modules = {getattr(value, "__module__", "") for value in exported}
    assert not any(module.startswith("agl.core.terminal.impl") for module in modules)


def test_importing_the_root_does_not_pull_in_impl() -> None:
    # In a fresh interpreter: any other test importing impl would bind it on the
    # package and make this pass or fail for reasons unrelated to __init__.
    source = "import sys, agl.core.terminal; print('agl.core.terminal.impl' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
