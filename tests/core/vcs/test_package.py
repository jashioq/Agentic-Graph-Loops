"""The package root is the API and nothing else."""

import subprocess
import sys

import agl.core.vcs as vcs


def test_root_exports_exactly_the_api() -> None:
    assert set(vcs.__all__) == {
        "BranchExistsError",
        "DirtyWorktreeError",
        "FileStatus",
        "MergeResult",
        "UnknownRefError",
        "Vcs",
        "VcsError",
        "Worktree",
    }


def test_root_re_exports_nothing_from_impl() -> None:
    exported = [getattr(vcs, name) for name in vcs.__all__]
    modules = {getattr(value, "__module__", "") for value in exported}
    assert not any(module.startswith("agl.core.vcs.impl") for module in modules)


def test_importing_the_root_does_not_pull_in_impl() -> None:
    # In a fresh interpreter: any other test importing impl would bind it on the
    # package and make this pass or fail for reasons unrelated to __init__.
    source = "import sys, agl.core.vcs; print('agl.core.vcs.impl' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_the_api_module_does_not_import_the_implementation() -> None:
    # The ABC describes what workflows need; it must not know how git does it.
    source = "import sys, agl.core.vcs.api; print('agl.core.vcs.impl' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
