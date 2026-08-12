"""The package root is the API and nothing else."""

import subprocess
import sys

import agl.core.store as store


def test_root_exports_exactly_the_api() -> None:
    assert set(store.__all__) == {"InvalidKeyError", "MissingKeyError", "Store"}


def test_root_re_exports_nothing_from_impl() -> None:
    exported = [getattr(store, name) for name in store.__all__]
    modules = {getattr(value, "__module__", "") for value in exported}
    assert not any(module.startswith("agl.core.store.impl") for module in modules)


def test_importing_the_root_does_not_pull_in_impl() -> None:
    # In a fresh interpreter: any other test importing impl would bind it on the
    # package and make this pass or fail for reasons unrelated to __init__.
    source = "import sys, agl.core.store; print('agl.core.store.impl' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
