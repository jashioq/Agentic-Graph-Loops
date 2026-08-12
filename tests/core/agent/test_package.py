"""The package root is the API and nothing else."""

import subprocess
import sys

import agl.core.agent as agent


def test_root_exports_exactly_the_api() -> None:
    assert set(agent.__all__) == {
        "AgentBudgetError",
        "AgentError",
        "AgentOption",
        "AgentQuestion",
        "AgentResult",
        "AgentRunner",
        "AgentSpec",
        "Tool",
    }


def test_root_re_exports_nothing_from_impl() -> None:
    exported = [getattr(agent, name) for name in agent.__all__]
    modules = {getattr(value, "__module__", "") for value in exported}
    assert not any(module.startswith("agl.core.agent.impl") for module in modules)


def test_importing_the_root_does_not_pull_in_impl() -> None:
    # In a fresh interpreter: any other test importing impl would bind it on the
    # package and make this pass or fail for reasons unrelated to __init__.
    source = "import sys, agl.core.agent; print('agl.core.agent.impl' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_importing_the_root_does_not_pull_in_the_sdk() -> None:
    # The API is plain dataclasses. Only `impl` talks to the SDK, and a workflow
    # importing the package root should not pay for loading it.
    source = "import sys, agl.core.agent; print('claude_agent_sdk' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
