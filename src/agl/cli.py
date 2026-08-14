"""`agl run` and `agl clean` — the composition root.

Layer: cli. This is the only file that constructs concrete `impl` classes;
everything below it takes them injected. `run` resolves a project, picks a
label, discovers a workflow by listing `agl.workflows`, and hands a built
`Deps` to that workflow's `Run`. `clean` removes what a label left behind —
worktrees, branches, and its run directory — tolerating anything already gone.
"""

import argparse
import asyncio
import importlib
import pkgutil
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from agl import workflows as workflows_pkg
from agl.config import ConfigError, ProjectConfig, agl_home, load_project
from agl.core import paths
from agl.core.agent.impl.claude_runner import ClaudeRunner
from agl.core.paths import InvalidNameError
from agl.core.store.impl.file_store import FileStore
from agl.core.terminal.impl.rich_terminal import RichTerminal
from agl.core.vcs import Vcs, VcsError
from agl.core.vcs.impl.git import Git

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return int(exit_.code) if exit_.code is not None else 0

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "clean":
        return _cmd_clean(args)
    raise AssertionError(f"unhandled command {args.command!r}")


# -- argument parsing -------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agl", description="Agentic Graph Loops.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="start a run of a workflow")
    run_parser.add_argument("workflow", help="the workflow to run, e.g. 'tickets'")
    run_parser.add_argument(
        "--name", help="the run's label; defaults to the current branch name"
    )
    run_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="how many tickets to work on at once (default: 3)",
    )
    run_parser.add_argument(
        "description", nargs="+", help="what to build, as free text; may start with '--'"
    )

    clean_parser = sub.add_parser("clean", help="remove a run's worktrees, branches, and files")
    clean_parser.add_argument("label", help="the run label to remove")

    return parser


# -- run ----------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        vcs: Vcs = Git(Path.cwd())
    except VcsError as error:
        return _fail(error)

    try:
        home = agl_home()
        config = load_project(home, vcs.root())
    except ConfigError as error:
        return _fail(error)

    label = args.name if args.name is not None else vcs.current_branch()
    try:
        paths.validate_label(label)
    except InvalidNameError as error:
        return _fail(error)

    available = _discover_workflows()
    if args.workflow not in available:
        listing = ", ".join(available) if available else "(none found)"
        return _fail(f"unknown workflow {args.workflow!r}; available: {listing}")

    module = importlib.import_module(f"agl.workflows.{args.workflow}.workflow")
    deps = _build_deps(module, home, label, config, vcs)
    description = " ".join(args.description)
    run_obj = module.Run(deps, label, description, args.max_concurrent)

    try:
        asyncio.run(run_obj.go())
    except KeyboardInterrupt:
        _print_interrupted(vcs, label)
        return 1
    except Exception as error:  # noqa: BLE001 - surfaced to the user, never a traceback
        return _fail(error)

    halt = getattr(getattr(run_obj, "state", None), "halt", None)
    if halt is not None:
        return _fail(getattr(halt, "reason", str(halt)))

    return 0


def _build_deps(
    module: ModuleType, home: Path, label: str, config: ProjectConfig, vcs: Vcs
) -> Any:
    return module.Deps(
        agent=ClaudeRunner(settings_path=None),
        vcs=vcs,
        store=FileStore(paths.run_dir(home, label)),
        terminal=RichTerminal(),
        config=config,
    )


def _discover_workflows() -> tuple[str, ...]:
    """Every workflow package under `agl.workflows`, sorted by name."""
    return tuple(
        sorted(info.name for info in pkgutil.iter_modules(workflows_pkg.__path__) if info.ispkg)
    )


def _print_interrupted(vcs: Vcs, label: str) -> None:
    namespace = f"{paths.branch_namespace(label)}/"
    trees = [w.path for w in vcs.list_worktrees() if w.branch.startswith(namespace)]
    print("\ninterrupted", file=sys.stderr)
    for tree in trees:
        print(f"  {tree}", file=sys.stderr)
    print(f"run `agl clean {label}` to remove them", file=sys.stderr)


# -- clean ----------------------------------------------------------------


def _cmd_clean(args: argparse.Namespace) -> int:
    label = args.label

    try:
        vcs: Vcs = Git(Path.cwd())
    except VcsError as error:
        return _fail(error)

    try:
        home = agl_home()
        config = load_project(home, vcs.root())
    except ConfigError as error:
        return _fail(error)

    try:
        paths.validate_label(label)
    except InvalidNameError as error:
        return _fail(error)

    namespace = paths.branch_namespace(label)
    run_directory = paths.run_dir(home, label)
    trees = paths.trees_dir(config.trees_root, config.name, label)
    tree_entries = sorted(trees.iterdir()) if trees.is_dir() else []
    branches = vcs.branches(namespace)

    if not branches and not tree_entries and not run_directory.exists():
        return _fail(f"unknown label {label!r}; runs found: {_existing_runs(home)}")

    removed_trees: list[Path] = []
    for entry in tree_entries:
        try:
            vcs.remove_worktree(entry, force=True)
            removed_trees.append(entry)
        except VcsError:
            pass
    vcs.prune_worktrees()

    removed_branches: list[str] = []
    for name in branches:
        try:
            vcs.delete_branch(name, force=True)
            removed_branches.append(name)
        except VcsError:
            pass

    removed_run_dir = run_directory.exists()
    shutil.rmtree(run_directory, ignore_errors=True)

    _report_clean(label, removed_trees, removed_branches, removed_run_dir)
    return 0


def _existing_runs(home: Path) -> str:
    runs_root = home / "runs"
    if not runs_root.is_dir():
        return "(none)"
    names = sorted(entry.name for entry in runs_root.iterdir() if entry.is_dir())
    return ", ".join(names) if names else "(none)"


def _report_clean(
    label: str, trees: list[Path], branches: list[str], removed_run_dir: bool
) -> None:
    print(f"cleaned {label!r}:")
    print(f"  worktrees: {', '.join(str(tree) for tree in trees) if trees else 'none'}")
    print(f"  branches: {', '.join(branches) if branches else 'none'}")
    print(f"  run directory: {'removed' if removed_run_dir else 'none'}")


# -- shared -----------------------------------------------------------------


def _fail(error: object) -> int:
    print(f"error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
