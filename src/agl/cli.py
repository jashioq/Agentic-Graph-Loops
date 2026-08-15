"""`agl run`, `agl clean`, and `agl init` — the composition root.

Layer: cli. This is the only file that constructs concrete `impl` classes;
everything below it takes them injected. `run` resolves a project, takes a
label via `--name`/`-n`, discovers a workflow by listing `agl.workflows`, and hands a built
`Deps` to that workflow's `Run`. `clean` removes what a label left behind —
worktrees, branches, and its run directory — tolerating anything already gone.
`init` writes the `config.toml` and `standards.md` a project needs before
`run` can find it.
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
from agl.config import (
    ConfigError,
    ProjectConfig,
    agl_home,
    find_project_by_repo,
    load_project,
    resolve_agl_home,
)
from agl.core.agent.impl.claude_runner import ClaudeRunner
from agl.core.store.impl.file_store import FileStore
from agl.core.terminal.impl.rich_terminal import RichTerminal
from agl.core.vcs import Vcs, VcsError
from agl.core.vcs.impl.git import Git
from agl.runtime import paths
from agl.runtime.paths import InvalidNameError

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
    if args.command == "init":
        return _cmd_init(args)
    raise AssertionError(f"unhandled command {args.command!r}")


# -- argument parsing -------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agl", description="Agentic Graph Loops.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="start a run of a workflow")
    run_parser.add_argument("workflow", help="the workflow to run, e.g. 'tickets'")
    run_parser.add_argument(
        "-n", "--name", required=True, help="the run's label"
    )
    run_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="how many tickets to work on at once (default: 3)",
    )
    run_parser.add_argument(
        "description",
        nargs="*",
        help=(
            "what to build, as free text; may start with '--'. "
            "Reads from stdin if omitted."
        ),
    )

    clean_parser = sub.add_parser("clean", help="remove a run's worktrees, branches, and files")
    clean_parser.add_argument("label", help="the run label to remove")

    init_parser = sub.add_parser(
        "init", help="create the project configuration for the repo in the current directory"
    )
    init_parser.add_argument(
        "-n", "--name",
        help="how to file this project under AGL_HOME; defaults to the repo's directory name",
    )

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

    label = args.name
    try:
        paths.validate_label(label)
    except InvalidNameError as error:
        return _fail(error)

    available = _discover_workflows()
    if args.workflow not in available:
        listing = ", ".join(available) if available else "(none found)"
        return _fail(f"unknown workflow {args.workflow!r}; available: {listing}")

    description = _read_description(args)
    if description is None:
        return _fail("give a description, either as an argument or piped on stdin")

    module = importlib.import_module(f"agl.workflows.{args.workflow}.workflow")
    deps = _build_deps(module, home, label, config, vcs)
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


def _read_description(args: argparse.Namespace) -> str | None:
    """The run description, or `None` if none was given.

    A positional description is used as-is and stdin is left untouched. Absent
    that, stdin is read whole — unless it is a terminal, which would otherwise
    leave a person waiting at an empty prompt with no sign it is waiting.
    """
    if args.description:
        text = " ".join(args.description)
    elif sys.stdin.isatty():
        return None
    else:
        text = sys.stdin.read()
    return text if text.strip() else None


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


# -- init ---------------------------------------------------------------

_BUILD_GUESSES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "gradlew",
        ("./gradlew", "compileDebugKotlin"),
        "Detected `gradlew` at the repo root. Adjust the task if this isn't\n"
        "# what should gate a merge.",
    ),
    (
        "Cargo.toml",
        ("cargo", "check"),
        "Detected `Cargo.toml` at the repo root.",
    ),
    (
        "package.json",
        ("npm", "run", "build"),
        "Detected `package.json` at the repo root.",
    ),
)

_BUILD_PLACEHOLDER: tuple[str, ...] = (
    "sh",
    "-c",
    "echo 'agl: no build command configured; set build in config.toml' >&2; exit 1",
)

_BUILD_PLACEHOLDER_COMMENT = (
    "No recognised build tool (gradlew, Cargo.toml, package.json) was found,\n"
    "# so this is a placeholder: it fails loudly on the first merge instead of\n"
    "# quietly passing. Set it to the real build or check command for this repo."
)


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        vcs: Vcs = Git(Path.cwd())
    except VcsError as error:
        return _fail(error)

    try:
        home = resolve_agl_home()
    except ConfigError as error:
        return _fail(error)

    repo_root = vcs.root()
    name = args.name if args.name is not None else repo_root.name
    try:
        paths.validate_project(name)
    except InvalidNameError as error:
        return _fail(error)

    project_dir = paths.project_dir(home, name)
    if project_dir.exists():
        return _fail(
            f"{project_dir} already exists; edit it directly or pass a different --name"
        )

    duplicate = find_project_by_repo(home, repo_root)
    if duplicate is not None:
        return _fail(f"{repo_root} is already configured in {duplicate}")

    build, build_note = _guess_build(repo_root)

    project_dir.mkdir(parents=True)
    config_path = project_dir / "config.toml"
    standards_path = project_dir / "standards.md"
    trees_root = repo_root.parent / ".trees"
    config_path.write_text(
        _render_config(name, repo_root, trees_root, build, build_note), encoding="utf-8"
    )
    standards_path.write_text(_STANDARDS_TEMPLATE, encoding="utf-8")
    (home / "runs").mkdir(parents=True, exist_ok=True)

    _report_init(project_dir, build, build_note)
    return 0


def _guess_build(repo_root: Path) -> tuple[tuple[str, ...], str]:
    """A build command guessed from a marker file, or a placeholder.

    The placeholder is deliberately a command that fails loudly on the first
    merge rather than one that looks plausible and quietly does the wrong
    thing.
    """
    for marker, build, comment in _BUILD_GUESSES:
        if (repo_root / marker).exists():
            return build, comment
    return _BUILD_PLACEHOLDER, _BUILD_PLACEHOLDER_COMMENT


def _render_config(
    name: str, repo_root: Path, trees_root: Path, build: tuple[str, ...], build_comment: str
) -> str:
    build_toml = "[" + ", ".join(_toml_str(part) for part in build) + "]"
    return (
        "# Identifies this project under AGL_HOME. `agl run` finds it by\n"
        "# matching `repo` below, not by this name or the directory it lives in.\n"
        f"name = {_toml_str(name)}\n"
        "\n"
        "# The git repository this project drives.\n"
        f"repo = {_toml_str(str(repo_root))}\n"
        "\n"
        "# Where one worktree per ticket is checked out while a run is in\n"
        "# progress. A sibling of the repo, never inside it, so the repo's own\n"
        "# working tree stays clean and editors/watchers don't index it three\n"
        "# times over.\n"
        f"trees_root = {_toml_str(str(trees_root))}\n"
        "\n"
        "# The command that gates every merge into the base branch, run from the\n"
        f"# repo root. {build_comment}\n"
        f"build = {build_toml}\n"
        "\n"
        "# Seconds to let one build run before it counts as a failure.\n"
        "build_timeout = 900\n"
    )


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


_STANDARDS_TEMPLATE = """\
# Standards

## Architecture

## Naming

## Testing

## Dependencies
"""


def _report_init(project_dir: Path, build: tuple[str, ...], build_note: str) -> None:
    print(f"Created {_display_path(project_dir)}/")
    if build == _BUILD_PLACEHOLDER:
        print("  config.toml    build: not detected — edit config.toml before running")
    else:
        print(f"  config.toml    build: {' '.join(build)}")
    print("  standards.md   empty")
    print()
    print(
        "standards.md is what the quality reviewer reviews against. While it is empty\n"
        "that reviewer has nothing specific to check and will fall back to generic code\n"
        "review — it will still run and still file findings, they just will not be about\n"
        "your conventions. Fill it in before relying on it."
    )
    print()
    print('Next: agl run tickets --name my-run "what you want built"')


def _display_path(path: Path) -> str:
    """`path`, with a leading `$HOME` collapsed to `~` for readability."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


# -- shared -----------------------------------------------------------------


def _fail(error: object) -> int:
    print(f"error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
