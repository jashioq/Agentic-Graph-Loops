"""Project configuration: `AGL_HOME` and one project's `config.toml`.

Layer: cli-only, alongside core but not of it — the one place that knows
`config.toml` as a file format. AGL lives outside the projects it drives::

    $AGL_HOME/projects/<dir>/config.toml
    $AGL_HOME/projects/<dir>/standards.md
    $AGL_HOME/runs/<label>/

`<dir>` is the user's own organisation, not identity: a project is identified by
its `repo` field, so loading one is a scan over every `config.toml`, and zero or
several matches are both errors a person can act on.
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ConfigError",
    "ProjectConfig",
    "agl_home",
    "find_project_by_repo",
    "load_project",
    "resolve_agl_home",
]

_REQUIRED_KEYS = ("name", "repo", "trees_root", "build", "build_timeout")


class ConfigError(Exception):
    """Anything wrong with `AGL_HOME` or a project's `config.toml`."""


@dataclass(frozen=True)
class ProjectConfig:
    """One project's configuration, resolved from its `config.toml`."""

    name: str
    repo: Path
    trees_root: Path
    build: tuple[str, ...]
    build_timeout: float
    standards: Path
    config_dir: Path


def agl_home() -> Path:
    """`AGL_HOME` from the environment. Raises `ConfigError` unset or missing."""
    home = resolve_agl_home()
    if not home.is_dir():
        raise ConfigError(f"AGL_HOME is set to {home}, which does not exist")
    return home


def resolve_agl_home() -> Path:
    """`AGL_HOME` from the environment, without requiring it to exist yet.

    For `agl init`, which creates it. Every other caller wants `agl_home`.
    """
    raw = os.environ.get("AGL_HOME")
    if not raw:
        raise ConfigError("AGL_HOME is not set; set it to AGL's home directory")
    return Path(raw)


def load_project(home: Path, repo_root: Path) -> ProjectConfig:
    """The project whose `repo` matches `repo_root`, scanning `projects/*/config.toml`.

    return: ProjectConfig - raises `ConfigError` on no match or several, naming them
    """
    repo_root = repo_root.resolve()
    projects_dir = home / "projects"
    matches: list[tuple[Path, ProjectConfig]] = []
    for config_path in sorted(projects_dir.glob("*/config.toml")):
        config = _load_one(config_path)
        if config.repo.resolve() == repo_root:
            matches.append((config_path, config))

    if not matches:
        raise ConfigError(
            f"No project configured for {repo_root}\n\n"
            f"Looked in {projects_dir}/*/config.toml\n"
            "Run `agl init` here to create one."
        )
    if len(matches) > 1:
        found = ", ".join(str(path) for path, _ in matches)
        raise ConfigError(f"multiple projects configured for {repo_root}: {found}")
    return matches[0][1]


def find_project_by_repo(home: Path, repo_root: Path) -> Path | None:
    """The `config.toml` that already claims `repo_root`, if any.

    For `agl init`, so a duplicate repo is caught before it becomes a failed run.
    """
    repo_root = repo_root.resolve()
    projects_dir = home / "projects"
    if not projects_dir.is_dir():
        return None
    for config_path in sorted(projects_dir.glob("*/config.toml")):
        config = _load_one(config_path)
        if config.repo.resolve() == repo_root:
            return config_path
    return None


def _load_one(config_path: Path) -> ProjectConfig:
    """Parse and validate one `config.toml`. Raises `ConfigError` naming the file."""
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{config_path}: malformed TOML: {error}") from error

    for key in _REQUIRED_KEYS:
        if key not in data:
            raise ConfigError(f"{config_path}: missing required key {key!r}")

    name = _require_str(data, "name", config_path)
    repo = _require_absolute_path(data, "repo", config_path)
    trees_root = _require_absolute_path(data, "trees_root", config_path)
    build = _require_build(data, config_path)
    build_timeout = _require_number(data, "build_timeout", config_path)

    config_dir = config_path.parent
    return ProjectConfig(
        name=name,
        repo=repo,
        trees_root=trees_root,
        build=build,
        build_timeout=build_timeout,
        standards=config_dir / "standards.md",
        config_dir=config_dir,
    )


def _require_str(data: dict[str, object], key: str, config_path: Path) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{config_path}: {key!r} must be a non-empty string")
    return value


def _require_absolute_path(data: dict[str, object], key: str, config_path: Path) -> Path:
    value = _require_str(data, key, config_path)
    path = Path(value)
    if not path.is_absolute():
        raise ConfigError(f"{config_path}: {key!r} must be an absolute path, got {value!r}")
    return path


def _require_number(data: dict[str, object], key: str, config_path: Path) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{config_path}: {key!r} must be a number")
    return float(value)


def _require_build(data: dict[str, object], config_path: Path) -> tuple[str, ...]:
    value = data["build"]
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{config_path}: 'build' must be a non-empty list of strings")
    return tuple(value)
