"""Project config: `AGL_HOME` resolution and `config.toml` loading."""

from pathlib import Path

import pytest

from agl.config import (
    ConfigError,
    ProjectConfig,
    agl_home,
    find_project_by_repo,
    load_project,
    resolve_agl_home,
)


def _write_config(path: Path, **overrides: object) -> Path:
    fields = {
        "name": "myproject",
        "repo": "/repo",
        "trees_root": "/trees",
        "build": ["./gradlew", "compileDebugKotlin"],
        "build_timeout": 900,
        **overrides,
    }
    lines = []
    for key, value in fields.items():
        if value is _OMIT:
            continue
        lines.append(f"{key} = {_toml(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _Omit:
    pass


_OMIT = _Omit()


def _toml(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    return str(value)


# -- agl_home ---------------------------------------------------------------


def test_agl_home_reads_the_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGL_HOME", str(tmp_path))
    assert agl_home() == tmp_path


def test_agl_home_unset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGL_HOME", raising=False)
    with pytest.raises(ConfigError, match="AGL_HOME"):
        agl_home()


def test_agl_home_nonexistent_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("AGL_HOME", str(missing))
    with pytest.raises(ConfigError, match="AGL_HOME"):
        agl_home()


# -- resolve_agl_home ---------------------------------------------------------


def test_resolve_agl_home_reads_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGL_HOME", str(tmp_path))
    assert resolve_agl_home() == tmp_path


def test_resolve_agl_home_unset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGL_HOME", raising=False)
    with pytest.raises(ConfigError, match="AGL_HOME"):
        resolve_agl_home()


def test_resolve_agl_home_does_not_require_the_directory_to_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "does-not-exist-yet"
    monkeypatch.setenv("AGL_HOME", str(missing))
    assert resolve_agl_home() == missing


# -- load_project: happy path ------------------------------------------------


def test_a_valid_config_loads_with_every_field(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    trees = tmp_path / "trees"
    config_path = _write_config(
        home / "projects" / "myproject" / "config.toml",
        repo=str(repo),
        trees_root=str(trees),
    )

    config = load_project(home, repo)

    assert config == ProjectConfig(
        name="myproject",
        repo=repo,
        trees_root=trees,
        build=("./gradlew", "compileDebugKotlin"),
        build_timeout=900,
        standards=config_path.parent / "standards.md",
        config_dir=config_path.parent,
    )


def test_resolution_finds_the_project_by_repo_not_directory_name(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _write_config(
        home / "projects" / "some-other-dir-name" / "config.toml",
        repo=str(repo),
        trees_root=str(tmp_path / "trees"),
    )

    config = load_project(home, repo)

    assert config.name == "myproject"


def test_two_projects_correct_one_chosen(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir(parents=True)
    repo_b.mkdir(parents=True)
    _write_config(
        home / "projects" / "a" / "config.toml",
        name="project-a",
        repo=str(repo_a),
        trees_root=str(tmp_path / "trees-a"),
    )
    _write_config(
        home / "projects" / "b" / "config.toml",
        name="project-b",
        repo=str(repo_b),
        trees_root=str(tmp_path / "trees-b"),
    )

    assert load_project(home, repo_b).name == "project-b"


# -- load_project: resolution errors -----------------------------------------


def test_no_match_raises_naming_the_repo(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "projects").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ConfigError, match=str(repo)):
        load_project(home, repo)


def test_no_match_names_the_search_path_and_agl_init(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "projects").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ConfigError) as excinfo:
        load_project(home, repo)

    message = str(excinfo.value)
    assert str(home / "projects") in message
    assert "agl init" in message


def test_two_configs_with_the_same_repo_raises_naming_both(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    first = _write_config(
        home / "projects" / "a" / "config.toml", repo=str(repo), trees_root=str(tmp_path / "t1")
    )
    second = _write_config(
        home / "projects" / "b" / "config.toml", repo=str(repo), trees_root=str(tmp_path / "t2")
    )

    with pytest.raises(ConfigError) as excinfo:
        load_project(home, repo)

    assert str(first) in str(excinfo.value)
    assert str(second) in str(excinfo.value)


# -- load_project: field validation ------------------------------------------


@pytest.mark.parametrize("key", ["name", "repo", "trees_root", "build", "build_timeout"])
def test_missing_required_key_raises_naming_key_and_file(tmp_path: Path, key: str) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    overrides: dict[str, object] = {
        "repo": str(repo),
        "trees_root": str(tmp_path / "trees"),
        key: _OMIT,
    }
    config_path = _write_config(home / "projects" / "myproject" / "config.toml", **overrides)

    with pytest.raises(ConfigError) as excinfo:
        load_project(home, repo)

    assert key in str(excinfo.value)
    assert str(config_path) in str(excinfo.value)


def test_relative_repo_raises(tmp_path: Path) -> None:
    home = tmp_path / "home"
    config_path = _write_config(
        home / "projects" / "myproject" / "config.toml",
        repo="relative/repo",
        trees_root=str(tmp_path / "trees"),
    )

    with pytest.raises(ConfigError) as excinfo:
        load_project(home, tmp_path / "repo")

    assert "repo" in str(excinfo.value)
    assert str(config_path) in str(excinfo.value)


def test_relative_trees_root_raises(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    config_path = _write_config(
        home / "projects" / "myproject" / "config.toml",
        repo=str(repo),
        trees_root="relative/trees",
    )

    with pytest.raises(ConfigError) as excinfo:
        load_project(home, repo)

    assert "trees_root" in str(excinfo.value)
    assert str(config_path) in str(excinfo.value)


def test_empty_build_raises(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    config_path = _write_config(
        home / "projects" / "myproject" / "config.toml",
        repo=str(repo),
        trees_root=str(tmp_path / "trees"),
        build=[],
    )

    with pytest.raises(ConfigError) as excinfo:
        load_project(home, repo)

    assert "build" in str(excinfo.value)
    assert str(config_path) in str(excinfo.value)


def test_malformed_toml_raises_naming_the_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    config_path = home / "projects" / "myproject" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("this is not [valid toml", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ConfigError) as excinfo:
        load_project(home, repo)

    assert str(config_path) in str(excinfo.value)


# -- find_project_by_repo -----------------------------------------------------


def test_find_project_by_repo_returns_none_when_nothing_matches(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "projects").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()

    assert find_project_by_repo(home, repo) is None


def test_find_project_by_repo_returns_none_when_projects_dir_is_absent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir()

    assert find_project_by_repo(home, repo) is None


def test_find_project_by_repo_finds_the_matching_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    config_path = _write_config(
        home / "projects" / "myproject" / "config.toml",
        repo=str(repo),
        trees_root=str(tmp_path / "trees"),
    )

    assert find_project_by_repo(home, repo) == config_path
