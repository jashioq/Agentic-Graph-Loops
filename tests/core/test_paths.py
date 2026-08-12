"""Pure layout arithmetic: no fixtures, no `tmp_path`, nothing on disk."""

from pathlib import Path

import pytest

from agl.core.paths import (
    InvalidLabelError,
    branch_namespace,
    bug_branch,
    project_config,
    project_dir,
    project_standards,
    run_dir,
    ticket_branch,
    trees_dir,
    validate_label,
    worktree_dir,
)

HOME = Path("/home/jan/.agl")
TREES = Path("/home/jan/.agl-trees")


# -- the home layout ------------------------------------------------------


def test_project_dir() -> None:
    assert project_dir(HOME, "acme-api") == Path("/home/jan/.agl/projects/acme-api")


def test_project_config_is_config_toml_in_the_project_dir() -> None:
    assert project_config(HOME, "acme-api") == project_dir(HOME, "acme-api") / "config.toml"


def test_project_standards_is_standards_md_in_the_project_dir() -> None:
    assert project_standards(HOME, "acme-api") == project_dir(HOME, "acme-api") / "standards.md"


def test_run_dir() -> None:
    assert run_dir(HOME, "add-auth") == Path("/home/jan/.agl/runs/add-auth")


def test_runs_and_projects_are_separate_trees() -> None:
    assert not run_dir(HOME, "add-auth").is_relative_to(project_dir(HOME, "acme-api"))


def test_two_runs_get_different_directories() -> None:
    assert run_dir(HOME, "add-auth") != run_dir(HOME, "add-billing")


def test_a_relative_home_stays_relative() -> None:
    assert run_dir(Path("agl-home"), "add-auth") == Path("agl-home/runs/add-auth")


# -- worktrees ------------------------------------------------------------


def test_trees_dir() -> None:
    assert trees_dir(TREES, "acme-api", "add-auth") == Path(
        "/home/jan/.agl-trees/acme-api/add-auth"
    )


def test_worktree_dir() -> None:
    assert worktree_dir(TREES, "acme-api", "add-auth", "T-03") == Path(
        "/home/jan/.agl-trees/acme-api/add-auth/T-03"
    )


def test_two_tickets_get_different_worktrees_under_the_same_trees_dir() -> None:
    first = worktree_dir(TREES, "acme-api", "add-auth", "T-03")
    second = worktree_dir(TREES, "acme-api", "add-auth", "T-04")
    assert first != second
    parent = trees_dir(TREES, "acme-api", "add-auth")
    assert first.parent == parent
    assert second.parent == parent


def test_two_runs_of_one_project_get_different_trees_dirs() -> None:
    assert trees_dir(TREES, "acme-api", "add-auth") != trees_dir(TREES, "acme-api", "add-billing")


def test_two_projects_get_different_trees_dirs() -> None:
    assert trees_dir(TREES, "acme-api", "add-auth") != trees_dir(TREES, "acme-web", "add-auth")


def test_worktrees_live_under_the_trees_root_not_the_agl_home() -> None:
    tree = worktree_dir(TREES, "acme-api", "add-auth", "T-03")
    assert tree.is_relative_to(TREES)
    assert not tree.is_relative_to(HOME)


# -- branches -------------------------------------------------------------


def test_ticket_branch() -> None:
    assert ticket_branch("add-auth", "T-03") == "agl/add-auth/T-03"


def test_bug_branch() -> None:
    assert bug_branch("add-auth", "T-03", 1) == "agl/add-auth/T-03-bug-1"


def test_a_bug_branch_is_not_a_path_child_of_its_parent_branch() -> None:
    # Git stores refs as files: `agl/add-auth/T-03` being a file means nothing
    # can live at `agl/add-auth/T-03/...`. Bug branches are siblings for that
    # reason, and this test is the guard on it.
    parent = ticket_branch("add-auth", "T-03")
    for n in (1, 2, 17):
        assert not bug_branch("add-auth", "T-03", n).startswith(parent + "/")


def test_bug_branches_are_flat_below_the_label() -> None:
    assert bug_branch("add-auth", "T-03", 1).count("/") == ticket_branch("add-auth", "T-03").count(
        "/"
    )


def test_bug_branch_numbers_are_distinct() -> None:
    assert bug_branch("add-auth", "T-03", 1) != bug_branch("add-auth", "T-03", 2)


def test_bug_branches_of_different_tickets_differ() -> None:
    assert bug_branch("add-auth", "T-03", 1) != bug_branch("add-auth", "T-04", 1)


def test_branch_namespace() -> None:
    assert branch_namespace("add-auth") == "agl/add-auth"


def test_branch_namespace_is_a_prefix_of_both_branch_kinds() -> None:
    namespace = branch_namespace("add-auth")
    assert ticket_branch("add-auth", "T-03").startswith(namespace + "/")
    assert bug_branch("add-auth", "T-03", 1).startswith(namespace + "/")


def test_the_agl_prefix_keeps_a_user_branch_named_like_the_label_clear() -> None:
    # A user's own `add-auth` branch must not sit inside what a clean deletes.
    assert not "add-auth".startswith(branch_namespace("add-auth"))
    assert branch_namespace("add-auth") != "add-auth"


def test_two_labels_get_different_namespaces() -> None:
    assert branch_namespace("add-auth") != branch_namespace("add-billing")


# -- label validation -----------------------------------------------------

VALID_LABELS = ["a", "add-auth", "t3", "0", "add-auth-2", "a-b-c-d"]

INVALID_LABELS = [
    "",
    "Add-Auth",
    "ADDAUTH",
    "-add-auth",
    "add auth",
    "add/auth",
    "add.auth",
    "add..auth",
    "..",
    "add_auth",
    "add-auth/",
    "/add-auth",
    "add\\auth",
    "add~auth",
    "add:auth",
    "añadir",
    " add-auth",
    "add-auth ",
    "add-auth\n",
]


@pytest.mark.parametrize("label", VALID_LABELS)
def test_valid_labels_pass(label: str) -> None:
    assert validate_label(label) is None


@pytest.mark.parametrize("label", INVALID_LABELS)
def test_invalid_labels_raise(label: str) -> None:
    with pytest.raises(InvalidLabelError):
        validate_label(label)


def test_invalid_label_error_is_an_exception() -> None:
    assert issubclass(InvalidLabelError, Exception)


def test_the_error_names_the_label() -> None:
    with pytest.raises(InvalidLabelError, match="Add-Auth"):
        validate_label("Add-Auth")


# -- purity ---------------------------------------------------------------


def test_path_functions_are_repeatable_and_create_nothing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    trees = tmp_path / "trees"
    calls = [
        lambda: project_dir(home, "acme-api"),
        lambda: project_config(home, "acme-api"),
        lambda: project_standards(home, "acme-api"),
        lambda: run_dir(home, "add-auth"),
        lambda: trees_dir(trees, "acme-api", "add-auth"),
        lambda: worktree_dir(trees, "acme-api", "add-auth", "T-03"),
    ]
    for call in calls:
        assert call() == call()
        assert not call().exists()
    assert list(tmp_path.iterdir()) == []


def test_the_module_does_no_io() -> None:
    import agl.core.paths as paths

    source = Path(str(paths.__file__)).read_text(encoding="utf-8")
    for forbidden in ("open(", "mkdir", "exists(", "os.", "shutil"):
        assert forbidden not in source
