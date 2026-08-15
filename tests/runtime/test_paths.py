"""Pure layout arithmetic: no fixtures, no `tmp_path`, nothing on disk."""

from pathlib import Path

import pytest

from agl.runtime.paths import (
    BRANCH_PREFIX,
    InvalidNameError,
    branch,
    branch_namespace,
    project_config,
    project_dir,
    run_dir,
    trees_dir,
    validate_label,
    validate_node_id,
    validate_project,
    worktree_dir,
)

HOME = Path("/home/jan/.agl")
TREES = Path("/home/jan/.agl-trees")


# -- the home layout ------------------------------------------------------


def test_project_dir() -> None:
    assert project_dir(HOME, "acme-api") == Path("/home/jan/.agl/projects/acme-api")


def test_project_config_is_config_toml_in_the_project_dir() -> None:
    assert project_config(HOME, "acme-api") == project_dir(HOME, "acme-api") / "config.toml"


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


def test_two_nodes_get_different_worktrees_under_the_same_trees_dir() -> None:
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


def test_branch() -> None:
    assert branch("add-auth", "T-03") == "agl/add-auth/T-03"


def test_branch_prefix_is_exported_and_starts_every_namespace() -> None:
    assert BRANCH_PREFIX == "agl"
    assert branch_namespace("add-auth").startswith(BRANCH_PREFIX + "/")


def test_a_composed_node_id_is_a_sibling_not_a_path_child() -> None:
    # Git stores refs as files: `agl/add-auth/T-03` being a file means nothing
    # can live at `agl/add-auth/T-03/...`. A workflow composing a derived id
    # hyphenates for that reason, and node-id validation — which refuses `/` —
    # is what keeps the composed branch flat. This test is the guard on it.
    parent = branch("add-auth", "T-03")
    for composed in ("T-03-bug-1", "T-03-bug-2", "T-03-followup"):
        assert not branch("add-auth", composed).startswith(parent + "/")
        assert branch("add-auth", composed).count("/") == parent.count("/")


def test_two_node_ids_get_different_branches() -> None:
    assert branch("add-auth", "T-03") != branch("add-auth", "T-04")


def test_branch_namespace() -> None:
    assert branch_namespace("add-auth") == "agl/add-auth"


def test_branch_namespace_is_a_prefix_of_every_branch() -> None:
    namespace = branch_namespace("add-auth")
    for node_id in ("T-03", "T-03-bug-1"):
        assert branch("add-auth", node_id).startswith(namespace + "/")


def test_the_agl_prefix_keeps_a_user_branch_named_like_the_label_clear() -> None:
    # A user's own `add-auth` branch must not sit inside what a clean deletes.
    assert not "add-auth".startswith(branch_namespace("add-auth"))
    assert branch_namespace("add-auth") != "add-auth"


def test_two_labels_get_different_namespaces() -> None:
    assert branch_namespace("add-auth") != branch_namespace("add-billing")


# -- validation -----------------------------------------------------------

VALID_LABELS = ["a", "add-auth", "t3", "0", "add-auth-2", "a-b-c-d"]

# Every one of these is illegal for a label, a project, and a node id alike:
# each would either escape its directory, nest a ref, or arrive at git as
# syntax rather than as a name.
INVALID_NAMES = [
    "",
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

# Uppercase is refused for the two names a person supplies, because a
# case-insensitive filesystem would fold `Add-Auth` onto `add-auth` and hand
# two runs the same directory. A node id is composed by a workflow from one
# scheme, so `T-03` is a name and not a collision.
UPPERCASE_NAMES = ["Add-Auth", "ADDAUTH"]


@pytest.mark.parametrize("label", VALID_LABELS)
def test_valid_labels_pass(label: str) -> None:
    assert validate_label(label) is None


@pytest.mark.parametrize("name", INVALID_NAMES)
def test_an_illegal_name_raises_for_every_kind(name: str) -> None:
    for validate in (validate_label, validate_project, validate_node_id):
        with pytest.raises(InvalidNameError):
            validate(name)


@pytest.mark.parametrize("name", UPPERCASE_NAMES)
def test_uppercase_is_refused_for_labels_and_projects(name: str) -> None:
    for validate in (validate_label, validate_project):
        with pytest.raises(InvalidNameError):
            validate(name)


@pytest.mark.parametrize("node_id", ["T-03", "T-03-bug-1", "t-03", "0", "N1"])
def test_valid_node_ids_pass(node_id: str) -> None:
    assert validate_node_id(node_id) is None


def test_invalid_name_error_is_an_exception() -> None:
    assert issubclass(InvalidNameError, Exception)


def test_the_error_names_the_offending_value() -> None:
    with pytest.raises(InvalidNameError, match="Add-Auth"):
        validate_label("Add-Auth")


def test_the_error_names_which_kind_was_wrong() -> None:
    with pytest.raises(InvalidNameError, match="node id"):
        validate_node_id("T 03")
    with pytest.raises(InvalidNameError, match="project"):
        validate_project("acme/api")


# -- every function that takes a name validates it -------------------------


def test_a_label_is_validated_wherever_it_is_taken() -> None:
    for call in (
        lambda: run_dir(HOME, "../escape"),
        lambda: trees_dir(TREES, "acme-api", "../escape"),
        lambda: worktree_dir(TREES, "acme-api", "../escape", "T-03"),
        lambda: branch_namespace("../escape"),
        lambda: branch("../escape", "T-03"),
    ):
        with pytest.raises(InvalidNameError):
            call()


def test_a_project_is_validated_wherever_it_is_taken() -> None:
    for call in (
        lambda: project_dir(HOME, "../escape"),
        lambda: project_config(HOME, "../escape"),
        lambda: trees_dir(TREES, "../escape", "add-auth"),
        lambda: worktree_dir(TREES, "../escape", "add-auth", "T-03"),
    ):
        with pytest.raises(InvalidNameError):
            call()


def test_a_node_id_is_validated_wherever_it_is_taken() -> None:
    for node_id in ("../escape", "T-03/bug-1", ".."):
        with pytest.raises(InvalidNameError):
            worktree_dir(TREES, "acme-api", "add-auth", node_id)
        with pytest.raises(InvalidNameError):
            branch("add-auth", node_id)


# -- purity ---------------------------------------------------------------


def test_path_functions_are_repeatable_and_create_nothing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    trees = tmp_path / "trees"
    calls = [
        lambda: project_dir(home, "acme-api"),
        lambda: project_config(home, "acme-api"),
        lambda: run_dir(home, "add-auth"),
        lambda: trees_dir(trees, "acme-api", "add-auth"),
        lambda: worktree_dir(trees, "acme-api", "add-auth", "T-03"),
    ]
    for call in calls:
        assert call() == call()
        assert not call().exists()
    assert list(tmp_path.iterdir()) == []


def test_the_module_does_no_io() -> None:
    import agl.runtime.paths as paths

    source = Path(str(paths.__file__)).read_text(encoding="utf-8")
    for forbidden in ("open(", "mkdir", "exists(", "os.", "shutil"):
        assert forbidden not in source


def test_the_module_speaks_no_workflow_vocabulary() -> None:
    import agl.runtime.paths as paths

    source = Path(str(paths.__file__)).read_text(encoding="utf-8").lower()
    for word in ("ticket", "bug", "spec", "standards"):
        assert word not in source, f"{word!r} leaked into paths.py"
