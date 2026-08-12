"""Documents on disk, against a real filesystem in `tmp_path`."""

from pathlib import Path

import pytest

from agl.core.store import InvalidKeyError, MissingKeyError
from agl.core.store.impl.file_store import FileStore


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    return FileStore(tmp_path / "run")


# -- text round trips -----------------------------------------------------


def test_write_then_read_returns_the_same_text(store: FileStore) -> None:
    store.write("spec.md", "# Add auth\n")
    assert store.read("spec.md") == "# Add auth\n"


def test_non_ascii_text_round_trips(store: FileStore) -> None:
    content = "Ünïcödé — ✅ 日本語 🎫\n"
    store.write("spec.md", content)
    assert store.read("spec.md") == content


def test_the_file_on_disk_is_utf8(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    store.write("spec.md", "café ✅")
    assert (tmp_path / "spec.md").read_bytes() == "café ✅".encode()


def test_writing_an_empty_document_is_allowed(store: FileStore) -> None:
    store.write("empty.md", "")
    assert store.read("empty.md") == ""
    assert store.exists("empty.md")


def test_overwriting_replaces_rather_than_appends(store: FileStore) -> None:
    store.write("spec.md", "first version\n")
    store.write("spec.md", "second\n")
    assert store.read("spec.md") == "second\n"


# -- the root -------------------------------------------------------------


def test_the_root_is_created_if_absent(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "add-auth"
    assert not root.exists()
    FileStore(root)
    assert root.is_dir()


def test_an_existing_root_with_documents_is_kept(tmp_path: Path) -> None:
    (tmp_path / "spec.md").write_text("already here\n", encoding="utf-8")
    assert FileStore(tmp_path).read("spec.md") == "already here\n"


def test_nested_keys_create_intermediate_directories(store: FileStore) -> None:
    store.write("reviews/round-1/T-03.md", "looks good\n")
    assert store.read("reviews/round-1/T-03.md") == "looks good\n"
    assert (store.root / "reviews" / "round-1").is_dir()


# -- missing keys ---------------------------------------------------------


def test_read_on_a_missing_key_raises(store: FileStore) -> None:
    with pytest.raises(MissingKeyError):
        store.read("nope.md")


def test_read_on_a_directory_raises_missing(store: FileStore) -> None:
    store.write("reviews/T-03.md", "x")
    with pytest.raises(MissingKeyError):
        store.read("reviews")


def test_delete_removes_the_file(store: FileStore) -> None:
    store.write("spec.md", "x")
    store.delete("spec.md")
    assert not (store.root / "spec.md").exists()


def test_a_second_delete_raises(store: FileStore) -> None:
    store.write("spec.md", "x")
    store.delete("spec.md")
    with pytest.raises(MissingKeyError):
        store.delete("spec.md")


def test_delete_on_a_missing_key_raises(store: FileStore) -> None:
    with pytest.raises(MissingKeyError):
        store.delete("nope.md")


# -- exists ---------------------------------------------------------------


def test_exists_tracks_write_and_delete(store: FileStore) -> None:
    assert not store.exists("spec.md")
    store.write("spec.md", "x")
    assert store.exists("spec.md")
    store.delete("spec.md")
    assert not store.exists("spec.md")


def test_exists_is_false_for_a_directory(store: FileStore) -> None:
    store.write("reviews/T-03.md", "x")
    assert not store.exists("reviews")


# -- list -----------------------------------------------------------------


def test_list_on_an_empty_store_is_empty(store: FileStore) -> None:
    assert store.list() == ()


def test_list_is_sorted_and_recursive(store: FileStore) -> None:
    for key in ("tickets.json", "spec.md", "reviews/T-03.md", "reviews/T-01.md"):
        store.write(key, "x")
    assert store.list() == ("reviews/T-01.md", "reviews/T-03.md", "spec.md", "tickets.json")


def test_list_filters_by_prefix(store: FileStore) -> None:
    for key in ("spec.md", "reviews/T-01.md", "reviews/T-03.md"):
        store.write(key, "x")
    assert store.list("reviews/") == ("reviews/T-01.md", "reviews/T-03.md")
    assert store.list("spec") == ("spec.md",)
    assert store.list("nothing") == ()


def test_list_returns_keys_not_paths(store: FileStore) -> None:
    store.write("reviews/T-03.md", "x")
    assert store.list() == ("reviews/T-03.md",)


def test_list_omits_directories(store: FileStore) -> None:
    store.write("reviews/T-03.md", "x")
    assert "reviews" not in store.list()


# -- json -----------------------------------------------------------------


def test_json_round_trips_a_dict(store: FileStore) -> None:
    store.write_json("tickets.json", {"id": "T-03", "done": False})
    assert store.read_json("tickets.json") == {"id": "T-03", "done": False}


def test_json_round_trips_a_list(store: FileStore) -> None:
    store.write_json("order.json", ["T-01", "T-02"])
    assert store.read_json("order.json") == ["T-01", "T-02"]


def test_json_round_trips_nested_structures(store: FileStore) -> None:
    value = {
        "tickets": [
            {"id": "T-01", "blockers": [], "meta": {"agent": "planner"}},
            {"id": "T-02", "blockers": ["T-01"], "meta": None},
        ],
        "count": 2,
    }
    store.write_json("tickets.json", value)
    assert store.read_json("tickets.json") == value


def test_json_round_trips_non_ascii(store: FileStore) -> None:
    store.write_json("tickets.json", {"title": "café ✅"})
    assert store.read_json("tickets.json") == {"title": "café ✅"}


def test_read_json_on_a_missing_key_raises_missing(store: FileStore) -> None:
    with pytest.raises(MissingKeyError):
        store.read_json("nope.json")


# -- keys -----------------------------------------------------------------

INVALID_KEYS = [
    "",
    "/spec.md",
    "/",
    "..",
    "../spec.md",
    "reviews/../../spec.md",
    "a/../../etc/passwd",
    "reviews/..",
    "reviews\\T-03.md",
    "spec\0.md",
]


@pytest.mark.parametrize("key", INVALID_KEYS)
def test_invalid_keys_are_rejected_by_every_method(store: FileStore, key: str) -> None:
    for call in (
        lambda: store.read(key),
        lambda: store.write(key, "x"),
        lambda: store.delete(key),
        lambda: store.exists(key),
    ):
        with pytest.raises(InvalidKeyError):
            call()


def test_the_escape_attempt_writes_nothing_outside_the_root(tmp_path: Path) -> None:
    store = FileStore(tmp_path / "run")
    outside = tmp_path / "run" / ".." / ".." / "etc"
    with pytest.raises(InvalidKeyError):
        store.write("a/../../etc/passwd", "pwned")
    assert not outside.exists()


def test_a_symlink_pointing_out_of_the_root_is_rejected(tmp_path: Path) -> None:
    # Syntactically clean, still outside once resolved — which is the check
    # that actually matters.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret\n", encoding="utf-8")
    store = FileStore(tmp_path / "run")
    (store.root / "escape").symlink_to(outside)
    with pytest.raises(InvalidKeyError):
        store.read("escape/secret.md")


def test_nested_keys_are_valid(store: FileStore) -> None:
    store.write("reviews/round-1/T-03.md", "x")
    assert store.exists("reviews/round-1/T-03.md")


def test_a_key_containing_a_dotdot_substring_is_valid(store: FileStore) -> None:
    # `..` is only an escape when it is a whole segment.
    store.write("T-03..md", "x")
    assert store.read("T-03..md") == "x"


# -- atomicity ------------------------------------------------------------


def test_a_write_that_fails_partway_leaves_the_previous_contents(
    store: FileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write("tickets.json", "original\n")

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(OSError):
        store.write("tickets.json", "replacement\n")
    monkeypatch.undo()
    assert store.read("tickets.json") == "original\n"


def test_a_failed_write_leaves_no_temporary_file_behind(
    store: FileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write("tickets.json", "original\n")

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(OSError):
        store.write("tickets.json", "replacement\n")
    monkeypatch.undo()
    assert store.list() == ("tickets.json",)
    assert list(store.root.iterdir()) == [store.root / "tickets.json"]


def test_a_failed_first_write_leaves_no_document(
    store: FileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(OSError):
        store.write("tickets.json", "content\n")
    monkeypatch.undo()
    assert not store.exists("tickets.json")
    assert store.list() == ()


def test_a_key_naming_the_root_itself_is_rejected(store: FileStore) -> None:
    for key in (".", "./"):
        with pytest.raises(InvalidKeyError):
            store.exists(key)
