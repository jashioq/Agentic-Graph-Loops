"""The filesystem implementation of `Store`: one directory, one run.

Layer: core. Keys become paths under a root the store owns. Writes go through a
temp file and `os.replace`, so nobody meets a half-written document. Text is
UTF-8 everywhere, explicitly, never left to the locale.

Because keys are paths, two ordinary keys collide: `rounds` cannot be a document
once `rounds/first` made it a directory. Both directions raise `InvalidKeyError`
rather than whichever `OSError` the platform picks.
"""

import os
import tempfile
from pathlib import Path

from agl.core.store.api import InvalidKeyError, MissingKeyError, Store

__all__ = ["FileStore"]


class FileStore(Store):
    """Documents as files under `root`, addressed by relative path-like keys."""

    def __init__(self, root: Path) -> None:
        """Takes ownership of `root`, creating it if absent and resolving it once."""
        root.mkdir(parents=True, exist_ok=True)
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        """The resolved directory holding every document in this store."""
        return self._root

    def read(self, key: str) -> str:
        path = self._resolve(key)
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as error:
            raise MissingKeyError(key) from error

    def write(self, key: str, content: str) -> None:
        path = self._resolve(key)
        self._require_no_collision(key, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._temp_file(path)
        try:
            temp.write_text(content, encoding="utf-8")
            os.replace(temp, path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if not path.is_file():
            # Nothing there, or a directory another key made: either way this
            # store holds no document under that key. Asked before `unlink`
            # rather than after, so that a `PermissionError` reaching the
            # `except` below can only mean the one thing it says — and so it is
            # left to propagate, because a file that exists and cannot be
            # removed is not a missing key.
            raise MissingKeyError(key)
        try:
            path.unlink()
        except FileNotFoundError as error:
            raise MissingKeyError(key) from error

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def list(self, prefix: str = "") -> tuple[str, ...]:
        """Keys under the root, sorted. Files only, and symlinked directories are not entered."""
        keys = (
            path.relative_to(self._root).as_posix()
            for path in self._root.rglob("*")
            if path.is_file()
        )
        return tuple(sorted(key for key in keys if key.startswith(prefix)))

    # -- internals --------------------------------------------------------

    def _resolve(self, key: str) -> Path:
        """Turns a key into a path inside the root, or raises `InvalidKeyError`.

        The containment check after the syntactic rules is the load-bearing one:
        a symlink leaves the root without a single `..` in the key.
        """
        if not key or key.startswith("/") or "\\" in key or "\0" in key:
            raise InvalidKeyError(key)
        if ".." in key.split("/"):
            raise InvalidKeyError(key)
        path = (self._root / key).resolve()
        if path == self._root or not path.is_relative_to(self._root):
            raise InvalidKeyError(key)
        return path

    def _require_no_collision(self, key: str, path: Path) -> None:
        """Refuses a key another key has already claimed as the wrong kind of thing.

        Checked in both directions, so a write fails with `InvalidKeyError`
        rather than an `IsADirectoryError` from the middle of it.
        """
        if path.is_dir():
            raise InvalidKeyError(
                f"{key!r} names a directory: another key has already been written under it"
            )
        walked = self._root
        for part in path.parent.relative_to(self._root).parts:
            walked = walked / part
            if walked.is_file():
                held = walked.relative_to(self._root).as_posix()
                raise InvalidKeyError(f"{key!r} would be written under {held!r}, a document")

    def _temp_file(self, path: Path) -> Path:
        """An empty file beside `path`, so `os.replace` stays within one device."""
        handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(handle)
        return Path(name)
