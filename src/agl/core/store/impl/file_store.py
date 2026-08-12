"""The filesystem implementation of `Store`: one directory, one run.

Layer: core. Keys become paths under a root the store owns. Writes go through a
temp file in the destination directory and then `os.replace`, so a reader — or a
person inspecting a failed run — never meets a half-written `tickets.json`.

Text is UTF-8 everywhere, stated explicitly rather than left to the locale.
"""

import os
import tempfile
from pathlib import Path

from agl.core.store.api import InvalidKeyError, MissingKeyError, Store

__all__ = ["FileStore"]


class FileStore(Store):
    """Documents as files under `root`, addressed by relative path-like keys."""

    def __init__(self, root: Path) -> None:
        """Take ownership of `root`, creating it if it does not exist.

        The root is resolved once, so every later containment check compares
        real paths and a symlinked root does not read as an escape.
        """
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
        try:
            path.unlink()
        except (FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError) as error:
            raise MissingKeyError(key) from error

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def list(self, prefix: str = "") -> tuple[str, ...]:
        """Keys under the root, sorted. `prefix` is matched against whole keys.

        Directories are not keys, so only files appear. Symlinked directories
        are not descended into — nothing outside the root is this store's.
        """
        keys = (
            path.relative_to(self._root).as_posix()
            for path in self._root.rglob("*")
            if path.is_file()
        )
        return tuple(sorted(key for key in keys if key.startswith(prefix)))

    # -- internals --------------------------------------------------------

    def _resolve(self, key: str) -> Path:
        """Turn a key into a path inside the root, or raise `InvalidKeyError`.

        The syntactic rules reject the obvious escapes; the containment check
        afterwards is the one that has to hold, since a symlink can leave the
        root without a single `..` in the key.
        """
        if not key or key.startswith("/") or "\\" in key or "\0" in key:
            raise InvalidKeyError(key)
        if ".." in key.split("/"):
            raise InvalidKeyError(key)
        path = (self._root / key).resolve()
        if path == self._root or not path.is_relative_to(self._root):
            raise InvalidKeyError(key)
        return path

    def _temp_file(self, path: Path) -> Path:
        """An empty file beside `path`, so `os.replace` stays within one device."""
        handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(handle)
        return Path(name)
