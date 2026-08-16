"""Store API: keyed documents scoped to one run.

Layer: core. Text is the primitive, JSON a thin layer over it: an implementation
provides five methods and gets `read_json`/`write_json` free. No opinion about
who may read what — access policy belongs to whoever hands out the keys.

A key is a relative path-like string: `notes.md`, `rounds/first/summary.md`.
Keys nest, so two valid keys can collide — once `rounds/first` holds a document,
`rounds` cannot — and an implementation raises `InvalidKeyError` either way round.
"""

import json
from abc import ABC, abstractmethod
from typing import Any

__all__ = ["InvalidKeyError", "MissingKeyError", "Store"]


class MissingKeyError(Exception):
    """Raised when a key names a document the store does not hold."""


class InvalidKeyError(Exception):
    """Raised when a key is malformed or would resolve outside the store root."""


class Store(ABC):
    """Documents addressed by key, for the lifetime of one run."""

    @abstractmethod
    def read(self, key: str) -> str:
        """The document's text. Raises `MissingKeyError`, or `InvalidKeyError`."""

    @abstractmethod
    def write(self, key: str, content: str) -> None:
        """Stores `content` under `key`, replacing anything already there.

        Atomic: a reader sees the whole old document or the whole new one.
        Raises `InvalidKeyError`, including on a collision, changing nothing.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Drops the document. Raises `MissingKeyError`, or `InvalidKeyError`."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether a document is stored under `key`. A container answers `False`."""

    @abstractmethod
    def list(self, prefix: str = "") -> tuple[str, ...]:
        """Every key starting with `prefix`, sorted; the default is all of them.

        param: prefix - a *string* prefix, so `"T-1"` also matches `T-10.md`;
            end it with `/` to mean a container
        """

    # -- JSON, in terms of the five above ---------------------------------

    def read_json(self, key: str) -> Any:
        """Parse the document as JSON. Raises what `read` raises, or `ValueError`."""
        return json.loads(self.read(key))

    def write_json(self, key: str, value: Any) -> None:
        """Serializes `value` as indented, newline-terminated JSON and stores it."""
        self.write(key, json.dumps(value, indent=2, ensure_ascii=False) + "\n")
