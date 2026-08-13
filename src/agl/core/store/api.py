"""Store API: keyed documents scoped to one run.

Layer: core. Text is the primitive and JSON is a thin layer on top, so an
implementation provides five methods and gets `read_json`/`write_json` free.

This module maps keys to documents and holds no opinion about who may read
what. Access policy belongs to the caller that decides which keys to hand out;
if a permissions concept ever appears here, it has leaked in from above.

A key is a relative path-like string: `notes.md`, `rounds/first/summary.md`.
What makes one valid is stated by the implementation that resolves it, which is
the only place that knows the root a key must stay inside.

Keys with a `/` in them nest, and nesting means two otherwise valid keys can
collide: once something is stored under `rounds/first`, `rounds` names a
container and cannot also name a document. An implementation that resolves keys
onto a hierarchy raises `InvalidKeyError` for either direction of that clash —
it is a fact about the pair of keys, not a fault in the write that discovered
it.
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
        """Store `content` under `key`, replacing anything already there.

        Raises `InvalidKeyError`, including when the key collides with one
        already stored — see the module docstring. A reader either sees the
        whole new document or the whole old one, never a half-written file, and
        a refused write changes nothing.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Drop the document. Raises `MissingKeyError`, or `InvalidKeyError`.

        `MissingKeyError` means there is no document under that key. A document
        that is there and cannot be removed is a different thing, and whatever
        the store raises for it is not caught and relabeled here.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether a document is stored under `key`. Raises `InvalidKeyError`.

        A key that names a container rather than a document answers `False`;
        the question has an answer either way, so it is not an error.
        """

    @abstractmethod
    def list(self, prefix: str = "") -> tuple[str, ...]:
        """Every key starting with `prefix`, sorted. The default prefix is all keys.

        A **string** prefix, not a segment one: `list("T-1")` returns `T-10.md`
        as well as `T-1.md`. A caller that means a container ends the prefix
        with `/`.
        """

    # -- JSON, in terms of the five above ---------------------------------

    def read_json(self, key: str) -> Any:
        """Parse the document as JSON. Raises what `read` raises, or `ValueError`."""
        return json.loads(self.read(key))

    def write_json(self, key: str, value: Any) -> None:
        """Serialize `value` as indented JSON and store it under `key`.

        Indented and newline-terminated because these documents get read by
        people inspecting a run, and diffed between runs.
        """
        self.write(key, json.dumps(value, indent=2, ensure_ascii=False) + "\n")
