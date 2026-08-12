"""Store core module. Re-exports the API only — never anything from `impl`.

Workflows import from here. Only `cli.py` may reach into `impl`.
"""

from agl.core.store.api import InvalidKeyError, MissingKeyError, Store

__all__ = ["InvalidKeyError", "MissingKeyError", "Store"]
