"""Terminal core module. Re-exports the API only — never anything from `impl`.

Workflows import from here. Only `cli.py` and demos may reach into `impl`.
"""

from agl.core.terminal.api import (
    Answer,
    Color,
    Component,
    LiveSession,
    Option,
    Question,
    Row,
    Rows,
    Screen,
    Spacer,
    Terminal,
    TerminalError,
    Text,
    Timer,
)

__all__ = [
    "Answer",
    "Color",
    "Component",
    "LiveSession",
    "Option",
    "Question",
    "Row",
    "Rows",
    "Screen",
    "Spacer",
    "Terminal",
    "TerminalError",
    "Text",
    "Timer",
]
