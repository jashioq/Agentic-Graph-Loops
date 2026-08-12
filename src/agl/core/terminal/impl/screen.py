"""Screen assembly: a `Screen` becomes a three-region Rich layout.

Layer: core (impl). Pure — no console, no state. Header and footer are sized to
the exact number of lines they render, and content takes whatever is left.
Content that overflows its region is cropped, not scrolled.
"""

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.text import Text as RichText

from agl.core.terminal.api import Row, Rows, Screen
from agl.core.terminal.impl.render import line_count, to_renderable

__all__ = ["to_layout"]


def to_layout(screen: Screen, now: float) -> Layout:
    """Lay `screen` out as of `now`: header pinned top, footer pinned bottom."""
    layout = Layout()
    layout.split_column(
        _region("header", screen.header, now),
        Layout(_visible(to_renderable(screen.content, now)), name="content", ratio=1),
        _region("footer", screen.footer, now),
    )
    return layout


def _region(name: str, component: Row | Rows | None, now: float) -> Layout:
    """A fixed-height region, collapsed to nothing when there is no component."""
    if component is None:
        # Size 0 alone does not collapse a region — hiding it is what drops the
        # region to zero lines.
        return Layout(RichText(""), name=name, size=0, visible=False)
    return Layout(
        _visible(to_renderable(component, now)), name=name, size=line_count(component, now)
    )


def _visible(renderable: RenderableType) -> RenderableType:
    """Keep empty renderables from being swapped for Rich's placeholder panel.

    `Layout` stores `renderable or _Placeholder(self)`, and an empty `Text` is
    falsy. A `Group` is always truthy, so a blank region stays blank.
    """
    return Group(renderable)
