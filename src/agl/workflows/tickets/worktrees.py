"""One ticket's worktree, from checkout through the review rounds to teardown.

Layer: workflows. Imports `agl.runtime.paths` and `agl.core.vcs`, and this
workflow's `models`.

`Worktrees` is the run's only place that creates or removes a worktree. The
kept-alive behaviour is the subtle part: a parent that files bugs must not
have its tree recreated on the second pass, because the bug tickets are
branched off it and the parent's own uncommitted-nothing-but-branched state is
what `base_for` points them at. `acquire` reuses a kept tree rather than
checking one out again; `keep` is how the scheduler body says "this ticket
is not done yet"; `release` is how it says "it is."
"""

from dataclasses import dataclass
from pathlib import Path

from agl.config import ProjectConfig
from agl.core.vcs import Vcs
from agl.runtime import paths
from agl.workflows.tickets.models import Ticket

__all__ = ["Work", "Worktrees"]


@dataclass(frozen=True)
class Work:
    """One ticket, bound to the worktree its work happens in."""

    ticket: Ticket
    tree: Path
    branch: str


class Worktrees:
    """Worktree lifecycle for one run: checkout, keep-alive, teardown."""

    def __init__(self, vcs: Vcs, config: ProjectConfig, label: str, base_branch: str) -> None:
        self._vcs = vcs
        self._config = config
        self._label = label
        self._base_branch = base_branch
        self._open: dict[str, Work] = {}

    def acquire(self, ticket: Ticket) -> Work:
        """The ticket's worktree: a kept one from a prior round, else a fresh checkout."""
        return self._open.pop(ticket.id, None) or self._checkout(ticket)

    def keep(self, work: Work) -> None:
        """Keep a worktree alive past this pass — its ticket is not done yet."""
        self._open[work.ticket.id] = work

    def release(self, work: Work) -> None:
        """Tear down a worktree whose ticket is fully done."""
        self._vcs.remove_worktree(work.tree)

    def tree_of(self, ticket_id: str) -> Path:
        """The tree of a still-open ticket — a bug's merge cwd is its parent's tree."""
        return self._open[ticket_id].tree

    def base_for(self, ticket: Ticket) -> str:
        """The branch a ticket's own branch is cut from: the run's base, or its parent's."""
        if ticket.parent is None:
            return self._base_branch
        return paths.branch(self._label, ticket.parent)

    def _checkout(self, ticket: Ticket) -> Work:
        branch = paths.branch(self._label, ticket.id)
        base = self.base_for(ticket)
        tree = self._vcs.add_worktree(self._worktree_dir(ticket.id), branch, base)
        return Work(ticket=ticket, tree=tree.path, branch=branch)

    def _worktree_dir(self, ticket_id: str) -> Path:
        return paths.worktree_dir(
            self._config.trees_root, self._config.name, self._label, ticket_id
        )
