"""A pool of git worktrees, one per node, for the length of one run.

Layer: runtime. Imports `agl.runtime.paths` and `agl.core.vcs`, and nothing
about any particular workflow: a node is a key, and where its branch is cut
from is the caller's business, handed in as `base`.

`Worktrees` is a run's only place that creates or removes a worktree. The
kept-alive behaviour is the subtle part: a node whose work is not finished must
not have its tree recreated on the second pass, because anything branched off
it points at the branch that tree is sitting on. `acquire` reuses a kept tree
rather than checking one out again; `keep` is how a scheduler body says "this
node is not done yet"; `release` is how it says "it is."
"""

from dataclasses import dataclass
from pathlib import Path

from agl.core.vcs import Vcs
from agl.runtime import paths

__all__ = ["Work", "Worktrees"]


@dataclass(frozen=True)
class Work:
    """One node's key, bound to the worktree and branch its work happens on."""

    key: str
    tree: Path
    branch: str


class Worktrees:
    """Worktree lifecycle for one run: checkout, keep-alive, teardown."""

    def __init__(self, vcs: Vcs, *, trees_root: Path, project: str, label: str) -> None:
        self._vcs = vcs
        self._trees_root = trees_root
        self._project = project
        self._label = label
        self._open: dict[str, Work] = {}

    def acquire(self, key: str, branch: str, base: str) -> Work:
        """The node's worktree: a kept one from a prior round, else a fresh checkout.

        A kept tree is handed back as it is, so `branch` and `base` describe
        only the checkout this call might have to make.
        """
        return self._open.pop(key, None) or self._checkout(key, branch, base)

    def keep(self, work: Work) -> None:
        """Keep a worktree alive past this pass — its node is not done yet."""
        self._open[work.key] = work

    def release(self, work: Work) -> None:
        """Tear down a worktree whose node is fully done."""
        self._vcs.remove_worktree(work.tree)

    def tree_of(self, key: str) -> Path:
        """The tree of a still-open node — where work derived from it happens."""
        return self._open[key].tree

    def branch_for(self, key: str) -> str:
        """The branch one node's work lands on, in this run's namespace."""
        return paths.branch(self._label, key)

    def _checkout(self, key: str, branch: str, base: str) -> Work:
        tree = self._vcs.add_worktree(self._worktree_dir(key), branch, base)
        return Work(key=key, tree=tree.path, branch=branch)

    def _worktree_dir(self, key: str) -> Path:
        return paths.worktree_dir(self._trees_root, self._project, self._label, key)
