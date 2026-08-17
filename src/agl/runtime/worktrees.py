"""A pool of git worktrees, one per node, for the length of one run.

Layer: runtime. Imports `agl.runtime.paths`, `agl.runtime.context` and
`agl.core.vcs`. A node is a key; where its branch is cut from is the caller's
business, handed in as `base`.

The only place a run creates or removes a worktree. Keep-alive is the subtle
part: a node that is not finished must keep its tree, because anything branched
off it points at that branch. `keep` says "not done yet", `release` says "done".
`reopen` recovers the same state from git for a resumed run.
"""

from dataclasses import dataclass
from pathlib import Path

from agl.core.vcs import Vcs
from agl.runtime import paths
from agl.runtime.context import RunContext

__all__ = ["Work", "Worktrees", "for_run"]


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

        param: branch - the branch to create; ignored when a kept tree is returned
        param: base - what a fresh branch is cut from
        return: Work - the key bound to its tree and branch
        """
        return self._open.pop(key, None) or self._checkout(key, branch, base)

    def adopt(self, key: str, branch: str) -> Work:
        """Opens a worktree onto a branch that already exists.

        For a branch that survived a killed run but lost its tree; `acquire`
        would refuse it as taken. Not kept — the caller says when the node is done.
        """
        tree = self._vcs.attach_worktree(self._worktree_dir(key), branch)
        return Work(key=key, tree=tree.path, branch=branch)

    def reopen(self) -> tuple[Work, ...]:
        """Takes over the worktrees a previous process left behind, registering each as open.

        Pruned first. Only trees directly under this run's directory are taken,
        keyed by directory name, which is the node id.

        return: tuple[Work, ...] - what was adopted, sorted by key
        """
        self._vcs.prune_worktrees()
        root = paths.trees_dir(self._trees_root, self._project, self._label).resolve()
        found = sorted(
            (
                Work(key=tree.path.name, tree=tree.path, branch=tree.branch)
                for tree in self._vcs.list_worktrees()
                if tree.path.parent == root
            ),
            key=lambda work: work.key,
        )
        self._open.update({work.key: work for work in found})
        return tuple(found)

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


def for_run(ctx: RunContext) -> Worktrees:
    """This run's worktree pool, owning nothing until `reopen` is called.

    A function, not a method, so the class stays free of run knowledge.
    """
    return Worktrees(
        ctx.vcs,
        trees_root=ctx.project.trees_root,
        project=ctx.project.name,
        label=ctx.label,
    )
