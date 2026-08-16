"""A pool of git worktrees, one per node, for the length of one run.

Layer: runtime. Imports `agl.runtime.paths`, `agl.runtime.context` and
`agl.core.vcs`, and nothing about any particular workflow: a node is a key, and
where its branch is cut from is the caller's business, handed in as `base`.

`Worktrees` is a run's only place that creates or removes a worktree. The
kept-alive behaviour is the subtle part: a node whose work is not finished must
not have its tree recreated on the second pass, because anything branched off
it points at the branch that tree is sitting on. `acquire` reuses a kept tree
rather than checking one out again; `keep` is how a scheduler body says "this
node is not done yet"; `release` is how it says "it is."

A resumed run starts with a pool that owns nothing and a repository full of
trees a dead process checked out, so `reopen` takes them over: the same
kept-alive state a first pass would have built, recovered from git rather than
from anything written down. `adopt` covers the other half of what a killed run
leaves — a branch whose tree is gone — and is the only other way a tree gets
made here.
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

        A kept tree is handed back as it is, so `branch` and `base` describe
        only the checkout this call might have to make.
        """
        return self._open.pop(key, None) or self._checkout(key, branch, base)

    def adopt(self, key: str, branch: str) -> Work:
        """Open a worktree onto a branch that already exists.

        The case where the branch survived but its tree did not: `acquire`
        creates a branch and would refuse this one as taken, so a resume needs
        the operation that only attaches. Like `acquire`, the tree comes back
        rather than being kept — the caller says whether the node is done.
        """
        tree = self._vcs.attach_worktree(self._worktree_dir(key), branch)
        return Work(key=key, tree=tree.path, branch=branch)

    def reopen(self) -> tuple[Work, ...]:
        """Take over the worktrees a previous process left behind.

        Pruned first, so a directory deleted under git is dropped from the
        registry rather than reported as a tree that is there. What is left is
        every registered worktree sitting directly under this run's trees
        directory, keyed by its directory name — which is the node id, because
        that is what `_worktree_dir` puts there. Another label's trees and the
        main worktree are somebody else's and are ignored.

        Each one is registered as open, so the resumed run reuses the tree it
        found for exactly the reason a first pass reuses a kept one.
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

    A function rather than a method, so the class stays free of run knowledge.
    A pool is addressed by project and label alone, so two processes over the
    same run name the same trees.
    """
    return Worktrees(
        ctx.vcs,
        trees_root=ctx.project.trees_root,
        project=ctx.project.name,
        label=ctx.label,
    )
