
# AGL — Agentic Graph Loops

A runtime for agent workflows. `core` holds the connectors, `runtime` holds the
reusable machinery a loop is built from, and a workflow assembles them into one
particular loop. The first workflow is a ticket orchestrator.

## Architecture rules

1. **Core is connectors.** Every core module talks to something outside AGL —
   the Claude SDK, git, the filesystem, the terminal, subprocesses — and nothing
   else does. A module that reaches nothing outside AGL belongs in `runtime`,
   however low-level it looks: `dag` and `paths` are why the rule exists.
2. **No core module imports another core module.** Enforced by import-linter
   over the four connectors — `terminal`, `agent`, `store`, `vcs`. Cross-module
   wiring happens in `runtime` or a workflow, never between connectors. The one
   module outside that contract is `core/command.py`, deliberately: it is the
   shared subprocess runner that `vcs` and the merge build gate both call, and
   it stays outside by knowing nothing about git, builds or agents.
3. **Core modules report by returning values, not by emitting.** The caller invoked
   the method and already knows what happened, so there is no `on_event` parameter
   anywhere. The one exception is `agent`: its calls run for minutes, so it takes an
   `on_activity` callback for the dashboard footer. Do not add reporting callbacks to
   any other module.
4. **Runtime reports; the workflow decides.** A runtime module does the work it
   is asked to do and reports what happened. It never decides what an outcome
   means and holds no shared mutable run state. Prefer a return value; where a
   module must ask mid-flight, take a callback the workflow supplied at
   construction, and give it a safe default. `runtime/merge.py` is the pattern:
   `MergeQueue` reports `CONFLICT` and asks its `resolve`; the workflow is what
   knows that means a halt. No `Halt` type exists below `workflows`.
5. **Layers:** `cli` → `workflows` → `runtime` → `core`. Never upward.
   `config.py` sits aside, cli-only: nothing below `cli` may import it, so what
   runtime needs from `config.toml` arrives as `ProjectSettings` data.
6. **Every core module with a stand-in is a package:** `api.py` holds the ABC and
   its data types; `impl/` holds the implementation. `__init__.py` re-exports the
   API only. Workflows and runtime import from the package root; only `cli.py`
   imports `impl`.
7. **`dag`, `paths` and `json_fields` are pure single files** under `runtime`. One
   implementation forever, so no ABC. Do not add a Protocol or ABC for something
   with one implementation. They are leaves: they import nothing else in AGL.
8. **Fakes, not mocks.** Real fake classes in `tests/fakes.py`, inheriting the ABC
   so an incomplete fake fails at instantiation.
9. **Test against the real thing where it's cheap.** Git in `tmp_path`, files in
   `tmp_path`. Fake only what is slow, costly, nondeterministic, or interactive.
10. **Pure where possible.** Rendering, graph algorithms, and conflict
    classification are pure functions. I/O lives at the edges.
11. **The ABC describes what workflows need**, not everything the implementation
    can do. Implementation-only helpers stay private to `impl/`.
12. **A workflow's state is one document in its run directory.** It is read
    fresh at every decision and written before anything acts on it; nothing
    derivable from it — the stage, the graph, the merge queue, which step a work
    item is on — is stored beside it. `runtime/record.py` owns the two
    documents; the workflow owns what goes in them. A workflow may also expose
    `resume(ctx)`, which settles its state against the world and re-enters the
    same loop; without one, a run of it cannot be resumed.

## Style

- Type hints on every public function. `mypy --strict` must pass.
- Frozen dataclasses for data. No metaclasses, no decorator registration,
  no dynamic imports.
- Files under ~300 lines. Split into a package when exceeded.
- Module docstring stating the contract and the layer, in ~8 lines or fewer.

### Docstrings are lean

Full prose documentation lives outside the code. A docstring exists to aid
someone who has already read it, so:

- One summary line saying what the function does: takes something, does
  something to it, returns something.
- Then `param:` and `return:` lines, **only where the signature does not
  already say it**:

      param: dag - the graph of work items and what blocks what
      param: scheduler - decides which node runs next
      return: Worktree - the tree the chosen node's work happens in

- Nothing else. No design rationale, no history, no justification of decisions
  already taken. An invariant a future edit could silently break is a one-line
  `#` comment on the code it constrains, not a paragraph.
- A one-line docstring is the common case. Reach for the `param:`/`return:`
  block when a call site genuinely cannot be written without it.

## TDD

Every module is built test-first: write the failing test, then the
implementation. Run the checks before claiming anything works.

## Commands

    agl init                   # write config.toml for the repo in this directory
    agl run <workflow> -n <label> <description>
    agl resume <label>         # continue a run that was stopped
    agl clean <label>          # remove a run's worktrees, branches, and files

`resume` takes the label and nothing else: everything the run was started with
is in its `run.json`, and where it got to is derived from its `state.json`.

    uv run pytest              # parallel by default (-n auto)
    uv run pytest -n0          # serial, for reading a failure one worker at a time
    uv run mypy
    uv run ruff check --fix
    uv run lint-imports

`-n auto` is in `addopts`, so pytest-xdist has to be installed. `-p no:xdist`
does not work; pass `-n0`.
