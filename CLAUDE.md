
# AGL — Agentic Graph Loops

A runtime for agent workflows. Core modules are reusable building blocks;
workflows compose them. The first workflow is a ticket orchestrator.

## Architecture rules

1. **No core module imports another core module.** Enforced by import-linter.
   Cross-module wiring happens in workflows, never in core.
2. **Core modules report by returning values, not by emitting.** The caller invoked
   the method and already knows what happened, so there is no `on_event` parameter
   anywhere. The one exception is `agent`: its calls run for minutes, so it takes an
   `on_activity` callback for the dashboard footer. Do not add reporting callbacks to
   any other module.
3. **Layers:** `cli` → `workflows` → `core`. Never upward.
4. **Every core module with a stand-in is a package:** `api.py` holds the ABC and
   its data types; `impl/` holds the implementation. `__init__.py` re-exports the
   API only. Workflows import from the package root; only `cli.py` imports `impl`.
5. **`dag` and `paths` are pure single files.** One implementation forever, so no
   ABC. Do not add a Protocol or ABC for something with one implementation.
6. **Fakes, not mocks.** Real fake classes in `tests/fakes.py`, inheriting the ABC
   so an incomplete fake fails at instantiation.
7. **Test against the real thing where it's cheap.** Git in `tmp_path`, files in
   `tmp_path`. Fake only what is slow, costly, nondeterministic, or interactive.
8. **Pure where possible.** Rendering, graph algorithms, and conflict
   classification are pure functions. I/O lives at the edges.
9. **The ABC describes what workflows need**, not everything the implementation
   can do. Implementation-only helpers stay private to `impl/`.

## Style

- Type hints on every public function. `mypy --strict` must pass.
- Frozen dataclasses for data. No metaclasses, no decorator registration,
  no dynamic imports.
- Files under ~300 lines. Split into a package when exceeded.
- Module docstring stating the contract and the layer.

## TDD

Every module is built test-first: write the failing test, then the
implementation. Run the checks before claiming anything works.

## Commands

    uv run pytest              # parallel by default (-n auto)
    uv run pytest -n0          # serial, for reading a failure one worker at a time
    uv run mypy
    uv run ruff check --fix
    uv run lint-imports

`-n auto` is in `addopts`, so pytest-xdist has to be installed. `-p no:xdist`
does not work; pass `-n0`.
