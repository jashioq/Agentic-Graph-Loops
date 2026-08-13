# Implement

Call `get_ticket` first — that is the one ticket you are building, and the
whole of your scope.

## Scope

Build exactly its deliverables. Nothing beyond them: an improvement you
notice along the way belongs in a note to the user, not in the diff. A
reviewer will read anything extra as scope creep and file a finding
against it, whether or not the change itself was good.

## Dependencies

Never add or bump a dependency. If the work needs one that is not already
in the project, stop and ask — do not work around it, and do not add it
anyway.

## Git

Do not commit, branch, checkout, or merge. That is handled for you outside
this call, and the tools to do it yourself are denied. Leave your changes
in the working tree and stop once the deliverables are met.

## When to ask

Use `AskUserQuestion` before deciding any of:

- A public API signature the ticket does not specify.
- A database or schema change.
- Any behaviour the spec is silent on where a reasonable implementer could
  go two different ways.

For everything else, use your judgement and get on with it — not every
small decision is worth a question.

## Reference

`read_standards` for how code in this repository is written. `read_spec`
for why this work exists, if the ticket's deliverables leave that unclear.
