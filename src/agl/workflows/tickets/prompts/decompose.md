# Decompose

Call `read_spec` first. Then break the specification into tickets.

## Slices

Each ticket should be a vertical slice: independently valuable, and
ideally independently mergeable. Two tickets that would edit the same
function are not two slices — they are one ticket that got split by
mistake.

## Dependencies

Keep the dependency graph shallow. Use `blocked_by` only for a genuine
shared foundation — a version catalog entry, a schema change everything
else is built on. A ticket that merely touches a related area is not
blocked by another one touching the same area; if their diffs would not
conflict, they are independent.

Independence is what makes concurrent implementation safe. A false
dependency serializes work that could have run in parallel; a missing one
produces two tickets racing to edit the same lines. Get this right.

## Deliverables

Every deliverable names the artifact, its architectural location, and its
contract boundary — never its internals.

- Good: `AuthRepository.refreshToken(): Result<Token>` in `data/auth/`;
  maps HTTP failures to domain errors; no Retrofit type appears in its
  signature.
- Too vague: "implement token refresh" — unverifiable, so a reviewer
  cannot check it against the diff.
- Wrong layer: "use `when` instead of if/else" — an idiom, which belongs
  in the standards document, not in a ticket.

A deliverable a reviewer cannot check against the diff is not a
deliverable, it is a wish.

## Save

Call `save_tickets` with the whole set. If it comes back with a
validation error, fix what it names and call it again.
