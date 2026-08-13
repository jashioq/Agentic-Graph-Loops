# Triage

Two reviewers' `HIGH` findings against this ticket, and its deliverables
so you know what was meant to be built. You do not have the code, on
purpose — this is the cheapest call in the run and it should stay that
way. Work only from what's below.

Findings:

$findings

Deliverables:

$deliverables

## Grouping

Group only these findings — `MEDIUM` and `LOW` are recorded elsewhere and
are not your concern. Group by what one agent could fix in a single pass:
the same file, the same seam, the same root cause. Two reviewers
describing the same problem in different words belong in one group, not
two.

Get this right: each group becomes its own branch, and parallel fixes to
one root cause are the most likely thing in this system to conflict at
merge time. Splitting one root cause into two groups creates that
conflict.

## Deliverables

Turn each group into a prescriptive, checkable deliverable: what must be
true when the fix is done, not a restatement of what's currently wrong.

If a finding is too vague to turn into a deliverable, say so plainly in
the group's title rather than inventing one — a made-up deliverable is
worse than an honest "unclear: needs a human."

## Coverage

Every `HIGH` finding above must appear in exactly one group, named by its
id in that group's `findings` list. This is checked mechanically — a
dropped finding fails the run.
