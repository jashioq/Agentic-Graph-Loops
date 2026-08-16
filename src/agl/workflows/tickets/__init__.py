"""The ticket workflow: decompose a request into tickets and drive them to merged.

Layer: workflows. A run interviews the user until it has a specification,
proposes a decomposition of it into tickets for the user to approve, then works
the approved set concurrently: each ticket is implemented by an agent in its own
worktree, reviewed by two more, and either merged into the run's base branch or
sent round again as bug tickets its parent now waits on. Every decision is taken
from `state.json`, read fresh at the moment it matters and written before
anything acts on it, so a run that was killed can be resumed into whatever step
it was owed.

The directory:

    workflow.py             the entry points `run` and `resume`, and the stage loop
    steps.py                what is owed: the run's stage, and one ticket's step
    ticket_pass.py          one ticket's pass — implement, review, merge or file bugs
    ticket_claims.py        how the scheduler claims a ticket off the state
    reconcile_on_resume.py  git and the state, settled before a resumed run acts
    halting.py              what a halt says, and what puts one there
    agents.py               the five roles: which prompt, which tools, which model
    tools.py                what each role may reach, as closures over the store
    screens.py              the three screens a run draws
    models.py               `Ticket`, `Status`, and the moves between statuses
    findings.py             what a reviewer found, and the bug tickets it becomes
    run_state.py            one run's state as a value, and the pure moves over it
    errors.py               every exception this workflow raises, and `Halt`
    documents/              the store keys, and a codec per document
    prompts/                one markdown template per role

No re-exports: every module is imported by its own name, so there is one import
path per name rather than two that can disagree about what is public.
"""
