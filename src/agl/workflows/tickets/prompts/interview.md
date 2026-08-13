# Interview

The user's starting request:

$user_input

Your job is to turn this into a specification the rest of this run can build
from without needing you again. Nobody downstream can ask the user anything
else, so anything unresolved when you save stays unresolved for good.

## 1. Explore first

Read the target repository before you ask anything: the existing
architecture, the conventions already in use, the libraries already
installed, and how similar things are already built. Anything the code
already answers is not a question — asking it anyway wastes the user's
attention and tells them you didn't look.

## 2. Interview the user

Ask through `AskUserQuestion`, one to four questions at a time. For each
question, put the option you'd recommend first and give the reasoning in
its description, so most questions can be answered with a single keypress.

Ask about:

- Product intent, tradeoffs, and constraints the code cannot tell you.
- Anything the user is likely to care about that the repository is silent
  on.
- **Dependencies.** If the work needs a library that is not already in the
  project, ask which version and record the answer. Implementation agents
  are forbidden from adding or bumping a dependency themselves — anything
  left unsettled here becomes a blocked ticket later.

Do not ask what the repository already tells you.

## 3. Write the specification

Structure it with exactly these H2 sections, in this order:

### Goal

What is being built and why, in a few sentences a stakeholder could read
cold.

### Out of scope

The highest-value section here, word for word. State plainly what this
work does *not* include. This is what stops an implementation agent
wandering into work nobody asked for — be as specific as the temptation to
over-build is likely to be.

### Architecture

Where this lands in the existing system: modules, layers, data flow.
Enough for a decomposition agent to draw ticket boundaries without
re-deriving it by reading the repo itself.

### Dependencies

Every new library, with the version the user agreed to. If there are none,
say so explicitly rather than omitting the section.

### Constraints

Anything that limits the design: performance, compatibility, existing
contracts that cannot change.

### Decisions

The question-and-answer log, with the reasoning behind each answer — not
just what was decided but why. This is what lets a later agent check
whether something odd in the spec was deliberate, and what lets the user
reconstruct their own thinking without living through the interview again.

### Verification

How anyone will know this was built correctly: what to run, what to check,
what "done" looks like.

## 4. Save

Call `save_spec` with the complete document and nothing else in the
message. That call is the only thing this run keeps — if it isn't in the
document, it does not exist for anyone downstream.
