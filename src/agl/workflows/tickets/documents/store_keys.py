"""Every key this workflow stores a document under.

Layer: workflows. Imports nothing from the workflow: the roles that write these
documents and the steps that read them back both ask here, so the two cannot
drift apart.
"""

__all__ = ["REVIEWERS", "SPEC_KEY", "STANDARDS_KEY", "TICKETS_KEY", "review_key"]

SPEC_KEY = "spec.md"
STANDARDS_KEY = "standards.md"
TICKETS_KEY = "tickets.json"

REVIEWERS: tuple[str, ...] = ("quality", "spec")
"""The two reviewers, named by the findings document each one writes.

Every other name a reviewer has is this one with something around it: role
`review-<source>`, prompt `review_<source>`, activity prefixed with it.
"""


def review_key(ticket_id: str, round_: int, source: str) -> str:
    """Where one reviewer's findings for one ticket and round are stored.

    Round is in the key because a ticket is reviewed again after its bugs merge,
    and round 1 must not be overwritten.
    """
    # `review_round` counts *completed* reviews, so the first review writes round 0.
    return f"reviews/{ticket_id}/round-{round_}/{source}.json"
