"""The keys this workflow stores its documents under."""

from agl.workflows.tickets.documents.store_keys import review_key


def test_review_key_shape() -> None:
    assert review_key("T-03", 1, "quality") == "reviews/T-03/round-1/quality.json"


def test_review_key_is_distinct_per_ticket_round_and_source() -> None:
    keys = {
        review_key("T-03", 1, "quality"),
        review_key("T-03", 2, "quality"),
        review_key("T-03", 1, "spec"),
        review_key("T-04", 1, "quality"),
    }

    assert len(keys) == 4
