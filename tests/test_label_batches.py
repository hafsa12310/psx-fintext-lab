from psx_fintext_lab.data.label_batches import validate_row

LABELS = ["positive", "negative", "neutral"]
EVENTS = ["financial_results", "other"]


def test_valid_row_is_cleaned() -> None:
    item = {"id": "a1", "sentiment": "Positive", "event_type": "weird", "confidence": 1.7}
    out = validate_row(item, LABELS, EVENTS)
    assert out["sentiment"] == "positive"  # lowercased
    assert out["event_type"] == "other"  # unknown event type replaced
    assert out["confidence"] == 1.0  # clamped to 0..1


def test_invalid_rows_are_rejected() -> None:
    assert validate_row({"id": "a1", "sentiment": "bullish"}, LABELS, EVENTS) is None
    assert validate_row({"sentiment": "neutral"}, LABELS, EVENTS) is None  # missing id
