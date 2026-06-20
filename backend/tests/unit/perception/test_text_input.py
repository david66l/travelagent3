"""Tests for text input perception."""

from perception import build_text_perception


def test_build_text_perception():
    output = build_text_perception("  hello world  ")
    assert output["user_input"] == "hello world"
    assert output["attachments_meta"] == []
    assert output["external_event"] is None
