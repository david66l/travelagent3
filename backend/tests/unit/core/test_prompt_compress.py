"""Unit tests for prompt compression."""

from core.prompt_compress import compress_message_history


def test_compress_truncates_long_messages():
    messages = [{"role": "user", "content": "x" * 5000}]
    out = compress_message_history(messages, max_chars_per_message=100)
    assert len(out[0]["content"]) == 101  # 100 chars + ellipsis


def test_compress_limits_message_count():
    messages = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]
    out = compress_message_history(messages, max_messages=5)
    assert len(out) == 5
    assert out[-1]["content"] == "msg-19"
