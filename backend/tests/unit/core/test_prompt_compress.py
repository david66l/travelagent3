"""Unit tests for prompt compression."""

from core.prompt_compress import compress_message_history


def test_compress_truncates_long_messages():
    messages = [{"role": "user", "content": "x" * 5000}]
    out = compress_message_history(messages, max_chars_per_message=100)
    assert len(out[0]["content"]) == 101  # 100 chars + ellipsis


def test_long_message_keeps_head_and_tail():
    # Head and tail carry instructions / conclusions; the middle is dropped.
    content = "HEAD" + ("x" * 5000) + "TAIL"
    out = compress_message_history(
        [{"role": "user", "content": content}], max_chars_per_message=100
    )
    body = out[0]["content"]
    assert body.startswith("HEAD")
    assert body.endswith("TAIL")
    assert "…" in body


def test_overflow_is_summarized_not_dropped():
    messages = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]
    out = compress_message_history(messages, max_messages=5)
    # Recent 5 kept verbatim ...
    assert [m["content"] for m in out[-5:]] == [f"msg-{i}" for i in range(15, 20)]
    # ... and the 15 older ones survive as one summary, not silently dropped.
    assert out[0]["role"] == "system"
    assert "msg-0" in out[0]["content"]


def test_system_message_survives_long_history():
    # Regression: a recency window must never evict the system prompt.
    messages = [{"role": "system", "content": "SYSTEM PROMPT"}]
    messages += [{"role": "user", "content": f"msg-{i}"} for i in range(50)]
    out = compress_message_history(messages, max_messages=5)
    assert out[0] == {"role": "system", "content": "SYSTEM PROMPT"}


def test_injected_summarizer_is_used():
    messages = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]
    out = compress_message_history(
        messages, max_messages=5, summarizer=lambda msgs: f"folded {len(msgs)} msgs"
    )
    assert "folded 15 msgs" in out[0]["content"]
