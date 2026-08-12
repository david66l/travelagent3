"""Prompt compression for long conversations (PRD §4.10.1).

Layered strategy — cheap, synchronous, no LLM on the hot path:

  1. ``system`` messages are never dropped or truncated. They carry the
     instructions; evicting the system prompt via a recency window (the old
     ``messages[-N:]`` could drop index 0) silently changes model behaviour.
  2. The most recent ``max_messages`` non-system messages are kept verbatim.
  3. Older overflow is *folded into one summary message* instead of being
     silently dropped, so earlier context survives in condensed form.
  4. A single over-long message keeps its head AND tail; the middle is the
     safe part to drop. A blind tail cut can sever a closing brace, a final
     instruction, or the conclusion of a RAG snippet.

The default summary is an offline extractive fold (zero latency, zero cost).
For very long sessions a real LLM summary can be plugged in via ``summarizer``
without touching callers — but note this project also persists durable facts
as a structured ``profile`` (see ``conversation_state.merge_profile``), so the
window rarely needs more than the extractive fold.
"""

from __future__ import annotations

from collections.abc import Callable

from core.settings import settings

Message = dict[str, str]
Summarizer = Callable[[list[Message]], str]

_SUMMARY_PREFIX = "[earlier conversation summary] "


def _truncate_middle(content: str, max_chars: int) -> str:
    """Keep ``content`` head + tail around an ellipsis, eliding the middle."""
    if len(content) <= max_chars:
        return content
    head = (max_chars + 1) // 2
    tail = max_chars - head
    if tail <= 0:
        return content[:max_chars] + "…"
    return content[:head] + "…" + content[-tail:]


def _extractive_summary(messages: list[Message]) -> str:
    """Offline fold of dropped messages: role-tagged, each clipped short."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = " ".join((msg.get("content") or "").split())
        if not text:
            continue
        if len(text) > 120:
            text = text[:120] + "…"
        parts.append(f"{role}: {text}")
    return " | ".join(parts)


def compress_message_history(
    messages: list[Message],
    *,
    max_messages: int | None = None,
    max_chars_per_message: int | None = None,
    summarizer: Summarizer | None = None,
) -> list[Message]:
    """Window + summarize overflow + per-message truncate, keeping system msgs.

    Output order: original system messages, then the earlier-context summary
    (if any), then the recent window.
    """
    if not messages:
        return []

    max_messages = max_messages or settings.prompt_compress_max_messages
    max_chars = max_chars_per_message or settings.prompt_compress_max_chars

    system_msgs = [m for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") != "system"]

    summary_head: list[Message] = []
    recent = convo
    if len(convo) > max_messages:
        overflow = convo[:-max_messages]
        recent = convo[-max_messages:]
        summary_text = (summarizer or _extractive_summary)(overflow)
        if summary_text:
            summary_text = _truncate_middle(summary_text, max_chars)
            summary_head.append(
                {"role": "system", "content": _SUMMARY_PREFIX + summary_text}
            )

    compressed: list[Message] = []
    for msg in recent:
        content = msg.get("content", "")
        if len(content) > max_chars:
            content = _truncate_middle(content, max_chars)
        compressed.append({"role": msg.get("role", "user"), "content": content})

    return [*system_msgs, *summary_head, *compressed]
