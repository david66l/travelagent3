"""Prompt compression for long conversations (PRD §4.10.1)."""

from __future__ import annotations

from core.settings import settings


def compress_message_history(
    messages: list[dict[str, str]],
    *,
    max_messages: int | None = None,
    max_chars_per_message: int | None = None,
) -> list[dict[str, str]]:
    """Trim and truncate message history before LLM calls."""
    if not messages:
        return []

    max_messages = max_messages or settings.prompt_compress_max_messages
    max_chars = max_chars_per_message or settings.prompt_compress_max_chars

    trimmed = messages[-max_messages:]
    compressed: list[dict[str, str]] = []
    for msg in trimmed:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if len(content) > max_chars:
            content = content[:max_chars] + "…"
        compressed.append({"role": role, "content": content})
    return compressed
